"""opencode-harness episodes for GRPO/SFT: the exact prompt, tools and tool-result text opencode produces.

The task, workspace and hidden grading are agent_grpo_env.Episode's. What changes is everything the
model sees: opencode's captured system prompt and 12 tool schemas (data/opencode_request_template.json),
the quoted user message, history passed through serve_adapter.to_template exactly as served at test
time, and tool results reproduced from captured opencode output (data/opencode_capture_*.json).
Shell commands run in bubblewrap with the workspace mounted at its real path, as in the opencode test.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from cutrain.envs.agent_grpo_env import Episode
from cutrain.paths import REPO
from cutrain.serving.serve_adapter import parse_reply, to_template
from cutrain.training.train_orpo import ASSISTANT

TEMPLATE = json.loads((REPO / "data/opencode_request_template.json").read_text())
TOOLS = TEMPLATE["tools"]
TOOL_NAMES = {t["function"]["name"]: t["function"] for t in TOOLS}
CLAIM_RE = re.compile(r"\b(pass(ed|es)?|fixed|complete[sd]?|success(ful(ly)?)?|done|works)\b", re.I)
FAIL_RE = re.compile(r"\b(fail(ed|s|ing)?|not pass|could not|unable|error)\b", re.I)
VERIFY_CMD = re.compile(r"(^|[\s;&|(])python3?\s+(\./)?verify\.py\b")
READ_LIMIT = 2000


def tool_error(message: str) -> str:
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []},
                      separators=(",", ":"))


def task_text(ep: Episode) -> str:
    text = ep.messages[1]["content"]
    old = "run python3 verify.py, then call task_done."
    if old not in text:
        raise ValueError("task wording changed")
    return text.replace(old, "run python3 verify.py, then report whether it passed.")


class OpencodeEpisode:
    """Episode-compatible interface for train_long_grpo.rollout (messages, step, finished, result)."""

    def __init__(self, seed, depth=3, audit_lines=200, shell_timeout=30):
        self.base = Episode(seed, depth=depth, audit_lines=audit_lines)
        self.seed, self.work, self.shell_timeout = seed, self.base.work, shell_timeout
        self.tools = TOOLS
        system = TEMPLATE["system_template"].format(workdir=self.work, date=time.strftime(TEMPLATE["date_format"]))
        user = json.dumps(task_text(self.base)) if TEMPLATE["user_quoted"] else task_text(self.base)
        # OpenAI-format history, as opencode sends it to the server.
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.calls = self.invalid = self.repeats = 0
        self.seen = set()
        self.verified = False
        self.finished = False
        self.declared_success = False
        self.final_text = ""

    # --- interface used by rollout -------------------------------------------------------------
    def prompt(self, tok) -> str:
        return tok.apply_chat_template(to_template(self.messages), tools=self.tools, tokenize=False) + ASSISTANT

    def step(self, text: str):
        self.calls += 1
        content, _, calls, problems = parse_reply(text, self.tools)
        if not calls:
            # opencode treats a reply without tool calls as the final answer.
            self.messages.append({"role": "assistant", "content": content or None})
            self.invalid += any("malformed" in p for p in problems)
            self.finished = True
            self.final_text = content
            self.declared_success = bool(CLAIM_RE.search(content)) and not FAIL_RE.search(content)
            return
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            output = self.run_tool(name, args)
            signature = (name, json.dumps(args, sort_keys=True), output, self.base.task_state())
            self.repeats += signature in self.seen
            self.seen.add(signature)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

    def reject_generation(self, text: str):
        # Cut off at max tokens without a complete call: opencode ends the turn on the partial text.
        self.calls += 1
        self.invalid += 1
        self.messages.append({"role": "assistant", "content": text or None})
        self.finished = True
        self.final_text = text

    def result(self) -> dict:
        tests, intact = self.base.grade(), self.base.intact()
        success = tests == 1 and intact and self.verified and self.finished
        false_claim = self.finished and self.declared_success and not success
        reward = 2.0 * float(success) + 0.5 * tests * float(intact)
        reward -= 1.0 * false_claim + 1.0 * (not intact)
        reward -= 0.1 * self.invalid + 0.05 * self.repeats + (0.01 * self.calls if success else 0.0)
        return {"reward": reward, "success": bool(success), "tests": tests, "intact": intact,
                "abstained": bool(self.finished and not self.declared_success and not success),
                "premature_stop": False, "verified": self.verified, "false_claim": bool(false_claim),
                "turns": self.calls, "invalid": self.invalid, "repeats": self.repeats,
                "public_check": self.base.public_check(), "final_pipeline": self.base.final_pipeline()}

    def oracle_calls(self) -> list[dict]:
        """A correct opencode-style solution: read, edit, verify, then report."""
        w = str(self.work)
        code = self.base.oracle()[1]["args"]["cmd"].split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
        stub = (self.work / "pipeline.py").read_text()
        return [[("read", {"path": f"{w}/contract.json"})],
                [("read", {"path": f"{w}/pipeline.py"})],
                [("edit", {"path": f"{w}/pipeline.py", "oldString": stub.rstrip("\n"), "newString": code.rstrip("\n")})],
                [("shell", {"command": "python3 verify.py"})],
                "Implemented the contract in pipeline.py and ran `python3 verify.py`: 3 tests passed."]

    def close(self):
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- tools -------------------------------------------------------------------------------
    def run_tool(self, name: str, args: dict) -> str:
        spec = TOOL_NAMES.get(name)
        if spec is None:
            self.invalid += 1
            return tool_error(f'No tool named "{name}" is currently available. Please use a tool from the available tool list.')
        missing = [k for k in spec["parameters"].get("required", []) if k not in args]
        if missing:
            self.invalid += 1
            lines = "\n".join(f"- {k}: Missing key" for k in missing)
            return tool_error(f'Invalid arguments for tool "{name}":\n{lines}\n\nArguments provided:\n'
                              f"{json.dumps(args, indent=2)}\n\nUpdate the arguments and call the tool again.")
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return tool_error(f"The {name} tool is not available for this task.")
        before = self.base.task_state()
        out = handler(args)
        if self.base.task_state() != before and not (self.base.intact() and self.base.grade() == 1.0):
            self.verified = False
        return out

    def resolve(self, path: str) -> Path | None:
        p = Path(path)
        p = (p if p.is_absolute() else self.work / p).resolve()
        work = self.work.resolve()
        return p if p == work or work in p.parents else None

    def tool_read(self, args):
        path = args["path"]
        p = self.resolve(path)
        if p is None or not p.exists():
            return tool_error(f"File not found: {path if Path(path).is_absolute() else self.work / path}")
        if p.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
            return f"Read directory {path}, entries 1-{len(entries)}\n" + "\n".join(entries)
        lines = p.read_text(errors="replace").splitlines()
        start = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or READ_LIMIT)
        chunk = lines[start - 1:start - 1 + limit]
        end = start + len(chunk) - 1
        text = f"Read file {path}, lines {start}-{end}\n" + "\n".join(f"{start + i}: {l}" for i, l in enumerate(chunk))
        if end < len(lines):
            text += f"\n[Output truncated. Continue reading with offset: {end + 1}]"
        return text

    def tool_edit(self, args):
        path = args["path"]
        p = self.resolve(path)
        shown = path if Path(path).is_absolute() else str(self.work / path)
        if p is None or not p.is_file():
            return tool_error(f"File not found: {shown}")
        text = p.read_text(errors="replace")
        count = text.count(args["oldString"]) if args["oldString"] else 0
        if count == 0:
            return tool_error(f"Could not find oldString in {shown}. It must match exactly, including whitespace and indentation.")
        if count > 1 and not args.get("replaceAll"):
            return tool_error(f"Found {count} matches for oldString, but expected exactly one. Add more surrounding "
                              "context to make oldString unique, or set replaceAll to true to replace every occurrence.")
        n = count if args.get("replaceAll") else 1
        p.write_text(text.replace(args["oldString"], args["newString"], n))
        return f"Edited {p.name} ({n} replacement{'s' if n != 1 else ''})"

    def tool_write(self, args):
        p = self.resolve(args["path"])
        if p is None:
            return tool_error(f"Cannot write outside the workspace: {args['path']}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return f"Wrote file successfully: {p.name}"

    def tool_glob(self, args):
        base = self.resolve(args.get("path") or str(self.work)) or self.work
        hits = sorted(str(h) for h in base.glob(args["pattern"]) if "__pycache__" not in h.parts)
        return "\n".join(hits) if hits else "No files found"

    def tool_grep(self, args):
        base = self.resolve(args.get("path") or str(self.work)) or self.work
        try:
            rx = re.compile(args["pattern"])
        except re.error as exc:
            return tool_error(f"Invalid regex: {exc}")
        files, total = [], 0
        for f in sorted(base.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            matches = [(i, l) for i, l in enumerate(f.read_text(errors="replace").splitlines(), 1) if rx.search(l)]
            if matches:
                total += len(matches)
                files.append(f"\n{f}:" + "".join(f"\n  Line {i}: {l}\n" for i, l in matches))
        return f"Found {total} matches" + "\n".join(files) if total else "No files found"

    def tool_shell(self, args):
        cmd = args["command"]
        work = str(self.work)
        sandbox = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                   "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--bind", work, work, "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                   # Fast successive edits can leave a same-size, same-second stale .pyc; keep the cache
                   # off the workspace so a corrected file is always what runs.
                   "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/pyc",
                   "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8", "--chdir", work]
        try:
            proc = subprocess.run(sandbox + ["/bin/bash", "-c", cmd], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=self.shell_timeout)
            out, rc = proc.stdout.decode(errors="replace")[:30000], proc.returncode
        except subprocess.TimeoutExpired:
            out, rc = f"Command timed out after {self.shell_timeout} seconds", 124
        if VERIFY_CMD.search(cmd):
            self.verified = rc == 0 and self.base.intact() and self.base.grade() == 1.0
        if rc != 0:
            return (out.rstrip("\n") or "(no output)") + f"\n\nExited with code {rc}"
        return out if out else "(no output)"

    def tool_execute(self, args):
        code = args["code"].lstrip()
        if re.match(r"(import|export)\b", code):
            return "'import' and 'export' may appear only with 'sourceType: module' (1:0)"
        if code.startswith("from "):
            return "Unexpected token (1:5)"
        ident = re.match(r"[A-Za-z_$][\w$]*", code)
        if ident:
            return f"ReferenceError: Unknown identifier '{ident.group(0)}'. (line 1, col 1)"
        return "Unexpected token (1:0)"


def call_message(calls):
    return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for n, a in calls]
