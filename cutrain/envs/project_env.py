"""Whole-project episodes in opencode's format: empty folder + plain-language outline -> finished project.

Everything the model sees is reproduced from captured opencode v2.0.27 traffic with the trimmed tool
set (data/opencode_trimmed_template.json): system prompt, the six tool schemas, the quoted user
message, tool-result text (including errors and large-output truncation) and context compaction
(data/opencode_compaction.json): when the rendered prompt passes the threshold, the next turn is
opencode's summary request; a summary without the template headings gets opencode's one retry, then
the session fails, as in opencode. After a summary the conversation restarts from the checkpoint.

Grading (project_grader) runs after the episode on the folder as left: the model's own tests rerun
with coverage, hidden CLI checks from the task, structure, owner guide, and whether a real spec draft
was written before any code. Shell commands run in bubblewrap with network (package installs) and a
persistent pip cache; nothing outside the project folder is writable.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from cutrain.envs import project_grader as grader
from cutrain.envs.project_tasks import Task
from cutrain.paths import REPO
from cutrain.serving.serve_adapter import parse_reply, to_template
from cutrain.training.train_orpo import ASSISTANT

TEMPLATE = json.loads((REPO / "data/opencode_trimmed_template.json").read_text())
COMPACTION = json.loads((REPO / "data/opencode_compaction.json").read_text())
TOOLS = TEMPLATE["tools"]
TOOL_SPECS = {t["function"]["name"]: t["function"] for t in TOOLS}
MODEL_NAME = "cu-app"
PIP_CACHE = Path.home() / ".cache/project-env-pip"
CLAIM_RE = re.compile(r"\b(pass(ed|es)?|complete[sd]?|success(ful(ly)?)?|done|works|built)\b", re.I)
FAIL_RE = re.compile(r"\b(fail(ed|s|ing)?|not pass|could not|unable|error|incomplete)\b", re.I)
SUMMARY_HEADINGS = ("## Objective", "## Next Move")
SHELL_LINE_LIMIT, SHELL_CHAR_LIMIT, READ_LIMIT, RESULT_CHARS_IN_TRANSCRIPT = 2000, 51200, 2000, 1250


def tool_error(message: str) -> str:
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []},
                      separators=(",", ":"))


def serialize_history(messages: list[dict]) -> str:
    """opencode's transcript format for the first compaction request."""
    lines = []
    for m in messages:
        content = m.get("content")
        if m["role"] == "user":
            lines.append(f"[User]: {content}")
        elif m["role"] == "assistant":
            if content:
                lines.append(f"[Assistant]: {content}")
            for c in m.get("tool_calls") or []:
                args = json.dumps(json.loads(c["function"]["arguments"]), separators=(",", ":"), ensure_ascii=False)
                lines.append(f"[Assistant tool call]: {c['function']['name']}({args})")
        elif m["role"] == "tool":
            text = content or ""
            if len(text) > RESULT_CHARS_IN_TRANSCRIPT:
                text = text[:RESULT_CHARS_IN_TRANSCRIPT] + "\n[truncated]"
            lines.append(f"[Tool result]: {text}")
    return "\n".join(lines)


class ProjectEpisode:
    """Episode interface for train_long_grpo.rollout: prompt(tok), step(text), finished, result()."""

    def __init__(self, task: Task, compact_at: int = 9500, shell_timeout: int = 180):
        self.task, self.compact_at, self.shell_timeout = task, compact_at, shell_timeout
        self.root = Path(tempfile.mkdtemp(prefix="model-apps-"))
        self.work = self.root / task.slug
        self.work.mkdir()
        self.tools = TOOLS
        self.system = TEMPLATE["system_template"].format(workdir=self.work, date=time.strftime(TEMPLATE["date_format"]),
                                                          model=MODEL_NAME)
        user = json.dumps(task.outline) if TEMPLATE["user_quoted"] else task.outline
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        self.compaction = None          # None, or {"request": [...], "retried": bool}
        self.checkpoint = False
        self.compactions = self.compaction_failures = 0
        self.calls = self.invalid = self.repeats = self.dir_writes = 0
        self.seen: set = set()
        self.created: list[str] = []    # relative paths in order of creation (tools or shell)
        self.finished = False
        self.final_text = ""

    # --- conversation ---------------------------------------------------------------------------
    def request_messages(self) -> list[dict]:
        return self.compaction["request"] if self.compaction else self.messages

    def prompt(self, tok) -> str:
        text = tok.apply_chat_template(to_template(self.request_messages()), tools=self.tools, tokenize=False) + ASSISTANT
        if self.compaction is None and len(tok(text, add_special_tokens=False).input_ids) > self.compact_at:
            self.begin_compaction()
            text = tok.apply_chat_template(to_template(self.request_messages()), tools=self.tools,
                                           tokenize=False) + ASSISTANT
        return text

    def begin_compaction(self):
        if self.checkpoint:
            request = self.messages + [{"role": "user", "content": COMPACTION["update_instruction"]}]
        else:
            request = [self.messages[0], {"role": "user", "content": serialize_history(self.messages[1:])},
                       {"role": "user", "content": COMPACTION["first_instruction"]}]
        self.compaction = {"request": request, "retried": False}

    def finish_compaction(self, text: str):
        content, _, calls, _ = parse_reply(text, self.tools)
        if not calls and all(h in content for h in SUMMARY_HEADINGS):
            wrapper = COMPACTION["checkpoint_wrapper"].replace("{summary}", content.strip())
            self.messages = [self.messages[0], {"role": "user", "content": wrapper}]
            self.compaction, self.checkpoint = None, True
            self.compactions += 1
            return
        if not self.compaction["retried"]:
            self.compaction["request"] = self.compaction["request"] + [
                {"role": "assistant", "content": content or None},
                {"role": "user", "content": COMPACTION["template_retry"]}]
            self.compaction["retried"] = True
            return
        self.compaction_failures += 1
        self.finished = True  # opencode ends the session when compaction fails twice

    def step(self, text: str):
        self.calls += 1
        if self.compaction:
            return self.finish_compaction(text)
        content, _, calls, problems = parse_reply(text, self.tools)
        if not calls:
            self.messages.append({"role": "assistant", "content": content or None})
            self.invalid += any("malformed" in p for p in problems)
            self.finished, self.final_text = True, content
            return
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for call in calls:
            name, args = call["function"]["name"], json.loads(call["function"]["arguments"])
            output = self.run_tool(name, args)
            signature = (name, json.dumps(args, sort_keys=True), output)
            # Count any identical repeat, not just repeated errors: re-running a build or the test
            # suite ("12 passed" again) is the exact "keeps calling tools" habit. This matches the
            # terminal env, which counts unchanged repeats regardless of outcome.
            self.repeats += signature in self.seen
            self.seen.add(signature)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

    def reject_generation(self, text: str):
        self.calls += 1
        self.invalid += 1
        if self.compaction:
            return self.finish_compaction(text)
        self.messages.append({"role": "assistant", "content": text or None})
        self.finished, self.final_text = True, text

    # --- tools ----------------------------------------------------------------------------------
    def run_tool(self, name: str, args: dict) -> str:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            self.invalid += 1
            return tool_error(f'No tool named "{name}" is currently available. Please use a tool from the available tool list.')
        missing = [k for k in spec["parameters"].get("required", []) if k not in args]
        if missing:
            self.invalid += 1
            lines = "\n".join(f"- {k}: Missing key" for k in missing)
            return tool_error(f'Invalid arguments for tool "{name}":\n{lines}\n\nArguments provided:\n'
                              f"{json.dumps(args, indent=2)}\n\nUpdate the arguments and call the tool again.")
        before = self.snapshot()
        out = getattr(self, f"tool_{name}")(args)
        self.note_created(before)
        return out

    def snapshot(self) -> set[str]:
        return {str(p.relative_to(self.work)) for p in self.work.rglob("*")
                if p.is_file() and not grader.SKIP & set(p.relative_to(self.work).parts)}

    def note_created(self, before: set[str]):
        for rel in sorted(self.snapshot() - before):
            if rel not in self.created:
                self.created.append(rel)

    def resolve(self, path: str) -> Path | None:
        p = Path(path)
        p = (p if p.is_absolute() else self.work / p).resolve()
        work = self.work.resolve()
        return p if p == work or work in p.parents else None

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.work.resolve()))

    def tool_read(self, args):
        path, p = args["path"], self.resolve(args["path"])
        if p is None or not p.exists():
            return tool_error(f"File not found: {path if Path(path).is_absolute() else self.work / path}")
        if p.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
            if not entries:  # as captured from opencode on the empty project folder
                return f"Read directory {path}, 0 entries"
            return f"Read directory {path}, entries 1-{len(entries)}\n" + "\n".join(entries)
        lines = p.read_text(errors="replace").splitlines()
        start = max(int(args.get("offset") or 1), 1)
        chunk = lines[start - 1:start - 1 + int(args.get("limit") or READ_LIMIT)]
        end = start + len(chunk) - 1
        text = f"Read file {path}, lines {start}-{end}\n" + "\n".join(f"{start + i}: {l}" for i, l in enumerate(chunk))
        if end < len(lines):
            text += f"\n[Output truncated. Continue reading with offset: {end + 1}]"
        return text

    def tool_write(self, args):
        p = self.resolve(args["path"])
        if p is None or p.is_dir():
            self.dir_writes += p is not None
            return tool_error(f"Unable to write {args['path']}")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return f"{'Wrote' if existed else 'Created'} file successfully: {self.rel(p)}"

    def tool_edit(self, args):
        path, p = args["path"], self.resolve(args["path"])
        shown = path if Path(path).is_absolute() else str(self.work / path)
        if p is None or not p.is_file():
            return tool_error(f"File not found: {shown}")
        if args["oldString"] == args["newString"]:
            return tool_error("No changes to apply: oldString and newString are identical.")
        text = p.read_text(errors="replace")
        count = text.count(args["oldString"]) if args["oldString"] else 0
        if count == 0:
            return tool_error(f"Could not find oldString in {shown}. It must match exactly, including whitespace and indentation.")
        if count > 1 and not args.get("replaceAll"):
            return tool_error(f"Found {count} matches for oldString, but expected exactly one. Add more surrounding "
                              "context to make oldString unique, or set replaceAll to true to replace every occurrence.")
        n = count if args.get("replaceAll") else 1
        p.write_text(text.replace(args["oldString"], args["newString"], n))
        return f"Edited {self.rel(p)} ({n} replacement{'s' if n != 1 else ''})"

    def tool_glob(self, args):
        base = self.resolve(args.get("path") or str(self.work)) or self.work
        hits = sorted(str(h) for h in base.glob(args["pattern"]) if not grader.SKIP & set(h.parts))
        return "\n".join(hits) if hits else "No files found"

    def tool_grep(self, args):
        base = self.resolve(args.get("path") or str(self.work)) or self.work
        try:
            rx = re.compile(args["pattern"])
        except re.error as exc:
            return tool_error(f"Invalid regex: {exc}")
        files, total = [], 0
        for f in sorted(base.rglob("*")):
            if not f.is_file() or grader.SKIP & set(f.relative_to(base).parts):
                continue
            hits = [(i, l) for i, l in enumerate(f.read_text(errors="replace").splitlines(), 1) if rx.search(l)]
            if hits:
                total += len(hits)
                files.append(f"\n{f}:" + "".join(f"\n  Line {i}: {l}\n" for i, l in hits))
        return f"Found {total} matches" + "\n".join(files) if total else "No files found"

    def tool_shell(self, args):
        work = str(self.work)
        PIP_CACHE.mkdir(parents=True, exist_ok=True)
        cmd = ["bwrap", "--unshare-all", "--share-net", "--die-with-parent", "--new-session",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
               "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", work, work,
               "--bind", str(PIP_CACHE), "/tmp/pip-cache", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
               "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8", "--setenv", "PIP_CACHE_DIR", "/tmp/pip-cache",
               "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/pyc", "--chdir", work, "/bin/bash", "-c", args["command"]]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.shell_timeout)
            out, rc = proc.stdout.decode(errors="replace"), proc.returncode
        except subprocess.TimeoutExpired:
            out, rc = f"Command timed out after {self.shell_timeout} seconds\n", 124
        lines = out.splitlines()
        saved = f"/home/pooh/.local/share/opencode/shell/{hashlib.sha1(work.encode()).hexdigest()}/sh_{uuid.uuid4().hex[:24]}.out"
        if len(lines) > SHELL_LINE_LIMIT:
            first = len(lines) - SHELL_LINE_LIMIT + 1
            out = "\n".join(lines[-SHELL_LINE_LIMIT:]) + f"\n\n[showing lines {first}-{len(lines)} of {len(lines)}; full output saved to {saved}]"
        elif len(out) > SHELL_CHAR_LIMIT:
            out = out[-SHELL_CHAR_LIMIT:] + f"\n\n[showing lines 1-{len(lines)} of {len(lines)}; full output saved to {saved}]"
        if rc != 0:
            return (out.rstrip("\n") or "(no output)") + f"\n\nExited with code {rc}"
        return out if out else "(no output)"

    # --- grading --------------------------------------------------------------------------------
    def spec_first(self) -> dict:
        """The first created file must be a markdown spec draft, written before any code."""
        first_md = next((r for r in self.created if r.lower().endswith(".md")), None)
        first_code = next((i for i, r in enumerate(self.created) if r.endswith(".py")), len(self.created))
        if first_md is None or self.created.index(first_md) > first_code:
            return {"ok": False, "file": first_md, "reason": "no markdown spec before the first code file"}
        quality = grader.spec_quality((self.work / first_md).read_text(errors="replace"), self.task.keywords)
        return {**quality, "file": first_md}

    def result(self) -> dict:
        own = grader.run_own_tests(self.work)
        checks = grader.run_checks(self.work, self.task.package, self.task.checks, sample=getattr(self.task, "sample", None))
        structure = grader.structure(self.work)
        spec = self.spec_first()
        readme = grader.readme_ok(self.work)
        tests_ok = own["passed"] and own["coverage"] >= 70
        functional = checks["passed"] / max(checks["total"], 1)
        claims = bool(CLAIM_RE.search(self.final_text)) and not FAIL_RE.search(self.final_text)
        success = (self.finished and not self.compaction_failures and spec["ok"] and structure["ok"] and readme
                   and tests_ok and functional == 1.0)
        false_claim = claims and not (tests_ok and functional == 1.0)
        reward = (2.0 * success + 0.3 * spec["ok"] + 0.2 * structure["ok"] + 0.2 * readme + 0.5 * tests_ok
                  + 0.8 * functional - 1.0 * false_claim - 0.5 * self.compaction_failures
                  - 0.1 * self.invalid - 0.1 * self.repeats - 0.1 * self.dir_writes)
        return {"reward": reward, "success": bool(success), "tests": functional, "intact": True,
                "verified": tests_ok, "false_claim": bool(false_claim), "abstained": False, "premature_stop": False,
                "turns": self.calls, "invalid": self.invalid, "repeats": self.repeats, "dir_writes": self.dir_writes,
                "compactions": self.compactions, "compaction_failures": self.compaction_failures,
                "spec": spec, "structure": structure, "readme": readme, "own_tests": own,
                "checks": {"passed": checks["passed"], "total": checks["total"]}, "created": self.created,
                "task": self.task.slug}

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
