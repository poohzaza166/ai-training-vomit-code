"""Episodes for the mixed GRPO run: MiMo-V2.6-RL-oss code and terminal_bench tasks in their own containers,
and DeepSeek-V4 / GLM-5.1 reasoning prompts scored against the teacher's final answer.

MimoEpisode is project_env.ProjectEpisode with the six opencode tools pointed into a docker container
instead of a local folder, so the model sees exactly the format it was trained on (same system prompt,
tool schemas, tool-result text, compaction). Grading follows MiMo's own contracts:

  code (opensource-code)  reset every path test_patch touches to the base commit, git apply test_patch
                          (it carries the hidden tests and mimo_test_command.sh), run test_command;
                          success iff it exits 0. (mimoagent/environments/datasets/opensource_code.py)
  terminal_bench          write tests_files to /tests, run /tests/test.sh, score from
                          /logs/verifier/reward.json or reward.txt; timeout or no file = 0.
                          (uni_agent/tasks/terminal_bench/reward.py)

The hidden tests only enter the container after the agent's last turn. Containers run with no network
and capped memory; the container is the sandbox, so tool paths are not confined to the work dir.

MathEpisode is one turn with no tools. The label is the teacher's boxed answer, or the one listed choice
its answer names; the reward is agreement with it, which is only as right as the teacher was.
"""
from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import time
import uuid

from cutrain.envs.project_env import CLAIM_RE, FAIL_RE, TEMPLATE, TOOLS, ProjectEpisode, tool_error
from cutrain.serving.serve_adapter import parse_reply, to_template
from cutrain.training.train_grpo_gan import ANSWER_RE, BOXED_RE, _norm_answer
from cutrain.training.train_orpo import ASSISTANT

REGISTRY = "docker.io/xiaomimimo/mimo-v2.6-rl-oss"
CONTAINER_MEMORY, CONTAINER_CPUS = "2g", "2"
VERIFIER_TIMEOUT_CAP = 600
GREP_LIMIT, GLOB_LIMIT = 200, 500
SKIP_DIRS = (".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache")


def hub_image(dataset_image: str) -> str:
    """format-code-task-001457:latest -> docker.io/xiaomimimo/...:format-code-task-001457
    (image-mapping.jsonl's rule: the repository name becomes the tag)."""
    return f"{REGISTRY}:{dataset_image.split(':', 1)[0]}"


def ensure_image(image: str, timeout: int = 3600) -> None:
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    proc = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"docker pull {image} failed: {proc.stderr.strip()[-500:]}")


def patch_touched_files(patch: str) -> list[str]:
    """Paths a unified diff creates, modifies or deletes (a/ and b/ sides, /dev/null dropped)."""
    files = []
    for line in patch.splitlines():
        m = re.match(r"^(?:---|\+\+\+) (?:a/|b/)?(.+?)(?:\t.*)?$", line)
        if m and m.group(1) != "/dev/null" and m.group(1) not in files:
            files.append(m.group(1))
    return files


def reset_test_files_cmd(patch: str, base_ref: str) -> str | None:
    """MiMo's per-file reset: restore files that exist at base, delete ones the patch adds, so an agent
    file at a path the patch creates cannot make git apply fail with 'already exists'."""
    touched = patch_touched_files(patch)
    if not touched or not base_ref:
        return None
    return ("while IFS= read -r tf; do\n"
            '  [ -z "$tf" ] && continue\n'
            f'  if git cat-file -e {base_ref}:"$tf" 2>/dev/null; then\n'
            f'    git checkout {base_ref} -- "$tf" 2>/dev/null || true\n'
            "  else\n"
            '    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n'
            '    rm -f "$tf"\n'
            "  fi\n"
            "done <<'EOF_RESET_TEST_FILES'\n" + "\n".join(touched) + "\nEOF_RESET_TEST_FILES\n")


def parse_reward_files(reward_json: str | None, reward_text: str | None) -> float:
    """Harbor's reward-file contract, as uni_agent reads it. Raises ValueError when neither is usable."""
    if reward_json is not None:
        payload = json.loads(reward_json)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("reward.json must be a non-empty object")
        rewards = {str(k): float(v) for k, v in payload.items()}
        if "reward" in rewards:
            return rewards["reward"]
        if len(rewards) == 1:
            return next(iter(rewards.values()))
        raise ValueError("reward.json has several metrics and no 'reward'")
    if reward_text is not None and reward_text.strip():
        return float(reward_text.strip())
    raise ValueError("verifier wrote neither reward.json nor reward.txt")


class Container:
    """One task container. Every command goes through `docker exec`."""

    def __init__(self, image: str, cwd: str, network: bool = False):
        self.name = f"mimo-{uuid.uuid4().hex[:12]}"
        self.cwd = cwd
        cmd = ["docker", "run", "-d", "--name", self.name, "--memory", CONTAINER_MEMORY,
               "--memory-swap", CONTAINER_MEMORY, "--cpus", CONTAINER_CPUS, "--pids-limit", "512",
               "-w", cwd, "--entrypoint", "sleep"]
        if not network:
            cmd += ["--network", "none"]
        proc = subprocess.run(cmd + [image, "infinity"], capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"docker run {image} failed: {proc.stderr.strip()[-500:]}")

    def run(self, script: str, *args: str, stdin: str | bytes | None = None, timeout: int = 180,
            cwd: str | None = None) -> tuple[int, str]:
        """bash -c script with positional args; (returncode, merged output). 124 on timeout."""
        data = stdin.encode() if isinstance(stdin, str) else stdin
        cmd = ["docker", "exec", "-i", "-w", cwd or self.cwd, self.name, "timeout", "-k", "5", str(timeout),
               "bash", "-c", script, "_", *args]
        try:
            proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout} seconds\n"
        return proc.returncode, proc.stdout.decode(errors="replace")

    def write(self, path: str, data: str | bytes) -> tuple[int, str]:
        return self.run('mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', path, stdin=data)

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=120)


class MimoEpisode(ProjectEpisode):
    """Episode interface for train_long_grpo.rollout, inside a MiMo task container."""

    def __init__(self, task: dict, compact_at: int = 9500, shell_timeout: int = 180):
        self.task, self.compact_at, self.shell_timeout = task, compact_at, shell_timeout
        self.kind = task["kind"]
        ensure_image(task["image"])
        self.box = Container(task["image"], task["cwd"])
        self.cwd = task["cwd"]
        self.tools = TOOLS
        self.system = TEMPLATE["system_template"].format(workdir=self.cwd, date=time.strftime(TEMPLATE["date_format"]),
                                                          model="cu-app")
        user = json.dumps(task["problem"]) if TEMPLATE["user_quoted"] else task["problem"]
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        self.compaction = None
        self.checkpoint = False
        self.compactions = self.compaction_failures = 0
        self.calls = self.invalid = self.repeats = self.dir_writes = 0
        self.seen: set = set()
        self.created: list[str] = []
        self.finished = False
        self.final_text = ""
        self.base_ref = ""
        try:
            self.setup()
        except Exception:
            self.close()
            raise

    def setup(self):
        if self.kind == "code":
            rc, out = self.box.run("git config --global --add safe.directory '*' && "
                                   "(git rev-parse --git-dir >/dev/null 2>&1 || "
                                   "(git init -q && git add -A && git -c user.email=b@b -c user.name=b "
                                   "commit -q -m baseline --allow-empty)) && git rev-parse HEAD")
            ref = out.strip().split()[-1] if out.strip() else ""
            if rc != 0 or len(ref) != 40:
                raise RuntimeError(f"{self.task['id']}: no base commit: {out[-300:]}")
            self.base_ref = ref
        else:
            self.box.run("mkdir -p /logs/agent /logs/verifier /logs/artifacts && "
                         "chmod 777 /logs/agent /logs/verifier /logs/artifacts", cwd="/")

    # --- tools: same names, arguments and result text as project_env, run in the container ---------
    def path(self, path: str) -> str:
        return path if path.startswith("/") else f"{self.cwd.rstrip('/')}/{path}"

    def snapshot(self) -> set[str]:
        return set()

    def note_created(self, before: set[str]):
        pass

    def tool_read(self, args):
        path, full = args["path"], self.path(args["path"])
        rc, out = self.box.run('if [ -d "$1" ]; then echo D; ls -1Ap -- "$1"; elif [ -f "$1" ]; then echo F; '
                               'cat -- "$1"; else exit 3; fi', full)
        if rc != 0:
            return tool_error(f"File not found: {full}")
        kind, _, body = out.partition("\n")
        if kind == "D":
            entries = [e for e in body.splitlines() if e]
            if not entries:
                return f"Read directory {path}, 0 entries"
            return f"Read directory {path}, entries 1-{len(entries)}\n" + "\n".join(entries)
        lines = body.splitlines()
        start = max(int(args.get("offset") or 1), 1)
        chunk = lines[start - 1:start - 1 + int(args.get("limit") or 2000)]
        end = start + len(chunk) - 1
        text = f"Read file {path}, lines {start}-{end}\n" + "\n".join(f"{start + i}: {l}" for i, l in enumerate(chunk))
        if end < len(lines):
            text += f"\n[Output truncated. Continue reading with offset: {end + 1}]"
        return text

    def tool_write(self, args):
        full = self.path(args["path"])
        rc, _ = self.box.run('[ -d "$1" ]', full)
        if rc == 0:
            self.dir_writes += 1
            return tool_error(f"Unable to write {args['path']}")
        existed = self.box.run('[ -e "$1" ]', full)[0] == 0
        rc, out = self.box.write(full, args["content"])
        if rc != 0:
            return tool_error(f"Unable to write {args['path']}: {out.strip()[-200:]}")
        return f"{'Wrote' if existed else 'Created'} file successfully: {args['path']}"

    def tool_edit(self, args):
        full = self.path(args["path"])
        rc, text = self.box.run('[ -f "$1" ] && cat -- "$1"', full)
        if rc != 0:
            return tool_error(f"File not found: {full}")
        if args["oldString"] == args["newString"]:
            return tool_error("No changes to apply: oldString and newString are identical.")
        count = text.count(args["oldString"]) if args["oldString"] else 0
        if count == 0:
            return tool_error(f"Could not find oldString in {full}. It must match exactly, including whitespace and indentation.")
        if count > 1 and not args.get("replaceAll"):
            return tool_error(f"Found {count} matches for oldString, but expected exactly one. Add more surrounding "
                              "context to make oldString unique, or set replaceAll to true to replace every occurrence.")
        n = count if args.get("replaceAll") else 1
        rc, out = self.box.write(full, text.replace(args["oldString"], args["newString"], n))
        if rc != 0:
            return tool_error(f"Unable to write {full}: {out.strip()[-200:]}")
        return f"Edited {args['path']} ({n} replacement{'s' if n != 1 else ''})"

    def tool_glob(self, args):
        base = self.path(args.get("path") or self.cwd)
        skip = "|".join(re.escape(d) for d in SKIP_DIRS)
        rc, out = self.box.run('cd -- "$1" 2>/dev/null || exit 3; shopt -s globstar nullglob dotglob; '
                               'for f in $2; do printf "%s\\n" "$PWD/$f"; done', base, args["pattern"])
        hits = sorted(h for h in out.splitlines() if h and not re.search(rf"/({skip})/", h))[:GLOB_LIMIT]
        return "\n".join(hits) if rc == 0 and hits else "No files found"

    def tool_grep(self, args):
        base = self.path(args.get("path") or self.cwd)
        try:
            re.compile(args["pattern"])
        except re.error as exc:
            return tool_error(f"Invalid regex: {exc}")
        excludes = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in SKIP_DIRS)
        rc, out = self.box.run(f'grep -rnIE {excludes} -e "$1" -- "$2" 2>/dev/null | head -n {GREP_LIMIT}',
                               args["pattern"], base)
        by_file: dict[str, list[str]] = {}
        for line in out.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                by_file.setdefault(parts[0], []).append(f"\n  Line {parts[1]}: {parts[2]}\n")
        total = sum(len(v) for v in by_file.values())
        if not total:
            return "No files found"
        return f"Found {total} matches" + "\n".join(f"\n{f}:" + "".join(v) for f, v in by_file.items())

    def tool_shell(self, args):
        rc, out = self.box.run(args["command"], timeout=self.shell_timeout)
        if rc == 124:
            out = out or f"Command timed out after {self.shell_timeout} seconds\n"
        lines = out.splitlines()
        if len(lines) > 2000:
            out = "\n".join(lines[-2000:]) + f"\n\n[showing lines {len(lines) - 1999}-{len(lines)} of {len(lines)}]"
        elif len(out) > 51200:
            out = out[-51200:] + f"\n\n[showing lines 1-{len(lines)} of {len(lines)}]"
        if rc != 0:
            return (out.rstrip("\n") or "(no output)") + f"\n\nExited with code {rc}"
        return out if out else "(no output)"

    # --- grading --------------------------------------------------------------------------------------
    def verify(self) -> dict:
        timeout = min(int(self.task["verifier_timeout"]), VERIFIER_TIMEOUT_CAP)
        if self.kind == "code":
            reset = reset_test_files_cmd(self.task["test_patch"], self.base_ref)
            if reset and self.box.run(reset)[0] != 0:
                return {"score": 0.0, "error": "reset_tests_failed"}
            remote = "/tmp/_mimo_test.patch"
            self.box.write(remote, self.task["test_patch"])
            rc, out = self.box.run(f"git apply --verbose {remote}")
            if rc != 0:
                return {"score": 0.0, "error": "apply_test_patch_failed", "output": out[-1500:]}
            rc, out = self.box.run(self.task["test_command"], timeout=timeout)
            return {"score": 1.0 if rc == 0 else 0.0, "returncode": rc, "output": out[-1500:],
                    **({"error": "verifier_timeout"} if rc == 124 else {})}
        self.box.run("rm -rf /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier", cwd="/")
        for name, b64 in self.task["tests_files"].items():
            self.box.write(f"/tests/{name}", base64.b64decode(b64))
        self.box.run("chmod +x /tests/test.sh", cwd="/")
        rc, out = self.box.run("/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1", timeout=timeout)
        if rc == 124:
            return {"score": 0.0, "error": "verifier_timeout"}
        _, rj = self.box.run('[ -f /logs/verifier/reward.json ] && cat /logs/verifier/reward.json', cwd="/")
        rj_ok = self.box.run("[ -f /logs/verifier/reward.json ]", cwd="/")[0] == 0
        _, rt = self.box.run('[ -f /logs/verifier/reward.txt ] && cat /logs/verifier/reward.txt', cwd="/")
        rt_ok = self.box.run("[ -f /logs/verifier/reward.txt ]", cwd="/")[0] == 0
        _, log = self.box.run("tail -c 1500 /logs/verifier/test-stdout.txt", cwd="/")
        try:
            score = parse_reward_files(rj if rj_ok else None, rt if rt_ok else None)
        except ValueError as exc:
            return {"score": 0.0, "error": str(exc), "output": log}
        return {"score": max(0.0, min(1.0, score)), "returncode": rc, "output": log}

    def result(self) -> dict:
        v = self.verify()
        success = v["score"] >= 1.0
        claims = bool(CLAIM_RE.search(self.final_text)) and not FAIL_RE.search(self.final_text)
        false_claim = claims and not success
        # Terminal verifiers can give partial credit; code is pass/fail. The small process terms are
        # what separates members of a group that all failed.
        reward = (2.0 * v["score"] - 0.5 * false_claim - 0.5 * self.compaction_failures
                  - 0.1 * self.invalid - 0.1 * self.repeats - 0.1 * self.dir_writes)
        return {"reward": reward, "success": bool(success), "tests": v["score"], "intact": True,
                "verified": bool(success), "false_claim": bool(false_claim), "abstained": False,
                "premature_stop": False, "turns": self.calls, "invalid": self.invalid, "repeats": self.repeats,
                "dir_writes": self.dir_writes, "compactions": self.compactions,
                "compaction_failures": self.compaction_failures, "verifier": v, "task": self.task["id"],
                "source": self.kind}

    def close(self):
        box = getattr(self, "box", None)
        if box is not None:
            box.close()


def norm_choice(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


ANSWER_SECTION_RE = re.compile(r"#+\s*(?:final\s+)?answer\s*\n+\s*(.+?)\s*$", re.I | re.S)


def final_answer(text: str) -> tuple[str | None, bool]:
    """(the completion's final answer, whether it was boxed). Boxed first; then a trailing '# Answer'
    section, the PRM800K habit the model has (it answered "Compute C(14,11)" with '# Answer\n\n364'
    and no box); then 'answer: x'. Only text after the reasoning counts."""
    final = text.rsplit("</think>", 1)[-1]
    boxed = BOXED_RE.findall(final)
    if boxed:
        return boxed[-1], True
    section = ANSWER_SECTION_RE.search(final)
    if section:
        return section.group(1).strip().splitlines()[0].strip("$ "), False
    found = ANSWER_RE.findall(final)
    return (found[-1], False) if found else (None, False)


def math_score(text: str, label: dict) -> tuple[float, bool]:
    """(1.0 when the completion's final answer matches the teacher's, whether it used the box asked for)."""
    answer, boxed = final_answer(text)
    if label["type"] == "boxed":
        return (1.0 if answer is not None and _norm_answer(answer) == _norm_answer(label["answer"]) else 0.0), boxed
    if answer is not None:
        return (1.0 if norm_choice(answer) == norm_choice(label["answer"]) else 0.0), boxed
    final = text.rsplit("</think>", 1)[-1]
    named = [c for c in label["choices"] if norm_choice(c) and re.search(rf"\b{re.escape(norm_choice(c))}\b",
                                                                          norm_choice(final))]
    return (1.0 if named == [label["answer"]] else 0.0), False


class MathEpisode:
    """One reasoning prompt, one answer, no tools."""

    def __init__(self, task: dict):
        self.task = task
        self.tools = []
        self.messages = [{"role": "user", "content": task["problem"]}]
        self.finished = False
        self.text = ""
        self.truncated = False
        self.calls = 0

    def prompt(self, tok) -> str:
        return tok.apply_chat_template(to_template(self.messages), tokenize=False) + ASSISTANT

    def step(self, text: str):
        self.calls += 1
        content, reasoning, _, _ = parse_reply(text, [])
        self.messages.append({"role": "assistant", "content": content or None})
        self.text, self.finished = text, True

    def reject_generation(self, text: str):
        self.calls += 1
        self.truncated = True
        self.messages.append({"role": "assistant", "content": text or None})
        self.text, self.finished = text, True

    def result(self) -> dict:
        correct, boxed = (0.0, False) if self.truncated else math_score(self.text, self.task["label"])
        # Correctness carries the reward; the box the prompt asked for is a small extra.
        reward = 2.0 * correct + 0.2 * (correct and boxed) - 0.3 * self.truncated
        return {"reward": reward, "success": bool(correct), "tests": correct, "intact": True,
                "verified": bool(correct), "false_claim": False, "abstained": False, "premature_stop": False,
                "turns": self.calls, "invalid": 0, "repeats": 0, "truncated": self.truncated,
                "task": self.task["id"], "source": self.task["kind"], "label": self.task["label"]["answer"]}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_episode(task: dict, compact_at: int = 9500):
    if task["kind"] in ("code", "terminal"):
        return MimoEpisode(task, compact_at=compact_at)
    return MathEpisode(task)


def load_tasks(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
