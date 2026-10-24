"""Same app-build trial as build_app_with_opencode.py, through the Cline CLI instead of opencode.

The model receives the spec file verbatim, once. Cline runs in the same bubblewrap sandbox (only the
project folder visible under /home besides the tools' own state), pointed at serve_adapter.py through
its openai-compatible provider (configured in --cline-home by `cline auth`). Nothing is fed back.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from cutrain.evaluation.build_app_with_opencode import SKIP_DIRS, independent_tests

HOME = Path.home()
NODE = HOME / ".local/opt/node/bin"
CLINE = HOME / ".local/opt/npm-global/bin/cline"


def sandbox(project: Path, cline_home: Path) -> list[str]:
    return ["bwrap", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
            "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/home",
            "--ro-bind", str(HOME / ".local/opt"), str(HOME / ".local/opt"),
            "--bind", str(cline_home), str(cline_home),
            "--bind", str(project), str(project), "--chdir", str(project)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cline-home", default=str(HOME / ".local/share/cline-eval"))
    ap.add_argument("--model", default="cu-app")
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=False)
    cline_home = Path(args.cline_home)
    prompt = Path(args.spec).read_text()
    env = {**os.environ, "PATH": f"{NODE}:{os.environ['PATH']}", "HOME": str(HOME)}
    cmd = [str(CLINE), "--json", "--auto-approve", "true", "-c", str(project),
           "--config", str(cline_home / "config"), "--data-dir", str(cline_home / "data"),
           "-P", "openai-compatible", "-m", args.model, "-t", str(args.timeout), prompt]
    started = time.monotonic()
    try:
        proc = subprocess.run(sandbox(project, cline_home) + cmd, env=env, capture_output=True, text=True,
                              timeout=args.timeout + 300)
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = None, True
    messages = []
    for line in stdout.splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    files = sorted(str(p.relative_to(project)) for p in project.rglob("*")
                   if p.is_file() and not SKIP_DIRS & set(p.relative_to(project).parts))
    record = {"harness": "cline", "spec": args.spec, "seconds": round(time.monotonic() - started, 1),
              "exit_code": rc, "timed_out": timed_out, "messages": len(messages), "files": files,
              "tests_rerun": independent_tests(project), "stdout_tail": stdout[-4000:],
              "stderr_tail": stderr[-4000:], "events": messages}
    Path(args.log).write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ("events", "stdout_tail")}, indent=1), flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
