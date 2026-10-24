"""Have the trained model build an app through the real opencode harness from a plain-language spec.

The model receives the spec file verbatim, once, in a single opencode session. No process instructions,
milestones, file names, libraries or code are given: planning, design and verification are the
model's own. Afterwards this driver collects evidence for evaluation (order of file writes, files
produced, an independent rerun of whatever tests the model wrote). Nothing is fed back to the model.

opencode runs in bubblewrap with the project folder and opencode's own state writable (network
allowed for package installs).

  python -m cutrain.evaluation.build_app_with_opencode --base-url http://127.0.0.1:8011/v1 --model cu-app \\
      --spec specs/dnd-music-bot.md --project out/apps/dnd-music-bot --log out/apps/dnd-music-bot.build.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from cutrain.evaluation.opencode_agent_test import OPENCODE, STATE_DIRS

SKIP_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".git"}


def sandbox(project: Path) -> list[str]:
    """Only the project folder is visible from /home: the training repo, specs and evaluation
    criteria stay hidden. System files, opencode's binary and its own state are mounted."""
    cmd = ["bwrap", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
           # /etc/resolv.conf links into /run; only the resolver is needed for package installs.
           "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
           "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/home",
           "--ro-bind", str(OPENCODE.parent.parent), str(OPENCODE.parent.parent)]
    for d in STATE_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(d), str(d)]
    return cmd + ["--bind", str(project), str(project), "--chdir", str(project)]


def write_order(events: list[dict], project: Path) -> list[str]:
    """Files in the order the model first created or edited them through write/edit tools."""
    order = []
    for e in events:
        if e.get("type") != "tool_use" or e["part"].get("tool") not in ("write", "edit"):
            continue
        path = str(e["part"].get("state", {}).get("input", {}).get("path", ""))
        rel = os.path.relpath(path, project) if os.path.isabs(path) else path
        if rel not in order:
            order.append(rel)
    return order


def independent_tests(project: Path) -> dict:
    """Rerun the model's own tests with whatever environment it created; report, never fix."""
    for venv in (".venv", "venv"):
        py = project / venv / "bin/python"
        if py.exists():
            break
    else:
        return {"ran": False, "reason": "no virtualenv in the project"}
    try:
        proc = subprocess.run(sandbox(project) + [str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              capture_output=True, text=True, timeout=600)
        return {"ran": True, "rc": proc.returncode, "tail": (proc.stdout + proc.stderr).strip().splitlines()[-20:]}
    except subprocess.TimeoutExpired:
        return {"ran": True, "rc": None, "tail": ["timed out"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True, help="plain-language spec, sent verbatim")
    ap.add_argument("--project", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--context", type=int, default=12000,
                    help="context limit told to opencode; it compacts near context - output (~10k prompt tokens, "
                         "matching project_env's training threshold)")
    ap.add_argument("--disable-tools", default="question,skill,subagent,webfetch,websearch,execute",
                    help="opencode built-in tools switched off in its config (the trained setup); '' keeps all")
    ap.add_argument("--output", type=int, default=2048, help="max tokens per model reply told to opencode")
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="seconds for the whole session")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=False)
    prompt = Path(args.spec).read_text()
    disabled = [t for t in args.disable_tools.split(",") if t]
    config = json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                         "model": f"local/{args.model}", "tools": {t: False for t in disabled},
                         "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "Local adapter",
                                                "options": {"baseURL": args.base_url},
                                                "models": {args.model: {"name": args.model, "tool_call": True,
                                                                        "limit": {"context": args.context,
                                                                                  "output": args.output}}}}}})
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": config, "OPENCODE_DISABLE_AUTOUPDATE": "1",
           "OPENCODE_DISABLE_PROJECT_CONFIG": "1"}
    started = time.monotonic()
    try:
        proc = subprocess.run(sandbox(project) + [str(OPENCODE), "run", "--standalone", "--auto",
                              "--format", "json", "-m", f"local/{args.model}", prompt],
                              env=env, capture_output=True, text=True, timeout=args.timeout)
        stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc, timed_out = None, True
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    tools = [e["part"] for e in events if e.get("type") == "tool_use"]
    texts = [e["part"].get("text", "") for e in events if e.get("type") == "text"]
    files = sorted(str(p.relative_to(project)) for p in project.rglob("*")
                   if p.is_file() and not SKIP_DIRS & set(p.relative_to(project).parts))
    record = {"spec": args.spec, "seconds": round(time.monotonic() - started, 1), "exit_code": rc,
              "timed_out": timed_out, "tool_calls": len(tools),
              "tools_used": sorted({t.get("tool") for t in tools}),
              "tool_errors": sum(t.get("state", {}).get("status") == "error" for t in tools),
              "write_order": write_order(events, project), "files": files,
              "final_message": texts[-1] if texts else "", "tests_rerun": independent_tests(project),
              "stderr_tail": stderr[-2000:], "events": events}
    Path(args.log).write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ("events", "final_message", "stderr_tail")},
                     indent=1), flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
