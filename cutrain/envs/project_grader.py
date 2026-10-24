"""Independent grading of a project folder: the project's own tests, coverage, hidden CLI checks, docs.

Everything runs in bubblewrap without network, with the folder mounted read-only for CLI checks on a
copy, using .venv-grader (pytest + coverage on the system Python the projects target). Nothing here
fixes the project; it only measures it.
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from cutrain.paths import REPO

GRADER_PY = REPO / ".venv-grader/bin/python"
SKIP = {".venv", "venv", "__pycache__", ".pytest_cache", ".git", "node_modules"}


def sandbox(folder: Path, writable: bool = True, cwd: Path | None = None) -> list[str]:
    """cwd: optional separate working folder (bound writable); the project is then importable via PYTHONPATH."""
    extra = ["--bind", str(cwd), str(cwd), "--setenv", "PYTHONPATH", str(folder)] if cwd else []
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(REPO / ".venv-grader"), str(REPO / ".venv-grader"),
            "--bind" if writable else "--ro-bind", str(folder), str(folder), "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8",
            "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/pyc", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            *extra, "--chdir", str(cwd or folder)]


def project_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(folder).parts))


def packages(folder: Path) -> list[Path]:
    return sorted(p.parent for p in folder.glob("*/__init__.py") if p.parent.name not in SKIP | {"tests"})


def run_own_tests(folder: Path, timeout: int = 300) -> dict:
    """Copy the project, run its tests under coverage with the grader's interpreter."""
    with tempfile.TemporaryDirectory(prefix="grade-") as tmp:
        work = Path(tmp) / folder.name
        shutil.copytree(folder, work, ignore=shutil.ignore_patterns(*SKIP))
        pkgs = [p.name for p in packages(work)]
        source = ",".join(pkgs) if pkgs else "."
        py = shlex.quote(str(GRADER_PY))
        cmd = (f"{py} -m coverage run --source={shlex.quote(source)} -m pytest -q -p no:cacheprovider; rc=$?; "
               f"{py} -m coverage report --format=total 2>/dev/null || echo 0; exit $rc")
        try:
            proc = subprocess.run(sandbox(work) + ["/bin/bash", "-c", cmd], capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ran": True, "passed": False, "tests": 0, "coverage": 0, "tail": ["timed out"]}
        out = proc.stdout + proc.stderr
        lines = out.strip().splitlines()
        cov = int(lines[-1]) if lines and lines[-1].strip().isdigit() else 0
        passed_n = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = bool(re.search(r"\d+ (failed|error)", out)) or proc.returncode != 0
        return {"ran": True, "passed": not failed and passed_n > 0, "tests": passed_n, "coverage": cov,
                "tail": lines[-12:]}


def run_checks(folder: Path, package: str, checks, timeout: int = 30, sample: str | None = None) -> dict:
    """Hidden CLI checks in order on a fresh copy, so state carries from one command to the next.

    A task's sample text, if any, is written as sample.txt and replaces the __SAMPLE__ argument.
    """
    results = []
    with tempfile.TemporaryDirectory(prefix="check-") as tmp:
        work = Path(tmp) / folder.name
        shutil.copytree(folder, work, ignore=shutil.ignore_patterns(*SKIP))
        # Commands run from a fresh empty folder: data the project saved while the model tried it
        # (in its own folder) must not leak into the hidden checks.
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        if sample is not None:
            (run_dir / "sample.txt").write_text(sample)
        for c in checks:
            args = ["sample.txt" if a == "__SAMPLE__" else a for a in c.args]
            try:
                proc = subprocess.run(sandbox(work, writable=False, cwd=run_dir) + ["/usr/bin/python3", "-m", package, *args],
                                      capture_output=True, text=True, timeout=timeout)
                out, err, rc = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired:
                out, err, rc = "", "timed out", -1
            exists = (work / package / "__init__.py").exists() or (work / package / "__main__.py").exists()
            if c.ok:
                good = rc == 0 and all(re.search(p, out) for p in c.expect)
            else:  # a polite refusal only counts from a program that exists
                good = exists and rc != 0 and "Traceback" not in err + out
            results.append({"args": args, "ok": good, "rc": rc, "out": out[-300:], "err": err[-300:]})
    return {"passed": sum(r["ok"] for r in results), "total": len(results), "results": results}


def readme_ok(folder: Path) -> bool:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = folder / name
        if p.is_file():
            text = p.read_text(errors="replace").lower()
            return len(text) >= 300 and all(re.search(k, text) for k in (r"install|setup|set up",
                                                                          r"run|usage|start", r"test"))
    return False


def structure(folder: Path) -> dict:
    pkgs = packages(folder)
    modules = [p for pkg in pkgs for p in pkg.rglob("*.py") if p.name != "__init__.py"]
    longest = max((len(p.read_text(errors="replace").splitlines()) for p in modules), default=0)
    tests = [p for p in folder.rglob("test_*.py") if not SKIP & set(p.relative_to(folder).parts)]
    return {"packages": [p.name for p in pkgs], "modules": len(modules), "longest_module": longest,
            "test_files": len(tests), "ok": bool(pkgs) and len(modules) >= 3 and longest <= 300 and bool(tests)}


def spec_quality(text: str, keywords: list[str]) -> dict:
    """A real spec draft, not a restatement: length, structure, and coverage of the requirements."""
    headings = len(re.findall(r"^#{1,3} ", text, re.M))
    bullets = len(re.findall(r"^\s*[-*\d]+[.)]?\s", text, re.M))
    low = text.lower()
    covered = sum(k.lower() in low for k in keywords) / max(len(keywords), 1)
    design = bool(re.search(r"design|structure|architecture|module|layout|component", low))
    verify = bool(re.search(r"test|verif|check", low))
    ok = len(text) >= 600 and headings >= 3 and bullets >= 5 and covered >= 0.6 and design and verify
    return {"ok": ok, "chars": len(text), "headings": headings, "bullets": bullets, "coverage": round(covered, 2),
            "design": design, "verification": verify}
