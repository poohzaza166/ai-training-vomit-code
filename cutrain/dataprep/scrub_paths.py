"""Rewrite this machine's own paths out of a unified dataset, so the model stops inventing them.

The opencode/Claude sessions were recorded on the host, so the data is full of real project roots
(/mnt/driveD/code/unity, /home/<user>/...). A model trained on that answers "list /workspace/app"
with `ls /mnt/driveD/code/unity/app`: it learned the paths, not the task. The harness only ever has
/workspace, so every project root in the data is rewritten to /workspace and home directories to
/home/user. Roots are found by frequency, not by a hand-written list, so this keeps working on the
next dump.

  python -m cutrain.dataprep.scrub_paths data/unified.jsonl          # in place (keeps a .bak)
  python -m cutrain.dataprep.scrub_paths data/unified.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
from pathlib import Path

ROOT_RE = re.compile(r"/(?:mnt|media|srv|data|opt/projects)/[\w.@+-]+(?:/[\w.@+-]+){1,2}")
HOME_RE = re.compile(r"/(?:home|Users)/([\w.@+-]+)")
KEEP_HOME = {"user", "root"}


def find_roots(path: str, min_hits: int, scan: int) -> list[str]:
    """Project roots in this dump, longest first so the deepest prefix wins the replace."""
    hits: collections.Counter = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if scan and i >= scan:
                break
            hits.update(ROOT_RE.findall(line))
    roots = [r for r, n in hits.items() if n >= min_hits]
    # a root whose parent is also frequent is covered by the parent; keep the shallowest of a chain
    roots = [r for r in roots if not any(r != o and r.startswith(o + "/") for o in roots)]
    return sorted(roots, key=len, reverse=True)


def scrub(text: str, roots: list[str]) -> str:
    for r in roots:
        text = text.replace(r, "/workspace")
    return HOME_RE.sub(lambda m: m.group(0) if m.group(1) in KEEP_HOME else "/home/user", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-hits", type=int, default=20, help="times a root must appear to count as one")
    ap.add_argument("--scan", type=int, default=20000, help="lines scanned when looking for roots (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in args.files:
        roots = find_roots(path, args.min_hits, args.scan)
        print(f"{path}: {len(roots)} roots -> /workspace")
        for r in roots[:12]:
            print("   ", r)
        if args.dry_run:
            with open(path, encoding="utf-8") as f:
                line = next(f)
            print("   sample:", scrub(line, roots)[:300])
            continue
        shutil.copy(path, path + ".bak")
        tmp = path + ".tmp"
        changed = 0
        with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                out = scrub(line, roots)
                changed += out != line
                dst.write(out)
        os.replace(tmp, path)
        print(f"   rewrote {changed} rows (backup at {Path(path).name}.bak)")


if __name__ == "__main__":
    main()
