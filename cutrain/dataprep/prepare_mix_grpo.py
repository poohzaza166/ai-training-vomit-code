"""Task pool for the mixed GRPO run (mimo_env.py): DeepSeek-V4 + GLM-5.1 reasoning, MiMo-V2.6-RL-oss code and
terminal_bench.

  select    pick candidates -> <out>/candidates.jsonl
            math: rows whose teacher answer gives a checkable label (a short \\boxed{} answer, or exactly
                  one of the prompt's listed choices); prompt and reasoning length capped so a 1,536-token
                  budget has a chance.
            code: MiMo opensource-code tasks whose image is at most --code-max-gb (images are 0.5-3.5 GB
                  each; all 2,698 would be terabytes).
            terminal: the 64 terminal_bench tasks from MiMo's "general" split (the other 925 general tasks
                  are scored mostly by an LLM judge and are left out).
  validate  pull each code/terminal image and run its verifier on the untouched task. Keep a task only
            if the verifier runs and scores 0: a task that already passes, or whose tests cannot run,
            teaches nothing.
  schedule  interleave validated tasks into <out>/train.jsonl (one row per GRPO group, in order) and
            hold some out as <out>/eval.jsonl.

  .venv-grpo/bin/python -m cutrain.dataprep.prepare_mix_grpo select --out data/mix_grpo
  .venv-grpo/bin/python -m cutrain.dataprep.prepare_mix_grpo validate --out data/mix_grpo
  .venv-grpo/bin/python -m cutrain.dataprep.prepare_mix_grpo schedule --out data/mix_grpo --groups 80
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

from cutrain.paths import REPO

MIMO = REPO / "data/mimo"
MATH_SOURCES = {  # repo, files (streamed), kind
    "deepseek": ("Jackrong/DeepSeek-V4-Distill-8000x", ("data/train.jsonl",)),
    "glm": ("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", ("main.jsonl",)),
}
BOXED = re.compile(r"\\boxed\{([^{}]{1,40})\}")
THINK = re.compile(r"<think>(.*?)</think>", re.S)
CHOICES = re.compile(r"Choices?:\s*(.+?)(?:\n\s*\n|\s+A:|\s+Answer:|$)", re.S | re.I)
ASK_BOXED = "\n\nGive your final answer in \\boxed{}."
ASK_CHOICE = "\n\nGive the exact text of the correct choice in \\boxed{}."


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def math_label(prompt: str, output: str) -> dict | None:
    final = THINK.sub("", output).strip()
    m = CHOICES.search(prompt)
    if m:
        choices = [c.strip(" .") for c in re.split(r"\s+-\s+|\n\s*-\s*", " " + m.group(1)) if c.strip(" .")]
        if 2 <= len(choices) <= 8:
            named = [c for c in choices if re.search(rf"\b{re.escape(norm(c))}\b", norm(final))] if final else []
            if len(named) == 1:
                return {"type": "choice", "answer": named[0], "choices": choices}
        return None
    boxed = BOXED.findall(final)
    if boxed and len(set(boxed)) == 1:
        return {"type": "boxed", "answer": boxed[-1].strip()}
    return None


def lines(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            yield raw.decode("utf-8", errors="replace")


def select_math(kind: str, want: int, max_prompt: int, max_think: int, stats: Counter) -> list[dict]:
    repo, files = MATH_SOURCES[kind]
    out = []
    for fname in files:
        for i, line in enumerate(lines(f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}")):
            if len(out) >= want or i > 200000:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, output = (row.get("input") or "").strip(), (row.get("output") or "").strip()
            think = THINK.search(output)
            if not prompt or not output or len(prompt) > max_prompt:
                stats[f"{kind}: prompt empty or too long"] += 1
                continue
            if think and len(think.group(1)) > max_think:
                stats[f"{kind}: reasoning too long"] += 1
                continue
            label = math_label(prompt, output)
            if label is None:
                stats[f"{kind}: no checkable label"] += 1
                continue
            ask = ASK_CHOICE if label["type"] == "choice" else ASK_BOXED
            out.append({"id": f"{kind}#{row.get('id', i)}", "kind": kind, "problem": prompt + ask, "label": label})
            stats[f"{kind}: kept {label['type']}"] += 1
    return out


def hub_gb(tag: str) -> float | None:
    url = f"https://hub.docker.com/v2/repositories/xiaomimimo/mimo-v2.6-rl-oss/tags/{tag}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return (json.load(r).get("full_size") or 0) / 1e9
        except Exception:  # noqa: BLE001 - rate limits; back off and retry
            time.sleep(5 * (attempt + 1))
    return None


def select_mimo(want_code: int, code_max_gb: float, max_problem: int, rng: random.Random, stats: Counter):
    import pandas as pd
    from cutrain.envs.mimo_env import hub_image
    code = pd.read_parquet(MIMO / "code.parquet")
    rows = [json.loads(e["instance_json"]) for e in code.extra_info]
    rng.shuffle(rows)
    out = []
    for ij in rows:
        if len([t for t in out if t["kind"] == "code"]) >= want_code:
            break
        if len(ij["problem_statement"]) > max_problem:
            stats["code: problem too long"] += 1
            continue
        tag = ij["docker_image"].split(":", 1)[0]
        gb = hub_gb(tag)
        if gb is None or gb > code_max_gb:
            stats["code: image too big or unknown"] += 1
            continue
        out.append({"id": ij["instance_id"], "kind": "code", "image": hub_image(ij["docker_image"]),
                    "image_gb": round(gb, 2), "cwd": ij["cwd"], "problem": ij["problem_statement"],
                    "test_patch": ij["test_patch"], "test_command": ij["test_command"],
                    "verifier_timeout": ij["verifier_timeout_sec"]})
        stats["code: kept"] += 1
    general = pd.read_parquet(MIMO / "general/train.parquet")
    for e in general.extra_info:
        ij = json.loads(e["instance_json"])
        if ij["dataset_type"] != "terminal_bench":
            continue
        out.append({"id": ij["instance_id"], "kind": "terminal", "image": hub_image(ij["docker_image"]),
                    "cwd": ij["cwd"], "problem": ij["problem_statement"],
                    "tests_files": json.loads(ij["tests_files"]) if isinstance(ij["tests_files"], str)
                    else ij["tests_files"],
                    "verifier_timeout": ij["verifier_timeout_sec"]})
        stats["terminal: kept"] += 1
    return out


def cmd_select(args):
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for kind in ("deepseek", "glm"):
        tasks += select_math(kind, args.math_per_source, args.max_prompt_chars, args.max_think_chars, stats)
        print(f"{kind}: {sum(t['kind'] == kind for t in tasks)}", flush=True)
    tasks += select_mimo(args.code, args.code_max_gb, args.max_problem_chars, rng, stats)
    with open(out / "candidates.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out / "select.report.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats), indent=1))


def cmd_validate(args):
    from cutrain.envs.mimo_env import MimoEpisode, ensure_image, load_tasks
    import subprocess
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out / "validated.jsonl")} if (out / "validated.jsonl").exists() else set()
    for task in load_tasks(out / "candidates.jsonl"):
        if task["kind"] not in ("code", "terminal") or task["id"] in done:
            continue
        started = time.time()
        record = {"id": task["id"], "kind": task["kind"]}
        try:
            ensure_image(task["image"])
            with MimoEpisode(task) as ep:
                v = ep.verify()
            record.update(score=v["score"], error=v.get("error"), output=(v.get("output") or "")[-400:],
                          ok=v["score"] == 0.0 and not v.get("error"))
        except Exception as exc:  # noqa: BLE001 - a broken task is a result, not a crash
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:400])
        record["seconds"] = round(time.time() - started, 1)
        if not record["ok"] and args.prune:
            subprocess.run(["docker", "rmi", task["image"]], capture_output=True)
        with open(out / "validated.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{'OK  ' if record['ok'] else 'DROP'} {task['kind']:8s} {task['id']:45s} {record['seconds']:6.1f}s "
              f"{record.get('error') or ''}", flush=True)


def cmd_schedule(args):
    from cutrain.envs.mimo_env import load_tasks
    out = Path(args.out)
    rng = random.Random(args.seed)
    ok = {json.loads(l)["id"] for l in open(out / "validated.jsonl") if json.loads(l)["ok"]}
    pools: dict[str, list] = {"math": [], "code": [], "terminal": []}
    for t in load_tasks(out / "candidates.jsonl"):
        if t["kind"] in ("deepseek", "glm"):
            pools["math"].append(t)
        elif t["id"] in ok:
            pools[t["kind"]].append(t)
    for p in pools.values():
        rng.shuffle(p)
    # One eval task per pool, held out; the rest train. The math pool is DeepSeek and GLM shuffled together.
    evals = [p.pop() for p in pools.values() if p]
    pattern = args.pattern.split(",")
    train, cursor = [], Counter()
    while len(train) < args.groups:
        progressed = False
        for kind in pattern:
            pool = pools[kind]
            if not pool or len(train) >= args.groups:
                continue
            train.append(pool[cursor[kind] % len(pool)])
            cursor[kind] += 1
            progressed = True
        if not progressed:
            raise SystemExit("every pool is empty")
    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as f:
            for t in rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    reuse = {k: (cursor[k], len(pools[k])) for k in pools}
    print(f"train {len(train)} groups: {dict(Counter(t['kind'] for t in train))}; eval {[t['id'] for t in evals]}")
    print("draws vs distinct tasks per pool:", reuse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--math-per-source", type=int, default=60)
    s.add_argument("--max-prompt-chars", type=int, default=2000)
    s.add_argument("--max-think-chars", type=int, default=6000)
    s.add_argument("--code", type=int, default=60)
    s.add_argument("--code-max-gb", type=float, default=1.0)
    s.add_argument("--max-problem-chars", type=int, default=6000)
    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    v.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                   help="remove the image of a task that fails validation")
    c = sub.add_parser("schedule")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--groups", type=int, default=80)
    c.add_argument("--pattern", default="math,code,math,code,terminal",
                   help="kinds in order, repeated; the default is 40%% math, 40%% code, 20%% terminal")
    args = ap.parse_args()
    {"select": cmd_select, "validate": cmd_validate, "schedule": cmd_schedule}[args.cmd](args)


if __name__ == "__main__":
    main()
