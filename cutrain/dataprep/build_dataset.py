"""Merge the prepared jsonl files into one training set, with the mix fixed at build time.

Why this exists: train_grpo_gan.py's --limit samples the merged pool *after* shuffling, so pool
composition decides what a round actually trains on - not intent. Adding a large public set silently
evicts the agentic rows (all 746k GLM-5.1 rows would leave ~23 agentic rows in a sample of 800).
Quotas here set the mix once, so --limit just decides how long a round takes.

--fit-tokens is the other half. train_orpo.fit() drops a row whose prompt+completion will not fit the
training budget even after its history is trimmed, and it drops the long-history rows first - which
are exactly the recorded sessions. Building with the same budget the run will use means the composition
printed here is the composition trained on, instead of a plan the loader quietly rewrites.

  python -m cutrain.dataprep.build_dataset --dry-run
  python -m cutrain.dataprep.build_dataset --out data/train.jsonl \\
      --max-prompt-tokens 1152 --max-completion-tokens 1024          # every usable row
  python -m cutrain.dataprep.build_dataset --out data/train.jsonl \\
      --max-prompt-tokens 1152 --max-completion-tokens 1024 \\
      --quota opencode=300 claude=100 hf:openthoughts-agent=200 hf:nvidia/When2Call=200
  python -m cutrain.dataprep.build_dataset --selftest
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cutrain.paths import REPO

DEFAULT_IN = ["data/unified.jsonl", "data/hf_agent.jsonl", "data/hf_reasoning.jsonl"]


def load(paths: list[str], stats: Counter) -> list[dict]:
    """Read every input, dropping rows that repeat an id (a re-run of a preparer overlaps itself)."""
    seen, rows = set(), []
    for p in paths:
        path = Path(p) if Path(p).is_absolute() else REPO / p
        if not path.exists():
            raise SystemExit(f"missing input: {path}\n"
                             "  prepare_data.py writes unified.jsonl; prepare_hf.py the hf_* ones")
        n = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen:
                stats["dropped: duplicate id"] += 1
                continue
            seen.add(r["id"])
            rows.append(r)
            n += 1
        stats[f"read: {path.name}"] = n
    return rows


def key_of(row: dict, keys: list[str]) -> str | None:
    """The quota bucket this row belongs to: the longest matching prefix, so 'hf:dapo' beats 'hf'."""
    hits = [k for k in keys if row["source"].startswith(k)]
    return max(hits, key=len) if hits else None


def apply_quotas(rows: list[dict], quotas: dict[str, int], rng: random.Random,
                 other: str, stats: Counter, keep=None) -> list[dict]:
    """Shuffle, then take at most quota[k] rows per bucket. A bucket short of its quota says so.

    `keep` (the --fit-tokens check) is asked last and only for a row a quota still has room for:
    it re-renders the chat template per row, so running it over the whole pool costs half an hour to
    decide rows the quotas were going to throw away anyway.
    """
    rng.shuffle(rows)
    keys = list(quotas)
    kept, taken = [], Counter()
    for r in rows:
        k = key_of(r, keys)
        if k is None:
            if other == "keep" and (keep is None or keep(r)):
                kept.append(r)
                taken["(unquoted)"] += 1
            elif other != "keep":
                stats["dropped: source not in --quota"] += 1
            continue
        if taken[k] >= quotas[k]:
            stats[f"dropped: {k} over quota"] += 1
            continue
        if keep is not None and not keep(r):
            continue
        kept.append(r)
        taken[k] += 1
    for k, want in quotas.items():
        if taken[k] < want:
            stats[f"SHORT: {k} wanted {want}, pool had {taken[k]}"] += 1
    rng.shuffle(kept)
    return kept


def fit_check(prompt_cap: int, completion_cap: int, model: str, stats: Counter):
    """Keep rows that fit BOTH caps, bounding prefill memory and learnability separately.

    train_orpo.fit only bounds prompt+completion, so one budget cannot express "prompts must stay
    under 1152 because that is what prefills four at a time, but completions may run to 1024 because
    this model keeps a KV cache on 4 of its 42 layers and decode is nearly free". Trimming history
    never changes the completion, so measure the completion first and hand fit a budget of
    prompt_cap + that - which makes its own test reduce to prompt <= prompt_cap.

    A row whose recorded action is longer than the run's --max-completion-length is dropped, not
    trimmed: the rollout would be cut off and mask_truncated_completions throws it away, so it costs
    a generation slot and teaches nothing.
    """
    from transformers import AutoTokenizer

    from cutrain.training.train_orpo import fit, render

    tok = AutoTokenizer.from_pretrained(model)
    tok.pad_token = tok.unk_token
    tok.bos_token = None

    def keep(row: dict) -> bool:
        pair = {**row, "rejected": row["chosen"]}
        got = render(pair, tok)
        if got is None:
            stats["dropped: chat-template mismatch"] += 1
            return False
        c = len(tok(got["chosen"])["input_ids"])
        if c > completion_cap:
            stats[f"dropped: completion over {completion_cap} tokens"] += 1
            return False
        if fit(pair, tok, prompt_cap + c) is None:
            stats[f"dropped: prompt will not trim under {prompt_cap} tokens"] += 1
            return False
        return True

    return keep


def report(rows: list[dict], title: str) -> None:
    by = Counter(r["source"] for r in rows)
    traces = Counter(r["source"] for r in rows if any(m.get("reasoning_content") for m in r["chosen"]))
    tools = Counter(r["source"] for r in rows if r["tools"])
    n = max(len(rows), 1)
    print(f"\n{title}: {len(rows)} rows")
    print(f"  {'source':30s} {'rows':>6} {'share':>7} {'<think>':>8} {'tools':>7}")
    for k, v in by.most_common():
        print(f"  {k:30s} {v:>6} {v / n:>6.1%} {traces[k] / v:>7.0%} {tools[k] / v:>6.0%}")
    print(f"  {'TOTAL':30s} {len(rows):>6} {1:>6.0%} {sum(traces.values()) / n:>7.0%} "
          f"{sum(tools.values()) / n:>6.0%}")


def selftest() -> None:
    """The quota logic, which is the only part with a branch worth breaking."""
    rng = random.Random(0)
    rows = ([{"id": f"a{i}", "source": "opencode", "tools": [1], "chosen": [{"reasoning_content": "x"}]} for i in range(10)]
            + [{"id": f"b{i}", "source": "hf:dapo-math", "tools": [], "chosen": [{}]} for i in range(10)]
            + [{"id": f"c{i}", "source": "hf:other", "tools": [], "chosen": [{}]} for i in range(5)])
    st: Counter = Counter()
    got = apply_quotas(list(rows), {"opencode": 3, "hf:dapo": 4}, rng, "drop", st)
    # the expensive check is consulted only for rows a quota still wants
    asked: list = []
    apply_quotas(list(rows), {"opencode": 2}, rng, "drop", Counter(),
                 keep=lambda r: (asked.append(r["id"]), True)[1])
    assert len(asked) == 2, asked
    assert Counter(r["source"] for r in got) == {"opencode": 3, "hf:dapo-math": 4}, got
    assert st["dropped: source not in --quota"] == 5  # hf:other matched no quota key
    # longest prefix wins, so a narrow key is not swallowed by a broad one
    assert key_of({"source": "hf:dapo-math"}, ["hf", "hf:dapo"]) == "hf:dapo"
    # a quota larger than the pool is reported, not silently met
    st2: Counter = Counter()
    apply_quotas(list(rows), {"opencode": 99}, rng, "drop", st2)
    assert any(k.startswith("SHORT: opencode") for k in st2), st2
    # keeping unquoted rows keeps all of them
    st3: Counter = Counter()
    assert len(apply_quotas(list(rows), {"opencode": 3}, rng, "keep", st3)) == 3 + 15
    print("ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=DEFAULT_IN, help="jsonl inputs to merge")
    ap.add_argument("--out", default=str(REPO / "data" / "train.jsonl"))
    ap.add_argument("--quota", nargs="*", default=[], metavar="SOURCE=N",
                    help="cap per source prefix, matched from the left ('hf' takes every hf row, "
                         "'hf:dapo' only the maths). No quotas = keep everything, report only")
    ap.add_argument("--other", choices=("drop", "keep"), default="drop",
                    help="what to do with a source no --quota names (default: drop, so the mix is "
                         "exactly what you asked for)")
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="keep only rows whose prompt trims under this (= the run's --max-prompt-length). "
                         "This is the one that costs VRAM: GRPO prefills --num-generations of them at once")
    ap.add_argument("--max-completion-tokens", type=int, default=0,
                    help="keep only rows whose recorded action fits this (= the run's "
                         "--max-completion-length). Needs --max-prompt-tokens too")
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", help="tokenizer for --fit-tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report the mix, write nothing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    stats: Counter = Counter()
    rows = load(args.data, stats)
    report(rows, "pool")

    quotas = {}
    for spec in args.quota:
        k, _, v = spec.partition("=")
        if not v.isdigit():
            ap.error(f"--quota wants SOURCE=N, got {spec!r}")
        quotas[k] = int(v)
    if bool(args.max_prompt_tokens) != bool(args.max_completion_tokens):
        ap.error("--max-prompt-tokens and --max-completion-tokens go together: one without the other "
                 "cannot say whether a row is unusable because of prefill or because of truncation")
    keep = (fit_check(args.max_prompt_tokens, args.max_completion_tokens, args.model, stats)
            if args.max_prompt_tokens else None)
    if quotas:
        rows = apply_quotas(rows, quotas, random.Random(args.seed), args.other, stats, keep)
        caps = f" + caps {args.max_prompt_tokens}/{args.max_completion_tokens}" if keep else ""
        report(rows, "after quotas" + caps)
    else:
        random.Random(args.seed).shuffle(rows)
        if keep:  # no quotas: every row has to be checked, which is the slow path by definition
            rows = [r for r in rows if keep(r)]
            report(rows, f"after caps {args.max_prompt_tokens}/{args.max_completion_tokens}")

    print()
    for k, v in sorted(stats.items()):
        print(f"  {v:7d}  {k}" if not k.startswith("SHORT") else f"  {k}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
