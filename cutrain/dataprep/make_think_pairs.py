"""Pairs that teach the model to think before acting: same action, reasoning vs empty <think>.

The ORPO-tuned policy emits an empty `<think></think>` in ~78% of turns, and GRPO cannot fix that:
group-relative advantage can only amplify behaviour the policy already samples, and if no sample in
a group reasons, the reasoning reward is flat and contributes no gradient. So the behaviour has to be
put back by supervised preference learning first.

Every unified row whose chosen step carries reasoning_content already contains both sides of the
pair: chosen as recorded, rejected = the identical tool call with the reasoning stripped. Nothing is
invented, and the only difference between the two sides is the thing being taught.

  python -m cutrain.dataprep.make_think_pairs data/unified.jsonl data/hf_agent.jsonl
  python -m cutrain.training.train_orpo --data data/think_pairs.jsonl --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cutrain.paths import REPO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(REPO / "data" / "think_pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--min-words", type=int, default=12, help="skip one-liners: they teach a token, not a habit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, stats = [], Counter()
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                chosen = r["chosen"][0]
                reasoning = (chosen.get("reasoning_content") or "").strip()
                if not reasoning:
                    stats[f"{r['source'].split(':')[0]}: no reasoning to teach"] += 1
                    continue
                if len(reasoning.split()) < args.min_words:
                    stats["reasoning too short"] += 1
                    continue
                if not (chosen.get("tool_calls") or chosen.get("content", "").strip()):
                    stats["empty action"] += 1
                    continue
                stripped = {k: v for k, v in chosen.items() if k != "reasoning_content"}
                rows.append({**r, "id": f"think:{r['id']}", "chosen": [chosen], "rejected": [stripped],
                             "rejected_origin": "synthetic:no_reasoning"})
                stats[f"{r['source'].split(':')[0]}: pair"] += 1
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(stats.items()):
        print(f"{v:7d}  {k}")
    print(f"wrote {len(rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
