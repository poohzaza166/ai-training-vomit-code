"""Export verified terminal trajectories for SFT; labels should cover assistant turns only."""
import argparse
import json
from pathlib import Path
from cutrain.envs.agent_grpo_env import TOOLS, Episode, action_text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=500042)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--audit-lines", type=int, nargs="+", default=[0, 64, 256])
    args = ap.parse_args()
    if args.episodes < 1 or args.depth < 1 or min(args.audit_lines) < 0:
        ap.error("episode count and depth must be positive, audit lines nonnegative")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for i in range(args.episodes):
            with Episode(args.seed + i, args.depth, args.audit_lines[i % len(args.audit_lines)]) as ep:
                ep.seed_history()
                # Mark seeded assistant turns separately: they are context, not teacher labels.
                first_label = len(ep.messages)
                for action in ep.oracle():
                    ep.step(action_text(action))
                result = ep.result()
                if not result["success"]:
                    raise RuntimeError(f"oracle failed for seed {args.seed + i}: {result}")
                row = {"episode_id": f"terminal-pipeline:{args.seed + i}", "source": "verified_local_oracle",
                       "family": "pipeline-repair", "messages": ep.messages, "tools": TOOLS,
                       "assistant_loss_indices": [j for j, m in enumerate(ep.messages)
                                                 if j >= first_label and m["role"] == "assistant"],
                       "verification": result}
                f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.episodes} verified episodes to {out}")


if __name__ == "__main__":
    main()
