"""Verified clean/recovery/stop-now examples and occasional genuinely unavailable-shell endings."""
import argparse
import json
from pathlib import Path
from cutrain.envs.agent_grpo_env import TOOLS, Episode, action_text


def build_row(seed, mode, audit_lines=0):
    with Episode(seed, depth=3, audit_lines=audit_lines) as ep:
        ep.seed_history()
        actions = ep.oracle()
        if mode == "recovery":
            ep.step(action_text({"tool": "run_command", "args": {"cmd": "python3 verify.py"}}))
            assert not ep.verified  # actual failing observation is context, not a positive label
        elif mode == "stop_now":
            for action in actions[:-1]:
                ep.step(action_text(action))
            assert ep.verified and ep.grade() == 1
        elif mode == "blocked":
            output, rc = ep.shell.run("sleep 60", timeout=0.1)
            assert rc == 124 and ep.shell.proc.poll() is not None
            call_id = ep.add_call({"tool": "run_command", "args": {"cmd": "sleep 60", "timeout": 0.1}})
            ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": output + "\n__RC__124"})
        elif mode != "clean":
            raise ValueError("unknown stopping demonstration mode")
        start = len(ep.messages)
        if mode == "blocked":
            ep.step(action_text({"tool": "task_done", "args": {
                "summary": "The terminal timed out and its session closed. I could not complete or verify the repair.",
                "success": False, "artifacts": []}}))
        else:
            for action in (actions[-1:] if mode == "stop_now" else actions):
                ep.step(action_text(action))
        verification = ep.result()
        if mode != "blocked" and not verification["success"]:
            raise RuntimeError("unverified demonstration")
        if mode == "blocked" and (verification["success"] or verification["false_claim"]):
            raise RuntimeError("invalid blocked demonstration")
        return {"episode_id": f"stopping:{seed}:{mode}", "family": "pipeline-repair", "mode": mode,
                "source": "verified_local_stopping_oracle", "messages": ep.messages, "tools": TOOLS,
                "assistant_loss_indices": [i for i in range(start, len(ep.messages))
                                           if ep.messages[i]["role"] == "assistant"],
                "verification": verification,
                "verified_blocker": {"kind": "terminal_unavailable", "shell_exited": True} if mode == "blocked" else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=700042)
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("episodes must be positive")
    modes = ["clean", "recovery", "stop_now", "clean", "recovery", "stop_now", "clean", "blocked"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for i in range(args.episodes):
            f.write(json.dumps(build_row(args.seed + i, modes[i % len(modes)], [0, 16, 48][i % 3])) + "\n")
    print(f"Wrote {args.episodes} verified stopping demonstrations to {out}")


if __name__ == "__main__":
    main()
