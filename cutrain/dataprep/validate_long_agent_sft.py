"""Check assistant labels and token lengths using the actual local ORPO tokenizer."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from cutrain.envs.agent_grpo_env import parse_action
from cutrain.serving.serve_adapter import render_turn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/long_agent_sft_memory.jsonl")
    ap.add_argument("--tokenizer", default="out/nemotron-nano-4b-orpo/adapter")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--out", default="out/long-grpo-recovery/sft-validation.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tok.bos_token = None
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    ids, lengths = set(), []
    for row in rows:
        blocker = row.get("verified_blocker")
        honest_blocked = (blocker == {"kind": "terminal_unavailable", "shell_exited": True}
                          and row["verification"].get("abstained")
                          and not row["verification"]["false_claim"]
                          and not row["verification"]["success"])
        if row["episode_id"] in ids or not (row["verification"]["success"] or honest_blocked):
            raise ValueError("duplicate ID or unsuccessful demonstration")
        ids.add(row["episode_id"])
        for i in row["assistant_loss_indices"]:
            if row["messages"][i]["role"] != "assistant":
                raise ValueError("non-assistant label")
            prompt, completion = render_turn(tok, row["messages"], i, row.get("tools"))
            if row["messages"][1]["content"] not in prompt:
                raise ValueError("original user requirement lost")
            if "<tools>" not in prompt or parse_action(completion) is None:
                raise ValueError("assistant label is not one native tool call against the declared tools")
            completion += tok.eos_token
            if tok.encode(completion, add_special_tokens=False)[-1] != tok.eos_token_id:
                raise ValueError("assistant completion is not terminated by the actual EOS token")
            lengths.append(len(tok.encode(prompt, add_special_tokens=False)) +
                           len(tok.encode(completion, add_special_tokens=False)))
    if not lengths or max(lengths) > args.max_context:
        raise ValueError("empty dataset or context overflow")
    result = {"episodes": len(rows), "assistant_examples": len(lengths),
              "min_tokens": min(lengths), "max_tokens": max(lengths),
              "max_context": args.max_context, "passed": True}
    result["verified_blocked_episodes"] = sum(bool(r.get("verified_blocker")) for r in rows)
    result["eos_checked_assistant_examples"] = len(lengths)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
