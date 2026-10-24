"""Verified opencode-format demonstrations for SFT (rendered_sft_v1 rows, one per assistant turn).

Every episode is executed in opencode_env (opencode's exact prompt, tools and tool-result text) and kept
only if the environment's hidden grade, integrity and verify checks all pass. Completions are the chat
template's own rendering of each assistant turn, and each is fed back through OpencodeEpisode.step, so
the text the model learns to emit is the text the serving parser accepts.

  python -m cutrain.dataprep.build_opencode_sft --episodes 80 --out data/opencode_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from cutrain.envs.opencode_env import OpencodeEpisode, call_message
from cutrain.serving.serve_adapter import to_template

FINALS = ["Implemented the contract in pipeline.py. `python3 verify.py` passed: 3 tests passed.",
          "pipeline.py now applies the override and every contract stage, returning each intermediate value. "
          "Ran `python3 verify.py`: 3 tests passed.",
          "Done. I updated `evaluate` in pipeline.py and verified it with `python3 verify.py`, which printed "
          "\"3 tests passed\". baseline.txt and verify.py are unchanged.",
          "The repair is complete and verified: `python3 verify.py` reports 3 tests passed."]
EXPR = {"add": "x + {v}", "multiply": "x * {v}", "floor": "max(x, {v})"}


def program(offset, stages):
    code = f"def evaluate(x):\n    x += {offset}\n    values = []\n"
    for st in stages:
        code += f"    x = {EXPR[st['op']].format(v=st['value'])}\n    values.append(x)\n"
    return code + "    return values\n"


def mutated(offset, stages, rng):
    """A plausible wrong first attempt: skipped override, or one stage with the wrong operation."""
    if rng.random() < 0.5:
        return program(0, stages) if offset else program(offset + 1, stages)
    k = rng.randrange(len(stages))
    wrong = [dict(s) for s in stages]
    wrong[k]["op"] = rng.choice([o for o in EXPR if o != stages[k]["op"]])
    return program(offset, wrong)


def plan(ep: OpencodeEpisode, rng: random.Random):
    w = str(ep.work)
    stages, offset = ep.base.stages, ep.base.offset
    good, stub = program(offset, stages), (ep.work / "pipeline.py").read_text()
    reads = [("read", {"path": f"{w}/contract.json"}), ("read", {"path": f"{w}/pipeline.py"})]
    if rng.random() < 0.3:
        reads.reverse()
    if rng.random() < 0.25:
        reads.insert(0, ("read", {"path": w}))
    if rng.random() < 0.3:
        reads.append(("read", {"path": f"{w}/verify.py"}))
    turns = [[r] for r in reads]
    use_write = rng.random() < 0.3

    def change(old, new):
        if use_write:
            return [("write", {"path": f"{w}/pipeline.py", "content": new})]
        return [("edit", {"path": f"{w}/pipeline.py", "oldString": old.rstrip("\n"), "newString": new.rstrip("\n")})]

    if rng.random() < 0.3:
        bad = mutated(offset, stages, rng)
        turns += [change(stub, bad), [("shell", {"command": "python3 verify.py"})],
                  [("read", {"path": f"{w}/contract.json"})], change(bad, good)]
    else:
        turns.append(change(stub, good))
    turns.append([("shell", {"command": "python3 verify.py"})])
    return turns + [rng.choice(FINALS)]


def completion_for(ep, tok, message):
    prompt = ep.prompt(tok)
    full = tok.apply_chat_template(to_template(ep.messages + [message]), tools=ep.tools, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("template rendering is not a continuation of the prompt")
    suffix = full[len(prompt):]
    if not suffix.endswith(tok.eos_token + "\n"):
        raise ValueError("unexpected end of rendered assistant turn")
    return prompt, suffix[:-len(tok.eos_token + "\n")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model path (for the tokenizer and chat template)")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--first-seed", type=int, default=3_000_000, help="disjoint from the opencode test (7,000,000+)")
    ap.add_argument("--max-tokens", type=int, default=11500, help="drop rows whose prompt+completion exceed this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.first_seed)
    rows, kept, dropped_long = [], 0, 0
    for i in range(args.episodes):
        seed = args.first_seed + i
        with OpencodeEpisode(seed) as ep:
            episode_rows = []
            for turn in plan(ep, rng):
                if isinstance(turn, str):
                    message = {"role": "assistant", "content": turn}
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": call_message(turn)}
                prompt, completion = completion_for(ep, tok, message)
                n = len(tok(prompt + completion, add_special_tokens=False).input_ids) + 1
                episode_rows.append({"format": "rendered_sft_v1", "source": "opencode_oracle",
                                     "source_id": f"opencode:{seed}:{len(episode_rows)}",
                                     "rendered_prompt": prompt, "completion": completion, "tokens": n})
                ep.step(completion)  # executes the tools exactly as a policy turn would
            result = ep.result()
        if not (result["success"] and not result["invalid"] and not result["false_claim"]):
            raise RuntimeError(f"oracle episode {seed} failed verification: {result}")
        long_rows = [r for r in episode_rows if r["tokens"] > args.max_tokens]
        dropped_long += len(long_rows)
        rows += [r for r in episode_rows if r["tokens"] <= args.max_tokens]
        kept += 1
    rng.shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"episodes": kept, "rows": len(rows), "dropped_over_max_tokens": dropped_long,
              "max_tokens": max(r["tokens"] for r in rows), "mean_tokens": sum(r["tokens"] for r in rows) / len(rows)}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
