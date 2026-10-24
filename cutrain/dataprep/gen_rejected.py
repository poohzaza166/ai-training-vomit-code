"""Swap rule-based rejected answers for real ones from local small models (on-policy + SLM negatives).

Each row whose rejected is synthetic is sent (same prompt + tools) to a model picked at random from --models.
If that answer does something different from chosen, it becomes the rejected answer ("model:<name>"); if it
does the same thing, the synthetic rejected stays. Rows with dataset / outcome / feedback rejected are untouched.
Progress is saved into --data as it goes, so rerunning resumes with the rows still synthetic.

Any OpenAI-compatible server works (default: Ollama). Ollama's OpenAI endpoint can't set the context size and
silently cuts prompts past its 4096 default from the front (system prompt + tools first), so use 8k variants:
  printf 'FROM nemotron-3-nano:4b\nPARAMETER num_ctx 8192\n' > Modelfile && ollama create nemotron-3-nano:4b-8k -f Modelfile
  (same for gemma4:e2b -> gemma4:e2b-8k)
Free the GPU before train_orpo.py: `ollama stop nemotron-3-nano:4b-8k gemma4:e2b-8k`.

  python -m cutrain.dataprep.gen_rejected --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from cutrain.paths import REPO


def to_api(msgs: list[dict]) -> list[dict]:
    """Unified messages -> OpenAI wire format: call ids, JSON-string arguments, tool_call_id on results."""
    out, pending = [], []
    for m in msgs:
        m = {k: v for k, v in m.items() if k != "reasoning_content"}  # not in the OpenAI message schema
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": f"call_{len(out)}_{k}", "type": "function", "function": {
                "name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])}}
                for k, c in enumerate(m["tool_calls"])]
            pending = [c["id"] for c in m["tool_calls"]]
        elif m["role"] == "tool":
            m["tool_call_id"] = pending.pop(0) if pending else "call_orphan"
        out.append(m)
    return out


def from_api(msg) -> dict:
    """OpenAI response message -> unified assistant message."""
    out = {"role": "assistant", "content": (msg.content or "").strip()}
    extra = msg.model_extra or {}
    reasoning = extra.get("reasoning") or extra.get("reasoning_content")  # Ollama / vLLM field names
    if reasoning and reasoning.strip():
        out["reasoning_content"] = reasoning.strip()
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": c.function.arguments}  # malformed arguments are still a fine negative
        calls.append({"type": "function", "function": {"name": c.function.name, "arguments": args}})
    if calls:
        out["tool_calls"] = calls
    return out


def action(m: dict):
    """What a step does, ignoring reasoning wording: its tool calls, else its answer text."""
    calls = [(c["function"]["name"], json.dumps(c["function"]["arguments"], sort_keys=True)) for c in m.get("tool_calls", [])]
    return calls or m.get("content", "").strip()


def save(rows: list[dict], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "data" / "unified.jsonl"), help="updated in place")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--models", nargs="+", default=["nemotron-3-nano:4b-8k", "gemma4:e2b-8k"],
                    help="picked at random per row; include the model you train for on-policy negatives")
    ap.add_argument("--limit", type=int, default=0, help="random sample of this many synthetic rows (0 = all)")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    rng = random.Random(args.seed)
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    todo = [i for i, r in enumerate(rows) if r["rejected_origin"].startswith("synthetic")]
    if args.limit:
        todo = rng.sample(todo, min(args.limit, len(todo)))
    jobs = [(i, rng.choice(args.models)) for i in todo]
    print(f"{len(jobs)} synthetic rows -> {args.models}")

    def ask(i: int, model: str):
        r = rows[i]
        # match chosen's format: if it has no thinking (Claude Code redacts it), a rejected with thinking would
        # teach "thinking = bad" instead of "this action = bad"
        think = "reasoning_content" in r["chosen"][0]
        resp = client.chat.completions.create(
            model=model, messages=to_api(r["prompt"]), temperature=args.temperature, max_tokens=args.max_tokens,
            **({"tools": r["tools"]} if r["tools"] else {}), **({} if think else {"reasoning_effort": "none"}))
        msg = from_api(resp.choices[0].message)
        if not think:
            msg.pop("reasoning_content", None)
        return msg, resp.choices[0].finish_reason

    stats: Counter = Counter()
    ex = ThreadPoolExecutor(args.workers)
    futs = {ex.submit(ask, i, m): (i, m) for i, m in jobs}
    try:
        for n, fut in enumerate(as_completed(futs), 1):
            i, model = futs[fut]
            try:
                msg, finish = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad request must not kill a long run
                stats[f"error {type(e).__name__}"] += 1
                continue
            if finish == "length" or not (msg["content"] or msg.get("tool_calls")):
                stats[f"{model}: truncated/empty, kept synthetic"] += 1
            elif action(msg) == action(rows[i]["chosen"][0]):
                stats[f"{model}: same as chosen, kept synthetic"] += 1
            else:
                rows[i]["rejected"], rows[i]["rejected_origin"] = [msg], f"model:{model}"
                stats[f"{model}: used as rejected"] += 1
            if n % 100 == 0:
                save(rows, args.data)
                print(f"{n}/{len(jobs)}", dict(stats), flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        save(rows, args.data)
    for k, v in sorted(stats.items()):
        print(f"{v:6d}  {k}")
    print("updated", args.data)


if __name__ == "__main__":
    main()
