"""Pairs that teach the model when to stop: task_done vs one more command.

After the thinking pass, every remaining probe failure was the same habit: the answer was already in
the tool output and the model ran another command anyway ("The file exists and has 42 lines. Let me
also check the first few lines"). The harness's own prompt says to call task_done as the final action
and never re-run a successful verification more than twice, and the training data barely contains any
stopping decisions, so there is nothing to learn it from.

The teacher supplies the missing label. For each step it sees the task, the last tool output and the
action the recorded session took next, and says whether the task was already finished. Both answers
produce a pair, and both are useful:

  finished  -> chosen = task_done(summary), rejected = the recorded next command  (stop sooner)
  not yet   -> chosen = the recorded next command, rejected = task_done(...)      (don't stop early)

  python -m cutrain.dataprep.make_stop_pairs --limit 700
  python -m cutrain.training.train_sft_think --data data/stop_pairs.jsonl --adapter <adapter>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cutrain.paths import REPO

TEACHER = (
    "You are grading one step of a computer-use agent. You get the user's task, the agent's own last "
    "action and the tool output it just received. Decide whether the task is ALREADY FINISHED - the "
    "output contains everything the task asked for, and another command would add nothing.\n"
    'Reply with JSON only: {"finished": true|false, "summary": "<1-2 sentences of what the result was, '
    'only when finished>"}\n'
    "Be strict: say finished only when the output itself answers the task. Exploration that is still "
    "missing a piece of the answer is not finished.")


def last_tool_output(prompt: list[dict]) -> str | None:
    for m in reversed(prompt):
        if m["role"] == "tool":
            return m["content"]
    return None


def task_of(prompt: list[dict]) -> str | None:
    return next((m["content"] for m in prompt if m["role"] == "user"), None)


def summarise(msg: dict) -> str:
    calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])[:200]})"
                      for c in msg.get("tool_calls", []))
    return "\n".join(filter(None, [msg.get("content", ""), calls and f"[called {calls}]"])) or "(nothing)"


def done_call(summary: str, success: bool = True) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "task_done", "arguments": {"summary": summary[:400], "success": success}}}]}


AGENT_SOURCES = ("opencode", "claude", "hf:openthoughts-agent")


def final_pairs(files: list[str], stats: Counter, limit: int, sources=AGENT_SOURCES):
    """Steps where the recorded session answered instead of running another command.

    No teacher needed and no rate limit: the session's own final answer is the label. chosen is that
    answer as a task_done call (what the harness asks for), rejected is re-running the command it had
    just run - the exact failure the probe shows.
    """
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["source"].startswith(sources):
                    continue  # chat data's "You're welcome!" is not a task report
                chosen = r["chosen"][0]
                calls = chosen.get("tool_calls") or []
                text = (chosen.get("content") or "").strip()
                already_done = len(calls) == 1 and calls[0]["function"]["name"] == "task_done"
                if already_done:
                    stats["session ended with task_done"] += 1
                elif calls or len(text) < 60:
                    continue
                prev = next((m for m in reversed(r["prompt"]) if m["role"] == "assistant"), {})
                if not prev.get("tool_calls"):
                    stats["final answer, but nothing to repeat as the negative"] += 1
                    continue
                stats["from a session that finished"] += 1
                yield {**r, "id": f"stop:{r['id']}", "chosen": [chosen if already_done else done_call(text, True)],
                       "rejected": [{k: v for k, v in prev.items() if k != "reasoning_content"}],
                       "rejected_origin": "outcome:kept_going"}
                if limit and stats["from a session that finished"] >= limit:
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=[str(REPO / "data" / "unified.jsonl"),
                                                 str(REPO / "data" / "hf_agent.jsonl")])
    ap.add_argument("--out", default=str(REPO / "data" / "stop_pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=400, help="rows sent to the teacher (0 = skip the teacher)")
    ap.add_argument("--sources", nargs="+", default=list(AGENT_SOURCES),
                    help="source prefixes whose final answers count as task reports")
    ap.add_argument("--finals", type=int, default=1200,
                    help="pairs taken from sessions that finished on their own (no teacher, no rate limit)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=2, help="the proxy stalls above a handful")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats: Counter = Counter()
    out_rows = list(final_pairs(args.files, stats, args.finals, tuple(args.sources)))
    if not args.limit:
        write(out_rows, args.out, stats)
        return

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or f"sk-{args.backend}-{args.model}",
                    timeout=300)
    rows: list[dict] = []
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # needs a tool result to judge, a task, and a recorded next action that is a tool call
                if last_tool_output(r["prompt"]) and task_of(r["prompt"]) and r["chosen"][0].get("tool_calls"):
                    rows.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]
    print(f"{len(out_rows)} pairs from finished sessions; {len(rows)} more steps -> {args.model}")

    def ask(r: dict):
        user = (f"# Task\n{task_of(r['prompt'])[:1500]}\n\n# The agent's last action\n"
                f"{summarise(next((m for m in reversed(r['prompt']) if m['role'] == 'assistant'), {}))[:600]}\n\n"
                f"# Tool output it just received\n{last_tool_output(r['prompt'])[:2000]}\n\n"
                f"# What the session did next\n{summarise(r['chosen'][0])[:400]}")
        out = None
        for delay in (0, 4, 12, 30):  # groq rate-limits a bulk pass hard; back off instead of dropping rows
            if delay:
                time.sleep(delay)
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=0, max_tokens=300,
                    **({"reasoning_effort": args.reasoning} if args.reasoning else {}),
                    messages=[{"role": "system", "content": TEACHER}, {"role": "user", "content": user}])
                msg = resp.choices[0].message
                out = msg.content or (msg.model_extra or {}).get("reasoning") or ""
                break
            except Exception as e:  # noqa: BLE001
                last = f"error {type(e).__name__}"
        if out is None:
            stats[last] += 1
            return None
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            stats["unparseable teacher reply"] += 1
            return None
        if not isinstance(d, dict) or "finished" not in d:
            stats["teacher gave no verdict"] += 1
            return None
        finished = bool(d["finished"])
        summary = str(d.get("summary") or "").strip() or "Task complete."
        if finished:
            chosen, rejected, kind = done_call(summary), r["chosen"][0], "stop_sooner"
        else:
            chosen, rejected, kind = r["chosen"][0], done_call(summary or "Done.", True), "dont_stop_early"
        if chosen == rejected:
            return None
        stats[kind] += 1
        return {**r, "id": f"stop:{r['id']}", "chosen": [chosen], "rejected": [rejected],
                "rejected_origin": f"teacher:{kind}"}

    with ThreadPoolExecutor(args.workers) as ex:
        for got in ex.map(ask, rows):
            if got:
                out_rows.append(got)
    write(out_rows, args.out, stats)


def write(rows: list[dict], out: str, stats: Counter) -> None:
    random.Random(0).shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(stats.items()):
        print(f"{v:6d}  {k}")
    print(f"wrote {len(rows)} pairs -> {out}")


if __name__ == "__main__":
    main()
