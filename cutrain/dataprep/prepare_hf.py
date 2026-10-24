"""Pull agent + reasoning rows off the Hub into prepare_data.py's unified schema, for GRPO.

Sources (each one for a different skill the harness needs):
  openthoughts  open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K - terminal agent episodes
                (SWE-bench-style). Their JSON protocol ({analysis, plan, commands:[{keystrokes}]})
                is rewritten into this harness's own run_command / task_done calls, and analysis +
                plan become the step's reasoning. This is the closest public data to what we serve.
  arcee         arcee-ai/agent-data, the glaive-function-calling-v2-extended part: native tool
                calling against hundreds of different tool schemas, which is what keeps the model
                from overfitting to our 19 tools. Read over HTTP range requests - the repo is one
                1.5 GB jsonl, and the tool-calling rows sit in its last third (Magpie chat first).
                ponytail: toolbench_* rows are skipped; their APIs are fake and their CoT/command
                text needs its own parser for ~14k rows of jokes-API calls.
  dapo          BytedTsinghua-SIA/DAPO-Math-17k - the open-r1 reasoning collection's best fit for
                GRPO: every row carries an integer ground truth, so the test reward is exact and
                needs no judge. (OpenR1-Math-220k answers are LaTeX strings, Big-Math is gated,
                KodCode/OpenCodeReasoning need a sandbox to run their tests - all left out.)

Rows are GRPO-shaped: prompt + chosen (+ answer for the verifiable ones), no rejected - that is
train_grpo_gan.py's job to sample. Everything else matches prepare_data.py so one --data list feeds both.

  python -m cutrain.dataprep.prepare_hf
  python -m cutrain.dataprep.prepare_hf --openthoughts 4000 --arcee 6000 --dapo 1500
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

from cutrain.paths import REPO

ARCEE_URL = "https://huggingface.co/datasets/arcee-ai/agent-data/resolve/main/arcee_agent_data.jsonl"
ARCEE_SUBSETS = ("glaive-function-calling-v2-extended",)
# reasoning traces, ShareGPT shape, <think> in the gpt turn. The V4 set's prompts are drawn from the
# GLM one and re-answered by DeepSeek-V4-Flash, so the two overlap by construction.
JACKRONG = {
    "v4": ("Jackrong/DeepSeek-V4-Distill-8000x", ("data/train.jsonl",), "hf:v4-distill"),
    # main.jsonl first: it averages 37 KB a row against Math.jsonl's 196 KB, so under an output-length
    # cap it yields far more keepers per GB streamed (Math + PHD-Science cost 8 GB for 40 rows at 600 words)
    "glm": ("Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
            ("main.jsonl", "Multilingual-STEM.jsonl", "PHD-Science.jsonl", "Math.jsonl"), "hf:glm-reasoning"),
}
JSON_BLOCK = re.compile(r"[\[{].*[\]}]", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)


# --- unified rows ---------------------------------------------------------------------------------

def steps(cid: str, source: str, category: str, msgs: list[dict], tools: list[dict],
          answer: str | None = None, max_history_chars: int = 8000):
    """One row per assistant turn: prompt = system + first user + the latest turns that fit."""
    head = next((i + 1 for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if head is None:
        return
    sizes = [len(json.dumps(m, ensure_ascii=False)) for m in msgs]
    start, total = head, 0
    for i, m in enumerate(msgs):
        if i < head:
            continue
        if m["role"] == "assistant":
            while total > max_history_chars and start < i:
                total -= sizes[start]
                start += 1
                while start < i and msgs[start]["role"] == "tool":  # never orphan a tool result
                    total -= sizes[start]
                    start += 1
            if m.get("content") or m.get("tool_calls"):
                row = {"id": f"{cid}:{i}", "source": source, "category": category, "tools": tools,
                       "prompt": msgs[:head] + msgs[start:i], "chosen": [m], "rejected": [],
                       "rejected_origin": "none"}
                if answer is not None:
                    row["answer"] = answer
                yield row
        total += sizes[i]


# --- open-thoughts terminal agent episodes --------------------------------------------------------

def _terminus(text: str) -> dict | None:
    """Their assistant turn: optional <think>, then the JSON envelope the terminus agent speaks."""
    think = THINK.search(text)
    body = THINK.sub("", text)
    hit = JSON_BLOCK.search(body)
    if not hit:
        return None
    try:
        env = json.loads(hit.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(env, dict):
        return None
    reasoning = "\n\n".join(filter(None, [think.group(1).strip() if think else "",
                                          env.get("analysis", ""), env.get("plan", "")])).strip()
    calls = [{"type": "function", "function": {"name": "run_command", "arguments": {"cmd": c["keystrokes"].rstrip("\n")}}}
             for c in env.get("commands") or [] if isinstance(c, dict) and c.get("keystrokes", "").strip()]
    if env.get("task_complete") and not calls:
        calls = [{"type": "function", "function": {"name": "task_done", "arguments": {
            "summary": (env.get("analysis") or "task complete")[:400], "success": True}}}]
    if not calls:
        return None
    msg = {"role": "assistant", "content": ""}
    if reasoning:
        msg["reasoning_content"] = reasoning
    msg["tool_calls"] = calls[:1]  # the harness runs one call per step; the rest become later steps
    return msg


def openthoughts_rows(limit: int, stats: Counter, system: str, tools: list[dict]):
    from datasets import load_dataset

    ds = load_dataset("open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K", split="train", streaming=True)
    n = 0
    for conv_idx, row in enumerate(ds):
        msgs: list[dict] = [{"role": "system", "content": system}]
        for k, m in enumerate(row["conversations"]):
            if m["role"] == "assistant":
                got = _terminus(m["content"])
                if got is None:
                    stats["openthoughts: unparsed assistant turn"] += 1
                    continue
                msgs.append(got)
            elif k == 0:
                # their first user turn is the terminus protocol spec + the task; keep the task and
                # drop the spec, or the model learns to answer in their JSON instead of calling tools
                task = m["content"]
                for marker in ("Task Description:", "Task:"):
                    if marker in task:
                        task = task.split(marker, 1)[1]
                        break
                else:
                    stats["openthoughts: no task marker"] += 1
                    continue
                msgs.append({"role": "user", "content": task.strip()[:6000]})
            elif msgs and msgs[-1].get("tool_calls"):
                body = m["content"].split("New Terminal Output:", 1)[-1]
                msgs.append({"role": "tool", "content": body.strip()[:4000]})
            else:
                msgs.append({"role": "user", "content": m["content"].strip()[:4000]})
        for r in steps(f"openthoughts#{row.get('trial_name') or conv_idx}", "hf:openthoughts-agent",
                       "execution", msgs, tools):
            yield r
            n += 1
            if limit and n >= limit:
                return


# --- arcee / glaive native tool calling -----------------------------------------------------------

def _hub_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("x-linked-size") or r.headers["content-length"])


def _lines(url: str, offset: int, length: int):
    """Range-read a window and hand back whole JSON lines (the partial ones at both ends go)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        buf = r.read().decode("utf-8", "ignore")
    for line in buf.split("\n")[1:-1]:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _glaive(conv: list[dict]) -> tuple[list[dict], list[dict]]:
    """ShareGPT turns -> (unified messages, tool schemas declared in the system prompt)."""
    tools, msgs = [], []
    for m in conv:
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(m["from"], m["from"])
        text = (m["value"] or "").strip()
        if role == "system":
            hit = JSON_BLOCK.search(text)
            if hit:
                try:
                    declared = json.loads(hit.group(0))
                    tools = [t if "function" in t else {"type": "function", "function": t}
                             for t in (declared if isinstance(declared, list) else [declared])]
                except json.JSONDecodeError:
                    pass
                text = text[: hit.start()].strip()
            msgs.append({"role": "system", "content": text})
        elif role == "assistant":
            calls = None
            if text.startswith("["):
                try:
                    calls = [{"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                             for c in json.loads(text) if isinstance(c, dict) and c.get("name")]
                except (json.JSONDecodeError, TypeError):
                    calls = None
            msgs.append({"role": "assistant", "content": "" if calls else text, **({"tool_calls": calls} if calls else {})})
        elif msgs and msgs[-1].get("tool_calls") and text.startswith("{"):
            msgs.append({"role": "tool", "content": text})  # glaive sends results back as a human turn
        else:
            msgs.append({"role": "user", "content": text})
    return msgs, tools


def arcee_rows(limit: int, stats: Counter, start: float = 0.66, window: int = 4_000_000):
    size = _hub_size(ARCEE_URL)
    offset, n = int(size * start), 0
    while offset < size and (not limit or n < limit):
        for i, row in enumerate(_lines(ARCEE_URL, offset, window)):
            if row.get("dataset") not in ARCEE_SUBSETS:
                stats[f"arcee: skipped {row.get('dataset')}"] += 1
                continue
            msgs, tools = _glaive(row["conversations"])
            if not tools or not any(m.get("tool_calls") for m in msgs):
                stats["arcee: no tools or no call"] += 1
                continue
            for r in steps(f"arcee#{offset}#{i}", "hf:arcee-glaive", "tool_call", msgs, tools):
                yield r
                n += 1
                if limit and n >= limit:
                    return
        offset += window


# --- DAPO verifiable math -------------------------------------------------------------------------

def dapo_rows(limit: int, stats: Counter):
    from datasets import load_dataset

    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", streaming=True)
    n = 0
    for i, row in enumerate(ds):
        answer = ((row.get("reward_model") or {}).get("ground_truth") or "").strip()
        prompt = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        if not answer or not prompt:
            stats["dapo: no ground truth"] += 1
            continue
        # chosen is the answer line the prompt itself asks for; the reward checks the answer, not the prose
        yield {"id": f"dapo#{i}", "source": "hf:dapo-math", "category": "reasoning", "tools": [],
               "prompt": prompt, "chosen": [{"role": "assistant", "content": f"Answer: {answer}"}],
               "rejected": [], "rejected_origin": "none", "answer": answer}
        n += 1
        if limit and n >= limit:
            return


# --- Jackrong reasoning traces --------------------------------------------------------------------

def _lines(url: str, chunk: int = 1 << 20):
    """Stream a remote jsonl line by line. main.jsonl alone is 19.6 GB - never download it."""
    buf = ""
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl"})) as r:
        while True:
            blob = r.read(chunk)
            if not blob:
                break
            buf += blob.decode("utf-8", "ignore")
            lines = buf.split("\n")
            buf = lines.pop()
            yield from lines


def jackrong_rows(key: str, limit: int, stats: Counter, max_output_words: int = 0):
    """Reasoning traces -> unified rows. No tools, no ground truth.

    Worth knowing before training on these: a row with no `answer` and no tool call in `chosen` makes
    test_score() return a flat 1.0 for every completion that calls nothing, and a group whose rewards
    are all equal has zero advantage - no gradient. Under GRPO they only reach the policy through the
    structure reward (non-empty <think>). Their real home is an SFT pass; see train_sft_think.py.

    Outputs are long (median ~1150 words, max 6462), so --jackrong-max-output-words exists to keep
    rows that stand a chance of fitting a --max-completion-length budget.
    """
    repo, files, source = JACKRONG[key]
    n = 0
    for fname in files:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
        for i, line in enumerate(_lines(url)):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats[f"{key}: unparsed line"] += 1
                continue
            prompt, out = (row.get("input") or "").strip(), (row.get("output") or "").strip()
            if not prompt or not out:
                stats[f"{key}: empty input or output"] += 1
                continue
            if max_output_words and len(out.split()) > max_output_words:
                stats[f"{key}: output over {max_output_words} words"] += 1
                continue
            think = THINK.search(out)
            msg = {"role": "assistant", "content": THINK.sub("", out).strip()}
            if think and think.group(1).strip():
                msg["reasoning_content"] = think.group(1).strip()
            else:
                stats[f"{key}: no <think> block"] += 1
            yield {"id": f"{key}#{fname}#{row.get('id', i)}", "source": source,
                   "category": "reasoning", "tools": [],
                   "prompt": [{"role": "user", "content": prompt}], "chosen": [msg],
                   "rejected": [], "rejected_origin": "none"}
            n += 1
            if limit and n >= limit:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "hf_agent.jsonl"))
    ap.add_argument("--openthoughts", type=int, default=3000, help="rows (0 = skip)")
    ap.add_argument("--arcee", type=int, default=4000, help="rows (0 = skip)")
    ap.add_argument("--dapo", type=int, default=1000, help="rows (0 = skip)")
    ap.add_argument("--v4-distill", type=int, default=0,
                    help="Jackrong/DeepSeek-V4-Distill-8000x reasoning traces, 7716 available (0 = skip)")
    ap.add_argument("--glm-reasoning", type=int, default=0,
                    help="Jackrong/GLM-5.1-Reasoning-1M-Cleaned, 746321 available (0 = skip)")
    ap.add_argument("--jackrong-max-output-words", type=int, default=600,
                    help="skip reasoning traces longer than this; their median is ~1150 words, which no "
                         "256-token completion budget can hold (0 = keep every length)")
    ap.add_argument("--arcee-start", type=float, default=0.66,
                    help="fraction of the jsonl to start range-reading at (the tool-calling rows)")
    ap.add_argument("--max-history-chars", type=int, default=8000)
    args = ap.parse_args()

    from cutrain.dataprep import prepare_data  # heavy (it builds the harness tool registry), and only openthoughts needs it

    harness = [t for t in prepare_data.HARNESS_TOOLS
               if t["function"]["name"] in ("run_command", "run_code", "task_done")]
    stats: Counter = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        gens = []
        if args.openthoughts:
            gens.append(openthoughts_rows(args.openthoughts, stats, prepare_data.BASE_SYSTEM, harness))
        if args.arcee:
            gens.append(arcee_rows(args.arcee, stats, args.arcee_start))
        if args.dapo:
            gens.append(dapo_rows(args.dapo, stats))
        for key, want in (("v4", args.v4_distill), ("glm", args.glm_reasoning)):
            if want:
                gens.append(jackrong_rows(key, want, stats, args.jackrong_max_output_words))
        for gen in gens:
            for row in gen:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[f"{row['source']} / {row['category']}"] += 1
    for k, v in sorted(stats.items()):
        if not k.startswith("arcee: skipped") or v > 1000:
            print(f"{v:7d}  {k}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
