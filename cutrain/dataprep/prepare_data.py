"""Build one unified ORPO dataset from Claude-style JSONL + Hugging Face preference sets.

Inputs:
  - Claude-style JSONL (auto-detected per line):
      Anthropic Messages record: {"system"?, "tools"?, "messages": [...], "rejected"?}
      Claude Code transcript line: {"type": "user"|"assistant", "message": {...}} (one file = one session)
  - opencode sessions (~/.local/share/opencode/opencode.db, read-only)
  - Hugging Face preference datasets

Unified row (one JSON per line, TRL conversational preference format + tools):
  {"id", "source", "category", "tools": [openai tool], "prompt": [msg], "chosen": [msg],
   "rejected": [msg], "rejected_origin": "dataset" | "synthetic:<kind>"}
msg = OpenAI chat message; assistant may carry "reasoning_content" and
"tool_calls": [{"type": "function", "function": {"name", "arguments": dict}}].
That is what the Nemotron chat template and the harness (tool_protocol: native) speak.

Claude `computer` / `bash` tool calls are mapped onto the harness tools (click, type_text,
run_command, ...) so the model learns the exact tool names it gets at inference.

  python -m cutrain.dataprep.prepare_data        # raw/*.jsonl + opencode db + default HF sets
  python -m cutrain.dataprep.prepare_data --claude "planning=~/dumps/p*.jsonl" --opencode --hf
"""
from __future__ import annotations

import argparse
import functools
import glob
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from cutrain.paths import HARNESS_ROOT, REPO

sys.path.insert(0, str(HARNESS_ROOT))
from src.harness.pipes import truncate_output  # noqa: E402
from src.harness.tools import gui, lsp, python_repl, report, skills_tools, terminal  # noqa: E402
from src.harness.tools.registry import GROUPS, ToolRegistry  # noqa: E402

# same registration as src/harness/runtime.build_registry, minus the runtime's pydantic/openai imports
_reg = ToolRegistry(HARNESS_ROOT / "src" / "prompts" / "overrides" / "tools.yaml")
for _mod in (terminal, python_repl, gui, lsp, skills_tools, report):
    _reg.register_module(_mod)
_reg.apply_overrides()
HARNESS_TOOLS = _reg.openai_tools()
BASE_SYSTEM = (HARNESS_ROOT / "src" / "prompts" / "base_system.md").read_text()
IMAGE = "[image]"  # ponytail: Nemotron-3-Nano-4B is text-only, screenshots become a placeholder
OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
DEFAULT_HF = ["mlabonne/orpo-dpo-mix-40k:default:train:3000", "nvidia/When2Call:train_pref:train:3000"]

INJECTED_RE = re.compile(r"<(system-reminder|ide_[a-z_]+)>.*?</\1>", re.S)  # Claude Code hook/IDE context
TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)
LIST_ITEM_RE = re.compile(r"\s*(\d+[.)]|[-*])\s")
# user pushback at the start of a follow-up message; --judge adds an SLM for paraphrases and typos
CORRECTION_RE = re.compile(
    r"^\W*(\[request interrupted|no\b|nope|wrong|that'?s (not|wrong)|not (what|quite|right)|i (said|meant|asked|wanted)"
    r"|you (didn'?t|did not|forgot|missed|broke)|why (did|would) you|undo|revert|don'?t\b"
    r"|still (not|broken|fail|doesn'?t|does not)|(it|this|that) (doesn'?t|does not|didn'?t|did not) work)", re.I)
KEYS = {"return": "enter", "escape": "esc", "page_down": "pagedown", "page_up": "pageup",
        "super": "win", "super_l": "win", "control": "ctrl", "control_l": "ctrl", "alt_l": "alt",
        "shift_l": "shift", "prior": "pageup", "next": "pagedown"}
CLICKS = {"left_click": ("click", {}), "middle_click": ("click", {"button": "middle"}),
          "right_click": ("right_click", {}), "double_click": ("double_click", {}),
          "mouse_move": ("move", {})}
JSON_TYPES = {str: "string", bool: "boolean", int: "integer", float: "number", list: "array", dict: "object"}


# --- Claude -> unified messages -------------------------------------------------------------

def _key(k: str) -> str:
    return KEYS.get(k.lower(), k.lower())


def map_call(name: str, inp: dict, state: dict) -> tuple[str, dict] | None:
    """Claude tool call -> harness tool call. None = no harness equivalent (conversation is cut there)."""
    if name in ("bash", "Bash"):
        return ("reset_shell", {}) if inp.get("restart") else ("run_command", {"cmd": inp["command"]})
    if name == "invalid":  # opencode's placeholder for a call the model garbled
        return None
    if name != "computer":
        return name, inp  # other tools (Read, Edit, TodoWrite, ...) stay as-is
    a = inp.get("action")
    if "coordinate" in inp:
        state["xy"] = inp["coordinate"]
    x, y = state.get("xy") or (None, None)  # computer_20241022 clicks at the last mouse_move
    if a == "screenshot":
        return "screenshot", {}
    if a in CLICKS and x is not None:
        return CLICKS[a][0], {"x": x, "y": y, **CLICKS[a][1]}
    if a == "left_click_drag" and "start_coordinate" in inp:
        sx, sy = inp["start_coordinate"]
        return "drag", {"x": sx, "y": sy, "to_x": x, "to_y": y}
    if a == "type":
        return "type_text", {"text": inp["text"]}
    if a == "key":
        parts = inp["text"].split("+")
        return ("hotkey", {"combo": "+".join(map(_key, parts))}) if len(parts) > 1 else ("press_key", {"key": _key(parts[0])})
    if a == "scroll" and inp.get("scroll_direction") in ("up", "down") and x is not None:
        n = int(inp.get("scroll_amount", 3))
        return "scroll", {"amount": n if inp["scroll_direction"] == "up" else -n, "x": x, "y": y}
    if a == "wait":
        return "run_command", {"cmd": f"sleep {inp.get('duration', 1)}"}
    return None  # triple_click, hold_key, zoom, cursor_position, horizontal scroll ...


def block_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append(IMAGE)
    return "\n".join(parts)


def assistant_msg(blocks, state: dict) -> tuple[dict | None, str | None]:
    """Anthropic assistant content -> (unified msg, unmapped-call reason)."""
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    reasoning, text, calls = [], [], []
    for b in blocks:
        t = b.get("type")
        if t == "thinking" and b.get("thinking", "").strip():  # Claude Code stores redacted (empty) thinking
            reasoning.append(b["thinking"])
        elif t == "text":
            text.append(b["text"])
        elif t == "tool_use":
            inp = b.get("input") or {}
            mapped = map_call(b["name"], inp, state)
            if mapped is None:
                return None, f"unmapped {b['name']}.{inp.get('action')}"
            calls.append({"type": "function", "function": {"name": mapped[0], "arguments": mapped[1]}})
    msg: dict = {"role": "assistant", "content": "\n".join(text).strip()}
    if reasoning:
        msg["reasoning_content"] = "\n".join(reasoning).strip()
    if calls:
        msg["tool_calls"] = calls
    return msg, None


def convert(messages: list, system: str) -> tuple[list[dict], str | None]:
    """Anthropic message list -> unified messages. Stops at the first call the harness can't express."""
    out: list[dict] = [{"role": "system", "content": system}]
    state: dict = {}
    for m in messages:
        if m["role"] == "assistant":
            msg, why = assistant_msg(m["content"], state)
            if msg is None:
                return out, why
            if out[-1]["role"] == "assistant":  # Claude Code writes one line per content block
                prev = out[-1]
                prev["content"] = "\n".join(filter(None, [prev["content"], msg["content"]]))
                if "reasoning_content" in msg:
                    prev["reasoning_content"] = "\n".join(filter(None, [prev.get("reasoning_content"), msg["reasoning_content"]]))
                prev.setdefault("tool_calls", []).extend(msg.get("tool_calls", []))
                if not prev["tool_calls"]:
                    del prev["tool_calls"]
            elif msg["content"] or msg.get("tool_calls") or msg.get("reasoning_content"):
                out.append(msg)
            continue
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        text = []
        for b in blocks:
            if b.get("type") == "tool_result":
                body = block_text(b.get("content")) or "(no output)"
                out.append({"role": "tool", "content": truncate_output(("error: " if b.get("is_error") else "") + body)})
            else:
                text.append(block_text([b]))
        text = INJECTED_RE.sub("", "\n".join(text)).strip()
        if text:
            if out[-1]["role"] == "user":
                out[-1]["content"] += "\n" + text
            else:
                out.append({"role": "user", "content": text})
    return out, None


def tool_schemas(declared: list | None, msgs: list[dict]) -> list[dict]:
    """Tool list for a conversation: the full harness set if it drives the GUI (what the model sees at
    inference), else only the harness tools it calls (~1.6k tokens saved); plus declared Anthropic tools
    and schemas inferred for undeclared ones (transcripts carry none)."""
    used = {c["function"]["name"] for m in msgs for c in m.get("tool_calls", [])}
    harness = {t["function"]["name"]: t for t in HARNESS_TOOLS}
    out = dict(harness) if used & set(GROUPS["gui"]) else {n: harness[n] for n in used if n in harness}
    for t in declared or []:
        if "input_schema" in t:  # built-ins (computer/bash/text_editor) have no schema
            out[t["name"]] = {"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
    seen: dict[str, list[dict]] = {}
    for m in msgs:
        for c in m.get("tool_calls", []):
            if c["function"]["name"] not in out:
                seen.setdefault(c["function"]["name"], []).append(c["function"]["arguments"])
    for name, calls in seen.items():
        props = {k: {"type": JSON_TYPES.get(type(v), "string")} for a in calls for k, v in a.items()}
        required = [k for k in props if all(k in a for a in calls)]
        out[name] = {"type": "function", "function": {"name": name, "description": "", "parameters": {
            "type": "object", "properties": props, "required": required}}}
    return list(out.values())


def read_claude(pattern: str):
    """Yield conversations (id, source, category, system, tools, messages, rejected) from Claude-style JSONL."""
    category, _, pattern = pattern.partition("=") if "=" in pattern else ("", "", pattern)
    for path in sorted(glob.glob(str(Path(pattern).expanduser()))):
        src, cat, transcript, plan = f"claude:{Path(path).name}", category or Path(path).stem, [], False
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in row:
                    yield f"{path}#{n}", src, cat, row.get("system"), row.get("tools"), row["messages"], row.get("rejected")
                elif row.get("type") in ("user", "assistant") and not row.get("isSidechain") and not row.get("isMeta"):
                    transcript.append(row["message"])
                    plan = plan or row.get("permissionMode") == "plan"
        # ponytail: transcript = file order; branches (parentUuid forks after rewinds) are not untangled
        if transcript:  # session files are uuid-named, so category comes from Claude Code's plan mode
            yield path, src, category or ("planning" if plan else "execution"), None, None, transcript, None


def read_opencode(db_path: str):
    """Yield conversations from opencode's SQLite store (opened read-only), as Anthropic-style messages.

    One opencode assistant message holds several model calls ("step-start" parts); each becomes an
    assistant turn followed by a user turn with its tool results. Agent plan/build -> planning/execution.
    """
    import sqlite3

    db = sqlite3.connect(f"file:{Path(db_path).expanduser()}?mode=ro", uri=True)
    rows = db.execute("select session_id, id, data from message order by session_id, time_created, id")
    for sid, group in itertools.groupby(rows, key=lambda r: r[0]):
        msgs, agent = [], None
        for _, mid, data in group:
            m = json.loads(data)
            parts = [json.loads(d) for (d,) in db.execute("select data from part where message_id = ? order by id", (mid,))]
            if m["role"] == "user":
                agent = agent or m.get("agent")
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": p["text"]} if p["type"] == "text"
                    else {"type": "image"} if p.get("mime", "").startswith("image/")
                    else {"type": "text", "text": f"[file {p.get('filename')}]"}
                    for p in parts if p["type"] in ("text", "file")]})
                continue
            steps: list[list[dict]] = [[]]
            for p in parts:
                if p["type"] == "step-start":
                    steps.append([])
                else:
                    steps[-1].append(p)
            for step in steps:
                blocks, results = [], []
                for p in step:
                    if p["type"] == "reasoning" and p["text"].strip():
                        blocks.append({"type": "thinking", "thinking": p["text"]})
                    elif p["type"] == "text":
                        blocks.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "tool":
                        st = p["state"]
                        blocks.append({"type": "tool_use", "id": p["callID"], "name": p["tool"], "input": st.get("input") or {}})
                        results.append({"type": "tool_result", "tool_use_id": p["callID"], "is_error": st.get("status") == "error",
                                        "content": st.get("output") or st.get("error") or ""})
                if blocks:
                    msgs.append({"role": "assistant", "content": blocks})
                if results:
                    msgs.append({"role": "user", "content": results})
        cat = {"plan": "planning", "build": "execution"}.get(agent, agent or "opencode")
        yield f"opencode:{sid}", "opencode", cat, None, None, msgs, None


# --- rejected synthesis ----------------------------------------------------------------------

def _jitter(call: dict, rng: random.Random) -> dict:
    a = dict(call["function"]["arguments"])
    for k in ("x", "y"):
        a[k] = max(0, int(a[k]) + rng.choice((-1, 1)) * rng.randint(120, 400))
    return {"type": "function", "function": {"name": call["function"]["name"], "arguments": a}}


def _drop_steps(text: str, rng: random.Random) -> str | None:
    lines = text.splitlines()
    items = [i for i, line in enumerate(lines) if LIST_ITEM_RE.match(line)]
    if len(items) >= 3:
        drop = set(rng.sample(items, len(items) // 2))
        return "\n".join(line for i, line in enumerate(lines) if i not in drop)
    return text[: len(text) // 3] if len(text) > 200 else None


def synth_rejected(m: dict, prev: dict | None, rng: random.Random) -> tuple[dict | None, str | None]:
    """Rule-based negative for one assistant step; gen_rejected.py swaps these for real model answers."""
    base = {k: v for k, v in m.items() if k != "tool_calls"}
    calls, prev_calls = m.get("tool_calls"), (prev or {}).get("tool_calls")
    opts: list[tuple[str, dict]] = []
    if calls:
        if {"x", "y"} <= calls[0]["function"]["arguments"].keys():
            opts.append(("wrong_coords", {**base, "tool_calls": [_jitter(calls[0], rng), *calls[1:]]}))
        if prev_calls and prev_calls != calls:
            opts.append(("repeat_prev", {**base, "tool_calls": prev_calls}))
        elif prev_calls and prev_calls == calls:
            # A step that repeats the previous action is the exact "keeps calling tools" habit.
            # Pairing it with "Done" would mark the repeat as correct and stopping as the error,
            # so emit no pair instead of reinforcing the loop.
            return None, "skip: repeated action"
        opts.append(("premature_done", {"role": "assistant", "content": "Done - the task is complete."}))
    else:
        if prev_calls:
            opts.append(("no_stop", {**base, "content": "", "tool_calls": prev_calls}))
        dropped = _drop_steps(m.get("content", ""), rng)
        if dropped:
            opts.append(("dropped_steps", {**base, "content": dropped}))
    if not opts:
        return None, None
    kind, rej = rng.choice(opts)
    return rej, f"synthetic:{kind}"


JUDGE_PROMPT = (
    "You label chat logs. After the assistant's turn the user replied. Label the reply PUSHBACK if it says the "
    "assistant's last answer or action was wrong, unwanted, incomplete, too complicated or did not work, or tells it to "
    "redo it differently. Label it FINE for approvals, 'continue', new requests, follow-up questions, or extra "
    "information.\n\nExamples:\n'yes' -> FINE\n'continue' -> FINE\n'what about the other board?' -> FINE\n"
    "'the log is in logs/ folder' -> FINE\n'it does not compile' -> PUSHBACK\n'use the project venv instead' -> PUSHBACK\n"
    "'you were supposed to test it' -> PUSHBACK\n\nReply with exactly one word: PUSHBACK or FINE.")
_judge: tuple | None = None  # (OpenAI client, model), set by --judge


@functools.lru_cache(maxsize=None)
def is_correction(prev: str, reply: str) -> bool:
    """Does the user's reply push back on the assistant turn `prev`? Regex first, then the SLM judge if set."""
    if CORRECTION_RE.match(reply):
        return True
    if _judge is None:
        return False
    client, model = _judge
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=8, reasoning_effort="none", messages=[
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Assistant's last turn:\n{prev[-1500:]}\n\nUser reply:\n{reply[:1500]}"}])
    return "PUSHBACK" in (r.choices[0].message.content or "").upper()


def _summary(m: dict) -> str:
    calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])[:200]})" for c in m.get("tool_calls", []))
    return "\n".join(filter(None, [m.get("content", ""), calls and f"[called {calls}]"]))


def _failed(msgs: list[dict], i: int) -> bool:
    """Tool results right after step i contain an error (is_error: failed/non-zero exit, user rejected the call)."""
    j = i + 1
    while j < len(msgs) and msgs[j]["role"] == "tool":
        if msgs[j]["content"].startswith("error:"):
            return True
        j += 1
    return False


def outcome_pairs(msgs: list[dict], head: int) -> dict[int, tuple]:
    """Real negatives from what happened next: assistant index -> (chosen, rejected, origin).

    - error: the step's tool call failed and the next step retried the same tool without error ->
      chosen = the step with the retry's call, rejected = the step as it was.
    - feedback: the user's next message pushes back (is_correction: "no", "not what I wanted", interrupt...)
      -> rejected = the step they reacted to, chosen = the agent's first step after the correction.
    Failed/corrected steps with no usable fix map to (None, None, reason) so they never become chosen.
    """
    out: dict[int, tuple] = {}
    for i in range(head, len(msgs)):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls") and _failed(msgs, i):
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "tool":
                j += 1
            fix = msgs[j] if j < len(msgs) and msgs[j]["role"] == "assistant" else {}
            if (fix.get("tool_calls") and not _failed(msgs, j)
                    and fix["tool_calls"][0]["function"]["name"] == m["tool_calls"][0]["function"]["name"]):
                out[i] = ({**m, "tool_calls": fix["tool_calls"]}, m, "outcome:error")
            else:
                out[i] = (None, None, "skip: failed step, no fix")
        elif m["role"] == "user" and i >= head:
            r = max((k for k in range(head, i) if msgs[k]["role"] == "assistant"), default=None)
            if r is None or not is_correction(_summary(msgs[r]), m["content"]):
                continue
            c = next((k for k in range(i + 1, len(msgs)) if msgs[k]["role"] != "tool"), None)
            if c is not None and msgs[c]["role"] == "assistant":
                out[r] = (msgs[c], msgs[r], "feedback:user_correction")
            else:
                out[r] = (None, None, "skip: corrected step, no follow-up")
    return out


# --- sources -> unified rows -----------------------------------------------------------------

def step_rows(convs, rng: random.Random, stats: Counter, max_chars: int):
    """Conversations -> one unified row per assistant step.

    Prompt = system + first user msg + as many of the latest messages as fit in `max_chars`
    (sliding window, oldest turns dropped). Without it every step repeats the whole history and a
    few hundred opencode sessions become GBs of JSONL.
    """
    for cid, source, cat, system, tools, messages, rejected in convs:
        msgs, why = convert(messages, block_text(system) or BASE_SYSTEM)
        if why:
            stats[f"cut: {why}"] += 1
        head = next((i + 1 for i, m in enumerate(msgs) if m["role"] == "user"), None)
        if head is None:
            continue
        schemas = tool_schemas(tools, msgs)
        sizes = [len(json.dumps(m, ensure_ascii=False)) for m in msgs]
        last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
        pairs = outcome_pairs(msgs, head)
        start, total, prev = head, 0, None
        for i in range(head, len(msgs)):
            m = msgs[i]
            if m["role"] == "assistant":
                while total > max_chars and start < i:  # drop oldest msg, never leave orphan tool results
                    total -= sizes[start]
                    start += 1
                    while start < i and msgs[start]["role"] == "tool":
                        total -= sizes[start]
                        start += 1
                chosen, rej, origin = pairs.get(i) or (m, None, None)
                if chosen is None:
                    stats[origin] += 1
                else:
                    if rej is None and rejected is not None and i == last and not why:
                        rej, _ = assistant_msg(rejected.get("content", rejected) if isinstance(rejected, dict) else rejected, {})
                        origin = "dataset"
                    if rej is None:
                        rej, origin = synth_rejected(m, prev, rng)
                    if rej is not None and rej != chosen:
                        yield {"id": f"{cid}:{i}", "source": source, "category": cat, "tools": schemas,
                               "prompt": msgs[:head] + msgs[start:i], "chosen": [chosen], "rejected": [rej],
                               "rejected_origin": origin}
                    else:
                        stats[f"skip: {origin or 'no rejected'}"] += 1
                prev = m
            total += sizes[i]


def hf_msg(m: dict) -> dict:
    out = {"role": m["role"], "content": m.get("content") or ""}
    hit = TOOLCALL_RE.search(out["content"]) if out["role"] == "assistant" else None
    if hit:
        try:
            calls = json.loads(hit.group(1))
        except json.JSONDecodeError:
            return out
        out["content"] = out["content"][: hit.start()].strip()
        out["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                             for c in calls]
    return out


def hf_rows(spec: str, stats: Counter):
    from datasets import load_dataset

    name, config, split, limit = (spec.split(":") + [""] * 3)[:4]
    ds = load_dataset(name, config or None, split=split or "train").shuffle(seed=0)
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))
    for i, row in enumerate(ds):
        tools = [{"type": "function", "function": json.loads(t) if isinstance(t, str) else t} for t in row.get("tools") or []]
        if "chosen_response" in row:  # nvidia/When2Call style
            prompt, chosen, rejected = row["messages"], row["chosen_response"], row["rejected_response"]
        elif isinstance(row["chosen"], list):  # full conversations (orpo-dpo-mix, ultrafeedback_binarized)
            prompt = row["chosen"][:-1] or [{"role": "user", "content": row["prompt"]}]
            chosen, rejected = row["chosen"][-1], row["rejected"][-1]
        else:  # plain strings
            prompt = [{"role": "user", "content": row["prompt"]}]
            chosen, rejected = {"role": "assistant", "content": row["chosen"]}, {"role": "assistant", "content": row["rejected"]}
        if not prompt or chosen == rejected:
            stats["skip: hf bad row"] += 1
            continue
        yield {"id": f"{name}#{i}", "source": f"hf:{name}", "category": "tool_call" if tools else "chat",
               "tools": tools, "prompt": [hf_msg(m) for m in prompt], "chosen": [hf_msg(chosen)],
               "rejected": [hf_msg(rejected)], "rejected_origin": "dataset"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude", nargs="*", default=[str(REPO / "raw" / "*.jsonl")], metavar="[CATEGORY=]GLOB",
                    help="Claude-style JSONL files; category defaults to file stem (planning.jsonl -> planning)")
    ap.add_argument("--opencode", nargs="*", default=[str(OPENCODE_DB)] if OPENCODE_DB.exists() else [], metavar="DB",
                    help="opencode SQLite store(s); pass --opencode with no value to skip")
    ap.add_argument("--hf", nargs="*", default=DEFAULT_HF, metavar="NAME[:CONFIG[:SPLIT[:LIMIT]]]",
                    help="HF preference datasets; pass --hf with no value to skip")
    ap.add_argument("--out", default=str(REPO / "data" / "unified.jsonl"))
    ap.add_argument("--max-history-chars", type=int, default=8000,
                    help="per-step prompt history budget (chars of JSON, ~4 per token); train_orpo.py trims the rest exactly")
    ap.add_argument("--judge", metavar="MODEL", help="SLM that labels follow-up user messages as corrections; "
                    "gemma4:e4b scored 22/22 on a hand-labelled sample (gemma4:e2b 18/22)")
    ap.add_argument("--judge-url", default="http://localhost:11434/v1", help="OpenAI-compatible server (default Ollama)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.judge:
        from openai import OpenAI

        global _judge
        _judge = (OpenAI(base_url=args.judge_url, api_key="none"), args.judge)

    rng, stats = random.Random(args.seed), Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        sources = ([step_rows(read_claude(p), rng, stats, args.max_history_chars) for p in args.claude]
                   + [step_rows(read_opencode(db), rng, stats, args.max_history_chars) for db in args.opencode]
                   + [hf_rows(s, stats) for s in args.hf])
        for rows in sources:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[f"{row['source']} / {row['category']} / {row['rejected_origin']}"] += 1
    for k, v in sorted(stats.items()):
        print(f"{v:7d}  {k}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
