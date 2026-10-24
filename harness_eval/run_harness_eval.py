"""Run harness_eval/cases.py through le-pooh-assistant's real turn loop (LLM_Parser.respond) against any
OpenAI-compatible endpoint, normally serve_adapter.py serving a trained adapter.

    .venv-harness/bin/python harness_eval/run_harness_eval.py --base-url http://127.0.0.1:8011/v1 \
        --model cu --out out/harness-eval/<name>

Nothing of the user's is touched: the harness runs on a copy of memory.db inside --out, calendar writes and
cloud escalation are dry-run stubs, and only reads reach the network (weather, time, fx, directions, live
web search, calendar list). Writes <out>/results.json and <out>/report.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, CATEGORIES  # noqa: E402

DEFAULT_HARNESS = Path("/home/pooh/plugin_assistant_harness/le-pooh-assistant")


def spoken_ok(text: str) -> bool:
    """The harness's voice rule: no markdown or lists, a couple of sentences."""
    if re.search(r"[*#`]|^\s*([-•]|\d+[.)])\s", text, re.M):
        return False
    return len(re.findall(r"[.!?](\s|$)", text)) <= 3 and len(text) <= 400


def grade(check: dict, turn: dict, todos: list[dict] | None, writes) -> list[str]:
    """Failed check descriptions for one turn; empty means pass."""
    fails = []
    ran = [c["tool"] for c in turn["calls"]]
    asked = [c["tool"] for c in turn["requested"]]
    for want in check.get("expect", []):
        if not any(alt in ran for alt in want.split("|")):
            fails.append(f"expected {want}, ran {ran or 'nothing'}")
    for bad in check.get("forbid", []):
        if bad in ran:
            fails.append(f"ran forbidden {bad}")
    if check.get("no_tools") and ran:
        fails.append(f"should answer directly, ran {ran}")
    if check.get("no_writes"):
        wrote = [c["tool"] for c in turn["calls"] if writes(c)]
        if wrote:
            fails.append(f"wrote via {wrote}")
    for bad in check.get("never_request", []):
        if bad in asked:
            fails.append(f"asked for {bad} (blocked or not)")
    for tool, want in check.get("args", {}).items():
        calls = [c["args"] or {} for c in turn["calls"] if c["tool"] == tool]
        if calls and not any(all(re.search(rx, str(a.get(k, "")), re.I) for k, rx in want.items()) for a in calls):
            fails.append(f"{tool} args {calls} do not match {want}")
    reply = turn["response"]
    for rx in check.get("say", []):
        if not re.search(rx, reply, re.I):
            fails.append(f"reply lacks /{rx}/")
    for rx in check.get("not_say", []):
        if re.search(rx, reply, re.I):
            fails.append(f"reply has /{rx}/")
    want = check.get("todos")
    if want is not None and todos is not None:
        def has(rx, done=None):
            return any(re.search(rx, t["text"], re.I) and (done is None or bool(t["done"]) == done) for t in todos)
        fails += [f"todo /{rx}/ not open" for rx in want.get("open", []) if not has(rx, False)]
        fails += [f"todo /{rx}/ not done" for rx in want.get("done", []) if not has(rx, True)]
        fails += [f"todo /{rx}/ still there" for rx in want.get("absent", []) if has(rx)]
        if "count" in want and len(todos) != want["count"]:
            fails.append(f"{len(todos)} todos, expected {want['count']}: {[t['text'] for t in todos]}")
    return fails


async def run(args) -> list[dict]:
    from bus import MessageBus
    from llm import LLM_Parser
    from loader import PluginLoader
    from memory import History
    from plugin_base import tool_result, writes_call

    bus = MessageBus()
    loader = PluginLoader(bus, "plugins")
    llm = LLM_Parser(bus, loader)
    llm.register()
    await loader.load_all_plugins()
    llm.register_tools()
    offered = [t["function"]["name"] for t in llm.tools]
    print("tools offered:", offered, flush=True)

    plugins = {inst.get_metadata().name: inst for inst in loader.get_all_plugin().values()}
    stubs: dict = {}
    for name, inst in plugins.items():
        real = inst.llm_run

        async def dispatch(_real=real, _inst=inst, _name=name, **kw):
            if _name in stubs:
                stub = stubs[_name]
                return stub if isinstance(stub, dict) and "ok" in stub else tool_result(stub)
            if _name in ("calendar", "not_a_good_answer") and writes_call(_inst, kw):
                return tool_result({"dry_run": True, "status": "created", "tool": _name, "args": kw})
            return await _real(**kw)
        inst.llm_run = dispatch

    def writes(call):
        p = plugins.get(call["tool"])
        return p is not None and writes_call(p, call["args"])

    todo = plugins.get("todolist")
    conn = todo.storage.conn if todo else None

    def todos_now():
        if conn is None:
            return None
        return [dict(r) for r in conn.execute("SELECT text, done FROM todos ORDER BY id")]

    # The harness's own History (on the run's private memory.db), so whatever it
    # keeps between turns is exactly what production keeps. A new session per case.
    history = History(bus)
    history._core_events()
    cur: dict = {}

    @bus.subscribe("plugin.invoked")
    async def on_invoked(data, **kw):
        cur["calls"].append({"tool": data.get("plugin"), "args": data.get("arguments") or {}})

    @bus.subscribe("llm.response.unparse")
    async def on_raw(data, **kw):
        msg = data.get("results")
        for tc in getattr(msg, "tool_calls", None) or []:
            cur["requested"].append({"tool": tc.function.name, "args": tc.function.arguments})

    @bus.subscribe("intent.response")
    async def on_resp(data, **kw):
        cur["response"] = data.get("response") or ""

    selected = [c for c in CASES if (not args.only or c["id"] in args.only)
                and (not args.category or c["cat"] in args.category)]
    results = []
    for case in selected:
        history.session_id = uuid.uuid4().hex
        stubs.clear()
        stubs.update(case.get("stubs", {}))
        if conn is not None:
            conn.execute("DELETE FROM todos")
            for item in case.get("seed", []):
                text, done = (item, 0) if isinstance(item, str) else item
                conn.execute("INSERT INTO todos (text, owner, done, created_ts) VALUES (?, 'user', ?, ?)",
                             (text, int(done), time.time() - 3600))
            conn.commit()
        turns = []
        for i, utt in enumerate(case["turns"]):
            cur.clear()
            cur.update(calls=[], requested=[], response="")
            t0 = time.monotonic()
            error = None
            try:
                # Through the bus like a real query: History records it, LLM_Parser answers.
                await bus.publish_async("intent.query", {"utterance": utt, "context": {}})
            except Exception as e:  # noqa: BLE001 - one broken case must not end the run
                error = f"{type(e).__name__}: {e}"
            if not cur["response"] and not error:
                error = "no intent.response (a subscriber raised; the bus swallows it)"
            turn = {"utterance": utt, "calls": cur["calls"], "requested": cur["requested"],
                    "response": cur["response"], "seconds": round(time.monotonic() - t0, 1), "error": error}
            last = i == len(case["turns"]) - 1
            check = case.get("check", {}) if last else case.get("turn_checks", {}).get(i)
            if check is not None:
                turn["fails"] = grade(check, turn, todos_now(), writes) + ([f"error {error}"] if error else [])
            turns.append(turn)
        fails = [f for t in turns for f in t.get("fails", [])]
        final = turns[-1]
        rec = {"id": case["id"], "cat": case["cat"], "why": case.get("why", ""), "pass": not fails,
               "fails": fails, "spoken": spoken_ok(final["response"]), "seconds": final["seconds"],
               "todos_after": todos_now(), "turns": turns}
        results.append(rec)
        print(f"[{'PASS' if rec['pass'] else 'FAIL'}] {case['cat']:10s} {case['id']:18s} {final['seconds']:5.1f}s "
              f"ran={[c['tool'] for c in final['calls']]} :: {final['response'][:110]!r}"
              + (f"\n        {fails}" if fails else ""), flush=True)
    return results


def report(results: list[dict], name: str) -> str:
    by = defaultdict(list)
    for r in results:
        by[r["cat"]].append(r)
    total = sum(r["pass"] for r in results)
    lines = [f"# Harness eval: {name}", "",
             f"**{total}/{len(results)} passed**, spoken style {sum(r['spoken'] for r in results)}/{len(results)}, "
             f"median turn {sorted(r['seconds'] for r in results)[len(results) // 2]}s", "",
             "| category | passed |", "|---|---|"]
    lines += [f"| {cat} | {sum(r['pass'] for r in by[cat])}/{len(by[cat])} |" for cat in CATEGORIES if by[cat]]
    lines += ["", "## Cases", "", "| | case | category | reply | failed checks |", "|---|---|---|---|---|"]
    for r in results:
        reply = r["turns"][-1]["response"].replace("|", "\\|").replace("\n", " ")[:140]
        fails = "; ".join(r["fails"]).replace("|", "\\|")[:200]
        lines.append(f"| {'✅' if r['pass'] else '❌'} | {r['id']} | {r['cat']} | {reply} | {fails} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    ap.add_argument("--only", nargs="*", help="case ids")
    ap.add_argument("--category", nargs="*", choices=list(CATEGORIES))
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Private copy of the harness's state and config, set before the harness imports read them.
    shutil.copy(args.harness / "memory.db", out / "memory.db")
    cfg = (args.harness / "config.yml").read_text()
    cfg = re.sub(r"(?m)^(\s*db_path:).*$", rf"\1 {out / 'memory.db'}", cfg)
    cfg = re.sub(r"(?m)^(\s*dir:\s*)traces\b.*$", rf"\1{out / 'traces'}", cfg)
    (out / "config.yml").write_text(cfg)
    os.environ.update(POOH_CONFIG=str(out / "config.yml"), LLM_BASE_URL=args.base_url, LLM_MODEL=args.model)
    os.chdir(args.harness / "app")
    sys.path.insert(0, str(args.harness / "app"))

    results = asyncio.run(run(args))
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    md = report(results, f"{args.model} ({out.name})")
    (out / "report.md").write_text(md)
    for leftover in ("memory.db", "memory.db-wal", "memory.db-shm"):
        (out / leftover).unlink(missing_ok=True)
    print("\n" + md.split("## Cases")[0])
    print(f"wrote {out / 'results.json'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
