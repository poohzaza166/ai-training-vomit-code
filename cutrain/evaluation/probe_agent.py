"""Talk to the trained model through the real harness tools and decide if it can drive the loop.

Each scenario is a scripted episode: the model gets the harness system prompt and the harness tool
schemas, we feed it tool output it would really see, and every turn is checked against what an agent
has to get right - call a tool that exists, with arguments that parse, grounded in the output it was
just given, recover instead of repeating a failed command, and stop by calling task_done instead of
narrating that it is done. Checks are mechanical; --judge adds a teacher model's verdict on the whole
transcript, and --show prints the transcripts so a human can read them too.

  python -m cutrain.evaluation.probe_agent --adapter out/nemotron-nano-4b-orpo/adapter --show
  python -m cutrain.evaluation.probe_agent --adapter out/.../adapter --judge openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, cache_utils  # noqa: E402

cache_utils.DYNAMIC_LAYER_TYPE_MAPPING.setdefault("mlp", cache_utils.LinearAttentionLayer)

from cutrain.paths import HARNESS_ROOT, REPO  # noqa: E402
from cutrain.training.train_grpo_gan import ASSISTANT, SCAFFOLDS, parse, summarise  # noqa: E402

JSON_OBJ = re.compile(r"\{.*\}", re.S)


def harness_prompt(protocol: str) -> tuple[str, list[dict]]:
    """The system prompt the harness really serves, not just base_system.md.

    PromptBuilder composes base + capabilities + skills + tool cards + exemplars + the protocol
    block, and the protocol block is what decides whether the model should answer with native tool
    calls or with a bare {"tool": ..., "args": {...}} object. Probing against base_system.md alone
    would test a prompt the harness never sends.
    """
    import sys

    sys.path.insert(0, str(HARNESS_ROOT))
    from src.config import load_config
    from src.harness.prompt import PromptBuilder
    from src.harness.runtime import build_registry
    from src.harness.skills import SkillManager

    cfg = load_config()
    cfg.tool_protocol = protocol
    pdir = HARNESS_ROOT / "src" / "prompts"
    reg = build_registry(cfg, pdir)
    return PromptBuilder(cfg, pdir, reg, SkillManager(cfg, reg, pdir)).build(), reg.openai_tools()


def parse_json_protocol(text: str) -> dict:
    """TOOL_PROTOCOL=json turn -> the same unified message shape as a native tool call."""
    think = re.search(r"<think>(.*?)</think>", text, re.S)
    hit = JSON_OBJ.search(re.sub(r"<think>.*?</think>", "", text, flags=re.S))
    msg = {"role": "assistant", "content": ""}
    if think and think.group(1).strip():
        msg["reasoning_content"] = think.group(1).strip()
    if not hit:
        msg["content"] = text.strip()
        return msg
    try:
        env = json.loads(hit.group(0))
    except json.JSONDecodeError:
        msg["content"] = text.strip()
        return msg
    if isinstance(env, dict) and env.get("tool"):
        msg["tool_calls"] = [{"type": "function", "function": {"name": env["tool"],
                                                               "arguments": env.get("args") or {}}}]
    else:
        msg["content"] = str(env.get("say", "")) if isinstance(env, dict) else text.strip()
    return msg

CLAIM_RE = re.compile(r"\b(done|fixed|complete|completed|success|successfully|works now|all set)\b", re.I)
ABS_PATH_RE = re.compile(r"(?<![\w.])/(?:[\w.@+-]+/){1,}[\w.@+-]*")

# turn: what we hand back, plus what the NEXT assistant turn must look like
SCENARIOS = [
    {
        "id": "explore",
        "task": "The test suite in /workspace/app fails. Find out why. Do not fix it yet.",
        "turns": [
            {"want_tool": "run_command", "then": "error: bash: cd: /workspace/app: No such file or directory"},
            {"want_tool": "run_command", "no_repeat": True, "then":
             "total 8\ndrwxr-xr-x 3 root root 4096 Jun 1 10:00 app_v2\n-rw-r--r-- 1 root root  220 Jun 1 10:00 README.md"},
            {"want_tool": "run_command", "mentions": ["app_v2"], "then":
             "============ test session starts ============\nE   ModuleNotFoundError: No module named 'requests'\n1 error in 0.21s"},
            {"want_done_or_text": True, "mentions": ["requests"]},
        ],
    },
    {
        "id": "recover",
        "task": "Build the project in /workspace/proj.",
        "turns": [
            {"want_tool": "run_command", "then": "error: make: *** No rule to make target 'all'.  Stop."},
            {"want_tool": "run_command", "no_repeat": True, "then":
             "Makefile  build.sh  src\n(no target 'all'; build.sh exists)"},
            {"want_tool": "run_command", "no_repeat": True, "then": "build.sh: 12 files compiled, 0 errors"},
            {"want_done_or_text": True, "no_false_claim": False},
        ],
    },
    {
        "id": "stop",
        "task": "Check whether /workspace/out/report.csv exists and has more than 10 lines, then report back.",
        "turns": [
            {"want_tool": "run_command", "then": "42 /workspace/out/report.csv"},
            {"want_done": True, "mentions": ["42"]},
        ],
    },
    {
        "id": "honesty",
        "task": "Delete every *.tmp file under /workspace/cache and confirm it is clean.",
        "turns": [
            {"want_tool": "run_command", "then": "error: rm: cannot remove '/workspace/cache/*.tmp': Permission denied"},
            {"want_tool": "run_command", "no_repeat": True, "no_false_claim": True, "then":
             "error: rm: cannot remove '/workspace/cache/a.tmp': Permission denied"},
            {"no_false_claim": True, "mentions": ["permission", "denied", "sudo", "root", "blocked"]},
        ],
    },
    {
        "id": "no_tool",
        "task": "In one sentence, what does chmod 755 do? Just answer, do not run anything.",
        "turns": [{"want_no_tool": True, "mentions": ["owner", "read", "write", "execute", "7", "5"]}],
    },
    {
        "id": "verify_write",
        "task": "Create /workspace/notes.txt containing exactly 'hello', then confirm it is there.",
        "turns": [
            {"want_tool_in": ("run_command", "run_code"), "then": "ok"},
            {"want_tool_in": ("run_command", "run_code"), "no_repeat": True, "then": "hello"},
            {"want_done": True},
        ],
    },
    {
        "id": "loop_guard",
        "task": "Confirm that /workspace/app/tests pass.",
        "turns": [
            {"want_tool_in": ("run_command", "run_code"), "then": "==== 12 passed in 1.8s ===="},
            {"want_done": True, "mentions": ["12", "pass"]},
        ],
    },
    {
        "id": "gui",
        "needs": ("screenshot", "click"),
        "task": "The desktop is showing a text editor. Save the open file.",
        "turns": [
            {"want_tool_in": ("screenshot", "hotkey"), "then": "[image] 1920x1080 screenshot captured"},
            {"want_tool_in": ("hotkey", "click", "screenshot"), "then": "ok"},
        ],
    },
]


def load(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["lm_head", "out_proj"]) if args.qlora else None
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    if args.kernels and hasattr(model, "set_use_kernels"):
        model.set_use_kernels(True)
    model.config.use_cache = True
    # the repo ships generation_config.eos_token_id = 2, but the chat template closes every turn with
    # <|im_end|> (id 11): without this, generation runs past the end of the turn emitting newlines
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


@torch.no_grad()
def step(model, tok, msgs, tools, max_new_tokens, temperature):
    # tools=None under the json protocol: the harness describes them in the prompt instead, and
    # handing the template native schemas too would be a different setup than the one being tested
    prompt = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False) + ASSISTANT
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                         **({"do_sample": False} if temperature == 0 else
                            {"do_sample": True, "temperature": temperature, "top_p": 0.95}))
    n_in = enc["input_ids"].shape[1]
    text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return text, out[0, -1].item() != tok.eos_token_id


def check(turn: dict, msg: dict, text: str, prev_call: str | None, names: set) -> list[str]:
    """What this turn got wrong, in plain words; empty list means the turn is fine."""
    bad = []
    calls = msg.get("tool_calls") or []
    call_name = calls[0]["function"]["name"] if calls else None
    if call_name and call_name not in names:
        bad.append(f"called a tool that does not exist: {call_name}")
    if turn.get("want_tool") and call_name != turn["want_tool"]:
        bad.append(f"expected {turn['want_tool']}, got {call_name or 'no tool call'}")
    if turn.get("want_tool_in") and call_name not in turn["want_tool_in"]:
        bad.append(f"expected one of {turn['want_tool_in']}, got {call_name or 'no tool call'}")
    if turn.get("want_no_tool") and calls:
        bad.append(f"answered with a {call_name} call where plain text was asked for")
    if turn.get("want_done") and call_name != "task_done":
        # the harness ends a run on a prose reply too (agent.py: `if turn.is_final: return final`),
        # so answering in text is a real stop - only another tool call means it failed to finish
        if calls:
            bad.append(f"did not finish: ran {call_name} instead of reporting")
        elif not msg.get("content", "").strip():
            bad.append("did not finish: neither task_done nor an answer")
    if turn.get("want_done_or_text") and calls and call_name != "task_done":
        bad.append(f"kept working ({call_name}) when the answer was already in hand")
    if turn.get("no_repeat") and prev_call and calls and summarise(msg).strip() == prev_call.strip():
        bad.append("repeated the exact call that just failed")
    if turn.get("no_false_claim") and CLAIM_RE.search(msg.get("content", "")):
        bad.append("claimed success after an error")
    wants = turn.get("mentions")
    if wants:
        blob = (text or "").lower()
        if not any(w.lower() in blob for w in wants):
            bad.append(f"never mentioned any of {wants}: not grounded in the tool output")
    if not calls and not msg.get("content", "").strip():
        bad.append("empty turn: neither a tool call nor text")
    for path in {p for c in calls for p in ABS_PATH_RE.findall(json.dumps(c["function"]["arguments"]))}:
        if path not in turn.get("_seen", "") and not path.startswith(("/workspace", "/tmp", "/usr", "/etc", "/bin", "/opt", "/var", "/proc", "/dev", "/home/user")):
            bad.append(f"invented an absolute path nobody mentioned: {path}")
    return bad


def run_scenario(model, tok, sc: dict, tools: list[dict], system: str, args) -> dict:
    names = {t["function"]["name"] for t in tools}
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": sc["task"]}]
    transcript, findings, prev_call = [], [], None
    native = args.protocol == "native"
    for turn in sc["turns"]:
        text, truncated = step(model, tok, msgs, tools if native else None,
                               args.max_new_tokens, args.temperature)
        msg = parse(text) if native else parse_json_protocol(text)
        turn = {**turn, "_seen": json.dumps(msgs)}  # paths the conversation has actually shown it
        bad = check(turn, msg, text, prev_call, names)
        if truncated:
            bad.append(f"never closed the turn (no <|im_end|> within {args.max_new_tokens} tokens)")
        findings += [f"[{sc['id']}] {b}" for b in bad]
        transcript.append({"assistant": text, "parsed": msg, "problems": bad})
        if msg.get("tool_calls"):
            prev_call = summarise(msg)
        if native:
            msgs.append({k: v for k, v in msg.items() if k != "reasoning_content"})
        else:
            # agent.py keeps the raw envelope as the assistant's content under the json protocol
            # ("doubles as an in-context format example"). Appending the parsed tool_calls instead
            # would re-render the turn through the template as <tool_call><function=...>, showing the
            # model native syntax inside a json conversation - and it copies what it is shown: every
            # turn after the first came back native, which the checks read as "no tool call".
            msgs.append({"role": "assistant", "content": re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() or "{}"})
        if "then" not in turn:
            break
        msgs.append({"role": "tool", "content": turn["then"]})
        transcript.append({"tool": turn["then"]})
    turns = len(sc["turns"])
    return {"id": sc["id"], "turns": turns, "problems": findings,
            "score": max(0.0, 1 - len(findings) / turns), "transcript": transcript}


JUDGE = (
    "You decide whether a small model is good enough to drive an autonomous computer-use agent loop "
    "(it calls tools, reads their output, and keeps going until the task is done).\n"
    "You get transcripts from scripted episodes. The tool output was fixed in advance, so judge only "
    "the model's turns: does it call the right tool with usable arguments, is each step grounded in the "
    "output it just got, does it change approach after an error instead of repeating it, does it stop by "
    "calling task_done rather than claiming success in prose, and does it ever claim something the output "
    "does not show?\n"
    'Reply with JSON only: {"verdict": "READY" or "NOT_READY", "scores": {"tool_use": 1-5, '
    '"grounding": 1-5, "recovery": 1-5, "stopping": 1-5, "honesty": 1-5}, '
    '"blockers": ["what must improve first", ...], "note": "one sentence"}')


def judge(results: list[dict], args) -> dict | None:
    from openai import OpenAI

    key = args.judge_api_key or f"sk-{args.judge_backend}-{args.judge}"
    client = OpenAI(base_url=args.judge_base_url, api_key=key, timeout=300)
    blob = []
    for r in results:
        blob.append(f"### episode {r['id']} (mechanical score {r['score']:.2f})")
        for t in r["transcript"]:
            if "tool" in t:
                blob.append(f"TOOL OUTPUT: {t['tool'][:600]}")
            else:
                blob.append(f"MODEL: {t['assistant'][:1200]}")
        blob.append("")
    out = None
    for delay in (0, 5, 15, 40):  # groq 429s under any sustained load; a single try drops the verdict
        if delay:
            time.sleep(delay)
        try:
            r = client.chat.completions.create(
                model=args.judge, temperature=0, max_tokens=900,
                **({"reasoning_effort": args.judge_reasoning} if args.judge_reasoning else {}),
                messages=[{"role": "system", "content": JUDGE},
                          {"role": "user", "content": "\n".join(blob)[:60000]}])
            msg = r.choices[0].message
            # gpt-oss puts its answer in `reasoning` and leaves content empty when it thinks first
            out = msg.content or (msg.model_extra or {}).get("reasoning") or ""
            break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:160]}"
    if out is None:
        print("judge unavailable:", last)
        return None
    m = re.search(r"\{.*\}", out, re.S)
    try:
        got = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        got = None
    if not got:
        print("judge gave nothing usable:", (out or "<empty response>")[:300])
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--adapter", help="LoRA adapter to probe (none = the base model)")
    ap.add_argument("--protocol", default="native", choices=("native", "json"),
                    help="how the harness asks for tool calls; .env TOOL_PROTOCOL decides which one is live")
    ap.add_argument("--scaffold", default="none", choices=sorted(SCAFFOLDS), help="reasoning method in the system prompt")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, the harness default")
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--show", action="store_true", help="print every transcript")
    ap.add_argument("--out", default=str(REPO / "out" / "probe.json"))
    ap.add_argument("--judge", help="teacher model that reads the transcripts and gives a verdict")
    ap.add_argument("--judge-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-backend", default="groq")
    ap.add_argument("--judge-api-key", default="")
    ap.add_argument("--judge-reasoning", default="low", help="reasoning_effort (groq: low/medium/high)")
    args = ap.parse_args()

    system, tools = harness_prompt(args.protocol)
    system += SCAFFOLDS[args.scaffold][0]
    print(f"protocol={args.protocol}  system prompt {len(system)} chars (~{len(system) // 4} tokens)  "
          f"{len(tools)} tools")
    model, tok = load(args)
    live = {t["function"]["name"] for t in tools}
    todo, skipped = [], []
    for sc in SCENARIOS:
        (todo if set(sc.get("needs", ())) <= live else skipped).append(sc)
    if skipped:
        print("skipped (tools not in this harness config):", ", ".join(sc["id"] for sc in skipped))
    results = [run_scenario(model, tok, sc, tools, system, args) for sc in todo]

    print()
    for r in results:
        print(f"{r['id']:9s} score {r['score']:.2f}  " + ("ok" if not r["problems"] else f"{len(r['problems'])} problems"))
        for p in r["problems"]:
            print("   -", p)
        if args.show:
            for t in r["transcript"]:
                print("   TOOL:" if "tool" in t else "   MODEL:", (t.get("tool") or t.get("assistant"))[:700].replace("\n", "\n     "))
    mean = sum(r["score"] for r in results) / len(results)
    print(f"\nmechanical score {mean:.2f} over {len(results)} episodes")

    verdict = judge(results, args) if args.judge else None
    if verdict:
        print("judge:", json.dumps(verdict, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"adapter": args.adapter, "scaffold": args.scaffold,
                                          "protocol": args.protocol, "mechanical": mean,
                                          "judge": verdict, "episodes": results},
                                         ensure_ascii=False, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
