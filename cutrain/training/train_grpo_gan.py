"""GRPO on the unified dataset with a self-critique (GAN-style) loop.

Generator = the policy, trained with GRPO over its own samples. Discriminator = the same family of
model prompted as a critic: it reads each sample's own reasoning and names the flaws in it. Reward
per sample is the weighted sum of

  test    graded match against the recorded chosen action (the verifiable part: right tool, right args)
  format  the chat template's own rules (<think> closed, <tool_call> balanced, nothing after the call)
  critic  1.0 when the critic finds no flaw, dropping with every flaw it names

The critic runs once per round, not once per completion: every sample is kept for free by the test
reward, and at the end of the round --critic-rows of them (the worst sample of each row) are sent to
the critic. A round over 800 prompts at 4 samples each would otherwise be 3200 calls, which no cloud
tier sustains - and a rate-limited call scores a neutral 1.0, so the run looks healthy and teaches
nothing. --critic-reward puts it back in the reward if the budget is there.

Every criticised sample goes back into the training set, two ways:
  - next round that prompt appears twice: bare (fix it unaided) and with its own critique pasted in
  - data/gan_pairs.jsonl gets a preference row (the critic-approved sample, else the recorded answer,
    vs the flawed one) in prepare_data.py's schema, so train_orpo.py can train on it unchanged
After every round the held-out split is generated greedily and scored (exact single-step action
match, --pass-threshold), and probe_agent's multi-turn episodes run as the autonomy gate: its
per-turn pass rate, and p**turns, are the only numbers that say anything about a run left alone.
Read that number honestly - 0.95 per turn is an 8% chance of surviving 50 turns.

`--critic-model self` makes the critic the policy itself: same weights, same adapter, no server and
no second model in VRAM, and it sharpens as the policy does. The default is a teacher on
`--critic-base-url`, an OpenAI-compatible server instead (Ollama, vLLM - see gen_rejected.py), which
only makes sense off this GPU: the defaults below already peak at 6.4 GB of the 8 GB card (measured:
QLoRA 4-bit, 1536-token prompts, 4 samples per prompt, embeddings parked in CPU RAM). On OOM, drop
--num-generations to 2 or --max-completion-length to 192 first.

  python -m cutrain.training.train_grpo_gan --adapter out/nemotron-nano-4b-orpo/adapter --rounds 2
  python -m cutrain.training.train_grpo_gan --no-critic          # plain verifiable-reward GRPO
  python -m cutrain.training.train_orpo --data data/gan_pairs.jsonl    # ORPO on the harvest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as train_orpo.py
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, cache_utils  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

# GRPO generates, and generation needs a cache. Nemotron-H's layer_types include "mlp" (a layer with no
# cache state at all), which transformers' dynamic-cache dispatch has no entry for -> KeyError: 'mlp'.
# "moe" is the same kind of stateless entry and maps to LinearAttentionLayer, so follow it.
cache_utils.DYNAMIC_LAYER_TYPE_MAPPING.setdefault("mlp", cache_utils.LinearAttentionLayer)

from cutrain.paths import HARNESS_ROOT, REPO
from cutrain.training.train_orpo import ASSISTANT, fit, merge  # prompt rendering + history trimming + bf16 merge

NEMOTRON_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj"]

# --- the model's own output format (see the chat template: <think>, <tool_call><function=...>) -----

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>[ \t]*\n?(.*?)\n?[ \t]*</parameter>", re.S)


def _value(raw: str):
    """Parameter bodies are plain text; JSON-decodable ones carry the type the schema asked for."""
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse(text: str) -> dict:
    """One completion -> unified assistant message (prepare_data.py's schema)."""
    think = THINK_RE.search(text)
    body = THINK_RE.sub("", text)
    calls = [{"type": "function", "function": {"name": name, "arguments": {k: _value(v) for k, v in PARAM_RE.findall(inner)}}}
             for name, inner in FUNC_RE.findall(body)]
    msg = {"role": "assistant", "content": CALL_RE.sub("", body).replace("<|im_end|>", "").strip()}
    if think and think.group(1).strip():
        msg["reasoning_content"] = think.group(1).strip()
    if calls:
        msg["tool_calls"] = calls
    return msg


def _same(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return " ".join(a.split()) == " ".join(b.split())
    return a == b or json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_score(got: dict, ref: dict, answer: str | None = None, text: str = "") -> float:
    """Graded action match: 0 wrong tool, 0.5 right tool, up to 1.0 as the arguments line up.

    Dense on purpose - a 0/1 action match leaves most groups with identical rewards, and GRPO
    learns nothing from a group whose advantages are all zero. Rows that carry a ground-truth
    answer are scored on that instead, which is the one fully objective signal in the mix.
    """
    if answer:
        return answer_score(text or got.get("content", ""), answer)
    g, r = got.get("tool_calls") or [], ref.get("tool_calls") or []
    if not r:
        return 1.0 if not g else 0.0  # the recorded step stopped and answered; calling a tool is the error
    if not g or g[0]["function"]["name"] != r[0]["function"]["name"]:
        return 0.0
    ra = r[0]["function"]["arguments"]
    ga = g[0]["function"]["arguments"]
    if not ra:
        return 1.0 if not ga else 0.75
    hit = sum(1 for k, v in ra.items() if k in ga and _same(ga[k], v))
    return 0.5 + 0.5 * hit / len(ra)


_EMPTY = re.match(r"(x?)", "")  # stand-in match object: .group(1) is "" when there is no <think>
ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]\s*\$?([^\n$]{1,80})", re.I)
BOXED_RE = re.compile(r"\\boxed\{([^{}]{1,80})\}")


def _norm_answer(a: str) -> str:
    a = a.strip().strip("$ \t.").replace(",", "").replace(" ", "").replace("\\!", "")
    return a.rstrip(".").lstrip("0") or "0"


def answer_score(text: str, answer: str) -> float:
    """Verifiable rows (DAPO math): the ground truth is a number, so the reward needs no judge."""
    found = BOXED_RE.findall(text) or ANSWER_RE.findall(text)
    if not found:
        return 0.0
    return 1.0 if _norm_answer(found[-1]) == _norm_answer(answer) else 0.0


PASS = 0.95  # test_score of an exact action match, modulo whitespace

ABS_PATH_RE = re.compile(r"(?<![\w.])/(?:[\w.@+-]+/){1,}[\w.@+-]*")
# the sandbox the harness actually gives it; anything else has to come from the conversation
SANDBOX = ("/workspace", "/tmp", "/usr", "/etc", "/bin", "/sbin", "/lib", "/opt", "/var", "/proc",
           "/sys", "/dev", "/root", "/home/user")


def grounding_score(msg: dict, prompt: str) -> float:
    """Punish absolute paths the conversation never mentioned.

    The recorded sessions were made on a real machine, so the model picked up its project roots and
    will answer "/workspace/app" tasks with the path it memorised. Only paths it was shown, or paths
    inside the container's own tree, count as grounded.
    """
    paths = {p for c in msg.get("tool_calls") or []
             for p in ABS_PATH_RE.findall(json.dumps(c["function"]["arguments"], ensure_ascii=False))}
    invented = [p for p in paths if not p.startswith(SANDBOX) and p not in prompt]
    return max(0.0, 1.0 - 0.5 * len(invented))


def format_score(text: str, truncated: bool) -> float:
    """The template's rules, the ones a parser downstream actually breaks on."""
    bad = 0
    if text.count("<think>") > 1 or text.count("<think>") != text.count("</think>"):
        bad += 1
    if text.count("<tool_call>") != text.count("</tool_call>"):
        bad += 1
    if text.count("<function=") != text.count("</function>"):
        bad += 1
    if "</tool_call>" in text and text.rsplit("</tool_call>", 1)[-1].strip():
        bad += 1  # "You may provide reasoning BEFORE the function call, but NOT after"
    if truncated:
        bad += 1
    return max(0.0, 1.0 - 0.34 * bad)


# --- structured reasoning scaffolds ---------------------------------------------------------------

# Each one is appended to the row's system prompt and asks for its own headings inside <think>.
# SECTIONS is what scaffold_score looks for, so a scaffold is measurable, not just decoration.
SCAFFOLDS = {
    "none": ("", (), ""),
    "polya": (  # Polya, How to Solve It (1945): understand, plan, carry out, look back
        "\n\n# Reasoning method\nInside <think>, work through Polya's four steps, each on its own line:\n"
        "UNDERSTAND: what is given, what is wanted, what is still unknown.\n"
        "PLAN: the single next action and why it gets you closer.\n"
        "CARRY OUT: what you expect that action to produce.\n"
        "LOOK BACK: how you will know from the result whether it worked.\n"
        "Then act.", ("UNDERSTAND:", "PLAN:", "CARRY OUT:", "LOOK BACK:"),
        "UNDERSTAND: the user wants the failing test named; I have no output yet.\n"
        "PLAN: run the suite, because the error text names the test.\n"
        "CARRY OUT: expect a pytest summary with one FAILED line.\n"
        "LOOK BACK: if nothing fails, the suite is not the one they meant."),
    "react": (  # Yao et al., ReAct: reason, act, observe
        "\n\n# Reasoning method\nInside <think>, write:\nTHOUGHT: what the last observation tells you.\n"
        "ACT: the one tool call you will make now.\nEXPECT: the observation that would confirm it.\n"
        "Then act.", ("THOUGHT:", "ACT:", "EXPECT:"),
        "THOUGHT: the listing shows app_v2, not app, so my first path was wrong.\n"
        "ACT: run the suite inside app_v2.\n"
        "EXPECT: a pytest summary naming the failing test."),
    "plan-solve": (  # Wang et al., Plan-and-Solve: plan the whole task first, then execute a step
        "\n\n# Reasoning method\nInside <think>, write:\nGOAL: the task in one sentence.\n"
        "STEPS: the numbered plan to finish it.\nNOW: which step this action does.\nThen act.",
        ("GOAL:", "STEPS:", "NOW:"),
        "GOAL: find why the suite fails, without fixing it.\n"
        "STEPS: 1 locate the project 2 run the suite 3 read the first error\n"
        "NOW: step 2."),
    "cove": (  # Dhuliawala et al., Chain-of-Verification: draft, then interrogate the draft
        "\n\n# Reasoning method\nInside <think>, write:\nDRAFT: the action you first want to take.\n"
        "CHECKS: two questions that would expose it as wrong.\nVERDICT: keep the draft or replace it, and why.\n"
        "Then act.", ("DRAFT:", "CHECKS:", "VERDICT:"),
        "DRAFT: rerun make all.\n"
        "CHECKS: did that target ever exist? does the listing show another entry point?\n"
        "VERDICT: replace it - the listing shows build.sh and no all target."),
}


def scaffold_block(name: str, example: bool) -> str:
    """The scaffold's instructions, plus a worked turn when asked.

    Without the example a 4B model reads the headings and ignores them: the adherence reward then
    scores every sample 0, and a reward with no spread inside a group is no signal at all in GRPO.
    """
    block, _, shot = SCAFFOLDS[name]
    if not block:
        return ""
    return block + (f"\n\nExample of the <think> block:\n{shot}" if example and shot else "")


BARE_THINKING = 0.3  # credit for reasoning at all, before any structure


def scaffold_score(text: str, sections: tuple) -> float:
    """Graded: 0 for an empty <think>, BARE_THINKING for reasoning, up to 1.0 for the asked-for shape.

    The ORPO-tuned policy emits an empty `<think></think>` in about three completions in four (it was
    trained on rows whose reasoning had been redacted), and the harness runs with THINKING=true. A
    pass/fail adherence score is flat 0 on that model, and a reward with no spread inside a group is
    no gradient in GRPO - so reasoning at all has to be worth something before structure is.
    """
    think = THINK_RE.search(text)
    body = (think.group(1) if think else "").strip()
    if not body:
        return 0.0
    if not sections:
        return 1.0
    # count the headings anywhere in the turn: asked for them inside <think>, the model puts them in
    # the body instead (4 of 19 turns under the json protocol), and that is still the scaffold being
    # followed - whether it breaks the output format is the format reward's business, not this one
    return BARE_THINKING + (1 - BARE_THINKING) * sum(s in text for s in sections) / len(sections)


# --- the discriminator ----------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You review one step of an AI computer-use agent. You get the task context, the agent's private "
    "reasoning and the action it chose. Name only real flaws: reasoning that does not follow from the "
    "context, a wrong or unnecessary tool, a wrong target (file, path, coordinate), an assumption not "
    "supported by the context, missing verification, or claiming success without evidence. Ignore style, "
    "wording and length.\n"
    'Reply with JSON only: {"verdict": "SOUND" or "FLAWED", "flaws": ["short flaw", ...]}\n'
    "Use an empty flaws list when the step is sound. At most 3 flaws, at most 15 words each.")


def summarise(msg: dict) -> str:
    calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])[:300]})"
                      for c in msg.get("tool_calls", []))
    return "\n".join(filter(None, [msg.get("content", ""), calls and f"[called {calls}]"])) or "(empty answer)"


class Critic:
    """Self-critique: prompt, flaw parsing, cache, and the memory that sharpens the next round."""

    def __init__(self, memory: Path, keep: int = 5):
        self.keep = keep
        self.memory_path = memory
        self.memory: list[str] = json.loads(memory.read_text()) if memory.exists() else []
        self.cache: dict[str, tuple[float, list[str]]] = {}
        self.seen: Counter = Counter()
        self.error_kinds: Counter = Counter()
        self.errors = 0

    def ask(self, system: str, user: str) -> str:
        raise NotImplementedError

    def system(self) -> str:
        if not self.memory:
            return CRITIC_SYSTEM
        return CRITIC_SYSTEM + "\n\nFlaws this agent repeats, check for them first:\n" + "\n".join(f"- {f}" for f in self.memory)

    def one(self, context: str, msg: dict) -> tuple[float, list[str]]:
        key = hashlib.sha1((context[-2000:] + json.dumps(msg, sort_keys=True)).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        user = (f"# Task context (tail)\n{context[-2000:]}\n\n# Agent reasoning\n"
                f"{msg.get('reasoning_content') or '(none given)'}\n\n# Agent action\n{summarise(msg)}")
        out = None
        for delay in (0, 4, 12, 30):  # a rate-limit window outlasts a single retry; back off instead
            if delay:
                time.sleep(delay)
            try:
                out = self.ask(self.system(), user)
                break
            except Exception as e:  # noqa: BLE001 - a dead critic must not poison the reward signal
                last = f"{type(e).__name__}: {str(e)[:90]}"
        if out is None:
            self.errors += 1
            self.error_kinds[last] += 1
            return 1.0, []
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:  # truncated JSON (a long flaw hit the token cap): keep the verdict only
            d = {"verdict": "FLAWED" if "FLAW" in out.upper() else "SOUND"}
        flaws = [" ".join(str(f).split())[:200] for f in (d.get("flaws") or []) if str(f).strip()][:3]
        verdict = str(d.get("verdict", "")).upper()
        if verdict.startswith("SOUND"):
            flaws = []
        elif verdict.startswith("FLAW") and not flaws:
            flaws = [" ".join(out.split())[:200] or "the critic named no flaw"]
        for f in flaws:
            self.seen[" ".join(f.lower().split())[:60]] += 1
        res = (max(0.0, 1.0 - 0.34 * len(flaws)), flaws)
        self.cache[key] = res
        return res

    def batch(self, contexts: list[str], msgs: list[dict]) -> list[tuple[float, list[str]]]:
        return [self.one(c, m) for c, m in zip(contexts, msgs)]

    def report_errors(self) -> str:
        """What went wrong, named. A silent critic scores every sample 1.0 and looks like success."""
        if not self.errors:
            return ""
        top = ", ".join(f"{n}x {k}" for k, n in self.error_kinds.most_common(3))
        return f"{self.errors} critic calls failed after a retry and scored a neutral 1.0 -> {top}"

    def remember(self) -> None:
        """Keep the flaws seen most this round; next round the critic looks for them first."""
        self.memory = [f for f, _ in self.seen.most_common(self.keep)]
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.memory, indent=1))
        self.seen.clear()


class SelfCritic(Critic):
    """The policy critiques itself - same weights, same adapter, so both sides of the loop co-evolve.

    Default because a server critic wants VRAM this training run is already holding: on one 8 GB card
    the policy peaks near 6.4 GB and an Ollama critic next to it does not fit.
    """

    def __init__(self, model, tok, memory: Path, max_new_tokens: int = 200, keep: int = 5):
        super().__init__(memory, keep)
        self.model, self.tok, self.max_new_tokens = model, tok, max_new_tokens

    @torch.no_grad()
    def ask(self, system: str, user: str) -> str:
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        try:
            p = self.tok.apply_chat_template([{"role": "system", "content": system},
                                              {"role": "user", "content": user}], tokenize=False)
            p += ASSISTANT + "<think></think>"  # the verdict, not a second chain of thought
            enc = self.tok(p, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            self.model.config.use_cache = False
            if was_training:
                self.model.train()


class ServerCritic(Critic):
    """Critic on an OpenAI-compatible server (Ollama, vLLM): another model, or the same one off-GPU."""

    def __init__(self, base_url: str, api_key: str, model: str, workers: int, memory: Path,
                 reasoning: str = "low", keep: int = 5):
        from openai import OpenAI  # here so the self-critic runs without the package

        super().__init__(memory, keep)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.model, self.workers, self.reasoning = model, workers, reasoning

    def ask(self, system: str, user: str) -> str:
        # reasoning_effort keeps a thinking critic cheap; groq only accepts low/medium/high, Ollama
        # wants "none", and some servers reject the field outright -> "" leaves it out
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=400,
            **({"reasoning_effort": self.reasoning} if self.reasoning else {}),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        msg = r.choices[0].message
        # gpt-oss answers in `reasoning` with an empty content field when it thinks first
        return msg.content or (msg.model_extra or {}).get("reasoning") or ""

    def batch(self, contexts: list[str], msgs: list[dict]) -> list[tuple[float, list[str]]]:
        # the local proxy stalls past a handful of concurrent requests (8 workers cost 1012 timeouts
        # in one round, every one of them scoring a neutral 1.0), so keep the fan-out small
        with ThreadPoolExecutor(min(self.workers, 3)) as ex:
            return list(ex.map(self.one, contexts, msgs))


# --- rows, rounds, harvest ------------------------------------------------------------------------

NOTE_HEAD = "<|im_start|>user\n<self_critique>\nA previous attempt at this step was criticised:\n"
NOTE_TAIL = "\nDo not repeat these.\n</self_critique><|im_end|>\n"


def inject(prompt: str, flaws: list[str]) -> str:
    """Paste a sample's own critique in just before the assistant turn it has to redo."""
    assert prompt.endswith(ASSISTANT)
    return prompt[: -len(ASSISTANT)] + NOTE_HEAD + "\n".join(f"- {f}" for f in flaws) + NOTE_TAIL + ASSISTANT


def load_rows(paths: list[str], tok, sources: list[str], budget: int, limit: int, rng: random.Random,
              scaffold: str = "none", example: bool = False) -> list[dict]:
    """Unified rows -> {row_id, prompt (rendered text), ref, answer, tools, category, msgs} for GRPO.

    Shuffled before rendering: with --limit set, rendering all 12k rows would cost minutes for rows
    that get thrown away (fit() re-renders once per dropped history turn). The scaffold rides on the
    row's own system message, so it reaches the model through the chat template like any instruction.
    """
    raw = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw += [json.loads(line) for line in f]
    raw = [r for r in raw if not sources or r["source"].startswith(tuple(sources))]
    rng.shuffle(raw)
    block = scaffold_block(scaffold, example)
    rows, dropped = [], 0
    for r in raw:
        msgs = r["prompt"]
        if block:
            msgs = ([{**msgs[0], "content": msgs[0]["content"] + block}] + msgs[1:]
                    if msgs and msgs[0]["role"] == "system"
                    else [{"role": "system", "content": block.strip()}] + msgs)  # dapo rows open on the user turn
        got = fit({**r, "prompt": msgs, "rejected": r["chosen"]}, tok, budget)  # trims history to fit
        if got is None:
            dropped += 1
            continue
        rows.append({"row_id": r["id"], "prompt": got["prompt"], "ref": r["chosen"][0],
                     "answer": r.get("answer"), "tools": r["tools"], "category": r["category"],
                     "msgs": msgs})
        if limit and len(rows) >= limit:
            break
    print(f"rows: {len(rows)} usable, {dropped} dropped (too long or template mismatch)"
          + (f", scaffold={scaffold}" if block else ""))
    return rows


def round_dataset(rows: list[dict], flaws: dict[str, list[str]], inject_critique: bool) -> Dataset:
    """One entry per row, plus a second hinted entry for every row the critic faulted last round.

    Only `prompt` and `row_id` go into Arrow: tool schemas differ per row and Arrow would merge the
    argument properties of unrelated rows together (the trap prepare_data.py's docstring warns about).
    """
    out = []
    for r in rows:
        out.append({"prompt": r["prompt"], "row_id": r["row_id"]})
        if inject_critique and flaws.get(r["row_id"]):
            out.append({"prompt": inject(r["prompt"], flaws[r["row_id"]]), "row_id": r["row_id"]})
    return Dataset.from_list(out)


class Harvest:
    """Every sample of this round, keyed by row, so the end of the round can build the pairs."""

    def __init__(self) -> None:
        self.by_row: dict[str, list[tuple[float, dict, list[str]]]] = {}

    def add(self, row_id: str, score: float, msg: dict, flaws: list[str]) -> None:
        self.by_row.setdefault(row_id, []).append((score, msg, flaws))

    def critique(self, critic, rows: dict[str, dict], n_rows: int, rng) -> int:
        """Critique the worst sample of up to n_rows rows. Returns the number of calls made.

        One pass at the end of the round instead of one call per completion inside the reward: a GRPO
        round over 800 prompts at 4 samples each is 3200 critic calls, which no cloud tier sustains -
        and a rate-limited critic returns a neutral 1.0, so the run looks healthy while learning
        nothing. The worst sample is the one most likely to carry a real flaw, and one flawed sample is
        all pairs() needs to build the preference row and next round's hint.
        """
        ids = [r for r in self.by_row if r in rows]
        rng.shuffle(ids)
        for row_id in ids[:n_rows]:
            samples = self.by_row[row_id]
            i = min(range(len(samples)), key=lambda j: samples[j][0])
            flaws = critic.one(rows[row_id]["prompt"], samples[i][1])[1]
            if flaws:
                samples[i] = (samples[i][0], samples[i][1], flaws)
        return len(ids[:n_rows])

    def pairs(self, rows: dict[str, dict], rnd: int) -> tuple[list[dict], dict[str, list[str]]]:
        """Flawed samples -> preference rows + the flaws to paste into next round's prompts."""
        out, flaws = [], {}
        for row_id, samples in self.by_row.items():
            faulted = [s for s in samples if s[2]]
            if not faulted or row_id not in rows:
                continue
            worst = min(faulted, key=lambda s: s[0])
            clean = [s for s in samples if not s[2] and s[0] >= PASS]
            chosen = max(clean, key=lambda s: s[0])[1] if clean else rows[row_id]["ref"]
            if chosen == worst[1]:
                continue
            flaws[row_id] = worst[2]
            out.append({"id": f"gan{rnd}:{row_id}", "source": f"gan:self_critique:round{rnd}",
                        "category": rows[row_id]["category"], "tools": rows[row_id]["tools"],
                        "prompt": rows[row_id]["msgs"], "chosen": [chosen], "rejected": [worst[1]],
                        "rejected_origin": "gan:self_critique",
                        "flaws": worst[2], "on_policy_chosen": bool(clean)})
        return out, flaws


# --- model, eval ----------------------------------------------------------------------------------

def _park(emb, inputs):
    if emb.weight.device.type != "cpu":
        emb.to("cpu")
    return (inputs[0].to("cpu"),)


def load_model(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token  # Nemotron ships neither pad nor a chat-template bos
    tok.bos_token = None
    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "out_proj"])  # mamba's fused kernel reads raw bf16 out_proj
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0},
        use_kernels=args.kernels)
    model.train()
    if args.kernels and hasattr(model, "set_use_kernels"):
        # Select CUDA training kernels before potentially moving the first parameter to CPU.
        model.set_use_kernels(True)
    if args.offload:  # frozen input embeddings (0.77 GB, a lookup) stay in CPU RAM, as train_orpo.py does
        emb = model.get_input_embeddings()
        emb.register_forward_pre_hook(_park)  # re-parks itself: Trainer.train() moves the model to GPU again
        emb.register_forward_hook(lambda mod, a, out: out.to("cuda"))
        emb.to("cpu")
    model.config.use_cache = False
    # the repo ships generation_config.eos_token_id = 2 while the chat template closes turns with
    # <|im_end|> (id 11); rollouts that never hit eos get masked out and teach nothing
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.adapter:  # continue the ORPO LoRA rather than stacking a second adapter on top of it
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print("continuing from", args.adapter)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=args.lora_targets))
    model.print_trainable_parameters()
    return model, tok


@torch.no_grad()
def run_test(model, tok, rows: list[dict], max_new_tokens: int, critic: Critic | None,
             sections: tuple = (), dump: str | None = None) -> dict:
    """The test: greedy answers on the held-out rows, scored against the recorded action or answer."""
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    exact = tool_ok = sound = fmt = struct = ground = thinks = 0.0
    out_file = open(dump, "a", encoding="utf-8") if dump else None
    try:
        for r in rows:
            enc = tok(r["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            text = tok.decode(out[0, n_in:], skip_special_tokens=True)
            msg = parse(text)
            score = test_score(msg, r["ref"], r.get("answer"), text)
            exact += score >= PASS
            tool_ok += score > 0
            fmt += format_score(text, out.shape[1] - n_in >= max_new_tokens)
            struct += scaffold_score(text, sections)
            thinks += bool((THINK_RE.search(text) or _EMPTY).group(1).strip())
            ground += grounding_score(msg, r["prompt"])
            if critic is not None:
                sound += not critic.one(r["prompt"], msg)[1]
            if out_file:
                out_file.write(json.dumps({"row_id": r["row_id"], "category": r["category"],
                                           "score": score, "completion": text}, ensure_ascii=False) + "\n")
    finally:
        if out_file:
            out_file.close()
        model.config.use_cache = False
        if was_training:
            model.train()
    n = max(len(rows), 1)
    return {"n": len(rows), "exact": exact / n, "tool": tool_ok / n, "format": fmt / n,
            "structure": struct / n, "grounded": ground / n, "thinks": thinks / n,
            "critic_sound": sound / n if critic is not None else float("nan")}


def _live_protocol() -> str:
    """What the harness is actually configured to send (.env TOOL_PROTOCOL), not a default guess."""
    import sys

    sys.path.insert(0, str(HARNESS_ROOT))
    from src.config import load_config

    return load_config().tool_protocol


def wblog(wb, payload: dict, commit: bool = True) -> None:
    """Log to whatever run the trainer opened. Never let the dashboard take the run down with it.

    Each round builds a fresh GRPOTrainer, which opens (and on exit finishes) its own wandb run, so
    a log landing between rounds can hit a closed run. A lost chart is cheaper than a lost round.
    """
    if wb is None:
        return
    try:
        wb.log(payload, commit=commit)
    except Exception as e:  # noqa: BLE001
        print(f"wandb log skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)


def run_probe(model, tok, args) -> dict:
    """probe_agent's multi-turn episodes, scored per turn: the gate that speaks to autonomy.

    run_test measures one recorded step. A run left alone overnight is hundreds of steps, and it
    survives only if every one of them holds - so the number that matters is the per-turn pass rate,
    and what it implies is p**turns. probe_agent already checks what a long chain breaks on (a tool
    that exists, arguments grounded in the output just received, a change of approach after an error,
    stopping instead of claiming success), so reuse it rather than growing a second harness.
    """
    from cutrain.evaluation import probe_agent  # here, not at module scope: probe_agent imports from this module

    protocol = args.probe_protocol or os.environ.get("TOOL_PROTOCOL") or _live_protocol()
    system, tools = probe_agent.harness_prompt(protocol)
    system += SCAFFOLDS[args.scaffold][0]
    live = {t["function"]["name"] for t in tools}
    todo = [sc for sc in probe_agent.SCENARIOS if set(sc.get("needs", ())) <= live]
    pa = argparse.Namespace(protocol=protocol, max_new_tokens=args.probe_tokens, temperature=0.0)
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    try:
        eps = [probe_agent.run_scenario(model, tok, sc, tools, system, pa) for sc in todo]
    finally:
        model.config.use_cache = False
        if was_training:
            model.train()
    turns = sum(e["turns"] for e in eps)
    bad = sum(len(e["problems"]) for e in eps)
    kinds = Counter(re.sub(r"[:(].*", "", p.split("] ", 1)[-1]).strip() for e in eps for p in e["problems"])
    return {"protocol": protocol, "per_turn": max(0.0, 1 - bad / max(turns, 1)), "turns": turns, "episodes": len(eps),
            "clean": sum(not e["problems"] for e in eps), "problems": dict(kinds),
            "by_episode": {e["id"]: round(e["score"], 3) for e in eps}}


def describe_probe(p: dict, horizons: list[int]) -> str:
    """Per-turn rate, and the only reading of it that answers "can I leave it running?"."""
    odds = "  ".join(f"{n}-turn run {p['per_turn'] ** n:7.2%}" for n in horizons)
    return (f"per-turn {p['per_turn']:.3f} over {p['turns']} turns, {p['clean']}/{p['episodes']} "
            f"episodes clean -> {odds}")


# --- main -----------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--adapter", help="LoRA adapter to continue from (e.g. out/nemotron-nano-4b-orpo/adapter)")
    ap.add_argument("--data", nargs="+", default=[str(REPO / "data" / "unified.jsonl"),
                                                  str(REPO / "data" / "hf_agent.jsonl")],
                    help="unified jsonl files (prepare_data.py and prepare_hf.py write these)")
    ap.add_argument("--out", default=str(REPO / "out" / "nemotron-nano-4b-grpo-gan"))
    ap.add_argument("--pairs", default=str(REPO / "data" / "gan_pairs.jsonl"),
                    help="self-critique preference rows, appended; prepare_data.py schema (feeds train_orpo.py)")
    ap.add_argument("--log", default=str(REPO / "data" / "critiques.jsonl"), help="every flaw the critic named")
    ap.add_argument("--sources", nargs="*", default=["opencode", "claude", "hf"],
                    help="source prefixes to keep, matched from the left: 'hf' takes every prepare_hf.py "
                         "row, 'hf:dapo' only the verifiable maths ones")
    ap.add_argument("--scaffold", default="none", choices=sorted(SCAFFOLDS),
                    help="structured-reasoning method pasted into the system prompt "
                         "(polya, react, plan-solve, cove); scored by the structure reward")
    ap.add_argument("--scaffold-example", action=argparse.BooleanOptionalAction, default=True,
                    help="include a worked <think> block for the scaffold (without it the model ignores "
                         "the headings and the structure reward is flat)")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: score the held-out rows under every --scaffold-ab value and stop")
    ap.add_argument("--scaffold-ab", nargs="+", default=sorted(SCAFFOLDS), help="scaffolds to compare in --eval-only")
    ap.add_argument("--dump", help="jsonl of every test completion (read what the model actually said)")
    ap.add_argument("--limit", type=int, default=800, help="rows kept after shuffling (0 = all)")
    ap.add_argument("--test-rows", type=int, default=64, help="held-out rows scored after every round")
    ap.add_argument("--pass-threshold", type=float, default=0.5, help="exact-match rate the test must reach")
    ap.add_argument("--rounds", type=int, default=2, help="GRPO pass + critique harvest per round")
    ap.add_argument("--max-prompt-length", type=int, default=1536)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--num-generations", type=int, default=4, help="group size; the GRPO baseline is its mean")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="per round; stop early (smoke runs)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL to the reference policy; 0 keeps no ref model in VRAM")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--weights", type=float, nargs=5, default=[1.0, 0.3, 0.7, 0.3, 0.5],
                    metavar=("TEST", "FORMAT", "CRITIC", "STRUCTURE", "GROUNDING"))
    ap.add_argument("--critic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--critic-rows", type=int, default=64,
                    help="rows critiqued at the end of each round (one call each). The whole API bill of "
                         "a round, bar the held-out test; raise it to harvest more preference pairs")
    ap.add_argument("--critic-reward", action=argparse.BooleanOptionalAction, default=False,
                    help="also score every completion with the critic as a GRPO reward. Off: a round is "
                         "~3200 calls, which no cloud tier sustains, and a 429 scores a neutral 1.0")
    ap.add_argument("--critic-base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible endpoint for the teacher critic; keep training's own GPU out of it")
    ap.add_argument("--critic-api-key", default="", help="default: sk-<critic-backend>-<critic-model>")
    ap.add_argument("--critic-backend", default="gemini",
                    help="prefix the local proxy wants in the key; groq's free tier 429s on a GRPO run "
                         "(596 of 640 calls in one round), gemini sustained 400 bulk calls with none")
    ap.add_argument("--critic-model", default="gemini-3.1-flash-lite",
                    help="teacher critic on --critic-base-url, or 'self' = the policy critiques its own "
                         "samples (no server, no extra VRAM, weaker)")
    ap.add_argument("--critic-workers", type=int, default=3, help="capped at 3; the proxy stalls above that")
    ap.add_argument("--critic-reasoning", default="low",
                    help="reasoning_effort for the critic: low/medium/high on groq, 'none' on Ollama, "
                         "empty to leave the field out")
    ap.add_argument("--inject-critique", action=argparse.BooleanOptionalAction, default=True,
                    help="also train the criticised prompt with its flaws pasted in (plus the bare one)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", nargs="+", default=NEMOTRON_TARGETS)
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the frozen input embeddings in CPU RAM (~0.8 GB of VRAM back)")
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vllm-server", help="base url of a trl vllm-serve process (much faster rollouts)")
    ap.add_argument("--log-completions", action="store_true", help="print a sample completion table every log step")
    ap.add_argument("--report-to", default="none", help="trainer logging backend: none, wandb, tensorboard")
    ap.add_argument("--wandb-project", default="nemotron-agent-grpo")
    ap.add_argument("--run-name", help="run name in the logging backend (default: the --out basename)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                    help="run probe_agent's multi-turn episodes after every round: the autonomy gate")
    ap.add_argument("--probe-protocol", choices=("native", "json"),
                    help="default: whatever .env's TOOL_PROTOCOL says. Gating on the other one would "
                         "score a prompt the harness never sends")
    ap.add_argument("--probe-tokens", type=int, default=400, help="max_new_tokens per probe turn")
    ap.add_argument("--probe-threshold", type=float, default=0.95,
                    help="per-turn pass rate the probe must reach. Note what it buys: a long run needs "
                         "p**turns, so 0.95 is ~8%% over 50 turns. Hands-off overnight wants 0.995+")
    ap.add_argument("--probe-horizon", type=int, nargs="+", default=[50, 200],
                    help="turn counts to report implied survival (p**n) for")
    ap.add_argument("--merge", action="store_true", help="also write merged bf16 model to <out>/merged")
    args = ap.parse_args()
    if "FP8" in args.model:
        ap.error("train on the BF16 repo; the FP8 repo is a modelopt export transformers can't load")
    if (args.batch * args.grad_accum) % args.num_generations:
        ap.error(f"batch*grad-accum ({args.batch * args.grad_accum}) must be a multiple of "
                 f"--num-generations ({args.num_generations}): one group cannot span generation batches")

    rng = random.Random(args.seed)
    model, tok = load_model(args)
    if args.eval_only:  # A/B the reasoning scaffolds on one model, no training in between
        results = {}
        for name in args.scaffold_ab:
            rows = load_rows(args.data, tok, args.sources, args.max_prompt_length + args.max_completion_length,
                             args.test_rows, random.Random(args.seed), name, args.scaffold_example)
            results[name] = run_test(model, tok, rows, args.max_completion_length, None,
                                     SCAFFOLDS[name][1], args.dump)
            r = results[name]
            print(f"{name:11s} exact {r['exact']:.3f}  right-tool {r['tool']:.3f}  format {r['format']:.3f}  "
                  f"reasoning {r['structure']:.3f}  thinks {r['thinks']:.3f}  "
                  f"grounded {r['grounded']:.3f}  (n={r['n']})", flush=True)
        best = max(results, key=lambda k: (results[k]["exact"], results[k]["tool"]))
        print(f"\nbest scaffold by exact, then right-tool: {best}")
        return
    rows = load_rows(args.data, tok, args.sources, args.max_prompt_length + args.max_completion_length,
                     args.limit, rng, args.scaffold, args.scaffold_example)
    if len(rows) <= args.test_rows:
        ap.error(f"only {len(rows)} rows usable; need more than --test-rows ({args.test_rows})")
    test_rows, train_rows = rows[: args.test_rows], rows[args.test_rows:]
    by_id = {r["row_id"]: r for r in train_rows}
    mem = Path(args.out) / "critic_memory.json"
    if not args.critic:
        critic = None
    elif args.critic_model == "self":
        critic = SelfCritic(model, tok, mem)
    else:
        key = args.critic_api_key or f"sk-{args.critic_backend}-{args.critic_model}"
        critic = ServerCritic(args.critic_base_url, key, args.critic_model, args.critic_workers, mem,
                              args.critic_reasoning)

    harvest = Harvest()
    wb = None
    if args.report_to == "wandb":
        import wandb as wb  # noqa: PLC0415 - optional dependency

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    sections = SCAFFOLDS[args.scaffold][1]

    def reward_test(prompts, completions, completion_ids, row_id, **kw):
        msgs = [parse(c) for c in completions]
        out = [test_score(m, by_id[rid]["ref"], by_id[rid].get("answer"), c)
               for m, rid, c in zip(msgs, row_id, completions)]
        for rid, m, s in zip(row_id, msgs, out):  # free: the round-end critique picks from these
            harvest.add(rid, s, m, [])
        kw["log_extra"]("action", [summarise(m)[:200] for m in msgs])
        kw["log_metric"]("test/exact", sum(s >= PASS for s in out) / len(out))
        return out

    def reward_format(prompts, completions, completion_ids, **kw):
        return [format_score(c, len(ids) >= args.max_completion_length)
                for c, ids in zip(completions, completion_ids)]

    def reward_structure(prompts, completions, completion_ids, **kw):
        return [scaffold_score(c, sections) for c in completions]

    def reward_grounding(prompts, completions, completion_ids, **kw):
        out = [grounding_score(parse(c), p) for c, p in zip(completions, prompts)]
        kw["log_metric"]("grounding/clean", sum(s >= 1.0 for s in out) / len(out))
        return out

    def reward_critic(prompts, completions, completion_ids, row_id, **kw):
        msgs = [parse(c) for c in completions]
        scored = critic.batch(list(prompts), msgs)  # harvesting is the round-end critique's job
        kw["log_extra"]("flaws", ["; ".join(f)[:200] for _, f in scored])
        kw["log_metric"]("critic/sound", sum(not f for _, f in scored) / len(scored))
        return [s for s, _ in scored]

    use_critic_reward = critic is not None and args.critic_reward
    funcs = [reward_test, reward_format] + ([reward_critic] if use_critic_reward else [])
    weights = list(args.weights[:2]) + ([args.weights[2]] if use_critic_reward else [])
    if args.weights[3]:  # reasoning: non-empty <think>, plus the scaffold's shape when one is set
        funcs.append(reward_structure)
        weights.append(args.weights[3])
    if args.weights[4]:
        funcs.append(reward_grounding)
        weights.append(args.weights[4])
    flaws: dict[str, list[str]] = {}
    history = []

    for rnd in range(args.rounds):
        train = round_dataset(train_rows, flaws, args.inject_critique)
        hinted = len(train) - len(train_rows)
        print(f"\n=== round {rnd}: {len(train)} prompts ({hinted} with a pasted critique) ===", flush=True)
        cfg = GRPOConfig(
            output_dir=f"{args.out}/round{rnd}", beta=args.beta, learning_rate=args.lr,
            num_generations=args.num_generations, max_completion_length=args.max_completion_length,
            temperature=args.temperature, per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
            max_steps=args.max_steps, reward_weights=weights, mask_truncated_completions=True,
            lr_scheduler_type="cosine", warmup_steps=5, bf16=True,
            optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
            logging_steps=1, save_strategy="no", report_to=args.report_to,
            run_name=f"{args.run_name or Path(args.out).name}-r{rnd}",
            log_completions=args.log_completions, num_completions_to_print=1, seed=args.seed + rnd,
            **({"use_vllm": True, "vllm_mode": "server", "vllm_server_base_url": args.vllm_server}
               if args.vllm_server else {}))
        harvest.by_row.clear()
        GRPOTrainer(model=model, reward_funcs=funcs, args=cfg, train_dataset=train, processing_class=tok).train()
        model.save_pretrained(f"{args.out}/round{rnd}/adapter")

        if critic is not None:
            asked = harvest.critique(critic, by_id, args.critic_rows, rng)
            print(f"critic: critiqued {asked} of {len(harvest.by_row)} rows", flush=True)
        pairs, flaws = harvest.pairs(by_id, rnd)
        if pairs:  # appended, so rounds and runs pile up into one preference set
            with open(args.pairs, "a", encoding="utf-8") as f, open(args.log, "a", encoding="utf-8") as g:
                for row in pairs:
                    g.write(json.dumps({"round": rnd, "id": row["id"], "flaws": row["flaws"],
                                        "on_policy_chosen": row["on_policy_chosen"],
                                        "rejected": row["rejected"][0]}, ensure_ascii=False) + "\n")
                    f.write(json.dumps({k: v for k, v in row.items() if k not in ("flaws", "on_policy_chosen")},
                                       ensure_ascii=False) + "\n")
        if critic is not None:
            print(f"critic: {len(pairs)} new preference rows -> {args.pairs}")
            if critic.errors:
                print("WARNING:", critic.report_errors(), flush=True)
            critic.remember()
            print("critic now watches for:", critic.memory or "(nothing yet)")

        res = run_test(model, tok, test_rows, args.max_completion_length, critic, sections, args.dump)
        res["round"] = rnd
        history.append(res)
        print(f"round {rnd} test: exact {res['exact']:.3f}  right-tool {res['tool']:.3f}  "
              f"format {res['format']:.3f}  reasoning {res['structure']:.3f}  "
              f"thinks {res['thinks']:.3f}  grounded {res['grounded']:.3f}  critic-sound {res['critic_sound']:.3f}  "
              f"(n={res['n']})", flush=True)
        wblog(wb, {f"heldout/{k}": v for k, v in res.items() if isinstance(v, (int, float))}, commit=False)
        if args.probe:
            res["probe"] = run_probe(model, tok, args)
            print(f"round {rnd} probe: {describe_probe(res['probe'], args.probe_horizon)}", flush=True)
            pr = res["probe"]
            wblog(wb, {"probe/per_turn": pr["per_turn"], "probe/episodes_clean": pr["clean"],
                       **{f"probe/survival_{h}turn": pr["per_turn"] ** h for h in args.probe_horizon},
                       **{f"probe/episode_{k}": v for k, v in pr["by_episode"].items()}})

    model.save_pretrained(f"{args.out}/adapter")
    best = max(history, key=lambda h: h["exact"])
    print("\nheld-out exact-match by round:", ", ".join(f"r{h['round']}={h['exact']:.3f}" for h in history))
    print(f"{'PASS' if best['exact'] >= args.pass_threshold else 'FAIL'}: best exact {best['exact']:.3f} "
          f"(round {best['round']}) vs threshold {args.pass_threshold:.3f} (single-step action match)")
    probes = [h["probe"] for h in history if h.get("probe")]
    if probes:
        top = max(probes, key=lambda p: p["per_turn"])
        print("autonomy gate:", describe_probe(top, args.probe_horizon))
        ok = top["per_turn"] >= args.probe_threshold
        print(f"{'PASS' if ok else 'FAIL'}: per-turn {top['per_turn']:.3f} vs threshold {args.probe_threshold:.3f}")
        for kind, k in sorted(top["problems"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {k:3d}x {kind}")
    if args.merge:
        del model
        torch.cuda.empty_cache()
        merge(args.model, args.out)


if __name__ == "__main__":
    main()
