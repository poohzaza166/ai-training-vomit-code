"""Multi-turn terminal GRPO built on train_grpo_gan.load_model and the existing ORPO adapter.

One group = independently reset copies of the same executable task.
Rollouts are sequential; assistant tokens alone receive the trajectory advantage.
Fixed token normalization (Dr. GRPO style), one update per rollout group, no critic model.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import platform
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:False" if "microsoft" in platform.release().lower()
                      else "expandable_segments:True")
if not os.environ.get("CC") and not shutil.which("gcc") and not shutil.which("clang"):
    compiler = Path(sys.executable).parent / "grpo-cc"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from cutrain.envs.agent_grpo_env import Episode, action_text
from cutrain.paths import REPO
from cutrain.serving.serve_adapter import render_turn
from cutrain.training.anti_loop import token_loops, apply_loop_reward, loop_preferences, evaluation_metrics
from cutrain.training.train_grpo_gan import load_model, NEMOTRON_TARGETS

MIX_TASKS: dict = {}


def policy_mode(model, training):
    model.train(training)
    base = model.get_base_model()
    if getattr(base, "use_kernels", False) and base.device.type != "cuda":
        from kernels import Device, Mode, kernelize
        kernelize(base, device=Device(type="cuda"), mode=Mode.TRAINING if training else Mode.INFERENCE)


def render(tok, messages):
    # The default generation template opens <think> without closing it; our parser needs
    # the complete assistant response, as in train_orpo.ASSISTANT.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) + "<|im_start|>assistant\n"


def group_advantages(rewards):
    # Mean-only baseline avoids inverse-std weighting of easy/hard tasks.
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def clipped_loss(new, old, advantage, denom, epsilon=0.2):
    ratio = torch.exp(new - old)
    return -torch.minimum(ratio * advantage,
                         ratio.clamp(1 - epsilon, 1 + epsilon) * advantage).sum() / denom


def log_probs(model, input_ids, prompt_length, chunk=32, offload=True, temperature=1.0):
    base = model.get_base_model()
    ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload and torch.is_grad_enabled() else contextlib.nullcontext()
    with ctx:
        hidden = base.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            use_cache=False).last_hidden_state
        hidden = hidden[:, prompt_length - 1:-1]
        targets = input_ids[:, prompt_length:]
        def part(h, labels):
            logits = base.lm_head(h).float() / temperature
            return -F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    labels.reshape(-1), reduction="none").view_as(labels)
        vals = []
        for start in range(0, targets.shape[1], chunk):
            h, labels = hidden[:, start:start + chunk], targets[:, start:start + chunk]
            vals.append(checkpoint(part, h, labels, use_reentrant=False) if torch.is_grad_enabled() else part(h, labels))
        return torch.cat(vals, dim=1).squeeze(0)


def create_episode(seed, args, tok, target):
    if getattr(args, "env", "terminal") == "opencode":
        # opencode's own prompt fixes the context (~10k tokens); target does not apply.
        from cutrain.envs.opencode_env import OpencodeEpisode
        return OpencodeEpisode(seed, depth=args.depth)
    if getattr(args, "env", "terminal") == "mix":
        # One scheduled task per GRPO group (prepare_mix_grpo.py schedule); eval seeds take the held-out set.
        # Cached here, not on args: save_adapter writes vars(args) into every checkpoint.
        from cutrain.envs.mimo_env import load_tasks, make_episode
        if args.mix_tasks not in MIX_TASKS:
            MIX_TASKS[args.mix_tasks] = (load_tasks(Path(args.mix_tasks) / "train.jsonl"),
                                         load_tasks(Path(args.mix_tasks) / "eval.jsonl"))
        train, held_out = MIX_TASKS[args.mix_tasks]
        index = seed - args.seed
        task = (held_out[(index - 1000000) % len(held_out)] if index >= 1000000
                else train[(index + args.mix_offset) % len(train)])
        return make_episode(task, compact_at=args.compact_at)
    if getattr(args, "env", "terminal") == "project":
        # Whole project from an empty folder in trimmed-opencode format; compaction keeps context bounded.
        from cutrain.envs.project_env import ProjectEpisode
        from cutrain.envs.project_tasks import make_task
        return ProjectEpisode(make_task(seed), compact_at=args.compact_at)
    # Estimate is followed by exact token counting, allowing nonuniform tokenizer behavior.
    lines = max(0, int(target / 12))
    for attempt in range(6):
        ep = Episode(seed, depth=args.depth, audit_lines=lines)
        try:
            ep.seed_history()
        except Exception:
            ep.close()
            raise
        n = len(tok.encode(ep.prompt(tok), add_special_tokens=False))
        if n >= target:
            # Crop only irrelevant audit text to land close to the requested starting length.
            msg = ep.messages[-1]
            ids = tok.encode(msg["content"], add_special_tokens=False)
            excess = n - target
            if excess > 0 and len(ids) > excess + 8:
                msg["content"] = tok.decode(ids[:len(ids) - excess], skip_special_tokens=False)
                n = len(tok.encode(ep.prompt(tok), add_special_tokens=False))
            if n + args.max_new_tokens > args.max_context:
                ep.close()
                raise ValueError("initial context plus completion exceeds max-context")
            return ep
        ep.close()
        lines = max(lines + 10, int(lines * 1.5))
    raise ValueError("Could not construct bounded context prefix")


def append_json(path, item):
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def add_feedback(ep, feedback):
    # Inside the task message, not a new turn after the seeded history: in A/B on the first
    # full run's failed tasks a separate feedback turn cut retry success from 18/32 to 10/32.
    if feedback:
        if ep.messages[1]["role"] != "user":
            raise ValueError("expected the task as the first user message")
        ep.messages[1]["content"] += "\n\n" + feedback


def rollout(model, tok, args, seed, target, sample=True, capture_old=True, feedback=None):
    records = []
    end_reason = "turn_budget"
    policy_mode(model, False)
    with create_episode(seed, args, tok, target) as ep:
        add_feedback(ep, feedback)
        initial_message_count = len(ep.messages)
        for turn in range(args.max_turns):
            prompt = ep.prompt(tok) if hasattr(ep, "prompt") else render(tok, ep.messages)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            n = ids.shape[1]
            # Hard cap without deleting original requirements / older tool results.
            if n + args.max_new_tokens > args.max_context:
                end_reason = "context_budget"
                break
            with torch.no_grad():
                model.config.use_cache = True
                kwargs = {"do_sample": sample, "max_new_tokens": args.max_new_tokens,
                          "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
                          "use_cache": True}
                if sample:
                    kwargs.update(temperature=args.temperature, top_p=1.0, top_k=0)
                kwargs.update(repetition_penalty=1.0, no_repeat_ngram_size=0)
                try:
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
                finally:
                    model.config.use_cache = False
                completion = out[:, n:]
                if completion.numel() == 0:
                    end_reason = "empty_completion"
                    break
                old = log_probs(model, out, n, args.logit_chunk, False, args.temperature).cpu() if capture_old else None
            text = tok.decode(completion[0], skip_special_tokens=True)
            loop = token_loops(completion[0].tolist(),
                               lambda x: tok.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                               min_tokens=args.loop_min_tokens)
            truncated = int(completion[0, -1]) != tok.eos_token_id
            records.append({"ids": out.cpu(), "prompt_length": n, "old": old,
                            "completion": text, "truncated": truncated,
                            "message_index": len(ep.messages), **loop})
            if truncated:
                ep.reject_generation(text)
            else:
                ep.step(text)
            print(f"rollout seed={seed} turn={turn + 1} context={n} generated={completion.numel()} "
                  f"action={text[-160:]!r}", flush=True)
            if ep.finished:
                end_reason = "task_done"
                break
        result = apply_loop_reward(ep.result(), records, args.token_loop_penalty)
        result.update(seed=seed, end_reason=end_reason, target_context=target,
                      initial_message_count=initial_message_count, feedback_conditioned=bool(feedback),
                      max_prompt_tokens=max((r["prompt_length"] for r in records), default=0),
                      generated_tokens=sum(r["ids"].shape[1] - r["prompt_length"] for r in records),
                      truncated_turns=sum(r["truncated"] for r in records))
        transcript = list(ep.messages)
    return records, result, transcript


def save_adapter(model, tok, directory, args, stats):
    # Written beside the target and renamed in: a kill mid-save leaves no half adapter, and a resumed
    # run that repeats the step writes the same checkpoint again.
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    (tmp / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    (tmp / "metrics.json").write_text(json.dumps(stats, indent=2, allow_nan=False))
    shutil.rmtree(directory, ignore_errors=True)
    os.rename(tmp, directory)


STATE_GLOB = "state-[0-9]*"


def save_state(out, model, opt, state):
    """Resume point (adapter, optimizer, RNG, loop state), renamed into place: the process may be killed at any time."""
    out = Path(out)
    tmp = out / "state.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    model.save_pretrained(tmp / "adapter")
    torch.save({"optimizer": opt.state_dict() if opt is not None else None,
                "rng": [random.getstate(), torch.get_rng_state(),
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []]},
               tmp / "train_state.pt")
    (tmp / "state.json").write_text(json.dumps(state, allow_nan=False))
    os.rename(tmp, out / f"state-{state['seq']:06d}")
    for old in sorted(out.glob(STATE_GLOB))[:-2]:
        shutil.rmtree(old)


def load_state(directory):
    directory = Path(directory)
    return json.loads((directory / "state.json").read_text()), torch.load(directory / "train_state.pt")


def sft_examples(tok, args, step, rows):
    if rows:
        row = rows[step % len(rows)]
        if row.get("format") == "rendered_sft_v1":
            if not row.get("rendered_prompt") or not row.get("completion"):
                raise ValueError("Empty rendered SFT example")
            yield row["rendered_prompt"], row["completion"], 1
            return
        messages = row["messages"]
        indices = row.get("assistant_loss_indices", [i for i, m in enumerate(messages) if m["role"] == "assistant"])
        if not indices:
            raise ValueError("SFT episode has no labelled assistant turns")
        for i in indices:
            if i < 1 or i >= len(messages) or messages[i]["role"] != "assistant":
                raise ValueError("Invalid SFT assistant loss index")
            # The target is the template's own rendering of the turn (reasoning, native calls), not raw content.
            prompt, completion = render_turn(tok, messages, i, row.get("tools"))
            yield prompt, completion, len(indices)
        return
    target = args.context_schedule[step % len(args.context_schedule)]
    with create_episode(args.seed + 500000 + step, args, tok, target) as ep:
        actions = ep.oracle()
        for action in actions:
            ep.step(action_text(action))
            i = max(j for j, m in enumerate(ep.messages) if m["role"] == "assistant")
            prompt, completion = render_turn(tok, ep.messages, i, ep.tools)
            yield prompt, completion, len(actions)


def sft_warmup(model, tok, opt, args, start=0, save=None):
    """Assistant-only SFT on verified traces; seeded context and tool output receive no labels.

    start skips steps already done by a paused run; save(next_step, force) records a resume point.
    """
    rows = []
    for path in args.sft_data or []:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if args.sft_data and not rows:
        raise ValueError("Empty SFT dataset")
    random.Random(args.seed).shuffle(rows)
    policy_mode(model, True)
    for step in range(start, args.sft_warmup_steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for prompt, answer, turns in sft_examples(tok, args, step, rows):
            prefix = tok.encode(prompt, add_special_tokens=False)
            suffix = tok.encode(answer + tok.eos_token, add_special_tokens=False)
            if len(prefix) + len(suffix) > args.max_context:
                raise ValueError("SFT example exceeds max-context; no evidence was silently truncated")
            ids = torch.tensor([prefix + suffix], device="cuda")
            lp = log_probs(model, ids, len(prefix), args.logit_chunk, args.activation_offload)
            loss = -lp.mean() / turns
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite SFT loss")
            loss.backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                             error_if_nonfinite=True)
        opt.step()
        metric = {"sft_step": step, "loss": total, "grad_norm": float(norm)}
        append_json(Path(args.out) / "sft_metrics.jsonl", metric)
        print(json.dumps(metric), flush=True)
        if save:
            save(step + 1)
    if save:
        save(args.sft_warmup_steps, force=True)


def evaluate(model, tok, args, label):
    results = []
    for target in args.context_schedule:
        for i in range(args.eval_episodes):
            _, res, transcript = rollout(model, tok, args, args.seed + 1000000 + i, target,
                                         sample=False, capture_old=False)
            res.update(phase=label)
            append_json(Path(args.out) / "episodes.jsonl", {**res, "messages": transcript})
            results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--adapter", default=str(REPO / "out/nemotron-nano-4b-orpo/adapter"))
    ap.add_argument("--out", required=True, help="new directory; never overwrite an existing run")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--context-schedule", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--env", choices=["terminal", "opencode", "project", "mix"], default="terminal",
                    help="opencode: opencode's captured prompt, tools and tool-result text (opencode_env.py); "
                         "project: whole projects from a plain-language outline (project_env.py); "
                         "mix: MiMo code/terminal containers + DeepSeek/GLM reasoning (mimo_env.py)")
    ap.add_argument("--mix-tasks", default=str(REPO / "data/mix_grpo"),
                    help="mix env: directory with train.jsonl (one task per group, in order) and eval.jsonl")
    ap.add_argument("--mix-offset", type=int, default=0,
                    help="mix env: start at this row of train.jsonl (resume from a step-N checkpoint with N)")
    ap.add_argument("--compact-at", type=int, default=9500,
                    help="project env: prompt tokens at which opencode-style compaction is requested")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--logit-chunk", type=int, default=32)
    ap.add_argument("--activation-offload", action=argparse.BooleanOptionalAction, default=False,
                    help="only enable when host RAM has measured headroom")
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=False,
                    help="park embeddings on CPU; default avoids mixed-device kernel selection")
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--sft-warmup-steps", type=int, default=0)
    ap.add_argument("--sft-lr", type=float, default=1e-5)
    ap.add_argument("--sft-data", nargs="+",
                    help="JSONL: rendered_sft_v1 rows, or episodes with messages, tools and optional assistant_loss_indices")
    ap.add_argument("--min-host-available-gib", type=float, default=3.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--profile", action="store_true", help="measure real long-context forward/backward without training")
    ap.add_argument("--critic", choices=["none", "self"], default="none",
                    help="self: review failed trajectories (logged) and retry the task in a fresh group; no critic reward")
    ap.add_argument("--retry-feedback", choices=["none", "facts", "facts+critic"], default="none",
                    help="what a retry group sees. none: the plain task again (30/32 retry successes in A/B on "
                         "the first full run's failed tasks); facts: 18/32; the old critic feedback: 4/32")
    ap.add_argument("--critic-max-new-tokens", type=int, default=192)
    ap.add_argument("--token-loop-penalty", type=float, default=0.5,
                    help="capped episode penalty for detected degenerate token cycles; 0 is the ablation")
    ap.add_argument("--loop-min-tokens", type=int, default=64)
    ap.add_argument("--loop-preference-data", nargs="+")
    ap.add_argument("--loop-preference-steps", type=int, default=0)
    ap.add_argument("--loop-preference-beta", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest state-* in --out, or start fresh if there is none; every "
                         "other argument must match the original run (train_ctl.sh pause/resume relies on this)")
    ap.add_argument("--checkpoint-minutes", type=float, default=10,
                    help="SFT/preference resume-point interval; GRPO records one after every step")
    args = ap.parse_args()
    if not 0 <= args.token_loop_penalty <= 1 or args.loop_min_tokens < 32:
        ap.error("token-loop-penalty must be in [0,1]; loop-min-tokens must be >=32")
    if args.loop_preference_steps < 0 or args.loop_preference_beta <= 0:
        ap.error("invalid preference step count or beta")
    if args.loop_preference_steps and not args.loop_preference_data:
        ap.error("loop-preference-steps requires observed, verified loop-preference-data")
    if args.env in ("opencode", "project") and (args.critic != "none" or args.profile):
        ap.error(f"--env {args.env} supports neither --critic self (long transcripts exceed the critic "
                 "budget) nor --profile; profile with the terminal env at the same context")
    if args.env == "project" and args.compact_at + args.max_new_tokens > args.max_context:
        ap.error("--compact-at plus --max-new-tokens must fit --max-context")
    if args.critic_max_new_tokens < 1 or args.critic_max_new_tokens >= args.max_context:
        ap.error("critic generation budget must be positive and smaller than max-context")
    if args.group_size < 2 or args.steps < 0 or args.max_turns < 1 or args.max_new_tokens < 1:
        ap.error("need group-size >= 2 and positive rollout limits; steps >= 0")
    if args.temperature <= 0 or args.logit_chunk < 1 or args.save_every < 1 or args.depth < 1:
        ap.error("temperature, chunk size, save interval and depth must be positive")
    if args.sft_warmup_steps < 0 or args.eval_episodes < 0 or args.lr <= 0 or args.sft_lr <= 0:
        ap.error("training/evaluation counts must be nonnegative and learning rates positive")
    if args.checkpoint_minutes < 0:
        ap.error("checkpoint-minutes must be nonnegative (0: a resume point after every SFT step)")
    if min(args.context_schedule) < 1 or max(args.context_schedule) + args.max_new_tokens > args.max_context:
        ap.error("context-schedule must fit max-context with completion headroom")
    if not torch.cuda.is_available():
        ap.error("CUDA is unavailable: repair the NVIDIA kernel driver before training; see docs/GRPO_PLAN_AND_REPORT.md")
    import psutil
    available = psutil.virtual_memory().available / 1024**3
    if available < args.min_host_available_gib:
        ap.error(f"only {available:.2f} GiB host RAM available; minimum {args.min_host_available_gib:.2f}")
    if args.activation_offload and available < 12:
        ap.error("activation offload requires at least 12 GiB available host RAM in this training setup")
    # Host RAM only backs long contexts through activation offload; without it the GPU profile decides.
    if args.activation_offload and available < 12 and args.max_context > 8192:
        ap.error("profile smaller contexts first; contexts above 8K require at least 12 GiB available host RAM")
    if Path(args.out).exists() and not args.resume:
        ap.error("out already exists; choose a new run directory (or --resume it)")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.lora_r, args.lora_targets = 16, NEMOTRON_TARGETS
    config = Path(args.out) / "config.json"
    current = {k: v for k, v in json.loads(json.dumps(vars(args))).items() if k != "resume"}
    if config.exists():
        original = {k: v for k, v in json.loads(config.read_text()).items() if k != "resume"}
        changed = sorted(k for k in original.keys() | current.keys() if original.get(k) != current.get(k))
        if changed:
            ap.error("--resume needs the original run's arguments; these differ: " + ", ".join(changed))
    else:
        config.write_text(json.dumps(vars(args), indent=2))
    st = {"seq": 0, "attempt": 0, "phase": "before", "step": 0, "before": [], "post_sft": [],
          "post_preference": [], "metrics": [], "updates": 0, "flat": 0, "pending_retry": None}
    saved, resume_phase, load_args = None, None, args
    latest = max(Path(args.out).glob(STATE_GLOB), default=None) if args.resume else None
    if latest:
        st, saved = load_state(latest)
        if st["phase"] == "done":
            print(f"{args.out} already finished; nothing to resume", flush=True)
            return
        resume_phase = st["phase"]
        load_args = argparse.Namespace(**{**vars(args), "adapter": str(latest / "adapter")})
        print(f"phase=resuming from {latest} ({resume_phase}, step {st['step']})", flush=True)
    st["attempt"] += 1
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("phase=loading_model (CPU/disk work and kernel downloads can leave GPU utilization at zero)", flush=True)
    model, tok = load_model(load_args)  # existing loader preserves ORPO lineage, QLoRA exclusions and EOS fix
    print("phase=model_ready", flush=True)
    # Policy scoring and sampling must agree: dropout stays off during both passes.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    if saved:
        random.setstate(saved["rng"][0])
        torch.set_rng_state(saved["rng"][1])
        torch.cuda.set_rng_state_all(saved["rng"][2])

    def checkpoint(active_opt, **changes):
        st.update(changes, seq=st["seq"] + 1)
        save_state(args.out, model, active_opt, st)

    def throttled(active_opt):
        # SFT/preference steps take seconds: a resume point every few minutes bounds both the work
        # lost on pause and the disk traffic.
        last = [time.monotonic()]
        def save(step, force=False):
            if force or time.monotonic() - last[0] >= args.checkpoint_minutes * 60:
                checkpoint(active_opt, step=step)
                last[0] = time.monotonic()
        return save

    def restore(active_opt, phase):
        if resume_phase == phase and saved["optimizer"] is not None:
            active_opt.load_state_dict(saved["optimizer"])

    if args.profile:
        results = []
        policy_mode(model, True)
        for target in args.context_schedule:
            print(f"phase=profile target_context={target} completion_tokens={args.max_new_tokens}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            with create_episode(args.seed, args, tok, target) as ep:
                prompt = tok.encode(ep.prompt(tok), add_special_tokens=False)
                completion = tok.encode(action_text(ep.oracle()[0]) + tok.eos_token, add_special_tokens=False)
                # Stress the full configured completion budget, not a convenient 19-token turn.
                completion = (completion * math.ceil(args.max_new_tokens / len(completion)))[:args.max_new_tokens]
                ids = torch.tensor([prompt + completion], device="cuda")
                lp = log_probs(model, ids, len(prompt), args.logit_chunk, args.activation_offload)
                loss = -lp.mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                result = {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                          "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                          "seconds": time.monotonic() - start, "loss": float(loss.detach()),
                          "grad_norm": float(norm)}
                results.append(result)
                print(json.dumps(result), flush=True)
                (Path(args.out) / "profile.json").write_text(json.dumps(results, indent=2))
        return

    critic = None
    if args.critic == "self":
        from cutrain.training.agent_critique import TrajectorySelfCritic
        critic = TrajectorySelfCritic(model, tok, Path(args.out) / "critic_memory.json",
                                      policy_mode, render, args.max_context, args.critic_max_new_tokens)
    # Phases run in order; a resumed run skips the finished ones. An interrupted evaluation runs again.
    if st["phase"] == "before":
        st["before"] = evaluate(model, tok, args, "before") if args.eval_episodes else []
        checkpoint(None, phase="sft", step=0)
    if st["phase"] == "sft":
        if args.sft_warmup_steps:
            sft_opt = torch.optim.AdamW(params, lr=args.sft_lr)
            restore(sft_opt, "sft")
            sft_warmup(model, tok, sft_opt, args, st["step"], throttled(sft_opt))
            del sft_opt
            opt.zero_grad(set_to_none=True)
            save_adapter(model, tok, Path(args.out) / "sft-adapter", args, {"steps": args.sft_warmup_steps})
            st["post_sft"] = evaluate(model, tok, args, "post_sft") if args.eval_episodes else []
        checkpoint(None, phase="preference", step=0)
    if st["phase"] == "preference":
        if args.loop_preference_steps:
            from cutrain.training.train_loop_preferences import preference_warmup
            pref_opt = torch.optim.AdamW(params, lr=args.sft_lr)
            restore(pref_opt, "preference")
            preference_warmup(model, tok, pref_opt, args, policy_mode, log_probs, append_json,
                              st["step"], throttled(pref_opt))
            del pref_opt
            opt.zero_grad(set_to_none=True)
            save_adapter(model, tok, Path(args.out) / "preference-adapter", args, {"steps": args.loop_preference_steps})
            st["post_preference"] = evaluate(model, tok, args, "post_preference") if args.eval_episodes else []
        checkpoint(None, phase="grpo", step=0)
    restore(opt, "grpo")
    updates, flat, metrics, pending_retry = st["updates"], st["flat"], st["metrics"], st["pending_retry"]
    # Rows of a step interrupted by a pause appear again with a higher attempt; the last attempt counted.
    for step in range(st["step"] if st["phase"] == "grpo" else args.steps, args.steps):
        # Curriculum progresses in blocks; later stages retain a quarter short-context rehearsal.
        stage = min(len(args.context_schedule) - 1, step * len(args.context_schedule) // max(args.steps, 1))
        target = args.context_schedule[0] if step % 4 == 3 else args.context_schedule[stage]
        seed = args.seed + step
        feedback, is_retry = None, False
        if pending_retry is not None:
            seed, target, feedback = pending_retry
            pending_retry, is_retry = None, True
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        group, results, transcripts = [], [], []
        for member in range(args.group_size):
            records, result, transcript = rollout(model, tok, args, seed, target, feedback=feedback)
            group.append(records)
            results.append(result)
            transcripts.append(transcript)
            append_json(Path(args.out) / "episodes.jsonl",
                        {**result, "step": step, "member": member, "phase": "train", "attempt": st["attempt"],
                         "messages": transcript})
        advantages = group_advantages([r["reward"] for r in results])
        for pair in loop_preferences(results, transcripts):
            append_json(Path(args.out) / "loop_preferences.jsonl", {"step": step, **pair})
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        norm = 0.0
        if max(advantages) - min(advantages) < 1e-8:
            flat += 1
        else:
            policy_mode(model, True)
            # Fixed denominator avoids rewarding shorter trajectories through length normalization.
            denom = args.group_size * args.max_turns * args.max_new_tokens
            for trajectory, advantage in zip(group, advantages):
                if abs(advantage) < 1e-8:
                    continue
                for rec in trajectory:
                    lp = log_probs(model, rec["ids"].to("cuda"), rec["prompt_length"],
                                   args.logit_chunk, args.activation_offload, args.temperature)
                    loss = clipped_loss(lp, rec["old"].to("cuda"), advantage, denom)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite GRPO loss")
                    loss.backward()
                    loss_total += loss.detach().item()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
            updates += 1
        review = None
        if critic is not None:
            from cutrain.training.agent_critique import make_feedback, harvest_verified
            sft_rows, pairs = harvest_verified(results, transcripts)
            for i, row in enumerate(sft_rows):
                append_json(Path(args.out) / "verified_rollouts_sft.jsonl",
                            {"episode_id": f"policy:{step}:{i}", **row})
            for pair in pairs:
                append_json(Path(args.out) / "trajectory_pairs.jsonl", {"step": step, **pair})
            # Critique once per ordinary group; the next group retries from a clean workspace.
            # Every member receives identical feedback (or none), preserving a common GRPO condition.
            if not is_retry and any(not r["success"] for r in results):
                worst = min(range(len(results)), key=lambda i: results[i]["reward"])
                review = critic.review(transcripts[worst], results[worst])
                note = None
                if args.retry_feedback != "none":
                    note = make_feedback(results[worst], review, critic_notes=args.retry_feedback == "facts+critic")
                append_json(Path(args.out) / "trajectory_critiques.jsonl",
                            {"step": step, "seed": seed, "member": worst,
                             "verification": results[worst], "feedback": note, **review})
                critic.remember()
                if step + 1 < args.steps:
                    pending_retry = (seed, target, note)
        metric = {"step": step, "attempt": st["attempt"], "target_context": target, "retry": is_retry,
                  "feedback_conditioned": bool(feedback),
                  "critic_status": review["status"] if review else None,
                  "rewards": [r["reward"] for r in results],
                  "task_rewards": [r["task_reward"] for r in results],
                  "token_loop_penalties": [r["token_loop_penalty"] for r in results],
                  "token_loop_turns": sum(r["token_loop_turns"] for r in results),
                  "action_repeats": sum(r["repeats"] for r in results),
                  "truncated_turns": sum(r["truncated_turns"] for r in results),
                  "advantages": advantages, "loss": loss_total, "grad_norm": float(norm),
                  "successes": sum(r["success"] for r in results), "updates": updates, "flat_groups": flat,
                  "seconds": time.monotonic() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        metrics.append(metric)
        print(json.dumps(metric), flush=True)
        append_json(Path(args.out) / "metrics.jsonl", metric)
        if (step + 1) % args.save_every == 0:
            save_adapter(model, tok, Path(args.out) / f"step-{step + 1:05d}", args, metric)
        checkpoint(opt, step=step + 1, updates=updates, flat=flat, metrics=metrics, pending_retry=pending_retry)
    if st["phase"] == "grpo":
        checkpoint(None, phase="after", step=0)
    before, post_sft, post_preference = st["before"], st["post_sft"], st["post_preference"]
    after = evaluate(model, tok, args, "after") if args.eval_episodes else []
    summary = {"updates": updates, "flat_groups": flat, "before": before, "post_sft": post_sft,
               "post_preference": post_preference, "after": after, "training": metrics,
               "evaluation_metrics": {k: evaluation_metrics(v) for k, v in
                                      (("before", before), ("post_sft", post_sft),
                                       ("post_preference", post_preference), ("after", after))}}
    save_adapter(model, tok, Path(args.out) / "adapter", args, summary)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    checkpoint(None, phase="done")
    print(f"saved {args.out}/adapter; {updates} GRPO updates, {flat} zero-advantage groups", flush=True)


if __name__ == "__main__":
    main()
