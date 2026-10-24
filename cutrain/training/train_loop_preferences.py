"""Memory-bounded single-turn ORPO on observed loops versus verified alternatives.

Invoked by train_long_grpo.py. Score both sides without graphs, then recompute
one side at a time with exact scalar chain-rule coefficients. Tool observations
and the shared prompt never receive loss. No reference model is loaded.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from cutrain.envs.agent_grpo_env import TOOLS, parse_action
from cutrain.training.anti_loop import token_loops
from cutrain.serving.serve_adapter import render_turn


def orpo_loss(chosen, rejected, beta):
    def odds(x):
        x = x.clamp(max=-1e-7)
        return x - torch.log(-torch.expm1(x))
    return -chosen - beta * F.logsigmoid(odds(chosen) - odds(rejected))


def preference_example(row, tok, args):
    if row.get("format") != "verified_loop_preference_v1":
        raise ValueError("expected single-turn verified loop preference; trajectory archives are not supported")
    v = row["chosen_verification"]
    if not (v["success"] and v["verified"] and v["intact"] and v["tests"] == 1):
        raise ValueError("chosen response has no independently verified successful trajectory")
    if row.get("source") != "observed_policy_loop_vs_verified_policy_success":
        raise ValueError("negative must be an observed policy loop, not synthetic repetition")
    if len(row["chosen"]) != 1 or len(row["rejected"]) != 1:
        raise ValueError("preference sides must each contain one assistant response")
    c, r = row["chosen"][0], row["rejected"][0]
    if c["role"] != "assistant" or r["role"] != "assistant":
        raise ValueError("invalid assistant preference pair")
    # Chosen is a transcript message: its text is the template's rendering. Rejected is the raw completion.
    prompt, chosen_text = render_turn(tok, row["prompt"] + [c], len(row["prompt"]), row.get("tools") or TOOLS)
    if parse_action(chosen_text) is None:
        raise ValueError("invalid assistant preference pair")
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    chosen = tok.encode(chosen_text, add_special_tokens=False)
    rejected = tok.encode(r["content"], add_special_tokens=False)
    if token_loops(chosen, decode, args.loop_min_tokens)["loop_tokens"]:
        raise ValueError("chosen response itself loops")
    if not token_loops(rejected, decode, args.loop_min_tokens)["loop_tokens"]:
        raise ValueError("rejected response has no detected loop; do not train an unverified preference")
    prefix = tok.encode(prompt, add_special_tokens=False)
    chosen += [tok.eos_token_id]
    if row["rejected_eos"]:
        rejected += [tok.eos_token_id]
    if not prefix or max(len(prefix) + len(chosen), len(prefix) + len(rejected)) > args.max_context:
        raise ValueError("preference exceeds context budget; original evidence is never truncated")
    return prefix, chosen, rejected


def preference_warmup(model, tok, opt, args, policy_mode, log_probs, append_json, start=0, save=None):
    rows = []
    for path in args.loop_preference_data:
        rows.extend(json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())
    if not rows:
        raise ValueError("no observed loop preference pairs available")
    examples = [preference_example(row, tok, args) for row in rows]
    policy_mode(model, True)
    for step in range(start, args.loop_preference_steps):
        prefix, chosen, rejected = examples[step % len(examples)]
        ids = [torch.tensor([prefix + suffix], device="cuda") for suffix in (chosen, rejected)]
        with torch.no_grad():
            scores = [log_probs(model, x, len(prefix), args.logit_chunk, False).mean() for x in ids]
        c, r = [x.detach().requires_grad_() for x in scores]
        loss = orpo_loss(c, r, args.loop_preference_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loop preference loss")
        coefficients = torch.autograd.grad(loss, (c, r))
        opt.zero_grad(set_to_none=True)
        for x, coefficient in zip(ids, coefficients):
            mean = log_probs(model, x, len(prefix), args.logit_chunk, args.activation_offload).mean()
            (mean * coefficient.detach()).backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             1.0, error_if_nonfinite=True)
        opt.step()
        metric = {"preference_step": step, "loss": float(loss.detach()), "grad_norm": float(norm),
                  "chosen_logp": float(c.detach()), "rejected_logp": float(r.detach())}
        append_json(Path(args.out) / "loop_preference_metrics.jsonl", metric)
        print(json.dumps(metric), flush=True)
        if save:
            save(step + 1)
    if save:
        save(args.loop_preference_steps, force=True)
