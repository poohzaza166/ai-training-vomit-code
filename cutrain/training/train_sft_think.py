"""Teach the model to think again: SFT on the reasoning side of the think-pairs.

Why not ORPO: GRPO cannot put this behaviour back (an empty `<think></think>` in ~98% of samples
means the reasoning reward has no spread inside a group, so no gradient), and train_orpo.py dies on
this box with `CUDA error: an illegal memory access` inside its CPU-offload path, while running it
without offload needs ~7.5 GB. Plain next-token loss on the chosen side needs one forward instead of
two, so it fits with room to spare, and it is the right tool anyway: the target behaviour is already
in the data, nothing needs to be ranked.

Loss covers the completion only (TRL's completion_only_loss on a prompt/completion dataset), so the
system prompt, the tool list and the tool output never pull on the weights.

  python -m cutrain.dataprep.make_think_pairs data/unified.jsonl data/hf_agent.jsonl
  python -m cutrain.training.train_sft_think --adapter out/nemotron-nano-4b-grpo-gan/round0/adapter
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, cache_utils  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

cache_utils.DYNAMIC_LAYER_TYPE_MAPPING.setdefault("mlp", cache_utils.LinearAttentionLayer)

from torch.utils.checkpoint import checkpoint  # noqa: E402

from cutrain.paths import REPO  # noqa: E402
from cutrain.training.train_grpo_gan import NEMOTRON_TARGETS, _park  # noqa: E402
from cutrain.training.train_orpo import _chunk_nll, fit  # noqa: E402


class ChunkedSFTTrainer(SFTTrainer):
    """Cross-entropy over completion tokens, built `chunk` positions at a time.

    TRL's own loss path allocates an lm_head-sized tensor (784 MiB here) per step on top of a 6.2 GB
    resident model and dies on an 8 GB card at every sequence length. train_orpo.py already solved
    the same problem for this model, so reuse its helper: logits for 256 positions at a time, never
    the whole vocab times the whole sequence, and the prompt contributes nothing.

    Checked against the stock loss on an identical batch (Qwen2.5-0.5B, CPU): 4.802957 both ways.
    test_sft_think.py covers the shift and the masking against a plain full-logits reference.
    """

    chunk = 256

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        inner = self.accelerator.unwrap_model(model)
        base = inner.get_base_model() if hasattr(inner, "get_base_model") else inner  # LoRA stays injected
        hidden = base.model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                            use_cache=False).last_hidden_state
        first = int((labels != -100).int().argmax(-1).min())  # every completion token sits at or after this
        hidden, labels = hidden[:, first - 1:-1], labels[:, first:]
        nll = torch.cat([checkpoint(_chunk_nll, base.lm_head, hidden[:, i:i + self.chunk],
                                    labels[:, i:i + self.chunk], use_reentrant=False)
                         for i in range(0, labels.shape[1], self.chunk)], dim=1)
        mask = labels != -100
        loss = (nll * mask).sum() / mask.sum().clamp(min=1)
        return (loss, None) if return_outputs else loss


def load_rows(path: str, tok, max_length: int, limit: int) -> Dataset:
    """Think-pairs -> {prompt, completion}; the completion is the step that reasons first."""
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            got = fit(r, tok, max_length)
            if got is None:
                dropped += 1
                continue
            rows.append({"prompt": got["prompt"], "completion": got["chosen"]})
            if limit and len(rows) >= limit:
                break
    print(f"rows: {len(rows)} kept, {dropped} dropped (too long or template mismatch)")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--adapter", help="LoRA adapter to keep training (default: a fresh one)")
    ap.add_argument("--data", default=str(REPO / "data" / "think_pairs.jsonl"))
    ap.add_argument("--out", default=str(REPO / "out" / "nemotron-think"))
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", nargs="+", default=NEMOTRON_TARGETS)
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True,
                    help="frozen input embeddings in CPU RAM (~0.8 GB of VRAM back)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.bos_token = None
    train = load_rows(args.data, tok, args.max_length, args.limit)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["lm_head", "out_proj"]) if args.qlora else None
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 quantization_config=quant, device_map={"": 0})
    model.train()
    if args.offload:
        emb = model.get_input_embeddings()
        emb.register_forward_pre_hook(_park)
        emb.register_forward_hook(lambda mod, a, out: out.to("cuda"))
        emb.to("cpu")
    if args.kernels and hasattr(model, "set_use_kernels"):
        model.set_use_kernels(True)
    model.config.use_cache = False
    model.generation_config.eos_token_id = tok.eos_token_id  # repo ships 2; the template ends on <|im_end|>
    model.generation_config.pad_token_id = tok.pad_token_id
    model.enable_input_require_grads()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print("continuing from", args.adapter)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=args.lora_targets))
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out, max_length=args.max_length, completion_only_loss=True, packing=False,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine", warmup_steps=10, bf16=True,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        logging_steps=5, save_strategy="no", report_to="none", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = ChunkedSFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok)
    trainer.train()
    model.save_pretrained(f"{args.out}/adapter")
    print("saved", f"{args.out}/adapter")


if __name__ == "__main__":
    main()
