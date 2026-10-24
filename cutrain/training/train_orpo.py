"""ORPO LoRA fine-tune of Nemotron-3-Nano-4B on the unified dataset (see prepare_data.py).

Why the BF16 repo: NVIDIA-Nemotron-3-Nano-4B-FP8 is a modelopt export (fp8 weights + separate
scales, no HF quantization_config), so transformers would load it as garbage. The BF16 repo holds
the same weights before quantization. Train here, merge, then serve FP8 (e.g. vLLM --quantization fp8).

  pip install -r "computer use training/requirements-train.txt"
  python -m cutrain.training.train_orpo --merge
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # ~0.8 GB fragmentation on 8 GB cards
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from transformers.utils import cached_file  # noqa: E402

try:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer
except ImportError:  # trl < 0.26
    from trl import ORPOConfig, ORPOTrainer

from cutrain.paths import REPO

ASSISTANT = "<|im_start|>assistant\n"


def _chunk_nll(lm_head, hidden, labels):
    logits = lm_head(hidden).float()
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").view_as(labels)


class CompletionORPOTrainer(ORPOTrainer):
    """ORPO with the NLL and log-probs over completion tokens only, logits built 256 positions at a time.

    Stock TRL ORPO builds full-vocab logits for every position (131k vocab: ~1 GB fp32 per 1k tokens of
    chosen+rejected, OOM on 8 GB) and puts the prompt (system, tool list, tool outputs) into the NLL term.
    """

    offload = True  # input embeddings + activation checkpoints (~0.26 MB/token here) live in CPU RAM
    chunk = 256

    def concatenated_forward(self, model, batch):
        cb = self.concatenated_inputs(batch, padding_value=self.padding_value, device=self.accelerator.device)
        labels = cb["concatenated_labels"]
        n_chosen = batch["chosen_labels"].shape[0]
        base = self.accelerator.unwrap_model(model).get_base_model()  # LoRA layers stay injected
        if self.offload:
            base.get_input_embeddings().to("cpu")  # no-op after the first step; accelerate.prepare moved it to GPU
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if self.offload else contextlib.nullcontext():
            hidden = base.model(input_ids=cb["concatenated_input_ids"], attention_mask=cb["concatenated_attention_mask"],
                                use_cache=False).last_hidden_state
        # chosen and rejected share the prompt, so every completion token sits at or after `first`
        first = int((labels != -100).int().argmax(-1).min())
        hidden, labels = hidden[:, first - 1:-1], labels[:, first:]
        nll = torch.cat([checkpoint(_chunk_nll, base.lm_head, hidden[:, i:i + self.chunk], labels[:, i:i + self.chunk],
                                    use_reentrant=False) for i in range(0, labels.shape[1], self.chunk)], dim=1)
        mask = labels != -100
        logps = -(nll * mask).sum(-1) / mask.sum(-1)  # average_log_prob, as stock ORPO
        chosen_nll = nll[:n_chosen][mask[:n_chosen]].mean()
        zero = logps.new_zeros(())  # ponytail: logits never kept, so TRL's logits/* metrics log 0
        return logps[:n_chosen], logps[n_chosen:], zero, zero, chosen_nll


def render(row: dict, tok) -> dict | None:
    """Unified messages -> prompt/chosen/rejected strings via the model's own chat template.

    Rendered here (not by TRL) so `tools` reach the template, and rows are plain JSON (no Arrow
    schema merging, which would pad tool arguments of other rows in as None).
    """
    kw = {"tools": row["tools"] or None, "tokenize": False}
    prompt = tok.apply_chat_template(row["prompt"], **kw) + ASSISTANT
    out = {"prompt": prompt}
    for k in ("chosen", "rejected"):
        full = tok.apply_chat_template(row["prompt"] + row[k], **kw)
        if not full.startswith(prompt):
            return None
        out[k] = full[len(prompt):].rstrip("\n")  # ends with <|im_end|> == eos, so TRL adds no second eos
    return out


def fit(row: dict, tok, max_length: int) -> dict | None:
    """Render; while too long, drop the oldest assistant turn (+ its tool results) after the first user msg.

    TRL ORPO never truncates the prompt itself. ponytail: re-renders per dropped turn; cheap because
    prepare_data.py --max-history-chars already bounds history. Batch the drops if that budget grows.
    """
    p = row["prompt"]
    first_user = next((i for i, m in enumerate(p) if m["role"] == "user"), None)
    while first_user is not None:
        r = render({**row, "prompt": p}, tok)
        if r is None:
            return None
        if len(tok(r["prompt"] + max(r["chosen"], r["rejected"], key=len))["input_ids"]) <= max_length:
            return r
        cut = next((i for i in range(first_user + 1, len(p)) if p[i]["role"] == "assistant"), None)
        if cut is None:
            return None
        end = cut + 1
        while end < len(p) and p[end]["role"] == "tool":
            end += 1
        p = p[:cut] + p[end:]
    return None


def load_rows(path: str, tok, max_length: int) -> Dataset:
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = fit(json.loads(line), tok, max_length)
            if r is None:
                dropped += 1
            else:
                rows.append(r)
    print(f"rows: {len(rows)} kept, {dropped} dropped (too long even with history trimmed, or template mismatch)")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--data", default=str(REPO / "data" / "unified.jsonl"))
    ap.add_argument("--out", default=str(REPO / "out" / "nemotron-nano-4b-orpo"))
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke runs)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="ORPO lambda (odds-ratio loss weight)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--adapter", help="LoRA adapter to continue training instead of starting a fresh one")
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True,
                    help="4-bit base weights (default; bf16 4B model alone needs ~8GB)")
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True,
                    help="prebuilt mamba-ssm/causal-conv1d from the HF hub (pip: kernels, einops); the pure "
                         "PyTorch Mamba fallback materializes ~48 GB at 2k tokens")
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True,
                    help="frozen input embeddings + activation checkpoints in CPU RAM (needed to fit 4k tokens in 8 GB)")
    ap.add_argument("--merge", action="store_true", help="also write merged bf16 model to <out>/merged")
    ap.add_argument("--merge-only", action="store_true", help="skip training, merge the existing <out>/adapter")
    args = ap.parse_args()
    if "FP8" in args.model:
        ap.error("train on the BF16 repo; the FP8 repo is a modelopt export transformers can't load (see docstring)")
    if args.merge_only:
        return merge(args.model, args.out)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.unk_token  # no pad token in the repo; id 0 matches generation_config.pad_token_id
    tok.bos_token = None  # chat template has no <s>; stop TRL from prepending one
    train = load_rows(args.data, tok, args.max_length)

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Mamba mixer casts activations to out_proj.weight.dtype and its fused kernel reads the raw
            # weight -> must stay a real bf16 tensor
            llm_int8_skip_modules=["lm_head", "out_proj"],
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0})
    model.train()
    if args.kernels:
        model.set_use_kernels(True)  # after .train(): kernelizing an eval model picks inference-only kernels
    if args.offload:  # trainer parks the frozen input embeddings (0.77 GB, lookup only) on CPU
        emb = model.get_input_embeddings()
        emb.register_forward_pre_hook(lambda mod, args: (args[0].to(mod.weight.device),))
        emb.register_forward_hook(lambda mod, args, out: out.to("cuda"))
    CompletionORPOTrainer.offload = args.offload
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # peft directly instead of trainer peft_config: TRL's prepare_model_for_kbit_training upcasts
    # embeddings/lm_head/out_proj to fp32, which does not fit in 8GB
    if args.adapter:  # keep training the same LoRA rather than stacking a second one on top of it
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print("continuing from", args.adapter)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            # attention + mamba in_proj + MLP; PEFT refuses mamba out_proj/conv1d
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    cfg = ORPOConfig(
        output_dir=args.out, beta=args.beta, learning_rate=args.lr, max_length=args.max_length,
        num_train_epochs=args.epochs, max_steps=args.max_steps, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_steps=10,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch", bf16=True,
        logging_steps=5, save_steps=200, save_total_limit=2, report_to="none",
    )
    trainer = CompletionORPOTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok)
    trainer.train()
    trainer.save_model(f"{args.out}/adapter")

    if args.merge:
        del model, trainer
        torch.cuda.empty_cache()
        merge(args.model, args.out)


def merge(model_id: str, out: str) -> None:
    """Fold <out>/adapter into the bf16 base (on CPU) and write <out>/merged."""
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, f"{out}/adapter").merge_and_unload()
    # repo's generation_config has temperature/top_p without do_sample; transformers 5 refuses to save that
    merged.generation_config.do_sample = True
    merged.save_pretrained(f"{out}/merged")
    # weights are saved in NVIDIA's layout but transformers 5 writes its own config schema (default MoE fields,
    # no hybrid_override_pattern) that llama.cpp/Ollama read as an 8-expert MoE -> ship the repo's config
    shutil.copy(cached_file(model_id, "config.json"), f"{out}/merged/config.json")
    AutoTokenizer.from_pretrained(model_id).save_pretrained(f"{out}/merged")
    print("merged ->", f"{out}/merged")


if __name__ == "__main__":
    main()
