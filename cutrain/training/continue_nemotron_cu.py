"""Launch terminal GRPO from the trainable checkpoint behind the repository's CU export.

Uses a user systemd service with the memory limits tested on this mirror.
--dry-run validates paths and prints commands without starting a service.
This does not train from GGUF, export to Ollama, or implement research tasks.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import uuid

from cutrain.paths import REPO

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["cu", "pilot"], default="cu",
                    help="cu: original ORPO adapter exported to Ollama; pilot: recent experimental GRPO adapter")
    ap.add_argument("--out", required=True, help="new run directory, relative to this repository")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--sft-warmup-steps", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--critic", choices=["none", "self"], default="self")
    ap.add_argument("--token-loop-penalty", type=float, default=0.5)
    ap.add_argument("--loop-preference-data", nargs="+")
    ap.add_argument("--loop-preference-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.steps < 0 or args.sft_warmup_steps < 0 or args.group_size < 2:
        ap.error("steps must be nonnegative and group-size at least two")

    root = "nemotron-nano-4b-orpo" if args.source == "cu" else "long-grpo-mirror-pilot"
    adapter = REPO / "out" / root / "adapter"
    data = REPO / "data/long_agent_sft_memory_v2.jsonl"
    stopping_data = REPO / "data/long_agent_stopping_sft.jsonl"
    python = REPO / ".venv-grpo/bin/python"
    out = (REPO / args.out).resolve()
    if out.exists():
        ap.error("output already exists; choose a new run directory")
    for path in (adapter / "adapter_config.json", adapter / "adapter_model.safetensors", data, stopping_data, python):
        if not path.is_file():
            ap.error(f"required file missing: {path}")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != BASE:
        ap.error("unexpected adapter base; inspect checkpoint lineage before using this launcher")

    model = BASE
    manifest = REPO / "out/long-grpo-recovery/base_model.json"
    if manifest.exists():
        snapshot = Path(json.loads(manifest.read_text())["snapshot"])
        if (snapshot / "config.json").is_file():
            model = str(snapshot)
    train = [str(python), "-u", "-m", "cutrain.training.train_long_grpo",
             "--model", model, "--adapter", str(adapter), "--out", str(out),
             "--sft-data", str(data), str(stopping_data), "--sft-warmup-steps", str(args.sft_warmup_steps),
             "--token-loop-penalty", str(args.token_loop_penalty), "--loop-min-tokens", "64",
             "--steps", str(args.steps), "--group-size", str(args.group_size),
             "--critic", args.critic,
             "--max-turns", "6", "--max-new-tokens", "384",
             "--context-schedule", "1024", "2048", "--max-context", "6144",
             "--eval-episodes", "1", "--save-every", "1", "--no-offload", "--no-activation-offload"]
    if not 0 <= args.token_loop_penalty <= 1 or args.loop_preference_steps < 0:
        ap.error("invalid token loop penalty or preference step count")
    if args.loop_preference_steps:
        if not args.loop_preference_data:
            ap.error("loop-preference-steps requires loop-preference-data")
        files = [(REPO / p).resolve() for p in args.loop_preference_data]
        if not all(p.is_file() for p in files):
            ap.error("missing loop preference data")
        train += ["--loop-preference-steps", str(args.loop_preference_steps),
                  "--loop-preference-data", *map(str, files)]
    unit = "nemotron-cu-" + uuid.uuid4().hex[:10]
    log = REPO / "out/logs" / (unit + ".log")
    launch = ["systemd-run", "--user", "--unit=" + unit,
              "--property=MemoryHigh=5G", "--property=MemoryMax=6G",
              "--property=MemorySwapMax=512M", "--property=WorkingDirectory=" + str(REPO),
              "--property=StandardOutput=append:" + str(log), "--property=StandardError=inherit", *train]
    print("Checkpoint:", adapter)
    print("Training:", shlex.join(train))
    print("Service command:", shlex.join(launch))
    print("Status:", "systemctl --user status " + unit)
    print("Log:", "tail -f " + shlex.quote(str(log)))
    print("Output:", out)
    if args.dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(launch, check=True)


if __name__ == "__main__":
    main()
