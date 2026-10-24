#!/usr/bin/env bash
# Run:    bash scripts/run_full_training.sh
# Check:  bash scripts/run_full_training.sh --prepare-only
#
# Pipeline: cutrain/dataprep/build_dataset.py -> mixed SFT -> short GRPO -> long GRPO
# The default runs in the background and survives an SSH disconnect.

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# SETTINGS — edit the defaults here, or supply environment variables when running.
# Example: SHORT_STEPS=80 LONG_STEPS=80 bash scripts/run_full_training.sh
# Skip the dataset build and SFT, starting GRPO from an existing SFT adapter:
#          SFT_ADAPTER=out/<run>/mixed-sft/adapter bash scripts/run_full_training.sh
# -----------------------------------------------------------------------------

SFT_EPOCHS="${SFT_EPOCHS:-1}"
SHORT_STEPS="${SHORT_STEPS:-40}"       # GRPO groups, not guaranteed optimizer steps
LONG_STEPS="${LONG_STEPS:-40}"
GROUP_SIZE="${GROUP_SIZE:-4}"          # Responses sampled per group, sequentially
EVAL_EPISODES="${EVAL_EPISODES:-4}"    # Held-out tasks per context length
SEED="${SEED:-42}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv-grpo/bin/python"
RUN_DIR="${RUN_DIR:-$ROOT/out/full-cu-$(date -u +%Y%m%dT%H%M%SZ)}"
START_ADAPTER="${START_ADAPTER:-$ROOT/out/nemotron-nano-4b-orpo/adapter}"
SFT_ADAPTER="${SFT_ADAPTER:-}"         # Set to skip the dataset build and mixed-SFT stage

# Resolve relative output paths before passing them to the background service.
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$ROOT/$RUN_DIR"
fi
if [[ -n "$SFT_ADAPTER" && "$SFT_ADAPTER" != /* ]]; then
    SFT_ADAPTER="$ROOT/$SFT_ADAPTER"
fi

# Use the downloaded base model associated with the CU adapter.
if [[ -z "${MODEL:-}" ]]; then
    MODEL="$("$PYTHON" - <<'PY'
import json
from pathlib import Path

manifest = Path('out/long-grpo-recovery/base_model.json')
print(json.loads(manifest.read_text())['snapshot'])
PY
)"
fi

# Shared trainer options. Each stage below adds its own context and checkpoint.
TRAINING_OPTIONS=(
    --model "$MODEL"
    --group-size "$GROUP_SIZE"
    --max-turns 8
    --max-new-tokens 384
    --lr 2e-6
    --sft-lr 5e-6
    --temperature 1.0
    --token-loop-penalty 0.5
    --loop-min-tokens 64
    --eval-episodes "$EVAL_EPISODES"
    --save-every 10
    --seed "$SEED"

    # Memory settings for the current remote machine.
    --logit-chunk 32
    --qlora
    --kernels
    --no-offload
    --no-activation-offload
)

# -----------------------------------------------------------------------------
# HELPERS — background launching, validation and command logging.
# The dataset preparation and four training stages are below these helpers.
# -----------------------------------------------------------------------------

fail() {
    echo "Error: $*" >&2
    exit 1
}

launch_background() {
    local unit="full-cu-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    local log="$ROOT/out/logs/$unit.log"
    mkdir -p "$ROOT/out/logs"

    systemd-run --user \
        --unit="$unit" \
        --property=MemoryHigh=5G \
        --property=MemoryMax=6G \
        --property=MemorySwapMax=512M \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$log" \
        --property=StandardError=inherit \
        /usr/bin/env \
            "RUN_DIR=$RUN_DIR" \
            "MODEL=$MODEL" \
            "START_ADAPTER=$START_ADAPTER" \
            "SFT_ADAPTER=$SFT_ADAPTER" \
            "SFT_EPOCHS=$SFT_EPOCHS" \
            "SHORT_STEPS=$SHORT_STEPS" \
            "LONG_STEPS=$LONG_STEPS" \
            "GROUP_SIZE=$GROUP_SIZE" \
            "EVAL_EPISODES=$EVAL_EPISODES" \
            "SEED=$SEED" \
        /bin/bash "$ROOT/scripts/run_full_training.sh" --worker

    printf 'Service: %s\nLog: %s\nOutput: %s\n' "$unit" "$log" "$RUN_DIR"
}

run_stage() {
    local stage="$1"
    shift

    local command=(
        "$PYTHON" -u -m cutrain.training.train_long_grpo
        "${TRAINING_OPTIONS[@]}"
        --out "$RUN_DIR/$stage"
        "$@"
    )

    echo
    echo "Stage: $stage"
    # One argument per line keeps logged commands readable and copyable.
    printf '%q' "${command[0]}"
    printf ' \\\n    %q' "${command[@]:1}"
    printf '\n'

    if [[ "$MODE" != --prepare-only ]]; then
        "${command[@]}"
    fi
}

MODE="${1:---launch}"
case "$MODE" in
    --launch | --worker | --prepare-only)
        ;;
    *)
        fail 'Usage: bash scripts/run_full_training.sh [--launch|--prepare-only]'
        ;;
esac

[[ ! -e "$RUN_DIR" ]] || fail "Output already exists: $RUN_DIR"
[[ -f "$START_ADAPTER/adapter_model.safetensors" ]] || fail "Missing adapter: $START_ADAPTER"
if [[ -n "$SFT_ADAPTER" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_model.safetensors" ]] || fail "Missing SFT adapter: $SFT_ADAPTER"
fi
[[ -f "$MODEL/config.json" ]] || fail "Missing base model: $MODEL"

if [[ "$MODE" == --launch ]]; then
    launch_background
    exit 0
fi

mkdir -p "$RUN_DIR"
trap 'rc=$?; printf "Failed at line %s (exit %s). See the run log.\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR

# Only one copy of this full-run script may execute at a time.
exec 9>"$ROOT/out/full-training.lock"
flock -n 9 || fail 'Another full training script is running'

# -----------------------------------------------------------------------------
# DATA — your existing builder controls the mixture and removes duplicate IDs.
# The next converter preserves reasoning/tools and drops over-budget examples.
# -----------------------------------------------------------------------------

if [[ -z "$SFT_ADAPTER" ]]; then
echo 'Preparing the dataset with cutrain.dataprep.build_dataset'

"$PYTHON" -u -m cutrain.dataprep.build_dataset \
    --data \
        data/unified.jsonl \
        data/hf_agent.jsonl \
        data/hf_reasoning.jsonl \
    --out "$RUN_DIR/selected_unified.jsonl" \
    --seed "$SEED" \
    --other drop \
    --quota \
        opencode=1000 \
        claude=400 \
        hf:openthoughts-agent=600 \
        hf:arcee-glaive=600 \
        hf:nvidia/When2Call=600 \
        hf:dapo-math=400 \
        hf:v4-distill=600 \
        hf:glm-reasoning=300 \
        hf:mlabonne/orpo-dpo-mix-40k=200 \
    | tee "$RUN_DIR/build_dataset.log"

# This also adds the verified long-context and stopping episodes.
# Do not use the builder's legacy trimming-based fit check for this pipeline.
"$PYTHON" -u -m cutrain.dataprep.prepare_full_run \
    --model "$MODEL" \
    --data "$RUN_DIR/selected_unified.jsonl" \
    --out "$RUN_DIR/mixed_sft.jsonl" \
    --seed "$SEED" \
    --max-context 4096

# One SFT step trains one retained completion. Derive the epoch length from data.
SFT_STEPS="$("$PYTHON" - "$RUN_DIR/mixed_sft.report.json" "$SFT_EPOCHS" <<'PY'
import json
import sys

with open(sys.argv[1]) as report_file:
    examples = json.load(report_file)['examples']
epochs = int(sys.argv[2])
if epochs < 1:
    raise ValueError('SFT_EPOCHS must be a positive integer')
print(examples * epochs)
PY
)"
else
    echo "Skipping dataset preparation and SFT; GRPO starts from $SFT_ADAPTER"
    printf '%s\n' "$SFT_ADAPTER" > "$RUN_DIR/SFT_ADAPTER.txt"
fi

# Checksummed below: the SFT data, or the adapter GRPO starts from when SFT is skipped.
if [[ -n "$SFT_ADAPTER" ]]; then
    TRAINING_INPUT="$SFT_ADAPTER/adapter_model.safetensors"
else
    TRAINING_INPUT="$RUN_DIR/mixed_sft.jsonl"
fi

# Record the environment and exact inputs for reproducibility; pip is not needed.
"$PYTHON" - <<'PY' > "$RUN_DIR/packages.txt"
from importlib.metadata import distributions

packages = [
    f"{package.metadata['Name']}=={package.version}"
    for package in distributions()
    if package.metadata['Name']
]
print('\n'.join(sorted(packages)))
PY

sha256sum \
    cutrain/dataprep/build_dataset.py \
    cutrain/training/train_long_grpo.py \
    cutrain/dataprep/prepare_full_run.py \
    scripts/run_full_training.sh \
    cutrain/envs/agent_grpo_env.py \
    cutrain/training/agent_critique.py \
    "$TRAINING_INPUT" \
    > "$RUN_DIR/SHA256SUMS"

# -----------------------------------------------------------------------------
# STAGE 1 — check memory near the largest planned sequence length.
# Failure here stops the run before any training updates.
# -----------------------------------------------------------------------------

run_stage profile \
    --adapter "${SFT_ADAPTER:-$START_ADAPTER}" \
    --profile \
    --steps 0 \
    --max-context 6144 \
    --context-schedule 5760

# -----------------------------------------------------------------------------
# STAGE 2 — SFT on your reasoning/agent data plus verified terminal examples.
# Continues the original CU adapter. No GRPO updates in this stage.
# -----------------------------------------------------------------------------

if [[ -z "$SFT_ADAPTER" ]]; then
run_stage mixed-sft \
    --adapter "$START_ADAPTER" \
    --sft-data "$RUN_DIR/mixed_sft.jsonl" \
    --sft-warmup-steps "$SFT_STEPS" \
    --steps 0 \
    --critic none \
    --max-context 4096 \
    --context-schedule 1024 2048
fi

# -----------------------------------------------------------------------------
# STAGE 3 — GRPO on executable terminal tasks, starting with shorter histories.
# Continues the SFT adapter. Self-critique and anti-loop rewards are enabled.
# -----------------------------------------------------------------------------

run_stage grpo-short \
    --adapter "${SFT_ADAPTER:-$RUN_DIR/mixed-sft/adapter}" \
    --steps "$SHORT_STEPS" \
    --critic self \
    --max-context 4096 \
    --context-schedule 1024 2048

# -----------------------------------------------------------------------------
# STAGE 4 — continue GRPO with longer histories and a larger total context cap.
# -----------------------------------------------------------------------------

run_stage grpo-long \
    --adapter "$RUN_DIR/grpo-short/adapter" \
    --steps "$LONG_STEPS" \
    --critic self \
    --max-context 6144 \
    --context-schedule 2048 4096

if [[ "$MODE" == --prepare-only ]]; then
    echo 'Preparation complete. Training commands printed; no training started.'
else
    FINAL_ADAPTER="$RUN_DIR/grpo-long/adapter"
    printf '%s\n' "$FINAL_ADAPTER" > "$RUN_DIR/FINAL_ADAPTER.txt"
    echo "Training complete: $FINAL_ADAPTER"
    echo 'Review the stage summary.json files before deployment.'
fi
