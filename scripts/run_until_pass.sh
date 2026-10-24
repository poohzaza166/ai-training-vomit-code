#!/usr/bin/env bash
# Run:    bash scripts/run_until_pass.sh           (detached user service; survives SSH loss)
# Status: tail -f out/logs/until-pass-*.log ; cat out/until-pass-*/progress.txt
#
# Train until the adapter passes the opencode agent test, or MAX_ROUNDS GRPO rounds run out:
#   1. SFT on verified opencode-format demonstrations (data/opencode_sft.jsonl) + rehearsal rows
#   2. opencode agent test (run_opencode_agent_test.sh's criteria) -> stop on PASS
#   3. GRPO round in opencode_env (fresh task seeds each round) -> test again -> repeat
# The test tasks (seeds 7,000,000+) are never trained on; training uses 3,000,000+ (SFT) and
# 100,000 * round (GRPO). Both are the same pipeline-repair task family, so a PASS shows the agent
# loop works in opencode for this kind of task, not general coding ability.

set -Euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-grpo/bin/python"
START_ADAPTER="${START_ADAPTER:-out/full-cu-20261015T153407Z/grpo-long/adapter}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
GRPO_STEPS="${GRPO_STEPS:-16}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TEST_TASKS="${TEST_TASKS:-12}"
REHEARSAL_ROWS="${REHEARSAL_ROWS:-150}"
PORT="${PORT:-8011}"

if [[ "${1:-}" != --worker ]]; then
    [[ "$(loginctl show-user "$USER" -p Linger --value)" == yes ]] \
        || { echo "Error: run 'loginctl enable-linger $USER' or the job dies with your SSH session" >&2; exit 1; }
    [[ -s data/opencode_sft.jsonl ]] || { echo "Error: build data/opencode_sft.jsonl first (python -m cutrain.dataprep.build_opencode_sft)" >&2; exit 1; }
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    unit="until-pass-$stamp"
    mkdir -p "$ROOT/out/logs"
    systemd-run --user --unit="$unit" \
        --property=MemoryHigh=5500M --property=MemoryMax=6500M --property=MemorySwapMax=1G \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$ROOT/out/logs/$unit.log" --property=StandardError=inherit \
        /usr/bin/env "STAMP=$stamp" "START_ADAPTER=$START_ADAPTER" "MAX_ROUNDS=$MAX_ROUNDS" \
            "GRPO_STEPS=$GRPO_STEPS" "GROUP_SIZE=$GROUP_SIZE" "TEST_TASKS=$TEST_TASKS" \
            "REHEARSAL_ROWS=$REHEARSAL_ROWS" "PORT=$PORT" \
        /bin/bash "$ROOT/scripts/run_until_pass.sh" --worker
    printf 'Service: %s\nLog: %s\nRun: %s\n' "$unit" "$ROOT/out/logs/$unit.log" "$ROOT/out/until-pass-$stamp"
    exit 0
fi

RUN="out/until-pass-$STAMP"
mkdir -p "$RUN"
MODEL="$("$PYTHON" -c "import json; print(json.load(open('out/long-grpo-recovery/base_model.json'))['snapshot'])")"
server_pid=""
trap '[[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null' EXIT

TRAIN=(--model "$MODEL" --env opencode --critic none --group-size "$GROUP_SIZE" --max-turns 10
       --max-new-tokens 512 --max-context 12800 --context-schedule 10500 --lr 2e-6 --sft-lr 5e-6
       --temperature 1.0 --token-loop-penalty 0.5 --loop-min-tokens 64 --eval-episodes 2 --save-every 100
       --logit-chunk 32 --qlora --kernels --no-offload --no-activation-offload)

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/progress.txt"; }

# Agent test of one adapter; prints its summary.json path. Same criteria as run_opencode_agent_test.sh.
agent_test() {
    local name="$1" adapter="$2" out="$RUN/test-$1"
    mkdir -p "$out"
    "$PYTHON" -u -m cutrain.serving.serve_adapter --model "$MODEL" --adapter "$adapter" --name "cu-$name" \
        --port "$PORT" --log "$out/serve.jsonl" > "$out/server.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 90); do
        grep -q '^serving ' "$out/server.log" && break
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 10
    done
    if grep -q '^serving ' "$out/server.log"; then
        "$PYTHON" -u -m cutrain.evaluation.opencode_agent_test --base-url "http://127.0.0.1:$PORT/v1" --model "cu-$name" \
            --server-log "$out/serve.jsonl" --out "$out" --tasks "$TEST_TASKS" --timeout 600 >&2 \
            || echo "test driver failed for $name" >&2
    else
        echo "server for $name did not start" >&2
        tail -5 "$out/server.log" >&2
    fi
    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    server_pid=""
    echo "$out/summary.json"
}

# Returns 0 on PASS; records the result line.
check() {
    local name="$1" adapter="$2" summary
    summary="$(agent_test "$name" "$adapter")"
    if [[ ! -f "$summary" ]]; then
        log "$name: no test result"
        return 1
    fi
    log "$name: $("$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"tasks {s['successes']}/{s['tasks']}, per-turn {s['per_turn']:.3f}, false claims {s['false_claims']} -> {'PASS' if s['pass'] else 'FAIL'}\")" "$summary") ($adapter)"
    "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$summary"
}

finish() {
    printf '%s\n' "$1" > "$RUN/PASSED_ADAPTER.txt"
    log "PASS: $1"
    echo UNTIL_PASS_DONE
    exit 0
}

# Rehearsal rows from the earlier mixed SFT keep previous skills from being overwritten.
"$PYTHON" - "$REHEARSAL_ROWS" "$RUN/rehearsal_sft.jsonl" <<'REHEARSAL'
import json
import random
import sys

rows = [l for l in open("out/full-cu-20261015T002706Z/mixed_sft.jsonl") if l.strip()]
random.Random(0).shuffle(rows)
open(sys.argv[2], "w").writelines(rows[:int(sys.argv[1])])
REHEARSAL
SFT_STEPS=$(( $(wc -l < data/opencode_sft.jsonl) + REHEARSAL_ROWS ))
sha256sum data/opencode_sft.jsonl data/opencode_request_template.json cutrain/envs/opencode_env.py cutrain/training/train_long_grpo.py \
    cutrain/serving/serve_adapter.py cutrain/evaluation/opencode_agent_test.py > "$RUN/SHA256SUMS"

log "SFT: $SFT_STEPS steps from $START_ADAPTER"
"$PYTHON" -u -m cutrain.training.train_long_grpo "${TRAIN[@]}" --adapter "$START_ADAPTER" --out "$RUN/sft" \
    --sft-data data/opencode_sft.jsonl "$RUN/rehearsal_sft.jsonl" --sft-warmup-steps "$SFT_STEPS" --steps 0 \
    || { log "SFT failed"; exit 1; }
current="$RUN/sft/adapter"
check sft "$current" && finish "$current"

for round in $(seq 1 "$MAX_ROUNDS"); do
    log "GRPO round $round: $GRPO_STEPS groups from $current"
    "$PYTHON" -u -m cutrain.training.train_long_grpo "${TRAIN[@]}" --adapter "$current" --out "$RUN/grpo-$round" \
        --steps "$GRPO_STEPS" --seed $(( 100000 * round )) \
        || { log "GRPO round $round failed"; exit 1; }
    current="$RUN/grpo-$round/adapter"
    check "round$round" "$current" && finish "$current"
done

log "Not passed after $MAX_ROUNDS GRPO rounds; see the test results above"
echo UNTIL_PASS_DONE
