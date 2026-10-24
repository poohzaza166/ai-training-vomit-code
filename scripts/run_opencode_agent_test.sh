#!/usr/bin/env bash
# Run:    bash scripts/run_opencode_agent_test.sh       (detached user service; survives SSH loss)
# Status: tail -f out/logs/opencode-agent-test-*.log
#
# Agent test through the opencode harness, for each adapter below in turn: serve it with
# cutrain/serving/serve_adapter.py, run cutrain/evaluation/opencode_agent_test.py on held-out tasks, stop the server.
# PASS: per-turn validity >= 0.95, task success >= 90%, and no unsupported success claims.

set -Euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-grpo/bin/python"
TASKS="${TASKS:-12}"
PORT="${PORT:-8011}"
FIRST_SEED="${FIRST_SEED:-7000000}"   # use unseen seeds for an independent evaluation
# Override with ADAPTER_LIST="name:path name:path"
read -r -a ADAPTERS <<< "${ADAPTER_LIST:-new:out/full-cu-20261015T153407Z/grpo-long/adapter prev:out/full-cu-20261015T002706Z/grpo-long/adapter orpo:out/nemotron-nano-4b-orpo/adapter}"

if [[ "${1:-}" != --worker ]]; then
    [[ "$(loginctl show-user "$USER" -p Linger --value)" == yes ]] \
        || { echo "Error: run 'loginctl enable-linger $USER' or the job dies with your SSH session" >&2; exit 1; }
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    unit="opencode-agent-test-$stamp"
    log="$ROOT/out/logs/$unit.log"
    mkdir -p "$ROOT/out/logs"
    systemd-run --user --unit="$unit" \
        --property=MemoryHigh=5500M --property=MemoryMax=6500M --property=MemorySwapMax=1G \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$log" --property=StandardError=inherit \
        /usr/bin/env "TASKS=$TASKS" "PORT=$PORT" "STAMP=$stamp" "FIRST_SEED=$FIRST_SEED" "ADAPTER_LIST=${ADAPTERS[*]}" \
        /bin/bash "$ROOT/scripts/run_opencode_agent_test.sh" --worker
    printf 'Service: %s\nLog: %s\nResults: %s\n' "$unit" "$log" "$ROOT/out/opencode-test/$stamp"
    exit 0
fi

MODEL="$("$PYTHON" -c "import json; print(json.load(open('out/long-grpo-recovery/base_model.json'))['snapshot'])")"
RESULTS="out/opencode-test/$STAMP"
mkdir -p "$RESULTS"
server_pid=""
trap '[[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null' EXIT

for pair in "${ADAPTERS[@]}"; do
    name="${pair%%:*}"
    adapter="${pair#*:}"
    out="$RESULTS/$name"
    mkdir -p "$out"
    echo "=== $(date -u +%FT%TZ) adapter $name: $adapter"
    "$PYTHON" -u -m cutrain.serving.serve_adapter --model "$MODEL" --adapter "$adapter" --name "cu-$name" \
        --port "$PORT" --log "$out/serve.jsonl" > "$out/server.log" 2>&1 &
    server_pid=$!
    ready=no
    for _ in $(seq 1 90); do
        grep -q '^serving ' "$out/server.log" && { ready=yes; break; }
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 10
    done
    if [[ "$ready" == yes ]]; then
        "$PYTHON" -u -m cutrain.evaluation.opencode_agent_test --base-url "http://127.0.0.1:$PORT/v1" --model "cu-$name" \
            --server-log "$out/serve.jsonl" --out "$out" --tasks "$TASKS" --first-seed "$FIRST_SEED" --timeout 600 \
            || echo "test driver failed for $name"
    else
        echo "server for $name did not start; see $out/server.log"
        tail -5 "$out/server.log"
    fi
    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    server_pid=""
done

"$PYTHON" - "$RESULTS" <<'SUMMARY' | tee "$RESULTS/summary.txt"
import json
import sys
from pathlib import Path

for d in sorted(Path(sys.argv[1]).iterdir()):
    f = d / "summary.json"
    if not f.exists():
        print(f"{d.name:5} no result")
        continue
    s = json.loads(f.read_text())
    print(f"{d.name:5} tasks {s['successes']}/{s['tasks']}  per-turn {s['per_turn']:.3f} over {s['turns']} turns  "
          f"false claims {s['false_claims']}  timeouts {s['timeouts']}  -> {'PASS' if s['pass'] else 'FAIL'}")
SUMMARY
echo OPENCODE_AGENT_TEST_DONE
