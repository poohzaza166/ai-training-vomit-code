#!/usr/bin/env bash
# Run:    bash scripts/run_probe_gate.sh            (launches a detached user service; survives SSH loss)
# Status: tail -f out/logs/probe-gate-*.log
#
# Agent gate: cutrain/evaluation/probe_agent.py through the real harness prompt and tools, for each adapter below.
# It waits for the harness (../src/harness) to appear, so it can be launched before copying it.
# PASS means per-turn pass rate >= 0.95 on the scripted probe turns; see docs/RUN_NOTES.md for limits.

set -Euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-grpo/bin/python"
HARNESS="$(dirname "$ROOT")/src/harness"
WAIT_HOURS="${WAIT_HOURS:-48}"
PROTOCOL="${PROTOCOL:-json}"   # live harness config uses TOOL_PROTOCOL=json
ADAPTERS=(
    "new:out/full-cu-20261015T153407Z/grpo-long/adapter"
    "prev:out/full-cu-20261015T002706Z/grpo-long/adapter"
    "orpo:out/nemotron-nano-4b-orpo/adapter"
)

if [[ "${1:-}" != --worker ]]; then
    [[ "$(loginctl show-user "$USER" -p Linger --value)" == yes ]] \
        || { echo "Error: run 'loginctl enable-linger $USER' or the job dies with your SSH session" >&2; exit 1; }
    unit="probe-gate-$(date -u +%Y%m%dT%H%M%SZ)"
    log="$ROOT/out/logs/$unit.log"
    mkdir -p "$ROOT/out/logs"
    systemd-run --user --unit="$unit" \
        --property=MemoryHigh=5G --property=MemoryMax=6G --property=MemorySwapMax=512M \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$log" --property=StandardError=inherit \
        /usr/bin/env "WAIT_HOURS=$WAIT_HOURS" "PROTOCOL=$PROTOCOL" \
        /bin/bash "$ROOT/scripts/run_probe_gate.sh" --worker
    printf 'Service: %s\nLog: %s\n' "$unit" "$log"
    exit 0
fi

deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
until [[ -d "$HARNESS" ]]; do
    if (( $(date +%s) > deadline )); then
        echo "Error: $HARNESS did not appear within $WAIT_HOURS hours" >&2
        exit 1
    fi
    echo "$(date -u +%FT%TZ) waiting for $HARNESS"
    sleep 300
done
echo "$(date -u +%FT%TZ) harness found: $HARNESS"
# Do not start mid-copy: wait until nothing under ../src has changed for two minutes.
while [[ -n "$(find "$(dirname "$HARNESS")" -mmin -2 -print -quit)" ]]; do
    echo "$(date -u +%FT%TZ) harness files still changing; waiting"
    sleep 60
done

MODEL="$("$PYTHON" -c "import json; print(json.load(open('out/long-grpo-recovery/base_model.json'))['snapshot'])")"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
outputs=()
for pair in "${ADAPTERS[@]}"; do
    name="${pair%%:*}"
    adapter="${pair#*:}"
    out="out/probe-gate-$stamp-$name.json"
    echo "=== probe $name: $adapter"
    if "$PYTHON" -u -m cutrain.evaluation.probe_agent --model "$MODEL" --adapter "$adapter" --protocol "$PROTOCOL" --out "$out"; then
        outputs+=("$name=$out")
    else
        echo "probe $name failed; continuing with the remaining adapters"
    fi
done

# Gate summary, using run_probe's per-turn definition: 1 - problems / turns.
"$PYTHON" - "${outputs[@]}" <<'SUMMARY' | tee "out/probe-gate-$stamp-summary.txt"
import json
import sys

for item in sys.argv[1:]:
    name, path = item.split("=", 1)
    episodes = json.load(open(path))["episodes"]
    turns = sum(e["turns"] for e in episodes)
    bad = sum(len(e["problems"]) for e in episodes)
    p = max(0.0, 1 - bad / max(turns, 1))
    clean = sum(not e["problems"] for e in episodes)
    print(f"{name:5} per-turn {p:.3f} ({turns - bad}/{turns} turns), {clean}/{len(episodes)} episodes clean, "
          f"50-turn survival {p ** 50:.1%} -> {'PASS' if p >= 0.95 else 'FAIL'}")
    for e in episodes:
        for problem in e["problems"]:
            print(f"      {e['id']}: {problem[:200]}")
SUMMARY
echo PROBE_GATE_DONE
