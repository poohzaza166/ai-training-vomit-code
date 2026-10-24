#!/usr/bin/env bash
# Serve one adapter with cutrain/serving/serve_adapter.py, run the harness eval against it, stop the server.
#
#   harness_eval/run_harness_eval.sh out/project-train-20261017T002500Z/sft/adapter sft-20261017
#   harness_eval/run_harness_eval.sh out/until-pass-20261016T024115Z/grpo-2/adapter grpo2-20261016 --category safety
#
# Extra arguments go to run_harness_eval.py. HARNESS=<dir> evaluates another copy of the harness. Results land in out/harness-eval/<name>/.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ADAPTER="${1:?adapter dir}"; NAME="${2:?run name}"; shift 2
PORT="${PORT:-8011}"
OUT="$ROOT/out/harness-eval/$NAME"
mkdir -p "$OUT"

if [[ ! -x .venv-harness/bin/python ]]; then
    python3 -m venv .venv-harness
    .venv-harness/bin/pip install -q openai httpx caldav icalendar pytz pyyaml requests numpy sqlite-vec fastapi
fi

MODEL="$(.venv-grpo/bin/python -c "import json; print(json.load(open('out/long-grpo-recovery/base_model.json'))['snapshot'])")"
.venv-grpo/bin/python -u -m cutrain.serving.serve_adapter --model "$MODEL" --adapter "$ADAPTER" --name cu --port "$PORT" \
    --max-context 20480 --max-new-tokens 1024 --log "$OUT/serve.jsonl" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "^serving" "$OUT/server.log"; do
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 3
done
# The first generation compiles kernels (~30 s); keep that out of the timed cases.
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' > /dev/null

.venv-harness/bin/python -u harness_eval/run_harness_eval.py --base-url "http://127.0.0.1:$PORT/v1" \
    --model cu --out "$OUT" ${HARNESS:+--harness "$HARNESS"} "$@" 2>&1 | tee "$OUT/run.log" | grep -E --line-buffered "^\[|^\*\*|^wrote|Traceback|Error"
