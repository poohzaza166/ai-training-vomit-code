#!/usr/bin/env bash
# Run:    bash scripts/run_build_app.sh        (detached user service; survives SSH loss)
# Status: tail -f out/logs/build-app-*.log
#
# Serves the trained adapter and sends specs/<APP>.md, verbatim and once, to opencode; the model plans and builds.
# The model's project lives outside this training repo ($PROJECT_ROOT/<APP>); build logs and evidence
# stay here under out/app-builds/.

set -Euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-grpo/bin/python"
ADAPTER="${ADAPTER:-out/until-pass-20261016T024115Z/grpo-2/adapter}"
APP="${APP:-dnd-music-bot}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/model-apps}"
PORT="${PORT:-8011}"
SPEC="${SPEC:-specs/$APP.md}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-2048}"   # max tokens per model reply
HARNESS="${HARNESS:-opencode}"           # opencode | cline

if [[ "${1:-}" != --worker ]]; then
    [[ "$(loginctl show-user "$USER" -p Linger --value)" == yes ]] \
        || { echo "Error: run 'loginctl enable-linger $USER' first" >&2; exit 1; }
    case "$(realpath -m "$PROJECT_ROOT/$APP")/" in
        "$ROOT"/*) echo "Error: the model's project must be outside the training repo" >&2; exit 1 ;;
    esac
    [[ ! -e "$PROJECT_ROOT/$APP" ]] || { echo "Error: $PROJECT_ROOT/$APP exists; choose another APP name" >&2; exit 1; }
    [[ -f "$SPEC" ]] || { echo "Error: missing $SPEC" >&2; exit 1; }
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    unit="build-app-$stamp"
    mkdir -p "$ROOT/out/logs"
    systemd-run --user --unit="$unit" \
        --property=MemoryHigh=5500M --property=MemoryMax=6500M --property=MemorySwapMax=1G \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$ROOT/out/logs/$unit.log" --property=StandardError=inherit \
        /usr/bin/env "ADAPTER=$ADAPTER" "APP=$APP" "PROJECT_ROOT=$PROJECT_ROOT" "PORT=$PORT" "SPEC=$SPEC" "OUTPUT_TOKENS=$OUTPUT_TOKENS" "HARNESS=$HARNESS" \
        /bin/bash "$ROOT/scripts/run_build_app.sh" --worker
    printf 'Service: %s\nLog: %s\nProject: %s\n' "$unit" "$ROOT/out/logs/$unit.log" "$PROJECT_ROOT/$APP"
    exit 0
fi

MODEL="$("$PYTHON" -c "import json; print(json.load(open('out/long-grpo-recovery/base_model.json'))['snapshot'])")"
EVIDENCE="out/app-builds"
mkdir -p "$EVIDENCE" "$PROJECT_ROOT"
"$PYTHON" -u -m cutrain.serving.serve_adapter --model "$MODEL" --adapter "$ADAPTER" --name cu-app --port "$PORT" \
    --max-context 20480 --max-new-tokens "$OUTPUT_TOKENS" --log "$EVIDENCE/$APP.serve.jsonl" > "$EVIDENCE/$APP.server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT
for _ in $(seq 1 90); do
    grep -q '^serving ' "$EVIDENCE/$APP.server.log" && break
    kill -0 "$server_pid" 2>/dev/null || { echo "server died"; tail -5 "$EVIDENCE/$APP.server.log"; exit 1; }
    sleep 10
done
if [[ "$HARNESS" == cline ]]; then
    # Cline's provider settings (base URL http://127.0.0.1:$PORT/v1, model cu-app) live in ~/.local/share/cline-eval.
    "$PYTHON" -u -m cutrain.evaluation.build_app_with_cline --spec "$SPEC" --project "$PROJECT_ROOT/$APP" --log "$EVIDENCE/$APP.build.json"
else
    "$PYTHON" -u -m cutrain.evaluation.build_app_with_opencode --base-url "http://127.0.0.1:$PORT/v1" --model cu-app \
        --spec "$SPEC" --output "$OUTPUT_TOKENS" --project "$PROJECT_ROOT/$APP" --log "$EVIDENCE/$APP.build.json"
fi
