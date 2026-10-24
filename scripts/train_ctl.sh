#!/usr/bin/env bash
# Pause training to free the GPU and RAM (e.g. before a game) and resume it afterwards.
#
#   bash scripts/train_ctl.sh pause           stop every running training service
#   bash scripts/train_ctl.sh resume [UNIT]   start paused runs again, from their last finished step
#                                     (UNIT: also a run stopped by a reboot or crash)
#   bash scripts/train_ctl.sh status
#
# Covers runs started by run_mix_grpo.sh and run_project_training.sh, which record out/logs/<unit>.launch.
# A paused run loses at most the step that was in progress: cutrain/training/train_long_grpo.py saves after every GRPO
# step (and every --checkpoint-minutes during SFT), and finished stages are skipped on resume.

set -Euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${TRAIN_CTL_LOGS:-$ROOT/out/logs}"
source "$ROOT/scripts/launch_lib.sh"

active() { systemctl --user is-active --quiet "$1"; }

progress() {
    local file="$ROOT/out/$1/progress.txt"
    [[ -s "$file" ]] && tail -n 1 "$file" || true
}

pause() {
    local f unit n=0
    for f in "$LOGS"/*.launch; do
        [[ -e "$f" ]] || continue
        unit="$(basename "$f" .launch)"
        active "$unit" || continue
        systemctl --user stop "$unit"
        touch "$LOGS/$unit.paused"
        echo "paused $unit"
        n=$((n + 1))
    done
    # Task containers run under dockerd, outside the service, and would keep their memory.
    docker ps -aq --filter name=mimo- 2> /dev/null | xargs -r docker rm -f > /dev/null 2>&1 || true
    ((n)) || echo "nothing was running"
}

resume() {
    local units=("$@") f unit record
    if ((${#units[@]} == 0)); then
        for f in "$LOGS"/*.paused; do
            [[ -e "$f" ]] && units+=("$(basename "$f" .paused)")
        done
    fi
    ((${#units[@]})) || { echo "nothing is paused"; return 0; }
    for unit in "${units[@]}"; do
        if active "$unit"; then
            echo "$unit is already running"
        elif [[ ! -s "$LOGS/$unit.launch" ]]; then
            echo "no launch record for $unit" >&2
            continue
        else
            mapfile -t record < "$LOGS/$unit.launch"
            launch_unit "$unit" "${record[@]}" RESUME=1 > /dev/null
            echo "resumed $unit"
        fi
        rm -f "$LOGS/$unit.paused"
    done
}

status() {
    local f unit last n=0
    for f in "$LOGS"/*.launch; do
        [[ -e "$f" ]] || continue
        unit="$(basename "$f" .launch)"
        last="$(progress "$unit")"
        if active "$unit"; then
            echo "running  $unit  $last"
        elif [[ -e "$LOGS/$unit.paused" ]]; then
            echo "paused   $unit  $last"
        elif [[ "$last" != *"ALL DONE"* ]]; then
            echo "stopped  $unit  $last  (resume with: bash scripts/train_ctl.sh resume $unit)"
        else
            continue
        fi
        n=$((n + 1))
    done
    ((n)) || echo "no training runs running or paused"
}

case "${1:-}" in
    pause) pause ;;
    resume) shift; resume "$@" ;;
    status) status ;;
    *) sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
