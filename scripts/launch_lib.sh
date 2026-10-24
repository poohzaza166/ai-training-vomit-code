# Sourced by the run_*.sh launchers and train_ctl.sh.
#
# launch_unit UNIT SCRIPT [KEY=VALUE ...]
#   Starts "SCRIPT --worker" with that environment as a detached user service (survives SSH loss) and
#   records how in out/logs/UNIT.launch, one item per line, so `train_ctl.sh resume` can start the same
#   run again after `train_ctl.sh pause`. MEMORY_HIGH / MEMORY_MAX in the environment set the service's
#   RAM limits (default 5500M / 6500M).

launch_unit() {
    local unit="$1" script="$2"
    shift 2
    local root high=5500M max=6500M kv
    root="$(cd -- "$(dirname -- "$script")/.." && pwd)"
    local record=("$script")
    for kv in "$@"; do
        [[ "$kv" != *$'\n'* ]] || { echo "launch_unit: newline in $kv" >&2; return 1; }
        case "$kv" in
            MEMORY_HIGH=*) high="${kv#*=}" ;;
            MEMORY_MAX=*) max="${kv#*=}" ;;
        esac
        [[ "$kv" == RESUME=* ]] || record+=("$kv")  # resume is per start, not part of the run
    done
    mkdir -p "$root/out/logs"
    printf '%s\n' "${record[@]}" > "$root/out/logs/$unit.launch"
    systemctl --user reset-failed "$unit" 2> /dev/null || true
    # TimeoutStopSec: pause should hand the GPU back quickly; the trainer saves after every step,
    # so killing it mid-step loses at most that step.
    systemd-run --user --quiet --unit="$unit" \
        --property=MemoryHigh="$high" --property=MemoryMax="$max" --property=MemorySwapMax=1G \
        --property=TimeoutStopSec=30 \
        --property="WorkingDirectory=$root" \
        --property="StandardOutput=append:$root/out/logs/$unit.log" --property=StandardError=inherit \
        /usr/bin/env "$@" /bin/bash "$script" --worker
}
