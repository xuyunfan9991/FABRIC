#!/bin/bash
# Keep one checkpoint per epoch.  train.py overwrites latest.pt in place every
# epoch, so without this the per-epoch weights are lost and metrics that need a
# forward pass (e.g. ONT top-1) can only ever be computed for the final model.
#
# latest.pt is self-contained: it carries both model_state_dict (this epoch) and
# best_model_state_dict (the selected best epoch), so one file per epoch is enough.
set -u

ROOT=/home2/xyf/project/FABRIC
SNAP=$ROOT/runs/checkpoint_snapshots
LOG=$SNAP/snapshot.log
RUNS="full_macro atac_macro"
POLL_SECONDS=60
GRACE_ROUNDS=3   # rounds to keep polling after both trainers exit

log() { echo "$(date '+%F %T')  $*" >>"$LOG"; }

exec 9>"$SNAP/.snapshot.lock"
if ! flock -n 9; then
    echo "another snapshot daemon is already running" >&2
    exit 1
fi

log "daemon started (pid $$, poll ${POLL_SECONDS}s)"
idle=0

while :; do
    for run in $RUNS; do
        run_dir=$ROOT/runs/fabric_v2_${run}_seed1103
        [ -f "$run_dir/history.tsv" ] || continue

        # history.tsv is written after latest.pt, so a new row means the
        # checkpoint for that epoch is already complete on disk.
        epoch=$(tail -1 "$run_dir/history.tsv" | cut -f2)
        case "$epoch" in '' | *[!0-9]*) continue ;; esac

        dest=$SNAP/$run/epoch_${epoch}.pt
        [ -f "$dest" ] && continue

        mkdir -p "$SNAP/$run"
        if cp "$run_dir/latest.pt" "$dest.partial" && mv "$dest.partial" "$dest"; then
            log "saved $run e$epoch ($(stat -c%s "$dest") bytes)"
        else
            rm -f "$dest.partial"
            log "FAILED to save $run e$epoch"
        fi
    done

    alive=0
    for run in $RUNS; do
        pgrep -f "run-dir runs/fabric_v2_${run}_seed1103" >/dev/null 2>&1 && alive=1
    done

    if [ "$alive" -eq 1 ]; then
        idle=0
    else
        idle=$((idle + 1))
        if [ "$idle" -ge "$GRACE_ROUNDS" ]; then
            log "both trainers gone; daemon exiting"
            exit 0
        fi
    fi

    sleep "$POLL_SECONDS"
done
