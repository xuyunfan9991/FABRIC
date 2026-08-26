#!/bin/bash
# Continue the early-stopped Full resident-cache seed-1103 run from epoch 53
# through epoch 75. Patience is raised to the safety cap, so only epoch 75 can
# terminate this continuation. The parent run remains append-only and read-only.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=1103
PY=/home/apps/anaconda3/bin/python
FIXTURE=data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset
CONFIG=configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_resident_continue75_nopatience_full.yaml
PARENT=runs/fabric_v2_full_rdtu_a0_shared6g2x_resident_continue100_seed1103
RUN=runs/fabric_v2_full_rdtu_a0_shared6g2x_resident_continue75_nopatience_seed1103
SESSION=fabric_full_rdtu_a0_resident_c75_nopatience_seed1103

if [ -n "$(git status --porcelain -- src/fabric)" ]; then
    echo "REFUSED: src/fabric has uncommitted changes"; exit 1
fi
if [ ! -f "$PARENT/latest.pt" ]; then
    echo "REFUSED: parent latest checkpoint is absent: $PARENT/latest.pt"; exit 1
fi
if [ -e "$RUN" ]; then
    echo "REFUSED: continuation run directory already exists: $RUN"; exit 1
fi

PYTHONPATH="$PWD/src" $PY - <<'EOF'
from pathlib import Path
import torch

from fabric.train import assert_execution_admitted, load_config

config = load_config(
    "configs/fabric_v2_full_cohort_reliability_dtu_macro_"
    "shared6g2x_resident_continue75_nopatience_full.yaml"
)
assert_execution_admitted(config, condition="full")
if (
    config["training"]["max_epochs"] != 75
    or config["training"]["early_stopping_patience"] != 75
):
    raise SystemExit("REFUSED: Full continuation is not cap=75/patience=75")
if config["execution"]["final_test_authorized"] is not False:
    raise SystemExit("REFUSED: held-out test is not closed")
parent = Path(
    "runs/fabric_v2_full_rdtu_a0_shared6g2x_resident_"
    "continue100_seed1103/latest.pt"
)
latest = torch.load(parent, map_location="cpu", weights_only=False)
if (
    latest.get("completed_epoch") != 53
    or latest.get("epochs_without_improvement") != 5
    or latest.get("training_complete") is not True
    or latest.get("held_out_test_evaluated") is not False
):
    raise SystemExit("REFUSED: parent is not the closed-test e53 patience stop")
print("preflight OK: Full e53 patience stop -> e75 cap, held-out test closed")
EOF

tmux new-session -d -s "$SESSION" -c "$PWD" \
    "taskset -c 0-15 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$PWD/src \
     $PY -m fabric.train \
     --config $CONFIG \
     --fixture $FIXTURE \
     --run-dir $RUN \
     --continue-from $PARENT/latest.pt \
     --condition full --seed $SEED --device cuda:0 \
     2>&1 | tee ${RUN}.log"
echo "launched Full: epoch 53 -> cap 75, patience disabled, GPU 0, cores 0-15, tmux $SESSION"
