#!/bin/bash
# Continue the completed resident-cache three-arm seed-1103 campaign from
# epoch 30 under a new max-epoch safety cap of 100. Parent runs are read-only;
# each arm writes a new append-only run with an explicit continuation manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=1103
PY=/home/apps/anaconda3/bin/python
FIXTURE=data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset
PREFIX=fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_resident_continue100

if [ -n "$(git status --porcelain -- src/fabric)" ]; then
    echo "REFUSED: src/fabric has uncommitted changes"; exit 1
fi
for cond in full atac rbp; do
    parent="runs/fabric_v2_${cond}_rdtu_a0_shared6g2x_resident_seed${SEED}"
    run="runs/fabric_v2_${cond}_rdtu_a0_shared6g2x_resident_continue100_seed${SEED}"
    if [ ! -f "${parent}/latest.pt" ]; then
        echo "REFUSED: parent latest checkpoint is absent: ${parent}/latest.pt"
        exit 1
    fi
    if [ -e "$run" ]; then
        echo "REFUSED: continuation run directory already exists: $run"
        exit 1
    fi
done

PYTHONPATH="$PWD/src" $PY - <<'EOF'
from pathlib import Path
import torch

from fabric.train import RUN_CONDITIONS, assert_execution_admitted, load_config

seed = 1103
prefix = "fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_resident_continue100"
for condition in RUN_CONDITIONS:
    config = load_config(f"configs/{prefix}_{condition}.yaml")
    assert_execution_admitted(config, condition=condition)
    if config["training"]["max_epochs"] != 100:
        raise SystemExit(f"REFUSED: {condition} safety cap is not 100")
    if config["execution"]["final_test_authorized"] is not False:
        raise SystemExit(f"REFUSED: {condition} held-out test is not closed")
    parent = Path(
        f"runs/fabric_v2_{condition}_rdtu_a0_shared6g2x_resident_seed{seed}"
    )
    latest = torch.load(parent / "latest.pt", map_location="cpu", weights_only=False)
    if (
        latest.get("completed_epoch") != 30
        or latest.get("training_complete") is not True
        or latest.get("held_out_test_evaluated") is not False
    ):
        raise SystemExit(f"REFUSED: {condition} parent is not a closed-test epoch-30 terminal checkpoint")
print("preflight OK: three epoch-30 parents, cap=100, profiles admitted, held-out test closed")
EOF

launch() {
    local cond=$1 gpu=$2 cores=$3
    local parent="fabric_v2_${cond}_rdtu_a0_shared6g2x_resident_seed${SEED}"
    local run="fabric_v2_${cond}_rdtu_a0_shared6g2x_resident_continue100_seed${SEED}"
    local session="fabric_${cond}_rdtu_a0_resident_c100_seed${SEED}"
    tmux new-session -d -s "$session" -c "$PWD" \
        "taskset -c ${cores} env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$PWD/src \
         $PY -m fabric.train \
         --config configs/${PREFIX}_${cond}.yaml \
         --fixture $FIXTURE \
         --run-dir runs/${run} \
         --continue-from runs/${parent}/latest.pt \
         --condition ${cond} --seed ${SEED} --device cuda:${gpu} \
         2>&1 | tee runs/${run}.log"
    echo "launched ${cond}: epoch 30 -> cap 100, GPU ${gpu}, cores ${cores}, tmux ${session}"
}

launch full 0 0-15
launch atac 1 64-79
launch rbp 1 80-95
