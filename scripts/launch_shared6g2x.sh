#!/bin/bash
# Launch the three-arm shared-GPU 2x-capacity paired run (weekend campaign).
#
# Packing (matches the canary evidence): full exclusive on GPU 0;
# atac + rbp share GPU 1.  CPU pinning is NUMA-aligned with each arm's GPU
# and thread counts follow the bench_omp_threads.py sweep (16 per arm).
#
# Refuses to launch unless src/fabric is clean, every config has
# training_authorized: true, the canary evidence says passed, and none of the
# run directories exists.  Each train process applies its own condition/profile
# admission and GPU-memory checks.  This is the exact first-launch record; it
# deliberately refuses to reuse any materialized run directory.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=1103
PY=/home/apps/anaconda3/bin/python
FIXTURE=data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset

if [ -n "$(git status --porcelain -- src/fabric)" ]; then
    echo "REFUSED: src/fabric has uncommitted changes"; exit 1
fi
for cond in full atac rbp; do
    run="runs/fabric_v2_${cond}_rdtu_a0_shared6g2x_seed${SEED}"
    if [ -e "$run" ]; then
        echo "REFUSED: historical run directory already exists: $run"
        echo "Use an explicit new run identity, or resume Full from its latest.pt."
        exit 1
    fi
done
$PY - <<'EOF'
import json, sys, yaml
evidence = json.load(open("data/processed/fabric_v2_shared6g2x_resources_v1/CanaryEvidence.json"))
if not evidence.get("canary_passed"):
    sys.exit("REFUSED: shared-GPU canary evidence is not a pass")
for cond in ("full", "atac", "rbp"):
    cfg = yaml.safe_load(open(
        f"configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_{cond}.yaml"))
    if cfg["execution"]["training_authorized"] is not True:
        sys.exit(f"REFUSED: {cond} config training_authorized is not true")
print("preflight OK: canary passed, all three configs authorized")
EOF

launch() {
    local cond=$1 gpu=$2 cores=$3
    local run="fabric_v2_${cond}_rdtu_a0_shared6g2x_seed${SEED}"
    tmux new-session -d -s "fabric_${cond}_rdtu_a0_2x_seed${SEED}" -c "$PWD" \
        "taskset -c ${cores} env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$PWD/src \
         $PY -m fabric.train \
         --config configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_${cond}.yaml \
         --fixture $FIXTURE \
         --run-dir runs/${run} \
         --condition ${cond} --seed ${SEED} --device cuda:${gpu} \
         2>&1 | tee runs/${run}.log"
    echo "launched ${cond}: GPU ${gpu}, cores ${cores}, tmux fabric_${cond}_rdtu_a0_2x_seed${SEED}"
}

launch full 0 0-15
launch atac 1 64-79
launch rbp  1 80-95
sleep 20
pgrep -af "[-]m fabric.train" | cut -c1-100
