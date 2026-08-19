#!/bin/bash
# The running evals are on e19 (first of 19,22,25,28,30).  Let e19 finish, then
# replace each run with an e30-only job so nothing already computed is wasted.
# The eval script skips epochs already present in its output file, so restarting
# against the same --out keeps e19 and computes only e30.
cd /home2/xyf/project/FABRIC
FIX=data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset

for cond in full atac; do
  (
    until [ -s "outputs/analysis/top1_${cond}.jsonl" ]; do sleep 20; done
    pkill -f "eval_top1.py --condition ${cond} " 2>/dev/null
    sleep 5
    device=cuda:0
    [ "$cond" = atac ] && device=cuda:1
    setsid nohup python scripts/eval_top1.py --condition "$cond" --fixture "$FIX" \
      --snapshot-dir "runs/checkpoint_snapshots/${cond}" \
      --run-dir "runs/fabric_v2_${cond}_seed1103" \
      --out "outputs/analysis/top1_${cond}.jsonl" --device "$device" --epochs 19,30 \
      >>"outputs/analysis/eval_${cond}.log" 2>&1 &
    echo "$(date '+%F %T')  ${cond}: e19 landed, restarted for e30 only" >>outputs/analysis/trim.log
  ) &
done
wait
