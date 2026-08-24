"""Shared-GPU canary: run ATAC and RBP profiling concurrently on one GPU.

The shared-GPU admission contract requires, beyond three exclusive
per-condition profiles, a canary in which the ATAC and RBP workloads run on
the SAME GPU covering both train and validation shapes, with at least 2 GiB
of physical free memory throughout, and with persisted evidence (a single
nvidia-smi screenshot does not qualify).  This script launches both
profilers concurrently on one device, samples NVML free/used memory on an
interval for the whole overlap, and writes the sample series plus a summary
verdict JSON next to the resource profiles.

Run from the repo root:

    python scripts/shared_gpu_canary.py --device cuda:1 \
        --artifact-dir data/processed/fabric_v2_shared6g2x_resources_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MINIMUM_FREE_BYTES = 2 * 1024**3


def gpu_memory(index: int) -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}",
         "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    return int(out[0]) * 1024**2, int(out[1]) * 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--config-template",
                        default="configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_{condition}.yaml")
    parser.add_argument("--prepared",
                        default="data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cpu-ranges", default="64-79,80-95",
                        help="one taskset range per condition, comma-separated")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    args = parser.parse_args()

    device_index = int(args.device.split(":")[1])
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started_iso = datetime.now(timezone.utc).isoformat()

    processes = {}
    ranges = args.cpu_ranges.split(",")
    for condition, cpu_range in zip(("atac", "rbp"), ranges, strict=True):
        config = args.config_template.format(condition=condition)
        output = artifact_dir / f"CanaryProfile.{condition}.json"
        log = (artifact_dir / f"canary_{condition}.log").open("w")
        processes[condition] = subprocess.Popen(
            ["taskset", "-c", cpu_range, "env",
             "OMP_NUM_THREADS=16", "MKL_NUM_THREADS=16", "PYTHONPATH=src",
             "/home/apps/anaconda3/bin/python", "-m", "fabric.profile_real",
             "--prepared", args.prepared, "--config", config,
             "--condition", condition, "--device", args.device,
             "--output", str(output)],
            stdout=log, stderr=subprocess.STDOUT,
        )

    samples = []
    sample_path = artifact_dir / "canary_memory_samples.tsv"
    with sample_path.open("w") as handle:
        handle.write("unix_time\tfree_bytes\tused_bytes\talive_processes\n")
        while any(p.poll() is None for p in processes.values()):
            free, used = gpu_memory(device_index)
            alive = sum(p.poll() is None for p in processes.values())
            samples.append((time.time(), free, used, alive))
            handle.write(f"{samples[-1][0]:.1f}\t{free}\t{used}\t{alive}\n")
            handle.flush()
            time.sleep(args.sample_seconds)

    exit_codes = {c: p.wait() for c, p in processes.items()}
    both_alive = [s for s in samples if s[3] == 2]
    minimum_free = min((s[1] for s in both_alive), default=0)
    verdict = (
        all(code == 0 for code in exit_codes.values())
        and bool(both_alive)
        and minimum_free >= MINIMUM_FREE_BYTES
    )
    evidence = {
        "schema_version": "fabric.shared_gpu_canary_evidence.v1",
        "started_utc": started_iso,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "conditions": list(processes),
        "exit_codes": exit_codes,
        "sample_interval_seconds": args.sample_seconds,
        "sample_count": len(samples),
        "samples_with_both_processes": len(both_alive),
        "minimum_free_bytes_while_both_alive": minimum_free,
        "minimum_free_requirement_bytes": MINIMUM_FREE_BYTES,
        "memory_sample_series": str(sample_path),
        "source_git_commit": subprocess.run(
            ["git", "log", "-1", "--format=%H"], check=True,
            capture_output=True, text=True).stdout.strip(),
        "canary_passed": verdict,
    }
    evidence_path = artifact_dir / "CanaryEvidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"canary {'PASSED' if verdict else 'FAILED'}: "
          f"min free while both alive = {minimum_free/2**30:.2f} GiB "
          f"({len(both_alive)}/{len(samples)} overlap samples) -> {evidence_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
