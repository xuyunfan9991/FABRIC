"""Benchmark the per-gene training step loop under a fixed intra-op thread count.

The full-cohort loop spends most wall time on CPU-side per-gene tensor work
(sampling, batch planning, EC subsetting) plus many small GPU launches.  On a
256-core host PyTorch defaults intra-op threads to the core count, so every
tiny op pays a fork/join on a huge OpenMP team.  This script mirrors the
_fit_condition inner loop (sample -> plan -> subset -> forward -> compatible
NLL -> backward -> clip -> step) over a mass-stratified sample of real genes
and times full passes, so thread settings can be compared like for like.

Launch one process per setting; the OpenMP team size is fixed at process
start:

    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python scripts/bench_omp_threads.py

Shard prefetching is not used here, so pass times include sequential shard
deserialization; that cost is identical across settings.  The warm pass is
reported separately (it additionally pays cold page-cache reads).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from fabric.likelihood import compatible_path_nll  # noqa: E402
from fabric.model import FABRICV2Model  # noqa: E402
from fabric.train import (  # noqa: E402
    _MODEL_CONDITION,
    BackedPreparedDataset,
    _model_spec,
    _plan_gene_cell_batches,
    _subset_gene_cells,
    build_optimizer,
    load_config,
    rows_for_split,
    sample_train_gene_cells_for_epoch,
    split_informative_molecule_mass,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fabric_v2_full_cohort_macro.yaml")
    parser.add_argument(
        "--fixture",
        default=(
            "data/processed/fabric_v2_real_dataset_atomic_introns_v1/prepared_dataset"
        ),
    )
    parser.add_argument(
        "--weight-table",
        default="data/processed/fabric_v2_compatible_ec_v1/G_fit.tsv",
        help="gene -> train mass map used only to stratify the gene sample",
    )
    parser.add_argument("--condition", default="full", choices=("full", "atac", "rbp"))
    parser.add_argument("--gene-count", type=int, default=150)
    parser.add_argument("--largest", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    threads = torch.get_num_threads()
    config = load_config(args.config)
    training = config["training"]
    prepared = BackedPreparedDataset.load(args.fixture)
    genes = prepared.genes

    table = pd.read_csv(args.weight_table, sep="\t").sort_values(
        "train_positive_informative_ec_mass"
    )
    ordered_ids = list(table["target_gene_id"].astype(str))
    position = {gene_id: i for i, (gene_id, _) in enumerate(genes.records)}
    largest = ordered_ids[-args.largest :]
    body = ordered_ids[: -args.largest]
    stride = max(1, len(body) // max(1, args.gene_count - args.largest))
    chosen = body[::stride][: args.gene_count - args.largest] + largest
    indices = [position[gene_id] for gene_id in chosen]

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    first = genes[indices[0]]
    model = FABRICV2Model(
        **_model_spec(first, config["model"]), readout_kind="path_context"
    ).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=float(config["optimizer"]["learning_rate"]),
        lambda_base=float(config["optimizer"]["lambda_base"]),
        lambda_int=float(config["optimizer"]["lambda_int"]),
    )
    condition = _MODEL_CONDITION[args.condition]
    clip = float(config["optimizer"]["gradient_clip_norm"])
    mass_cache: dict[str, float] = {}

    def one_pass(epoch: int) -> float:
        model.train()
        started = time.perf_counter()
        for order, index in enumerate(indices):
            gene = genes[index]
            rows = rows_for_split(gene, "train")
            optimizer.zero_grad(set_to_none=True)
            sample = sample_train_gene_cells_for_epoch(
                gene,
                rows,
                max_gene_cells=int(
                    training["max_train_gene_cells_per_gene_per_epoch"]
                ),
                seed=1103,
                epoch=epoch,
                gene_order_0based=order,
                gene_count=len(indices),
            )
            plan = _plan_gene_cell_batches(
                gene,
                sample.selected_cells,
                sample.selected_rows,
                model_config=config["model"],
                resources=config["resources"],
                phase="train",
            )
            for cell_batch in plan.batches:
                cell_mask = torch.isin(
                    gene.row_cell_index[sample.selected_rows], cell_batch
                )
                batch_rows = sample.selected_rows[cell_mask]
                batch_input, row_cell_index = _subset_gene_cells(
                    gene, cell_batch, batch_rows, model
                )
                details = compatible_path_nll(
                    model(batch_input, condition=condition).path_logits,
                    gene.compatible_path_indices[batch_rows].to(device),
                    gene.compatible_path_mask[batch_rows].to(device),
                    gene.molecule_count[batch_rows].to(device),
                    row_cell_index=row_cell_index,
                    return_details=True,
                )
                gene_mass = mass_cache.get(gene.gene_id)
                if gene_mass is None:
                    gene_mass = split_informative_molecule_mass((gene,), "train")
                    mass_cache[gene.gene_id] = gene_mass
                (
                    details.weighted_sum * sample.inclusion_multiplier / gene_mass
                ).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - started

    def load_average() -> float:
        return os.getloadavg()[0]

    warm = one_pass(1)
    print(
        f"threads={threads}\tgenes={len(indices)}\tpass=warm\t"
        f"seconds={warm:.2f}\tper_gene_ms={1000*warm/len(indices):.1f}\t"
        f"load1={load_average():.0f}",
        flush=True,
    )
    for repeat in range(args.passes):
        elapsed = one_pass(2 + repeat)
        print(
            f"threads={threads}\tgenes={len(indices)}\tpass={repeat + 1}\t"
            f"seconds={elapsed:.2f}\tper_gene_ms={1000*elapsed/len(indices):.1f}\t"
            f"load1={load_average():.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
