"""Compute ONT top-1 accuracy for one or more per-epoch checkpoints.

The per-epoch monitor only records validation NLL and ONT matrix KL.  KL alone
cannot distinguish "the model picks the right dominant isoform but is
over-confident" (a calibration issue) from "the model picks the wrong isoform"
(a real objective problem), because both raise KL.  Top-1 accuracy separates them.

Everything top-1 needs is already available inside the KL loop: for each
validation cell-gene we have the ONT counts per path and the model logits per
path, on the same path axis.  So this re-runs one validation forward pass per
checkpoint and, in the same sweep, recomputes KL (as a correctness check against
history.tsv) alongside the top-1 statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric.evaluate import OntMatrixKlTarget  # noqa: E402
from fabric.likelihood import compatible_path_nll  # noqa: E402
from fabric.model import FABRICV2Model  # noqa: E402
from fabric.train import (  # noqa: E402
    _MODEL_CONDITION,
    BackedPreparedDataset,
    _evaluate_split,
    _model_spec,
    load_config,
)


def ont_top1_and_kl(snapshot, target: OntMatrixKlTarget):
    """Mirror compute_validation_ont_matrix_kl, additionally scoring top-1.

    Eligibility matches the KL metric exactly: a validation cell-gene counts only
    when it has non-zero ONT mass on at least two distinct paths, so the
    denominators here are directly comparable to the reported KL.
    """
    path_position = {value: index for index, value in enumerate(target.path_ids)}
    cell_position = {value: index for index, value in enumerate(target.cell_ids)}
    gene_paths: dict[str, tuple[str, ...]] = {}
    for gene_id, path_id in zip(target.path_gene_ids, target.path_ids, strict=True):
        gene_paths.setdefault(gene_id, tuple())
        gene_paths[gene_id] = (*gene_paths[gene_id], path_id)

    per_gene: dict[str, dict[str, float]] = {}
    weighted_kl_sum = 0.0
    eligible_count_mass = 0
    eligible_cell_genes = 0
    top1_hits = 0
    top1_tie_aware_hits = 0
    top1_count_weighted_hits = 0
    unique_top1_hits = 0
    unique_top1_total = 0
    observed_tie_cell_genes = 0

    for prediction in snapshot.predictions:
        gene_id = str(getattr(prediction, "gene_id"))
        path_ids = tuple(map(str, getattr(prediction, "path_ids")))
        if gene_id not in gene_paths or path_ids != gene_paths[gene_id]:
            raise ValueError(f"path axis differs for gene {gene_id}")
        cell_ids = tuple(map(str, getattr(prediction, "cell_ids")))
        logits = (
            getattr(prediction, "path_logits")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        row_index = [path_position[value] for value in path_ids]
        column_index = [cell_position[value] for value in cell_ids]
        counts = (
            target.counts[row_index][:, column_index]
            .toarray()
            .T.astype(np.float64, copy=False)
        )

        gene_record = per_gene.setdefault(
            gene_id,
            {
                "ont_eligible_cell_genes": 0,
                "ont_count_mass": 0,
                "top1_hits": 0,
                "top1_count_weighted_hits": 0,
                "tie_aware_hits": 0,
                "kl_weighted_sum": 0.0,
            },
        )

        for cell_counts, cell_logits in zip(counts, logits, strict=True):
            total = int(cell_counts.sum())
            if total == 0 or int(np.count_nonzero(cell_counts)) < 2:
                continue

            q = cell_counts / total
            positive = q > 0
            log_probability = cell_logits - logsumexp(cell_logits)
            kl = float(
                np.sum(q[positive] * (np.log(q[positive]) - log_probability[positive]))
            )
            weighted_kl_sum += total * max(0.0, kl)
            eligible_count_mass += total
            eligible_cell_genes += 1
            gene_record["ont_eligible_cell_genes"] += 1
            gene_record["ont_count_mass"] += total
            gene_record["kl_weighted_sum"] += total * max(0.0, kl)

            # Ties are broken by lowest path index on both sides, matching
            # compute_ont_matrix_agreement's `min(...)` over tied identifiers.
            observed_top = np.flatnonzero(cell_counts == cell_counts.max())
            predicted_winner = int(np.argmax(cell_logits))
            observed_winner = int(observed_top[0])

            hit = predicted_winner == observed_winner
            top1_hits += int(hit)
            top1_count_weighted_hits += total * int(hit)
            tie_aware = int(predicted_winner in set(observed_top.tolist()))
            top1_tie_aware_hits += tie_aware
            gene_record["top1_hits"] += int(hit)
            gene_record["top1_count_weighted_hits"] += total * int(hit)
            gene_record["tie_aware_hits"] += tie_aware
            if observed_top.size == 1:
                unique_top1_total += 1
                unique_top1_hits += int(hit)
            else:
                observed_tie_cell_genes += 1

    if eligible_cell_genes == 0:
        raise ValueError("no eligible validation cell-genes")

    aggregate = {
        "ont_matrix_kl_count_weighted": weighted_kl_sum / eligible_count_mass,
        "ont_top1_acc_macro": top1_hits / eligible_cell_genes,
        "ont_top1_acc_count_weighted": top1_count_weighted_hits / eligible_count_mass,
        "ont_top1_acc_tie_aware": top1_tie_aware_hits / eligible_cell_genes,
        "ont_unique_top1_acc": (
            unique_top1_hits / unique_top1_total if unique_top1_total else float("nan")
        ),
        "eligible_cell_gene_count": eligible_cell_genes,
        "eligible_count_mass": eligible_count_mass,
        "observed_tie_cell_gene_count": observed_tie_cell_genes,
    }
    return aggregate, per_gene


def per_gene_nll(snapshot) -> dict[str, tuple[float, float]]:
    """Decompose the split NLL into per-gene (weighted_sum, molecule_mass)."""
    out: dict[str, tuple[float, float]] = {}
    for prediction in snapshot.predictions:
        details = compatible_path_nll(
            prediction.path_logits,
            prediction.compatible_path_indices,
            prediction.compatible_path_mask,
            prediction.molecule_count,
            row_cell_index=prediction.row_cell_index,
            return_details=True,
        )
        out[str(prediction.gene_id)] = (
            float(details.weighted_sum),
            float(details.molecule_mass),
        )
    return out


PER_GENE_COLUMNS = (
    "gene_id",
    "nll_weighted_sum",
    "nll_molecule_mass",
    "ont_eligible_cell_genes",
    "ont_count_mass",
    "top1_hits",
    "top1_count_weighted_hits",
    "tie_aware_hits",
    "kl_weighted_sum",
)


def write_per_gene(path: Path, per_gene: dict, nll_by_gene: dict) -> None:
    with path.open("w") as handle:
        handle.write("\t".join(PER_GENE_COLUMNS) + "\n")
        for gene_id in sorted(set(per_gene) | set(nll_by_gene)):
            stats = per_gene.get(gene_id, {})
            weighted_sum, mass = nll_by_gene.get(gene_id, (float("nan"), 0.0))
            handle.write(
                "\t".join(
                    str(value)
                    for value in (
                        gene_id,
                        weighted_sum,
                        mass,
                        stats.get("ont_eligible_cell_genes", 0),
                        stats.get("ont_count_mass", 0),
                        stats.get("top1_hits", 0),
                        stats.get("top1_count_weighted_hits", 0),
                        stats.get("tie_aware_hits", 0),
                        stats.get("kl_weighted_sum", 0.0),
                    )
                )
                + "\n"
            )


def history_kl(run_dir: Path, epoch: int) -> float | None:
    with (run_dir / "history.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["epoch"]) == epoch:
                return float(row["ont_matrix_kl_count_weighted"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("full", "atac", "rbp"))
    parser.add_argument("--config", default="configs/fabric_v2_full_cohort.yaml")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", help="comma-separated subset, e.g. 18,24,30")
    args = parser.parse_args()

    config = load_config(args.config)
    snapshot_dir = Path(args.snapshot_dir)
    run_dir = Path(args.run_dir)
    out_path = Path(args.out)

    available = {}
    for path in snapshot_dir.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
        if match:
            available[int(match.group(1))] = path
    if args.epochs:
        wanted = [int(v) for v in args.epochs.split(",")]
        available = {e: p for e, p in available.items() if e in wanted}
    if not available:
        raise SystemExit(f"no checkpoints found under {snapshot_dir}")

    print(f"loading prepared dataset from {args.fixture}", flush=True)
    prepared = BackedPreparedDataset.load(args.fixture)
    genes = prepared.genes
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")

    target = OntMatrixKlTarget.load(config["monitor"]["target_root"])
    model = FABRICV2Model(
        **_model_spec(genes[0], config["model"]), readout_kind="path_context"
    ).to(device)
    model_condition = _MODEL_CONDITION[args.condition]

    done = {}
    if out_path.exists():
        with out_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                done[int(record["epoch"])] = record
        print(f"resuming: {len(done)} epochs already scored", flush=True)

    for epoch in sorted(available):
        if epoch in done:
            continue
        started = time.time()
        checkpoint = torch.load(available[epoch], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        snapshot = _evaluate_split(
            genes,
            model,
            condition=model_condition,
            split="val",
            model_config=config["model"],
            resources=config["resources"],
        )
        record, per_gene = ont_top1_and_kl(snapshot, target)
        nll_by_gene = per_gene_nll(snapshot)
        per_gene_path = out_path.parent / (
            f"per_gene_{args.condition}_e{epoch}.tsv"
        )
        write_per_gene(per_gene_path, per_gene, nll_by_gene)
        record["per_gene_path"] = str(per_gene_path)
        record["per_gene_count"] = len(nll_by_gene)
        record["epoch"] = epoch
        record["condition"] = args.condition
        record["validation_compatible_path_nll"] = float(snapshot.nll)
        record["elapsed_seconds"] = round(time.time() - started, 1)

        reference = history_kl(run_dir, epoch)
        record["history_kl"] = reference
        record["kl_matches_history"] = (
            reference is not None
            and abs(reference - record["ont_matrix_kl_count_weighted"]) < 1e-9
        )

        with out_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"e{epoch:<3} top1={record['ont_top1_acc_macro']:.4f}  "
            f"cw={record['ont_top1_acc_count_weighted']:.4f}  "
            f"tie_aware={record['ont_top1_acc_tie_aware']:.4f}  "
            f"kl={record['ont_matrix_kl_count_weighted']:.4f}  "
            f"kl_check={'OK' if record['kl_matches_history'] else 'MISMATCH'}  "
            f"({record['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        del snapshot
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
