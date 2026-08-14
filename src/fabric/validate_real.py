"""Strict identity and tensor validation for the real backed V2 dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .real_prepared import CHROMOSOMES
from .train import BackedPreparedDataset, _input_dimensions, _validate_genes


def _finite_tensor(value: torch.Tensor) -> bool:
    checked = (
        value
        if value.layout == torch.strided
        else value.to_sparse_coo().coalesce().values()
    )
    return bool(torch.isfinite(checked).all().item())


def validate_real_dataset(
    real_root: str | Path, compatible_root: str | Path
) -> dict[str, object]:
    root = Path(real_root)
    compatible = Path(compatible_root)
    prepared_root = root / "prepared_dataset"
    prepared = BackedPreparedDataset.load(prepared_root)
    prepared_manifest = json.loads(
        (prepared_root / "PreparedDatasetManifest.json").read_text()
    )
    source_validation = json.loads((root / "SourceValidation.json").read_text())
    split_rows = pd.read_parquet(
        source_validation["sources"]["split_rows"], columns=["cell_id", "split"]
    )
    cell_metadata = pd.read_parquet(
        root / "cell_context" / "cell_metadata.parquet",
        columns=["cell_id", "split"],
    )
    if not split_rows.astype(str).equals(cell_metadata.astype(str)):
        raise ValueError("real cell metadata identity/order differs from the frozen split")
    split_counts = split_rows["split"].astype(str).value_counts().to_dict()
    if split_counts != {"train": 174_357, "val": 21_788, "test": 21_788}:
        raise ValueError("frozen 217,933-cell train/validation/test split counts drifted")
    split_by_cell = split_rows.set_index("cell_id")["split"].astype(str)
    test_compatible_rows_found = 0
    for path in sorted((compatible / "compatible_ec").glob("part-chr*.parquet")):
        test_compatible_rows_found += len(
            pd.read_parquet(path, filters=[("split", "==", "test")], columns=["split"])
        )
    if test_compatible_rows_found:
        raise ValueError("compatible-EC artifact physically contains forbidden test rows")
    frozen_g_fit = tuple(
        pd.read_csv(compatible / "G_fit.tsv", sep="\t")["target_gene_id"].astype(str)
    )
    if prepared.informative_gene_ids != frozen_g_fit or len(frozen_g_fit) != 17_600:
        raise ValueError("backed dataset gene axis differs from frozen 17,600-gene G_fit")
    legal = pd.read_parquet(
        compatible / "legal_structural_paths.parquet",
        columns=["gene_id", "path_id"],
    )
    graph_paths = pd.read_parquet(
        root / "graph" / "path_table.parquet", columns=["gene_id", "path_id"]
    )
    if not graph_paths[["gene_id", "path_id"]].astype(str).equals(
        legal[["gene_id", "path_id"]].astype(str)
    ):
        raise ValueError("real graph/path identity or order differs from frozen legal paths")
    if len(graph_paths) != 90_672 or graph_paths["gene_id"].nunique() != 17_706:
        raise ValueError("real graph does not cover 17,706 genes and 90,672 paths")
    graph_audit = pd.read_parquet(root / "graph" / "candidate_graph_fit_audit.parquet")
    graph_only = graph_audit.loc[
        graph_audit["support_status"].astype(str).eq(
            "graph_only_zero_train_informative_mass"
        ),
        "target_gene_id",
    ].astype(str)
    if len(graph_only) != 106 or set(graph_only) & set(frozen_g_fit):
        raise ValueError("106-gene graph-only audit differs from G_fit complement")
    gate_gene_ids = []
    for chromosome in CHROMOSOMES:
        event_manifest = root / "events" / "shard_manifests" / f"{chromosome}.json"
        gate_manifest = root / "events" / "gated" / "shard_manifests" / f"{chromosome}.json"
        if not event_manifest.is_file() or not gate_manifest.is_file():
            raise FileNotFoundError(f"event/gate manifest missing for {chromosome}")
        gate_record = json.loads(gate_manifest.read_text())
        if gate_record["test_rows_or_test_statistics_used"] is not False:
            raise ValueError("gate manifest reports test exposure")
        gate_gene_ids.extend(
            str(value["gene_id"]) for value in gate_record["gene_records"]
        )
    if set(gate_gene_ids) != set(frozen_g_fit) or len(gate_gene_ids) != len(frozen_g_fit):
        raise ValueError("gate gene identities differ from complete G_fit")
    gate_tensor_gene_ids = [
        path.stem for path in (root / "gates").glob("chr*/*.pt")
    ]
    if (
        len(gate_tensor_gene_ids) != len(set(gate_tensor_gene_ids))
        or set(gate_tensor_gene_ids) != set(frozen_g_fit)
    ):
        raise ValueError("production gate tensor files differ from the exact G_fit axis")
    interaction_manifest = json.loads(
        (root / "design" / "InteractionSupportManifest.json").read_text()
    )
    for modality in ("DNA", "RNA"):
        audit = interaction_manifest["modalities"][modality]["combined_rank_audit"]
        if not isinstance(audit, dict) or audit.get("status") != "PASS":
            raise ValueError(f"{modality} combined route design rank is not admitted")
    train_only_records = (
        json.loads((root / "cis" / "CISManifest.json").read_text()),
        json.loads((root / "cell_context" / "ATACMappingManifest.json").read_text()),
        json.loads((root / "cell_context" / "RNAActivityManifest.json").read_text()),
        json.loads((root / "design" / "RawInteractionSupportManifest.json").read_text()),
        interaction_manifest,
        json.loads((root / "design" / "BaseDesignManifest.json").read_text()),
    )
    if (
        train_only_records[0].get("test_rows_or_test_outcomes_read") is not False
        or train_only_records[1].get("target_test_cell_count") != 0
        or train_only_records[1].get("test_compatible_rows_or_predictions_read") is not False
        or train_only_records[2].get("test_cells_or_compatible_rows_read") is not False
        or train_only_records[3].get("test_rows_or_test_statistics_used") is not False
        or train_only_records[4].get("test_rows_or_test_statistics_used") is not False
        or train_only_records[4].get("validation_test_may_activate_columns") is not False
        or train_only_records[5].get("test_rows_or_test_outcomes_read") is not False
    ):
        raise ValueError("one or more train-only processing manifests report held-out exposure")

    dimensions = None
    train_mass = 0
    validation_mass = 0
    ec_rows = 0
    cell_gene_instances = 0
    total_paths = 0
    total_edges = 0
    for index, gene in enumerate(prepared.genes):
        if gene.gene_id != frozen_g_fit[index]:
            raise ValueError("backed gene shard order differs from G_fit")
        _validate_genes((gene,))
        current_dimensions = _input_dimensions(gene)
        if dimensions is None:
            dimensions = current_dimensions
        elif current_dimensions != dimensions:
            raise ValueError("prepared genes do not share frozen feature dimensions")
        if set(gene.cell_split) - {"train", "val"}:
            raise ValueError(f"prepared shard contains test cells: {gene.gene_id}")
        expected_gene_splits = tuple(
            split_by_cell.loc[list(gene.cell_ids)].astype(str)
        )
        if tuple(gene.cell_split) != expected_gene_splits:
            raise ValueError(f"prepared gene cell/split identity differs: {gene.gene_id}")
        for tensor in (
            gene.model_input.cis_features,
            gene.model_input.dna.route_base_features,
            gene.model_input.dna.route_interaction_features,
            gene.model_input.dna.gate,
            gene.model_input.rna.route_base_features,
            gene.model_input.rna.route_interaction_features,
            gene.model_input.rna.gate,
            gene.molecule_count,
        ):
            if not _finite_tensor(tensor):
                raise ValueError(f"prepared tensor is non-finite: {gene.gene_id}")
        # NLL weights are deliberately stored as float32, but their source
        # contract is positive integer molecule multiplicity.  Validate the
        # represented values rather than conflating that contract with the
        # runtime tensor dtype.
        if not bool((gene.molecule_count > 0).all()) or not torch.equal(
            gene.molecule_count, gene.molecule_count.round()
        ):
            raise ValueError(
                f"prepared molecule mass is not positive integral-valued: {gene.gene_id}"
            )
        # _validate_genes checks the left-aligned mask and strictly increasing
        # in-range compatible indices with vectorized tensor operations.  Do
        # not reintroduce a Python loop over ~120 million real EC rows here.
        row_splits = np.asarray(gene.cell_split, dtype=object)[
            gene.row_cell_index.numpy()
        ]
        mass = gene.molecule_count.numpy().astype(np.int64)
        train_mass += int(mass[row_splits == "train"].sum())
        validation_mass += int(mass[row_splits == "val"].sum())
        ec_rows += len(mass)
        cell_gene_instances += len(gene.cell_ids)
        total_paths += len(gene.path_ids)
        total_edges += gene.model_input.cis_features.shape[0]
    upstream = json.loads((compatible / "CompatibilityArtifactManifest.json").read_text())
    if (
        upstream.get("run_counts", {}).get(
            "duplicate_cell_gene_umi_primary_records"
        )
        != 0
    ):
        raise ValueError("upstream compatible artifact contains duplicate cell-gene-UMIs")
    candidate_expected = {
        str(row["split"]): int(row["proper_subset_compatible_molecule_mass"])
        for row in upstream["split_conservation"]
    }
    expected = {
        "train": int(
            prepared_manifest["expected_train_informative_molecule_mass"]
        ),
        "val": int(
            prepared_manifest["expected_validation_informative_molecule_mass"]
        ),
    }
    graph_only_mass = {
        "train": int(
            prepared_manifest["graph_only_train_informative_molecule_mass"]
        ),
        "val": int(
            prepared_manifest["graph_only_validation_informative_molecule_mass"]
        ),
    }
    if graph_only_mass["train"] != 0 or any(
        expected[split] + graph_only_mass[split] != candidate_expected[split]
        for split in ("train", "val")
    ):
        raise ValueError("G_fit and graph-only split mass do not close upstream conservation")
    if train_mass != expected["train"] or validation_mass != expected["val"]:
        raise ValueError("prepared split molecule mass differs from upstream conservation")
    if upstream["test_rows_written"] is not False or upstream["artifact_complete"] is not True:
        raise ValueError("upstream compatible artifact is incomplete or exposes test rows")
    return {
        "schema_version": "fabric.real_dataset_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ADMITTED_FOR_PRELAUNCH",
        "candidate_gene_count": 17_706,
        "g_fit_gene_count": 17_600,
        "graph_only_gene_count": 106,
        "graph_path_count": len(graph_paths),
        "prepared_g_fit_path_count": total_paths,
        "prepared_g_fit_edge_count": total_edges,
        "prepared_ec_row_count": ec_rows,
        "prepared_cell_gene_instance_count": cell_gene_instances,
        "train_informative_molecule_mass": train_mass,
        "validation_informative_molecule_mass": validation_mass,
        "graph_only_train_informative_molecule_mass": graph_only_mass["train"],
        "graph_only_validation_informative_molecule_mass": graph_only_mass["val"],
        "test_compatible_row_count": 0,
        "frozen_cell_count": len(split_rows),
        "frozen_split_counts": split_counts,
        "cell_and_split_identity_valid": True,
        "shared_model_dimensions": list(dimensions or ()),
        "ordered_compatible_sets_valid": True,
        "integer_molecule_mass_conserved": True,
        "all_model_tensors_finite": True,
        "train_only_gate_and_interaction_support": True,
        "historical_7198_graph_or_ec_used": False,
        "historical_167235_split_used": False,
        "test_predictions_or_metrics_computed": False,
        "training_started": False,
        "final_test_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--compatible-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    record = validate_real_dataset(args.real_root, args.compatible_root)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
