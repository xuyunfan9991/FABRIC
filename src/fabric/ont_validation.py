"""Build the validation-only ONT count target for the one epoch KL diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
from scipy import sparse
from scipy.io import mmread

from .evaluate import OntMatrixKlTarget


SCOPE_POLICY = (
    "likelihood_informative_validation_cell_gene_with_at_least_two_"
    "positive_ont_paths"
)


def _expected_validation_cell_gene_axis(
    compatible_ec_root: Path,
    *,
    gene_order: dict[str, int],
    cell_order: dict[str, int],
) -> pd.DataFrame:
    """Freeze the exact likelihood-informative validation prediction scope."""

    dataset = pads.dataset(compatible_ec_root / "compatible_ec", format="parquet")
    rows = dataset.to_table(
        columns=["cell_id", "target_gene_id"],
        filter=(pads.field("split") == "val")
        & (pads.field("final_fate") == "likelihood_informative"),
    )
    rows = rows.filter(
        pc.is_in(rows["target_gene_id"], value_set=pa.array(list(gene_order)))
    )
    unique = rows.group_by(["target_gene_id", "cell_id"]).aggregate([]).to_pandas()
    unique["gene_order_0based"] = unique["target_gene_id"].astype(str).map(gene_order)
    unique["cell_order_0based"] = unique["cell_id"].astype(str).map(cell_order)
    if unique.empty or unique[["gene_order_0based", "cell_order_0based"]].isna().any().any():
        raise ValueError(
            "compatible EC validation cell-gene scope differs from G_fit/validation axes"
        )
    unique = unique.sort_values(
        ["gene_order_0based", "cell_order_0based"], kind="mergesort"
    ).reset_index(drop=True)
    unique.insert(
        0, "expected_instance_order_0based", np.arange(len(unique), dtype=np.int64)
    )
    return unique[
        [
            "expected_instance_order_0based",
            "cell_id",
            "target_gene_id",
            "gene_order_0based",
            "cell_order_0based",
        ]
    ]


def build_ont_validation_kl_target(
    *,
    matrix_path: str | Path,
    matrix_cell_index_path: str | Path,
    legal_paths_path: str | Path,
    g_fit_path: str | Path,
    compatible_ec_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Materialize G_fit paths × validation cells; never select or store test counts."""

    matrix_path = Path(matrix_path)
    matrix_cell_index_path = Path(matrix_cell_index_path)
    legal_paths_path = Path(legal_paths_path)
    g_fit_path = Path(g_fit_path)
    compatible_ec_root = Path(compatible_ec_root)
    output_root = Path(output_root)

    cells = pd.read_parquet(matrix_cell_index_path)
    required_cells = {"matrix_column_0based", "cell_id", "split"}
    if not required_cells.issubset(cells.columns):
        raise ValueError("matrix cell index misses validation target columns")
    if (
        cells["cell_id"].duplicated().any()
        or cells["matrix_column_0based"].duplicated().any()
    ):
        raise ValueError("matrix cell index contains duplicate cell/column identities")
    validation_cells = (
        cells.loc[cells["split"].astype(str).eq("val")]
        .sort_values("matrix_column_0based", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(validation_cells) != 21_788:
        raise ValueError("ONT KL target requires exactly 21,788 validation cells")
    if cells["split"].astype(str).eq("test").sum() != 21_788:
        raise ValueError("frozen split identity differs before validation target build")

    g_fit = pd.read_csv(g_fit_path, sep="\t")
    if "target_gene_id" not in g_fit or g_fit["target_gene_id"].duplicated().any():
        raise ValueError("G_fit axis is missing or duplicated")
    gene_ids = g_fit["target_gene_id"].astype(str).tolist()
    if len(gene_ids) != 17_600:
        raise ValueError("ONT KL target requires exactly 17,600 G_fit genes")
    gene_order = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    cell_order = {
        str(cell_id): index
        for index, cell_id in enumerate(validation_cells["cell_id"].astype(str))
    }
    expected_axis = _expected_validation_cell_gene_axis(
        compatible_ec_root,
        gene_order=gene_order,
        cell_order=cell_order,
    )

    paths = pd.read_parquet(legal_paths_path)
    required_paths = {
        "matrix_row_0based",
        "gene_id",
        "path_id",
        "path_order_0based",
    }
    if not required_paths.issubset(paths.columns):
        raise ValueError("legal path table misses ONT KL target columns")
    paths = paths.loc[paths["gene_id"].astype(str).isin(gene_order)].copy()
    paths["gene_id"] = paths["gene_id"].astype(str)
    paths["path_id"] = paths["path_id"].astype(str)
    paths["g_fit_order_0based"] = paths["gene_id"].map(gene_order)
    paths = paths.sort_values(
        ["g_fit_order_0based", "path_order_0based"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        paths["path_id"].duplicated().any()
        or paths["matrix_row_0based"].duplicated().any()
        or paths["gene_id"].nunique() != 17_600
    ):
        raise ValueError("G_fit ONT path identity is incomplete or duplicated")
    paths["path_order_global_0based"] = np.arange(len(paths), dtype=np.int64)

    matrix = mmread(matrix_path)
    if not sparse.issparse(matrix):
        raise TypeError("ONT MatrixMarket target source must be sparse")
    if matrix.shape != (101_067, 217_933):
        raise ValueError("ONT MatrixMarket shape differs from the frozen source")
    matrix = matrix.tocsr()
    if (
        not np.isfinite(matrix.data).all()
        or bool((matrix.data < 0).any())
        or not np.equal(matrix.data, np.floor(matrix.data)).all()
    ):
        raise ValueError("ONT MatrixMarket values must be non-negative integers")
    counts = matrix[paths["matrix_row_0based"].to_numpy(dtype=np.int64)][
        :, validation_cells["matrix_column_0based"].to_numpy(dtype=np.int64)
    ].tocsr()
    del matrix

    path_axis = paths[
        [
            "path_order_global_0based",
            "gene_id",
            "g_fit_order_0based",
            "path_order_0based",
            "path_id",
            "matrix_row_0based",
        ]
    ].rename(columns={"path_order_global_0based": "path_order_0based_global"})
    path_axis = path_axis.rename(
        columns={
            "path_order_0based_global": "path_order_0based",
            "path_order_0based": "gene_path_order_0based",
        }
    )
    cell_axis = validation_cells[["matrix_column_0based", "cell_id", "split"]].copy()
    cell_axis.insert(0, "cell_order_0based", np.arange(len(cell_axis), dtype=np.int64))

    output_root.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_root / "validation_ont_counts.npz", counts, compressed=True)
    path_axis.to_parquet(output_root / "path_axis.parquet", index=False)
    cell_axis.to_parquet(output_root / "cell_axis.parquet", index=False)
    expected_axis.to_parquet(
        output_root / "expected_cell_gene_axis.parquet", index=False
    )
    manifest: dict[str, object] = {
        "schema_version": "fabric.ont_validation_kl_target.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "ont_matrix_kl_count_weighted",
        "scope_policy": SCOPE_POLICY,
        "matrix_identity": "tx_matrix_ONT:101067x217933:raw_integer_counts",
        "path_identity": "fabric_v2_compatible_ec_v1:G_fit_ordered_paths",
        "split_identity": "embryo_stratified_cell_80_10_10_v1:seed=20260725:val",
        "g_fit_gene_count": 17_600,
        "path_count": len(path_axis),
        "validation_cell_count": len(cell_axis),
        "expected_cell_gene_count": len(expected_axis),
        "counts_shape": list(counts.shape),
        "counts_nnz": int(counts.nnz),
        "counts_total": int(counts.sum()),
        "counts": "validation_ont_counts.npz",
        "path_axis": "path_axis.parquet",
        "cell_axis": "cell_axis.parquet",
        "expected_cell_gene_axis": "expected_cell_gene_axis.parquet",
        "test_cells_or_counts_included": False,
        "training_started": False,
        "optimizer_step_called": False,
        "test_predictions_or_metrics_computed": False,
    }
    (output_root / "OntMatrixKlTargetManifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    OntMatrixKlTarget.load(output_root)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--matrix-cell-index", required=True)
    parser.add_argument("--legal-paths", required=True)
    parser.add_argument("--g-fit", required=True)
    parser.add_argument("--compatible-ec-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    manifest = build_ont_validation_kl_target(
        matrix_path=args.matrix,
        matrix_cell_index_path=args.matrix_cell_index,
        legal_paths_path=args.legal_paths,
        g_fit_path=args.g_fit,
        compatible_ec_root=args.compatible_ec_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
