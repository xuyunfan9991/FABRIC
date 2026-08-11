"""Build the frozen ENSG00000275074 graph/EC fixture from normalized inputs.

This script reads only the imported Parquet tables named by the FABRIC V1
contract.  It does not import PRISM, scan BAM/GTF, or follow a mutable graph
pointer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads


GENE_ID = "ENSG00000275074"
GRAPH_GENERATION = Path(
    "/home2/xyf/project/PRISM/data/i6_multi7198_terminal_v2/artifacts/graph/"
    "generations/d51be2344bb0e5acde0e5cf9b8e5aded7b38c2666a639c836051f7b48f714bc4"
)
COMPATIBILITY_EC = Path(
    "/home2/xyf/project/PRISM/data/i6_multi7198_terminal_v2/supervision/.artifact/"
    "generations/c6c36a3a3144179de935816e3b531df0e3619fcd66d94292d360d5184abf3ab7/"
    "outputs/supervision/compatibility_equivalence_classes.parquet"
)
CELL_SPLIT = Path(
    "/home2/xyf/project/PRISM/data/final_multimodal_simple_12run_v1/D2/split_manifest/"
    "generations/87e623496acaa2799f4fdcbe6befbc965b70ca7324eada288d11d4c7965195d9/"
    "outputs/split_rows.parquet"
)
OUTPUT_ROOT = Path(__file__).parent / "fixtures" / "real"

NODE_COLUMNS = (
    "gene_id",
    "node_id",
    "node_type",
    "chrom",
    "strand",
    "pos_0based",
    "site_start_0based",
    "site_end_0based",
    "relative_gene_pos",
    "annotation_confidence",
    "site_prior_score",
)
EDGE_COLUMNS = (
    "gene_id",
    "edge_id",
    "edge_type",
    "src_node_id",
    "dst_node_id",
    "src_node_type",
    "dst_node_type",
    "chrom",
    "strand",
    "start_0based",
    "end_0based_exclusive",
    "span_bp",
    "length_bp",
    "relative_edge_pos",
    "annotation_confidence",
    "edge_prior_score",
)
PATH_COLUMNS = (
    "gene_id",
    "path_id",
    "transcript_id",
    "chrom",
    "strand",
    "tss_node_id",
    "pas_node_id",
    "n_edges",
    "path_length_bp",
)
PATH_EDGE_COLUMNS = (
    "gene_id",
    "path_id",
    "transcript_id",
    "edge_order",
    "edge_id",
    "edge_type",
    "src_node_id",
    "dst_node_id",
    "chrom",
    "strand",
)
EC_COLUMNS = (
    "cell_id",
    "gene_id",
    "compatibility_class_id",
    "compatible_path_ids",
    "compatible_path_ids_key",
    "compatible_path_count",
    "molecule_count",
    "mean_mapq",
    "mean_aligned_bp",
    "bucket_qc_flags",
    "split",
)


def _read_gene_parquet(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    table = pads.dataset(path, format="parquet").to_table(
        columns=list(columns), filter=pads.field("gene_id") == GENE_ID
    )
    return table.to_pandas()


def _canonical_split_cell_id(value: str) -> str:
    value = str(value)
    if not value.startswith("RNA__"):
        raise ValueError(
            f"current split cell lacks the exact RNA__ namespace: {value!r}"
        )
    canonical = value[5:]
    if not canonical:
        raise ValueError("current split contains an empty canonical RNA cell ID")
    return canonical


def build_fixture(output_root: Path = OUTPUT_ROOT) -> None:
    graph_root = GRAPH_GENERATION / "outputs" / "graph"
    nodes = _read_gene_parquet(graph_root / "node_table.parquet", NODE_COLUMNS)
    edges = _read_gene_parquet(graph_root / "edge_table.parquet", EDGE_COLUMNS)
    paths = _read_gene_parquet(graph_root / "path_table.parquet", PATH_COLUMNS)
    path_edges = _read_gene_parquet(
        graph_root / "path_edge_table.parquet", PATH_EDGE_COLUMNS
    )
    if (len(nodes), len(edges), len(paths), len(path_edges)) != (9, 10, 2, 14):
        raise ValueError("frozen source graph counts changed for ENSG00000275074")

    # SOURCE/SINK and START/END are import-boundary scaffolding, not FABRIC V1
    # processing objects.  Legal path incidence is the authority for retained
    # edges, so an unreferenced edge cannot enter the fixture graph.
    path_edges = path_edges.loc[
        ~path_edges["edge_type"].astype(str).isin({"START", "END"})
    ].copy()
    path_edges["edge_order"] = path_edges.groupby("path_id", sort=False).cumcount()
    used_edge_ids = set(path_edges["edge_id"].astype(str))
    edges = edges.loc[edges["edge_id"].astype(str).isin(used_edge_ids)].copy()
    if set(edges["edge_id"].astype(str)) != used_edge_ids:
        raise ValueError("a legal path edge is absent from the imported edge table")
    used_node_ids = set(path_edges["src_node_id"].astype(str)) | set(
        path_edges["dst_node_id"].astype(str)
    )
    nodes = nodes.loc[nodes["node_id"].astype(str).isin(used_node_ids)].copy()
    if set(nodes["node_id"].astype(str)) != used_node_ids:
        raise ValueError("a legal path endpoint is absent from the imported node table")
    edge_counts = path_edges.groupby("path_id", sort=False).size()
    paths["n_edges"] = paths["path_id"].astype(str).map(edge_counts).astype("int64")
    if (len(nodes), len(edges), len(paths), len(path_edges)) != (7, 7, 2, 10):
        raise ValueError(
            "normalized fixture graph counts differ from the frozen contract"
        )

    split = pd.read_parquet(CELL_SPLIT)
    split = split.copy()
    split["canonical_cell_id"] = split["cell_id"].map(_canonical_split_cell_id)
    if split["canonical_cell_id"].duplicated().any():
        raise ValueError("current split is not unique after exact RNA__ removal")
    ec = _read_gene_parquet(COMPATIBILITY_EC, EC_COLUMNS)
    if len(ec) != 97 or int(ec["molecule_count"].sum()) != 98:
        raise ValueError("frozen source EC counts changed for ENSG00000275074")
    ec = ec.rename(columns={"split": "source_ec_split"})
    bound = ec.merge(
        split[["canonical_cell_id", "split"]],
        left_on="cell_id",
        right_on="canonical_cell_id",
        how="inner",
        validate="many_to_one",
    ).drop(columns="canonical_cell_id")
    bound = bound.sort_values(
        ["cell_id", "compatible_path_ids_key"], kind="mergesort"
    ).reset_index(drop=True)
    if len(bound) != 66 or int(bound["molecule_count"].sum()) != 66:
        raise ValueError("current split coverage changed for the frozen fixture")
    split_counts = bound["split"].value_counts().to_dict()
    if split_counts != {"train": 56, "test": 5, "val": 5}:
        raise ValueError(f"current split counts changed: {split_counts}")
    split_mismatches = int((bound["source_ec_split"] != bound["split"]).sum())
    if split_mismatches != 10:
        raise ValueError("stale EC split mismatch count changed for the frozen fixture")
    path_ids = set(paths["path_id"].astype(str))
    if any(set(map(str, values)) - path_ids for values in bound["compatible_path_ids"]):
        raise ValueError("fixture EC references a path outside the selected graph")
    selected_split = split.loc[
        split["canonical_cell_id"].isin(set(bound["cell_id"].astype(str))),
        ["cell_id", "rna_embryo_id", "split"],
    ].sort_values("cell_id", kind="mergesort")
    if len(selected_split) != len(bound):
        raise ValueError("fixture split and EC cells are not one-to-one")

    graph_output = output_root / "graph_generation" / "outputs" / "graph"
    graph_output.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(graph_output / "node_table.parquet", index=False)
    edges.to_parquet(graph_output / "edge_table.parquet", index=False)
    paths.to_parquet(graph_output / "path_table.parquet", index=False)
    path_edges.to_parquet(graph_output / "path_edge_table.parquet", index=False)
    bound.to_parquet(
        output_root / "compatibility_equivalence_classes.parquet", index=False
    )
    selected_split.to_parquet(output_root / "split_rows.parquet", index=False)

    metadata = {
        "schema_version": "fabric.real-fixture.v1",
        "gene_id": GENE_ID,
        "source_graph_generation": str(GRAPH_GENERATION),
        "source_compatibility_ec": str(COMPATIBILITY_EC),
        "source_cell_split": str(CELL_SPLIT),
        "normalization": {
            "cell_id": "remove_exact_RNA__prefix_from_split_only",
            "graph": "drop_SOURCE_SINK_START_END_and_edges_not_in_path_edge",
            "split": "inner_join_EC_to_current_D2_split_then_replace_EC_split",
        },
        "counts": {
            "source_nodes": 9,
            "source_edges": 10,
            "source_paths": 2,
            "source_path_edges": 14,
            "nodes": 7,
            "edges": 7,
            "paths": 2,
            "path_edges": 10,
            "source_ec_rows": 97,
            "source_ec_molecules": 98,
            "ec_rows": 66,
            "ec_molecules": 66,
            "excluded_ec_rows_outside_current_split": 31,
            "stale_split_mismatch_rows": 10,
            "train_rows": 56,
            "val_rows": 5,
            "test_rows": 5,
        },
        "fixed_logit_likelihood_reference": {
            "path_logits": {
                "path:ENSG00000275074:ENST00000613958": 0.75,
                "path:ENSG00000275074:ENST00000611621": -0.25,
            },
            "row_nll": {
                "path:ENSG00000275074:ENST00000613958": 0.3132616875182228,
                "path:ENSG00000275074:ENST00000611621": 1.3132616875182228,
            },
            "molecule_weighted_mean_nll": 1.025382899639435,
        },
    }
    (output_root / "fixture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    build_fixture()
