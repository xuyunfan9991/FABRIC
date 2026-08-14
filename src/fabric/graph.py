"""Imported processing graphs, legal paths, and sparse path incidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from scipy import sparse

# SOURCE/SINK and START/END exist in the imported tables only as serialization
# sentinels.  FABRIC's scientific graph starts at TSS, ends at PAS, and counts
# only real processing transitions in M_edge and ell_p.
NODE_TYPES = ("TSS", "donor", "acceptor", "PAS")
EDGE_TYPES = ("EXON_CONTINUATION", "SPLICE", "RETAINED_INTRON")
IMPORTED_NODE_TYPES = ("SOURCE", *NODE_TYPES, "SINK")
IMPORTED_EDGE_TYPES = ("START", *EDGE_TYPES, "END")


def canonical_rna_cell_id(value: str) -> str:
    """Remove only the documented RNA namespace at the graph import boundary."""

    cell_id = str(value)
    if cell_id.startswith("RNA__"):
        cell_id = cell_id[5:]
    if not cell_id:
        raise ValueError("RNA cell_id must be non-empty")
    return cell_id


def load_split_rows(path: str | Path) -> pd.DataFrame:
    """Load the frozen RNA split and fail on namespace-induced collisions."""

    rows = pd.read_parquet(path, columns=["cell_id", "rna_embryo_id", "split"])
    rows = rows.copy()
    rows["source_split_cell_id"] = rows["cell_id"].astype(str)
    rows["cell_id"] = rows["source_split_cell_id"].map(canonical_rna_cell_id)
    if rows["cell_id"].duplicated().any():
        raise ValueError("cell split contains duplicate canonical cell_id values")
    allowed = {"train", "val", "test"}
    observed = set(rows["split"].astype(str))
    if observed != allowed:
        raise ValueError(
            f"cell split labels differ from {sorted(allowed)}: {sorted(observed)}"
        )
    return rows


@dataclass(frozen=True)
class GraphTables:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    paths: pd.DataFrame
    path_edges: pd.DataFrame


@dataclass(frozen=True)
class GeneGraph:
    gene_id: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    paths: pd.DataFrame
    path_edges: pd.DataFrame
    edge_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    transcript_aliases: tuple[tuple[str, ...], ...]
    path_edge_rows: tuple[tuple[int, ...], ...]
    path_node_rows: tuple[tuple[str, ...], ...]
    path_first_edge_indices: tuple[int, ...]
    path_last_edge_indices: tuple[int, ...]
    path_log_edge_count: tuple[float, ...]
    path_edge_incidence: sparse.csr_matrix
    local_edge_index: np.ndarray
    edge_features: np.ndarray
    variable_edge_count: int
    centered_path_incidence_energy: float


def load_graph_tables(
    graph_generation: str | Path, *, gene_ids: Sequence[str] | None = None
) -> GraphTables:
    """Read only the four normalized graph tables, optionally filtered by gene."""

    root = Path(graph_generation) / "outputs" / "graph"
    expression = None
    if gene_ids is not None:
        selected = [str(value) for value in gene_ids]
        if not selected:
            raise ValueError("gene_ids must be non-empty when supplied")
        expression = pads.field("gene_id").isin(selected)

    def read(name: str, columns: Sequence[str]) -> pd.DataFrame:
        table = pads.dataset(root / name, format="parquet").to_table(
            columns=list(columns), filter=expression
        )
        return table.to_pandas()

    nodes = read(
        "node_table.parquet",
        (
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
        ),
    )
    edges = read(
        "edge_table.parquet",
        (
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
        ),
    )
    paths = read(
        "path_table.parquet",
        (
            "gene_id",
            "path_id",
            "transcript_id",
            "chrom",
            "strand",
            "tss_node_id",
            "pas_node_id",
            "n_edges",
            "path_length_bp",
        ),
    )
    path_edges = read(
        "path_edge_table.parquet",
        (
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
        ),
    )
    return GraphTables(nodes=nodes, edges=edges, paths=paths, path_edges=path_edges)


def split_gene_graphs(tables: GraphTables) -> Iterator[GeneGraph]:
    genes = tables.paths["gene_id"].astype(str).drop_duplicates().tolist()
    nodes_by_gene = tables.nodes.groupby(
        tables.nodes["gene_id"].astype(str), sort=False
    )
    edges_by_gene = tables.edges.groupby(
        tables.edges["gene_id"].astype(str), sort=False
    )
    paths_by_gene = tables.paths.groupby(
        tables.paths["gene_id"].astype(str), sort=False
    )
    path_edges_by_gene = tables.path_edges.groupby(
        tables.path_edges["gene_id"].astype(str), sort=False
    )
    for gene_id in genes:
        yield build_gene_graph(
            gene_id,
            nodes=nodes_by_gene.get_group(gene_id),
            edges=edges_by_gene.get_group(gene_id),
            paths=paths_by_gene.get_group(gene_id),
            path_edges=path_edges_by_gene.get_group(gene_id),
        )


def build_gene_graph(
    gene_id: str,
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    paths: pd.DataFrame,
    path_edges: pd.DataFrame,
) -> GeneGraph:
    """Validate one imported gene and construct sparse, order-preserving tensors."""

    gene_id = str(gene_id)
    for label, frame in (
        ("node", nodes),
        ("edge", edges),
        ("path", paths),
        ("path_edge", path_edges),
    ):
        observed = set(frame["gene_id"].astype(str))
        if observed != {gene_id}:
            raise ValueError(f"{label} rows do not belong exactly to gene {gene_id}")
    nodes = nodes.reset_index(drop=True).copy()
    edges = edges.reset_index(drop=True).copy()
    paths = paths.reset_index(drop=True).copy()
    path_edges = path_edges.reset_index(drop=True).copy()
    _require_unique(nodes, "node_id", "node")
    _require_unique(edges, "edge_id", "edge")
    _require_unique(paths, "path_id", "path")
    _require_unique(paths, "transcript_id", "transcript")
    declared_path_ids = set(paths["path_id"].astype(str))
    observed_path_edge_ids = set(path_edges["path_id"].astype(str))
    if observed_path_edge_ids != declared_path_ids:
        missing = sorted(declared_path_ids - observed_path_edge_ids)
        extra = sorted(observed_path_edge_ids - declared_path_ids)
        raise ValueError(
            "path and path-edge table identities differ: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if not set(nodes["node_type"].astype(str)).issubset(IMPORTED_NODE_TYPES):
        raise ValueError(f"gene {gene_id} contains an unknown imported node type")
    if not set(edges["edge_type"].astype(str)).issubset(IMPORTED_EDGE_TYPES):
        raise ValueError(f"gene {gene_id} contains an unknown imported edge type")
    strands = (
        set(nodes["strand"].astype(str))
        | set(edges["strand"].astype(str))
        | set(paths["strand"].astype(str))
    )
    chroms = (
        set(nodes["chrom"].astype(str))
        | set(edges["chrom"].astype(str))
        | set(paths["chrom"].astype(str))
    )
    if len(strands) != 1 or not strands.issubset({"+", "-"}):
        raise ValueError(f"gene {gene_id} does not have one valid strand")
    if len(chroms) != 1:
        raise ValueError(f"gene {gene_id} spans multiple contigs")
    intervals = edges[["start_0based", "end_0based_exclusive"]].to_numpy(dtype=np.int64)
    if bool((intervals[:, 0] < 0).any()) or bool(
        (intervals[:, 1] < intervals[:, 0]).any()
    ):
        raise ValueError(f"gene {gene_id} contains invalid edge coordinates")
    if not np.isfinite(
        edges[
            [
                "span_bp",
                "length_bp",
                "relative_edge_pos",
                "annotation_confidence",
                "edge_prior_score",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(f"gene {gene_id} contains non-finite edge features")

    # Validate the imported ordering before conversion.  In particular, do not
    # use path.split_admission_flag or every edge-table row: legal paths are the
    # authority for which processing transitions actually exist in FABRIC.
    imported_edges = edges.set_index(edges["edge_id"].astype(str), drop=False)
    imported_groups = {
        str(key): value for key, value in path_edges.groupby("path_id", sort=False)
    }
    for _, path in paths.iterrows():
        path_id = str(path["path_id"])
        if path_id not in imported_groups:
            raise ValueError(f"path {path_id} has no ordered edge rows")
        ordered = imported_groups[path_id].sort_values("edge_order", kind="mergesort")
        orders = ordered["edge_order"].to_numpy(dtype=np.int64)
        if not np.array_equal(orders, np.arange(len(ordered), dtype=np.int64)):
            raise ValueError(f"path {path_id} edge_order is not contiguous from zero")
        if len(ordered) != int(path["n_edges"]):
            raise ValueError(
                f"path {path_id} n_edges differs from imported path-edge rows"
            )
        ids = ordered["edge_id"].astype(str).tolist()
        unknown = sorted(set(ids) - set(imported_edges.index))
        if unknown:
            raise ValueError(f"path {path_id} references unknown edges: {unknown[:5]}")
        if set(ordered["transcript_id"].astype(str)) != {
            str(path["transcript_id"])
        }:
            raise ValueError(
                f"path {path_id} transcript identity differs between path tables"
            )
        if set(ordered["chrom"].astype(str)) != {str(path["chrom"])} or set(
            ordered["strand"].astype(str)
        ) != {str(path["strand"])}:
            raise ValueError(
                f"path {path_id} chromosome/strand differs between path tables"
            )
        selected = imported_edges.loc[ids]
        if not ordered["edge_type"].astype(str).reset_index(drop=True).equals(
            selected["edge_type"].astype(str).reset_index(drop=True)
        ) or not ordered["chrom"].astype(str).reset_index(drop=True).equals(
            selected["chrom"].astype(str).reset_index(drop=True)
        ) or not ordered["strand"].astype(str).reset_index(drop=True).equals(
            selected["strand"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"path {path_id} edge type/chromosome/strand differs from edge table"
            )
        if not ordered["src_node_id"].astype(str).reset_index(drop=True).equals(
            selected["src_node_id"].astype(str).reset_index(drop=True)
        ) or not ordered["dst_node_id"].astype(str).reset_index(drop=True).equals(
            selected["dst_node_id"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"path {path_id} endpoint identity differs from edge table"
            )
        src = ordered["src_node_id"].astype(str).tolist()
        dst = ordered["dst_node_id"].astype(str).tolist()
        if any(dst[index] != src[index + 1] for index in range(len(ids) - 1)):
            raise ValueError(
                f"path {path_id} is not continuous in imported transcript order"
            )

    processing = path_edges["edge_type"].astype(str).isin(EDGE_TYPES)
    path_edges = path_edges.loc[processing].copy()
    if path_edges.empty:
        raise ValueError(f"gene {gene_id} has no processing path edges")
    processing_groups = {
        str(key): value.sort_values("edge_order", kind="mergesort")
        for key, value in path_edges.groupby("path_id", sort=False)
    }
    for path in paths.itertuples(index=False):
        path_id = str(path.path_id)
        ordered = processing_groups.get(path_id)
        if ordered is None or ordered.empty:
            raise ValueError(f"path {path_id} has no processing path edges")
        if str(ordered.iloc[0]["src_node_id"]) != str(path.tss_node_id):
            raise ValueError(
                f"path {path_id} does not start at its declared TSS before collapse"
            )
        if str(ordered.iloc[-1]["dst_node_id"]) != str(path.pas_node_id):
            raise ValueError(
                f"path {path_id} does not end at its declared PAS before collapse"
            )
    paths, path_edges = _collapse_identical_structural_paths(paths, path_edges)
    counts = path_edges.groupby("path_id", sort=False).size()
    if set(counts.index.astype(str)) != set(paths["path_id"].astype(str)):
        raise ValueError(f"gene {gene_id} has a legal path with no processing edges")
    path_edges["edge_order"] = path_edges.groupby("path_id", sort=False).cumcount()
    referenced_edge_ids = set(path_edges["edge_id"].astype(str))
    edges = edges.loc[edges["edge_id"].astype(str).isin(referenced_edge_ids)].copy()
    if set(edges["edge_type"].astype(str)) - set(EDGE_TYPES):
        raise ValueError("virtual START/END edges remained after processing conversion")
    referenced_node_ids = set(edges["src_node_id"].astype(str)) | set(
        edges["dst_node_id"].astype(str)
    )
    nodes = nodes.loc[nodes["node_id"].astype(str).isin(referenced_node_ids)].copy()
    if set(nodes["node_type"].astype(str)) - set(NODE_TYPES):
        raise ValueError(
            "virtual SOURCE/SINK nodes remained after processing conversion"
        )
    paths["n_edges"] = (
        paths["path_id"]
        .astype(str)
        .map({str(key): int(value) for key, value in counts.items()})
    )
    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    path_edges = path_edges.reset_index(drop=True)
    node_by_id = nodes.set_index(nodes["node_id"].astype(str), drop=False)
    if set(edges["src_node_id"].astype(str)) | set(
        edges["dst_node_id"].astype(str)
    ) != set(node_by_id.index):
        raise ValueError("processing edge endpoints and retained node axis differ")
    if not np.array_equal(
        edges["src_node_type"].astype(str).to_numpy(),
        node_by_id.loc[edges["src_node_id"].astype(str), "node_type"]
        .astype(str)
        .to_numpy(),
    ) or not np.array_equal(
        edges["dst_node_type"].astype(str).to_numpy(),
        node_by_id.loc[edges["dst_node_id"].astype(str), "node_type"]
        .astype(str)
        .to_numpy(),
    ):
        raise ValueError("processing edge endpoint types differ from retained nodes")
    positions = nodes["pos_0based"].to_numpy(dtype=np.int64)
    site_starts = nodes["site_start_0based"].to_numpy(dtype=np.int64)
    site_ends = nodes["site_end_0based"].to_numpy(dtype=np.int64)
    if (
        bool((positions < 0).any())
        or not np.array_equal(site_starts, positions)
        or not np.array_equal(site_ends, positions + 1)
    ):
        raise ValueError("processing node boundary coordinates are invalid")

    edge_ids = tuple(edges["edge_id"].astype(str))
    path_ids = tuple(paths["path_id"].astype(str))
    transcript_aliases = tuple(
        tuple(str(value) for value in aliases)
        for aliases in paths["transcript_aliases"]
    )
    edge_index = {value: index for index, value in enumerate(edge_ids)}
    edge_by_id = edges.set_index(edges["edge_id"].astype(str), drop=False)
    grouped = {
        str(key): value for key, value in path_edges.groupby("path_id", sort=False)
    }
    path_edge_rows: list[tuple[int, ...]] = []
    path_node_rows: list[tuple[str, ...]] = []
    incidence_row: list[int] = []
    incidence_col: list[int] = []
    local_pairs: set[tuple[int, int]] = set()
    for path_row, path in paths.iterrows():
        path_id = str(path["path_id"])
        if path_id not in grouped:
            raise ValueError(f"path {path_id} has no ordered edge rows")
        ordered = grouped[path_id].sort_values("edge_order", kind="mergesort")
        orders = ordered["edge_order"].to_numpy(dtype=np.int64)
        if not np.array_equal(orders, np.arange(len(ordered), dtype=np.int64)):
            raise ValueError(f"path {path_id} edge_order is not contiguous from zero")
        if len(ordered) != int(path["n_edges"]):
            raise ValueError(f"path {path_id} n_edges differs from path-edge rows")
        ids = ordered["edge_id"].astype(str).tolist()
        unknown = sorted(set(ids) - set(edge_ids))
        if unknown:
            raise ValueError(f"path {path_id} references unknown edges: {unknown[:5]}")
        indices = tuple(edge_index[value] for value in ids)
        if len(set(indices)) != len(indices):
            raise ValueError(f"path {path_id} repeats a processing edge")
        selected_edges = edge_by_id.loc[ids]
        if not ordered["src_node_id"].astype(str).reset_index(drop=True).equals(
            selected_edges["src_node_id"].astype(str).reset_index(drop=True)
        ) or not ordered["dst_node_id"].astype(str).reset_index(drop=True).equals(
            selected_edges["dst_node_id"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"path {path_id} endpoint identity differs from edge table"
            )
        src = ordered["src_node_id"].astype(str).tolist()
        dst = ordered["dst_node_id"].astype(str).tolist()
        if any(dst[index] != src[index + 1] for index in range(len(ids) - 1)):
            raise ValueError(f"path {path_id} is not continuous in transcript order")
        node_sequence = tuple([src[0], *dst])
        if node_sequence[0] != str(path["tss_node_id"]):
            raise ValueError(f"path {path_id} does not start at its declared TSS")
        if node_sequence[-1] != str(path["pas_node_id"]):
            raise ValueError(f"path {path_id} does not end at its declared PAS")
        path_positions = node_by_id.loc[list(node_sequence), "pos_0based"].to_numpy(
            dtype=np.int64
        )
        direction = np.diff(path_positions)
        strand = str(path["strand"])
        if (strand == "+" and bool((direction < 0).any())) or (
            strand == "-" and bool((direction > 0).any())
        ):
            raise ValueError(
                f"path {path_id} node order disagrees with transcript strand"
            )
        coincident = np.flatnonzero(direction == 0)
        for index in coincident:
            edge = selected_edges.iloc[int(index)]
            if (
                str(edge.edge_type) != "EXON_CONTINUATION"
                or int(edge.span_bp) != 0
                or (str(edge.src_node_type), str(edge.dst_node_type))
                != ("acceptor", "donor")
            ):
                raise ValueError(
                    f"path {path_id} has an invalid coincident-site transition"
                )
        path_edge_rows.append(indices)
        path_node_rows.append(node_sequence)
        incidence_row.extend([path_row] * len(indices))
        incidence_col.extend(indices)
        for left, right in zip(indices[:-1], indices[1:]):
            local_pairs.add((left, right))
            local_pairs.add((right, left))
    values = np.ones(len(incidence_row), dtype=np.float32)
    incidence = sparse.csr_matrix(
        (values, (incidence_row, incidence_col)), shape=(len(paths), len(edges))
    )
    local_edge_index = (
        np.asarray(sorted(local_pairs), dtype=np.int64).T
        if local_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    catalog_incidence_mean = np.asarray(incidence.mean(axis=0)).reshape(-1)
    variable_edge_count = int(
        ((catalog_incidence_mean > 0.0) & (catalog_incidence_mean < 1.0)).sum()
    )
    centered_path_incidence_energy = float(
        np.sum(catalog_incidence_mean * (1.0 - catalog_incidence_mean))
    )
    return GeneGraph(
        gene_id=gene_id,
        nodes=nodes,
        edges=edges,
        paths=paths,
        path_edges=path_edges,
        edge_ids=edge_ids,
        path_ids=path_ids,
        transcript_aliases=transcript_aliases,
        path_edge_rows=tuple(path_edge_rows),
        path_node_rows=tuple(path_node_rows),
        path_first_edge_indices=tuple(row[0] for row in path_edge_rows),
        path_last_edge_indices=tuple(row[-1] for row in path_edge_rows),
        path_log_edge_count=tuple(float(np.log1p(len(row))) for row in path_edge_rows),
        path_edge_incidence=incidence,
        local_edge_index=local_edge_index,
        edge_features=edge_feature_matrix(edges),
        variable_edge_count=variable_edge_count,
        centered_path_incidence_energy=centered_path_incidence_energy,
    )


def edge_feature_matrix(edges: pd.DataFrame) -> np.ndarray:
    """Fixed visible structural features for the only CIS encoder."""

    rows: list[list[float]] = []
    for row in edges.itertuples(index=False):
        edge_type = str(row.edge_type)
        src_type = str(row.src_node_type)
        dst_type = str(row.dst_node_type)
        feature = [float(edge_type == value) for value in EDGE_TYPES]
        feature.extend(float(src_type == value) for value in NODE_TYPES)
        feature.extend(float(dst_type == value) for value in NODE_TYPES)
        feature.extend(
            (
                float(np.log1p(float(row.span_bp)) / np.log1p(1_000_000.0)),
                float(np.log1p(float(row.length_bp)) / np.log1p(1_000_000.0)),
                float(row.relative_edge_pos),
                float(row.annotation_confidence),
                float(row.edge_prior_score),
            )
        )
        rows.append(feature)
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("fixed edge feature matrix contains non-finite values")
    return result


def _collapse_identical_structural_paths(
    paths: pd.DataFrame, path_edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse duplicate ordered edge sequences and retain transcript aliases.

    The legal candidate axis is structural.  Transcript annotations with the
    same ordered processing-edge sequence therefore contribute one softmax
    entry, while every original transcript ID remains attached to that entry
    for reporting and crosswalk validation.
    """

    grouped = {
        str(path_id): group.sort_values("edge_order", kind="mergesort")
        for path_id, group in path_edges.groupby("path_id", sort=False)
    }
    path_rows = {
        str(row.path_id): row
        for row in paths.itertuples(index=False)
    }
    sequence_groups: dict[tuple[str, ...], list[str]] = {}
    for path_id in paths["path_id"].astype(str):
        ordered = grouped.get(path_id)
        if ordered is None:
            raise ValueError(f"path {path_id} has no processing edge rows")
        sequence = tuple(ordered["edge_id"].astype(str))
        sequence_groups.setdefault(sequence, []).append(path_id)

    collapsed_paths: list[dict[str, object]] = []
    collapsed_path_edges: list[pd.DataFrame] = []
    for sequence, source_path_ids in sequence_groups.items():
        canonical_path_id = min(source_path_ids)
        canonical = paths.loc[
            paths["path_id"].astype(str) == canonical_path_id
        ].iloc[0].to_dict()
        source_transcripts = [
            getattr(path_rows[path_id], "transcript_id")
            for path_id in source_path_ids
        ]
        if any(pd.isna(value) for value in source_transcripts):
            raise ValueError("structural path transcript aliases cannot be missing")
        aliases = sorted({str(value) for value in source_transcripts})
        if not aliases or any(not alias for alias in aliases):
            raise ValueError("structural path transcript aliases must be non-empty")
        source_path_rows = paths.loc[
            paths["path_id"].astype(str).isin(source_path_ids)
        ]
        if source_path_rows["path_length_bp"].nunique(dropna=False) != 1:
            raise ValueError(
                "identical structural paths disagree on declared path_length_bp"
            )
        canonical["path_id"] = canonical_path_id
        canonical["transcript_id"] = aliases[0]
        canonical["transcript_aliases"] = aliases
        canonical["source_path_ids"] = sorted(source_path_ids)
        collapsed_paths.append(canonical)

        ordered = grouped[canonical_path_id].copy()
        ordered["path_id"] = canonical_path_id
        ordered["transcript_id"] = aliases[0]
        if tuple(ordered["edge_id"].astype(str)) != sequence:
            raise AssertionError("canonical structural path sequence changed")
        collapsed_path_edges.append(ordered)

    result_paths = pd.DataFrame(collapsed_paths).reset_index(drop=True)
    result_edges = pd.concat(collapsed_path_edges, ignore_index=True)
    if result_paths["path_id"].astype(str).duplicated().any():
        raise AssertionError("collapsed structural path IDs are not unique")
    return result_paths, result_edges


def load_compatibility_rows(
    path: str | Path,
    *,
    gene_ids: Sequence[str],
    cell_ids: Sequence[str] | None = None,
    batch_size: int = 65_536,
) -> Iterator[pd.DataFrame]:
    """Stream selected EC rows from the 59M-row normalized Parquet source."""

    genes = [str(value) for value in gene_ids]
    if not genes:
        raise ValueError("gene_ids must be non-empty")
    expression = pads.field("gene_id").isin(genes)
    if cell_ids is not None:
        cells = [canonical_rna_cell_id(value) for value in cell_ids]
        expression = expression & pads.field("cell_id").isin(cells)
    scanner = pads.dataset(path, format="parquet").scanner(
        columns=[
            "cell_id",
            "gene_id",
            "compatibility_class_id",
            "compatible_path_ids",
            "compatible_path_count",
            "molecule_count",
            "mean_mapq",
            "mean_aligned_bp",
            "bucket_qc_flags",
            "split",
        ],
        filter=expression,
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if len(frame):
            yield frame


def bind_authoritative_split(
    ec_rows: pd.DataFrame, split_rows: pd.DataFrame
) -> pd.DataFrame:
    """Replace the stale embedded EC split using globally unique cell IDs."""

    rows = ec_rows.copy()
    rows["cell_id"] = rows["cell_id"].astype(str).map(canonical_rna_cell_id)
    authority = split_rows[["cell_id", "split"]].rename(
        columns={"split": "fabric_split"}
    )
    merged = rows.merge(authority, on="cell_id", how="left", validate="many_to_one")
    if merged["fabric_split"].isna().any():
        missing = merged.loc[merged["fabric_split"].isna(), "cell_id"].unique().tolist()
        raise ValueError(f"EC cells absent from authoritative split: {missing[:10]}")
    merged = merged.drop(columns=["split"]).rename(columns={"fabric_split": "split"})
    return merged


def normalize_compatibility_path_order(
    rows: pd.DataFrame, graph: GeneGraph
) -> pd.DataFrame:
    """Convert external EC path lists to the path-table row order explicitly.

    The source EC serializes path IDs lexicographically, whereas FABRIC tensor
    indices follow the preserved path-table order.  Compatibility is a set;
    this import conversion records the source order and creates the sole
    canonical internal ordering instead of mistaking a valid set for drift.
    """

    normalized = rows.copy()
    path_index = {value: index for index, value in enumerate(graph.path_ids)}
    source_lists: list[list[str]] = []
    ordered_lists: list[list[str]] = []
    index_lists: list[list[int]] = []
    for row in normalized.itertuples(index=False):
        compatible = [str(value) for value in row.compatible_path_ids]
        if not compatible or len(set(compatible)) != len(compatible):
            raise ValueError("compatible path set must be non-empty and unique")
        if len(compatible) != int(row.compatible_path_count):
            raise ValueError("compatible_path_count differs from path list")
        unknown = sorted(set(compatible) - set(graph.path_ids))
        if unknown:
            raise ValueError(f"EC row references unknown legal paths: {unknown[:5]}")
        indices = sorted(path_index[value] for value in compatible)
        source_lists.append(compatible)
        ordered_lists.append([graph.path_ids[index] for index in indices])
        index_lists.append(indices)
    normalized["source_compatible_path_ids"] = source_lists
    normalized["compatible_path_ids"] = ordered_lists
    normalized["compatible_path_indices"] = index_lists
    return normalized


def validate_compatibility_rows(rows: pd.DataFrame, graph: GeneGraph) -> None:
    """Verify EC gene/path identities against the imported legal-path order."""

    if set(rows["gene_id"].astype(str)) != {graph.gene_id}:
        raise ValueError("compatibility rows and graph gene differ")
    path_index = {value: index for index, value in enumerate(graph.path_ids)}
    for row in rows.itertuples(index=False):
        compatible = [str(value) for value in row.compatible_path_ids]
        if not compatible or len(set(compatible)) != len(compatible):
            raise ValueError("compatible path set must be non-empty and unique")
        if len(compatible) != int(row.compatible_path_count):
            raise ValueError("compatible_path_count differs from path list")
        unknown = sorted(set(compatible) - set(graph.path_ids))
        if unknown:
            raise ValueError(f"EC row references unknown legal paths: {unknown[:5]}")
        indices = [path_index[value] for value in compatible]
        if indices != sorted(indices):
            raise ValueError("compatible path IDs do not follow canonical path order")
        if int(row.molecule_count) <= 0:
            raise ValueError("imported EC molecule_count must be positive")


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    values = frame[column].astype(str)
    if values.str.len().eq(0).any() or values.duplicated().any():
        raise ValueError(f"{label} {column} values must be unique and non-empty")
