"""Build the one real, test-blind FABRIC V2 dataset.

The builder is deliberately table-first and resumable.  It imports the frozen
17,706-gene ONT structural-path catalog directly, reconstructs processing
graphs without reading the historical PRISM graph, and keeps test compatible
rows absent.  Every stage writes ordinary Parquet/JSON records with readable
identities; this module adds no checksum or content-hash layer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import hdf5plugin  # noqa: F401 - registers the HDF5 compression filters
import yaml
from pyfaidx import Fasta
from sklearn.neighbors import NearestNeighbors

from .cis import (
    CISFeatureManifest,
    CISNormalizationPolicy,
    CISSequenceFeatureSpec,
    SEQUENCE_FEATURES,
    apply_cis_normalization,
    build_explicit_cis_table,
    fit_cis_normalization,
)
from .dataset import (
    ATACMappingContext,
    ActivityContext,
    build_gate_keys,
    build_raw_gate_signals,
    fit_gate_admission,
    normalize_log1p_counts,
    transform_gates,
)
from .graph import GraphTables, build_gene_graph
from .motifs import (
    accessibility_only_hits,
    assign_unique_peak_to_dna_hits,
    build_factor_catalog,
    build_candidate_routes,
    build_graph_anchor_regions,
    cap_and_finalize_routes,
    collapse_physical_events,
    parse_cisbp_motifs,
    parse_meme_motifs,
    transcript_relative_interval,
)


EXPECTED = {
    "candidate_gene_count": 17_706,
    "path_count": 90_672,
    "cell_count": 217_933,
    "train_cell_count": 174_357,
    "val_cell_count": 21_788,
    "test_cell_count": 21_788,
    "g_fit_gene_count": 17_600,
    "graph_only_gene_count": 106,
}

_ATTRIBUTE = re.compile(r'(\w+) "([^"]*)"')


def _read_yaml(path: str | Path) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def validate_external_inputs(manifest_path: str | Path) -> dict[str, Path]:
    """Resolve and validate every real source before a derived stage starts."""

    manifest = _read_yaml(manifest_path)
    if manifest.get("contract") != "FABRIC_ARCHITECTURE_V2":
        raise ValueError("external inputs are not bound to FABRIC_ARCHITECTURE_V2")
    sources = manifest.get("sources")
    derived = manifest.get("derived")
    if not isinstance(sources, dict) or not isinstance(derived, dict):
        raise TypeError("external inputs require sources and derived mappings")
    forbidden_fragments = (
        "final_multimodal_simple_12run_v1/D2/graph",
        "fabric_v1_compatible_ec",
        "i6_multi7198",
        "167235",
    )
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for role, raw in sources.items():
        path = Path(str(raw))
        if any(value in str(path) for value in forbidden_fragments):
            raise ValueError(f"historical V1 input is forbidden for {role}: {path}")
        paths[str(role)] = path
        if not path.exists():
            missing.append(f"{role}={path}")
    if missing:
        raise FileNotFoundError("real V2 sources are absent: " + "; ".join(missing))
    paths["real_dataset"] = Path(str(derived["real_dataset"]))
    paths["rna_atac_neighbors"] = Path(str(derived["rna_atac_neighbors"]))

    compatibility = json.loads(paths["compatibility_manifest"].read_text())
    required_manifest = {
        "artifact_complete": True,
        "admission_pass": True,
        "test_rows_written": False,
        "test_predictions_or_metrics_computed": False,
        "matrix_structural_path_count": EXPECTED["path_count"],
    }
    for key, expected in required_manifest.items():
        if compatibility.get(key) != expected:
            raise ValueError(
                f"compatibility manifest {key}={compatibility.get(key)!r}, "
                f"expected {expected!r}"
            )
    if compatibility.get("build_scope") != "complete_train_validation":
        raise ValueError("compatible EC artifact is not exactly train+validation")
    if compatibility.get("compatible_test_row_exposure") != (
        "not_materialized_before_checkpoint"
    ):
        raise ValueError("test compatible-row exposure is not fail-closed")

    path_table = pd.read_parquet(paths["legal_structural_paths"])
    if len(path_table) != EXPECTED["path_count"]:
        raise ValueError("frozen legal structural path count drift")
    if path_table["path_id"].astype(str).duplicated().any():
        raise ValueError("frozen legal structural path IDs are duplicated")
    candidate_ids = tuple(map(str, compatibility["candidate_gene_ids"]))
    if len(candidate_ids) != EXPECTED["candidate_gene_count"] or len(
        set(candidate_ids)
    ) != len(candidate_ids):
        raise ValueError("frozen candidate gene identity is not 17,706 unique IDs")
    if set(path_table["gene_id"].astype(str)) != set(candidate_ids):
        raise ValueError("legal path genes differ from the frozen candidate set")

    g_fit = pd.read_csv(paths["g_fit"], sep="\t")
    g_fit_ids = tuple(g_fit["target_gene_id"].astype(str))
    if len(g_fit_ids) != EXPECTED["g_fit_gene_count"] or len(set(g_fit_ids)) != len(
        g_fit_ids
    ):
        raise ValueError("G_fit identity is not 17,600 unique genes")
    support = pd.read_parquet(paths["candidate_support_status"])
    graph_only = support.loc[
        support["support_status"].astype(str).eq(
            "graph_only_zero_train_informative_mass"
        ),
        "target_gene_id",
    ].astype(str)
    if len(graph_only) != EXPECTED["graph_only_gene_count"]:
        raise ValueError("graph-only audit does not contain exactly 106 genes")
    if set(g_fit_ids) | set(graph_only) != set(candidate_ids) or set(g_fit_ids) & set(
        graph_only
    ):
        raise ValueError("G_fit and graph-only sets do not partition the candidate set")

    split = pd.read_parquet(paths["split_rows"], columns=["cell_id", "split"])
    if len(split) != EXPECTED["cell_count"] or split["cell_id"].astype(str).duplicated().any():
        raise ValueError("frozen split is not exactly 217,933 unique cells")
    counts = split["split"].astype(str).value_counts().to_dict()
    expected_counts = {
        "train": EXPECTED["train_cell_count"],
        "val": EXPECTED["val_cell_count"],
        "test": EXPECTED["test_cell_count"],
    }
    if counts != expected_counts:
        raise ValueError(f"frozen split counts drift: {counts}")
    matrix_cells = pd.read_parquet(paths["matrix_cell_index"])
    matrix_cell_column = "cell_id" if "cell_id" in matrix_cells else "resolved_cell_id"
    if matrix_cells[matrix_cell_column].astype(str).duplicated().any() or set(
        matrix_cells[matrix_cell_column].astype(str)
    ) != set(split["cell_id"].astype(str)):
        raise ValueError("matrix cell identity differs from the frozen split")
    if "matrix_column_0based" not in matrix_cells or not np.array_equal(
        matrix_cells["matrix_column_0based"].to_numpy(np.int64),
        np.arange(len(matrix_cells), dtype=np.int64),
    ):
        raise ValueError("matrix cell index is not contiguous in matrix-column order")
    matrix_assignment = matrix_cells[
        [matrix_cell_column, "split"]
    ].rename(columns={matrix_cell_column: "cell_id", "split": "matrix_split"})
    joined = split.merge(matrix_assignment, on="cell_id", how="left", validate="one_to_one")
    if joined["matrix_split"].isna().any() or not np.array_equal(
        joined["split"].astype(str).to_numpy(),
        joined["matrix_split"].astype(str).to_numpy(),
    ):
        raise ValueError("matrix cell split assignments differ from split_rows")
    return paths


def _parse_gtf_exons(
    path: Path, transcript_ids: set[str]
) -> dict[str, tuple[str, str, str, tuple[int, ...], tuple[int, ...]]]:
    """Read only selected exon rows as 0-based half-open transcript models."""

    exons: dict[str, list[tuple[int, int]]] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue
            attrs = dict(_ATTRIBUTE.findall(fields[8]))
            transcript_id = attrs.get("transcript_id", "").split(".")[0]
            if transcript_id not in transcript_ids:
                continue
            gene_id = attrs.get("gene_id", "").split(".")[0]
            chrom, strand = fields[0], fields[6]
            current = (gene_id, chrom, strand)
            if transcript_id in metadata and metadata[transcript_id] != current:
                raise ValueError(f"GTF transcript metadata is inconsistent: {transcript_id}")
            metadata[transcript_id] = current
            exons.setdefault(transcript_id, []).append((int(fields[3]) - 1, int(fields[4])))
    result = {}
    for transcript_id, intervals in exons.items():
        gene_id, chrom, strand = metadata[transcript_id]
        # GTF and the frozen compatible-path artifact both serialize exon
        # arrays in genomic ascending order.  Transcript orientation is
        # applied only while constructing each processing path.
        ordered = sorted(intervals)
        result[transcript_id] = (
            gene_id,
            chrom,
            strand,
            tuple(value[0] for value in ordered),
            tuple(value[1] for value in ordered),
        )
    return result


def _crosscheck_path_gtfs(
    paths: pd.DataFrame, authoritative_gtf: Path, matrix_gtf: Path
) -> pd.DataFrame:
    transcript_ids = set(paths["resolved_transcript_id"].astype(str))
    authority = _parse_gtf_exons(authoritative_gtf, transcript_ids)
    matrix = _parse_gtf_exons(matrix_gtf, transcript_ids)
    rows = []
    for row in paths.itertuples(index=False):
        transcript_id = str(row.resolved_transcript_id)
        expected = (
            str(row.gene_id),
            str(row.chrom),
            str(row.strand),
            tuple(map(int, row.exon_starts_0based)),
            tuple(map(int, row.exon_ends_0based_exclusive)),
        )
        matched = []
        for source, table in (("authoritative_gtf", authority), ("matrix_matched_gtf", matrix)):
            observed = table.get(transcript_id)
            if observed is not None and observed != expected:
                raise ValueError(
                    f"frozen path differs from {source} for {transcript_id}"
                )
            if observed is not None:
                matched.append(source)
        if not matched:
            raise ValueError(f"frozen structural path is absent from both GTFs: {transcript_id}")
        rows.append(
            {
                "path_id": str(row.path_id),
                "resolved_transcript_id": transcript_id,
                "target_gene_id": str(row.gene_id),
                "gtf_sources_matching_exact_exons": matched,
                "status": "exact_exon_identity_confirmed",
            }
        )
    return pd.DataFrame(rows)


def _node_type_for_position(
    position: int,
    *,
    strand: str,
    path_starts: tuple[int, ...],
    path_ends: tuple[int, ...],
    exon_index: int,
) -> str:
    last = len(path_starts) - 1
    if strand == "+":
        if exon_index == 0 and position == path_starts[0]:
            return "TSS"
        if exon_index == last and position == path_ends[last]:
            return "PAS"
        return "acceptor" if position == path_starts[exon_index] else "donor"
    if exon_index == 0 and position == path_ends[0]:
        return "TSS"
    if exon_index == last and position == path_starts[last]:
        return "PAS"
    return "donor" if position == path_starts[exon_index] else "acceptor"


def compile_gene_graph_tables(path_rows: pd.DataFrame) -> GraphTables:
    """Reconstruct one union processing graph from frozen ordered exon paths."""

    path_rows = path_rows.sort_values("path_order_0based", kind="mergesort")
    gene_id = str(path_rows.iloc[0]["gene_id"])
    if set(path_rows["gene_id"].astype(str)) != {gene_id}:
        raise ValueError("gene graph input mixes gene identities")
    if not np.array_equal(
        path_rows["path_order_0based"].to_numpy(np.int64),
        np.arange(len(path_rows), dtype=np.int64),
    ):
        raise ValueError(f"frozen path order is not contiguous for {gene_id}")
    chroms = set(path_rows["chrom"].astype(str))
    strands = set(path_rows["strand"].astype(str))
    if len(chroms) != 1 or len(strands) != 1:
        raise ValueError(f"gene {gene_id} spans multiple chromosome/strand values")
    chrom, strand = next(iter(chroms)), next(iter(strands))
    if strand not in {"+", "-"}:
        raise ValueError(f"invalid strand for {gene_id}: {strand}")

    all_intron_pairs: set[tuple[int, int]] = set()
    models: list[tuple[object, tuple[int, ...], tuple[int, ...]]] = []
    for row in path_rows.itertuples(index=False):
        starts = tuple(map(int, row.exon_starts_0based))
        ends = tuple(map(int, row.exon_ends_0based_exclusive))
        if not starts or len(starts) != len(ends) or any(a >= b for a, b in zip(starts, ends)):
            raise ValueError(f"invalid exon intervals for {row.path_id}")
        genomic_order = list(zip(starts, ends))
        if genomic_order != sorted(genomic_order):
            raise ValueError(f"frozen exon array is not genomic-order for {row.path_id}")
        ordered = genomic_order if strand == "+" else genomic_order[::-1]
        starts = tuple(value[0] for value in ordered)
        ends = tuple(value[1] for value in ordered)
        for index in range(len(starts) - 1):
            if strand == "+":
                all_intron_pairs.add((ends[index], starts[index + 1]))
            else:
                all_intron_pairs.add((starts[index], ends[index + 1]))
        models.append((row, starts, ends))

    node_types: dict[int, set[str]] = {}
    path_nodes: dict[str, list[tuple[int, str]]] = {}
    path_edge_types: dict[str, list[str]] = {}
    for row, starts, ends in models:
        ordered_nodes: list[tuple[int, str]] = []
        junction_after: set[int] = set()
        retained_pairs: set[tuple[int, int]] = set()
        for exon_index, (start, end) in enumerate(zip(starts, ends)):
            boundaries: set[int] = {start, end}
            internal_types: dict[int, set[str]] = {}
            retained_intervals = sorted(
                (min(donor, acceptor), max(donor, acceptor))
                for donor, acceptor in all_intron_pairs
                if start < min(donor, acceptor)
                and max(donor, acceptor) < end
            )
            overlap_component: list[tuple[int, int]] = []
            component_end = -1
            for low, high in retained_intervals:
                if overlap_component and low <= component_end:
                    overlap_component.append((low, high))
                    component_end = max(component_end, high)
                else:
                    if len(overlap_component) > 1:
                        raise ValueError(
                            "retained exon covers an unresolved overlapping-intron "
                            f"component for {gene_id}/{row.path_id}: "
                            f"exon=({start}, {end}), introns={overlap_component}"
                        )
                    overlap_component = [(low, high)]
                    component_end = high
            if len(overlap_component) > 1:
                raise ValueError(
                    "retained exon covers an unresolved overlapping-intron "
                    f"component for {gene_id}/{row.path_id}: "
                    f"exon=({start}, {end}), introns={overlap_component}"
                )
            for low, high in retained_intervals:
                donor, acceptor = (low, high) if strand == "+" else (high, low)
                retained_pairs.add((donor, acceptor))
                boundaries.update((donor, acceptor))
                internal_types.setdefault(donor, set()).add("donor")
                internal_types.setdefault(acceptor, set()).add("acceptor")
            transcript_order = sorted(boundaries, reverse=strand == "-")
            for position in transcript_order:
                endpoint_type = _node_type_for_position(
                    position,
                    strand=strand,
                    path_starts=starts,
                    path_ends=ends,
                    exon_index=exon_index,
                )
                # Internal retained-intron boundaries inherit their union-graph
                # donor/acceptor type, not the enclosing exon endpoint type.
                if position not in {start, end}:
                    candidate_types = internal_types.get(position, set())
                    if len(candidate_types) != 1:
                        raise ValueError(
                            f"ambiguous internal processing-site type at {gene_id}:{position}"
                        )
                    endpoint_type = next(iter(candidate_types))
                node_types.setdefault(position, set()).add(endpoint_type)
                ordered_nodes.append((position, endpoint_type))
            if exon_index < len(starts) - 1:
                junction_after.add(len(ordered_nodes) - 1)
        edge_types = []
        for index, (left, right) in enumerate(zip(ordered_nodes[:-1], ordered_nodes[1:])):
            pair = (left[0], right[0])
            if index in junction_after:
                edge_types.append("SPLICE")
            elif pair in retained_pairs:
                edge_types.append("RETAINED_INTRON")
            else:
                edge_types.append("EXON_CONTINUATION")
        path_nodes[str(row.path_id)] = ordered_nodes
        path_edge_types[str(row.path_id)] = edge_types

    gene_start = min(position for nodes in path_nodes.values() for position, _ in nodes)
    gene_end = max(position for nodes in path_nodes.values() for position, _ in nodes)
    span = gene_end - gene_start
    # The same genomic boundary can legitimately be a TSS in one path and a
    # donor in another.  Node identity is therefore (position, processing
    # type), while a path still has strictly monotone genomic positions.
    node_id = {
        (position, node_type): f"node:{gene_id}:{node_type}:{chrom}:{position}:{strand}"
        for position, types in node_types.items()
        for node_type in sorted(types)
    }
    nodes = pd.DataFrame(
        [
            {
                "gene_id": gene_id,
                "node_id": node_id[(position, node_type)],
                "node_type": node_type,
                "chrom": chrom,
                "strand": strand,
                "pos_0based": position,
                "site_start_0based": position,
                "site_end_0based": position + 1,
                "relative_gene_pos": (
                    (position - gene_start) / span
                    if strand == "+"
                    else (gene_end - position) / span
                ),
                "annotation_confidence": 1.0,
                "site_prior_score": 0.0,
            }
            for position, types in sorted(node_types.items())
            for node_type in sorted(types)
        ]
    )
    edge_records: dict[tuple[str, int, str, int, str], dict[str, object]] = {}
    path_edge_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    row_by_path = path_rows.set_index(path_rows["path_id"].astype(str), drop=False)
    for path_id in path_rows["path_id"].astype(str):
        row = row_by_path.loc[path_id]
        ordered_nodes = path_nodes[path_id]
        edge_types = path_edge_types[path_id]
        sequence = []
        for order, ((src_pos, src_type), (dst_pos, dst_type), edge_type) in enumerate(
            zip(ordered_nodes[:-1], ordered_nodes[1:], edge_types)
        ):
            key = (edge_type, src_pos, src_type, dst_pos, dst_type)
            edge_id = (
                f"edge:{gene_id}:{edge_type}:{src_type}:{src_pos}>"
                f"{dst_type}:{dst_pos}:{strand}"
            )
            start, end = sorted((src_pos, dst_pos))
            span_bp = end - start
            record = {
                "gene_id": gene_id,
                "edge_id": edge_id,
                "edge_type": edge_type,
                "src_node_id": node_id[(src_pos, src_type)],
                "dst_node_id": node_id[(dst_pos, dst_type)],
                "src_node_type": src_type,
                "dst_node_type": dst_type,
                "chrom": chrom,
                "strand": strand,
                "start_0based": start,
                "end_0based_exclusive": end,
                "span_bp": span_bp,
                "length_bp": 0 if edge_type == "SPLICE" else span_bp,
                "relative_edge_pos": (
                    (start - gene_start) / span
                    if strand == "+"
                    else (gene_end - end) / span
                ),
                "annotation_confidence": 1.0,
                "edge_prior_score": 0.0,
            }
            if key in edge_records and edge_records[key] != record:
                raise ValueError(f"union edge identity conflict in {gene_id}")
            edge_records[key] = record
            sequence.append(record)
            path_edge_records.append(
                {
                    "gene_id": gene_id,
                    "path_id": path_id,
                    "transcript_id": str(row["resolved_transcript_id"]),
                    "edge_order": order,
                    "edge_id": edge_id,
                    "edge_type": edge_type,
                    "src_node_id": node_id[(src_pos, src_type)],
                    "dst_node_id": node_id[(dst_pos, dst_type)],
                    "chrom": chrom,
                    "strand": strand,
                }
            )
        path_records.append(
            {
                "gene_id": gene_id,
                "path_id": path_id,
                "transcript_id": str(row["resolved_transcript_id"]),
                "transcript_aliases": list(row["transcript_aliases"]),
                "chrom": chrom,
                "strand": strand,
                "tss_node_id": node_id[ordered_nodes[0]],
                "pas_node_id": node_id[ordered_nodes[-1]],
                "n_edges": len(sequence),
                "path_length_bp": int(sum(value["length_bp"] for value in sequence)),
            }
        )
    edges = pd.DataFrame(edge_records.values()).sort_values(
        ["start_0based", "end_0based_exclusive", "edge_type", "edge_id"],
        kind="mergesort",
    )
    tables = GraphTables(
        nodes=nodes.reset_index(drop=True),
        edges=edges.reset_index(drop=True),
        paths=pd.DataFrame(path_records).reset_index(drop=True),
        path_edges=pd.DataFrame(path_edge_records).reset_index(drop=True),
    )
    graph = build_gene_graph(gene_id, **asdict(tables))
    if graph.path_ids != tuple(path_rows["path_id"].astype(str)):
        raise ValueError(f"runtime graph changed frozen path order for {gene_id}")
    return GraphTables(graph.nodes, graph.edges, graph.paths, graph.path_edges)


def build_graph_stage(paths: Mapping[str, Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    graph_root = output / "graph"
    graph_root.mkdir(parents=True, exist_ok=True)
    legal = pd.read_parquet(paths["legal_structural_paths"])
    gtf_audit = _crosscheck_path_gtfs(
        legal, paths["authoritative_gtf"], paths["matrix_matched_gtf"]
    )
    tables: dict[str, list[pd.DataFrame]] = {
        "node_table": [],
        "edge_table": [],
        "path_table": [],
        "path_edge_table": [],
    }
    for _, group in legal.groupby("gene_id", sort=False):
        compiled = compile_gene_graph_tables(group)
        tables["node_table"].append(compiled.nodes)
        tables["edge_table"].append(compiled.edges)
        tables["path_table"].append(compiled.paths)
        tables["path_edge_table"].append(compiled.path_edges)
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            graph_root / f"{name}.parquet", index=False
        )
    gtf_audit.to_parquet(graph_root / "gtf_path_identity_audit.parquet", index=False)

    support = pd.read_parquet(paths["candidate_support_status"])
    support[[
        "target_gene_id",
        "support_status",
        "train_positive_informative_ec_mass",
        "validation_positive_informative_ec_mass",
    ]].to_parquet(graph_root / "candidate_graph_fit_audit.parquet", index=False)
    record = {
        "schema_version": "fabric.real_graph_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(paths["legal_structural_paths"]),
        "source_gtfs": [str(paths["authoritative_gtf"]), str(paths["matrix_matched_gtf"])],
        "candidate_gene_count": int(legal["gene_id"].nunique()),
        "path_count": len(legal),
        "node_count": sum(len(value) for value in tables["node_table"]),
        "edge_count": sum(len(value) for value in tables["edge_table"]),
        "edge_length_semantics": {
            "span_bp": "genomic distance between processing sites",
            "SPLICE.length_bp": 0,
            "EXON_CONTINUATION.length_bp": "span_bp",
            "RETAINED_INTRON.length_bp": "span_bp",
            "path_length_bp": "sum of retained edge length_bp",
        },
        "graph_only_gene_count": int(
            support["support_status"].astype(str).eq(
                "graph_only_zero_train_informative_mass"
            ).sum()
        ),
        "historical_graph_used": False,
    }
    (graph_root / "GraphManifest.json").write_text(json.dumps(record, indent=2) + "\n")


def _revcomp(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def _fetch_oriented(fasta: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    contig_length = len(fasta[chrom])
    start, end = max(0, start), min(contig_length, end)
    if end <= start:
        return ""
    sequence = str(fasta[chrom][start:end]).upper()
    return _revcomp(sequence) if strand == "-" else sequence


def _fraction(sequence: str, alphabet: set[str]) -> float:
    bases = [base for base in sequence if base in {"A", "C", "G", "T"}]
    return 0.0 if not bases else sum(base in alphabet for base in bases) / len(bases)


def _max_consensus_fraction(sequence: str, patterns: Sequence[str]) -> float:
    if not sequence:
        return 0.0
    best = 0.0
    for pattern in patterns:
        width = len(pattern)
        if len(sequence) < width:
            continue
        allowed = {
            "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
            "R": {"A", "G"}, "Y": {"C", "T"}, "W": {"A", "T"},
            "N": {"A", "C", "G", "T"},
        }
        for offset in range(len(sequence) - width + 1):
            score = sum(
                sequence[offset + index] in allowed[code]
                for index, code in enumerate(pattern)
            ) / width
            best = max(best, score)
    return best


def _cis_sequence_scores(edges: pd.DataFrame, nodes: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    node = nodes.set_index(nodes["node_id"].astype(str), drop=False)
    rows = []
    for edge in edges.itertuples(index=False):
        chrom, strand = str(edge.chrom), str(edge.strand)
        src = node.loc[str(edge.src_node_id)]
        dst = node.loc[str(edge.dst_node_id)]
        sites = {str(src.node_type): int(src.pos_0based), str(dst.node_type): int(dst.pos_0based)}
        edge_sequence = _fetch_oriented(
            fasta, chrom, int(edge.start_0based), int(edge.end_0based_exclusive), strand
        )
        if len(edge_sequence) != int(edge.end_0based_exclusive) - int(edge.start_0based):
            raise ValueError(f"CIS edge interval crosses a reference-contig boundary: {edge.edge_id}")
        record: dict[str, object] = {"edge_id": str(edge.edge_id)}
        record["edge_gc_fraction"] = _fraction(edge_sequence, {"G", "C"})
        record["edge_gc_fraction_available"] = True
        feature_specs = {
            "donor_strength": "donor",
            "acceptor_strength": "acceptor",
            "branchpoint_score": "acceptor",
            "polypyrimidine_tract_score": "acceptor",
            "tss_core_promoter_score": "TSS",
            "polya_hexamer_score": "PAS",
            "pas_downstream_u_gu_fraction": "PAS",
        }
        for feature, site_type in feature_specs.items():
            available = site_type in sites
            record[f"{feature}_available"] = available
            if not available:
                record[feature] = 0.0
                continue
            position = sites[site_type]
            if feature == "donor_strength":
                start, end = transcript_relative_interval(position, -3, 6, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _max_consensus_fraction(seq, ["NNNGTRAGT"])
            elif feature == "acceptor_strength":
                start, end = transcript_relative_interval(position, -20, 3, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _max_consensus_fraction(seq, ["YYYYYYYYYYYYYYYYYYAGNNN"])
            elif feature == "branchpoint_score":
                start, end = transcript_relative_interval(position, -50, -5, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _max_consensus_fraction(seq, ["YTNAY"])
            elif feature == "polypyrimidine_tract_score":
                start, end = transcript_relative_interval(position, -25, -3, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _fraction(seq, {"C", "T"})
            elif feature == "tss_core_promoter_score":
                start, end = transcript_relative_interval(position, -40, 10, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _max_consensus_fraction(seq, ["TATAWA"])
            elif feature == "polya_hexamer_score":
                start, end = transcript_relative_interval(position, -50, 0, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _max_consensus_fraction(
                    seq, ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA"]
                )
            else:
                start, end = transcript_relative_interval(position, 0, 50, strand)
                seq = _fetch_oriented(fasta, chrom, start, end, strand)
                record[feature] = _fraction(seq, {"T", "G"})
            if len(seq) != end - start:
                record[f"{feature}_available"] = False
                record[feature] = 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _cis_manifest() -> CISFeatureManifest:
    scanner = {
        "edge_gc_fraction": ("direct_base_fraction", "edge interval"),
        "donor_strength": ("fixed_consensus_fraction", "donor -3:+6 transcript-oriented"),
        "acceptor_strength": ("fixed_consensus_fraction", "acceptor -20:+3 transcript-oriented"),
        "branchpoint_score": ("fixed_consensus_fraction", "acceptor -50:-5 transcript-oriented"),
        "polypyrimidine_tract_score": ("direct_base_fraction", "acceptor -25:-3 transcript-oriented"),
        "tss_core_promoter_score": ("fixed_consensus_fraction", "TSS -40:+10 transcript-oriented"),
        "polya_hexamer_score": ("fixed_consensus_fraction", "PAS -50:0 transcript-oriented"),
        "pas_downstream_u_gu_fraction": ("direct_base_fraction", "PAS 0:+50 transcript-oriented"),
    }
    return CISFeatureManifest(
        reference_build="GRCh38",
        strand_convention="all sequence windows are transcript-oriented",
        sequence_features=tuple(
            CISSequenceFeatureSpec(
                feature_name=feature,
                scanner_name=scanner[feature][0],
                scanner_version="fabric.real_dataset.v1",
                sequence_window=scanner[feature][1],
                fixed_transform="identity final score",
            )
            for feature in SEQUENCE_FEATURES
        ),
        normalization=CISNormalizationPolicy(),
    )


def build_cis_stage(paths: Mapping[str, Path], output: Path) -> None:
    graph_root, cis_root = output / "graph", output / "cis"
    cis_root.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(graph_root / "edge_table.parquet")
    nodes = pd.read_parquet(graph_root / "node_table.parquet")
    scores = _cis_sequence_scores(edges, nodes, paths["reference_fasta"])
    structural = edges.rename(columns={"gene_id": "target_gene_id"})
    manifest = _cis_manifest()
    raw = build_explicit_cis_table(structural, scores, manifest=manifest)
    g_fit = pd.read_csv(paths["g_fit"], sep="\t")["target_gene_id"].astype(str)
    fitted = fit_cis_normalization(
        raw, train_admitted_gene_ids=tuple(g_fit), manifest=manifest
    )
    normalized = apply_cis_normalization(raw, normalization=fitted, manifest=manifest)
    scores.to_parquet(cis_root / "sequence_scores.parquet", index=False)
    raw.to_frame().to_parquet(cis_root / "raw_cis_features.parquet", index=False)
    normalized.to_frame().to_parquet(
        cis_root / "normalized_cis_features.parquet", index=False
    )
    record = {
        "schema_version": "fabric.real_cis_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_fasta": str(paths["reference_fasta"]),
        "reference_build": "GRCh38",
        "train_only_population": "17,600 G_fit unique structural edges, each once",
        "candidate_edge_count": len(edges),
        "g_fit_gene_count": len(g_fit),
        "model_feature_order": list(normalized.column_names),
        "model_design_rank_closure": {
            "population": "train_admitted_unique_structural_edges",
            "explicit_model_bias_included": True,
            "candidate_column_count": len(raw.column_names),
            "retained_column_count": len(normalized.column_names),
            "status": "FULL_COLUMN_RANK_WITH_BIAS",
        },
        "sequence_feature_specs": [asdict(value) for value in manifest.sequence_features],
        "normalization_statistics": [asdict(value) for value in fitted.statistics],
        "test_rows_or_test_outcomes_read": False,
    }
    (cis_root / "CISManifest.json").write_text(json.dumps(record, indent=2) + "\n")


def _admissible_atac_neighbors(
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    maximum_distance: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only absolute-distance-admitted neighbors and renormalize them."""

    distance_row = np.asarray(distances, dtype=np.float64)
    index_row = np.asarray(indices, dtype=np.int64)
    if distance_row.ndim != 1 or index_row.shape != distance_row.shape:
        raise ValueError("ATAC neighbor distances and indices must be aligned vectors")
    legal = np.isfinite(distance_row) & (distance_row <= maximum_distance)
    admitted_distances = distance_row[legal]
    admitted_indices = index_row[legal]
    if not len(admitted_distances):
        return admitted_distances, admitted_indices, np.empty(0, dtype=np.float64)
    raw_weight = np.exp(-admitted_distances / temperature)
    weights = raw_weight / raw_weight.sum()
    return admitted_distances, admitted_indices, weights


def build_cell_context_stage(paths: Mapping[str, Path], output: Path) -> None:
    """Freeze train/validation RNA-to-ATAC neighbors in exact biological strata."""

    import anndata as ad

    context_root = output / "cell_context"
    context_root.mkdir(parents=True, exist_ok=True)
    split = pd.read_parquet(paths["split_rows"], columns=["cell_id", "split"])
    selected = split["split"].astype(str).isin(["train", "val"])
    target = split.loc[selected].copy().reset_index(drop=True)
    if len(target) != EXPECTED["train_cell_count"] + EXPECTED["val_cell_count"]:
        raise ValueError("train+validation target cell count drift")

    rna = ad.read_h5ad(paths["rna_counts"], backed="r")
    rna_ids = pd.Index("RNA__" + rna.obs_names.astype(str))
    if len(rna_ids) != EXPECTED["cell_count"] or rna_ids.has_duplicates:
        raise ValueError("RNA count cell axis is not the frozen 217,933-cell identity")
    if set(rna_ids) != set(split["cell_id"].astype(str)):
        raise ValueError("RNA count cells differ from the frozen split")
    metadata = rna.obs.copy().reset_index(drop=True)
    metadata.insert(0, "cell_id", rna_ids)
    metadata = split.merge(metadata, on="cell_id", how="left", validate="one_to_one")
    if metadata["developmental_system"].isna().any() or metadata["stage"].isna().any():
        raise ValueError("RNA cell metadata is incomplete on the frozen cell axis")
    metadata.to_parquet(context_root / "cell_metadata.parquet", index=False)

    glue = ad.read_h5ad(paths["rna_atac_glue_embedding"], backed="r")
    glue_obs = glue.obs.copy().reset_index(drop=True)
    glue_obs.insert(0, "glue_cell_id", glue.obs_names.astype(str))
    modalities = glue_obs["modality"].astype(str)
    rna_mask = modalities.eq("RNA").to_numpy()
    atac_mask = modalities.eq("ATAC").to_numpy()
    if int(rna_mask.sum()) != 205_864 or int(atac_mask.sum()) != 232_474:
        raise ValueError("GLUE RNA/ATAC modality counts drift")
    embedding = np.asarray(glue.obsm["X_glue"], dtype=np.float32)
    if embedding.shape != (438_338, 50) or not np.isfinite(embedding).all():
        raise ValueError("GLUE embedding is not finite full shape [438338,50]")

    glue_rna = glue_obs.loc[rna_mask].copy()
    glue_rna["cell_id"] = glue_rna["glue_cell_id"].astype(str)
    if glue_rna["cell_id"].duplicated().any() or not set(glue_rna["cell_id"]).issubset(
        set(split["cell_id"].astype(str))
    ):
        raise ValueError("GLUE RNA identities are not a unique subset of frozen RNA cells")
    # Freeze the stage-to-window transformation from overlapping RNA metadata;
    # it is never inferred from held-out compatible outcomes.
    stage_map_rows = metadata[["cell_id", "stage"]].merge(
        glue_rna[["cell_id", "stage_window"]],
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )
    stage_map = stage_map_rows.groupby("stage", sort=True)["stage_window"].agg(
        lambda values: sorted(set(map(str, values)))
    )
    ambiguous = {str(key): value for key, value in stage_map.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"RNA stage maps to multiple GLUE stage windows: {ambiguous}")
    stage_to_window = {str(key): value[0] for key, value in stage_map.items()}
    target_meta = target.merge(
        metadata[["cell_id", "stage", "developmental_system"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    target_meta["stage_window"] = target_meta["stage"].astype(str).map(stage_to_window)
    if target_meta["stage_window"].isna().any():
        missing = sorted(set(target_meta.loc[target_meta["stage_window"].isna(), "stage"]))
        raise ValueError(f"target RNA stages lack a frozen GLUE window: {missing}")

    glue_rna_position = {
        value: index for index, value in enumerate(glue_rna["cell_id"].astype(str))
    }
    glue_rna_embedding = embedding[rna_mask]
    glue_atac = glue_obs.loc[atac_mask].copy().reset_index(drop=True)
    glue_atac_embedding = embedding[atac_mask]
    atac = ad.read_h5ad(paths["atac_peak_counts"], backed="r")
    atac_ids = tuple(atac.obs_names.astype(str))
    glue_atac_ids = tuple(
        value.removeprefix("ATAC__")
        for value in glue_atac["glue_cell_id"].astype(str)
    )
    if glue_atac_ids != atac_ids:
        raise ValueError("GLUE ATAC axis differs from the peak-count matrix axis")
    if set(glue_atac["developmental_system"].astype(str)) != {"Unknown"}:
        raise ValueError(
            "combined GLUE ATAC developmental_system is neither the documented "
            "Unknown placeholder nor a reviewed usable label"
        )
    peak_system = atac.obs["developmental_system"].astype(str).to_numpy()
    if np.any(pd.isna(peak_system)) or set(peak_system) == {"Unknown"}:
        raise ValueError("peak-count ATAC developmental-system labels are unavailable")
    glue_atac["developmental_system"] = peak_system

    neighbor_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    k = 30
    target_meta["has_glue_embedding"] = target_meta["cell_id"].astype(str).isin(
        glue_rna_position
    )
    for (system, window), group in target_meta.groupby(
        ["developmental_system", "stage_window"], sort=True, dropna=False
    ):
        atlas_selector = (
            glue_atac["developmental_system"].astype(str).eq(str(system))
            & glue_atac["stage_window"].astype(str).eq(str(window))
        ).to_numpy()
        atlas_rows = np.flatnonzero(atlas_selector)
        embedded = group.loc[group["has_glue_embedding"]].copy()
        if len(atlas_rows) < k:
            for row in group.itertuples(index=False):
                audit_rows.append(
                    {
                        "cell_id": str(row.cell_id),
                        "split": str(row.split),
                        "developmental_system": str(system),
                        "stage_window": str(window),
                        "has_glue_embedding": bool(row.has_glue_embedding),
                        "atlas_cell_count": len(atlas_rows),
                        "nearest_distance": np.nan,
                        "distance_threshold": np.nan,
                        "mapping_valid": False,
                        "failure_reason": "stratum_has_fewer_than_30_atac_cells",
                    }
                )
            continue
        model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        model.fit(glue_atac_embedding[atlas_rows])
        if embedded.empty:
            distances = np.empty((0, k), dtype=np.float32)
            indices = np.empty((0, k), dtype=np.int64)
        else:
            query_rows = [glue_rna_position[value] for value in embedded["cell_id"].astype(str)]
            distances, local_indices = model.kneighbors(glue_rna_embedding[query_rows])
            indices = atlas_rows[local_indices]
        train_nearest = distances[
            embedded["split"].astype(str).eq("train").to_numpy(), 0
        ]
        if not len(train_nearest):
            raise ValueError(f"GLUE stratum has no train RNA query: {system}/{window}")
        distance_threshold = float(np.quantile(train_nearest, 0.99))
        temperature = float(np.median(train_nearest))
        temperature = max(temperature, 1.0e-6)
        thresholds.append(
            {
                "developmental_system": str(system),
                "stage_window": str(window),
                "train_embedded_rna_count": int(len(train_nearest)),
                "atac_atlas_count": int(len(atlas_rows)),
                "nearest_distance_q99": distance_threshold,
                "distance_weight_temperature": temperature,
            }
        )
        for source_row, distance_row, index_row in zip(
            embedded.itertuples(index=False), distances, indices, strict=True
        ):
            legal_distances, legal_indices, weights = _admissible_atac_neighbors(
                distance_row,
                index_row,
                maximum_distance=distance_threshold,
                temperature=temperature,
            )
            valid = bool(len(legal_distances))
            if valid:
                neighbor_rows.append(
                    pd.DataFrame(
                        {
                            "cell_id": str(source_row.cell_id),
                            "split": str(source_row.split),
                            "developmental_system": str(system),
                            "stage_window": str(window),
                            "neighbor_rank": np.arange(len(legal_distances), dtype=np.int16),
                            "atac_cell_id": [atac_ids[index] for index in legal_indices],
                            "atac_matrix_row_0based": legal_indices.astype(np.int64),
                            "distance": legal_distances.astype(np.float32),
                            "weight": weights.astype(np.float32),
                            "mapping_valid": True,
                        }
                    )
                )
            ess = 1.0 / float(np.square(weights).sum()) if valid else np.nan
            audit_rows.append(
                {
                    "cell_id": str(source_row.cell_id),
                    "split": str(source_row.split),
                    "developmental_system": str(system),
                    "stage_window": str(window),
                    "has_glue_embedding": True,
                    "atlas_cell_count": len(atlas_rows),
                    "nearest_distance": float(distance_row[0]),
                    "distance_threshold": distance_threshold,
                    "candidate_neighbor_count": k,
                    "valid_neighbor_count": len(legal_distances),
                    "ess_atac": ess,
                    "maximum_neighbor_weight": float(weights.max()) if valid else np.nan,
                    "weighted_mean_distance": (
                        float(np.dot(weights, legal_distances)) if valid else np.nan
                    ),
                    "mapping_valid": valid,
                    "failure_reason": "" if valid else "nearest_distance_above_train_q99",
                }
            )
        missing_embedding = group.loc[~group["has_glue_embedding"]]
        for row in missing_embedding.itertuples(index=False):
            audit_rows.append(
                {
                    "cell_id": str(row.cell_id),
                    "split": str(row.split),
                    "developmental_system": str(system),
                    "stage_window": str(window),
                    "has_glue_embedding": False,
                    "atlas_cell_count": len(atlas_rows),
                    "nearest_distance": np.nan,
                    "distance_threshold": distance_threshold,
                    "mapping_valid": False,
                    "failure_reason": "rna_cell_absent_from_frozen_glue_embedding",
                }
            )
    neighbors = pd.concat(neighbor_rows, ignore_index=True)
    audit = pd.DataFrame(audit_rows).sort_values("cell_id", kind="mergesort")
    if len(audit) != len(target_meta) or audit["cell_id"].astype(str).duplicated().any():
        raise ValueError("ATAC mapping audit does not cover each train/validation cell once")
    if set(neighbors["split"].astype(str)) - {"train", "val"}:
        raise ValueError("ATAC neighbor table contains held-out test cells")
    valid_counts = neighbors.groupby("cell_id", sort=False).size()
    expected_valid = set(
        audit.loc[
            audit["mapping_valid"].astype(bool),
            "cell_id",
        ].astype(str)
    )
    if (
        set(valid_counts.index.astype(str)) != expected_valid
        or bool((valid_counts < 1).any())
        or bool((valid_counts > k).any())
    ):
        raise ValueError("valid RNA targets do not have 1..30 admitted ATAC neighbors")
    weight_sums = neighbors.groupby("cell_id", sort=False)["weight"].sum()
    if not np.allclose(weight_sums.to_numpy(np.float64), 1.0, atol=1.0e-6, rtol=0):
        raise ValueError("admitted ATAC neighbor weights do not sum to one")
    neighbors.to_parquet(context_root / "rna_atac_neighbors.parquet", index=False)
    audit.to_parquet(context_root / "ATACMappingAudit.parquet", index=False)
    pd.DataFrame(thresholds).to_parquet(
        context_root / "ATACMappingTrainThresholds.parquet", index=False
    )
    manifest = {
        "schema_version": "fabric.atac_mapping_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_population": "train_validation_only",
        "target_cell_count": len(target_meta),
        "target_train_cell_count": int(target_meta["split"].eq("train").sum()),
        "target_validation_cell_count": int(target_meta["split"].eq("val").sum()),
        "target_test_cell_count": 0,
        "maximum_candidate_k": k,
        "admitted_k_policy": "1..30 neighbors within the frozen absolute distance range",
        "strata": ["developmental_system", "stage_window"],
        "stratum_field_sources": {
            "developmental_system": "atac_peak_counts.obs.developmental_system",
            "stage_window": "rna_atac_glue_embedding.obs.stage_window",
        },
        "distance": "euclidean_in_frozen_50d_X_glue",
        "train_only_threshold": "per-stratum q99 nearest RNA-to-ATAC distance",
        "weight": (
            "exp(-distance/train_median_nearest_distance), normalized only across "
            "distance-admitted neighbors per RNA cell"
        ),
        "missing_rna_glue_embedding_policy": "explicit mapping_invalid, zero ATAC gate mask",
        "valid_mapping_count": int(audit["mapping_valid"].sum()),
        "invalid_mapping_count": int((~audit["mapping_valid"]).sum()),
        "test_compatible_rows_or_predictions_read": False,
    }
    (context_root / "ATACMappingManifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    rna.file.close()
    glue.file.close()
    atac.file.close()


def _gene_symbol_axis(gtf: Path) -> tuple[dict[str, str], dict[str, str]]:
    symbol_to_id: dict[str, str] = {}
    id_to_symbol: dict[str, str] = {}
    with gtf.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = dict(_ATTRIBUTE.findall(fields[8]))
            gene_id = attrs.get("gene_id", "").split(".")[0]
            symbol = attrs.get("gene_name", "")
            if not gene_id or not symbol:
                continue
            key = symbol.upper()
            if key in symbol_to_id and symbol_to_id[key] != gene_id:
                # An ambiguous symbol is not a lawful expression-axis mapping.
                symbol_to_id[key] = ""
            else:
                symbol_to_id[key] = gene_id
            id_to_symbol[gene_id] = symbol
    return symbol_to_id, id_to_symbol


def _jaspar_symbols(name: str) -> list[str]:
    cleaned = re.sub(r"\([^)]*\)", "", str(name)).strip()
    return [value for value in cleaned.split("::") if value]


def _is_jaspar_heterodimer(name: str) -> bool:
    return "::" in re.sub(r"\([^)]*\)", "", str(name))


def build_factor_stage(paths: Mapping[str, Path], output: Path) -> None:
    """Map DNA/RNA motif libraries to the frozen RNA Ensembl-gene axis."""

    import anndata as ad

    event_root = output / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    rna = ad.read_h5ad(paths["rna_counts"], backed="r")
    rna_axis = tuple(value.split(".")[0] for value in rna.var_names.astype(str))
    if len(rna_axis) != 32_351 or len(set(rna_axis)) != len(rna_axis):
        raise ValueError("RNA gene axis is not 32,351 unique Ensembl IDs")
    axis_set = set(rna_axis)
    symbol_to_id, id_to_symbol = _gene_symbol_axis(paths["rna_gene_gtf"])

    dna_index = pd.read_csv(paths["dna_motif_index"], sep="\t")
    dna_library = parse_meme_motifs(paths["dna_motif_library"])
    if len(dna_index) != len(dna_library) or set(dna_index["motif_id"].astype(str)) != set(
        dna_library
    ):
        raise ValueError("JASPAR index and MEME library identities differ")
    mapping_rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for row in dna_index.itertuples(index=False):
        motif_id, name = str(row.motif_id), str(row.tf_name)
        symbols = _jaspar_symbols(name)
        if _is_jaspar_heterodimer(name):
            excluded.append(
                {
                    "modality": "DNA",
                    "motif_id": motif_id,
                    "source_name": name,
                    "candidate_symbols": symbols,
                    "reason": "heterodimer_complex_unmodeled",
                }
            )
            continue
        gene_ids = [symbol_to_id.get(value.upper(), "") for value in symbols]
        if not symbols or any(not value or value not in axis_set for value in gene_ids):
            excluded.append(
                {
                    "modality": "DNA",
                    "motif_id": motif_id,
                    "source_name": name,
                    "candidate_symbols": symbols,
                    "reason": "complete_factor_identity_not_on_frozen_rna_axis",
                }
            )
            continue
        if len(set(gene_ids)) != len(gene_ids):
            excluded.append(
                {
                    "modality": "DNA",
                    "motif_id": motif_id,
                    "source_name": name,
                    "candidate_symbols": symbols,
                    "reason": "compound_factor_members_not_unique",
                }
            )
            continue
        if len(gene_ids) == 1:
            entity = gene_ids[0]
            kind = "unique"
            candidate_ids = [entity]
        else:
            candidate_ids = sorted(gene_ids)
            entity = "factor_group:" + "+".join(candidate_ids)
            kind = "factor_equivalence_group"
        mapping_rows.append(
            {
                "modality": "DNA",
                "motif_id": motif_id,
                "factor_identity_kind": kind,
                "factor_entity_id": entity,
                "candidate_factor_ids": candidate_ids,
                "activity_entity_id": entity,
                "activity_gene_ids": candidate_ids,
                "activity_proxy_rule": (
                    "single_frozen_rna_gene_log1p_activity"
                    if kind == "unique"
                    else "sum_frozen_member_rna_counts_then_log1p"
                ),
                "motif_equivalence_family_id": motif_id,
                "source_priority": 0,
                "source_local_rank": np.nan,
                "source_name": name,
            }
        )

    rna_map = pd.read_csv(paths["rna_motif_gene_map"], sep="\t")
    rna_library = parse_cisbp_motifs(
        paths["rna_motif_directory"], motif_ids=tuple(rna_map["motif_id"].astype(str))
    )
    if set(rna_library) != set(rna_map["motif_id"].astype(str)):
        raise ValueError("CisBP-RNA map and PWM library identities differ")
    for motif_id, motif_rows in rna_map.groupby("motif_id", sort=False):
        motif_id = str(motif_id)
        gene_ids = sorted(
            set(motif_rows["gene_id"].astype(str).str.split(".").str[0])
        )
        source_names = sorted(set(motif_rows["rbp_gene"].astype(str)))
        if not gene_ids or any(gene_id not in axis_set for gene_id in gene_ids):
            excluded.append(
                {
                    "modality": "RNA",
                    "motif_id": motif_id,
                    "source_name": ";".join(source_names),
                    "candidate_symbols": source_names,
                    "reason": "complete_rbp_group_not_on_frozen_rna_axis",
                }
            )
            continue
        if len(gene_ids) == 1:
            entity = gene_ids[0]
            kind = "unique"
        else:
            entity = "factor_group:" + "+".join(gene_ids)
            kind = "factor_equivalence_group"
        mapping_rows.append(
            {
                "modality": "RNA",
                "motif_id": motif_id,
                "factor_identity_kind": kind,
                "factor_entity_id": entity,
                "candidate_factor_ids": gene_ids,
                "activity_entity_id": entity,
                "activity_gene_ids": gene_ids,
                "activity_proxy_rule": (
                    "single_frozen_rna_gene_log1p_activity"
                    if kind == "unique"
                    else "sum_frozen_member_rna_counts_then_log1p"
                ),
                "motif_equivalence_family_id": motif_id,
                "source_priority": (
                    0
                    if motif_rows["evidence_status"].astype(str).eq("direct").any()
                    else 1
                ),
                "source_local_rank": np.nan,
                "source_name": ";".join(source_names),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    catalog = build_factor_catalog(mapping, frozen_rna_gene_axis=rna_axis)
    if not bool(catalog.motif_mapping["source_valid"].all()):
        raise ValueError("admitted motif mapping contains an invalid RNA activity axis")
    catalog.motif_mapping.to_parquet(event_root / "motif_factor_mapping.parquet", index=False)
    catalog.factors.to_parquet(event_root / "factor_catalog.parquet", index=False)
    pd.DataFrame(excluded).to_parquet(event_root / "excluded_motifs.parquet", index=False)
    manifest = {
        "schema_version": "fabric.factor_catalog_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rna_gene_axis_source": str(paths["rna_counts"]),
        "rna_gene_axis_count": len(rna_axis),
        "dna_library_motif_count": len(dna_library),
        "rna_library_motif_count": len(rna_library),
        "admitted_dna_motif_count": int(mapping["modality"].eq("DNA").sum()),
        "admitted_rna_motif_count": int(mapping["modality"].eq("RNA").sum()),
        "excluded_motif_count": len(excluded),
        "compound_factor_policy": "all named members must map; groups stay groups",
        "partial_group_fallback": False,
    }
    (event_root / "FactorCatalogManifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    rna.file.close()


def build_activity_stage(paths: Mapping[str, Path], output: Path) -> None:
    """Materialize real CP10K-log1p factor/group activities without gene filtering."""

    import anndata as ad
    from scipy import sparse

    context_root = output / "cell_context"
    factor_path = output / "events" / "factor_catalog.parquet"
    if not factor_path.is_file():
        raise FileNotFoundError("factor stage must complete before RNA activity")
    factors = pd.read_parquet(factor_path).sort_values(
        "activity_entity_id", kind="mergesort"
    ).reset_index(drop=True)
    split = pd.read_parquet(paths["split_rows"], columns=["cell_id", "split"])
    rna = ad.read_h5ad(paths["rna_counts"], backed="r")
    source_ids = tuple("RNA__" + rna.obs_names.astype(str))
    source_axis = pd.DataFrame(
        {
            "cell_id": source_ids,
            "source_rna_row_0based": np.arange(len(source_ids), dtype=np.int64),
        }
    )
    target = source_axis.merge(split, on="cell_id", how="left", validate="one_to_one")
    if target["split"].isna().any():
        raise ValueError("RNA source axis contains cells absent from the frozen split")
    target = target.loc[target["split"].astype(str).isin(["train", "val"])].copy()
    target_ids = tuple(target["cell_id"].astype(str))
    gene_axis = tuple(value.split(".")[0] for value in rna.var_names.astype(str))
    gene_index = {value: index for index, value in enumerate(gene_axis)}
    membership_rows: list[int] = []
    membership_columns: list[int] = []
    for entity_column, row in enumerate(factors.itertuples(index=False)):
        members = [str(value) for value in row.activity_gene_ids]
        missing = sorted(set(members) - set(gene_index))
        if missing or not bool(row.source_valid):
            raise ValueError(
                f"admitted activity entity lacks source genes: {row.activity_entity_id}/{missing}"
            )
        membership_rows.extend(gene_index[value] for value in members)
        membership_columns.extend([entity_column] * len(members))
    membership = sparse.csc_matrix(
        (
            np.ones(len(membership_rows), dtype=np.float64),
            (membership_rows, membership_columns),
        ),
        shape=(len(gene_axis), len(factors)),
    )

    values_path = context_root / "rna_activity_cp10k_log1p.npy"
    observed_path = context_root / "rna_activity_observed.npy"
    library_path = context_root / "rna_library_size.npy"
    values = np.lib.format.open_memmap(
        values_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(target_ids), len(factors)),
    )
    observed = np.lib.format.open_memmap(
        observed_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(target_ids),),
    )
    library_size = np.lib.format.open_memmap(
        library_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(target_ids),),
    )
    # Iterate contiguous source ranges and then remove test rows locally.  This
    # preserves the test-blind output while avoiding pathological random HDF5
    # access from split-table order.
    chunk_size = 4096
    output_start = 0
    split_by_source = source_axis.merge(
        split, on="cell_id", how="left", validate="one_to_one"
    )["split"].astype(str).to_numpy()
    for source_start in range(0, len(source_ids), chunk_size):
        source_stop = min(len(source_ids), source_start + chunk_size)
        include = np.isin(split_by_source[source_start:source_stop], ["train", "val"])
        chunk = sparse.csr_matrix(rna.X[source_start:source_stop], dtype=np.float64)[include]
        stop = output_start + chunk.shape[0]
        if chunk.data.size and (
            not np.isfinite(chunk.data).all() or bool((chunk.data < 0).any())
        ):
            raise ValueError("RNA count matrix contains non-finite or negative counts")
        totals = np.asarray(chunk.sum(axis=1)).reshape(-1)
        valid = totals > 0
        library_size[output_start:stop] = totals
        observed[output_start:stop] = valid
        raw_entity = (chunk @ membership).toarray()
        transformed = np.zeros(raw_entity.shape, dtype=np.float32)
        transformed[valid] = np.log1p(
            10_000.0 * raw_entity[valid] / totals[valid, None]
        ).astype(np.float32)
        values[output_start:stop] = transformed
        output_start = stop
    if output_start != len(target_ids):
        raise RuntimeError("RNA activity output traversal did not close on target axis")
    values.flush()
    observed.flush()
    library_size.flush()
    del values, observed, library_size
    target.assign(activity_row_0based=np.arange(len(target), dtype=np.int64)).to_parquet(
        context_root / "rna_activity_cell_axis.parquet", index=False
    )
    factors[[
        "activity_entity_id",
        "factor_identity_kind",
        "activity_gene_ids",
        "activity_proxy_rule",
    ]].assign(activity_column_0based=np.arange(len(factors), dtype=np.int64)).to_parquet(
        context_root / "rna_activity_entity_axis.parquet", index=False
    )
    record = {
        "schema_version": "fabric.rna_activity_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_population": "train_validation_only",
        "cell_count": len(target_ids),
        "entity_count": len(factors),
        "full_library_gene_denominator_count": len(gene_axis),
        "normalization": "log1p(10000 * entity_raw_count_sum / full_library_size)",
        "group_rule": "sum raw member counts before CP10K-log1p",
        "observed_zero_rule": "zero is observed when full RNA library size is positive",
        "test_cells_or_compatible_rows_read": False,
    }
    (context_root / "RNAActivityManifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    rna.file.close()


def _load_real_graphs(graph_root: Path) -> Iterable[object]:
    nodes = pd.read_parquet(graph_root / "node_table.parquet")
    edges = pd.read_parquet(graph_root / "edge_table.parquet")
    paths = pd.read_parquet(graph_root / "path_table.parquet")
    path_edges = pd.read_parquet(graph_root / "path_edge_table.parquet")
    node_groups = {str(key): value for key, value in nodes.groupby("gene_id", sort=False)}
    edge_groups = {str(key): value for key, value in edges.groupby("gene_id", sort=False)}
    path_groups = {str(key): value for key, value in paths.groupby("gene_id", sort=False)}
    path_edge_groups = {
        str(key): value for key, value in path_edges.groupby("gene_id", sort=False)
    }
    for gene_id in paths["gene_id"].astype(str).drop_duplicates():
        yield build_gene_graph(
            gene_id,
            nodes=node_groups[gene_id],
            edges=edge_groups[gene_id],
            paths=path_groups[gene_id],
            path_edges=path_edge_groups[gene_id],
        )


def _ordered_peak_interval_arrays(
    peaks: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the arrays required by the frozen monotone interval query."""

    starts = peaks["start"].to_numpy(np.int64)
    ends = peaks["end"].to_numpy(np.int64)
    if len(starts) > 1 and (
        bool((starts[1:] < starts[:-1]).any())
        or bool((ends[1:] < ends[:-1]).any())
    ):
        raise ValueError(
            "consensus peak interval query requires monotone starts and ends"
        )
    return starts, ends, peaks["peak_row_0based"].to_numpy(np.int64)


def build_anchor_stage(paths: Mapping[str, Path], output: Path) -> None:
    """Freeze graph-defined DNA/RNA windows and exact consensus-peak membership."""

    import anndata as ad

    graph_root = output / "graph"
    event_root = output / "events"
    if not (graph_root / "GraphManifest.json").is_file():
        raise FileNotFoundError("graph stage must complete before event anchors")
    event_root.mkdir(parents=True, exist_ok=True)
    contig_lengths: dict[str, int] = {}
    with paths["reference_fasta_index"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            contig_lengths[fields[0]] = int(fields[1])
    dna_flanks = {"TSS": (2_000, 500), "donor": (250, 250), "acceptor": (250, 250), "PAS": (500, 2_000)}
    rna_flanks = {"TSS": (250, 250), "donor": (250, 250), "acceptor": (250, 250), "PAS": (250, 250)}
    anchors: list[pd.DataFrame] = []
    for graph in _load_real_graphs(graph_root):
        positions = graph.nodes["pos_0based"].to_numpy(np.int64)
        bounds = (int(positions.min()), int(positions.max()) + 1)
        anchors.append(
            build_graph_anchor_regions(
                graph,
                modality="DNA",
                site_flanks=dna_flanks,
                maximum_short_exon_bp=500,
                contig_lengths=contig_lengths,
                gene_bounds=bounds,
            )
        )
        anchors.append(
            build_graph_anchor_regions(
                graph,
                modality="RNA",
                site_flanks=rna_flanks,
                maximum_short_exon_bp=500,
                contig_lengths=contig_lengths,
                gene_bounds=bounds,
            )
        )
    anchor_table = pd.concat(anchors, ignore_index=True)
    if anchor_table.duplicated(
        ["target_gene_id", "modality", "anchor_region_id", "anchor_site_id", "edge_id"]
    ).any():
        raise ValueError("graph anchor carrier identities are duplicated")
    anchor_table.to_parquet(event_root / "graph_anchor_regions.parquet", index=False)

    peak_bed = pd.read_csv(
        paths["consensus_peak_bed"],
        sep="\t",
        header=None,
        names=["chromosome", "start", "end"],
    )
    peak_bed["peak_id"] = (
        peak_bed["chromosome"].astype(str)
        + ":"
        + peak_bed["start"].astype(str)
        + "-"
        + peak_bed["end"].astype(str)
    )
    peak_bed["peak_row_0based"] = np.arange(len(peak_bed), dtype=np.int64)
    peak_bed["peak_support"] = np.float32(1.0)
    if len(peak_bed) != 753_753 or peak_bed["peak_id"].duplicated().any():
        raise ValueError("consensus peak catalog is not 753,753 unique intervals")
    atac = ad.read_h5ad(paths["atac_peak_counts"], backed="r")
    if tuple(peak_bed["peak_id"].astype(str)) != tuple(atac.var_names.astype(str)):
        raise ValueError("consensus BED order differs from the ATAC peak matrix axis")
    atac.file.close()

    assignments: set[tuple[str, int]] = set()
    dna = anchor_table.loc[anchor_table["modality"].astype(str).eq("DNA")]
    peak_by_chrom = {
        str(chrom): group.sort_values("start", kind="mergesort")
        for chrom, group in peak_bed.groupby("chromosome", sort=False)
    }
    arrays = {
        chrom: _ordered_peak_interval_arrays(frame)
        for chrom, frame in peak_by_chrom.items()
    }
    for row in dna.itertuples(index=False):
        chrom = str(row.chromosome)
        if chrom not in arrays:
            continue
        starts, ends, peak_rows = arrays[chrom]
        left = int(np.searchsorted(ends, int(row.region_start), side="right"))
        right = int(np.searchsorted(starts, int(row.region_end), side="left"))
        for peak_row in peak_rows[left:right]:
            assignments.add((str(row.target_gene_id), int(peak_row)))
    assigned = pd.DataFrame(
        sorted(assignments), columns=["target_gene_id", "peak_row_0based"]
    ).merge(peak_bed, on="peak_row_0based", how="left", validate="many_to_one")
    if assigned.empty or assigned.duplicated(["target_gene_id", "peak_id"]).any():
        raise ValueError("DNA peak-to-gene assignments are empty or duplicated")
    assigned["strand"] = assigned["target_gene_id"].map(
        dna.drop_duplicates("target_gene_id").set_index("target_gene_id")["strand"]
    )
    if assigned["strand"].isna().any():
        raise ValueError("DNA peak assignment lacks target-gene strand")
    assigned.to_parquet(event_root / "dna_peak_gene_assignments.parquet", index=False)
    peak_bed.to_parquet(event_root / "consensus_peaks.parquet", index=False)
    record = {
        "schema_version": "fabric.motif_routing_window_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": "GRCh38 0-based half-open",
        "dna_site_flanks_transcript_oriented": {key: list(value) for key, value in dna_flanks.items()},
        "rna_site_flanks_transcript_oriented": {key: list(value) for key, value in rna_flanks.items()},
        "rna_maximum_short_exon_bp": 500,
        "site_window_clipping": "reference_contig_only",
        "dna_edge_policy": "complete processing-edge interval, motif events require consensus-peak overlap",
        "consensus_peak_interval_query_contract": "start_and_end_monotone_validated_per_contig",
        "peak_support_semantics": "binary membership in the frozen shared consensus BED",
        "anchor_carrier_row_count": len(anchor_table),
        "dna_peak_gene_pair_count": len(assigned),
        "unique_assigned_peak_count": int(assigned["peak_id"].nunique()),
        "test_outcomes_read": False,
    }
    (event_root / "MotifRoutingWindowManifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def build_atac_normalization_stage(paths: Mapping[str, Path], output: Path) -> None:
    """Normalize every real ATAC cell once on the full 753,753-peak denominator."""

    import anndata as ad
    from scipy import sparse

    context_root = output / "cell_context"
    context_root.mkdir(parents=True, exist_ok=True)
    atac = ad.read_h5ad(paths["atac_peak_counts"])
    if atac.shape != (232_474, 753_753):
        raise ValueError("ATAC peak-count matrix full shape drift")
    normalized, library_size = normalize_log1p_counts(atac.X, target_sum=10_000.0)
    sparse.save_npz(
        context_root / "atac_cell_peak_cp10k_log1p.npz",
        normalized,
        compressed=True,
    )
    np.save(context_root / "atac_library_size.npy", library_size)
    pd.DataFrame(
        {
            "atac_cell_id": atac.obs_names.astype(str),
            "atac_row_0based": np.arange(atac.n_obs, dtype=np.int64),
        }
    ).to_parquet(context_root / "atac_cell_axis.parquet", index=False)
    pd.DataFrame(
        {
            "peak_id": atac.var_names.astype(str),
            "peak_column_0based": np.arange(atac.n_vars, dtype=np.int64),
        }
    ).to_parquet(context_root / "atac_peak_axis.parquet", index=False)
    record = {
        "schema_version": "fabric.atac_normalization_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_count": atac.n_obs,
        "peak_count": atac.n_vars,
        "normalization": "log1p(10000 * peak_count / full_cell_peak_library_size)",
        "full_peak_denominator": True,
        "mapped_to_rna_after_source_normalization": True,
        "runtime_sparse_layout": "mmap CSC arrays derived from the formal CSR artifact",
    }
    (context_root / "ATACNormalizationManifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def build_atac_csc_backing_stage(output: Path) -> None:
    """Materialize mmap-able CSC arrays for parallel read-only gate workers."""

    from scipy import sparse

    context_root = output / "cell_context"
    source = context_root / "atac_cell_peak_cp10k_log1p.npz"
    if not source.is_file():
        raise FileNotFoundError(f"normalized ATAC sparse source is absent: {source}")
    matrix = sparse.load_npz(source).tocsc()
    backing = context_root / "atac_csc_backing"
    backing.mkdir(parents=True, exist_ok=True)
    np.save(backing / "data.npy", matrix.data.astype(np.float32, copy=False))
    np.save(backing / "indices.npy", matrix.indices)
    np.save(backing / "indptr.npy", matrix.indptr)
    record = {
        "schema_version": "fabric.atac_csc_backing.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "semantics": "read-only mmap CSC of full-denominator CP10K-log1p ATAC",
    }
    (backing / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


def _load_atac_csc_backing(context_root: Path):
    from scipy import sparse

    backing = context_root / "atac_csc_backing"
    manifest = json.loads((backing / "manifest.json").read_text())
    data = np.load(backing / "data.npy", mmap_mode="r")
    indices = np.load(backing / "indices.npy", mmap_mode="r")
    indptr = np.load(backing / "indptr.npy", mmap_mode="r")
    matrix = sparse.csc_matrix(
        (data, indices, indptr), shape=tuple(manifest["shape"]), copy=False
    )
    if matrix.nnz != int(manifest["nnz"]):
        raise ValueError("ATAC CSC backing nnz differs from its manifest")
    return matrix


def _moods_scanner(motifs: Mapping[str, object], *, reverse_strand: bool):
    import MOODS.scan
    import MOODS.tools

    background = [0.25, 0.25, 0.25, 0.25]
    matrices = []
    thresholds = []
    identities: list[tuple[str, str, int, float, float]] = []
    for motif_id, motif in motifs.items():
        matrix = MOODS.tools.log_odds(
            np.asarray(motif.probabilities, dtype=np.float64).T.tolist(),
            background,
            0.1,
        )
        threshold = MOODS.tools.threshold_from_p(matrix, background, 1.0e-6)
        minimum = float(sum(min(column) for column in zip(*matrix)))
        maximum = float(sum(max(column) for column in zip(*matrix)))
        matrices.append(matrix)
        thresholds.append(threshold)
        identities.append((str(motif_id), "+", int(motif.width), minimum, maximum))
        if reverse_strand:
            reverse = MOODS.tools.reverse_complement(matrix)
            matrices.append(reverse)
            thresholds.append(threshold)
            identities.append((str(motif_id), "-", int(motif.width), minimum, maximum))
    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(matrices, background, thresholds)
    return scanner, identities


def _scan_segments(
    segments: pd.DataFrame,
    *,
    fasta: Fasta,
    scanner: object,
    scanner_identities: Sequence[tuple[str, str, int, float, float]],
    oriented_by_gene_strand: bool,
) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(
            columns=["segment_id", "motif_id", "start", "end", "hit_strand", "motif_score"]
        )
    pieces: list[str] = []
    concat_starts: list[int] = []
    concat_ends: list[int] = []
    metadata = []
    cursor = 0
    separator = "N" * 64
    for row in segments.itertuples(index=False):
        strand = str(row.strand) if oriented_by_gene_strand else "+"
        sequence = _fetch_oriented(
            fasta,
            str(row.chromosome),
            int(row.start),
            int(row.end),
            strand,
        )
        if len(sequence) != int(row.end) - int(row.start):
            raise ValueError("motif scan segment was clipped after anchor construction")
        concat_starts.append(cursor)
        concat_ends.append(cursor + len(sequence))
        metadata.append(row)
        pieces.append(sequence)
        pieces.append(separator)
        cursor += len(sequence) + len(separator)
    sequence = "".join(pieces)
    starts_array = np.asarray(concat_starts, dtype=np.int64)
    ends_array = np.asarray(concat_ends, dtype=np.int64)
    rows: list[dict[str, object]] = []
    results = scanner.scan(sequence)
    for identity, matches in zip(scanner_identities, results, strict=True):
        motif_id, hit_strand, width, minimum, maximum = identity
        denominator = max(maximum - minimum, 1.0e-12)
        for match in matches:
            offset = int(match.pos)
            segment_index = int(np.searchsorted(starts_array, offset, side="right") - 1)
            if segment_index < 0 or offset + width > ends_array[segment_index]:
                continue
            row = metadata[segment_index]
            local = offset - int(starts_array[segment_index])
            if oriented_by_gene_strand and str(row.strand) == "-":
                start = int(row.end) - local - width
                source_strand = "-"
            else:
                start = int(row.start) + local
                source_strand = hit_strand
            rows.append(
                {
                    "segment_id": str(row.segment_id),
                    "motif_id": motif_id,
                    "start": start,
                    "end": start + width,
                    "hit_strand": source_strand,
                    "motif_score": float((float(match.score) - minimum) / denominator),
                }
            )
    return pd.DataFrame(rows)


def _motif_source_hits(
    scanned: pd.DataFrame,
    mapping: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    target_gene_id: str,
    modality: str,
    gene_strand: str,
) -> pd.DataFrame:
    if scanned.empty:
        return pd.DataFrame()
    joined = scanned.merge(mapping, on="motif_id", how="left", validate="many_to_one")
    joined = joined.merge(
        segments[
            ["segment_id", "source_window_id", "chromosome", "peak_id", "peak_support"]
        ],
        on="segment_id",
        how="left",
        validate="many_to_one",
    )
    if joined["factor_entity_id"].isna().any() or joined["chromosome"].isna().any():
        raise ValueError("motif scan result lacks factor or segment provenance")
    joined["source_local_rank"] = joined.groupby(
        ["motif_id", "source_window_id"], sort=False
    )["motif_score"].rank(method="first", ascending=False)
    if modality == "RNA":
        joined["orientation"] = "transcribed"
    else:
        joined["orientation"] = np.where(
            joined["hit_strand"].astype(str).eq(gene_strand),
            "same_transcript",
            "opposite_transcript",
        )
    joined["target_gene_id"] = target_gene_id
    joined["modality"] = modality
    joined["strand"] = gene_strand
    joined["source_valid"] = True
    joined["calibrated_motif_quality"] = np.nan
    joined["source_hit_id"] = (
        modality
        + "|"
        + target_gene_id
        + "|"
        + joined["source_window_id"].astype(str)
        + "|"
        + joined["motif_id"].astype(str)
        + "|"
        + joined["start"].astype(str)
        + "|"
        + joined["hit_strand"].astype(str)
    )
    if joined["source_hit_id"].duplicated().any():
        raise ValueError("motif source-hit identities are duplicated")
    return joined[
        [
            "source_hit_id",
            "source_window_id",
            "target_gene_id",
            "factor_entity_id",
            "factor_identity_kind",
            "candidate_factor_ids",
            "activity_entity_id",
            "activity_gene_ids",
            "activity_proxy_rule",
            "modality",
            "motif_id",
            "motif_equivalence_family_id",
            "chromosome",
            "start",
            "end",
            "strand",
            "motif_score",
            "calibrated_motif_quality",
            "orientation",
            "peak_id",
            "peak_support",
            "source_priority",
            "source_local_rank",
            "source_valid",
        ]
    ]


def build_event_scan_stage(
    paths: Mapping[str, Path],
    output: Path,
    *,
    chromosome: str,
    pilot_gene_id: str | None = None,
    gene_shard_index: int | None = None,
    gene_shard_count: int | None = None,
) -> None:
    """Build one chromosome's real physical-event and post-cap route shard."""

    event_root = output / "events"
    if (gene_shard_index is None) != (gene_shard_count is None):
        raise ValueError("gene shard index/count must be provided together")
    if gene_shard_count is not None and (
        pilot_gene_id is not None
        or gene_shard_count < 1
        or gene_shard_index is None
        or not 0 <= gene_shard_index < gene_shard_count
    ):
        raise ValueError("event gene-shard specification is invalid")
    if pilot_gene_id is not None:
        shard_output_root = event_root / "pilot"
        suffix = f"{chromosome}-{pilot_gene_id}"
    elif gene_shard_count is not None:
        shard_output_root = event_root / "gene_chunks"
        suffix = (
            f"{chromosome}-genechunk-{gene_shard_index:02d}-of-{gene_shard_count:02d}"
        )
    else:
        shard_output_root = event_root
        suffix = chromosome
    completed_manifest = shard_output_root / "shard_manifests" / f"{suffix}.json"
    completed_tables = (
        "physical_events", "candidate_routes", "event_routes", "cap_audit",
        "route_degree_cap_audit", "catalog_burden",
    )
    if completed_manifest.is_file() and all(
        (shard_output_root / name / f"part-{suffix}.parquet").is_file()
        for name in completed_tables
    ):
        raise FileExistsError(
            "completed event shard already exists without a current-source/upstream "
            f"identity proof: {suffix}; use a fresh output root"
        )
    required = [
        event_root / "graph_anchor_regions.parquet",
        event_root / "dna_peak_gene_assignments.parquet",
        event_root / "motif_factor_mapping.parquet",
    ]
    missing = [str(value) for value in required if not value.is_file()]
    if missing:
        raise FileNotFoundError("event prerequisites are absent: " + "; ".join(missing))
    anchors = pd.read_parquet(required[0])
    anchors = anchors.loc[anchors["chromosome"].astype(str).eq(chromosome)].copy()
    if pilot_gene_id is not None:
        anchors = anchors.loc[
            anchors["target_gene_id"].astype(str).eq(str(pilot_gene_id))
        ].copy()
    if gene_shard_count is not None:
        gene_axis = tuple(anchors["target_gene_id"].astype(str).drop_duplicates())
        selected_gene_ids = gene_axis[gene_shard_index::gene_shard_count]
        anchors = anchors.loc[
            anchors["target_gene_id"].astype(str).isin(selected_gene_ids)
        ].copy()
    if anchors.empty:
        raise ValueError(f"no graph anchors exist on {chromosome}")
    assignments = pd.read_parquet(required[1])
    assignments = assignments.loc[
        assignments["chromosome"].astype(str).eq(chromosome)
    ].copy()
    if pilot_gene_id is not None:
        assignments = assignments.loc[
            assignments["target_gene_id"].astype(str).eq(str(pilot_gene_id))
        ].copy()
    if gene_shard_count is not None:
        assignments = assignments.loc[
            assignments["target_gene_id"].astype(str).isin(selected_gene_ids)
        ].copy()
    mapping = pd.read_parquet(required[2])
    dna_mapping = mapping.loc[mapping["modality"].astype(str).eq("DNA")].copy()
    rna_mapping = mapping.loc[mapping["modality"].astype(str).eq("RNA")].copy()
    dna_all = parse_meme_motifs(paths["dna_motif_library"])
    dna_motifs = {value: dna_all[value] for value in dna_mapping["motif_id"].astype(str)}
    rna_motifs = parse_cisbp_motifs(
        paths["rna_motif_directory"], motif_ids=tuple(rna_mapping["motif_id"].astype(str))
    )
    dna_scanner, dna_identities = _moods_scanner(dna_motifs, reverse_strand=True)
    rna_scanner, rna_identities = _moods_scanner(rna_motifs, reverse_strand=False)
    fasta = Fasta(str(paths["reference_fasta"]), as_raw=True, sequence_always_upper=True)

    unique_peaks = assignments.drop_duplicates("peak_id").sort_values(
        ["start", "end", "peak_id"], kind="mergesort"
    )
    dna_segments = unique_peaks.rename(columns={"peak_id": "segment_id"}).copy()
    dna_segments["source_window_id"] = "peak:" + dna_segments["segment_id"].astype(str)
    dna_segments["peak_id"] = dna_segments["segment_id"]
    dna_segments["strand"] = "+"
    dna_base = _scan_segments(
        dna_segments,
        fasta=fasta,
        scanner=dna_scanner,
        scanner_identities=dna_identities,
        oriented_by_gene_strand=False,
    )
    if not dna_base.empty:
        dna_base = dna_base.merge(dna_mapping, on="motif_id", how="left", validate="many_to_one")
        dna_base = dna_base.merge(
            dna_segments[["segment_id", "source_window_id", "chromosome", "peak_id", "peak_support"]],
            on="segment_id",
            how="left",
            validate="many_to_one",
        )
    dna_by_peak = (
        {str(key): value for key, value in dna_base.groupby("peak_id", sort=False)}
        if not dna_base.empty
        else {}
    )
    anchor_by_gene = {
        str(key): value for key, value in anchors.groupby("target_gene_id", sort=False)
    }
    assignment_by_gene = {
        str(key): value for key, value in assignments.groupby("target_gene_id", sort=False)
    }
    output_tables: dict[str, list[pd.DataFrame]] = {
        "physical_events": [],
        "candidate_routes": [],
        "event_routes": [],
        "cap_audit": [],
        "route_degree_cap_audit": [],
        "catalog_burden": [],
    }
    processed_genes = []
    for gene_id, gene_anchors in anchor_by_gene.items():
        strands = set(gene_anchors["strand"].astype(str))
        if len(strands) != 1:
            raise ValueError(f"event anchors have multiple strands for {gene_id}")
        gene_strand = next(iter(strands))
        source_parts: list[pd.DataFrame] = []
        gene_assignments = assignment_by_gene.get(gene_id)
        if gene_assignments is not None:
            dna_parts = [
                dna_by_peak[peak_id]
                for peak_id in gene_assignments["peak_id"].astype(str)
                if peak_id in dna_by_peak
            ]
            if dna_parts:
                dna_joined = pd.concat(dna_parts, ignore_index=True)
                dna_joined["source_local_rank"] = dna_joined.groupby(
                    ["motif_id", "source_window_id"], sort=False
                )["motif_score"].rank(method="first", ascending=False)
                dna_joined["target_gene_id"] = gene_id
                dna_joined["modality"] = "DNA"
                dna_joined["strand"] = gene_strand
                dna_joined["orientation"] = np.where(
                    dna_joined["hit_strand"].astype(str).eq(gene_strand),
                    "same_transcript",
                    "opposite_transcript",
                )
                dna_joined["source_valid"] = True
                dna_joined["calibrated_motif_quality"] = np.nan
                dna_joined["source_hit_id"] = (
                    "DNA|"
                    + gene_id
                    + "|"
                    + dna_joined["source_window_id"].astype(str)
                    + "|"
                    + dna_joined["motif_id"].astype(str)
                    + "|"
                    + dna_joined["start"].astype(str)
                    + "|"
                    + dna_joined["hit_strand"].astype(str)
                )
                dna_joined = assign_unique_peak_to_dna_hits(dna_joined)
                source_parts.append(
                    dna_joined[
                        [
                            "source_hit_id", "source_window_id", "target_gene_id",
                            "factor_entity_id", "factor_identity_kind", "candidate_factor_ids",
                            "activity_entity_id", "activity_gene_ids", "activity_proxy_rule",
                            "modality", "motif_id", "motif_equivalence_family_id",
                            "chromosome", "start", "end", "strand", "motif_score",
                            "calibrated_motif_quality", "orientation", "peak_id",
                            "peak_support", "source_priority", "source_local_rank", "source_valid",
                            "source_window_ids", "overlapping_peak_ids",
                        ]
                    ]
                )
            open_peaks = gene_assignments[
                ["peak_id", "chromosome", "start", "end", "strand", "peak_support"]
            ]
            source_parts.append(accessibility_only_hits(open_peaks, target_gene_id=gene_id))

        rna_windows = gene_anchors.loc[
            gene_anchors["modality"].astype(str).eq("RNA")
        ].drop_duplicates(
            ["anchor_region_id", "region_start", "region_end", "strand"]
        )
        if not rna_windows.empty:
            rna_segments = rna_windows.rename(
                columns={
                    "anchor_region_id": "segment_id",
                    "region_start": "start",
                    "region_end": "end",
                }
            ).copy()
            rna_segments["source_window_id"] = rna_segments["segment_id"]
            rna_segments["peak_id"] = None
            rna_segments["peak_support"] = np.float32(0.0)
            rna_scanned = _scan_segments(
                rna_segments,
                fasta=fasta,
                scanner=rna_scanner,
                scanner_identities=rna_identities,
                oriented_by_gene_strand=True,
            )
            rna_hits = _motif_source_hits(
                rna_scanned,
                rna_mapping,
                rna_segments,
                target_gene_id=gene_id,
                modality="RNA",
                gene_strand=gene_strand,
            )
            if not rna_hits.empty:
                source_parts.append(rna_hits)
        if not source_parts:
            raise ValueError(f"gene {gene_id} has no real DNA/Open/RNA source events")
        source_hits = pd.concat(source_parts, ignore_index=True)
        physical = collapse_physical_events(
            source_hits, minimum_overlap_bp=1, minimum_reciprocal_overlap=0.0
        )
        physical, _ = build_gate_keys(physical)
        candidate = build_candidate_routes(physical, gene_anchors)
        catalog = cap_and_finalize_routes(
            physical, candidate, events_per_bucket_cap=16
        )
        for name in output_tables:
            output_tables[name].append(getattr(catalog, name))
        processed_genes.append(gene_id)

    shard_names = {
        "physical_events": "physical_events",
        "candidate_routes": "candidate_routes",
        "event_routes": "event_routes",
        "cap_audit": "cap_audit",
        "route_degree_cap_audit": "route_degree_cap_audit",
        "catalog_burden": "catalog_burden",
    }
    counts = {}
    for key, directory in shard_names.items():
        destination = shard_output_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        frame = pd.concat(output_tables[key], ignore_index=True)
        if key == "route_degree_cap_audit":
            frame = _normalize_route_degree_audit_frame(frame)
        frame.to_parquet(destination / f"part-{suffix}.parquet", index=False)
        counts[key] = len(frame)
    record = {
        "schema_version": "fabric.event_scan_shard_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosome": chromosome,
        "gene_shard_index": gene_shard_index,
        "gene_shard_count": gene_shard_count,
        "gene_count": len(processed_genes),
        "gene_ids": processed_genes,
        "moods_version": "1.9.4.1",
        "motif_hit_threshold": "MOODS threshold_from_p with p=1e-6, uniform background",
        "motif_score_semantics": "within-PWM relative log-odds score; ranking only",
        "motif_score_in_model": False,
        "motif_equivalence": "singleton motif_id families",
        "collapse_overlap": "at least 1 bp connected components within exact family/peak bucket",
        "events_per_anchor_and_evidence_class_cap": 16,
        "counts": counts,
        "test_outcomes_read": False,
    }
    shard_root = shard_output_root / "shard_manifests"
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / f"{suffix}.json").write_text(json.dumps(record, indent=2) + "\n")


def _route_count_records(value: object) -> list[dict[str, object]]:
    """Represent a sparse edge-count map without an unbounded Arrow struct schema."""

    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = (
            (record["edge_id"], record["route_count"])
            for record in value
            if record is not None
        )
    else:
        raise TypeError(f"unsupported per-edge route-count value: {type(value)!r}")
    return [
        {"edge_id": str(edge_id), "route_count": int(count)}
        for edge_id, count in sorted(items)
        if count is not None and not pd.isna(count)
    ]


def _normalize_route_degree_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("per_edge_route_counts_pre", "per_edge_route_counts_post"):
        frame[column] = frame[column].map(_route_count_records)
    return frame


def merge_event_gene_chunks(
    output: Path, *, chromosome: str, gene_shard_count: int
) -> None:
    """Merge deterministic per-gene scan chunks into one chromosome catalog."""

    if gene_shard_count < 1:
        raise ValueError("event gene shard count must be positive")
    event_root = output / "events"
    final_manifest = event_root / "shard_manifests" / f"{chromosome}.json"
    table_names = (
        "physical_events", "candidate_routes", "event_routes", "cap_audit",
        "route_degree_cap_audit", "catalog_burden",
    )
    if final_manifest.is_file() and all(
        (event_root / name / f"part-{chromosome}.parquet").is_file()
        for name in table_names
    ):
        return
    chunk_root = event_root / "gene_chunks"
    manifests = []
    suffixes = []
    for index in range(gene_shard_count):
        suffix = f"{chromosome}-genechunk-{index:02d}-of-{gene_shard_count:02d}"
        path = chunk_root / "shard_manifests" / f"{suffix}.json"
        if not path.is_file():
            raise FileNotFoundError(f"event gene chunk manifest is absent: {path}")
        record = json.loads(path.read_text())
        if (
            record.get("chromosome") != chromosome
            or record.get("gene_shard_index") != index
            or record.get("gene_shard_count") != gene_shard_count
        ):
            raise ValueError(f"event chunk identity differs: {path}")
        manifests.append(record)
        suffixes.append(suffix)
    anchor_genes = tuple(
        pd.read_parquet(
            event_root / "graph_anchor_regions.parquet",
            filters=[("chromosome", "==", chromosome)],
            columns=["target_gene_id"],
        )["target_gene_id"].astype(str).drop_duplicates()
    )
    chunk_genes = [str(value) for record in manifests for value in record["gene_ids"]]
    if len(chunk_genes) != len(set(chunk_genes)) or set(chunk_genes) != set(anchor_genes):
        raise ValueError(f"event gene chunks do not close the {chromosome} gene axis")
    counts = {}
    sort_columns = {
        "physical_events": ["event_id"],
        "candidate_routes": ["route_id"],
        "event_routes": ["route_id"],
        "cap_audit": ["cap_bucket_id"],
        "route_degree_cap_audit": ["audit_population", "event_id", "anchor_region_id"],
        "catalog_burden": ["audit_population", "target_gene_id", "modality", "edge_token_id"],
    }
    for table_name in table_names:
        frames = [
            pd.read_parquet(
                chunk_root / table_name / f"part-{suffix}.parquet"
            )
            for suffix in suffixes
        ]
        merged = pd.concat(frames, ignore_index=True)
        if table_name == "route_degree_cap_audit":
            merged = _normalize_route_degree_audit_frame(merged)
        available_sort = [value for value in sort_columns[table_name] if value in merged]
        if available_sort:
            merged = merged.sort_values(available_sort, kind="mergesort").reset_index(drop=True)
        destination = event_root / table_name
        destination.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(destination / f"part-{chromosome}.parquet", index=False)
        counts[table_name] = len(merged)
        expected_count = sum(int(record["counts"][table_name]) for record in manifests)
        if len(merged) != expected_count:
            raise ValueError(f"merged {table_name} count differs from chunks")
    record = {
        "schema_version": "fabric.event_scan_shard_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosome": chromosome,
        "gene_count": len(anchor_genes),
        "gene_ids": list(anchor_genes),
        "moods_version": "1.9.4.1",
        "motif_hit_threshold": "MOODS threshold_from_p with p=1e-6, uniform background",
        "motif_score_semantics": "within-PWM relative log-odds score; ranking only",
        "motif_score_in_model": False,
        "motif_equivalence": "singleton motif_id families",
        "collapse_overlap": "at least 1 bp connected components within exact family/peak bucket",
        "events_per_anchor_and_evidence_class_cap": 16,
        "source_gene_chunk_count": gene_shard_count,
        "source_gene_chunk_manifests": [
            f"events/gene_chunks/shard_manifests/{suffix}.json" for suffix in suffixes
        ],
        "counts": counts,
        "test_outcomes_read": False,
    }
    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.write_text(json.dumps(record, indent=2) + "\n")


def build_gate_shard_stage(
    paths: Mapping[str, Path],
    output: Path,
    *,
    chromosome: str,
    pilot_gene_id: str | None = None,
    gene_shard_index: int | None = None,
    gene_shard_count: int | None = None,
) -> None:
    """Fit train-only molecule-weighted gates for one real chromosome shard."""

    from scipy import sparse
    import torch
    from .motifs import EVENT_ROUTE_COLUMNS

    event_root = output / "events"
    frozen_g_fit = tuple(
        pd.read_csv(paths["g_fit"], sep="\t")["target_gene_id"].astype(str)
    )
    if len(frozen_g_fit) != 17_600 or len(set(frozen_g_fit)) != len(frozen_g_fit):
        raise ValueError("gate construction requires the unique frozen 17,600-gene G_fit axis")
    frozen_g_fit_set = set(frozen_g_fit)
    if pilot_gene_id is not None and str(pilot_gene_id) not in frozen_g_fit_set:
        raise ValueError("gate pilot gene is outside the frozen G_fit axis")
    shard_event_root = event_root if pilot_gene_id is None else event_root / "pilot"
    if (gene_shard_index is None) != (gene_shard_count is None):
        raise ValueError("gate gene shard index/count must be provided together")
    if gene_shard_count is not None and (
        pilot_gene_id is not None
        or gene_shard_count < 1
        or gene_shard_index is None
        or not 0 <= gene_shard_index < gene_shard_count
    ):
        raise ValueError("gate gene-shard specification is invalid")
    shard_suffix = (
        f"{chromosome}-{pilot_gene_id}"
        if pilot_gene_id is not None
        else (
            f"{chromosome}-genechunk-{gene_shard_index:02d}-of-{gene_shard_count:02d}"
            if gene_shard_count is not None
            else chromosome
        )
    )
    completed_gated_root = (
        shard_event_root / "gated" / "gene_chunks"
        if gene_shard_count is not None
        else shard_event_root / "gated"
    )
    completed_manifest = completed_gated_root / "shard_manifests" / f"{shard_suffix}.json"
    completed_tables = (
        "physical_events", "candidate_routes", "event_routes", "cap_audit",
        "route_degree_cap_audit", "catalog_burden", "gate_admission",
    )
    if completed_manifest.is_file() and all(
        (completed_gated_root / name / f"part-{shard_suffix}.parquet").is_file()
        for name in completed_tables
    ):
        record = json.loads(completed_manifest.read_text())
        recorded_gene_ids = [str(value["gene_id"]) for value in record["gene_records"]]
        if (
            len(recorded_gene_ids) == len(set(recorded_gene_ids))
            and set(recorded_gene_ids) <= frozen_g_fit_set
            and all(
                (output / value["relative_path"]).is_file()
                for value in record["gene_records"]
            )
        ):
            return
    source_suffix = (
        f"{chromosome}-{pilot_gene_id}" if pilot_gene_id is not None else chromosome
    )
    context_root = output / "cell_context"
    required = [
        shard_event_root / "physical_events" / f"part-{source_suffix}.parquet",
        shard_event_root / "candidate_routes" / f"part-{source_suffix}.parquet",
        context_root / "rna_activity_cp10k_log1p.npy",
        context_root / "rna_activity_observed.npy",
        context_root / "rna_library_size.npy",
        context_root / "rna_activity_cell_axis.parquet",
        context_root / "rna_activity_entity_axis.parquet",
        context_root / "rna_atac_neighbors.parquet",
        context_root / "ATACMappingAudit.parquet",
        context_root / "atac_csc_backing" / "manifest.json",
        context_root / "atac_cell_axis.parquet",
        context_root / "atac_peak_axis.parquet",
    ]
    missing = [str(value) for value in required if not value.is_file()]
    if missing:
        raise FileNotFoundError("gate prerequisites are absent: " + "; ".join(missing))
    if gene_shard_count is not None:
        source_manifest = event_root / "shard_manifests" / f"{chromosome}.json"
        if not source_manifest.is_file():
            raise FileNotFoundError(
                f"merged event chromosome manifest is absent: {source_manifest}"
            )
        source_record = json.loads(source_manifest.read_text())
        if (
            source_record.get("chromosome") != chromosome
            or source_record.get("test_outcomes_read") is not False
        ):
            raise ValueError(f"event chromosome manifest identity differs: {source_manifest}")
        gene_axis = tuple(str(value) for value in source_record["gene_ids"])
        selected_gene_ids = gene_axis[gene_shard_index::gene_shard_count]
        fit_selected_gene_ids = tuple(
            value for value in selected_gene_ids if value in frozen_g_fit_set
        )
        physical = pd.read_parquet(
            required[0], filters=[("target_gene_id", "in", list(fit_selected_gene_ids))]
        )
        candidate = pd.read_parquet(
            required[1], filters=[("target_gene_id", "in", list(fit_selected_gene_ids))]
        )
        observed_gene_ids = set(physical["target_gene_id"].astype(str))
        if observed_gene_ids != set(fit_selected_gene_ids):
            raise ValueError(
                f"filtered event rows do not close the selected {chromosome} G_fit gate axis"
            )
    else:
        physical = pd.read_parquet(required[0])
        candidate = pd.read_parquet(required[1])
        physical = physical.loc[
            physical["target_gene_id"].astype(str).isin(frozen_g_fit_set)
        ].copy()
        candidate = candidate.loc[
            candidate["target_gene_id"].astype(str).isin(frozen_g_fit_set)
        ].copy()
    ec_path = paths["compatible_ec"] / "compatible_ec" / f"part-{chromosome}.parquet"
    ec_filters = [("final_fate", "==", "likelihood_informative")]
    if gene_shard_count is not None:
        ec_filters.append(("target_gene_id", "in", list(fit_selected_gene_ids)))
    # The upstream shard retains all observation-process fates for audit.  The
    # gate contract is defined only on K^inf, so push the exact fate predicate
    # into the parquet reader instead of materializing millions of audit-only
    # rows in RAM and filtering them afterwards.
    ec = pd.read_parquet(
        ec_path,
        filters=ec_filters,
    )
    if pilot_gene_id is not None:
        ec = ec.loc[
            ec["target_gene_id"].astype(str).eq(str(pilot_gene_id))
        ].copy()
    if ec.empty or set(ec["split"].astype(str)) - {"train", "val"}:
        raise ValueError(f"informative EC shard is empty or contains test: {chromosome}")
    activity = np.load(required[2], mmap_mode="r")
    activity_observed_axis = np.load(required[3], mmap_mode="r")
    activity_library_size = np.load(required[4], mmap_mode="r")
    activity_cells = pd.read_parquet(required[5])
    activity_entities = pd.read_parquet(required[6])
    if activity.shape != (len(activity_cells), len(activity_entities)):
        raise ValueError("RNA activity matrix differs from its frozen axes")
    if (
        activity_observed_axis.shape != (len(activity_cells),)
        or activity_library_size.shape != (len(activity_cells),)
        or not np.array_equal(activity_observed_axis, activity_library_size > 0)
    ):
        raise ValueError("RNA activity observation mask differs from full-library size")
    activity_cell_index = {
        value: index for index, value in enumerate(activity_cells["cell_id"].astype(str))
    }
    entity_ids = tuple(activity_entities["activity_entity_id"].astype(str))

    # Gate construction repeatedly selects narrow peak columns.  Keep the
    # normalized matrix in CSC so this operation does not rescan all 753,753
    # CSR columns for every gene.
    normalized_atac = _load_atac_csc_backing(context_root)
    atac_cells = pd.read_parquet(required[10])
    peak_axis = pd.read_parquet(required[11])
    if normalized_atac.shape != (len(atac_cells), len(peak_axis)):
        raise ValueError("normalized ATAC matrix differs from its frozen axes")
    atac_cell_index = {
        value: index for index, value in enumerate(atac_cells["atac_cell_id"].astype(str))
    }
    peak_index = {
        value: index for index, value in enumerate(peak_axis["peak_id"].astype(str))
    }
    mapping_audit = pd.read_parquet(required[8]).set_index("cell_id", drop=False)
    neighbors = pd.read_parquet(required[7])
    neighbors = neighbors.loc[neighbors["mapping_valid"].astype(bool)].copy()
    target_index = activity_cell_index
    weight_rows = neighbors["cell_id"].astype(str).map(target_index)
    weight_cols = neighbors["atac_cell_id"].astype(str).map(atac_cell_index)
    if weight_rows.isna().any() or weight_cols.isna().any():
        raise ValueError("ATAC neighbor identities differ from frozen axes")
    weights = sparse.csr_matrix(
        (
            neighbors["weight"].to_numpy(np.float64),
            (weight_rows.to_numpy(np.int64), weight_cols.to_numpy(np.int64)),
        ),
        shape=(len(activity_cells), len(atac_cells)),
    )
    sums = np.asarray(weights.sum(axis=1)).reshape(-1)
    valid_global = activity_cells["cell_id"].astype(str).map(
        mapping_audit["mapping_valid"].astype(bool)
    ).to_numpy(bool)
    if not np.allclose(sums[valid_global], 1.0, atol=1.0e-6, rtol=0):
        raise ValueError("valid global ATAC mapping weights do not sum to one")

    event_groups = {
        str(key): value for key, value in physical.groupby("target_gene_id", sort=False)
    }
    candidate_groups = {
        str(key): value for key, value in candidate.groupby("target_gene_id", sort=False)
    }
    ec_groups = {str(key): value for key, value in ec.groupby("target_gene_id", sort=False)}
    gates_root = (
        output / "gates" / chromosome
        if pilot_gene_id is None
        else output / "gates" / "pilot" / chromosome
    )
    gates_root.mkdir(parents=True, exist_ok=True)
    final_tables: dict[str, list[pd.DataFrame]] = {
        "physical_events": [],
        "candidate_routes": [],
        "event_routes": [],
        "cap_audit": [],
        "route_degree_cap_audit": [],
        "catalog_burden": [],
    }
    admission_rows: list[pd.DataFrame] = []
    gene_records = []
    thresholds = {
        channel: {
            "minimum_valid_cells": 25,
            "minimum_effective_cells": 10.0,
            "minimum_informative_molecules": 50.0,
            "minimum_standard_deviation": 1.0e-4,
        }
        for channel in ("RNA", "DNA", "Open")
    }
    for gene_id in physical["target_gene_id"].astype(str).drop_duplicates():
        if gene_id not in ec_groups:
            # This is expected only for the 106 graph-only genes, which have no
            # likelihood-fit tensor and remain audit-only.
            continue
        gene_ec = ec_groups[gene_id]
        gene_mass = gene_ec.groupby(["cell_id", "split"], sort=False)[
            "molecule_count"
        ].sum().reset_index()
        gene_mass["activity_row"] = gene_mass["cell_id"].astype(str).map(
            activity_cell_index
        )
        if gene_mass["activity_row"].isna().any():
            raise ValueError(f"gene EC cells are absent from RNA activity: {gene_id}")
        gene_mass = gene_mass.sort_values("activity_row", kind="mergesort").reset_index(drop=True)
        rows = gene_mass["activity_row"].to_numpy(np.int64)
        cell_ids = tuple(gene_mass["cell_id"].astype(str))
        splits = tuple(gene_mass["split"].astype(str))
        activity_values = np.asarray(activity[rows], dtype=np.float32)
        activity_observed = np.broadcast_to(
            np.asarray(activity_observed_axis[rows], dtype=bool)[:, None],
            activity_values.shape,
        ).copy()
        activity_context = ActivityContext(
            cell_ids=cell_ids,
            activity_entity_ids=entity_ids,
            values=activity_values,
            observed=activity_observed,
            library_size=np.asarray(activity_library_size[rows], dtype=np.float64),
        )
        events, gate_keys = build_gate_keys(event_groups[gene_id])
        peak_ids = tuple(sorted(set(gate_keys["peak_id"].dropna().astype(str))))
        missing_peaks = sorted(set(peak_ids) - set(peak_index))
        if missing_peaks:
            raise ValueError(f"gate peaks are absent from ATAC matrix: {missing_peaks[:5]}")
        if peak_ids:
            peak_columns = [peak_index[value] for value in peak_ids]
            mapped = (
                weights[rows] @ normalized_atac[:, peak_columns]
            ).tocsr().astype(np.float32)
        else:
            mapped = sparse.csr_matrix((len(rows), 0), dtype=np.float32)
        gene_mapping_valid = valid_global[rows]
        atac_context = ATACMappingContext(
            cell_ids=cell_ids,
            peak_ids=peak_ids,
            accessibility=mapped,
            mapping_valid=gene_mapping_valid,
            diagnostics=mapping_audit.loc[list(cell_ids)].reset_index(drop=True),
        )
        raw = build_raw_gate_signals(
            gate_keys, activity=activity_context, atac=atac_context
        )
        mass = np.broadcast_to(
            gene_mass["molecule_count"].to_numpy(np.float64)[:, None], raw.raw.shape
        )
        admission = fit_gate_admission(
            raw,
            gate_keys,
            train_mask=gene_mass["split"].astype(str).eq("train").to_numpy(),
            informative_molecule_mass=mass,
            thresholds_by_channel=thresholds,
        )
        values = transform_gates(raw, admission)
        candidate_gene = candidate_groups.get(gene_id)
        if candidate_gene is None:
            raise ValueError(f"event gene has no candidate routes: {gene_id}")
        catalog = cap_and_finalize_routes(
            events,
            candidate_gene[list(EVENT_ROUTE_COLUMNS)],
            events_per_bucket_cap=16,
            gate_admission=admission,
        )
        for name in final_tables:
            final_tables[name].append(getattr(catalog, name))
        admission_rows.append(admission)
        relative_path = (
            f"gates/{chromosome}/{gene_id}.pt"
            if pilot_gene_id is None
            else f"gates/pilot/{chromosome}/{gene_id}.pt"
        )
        torch.save(
            {
                "gene_id": gene_id,
                "cell_ids": cell_ids,
                "cell_split": splits,
                "gate_values": values,
                "informative_molecule_mass": gene_mass["molecule_count"].to_numpy(np.int64),
            },
            output / relative_path,
        )
        gene_records.append(
            {
                "gene_id": gene_id,
                "relative_path": relative_path,
                "cell_count": len(cell_ids),
                "gate_key_count": len(values.gate_key_ids),
                "active_gate_key_count": int(admission["gate_key_active"].sum()),
                "model_active_event_count": int(catalog.physical_events["model_active"].sum()),
            }
        )
    gated_root = completed_gated_root
    counts = {}
    for name, frames in final_tables.items():
        destination = gated_root / name
        destination.mkdir(parents=True, exist_ok=True)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if name == "route_degree_cap_audit" and not frame.empty:
            frame = _normalize_route_degree_audit_frame(frame)
        frame.to_parquet(destination / f"part-{shard_suffix}.parquet", index=False)
        counts[name] = len(frame)
    admission_root = gated_root / "gate_admission"
    admission_root.mkdir(parents=True, exist_ok=True)
    pd.concat(admission_rows, ignore_index=True).to_parquet(
        admission_root / f"part-{shard_suffix}.parquet", index=False
    )
    manifest = {
        "schema_version": "fabric.gate_shard_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosome": chromosome,
        "pilot_gene_id": pilot_gene_id,
        "gene_shard_index": gene_shard_index,
        "gene_shard_count": gene_shard_count,
        "g_fit_gene_count": len(gene_records),
        "gene_records": gene_records,
        "thresholds_by_channel": thresholds,
        "normalization_population": "train likelihood-informative molecule mass",
        "validation_statistics_used": False,
        "test_rows_or_test_statistics_used": False,
        "counts": counts,
    }
    manifest_root = gated_root / "shard_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / f"{shard_suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def merge_gate_gene_chunks(
    output: Path, *, chromosome: str, gene_shard_count: int, g_fit_path: Path
) -> None:
    """Merge deterministic train-fitted gate chunks into one chromosome shard."""

    if gene_shard_count < 1:
        raise ValueError("gate gene shard count must be positive")
    gated_root = output / "events" / "gated"
    final_manifest = gated_root / "shard_manifests" / f"{chromosome}.json"
    frozen_g_fit = set(
        pd.read_csv(g_fit_path, sep="\t")["target_gene_id"].astype(str)
    )
    if len(frozen_g_fit) != 17_600:
        raise ValueError("gate merge requires the frozen 17,600-gene G_fit axis")
    event_record = json.loads(
        (output / "events" / "shard_manifests" / f"{chromosome}.json").read_text()
    )
    expected_gene_ids = {
        str(value) for value in event_record["gene_ids"] if str(value) in frozen_g_fit
    }
    table_names = (
        "physical_events", "candidate_routes", "event_routes", "cap_audit",
        "route_degree_cap_audit", "catalog_burden", "gate_admission",
    )
    if final_manifest.is_file() and all(
        (gated_root / name / f"part-{chromosome}.parquet").is_file()
        for name in table_names
    ):
        existing = json.loads(final_manifest.read_text())
        existing_gene_ids = {
            str(value["gene_id"]) for value in existing.get("gene_records", ())
        }
        if (
            existing_gene_ids == expected_gene_ids
            and len(existing_gene_ids) == len(existing.get("gene_records", ()))
        ):
            return
    chunk_root = gated_root / "gene_chunks"
    manifests = []
    suffixes = []
    for index in range(gene_shard_count):
        suffix = f"{chromosome}-genechunk-{index:02d}-of-{gene_shard_count:02d}"
        path = chunk_root / "shard_manifests" / f"{suffix}.json"
        if not path.is_file():
            raise FileNotFoundError(f"gate gene chunk manifest is absent: {path}")
        record = json.loads(path.read_text())
        if (
            record.get("chromosome") != chromosome
            or record.get("gene_shard_index") != index
            or record.get("gene_shard_count") != gene_shard_count
            or record.get("test_rows_or_test_statistics_used") is not False
        ):
            raise ValueError(f"gate chunk identity differs: {path}")
        manifests.append(record)
        suffixes.append(suffix)
    gene_records = [value for record in manifests for value in record["gene_records"]]
    gene_ids = [str(value["gene_id"]) for value in gene_records]
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError(f"gate gene chunks overlap on {chromosome}")
    if set(gene_ids) != expected_gene_ids:
        raise ValueError(f"gate gene chunks do not close the {chromosome} G_fit axis")
    sort_columns = {
        "physical_events": ["event_id"],
        "candidate_routes": ["route_id"],
        "event_routes": ["route_id"],
        "cap_audit": ["cap_bucket_id"],
        "route_degree_cap_audit": ["audit_population", "event_id", "anchor_region_id"],
        "catalog_burden": ["audit_population", "target_gene_id", "modality", "edge_token_id"],
        "gate_admission": ["target_gene_id", "gate_key_id"],
    }
    counts = {}
    for table_name in table_names:
        merged = pd.concat(
            [
                pd.read_parquet(chunk_root / table_name / f"part-{suffix}.parquet")
                for suffix in suffixes
            ],
            ignore_index=True,
        )
        if table_name == "route_degree_cap_audit":
            merged = _normalize_route_degree_audit_frame(merged)
        sort = [value for value in sort_columns[table_name] if value in merged]
        if sort:
            merged = merged.sort_values(sort, kind="mergesort").reset_index(drop=True)
        destination = gated_root / table_name
        destination.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(destination / f"part-{chromosome}.parquet", index=False)
        counts[table_name] = len(merged)
    if not all((output / value["relative_path"]).is_file() for value in gene_records):
        raise FileNotFoundError("one or more merged gate gene tensors are absent")
    record = {
        "schema_version": "fabric.gate_shard_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosome": chromosome,
        "pilot_gene_id": None,
        "g_fit_gene_count": len(gene_records),
        "gene_records": sorted(gene_records, key=lambda value: str(value["gene_id"])),
        "thresholds_by_channel": manifests[0]["thresholds_by_channel"],
        "normalization_population": "train likelihood-informative molecule mass",
        "validation_statistics_used": False,
        "test_rows_or_test_statistics_used": False,
        "source_gene_chunk_count": gene_shard_count,
        "counts": counts,
    }
    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.write_text(json.dumps(record, indent=2) + "\n")


def _write_source_validation(paths: Mapping[str, Path], output: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--", "src/fabric"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("real dataset build requires a committed src/fabric source tree")
    source_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_paths = {
        key: str(value) for key, value in paths.items() if key != "real_dataset"
    }
    destination = output / "SourceValidation.json"
    if destination.is_file():
        existing = json.loads(destination.read_text())
        expected_identity = {
            "source_git_commit": source_commit,
            "sources": source_paths,
            "expected_counts": EXPECTED,
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(
                "real dataset build identity differs from the existing output root; "
                "use a fresh output root"
            )
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "existing real dataset output lacks a current build identity; "
            "use a fresh output root"
        )
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "fabric.real_source_validation.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ADMITTED",
        "source_git_commit": source_commit,
        "sources": source_paths,
        "expected_counts": EXPECTED,
        "historical_7198_graph_or_ec_used": False,
        "historical_167235_split_used": False,
        "test_compatible_rows_read": False,
        "test_predictions_or_metrics_computed": False,
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-inputs", default="data/external_inputs.yaml")
    parser.add_argument(
        "--stage",
        choices=(
            "validate",
            "graph",
            "cis",
            "cell-context",
            "factors",
            "activity",
            "anchors",
            "atac-normalization",
            "atac-csc-backing",
            "event-scan",
            "event-merge",
            "gates",
            "gate-merge",
            "all-static",
        ),
        default="all-static",
    )
    parser.add_argument("--chromosome")
    parser.add_argument("--pilot-gene-id")
    parser.add_argument("--gene-shard-index", type=int)
    parser.add_argument("--gene-shard-count", type=int)
    args = parser.parse_args(argv)
    paths = validate_external_inputs(args.external_inputs)
    output = paths["real_dataset"]
    _write_source_validation(paths, output)
    if args.stage in {"graph", "all-static"}:
        build_graph_stage(paths, output)
    if args.stage in {"cis", "all-static"}:
        if not (output / "graph" / "GraphManifest.json").is_file():
            raise FileNotFoundError("graph stage must complete before CIS")
        build_cis_stage(paths, output)
    if args.stage in {"cell-context", "all-static"}:
        build_cell_context_stage(paths, output)
    if args.stage in {"factors", "all-static"}:
        build_factor_stage(paths, output)
    if args.stage in {"activity", "all-static"}:
        build_activity_stage(paths, output)
    if args.stage in {"anchors", "all-static"}:
        build_anchor_stage(paths, output)
    if args.stage in {"atac-normalization", "all-static"}:
        build_atac_normalization_stage(paths, output)
    if args.stage == "atac-csc-backing":
        build_atac_csc_backing_stage(output)
    if args.stage == "event-scan":
        if not args.chromosome:
            raise ValueError("event-scan requires --chromosome")
        build_event_scan_stage(
            paths,
            output,
            chromosome=args.chromosome,
            pilot_gene_id=args.pilot_gene_id,
            gene_shard_index=args.gene_shard_index,
            gene_shard_count=args.gene_shard_count,
        )
    if args.stage == "event-merge":
        if not args.chromosome or args.gene_shard_count is None:
            raise ValueError("event-merge requires --chromosome and --gene-shard-count")
        merge_event_gene_chunks(
            output,
            chromosome=args.chromosome,
            gene_shard_count=args.gene_shard_count,
        )
    if args.stage == "gates":
        if not args.chromosome:
            raise ValueError("gates requires --chromosome")
        build_gate_shard_stage(
            paths,
            output,
            chromosome=args.chromosome,
            pilot_gene_id=args.pilot_gene_id,
            gene_shard_index=args.gene_shard_index,
            gene_shard_count=args.gene_shard_count,
        )
    if args.stage == "gate-merge":
        if not args.chromosome or args.gene_shard_count is None:
            raise ValueError("gate-merge requires --chromosome and --gene-shard-count")
        merge_gate_gene_chunks(
            output,
            chromosome=args.chromosome,
            gene_shard_count=args.gene_shard_count,
            g_fit_path=paths["g_fit"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
