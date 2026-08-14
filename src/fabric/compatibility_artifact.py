"""Build the external FABRIC V2 compatible-EC delivery from a frozen ONT BAM.

This is a data-preparation command, not a trainer fallback.  It consumes only
the frozen matrix-matched path catalog, cell split, and an existing alignment;
it never aligns reads, discovers transcripts, or extends the legal path axis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from scipy import sparse
from scipy.io import mmread
import yaml

EMPTY_FATE = "no_matrix_isoform_compatible"
INFORMATIVE_FATE = "likelihood_informative"
FULL_FATE = "matrix_catalog_compatible_uninformative"
TECHNICAL_FAILURE_FATE = "pre_compatibility_technical_qc_failure"
COMPATIBILITY_FATES = (EMPTY_FATE, INFORMATIVE_FATE, FULL_FATE)
READ_NAME_RE = re.compile(
    r"^(?P<barcode>[ACGT]{16})_(?P<umi>[ACGT]{12})#"
    r"(?P<uuid>[^_]+)_(?P<strand>[+-])$"
)
MATCH_CIGAR_OPS = {0, 7, 8}
REF_SKIP_CIGAR_OP = 3
DELETION_CIGAR_OP = 2
INSERTION_CIGAR_OP = 1
SOFT_CLIP_CIGAR_OP = 4
HARD_CLIP_CIGAR_OP = 5


@dataclass(frozen=True)
class CompatibilityPolicy:
    minimum_mapq: int
    minimum_junction_anchor_bp: int
    maximum_deletion_bp: int
    junction_tolerance_bp: int
    terminal_tolerance_bp: int
    ir_minimum_exon_aligned_bp_each_side: int
    ir_minimum_intron_aligned_bp_each_side: int
    require_primary: bool = True
    reject_sa_tag: bool = True
    library_strand_protocol: str = "forward"

    def __post_init__(self) -> None:
        integer_fields = (
            "minimum_mapq",
            "minimum_junction_anchor_bp",
            "maximum_deletion_bp",
            "junction_tolerance_bp",
            "terminal_tolerance_bp",
            "ir_minimum_exon_aligned_bp_each_side",
            "ir_minimum_intron_aligned_bp_each_side",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"compatibility policy {name} must be non-negative int")
        if self.minimum_junction_anchor_bp < 1:
            raise ValueError("minimum_junction_anchor_bp must be positive")
        if self.ir_minimum_exon_aligned_bp_each_side < 1:
            raise ValueError("IR exon boundary support must be positive")
        if self.ir_minimum_intron_aligned_bp_each_side < 1:
            raise ValueError("IR intron boundary support must be positive")
        if self.library_strand_protocol != "forward":
            raise ValueError("the frozen ONT library strand protocol must be forward")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CompatibilityPolicy":
        return cls(
            minimum_mapq=int(raw["minimum_mapq"]),
            minimum_junction_anchor_bp=int(raw["minimum_junction_anchor_bp"]),
            maximum_deletion_bp=int(raw["maximum_deletion_bp"]),
            junction_tolerance_bp=int(raw["junction_tolerance_bp"]),
            terminal_tolerance_bp=int(raw["terminal_tolerance_bp"]),
            ir_minimum_exon_aligned_bp_each_side=int(
                raw["ir_minimum_exon_aligned_bp_each_side"]
            ),
            ir_minimum_intron_aligned_bp_each_side=int(
                raw["ir_minimum_intron_aligned_bp_each_side"]
            ),
            require_primary=bool(raw.get("require_primary", True)),
            reject_sa_tag=bool(raw.get("reject_sa_tag", True)),
            library_strand_protocol=str(
                raw.get("library_strand_protocol", "forward")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "minimum_mapq",
                "minimum_junction_anchor_bp",
                "maximum_deletion_bp",
                "junction_tolerance_bp",
                "terminal_tolerance_bp",
                "ir_minimum_exon_aligned_bp_each_side",
                "ir_minimum_intron_aligned_bp_each_side",
                "require_primary",
                "reject_sa_tag",
                "library_strand_protocol",
            )
        }


@dataclass(frozen=True)
class PathEvidence:
    path_id: str
    matrix_row_0based: int
    exons: tuple[tuple[int, int], ...]
    junctions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RetainedIntronOpportunity:
    intron: tuple[int, int]
    spliced_path_ids: tuple[str, ...]
    retained_path_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenePathCatalog:
    gene_id: str
    chrom: str
    strand: str
    start_0based: int
    end_0based_exclusive: int
    ordered_path_ids: tuple[str, ...]
    paths: Mapping[str, PathEvidence]
    retained_introns: tuple[RetainedIntronOpportunity, ...]


@dataclass(frozen=True)
class AlignmentEvidence:
    covered_blocks: tuple[tuple[int, int], ...]
    observed_junctions: tuple[tuple[int, int], ...]
    junction_anchors: tuple[tuple[int, int], ...]
    deletion_intervals: tuple[tuple[int, int], ...]
    aligned_reference_bp: int
    soft_clip_bp: int
    hard_clip_bp: int


@dataclass(frozen=True)
class CompatibilityResult:
    pre_compatibility_qc_pass: bool
    technical_reason_code: str
    compatible_path_ids: tuple[str, ...]
    final_fate: str
    ir_alignment_supported_count: int
    ir_evidence_censored_count: int
    multi_intron_unspliced_pattern: bool
    ir_biogenesis_context: str


@dataclass
class GeneAccumulator:
    ec_counts: Counter[tuple[object, ...]] = field(default_factory=Counter)
    cell_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    cell_splits: dict[str, str] = field(default_factory=dict)
    observed_cell_umi_keys: set[tuple[str, str]] = field(default_factory=set)


def _register_unique_cell_gene_umi(
    accumulator: GeneAccumulator,
    *,
    cell_id: str,
    gene_id: str,
    umi: str,
) -> None:
    """Admit one molecular identity before any count or artifact mutation."""

    key = (cell_id, umi)
    if key in accumulator.observed_cell_umi_keys:
        raise ValueError(
            f"duplicate primary record for cell-gene-UMI {cell_id}/{gene_id}/{umi}"
        )
    accumulator.observed_cell_umi_keys.add(key)


def parse_read_name(query_name: str) -> dict[str, str]:
    match = READ_NAME_RE.match(str(query_name))
    if match is None:
        return {
            "barcode": "",
            "umi": "",
            "read_uuid": str(query_name),
            "read_name_strand": "",
            "parse_status": "unparsed_read_name",
        }
    return {
        "barcode": match.group("barcode"),
        "umi": match.group("umi"),
        "read_uuid": match.group("uuid"),
        "read_name_strand": match.group("strand"),
        "parse_status": "parsed",
    }


def parse_alignment_evidence(read: pysam.AlignedSegment) -> AlignmentEvidence:
    """Convert one primary genomic alignment to exact half-open evidence."""

    ref_pos = int(read.reference_start)
    block_start: int | None = None
    block_end: int | None = None
    blocks: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []
    anchors: list[tuple[int, int]] = []
    deletions: list[tuple[int, int]] = []
    aligned_reference_bp = 0
    soft_clip_bp = 0
    hard_clip_bp = 0
    cigar = tuple(read.cigartuples or ())

    for index, (operation, raw_length) in enumerate(cigar):
        length = int(raw_length)
        if operation in MATCH_CIGAR_OPS:
            if block_start is None:
                block_start = ref_pos
            ref_pos += length
            block_end = ref_pos
            aligned_reference_bp += length
            continue
        if operation == DELETION_CIGAR_OP:
            if block_start is not None and block_end is not None:
                blocks.append((block_start, block_end))
            deletions.append((ref_pos, ref_pos + length))
            ref_pos += length
            aligned_reference_bp += length
            block_start = None
            block_end = None
            continue
        if operation == REF_SKIP_CIGAR_OP:
            if block_start is not None and block_end is not None:
                blocks.append((block_start, block_end))
            junctions.append((ref_pos, ref_pos + length))
            anchors.append(
                (
                    _contiguous_match_anchor(cigar, index - 1, -1),
                    _contiguous_match_anchor(cigar, index + 1, 1),
                )
            )
            ref_pos += length
            block_start = None
            block_end = None
            continue
        if operation == INSERTION_CIGAR_OP:
            continue
        if operation == SOFT_CLIP_CIGAR_OP:
            soft_clip_bp += length
            continue
        if operation == HARD_CLIP_CIGAR_OP:
            hard_clip_bp += length
            continue
    if block_start is not None and block_end is not None:
        blocks.append((block_start, block_end))
    return AlignmentEvidence(
        covered_blocks=tuple(_merge_intervals(blocks)),
        observed_junctions=tuple(junctions),
        junction_anchors=tuple(anchors),
        deletion_intervals=tuple(deletions),
        aligned_reference_bp=aligned_reference_bp,
        soft_clip_bp=soft_clip_bp,
        hard_clip_bp=hard_clip_bp,
    )


def _contiguous_match_anchor(
    cigar: tuple[tuple[int, int], ...], index: int, step: int
) -> int:
    result = 0
    while 0 <= index < len(cigar) and int(cigar[index][0]) in MATCH_CIGAR_OPS:
        result += int(cigar[index][1])
        index += step
    return result


def build_path_catalog(
    selection_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, GenePathCatalog], str]:
    """Recover exactly the frozen 17,706-gene/90,672-path matrix catalog."""

    root = Path(selection_dir)
    summary = json.loads((root / "selection_summary.json").read_text())
    selected = pd.read_csv(
        root / "selected_ont_training_gene_catalog.tsv", sep="\t"
    )
    crosswalk = pd.read_parquet(root / "transcript_crosswalk_audit.parquet")
    gene_ids = tuple(selected["gene_id"].astype(str))
    if len(gene_ids) != 17_706 or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("frozen candidate gene catalog is not exactly 17,706 unique IDs")
    paths = crosswalk.loc[crosswalk["gene_id"].astype(str).isin(gene_ids)].copy()
    paths = paths.sort_values(
        ["gene_id", "matrix_row_0based"], kind="mergesort"
    ).reset_index(drop=True)
    if len(paths) != 90_672:
        raise ValueError("frozen selected matrix path catalog is not exactly 90,672 rows")
    if paths["resolved_transcript_id"].astype(str).duplicated().any():
        raise ValueError("selected matrix transcript IDs are not unique")
    if paths.groupby("gene_id")["path_signature"].nunique().sum() != len(paths):
        raise ValueError("selected matrix catalog contains structural alias collisions")
    paths["path_id"] = paths["resolved_transcript_id"].astype(str)
    paths["path_order_0based"] = paths.groupby("gene_id", sort=False).cumcount()
    parsed_exons = paths["path_signature"].map(_parse_path_signature)
    paths["exon_starts_0based"] = parsed_exons.map(
        lambda values: [start for start, _ in values]
    )
    paths["exon_ends_0based_exclusive"] = parsed_exons.map(
        lambda values: [end for _, end in values]
    )
    paths["transcript_aliases"] = paths["resolved_transcript_id"].map(
        lambda value: [str(value)]
    )
    paths["model_isoform_universe"] = (
        "resolved_ont_matrix_structural_paths_only"
    )

    catalogs: dict[str, GenePathCatalog] = {}
    for gene_id, group in paths.groupby("gene_id", sort=True):
        chroms = set(group["chrom"].astype(str))
        strands = set(group["strand"].astype(str))
        if len(chroms) != 1 or len(strands) != 1:
            raise ValueError(f"candidate gene spans multiple chromosome/strand: {gene_id}")
        evidence: dict[str, PathEvidence] = {}
        for row in group.itertuples(index=False):
            exons = tuple(
                zip(
                    map(int, row.exon_starts_0based),
                    map(int, row.exon_ends_0based_exclusive),
                    strict=True,
                )
            )
            junctions = tuple(
                (exons[index][1], exons[index + 1][0])
                for index in range(len(exons) - 1)
            )
            evidence[str(row.path_id)] = PathEvidence(
                path_id=str(row.path_id),
                matrix_row_0based=int(row.matrix_row_0based),
                exons=exons,
                junctions=junctions,
            )
        ordered_path_ids = tuple(group["path_id"].astype(str))
        catalogs[str(gene_id)] = GenePathCatalog(
            gene_id=str(gene_id),
            chrom=next(iter(chroms)),
            strand=next(iter(strands)),
            start_0based=min(start for item in evidence.values() for start, _ in item.exons),
            end_0based_exclusive=max(
                end for item in evidence.values() for _, end in item.exons
            ),
            ordered_path_ids=ordered_path_ids,
            paths=evidence,
            retained_introns=_retained_intron_opportunities(
                ordered_path_ids, evidence
            ),
        )
    if set(catalogs) != set(gene_ids):
        raise ValueError("path catalog does not cover every frozen candidate gene")
    expected = summary["selection"]
    if int(expected["selected_ont_training_gene_catalog"]) != len(catalogs):
        raise ValueError("selection summary candidate count drift")
    if int(expected["selected_matrix_structural_paths"]) != len(paths):
        raise ValueError("selection summary path count drift")
    identity = (
        f"{(root / 'transcript_crosswalk_audit.parquet').resolve()}::"
        f"selected_gene_catalog_ordered_matrix_paths::{len(paths)}_rows"
    )
    return paths, catalogs, identity


def _parse_path_signature(value: object) -> tuple[tuple[int, int], ...]:
    exons: list[tuple[int, int]] = []
    for token in str(value).split(";"):
        start_text, end_text = token.split("-", maxsplit=1)
        start = int(start_text) - 1
        end = int(end_text)
        if start < 0 or end <= start:
            raise ValueError(f"invalid frozen GTF exon interval: {token}")
        exons.append((start, end))
    exons.sort()
    if not exons:
        raise ValueError("path requires at least one exon")
    if any(exons[index][1] > exons[index + 1][0] for index in range(len(exons) - 1)):
        raise ValueError("path contains overlapping exons")
    return tuple(exons)


def _retained_intron_opportunities(
    ordered_path_ids: Sequence[str], paths: Mapping[str, PathEvidence]
) -> tuple[RetainedIntronOpportunity, ...]:
    unique_junctions = sorted(
        {junction for path in paths.values() for junction in path.junctions}
    )
    opportunities: list[RetainedIntronOpportunity] = []
    for intron in unique_junctions:
        start, end = intron
        spliced = tuple(
            path_id
            for path_id in ordered_path_ids
            if any(junction == intron for junction in paths[path_id].junctions)
        )
        retained = tuple(
            path_id
            for path_id in ordered_path_ids
            if any(
                exon_start <= start and exon_end >= end
                for exon_start, exon_end in paths[path_id].exons
            )
        )
        if spliced and retained:
            opportunities.append(
                RetainedIntronOpportunity(intron, spliced, retained)
            )
    return tuple(opportunities)


def build_cell_lookup(
    selection_dir: str | Path,
    library_to_matrix_prefix: Mapping[str, Mapping[str, str]],
) -> tuple[dict[tuple[str, str, str], tuple[str, str]], str]:
    """Build an exact `(XE, TS, barcode) -> (cell_id, split)` lookup."""

    root = Path(selection_dir)
    rows = pd.read_parquet(root / "matrix_cell_index.parquet")
    split_rows = pd.read_parquet(root / "split_rows.parquet")
    if len(rows) != 217_933 or len(split_rows) != 217_933:
        raise ValueError("frozen cell/split axis is not exactly 217,933 rows")
    if set(rows["cell_id"].astype(str)) != set(split_rows["cell_id"].astype(str)):
        raise ValueError("matrix cell and split identities differ")
    split = split_rows.set_index("cell_id")["split"].astype(str).to_dict()
    rows = rows.copy()
    rows["barcode"] = rows["matrix_barcode"].str.extract(r"_([ACGT]{16})$")[0]
    rows["matrix_prefix"] = rows["matrix_barcode"].str.replace(
        r"_[ACGT]{16}$", "", regex=True
    )
    if rows["barcode"].isna().any():
        raise ValueError("matrix cell axis contains an unparsable barcode")
    declared_prefixes: dict[str, tuple[str, str]] = {}
    for raw_xe, libraries in library_to_matrix_prefix.items():
        xe = str(raw_xe).lower()
        for raw_library, raw_prefix in libraries.items():
            key = str(raw_prefix)
            if key in declared_prefixes:
                raise ValueError(f"matrix prefix has multiple library owners: {key}")
            declared_prefixes[key] = (xe, str(raw_library))
    observed_prefixes = set(rows["matrix_prefix"].astype(str))
    if set(declared_prefixes) != observed_prefixes:
        raise ValueError(
            "library-to-matrix-prefix map does not cover the exact frozen axis: "
            f"missing={sorted(observed_prefixes - set(declared_prefixes))} "
            f"extra={sorted(set(declared_prefixes) - observed_prefixes)}"
        )
    lookup: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows.itertuples(index=False):
        xe, library = declared_prefixes[str(row.matrix_prefix)]
        key = (xe, library, str(row.barcode))
        value = (str(row.cell_id), str(split[str(row.cell_id)]))
        if key in lookup:
            raise ValueError(f"canonical BAM cell key is duplicated: {key}")
        lookup[key] = value
    identity = (
        f"{(root / 'matrix_cell_index.parquet').resolve()}+"
        f"{(root / 'split_rows.parquet').resolve()}::"
        f"XE_TS_barcode_to_cell_split::{len(lookup)}_rows"
    )
    return lookup, identity


def evaluate_alignment_compatibility(
    evidence: AlignmentEvidence,
    catalog: GenePathCatalog,
    policy: CompatibilityPolicy,
    *,
    mapq: int,
    alignment_strand: str,
    is_primary: bool,
    sa_tag_present: bool,
    read_name_parse_status: str = "parsed",
) -> CompatibilityResult:
    """Apply one frozen QC/compatibility rule without using the `TX` path label."""

    reasons: list[str] = []
    if policy.require_primary and not is_primary:
        reasons.append("not_primary_alignment")
    if read_name_parse_status != "parsed":
        reasons.append("unparsed_read_name")
    if mapq < policy.minimum_mapq:
        reasons.append("mapq_below_minimum")
    if policy.reject_sa_tag and sa_tag_present:
        reasons.append("sa_tag_present_chimeric")
    if alignment_strand != catalog.strand:
        reasons.append("alignment_gene_strand_conflict")
    if any(
        left < policy.minimum_junction_anchor_bp
        or right < policy.minimum_junction_anchor_bp
        for left, right in evidence.junction_anchors
    ):
        reasons.append("junction_anchor_below_minimum")
    if any(
        end - start > policy.maximum_deletion_bp
        for start, end in evidence.deletion_intervals
    ):
        reasons.append("deletion_above_maximum")
    if not evidence.covered_blocks and not evidence.observed_junctions:
        reasons.append("no_aligned_gene_evidence")
    if reasons:
        return CompatibilityResult(
            False,
            ";".join(sorted(set(reasons))),
            (),
            TECHNICAL_FAILURE_FATE,
            0,
            0,
            False,
            "not_applicable_technical_qc_failure",
        )

    supported_ir: list[tuple[int, int]] = []
    censored_ir: list[tuple[int, int]] = []
    for opportunity in catalog.retained_introns:
        intron = opportunity.intron
        if any(
            _junction_matches(observed, intron, policy.junction_tolerance_bp)
            for observed in evidence.observed_junctions
        ):
            continue
        if _bilateral_ir_supported(evidence.covered_blocks, intron, policy):
            supported_ir.append(intron)
        elif any(_overlap_bp(block, intron) > 0 for block in evidence.covered_blocks):
            censored_ir.append(intron)

    informative_blocks = tuple(evidence.covered_blocks)
    for intron in censored_ir:
        informative_blocks = tuple(
            fragment
            for block in informative_blocks
            for fragment in _subtract_interval(block, intron)
        )
    informative_blocks = tuple(_merge_intervals(informative_blocks))

    compatible: list[str] = []
    for path_id in catalog.ordered_path_ids:
        path = catalog.paths[path_id]
        junction_ok = all(
            any(
                _junction_matches(
                    observed, candidate, policy.junction_tolerance_bp
                )
                for candidate in path.junctions
            )
            for observed in evidence.observed_junctions
        )
        blocks_ok = _blocks_match_path(
            informative_blocks,
            path.exons,
            policy.terminal_tolerance_bp,
        )
        if junction_ok and blocks_ok:
            compatible.append(path_id)

    if not compatible:
        fate = EMPTY_FATE
    elif len(compatible) == len(catalog.ordered_path_ids):
        fate = FULL_FATE
    else:
        fate = INFORMATIVE_FATE
    if supported_ir:
        other_junction_count = sum(
            not any(
                _junction_matches(junction, intron, policy.junction_tolerance_bp)
                for intron in supported_ir
            )
            for junction in evidence.observed_junctions
        )
        context = (
            "processed_context_supported"
            if other_junction_count > 0
            else "mature_vs_nascent_unresolved"
        )
    else:
        context = "not_applicable_no_ir_alignment_support"
    return CompatibilityResult(
        True,
        "",
        tuple(compatible),
        fate,
        len(supported_ir),
        len(censored_ir),
        len(supported_ir) >= 2,
        context,
    )


def _bilateral_ir_supported(
    blocks: Sequence[tuple[int, int]],
    intron: tuple[int, int],
    policy: CompatibilityPolicy,
) -> bool:
    start, end = intron
    if end - start < 2 * policy.ir_minimum_intron_aligned_bp_each_side:
        return False
    required_start = start - policy.ir_minimum_exon_aligned_bp_each_side
    required_end = end + policy.ir_minimum_exon_aligned_bp_each_side
    return any(block_start <= required_start and block_end >= required_end for block_start, block_end in blocks)


def _junction_matches(
    observed: tuple[int, int], candidate: tuple[int, int], tolerance: int
) -> bool:
    return (
        abs(int(observed[0]) - int(candidate[0])) <= tolerance
        and abs(int(observed[1]) - int(candidate[1])) <= tolerance
    )


def _blocks_match_path(
    blocks: Sequence[tuple[int, int]],
    exons: Sequence[tuple[int, int]],
    tolerance: int,
) -> bool:
    return all(
        any(
            block_start < exon_end
            and block_end > exon_start
            and block_start >= exon_start - tolerance
            and block_end <= exon_end + tolerance
            for exon_start, exon_end in exons
        )
        for block_start, block_end in blocks
    )


def _subtract_interval(
    interval: tuple[int, int], removed: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    start, end = interval
    remove_start, remove_end = removed
    if remove_end <= start or remove_start >= end:
        return (interval,)
    pieces = []
    if start < remove_start:
        pieces.append((start, min(end, remove_start)))
    if remove_end < end:
        pieces.append((max(start, remove_end), end))
    return tuple(piece for piece in pieces if piece[1] > piece[0])


def _overlap_bp(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = sorted(
        (int(start), int(end))
        for start, end in intervals
        if int(end) > int(start)
    )
    if not ordered:
        return []
    result = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = result[-1]
        if start <= previous_end:
            result[-1] = (previous_start, max(previous_end, end))
        else:
            result.append((start, end))
    return result


class ParquetBatchWriter:
    def __init__(self, path: Path, schema: pa.Schema, *, batch_size: int = 100_000):
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict[str, object]] = []
        self.writer: pq.ParquetWriter | None = None

    def append(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def extend(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.path, self.schema, compression="zstd"
            )
        self.writer.write_table(table, row_group_size=self.batch_size)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path)
        else:
            self.writer.close()


MOLECULE_SCHEMA = pa.schema(
    [
        ("molecule_id", pa.string()),
        ("read_uuid", pa.string()),
        ("cell_id", pa.string()),
        ("cell_barcode", pa.string()),
        ("umi", pa.string()),
        ("library_id", pa.string()),
        ("donor_id", pa.string()),
        ("reporting_cell_state", pa.string()),
        ("target_gene_id", pa.string()),
        ("split", pa.string()),
        ("chrom", pa.string()),
        ("alignment_start_0based", pa.int64()),
        ("alignment_end_0based_exclusive", pa.int64()),
        ("alignment_strand", pa.string()),
        ("mapq", pa.int64()),
        ("tx_tag", pa.string()),
        ("cigarstring", pa.string()),
        ("pre_compatibility_qc_pass", pa.bool_()),
        ("technical_reason_code", pa.string()),
        ("compatible_path_ids", pa.list_(pa.string())),
        ("final_fate", pa.string()),
        ("ir_alignment_supported_count", pa.int64()),
        ("ir_evidence_censored_count", pa.int64()),
        ("multi_intron_unspliced_pattern", pa.bool_()),
        ("ir_biogenesis_context", pa.string()),
        ("internal_priming_status", pa.string()),
        ("genomic_dna_contamination_status", pa.string()),
        ("protocol_mature_transcript_evidence_status", pa.string()),
        ("molecule_count", pa.int64()),
    ]
)


EC_SCHEMA = pa.schema(
    [
        ("compatibility_class_id", pa.string()),
        ("cell_id", pa.string()),
        ("target_gene_id", pa.string()),
        ("split", pa.string()),
        ("library_id", pa.string()),
        ("donor_id", pa.string()),
        ("reporting_cell_state", pa.string()),
        ("pre_compatibility_qc_pass", pa.bool_()),
        ("technical_reason_code", pa.string()),
        ("compatible_path_ids", pa.list_(pa.string())),
        ("compatible_path_ids_key", pa.string()),
        ("final_fate", pa.string()),
        ("molecule_count", pa.int64()),
    ]
)


CELL_GENE_SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("target_gene_id", pa.string()),
        ("split", pa.string()),
        ("captured_gene_assigned_mass", pa.int64()),
        ("pre_compatibility_mass", pa.int64()),
        ("technical_qc_failure_mass", pa.int64()),
        ("empty_compatible_mass", pa.int64()),
        ("proper_subset_compatible_mass", pa.int64()),
        ("full_set_compatible_mass", pa.int64()),
    ]
)

RECONCILIATION_SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("target_gene_id", pa.string()),
        ("split", pa.string()),
        ("rna_embryo_id", pa.string()),
        ("matrix_library_prefix", pa.string()),
        ("matrix_mapped_count", pa.int64()),
        ("matrix_positive_transcript_count", pa.int32()),
        ("captured_gene_assigned_mass", pa.int64()),
        ("pre_compatibility_mass", pa.int64()),
        ("technical_qc_failure_mass", pa.int64()),
        ("empty_compatible_mass", pa.int64()),
        ("proper_subset_compatible_mass", pa.int64()),
        ("full_set_compatible_mass", pa.int64()),
        ("other_explicit_fate_mass", pa.int64()),
        ("matrix_scope_fate", pa.string()),
        ("matrix_pre_compatibility_relation", pa.string()),
        ("matrix_minus_pre_compatibility_mass", pa.int64()),
    ]
)


def build_compatible_ec_artifact(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_primary_records: int | None = None,
    chromosomes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build train/validation molecule fates and EC rows from the frozen BAM."""

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("compatibility producer config must contain a mapping")
    inputs = raw["inputs"]
    if not isinstance(inputs, Mapping):
        raise TypeError("compatibility producer inputs must contain a mapping")
    selection_dir = Path(str(inputs["selection_dir"])).resolve()
    bam_path = Path(str(inputs["bam"])).resolve()
    bam_index = Path(str(inputs["bam_index"])).resolve()
    reference_fasta = Path(str(inputs["reference_fasta"])).resolve()
    reference_fai = Path(str(inputs["reference_fai"])).resolve()
    authoritative_gtf = Path(str(inputs["authoritative_gtf"])).resolve()
    matrix_gtf = Path(str(inputs["matrix_matched_gtf"])).resolve()
    ont_matrix = Path(str(inputs["ont_transcript_matrix"])).resolve()
    ont_transcripts = Path(str(inputs["ont_transcripts"])).resolve()
    ont_barcodes = Path(str(inputs["ont_barcodes"])).resolve()
    for path in (
        selection_dir,
        bam_path,
        bam_index,
        reference_fasta,
        reference_fai,
        authoritative_gtf,
        matrix_gtf,
        ont_matrix,
        ont_transcripts,
        ont_barcodes,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir is None:
        output_dir = raw["output_dir"]
    destination = Path(str(output_dir)).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), destination / "producer_source_snapshot.py")
    shutil.copy2(config_path, destination / "producer_config_snapshot.yaml")
    policy = CompatibilityPolicy.from_mapping(raw["compatibility_policy"])
    policy_identity = str(raw["compatibility_policy_id"])
    if not policy_identity.strip():
        raise ValueError("compatibility_policy_id must be nonempty")
    library_map = raw["library_to_matrix_prefix"]
    if not isinstance(library_map, Mapping):
        raise TypeError("library_to_matrix_prefix must be a mapping")

    path_table, catalogs, path_identity = build_path_catalog(selection_dir)
    path_table.to_parquet(destination / "legal_structural_paths.parquet", index=False)
    cell_lookup, split_identity = build_cell_lookup(selection_dir, library_map)
    summary = json.loads((selection_dir / "selection_summary.json").read_text())
    split_row_identity = split_identity

    authoritative_tx_gene, authoritative_gene_loci = _read_transcript_annotation(
        authoritative_gtf
    )
    candidate_ids = tuple(sorted(catalogs))
    candidate_set = set(candidate_ids)
    selected = pd.read_csv(
        selection_dir / "selected_ont_training_gene_catalog.tsv", sep="\t"
    ).set_index("gene_id")
    support = {gene_id: Counter() for gene_id in candidate_ids}
    audit_groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    run_counts: Counter[str] = Counter()
    accumulators: dict[str, GeneAccumulator] = {}
    flushed: set[str] = set()

    molecule_writer = ParquetBatchWriter(
        destination / "molecule_fates.parquet", MOLECULE_SCHEMA
    )
    ec_writer = ParquetBatchWriter(destination / "compatible_ec.parquet", EC_SCHEMA)
    cell_gene_writer = ParquetBatchWriter(
        destination / "compatibility_cell_gene_counts.parquet", CELL_GENE_SCHEMA
    )

    chosen_chromosomes = tuple(
        chromosomes
        if chromosomes is not None
        else sorted({catalog.chrom for catalog in catalogs.values()}, key=_chrom_key)
    )
    genes_by_chrom: dict[str, list[str]] = defaultdict(list)
    for gene_id, catalog in catalogs.items():
        genes_by_chrom[catalog.chrom].append(gene_id)

    def flush_gene(gene_id: str) -> None:
        accumulator = accumulators.pop(gene_id, None)
        flushed.add(gene_id)
        if accumulator is None:
            return
        for ec_index, (key, molecule_count) in enumerate(
            sorted(accumulator.ec_counts.items(), key=lambda item: repr(item[0]))
        ):
            (
                cell_id,
                split,
                library_id,
                donor_id,
                cell_state,
                qc_pass,
                technical_reason,
                path_ids,
                final_fate,
            ) = key
            ec_writer.append(
                {
                    "compatibility_class_id": f"ec:{gene_id}:{ec_index:08d}",
                    "cell_id": cell_id,
                    "target_gene_id": gene_id,
                    "split": split,
                    "library_id": library_id,
                    "donor_id": donor_id,
                    "reporting_cell_state": cell_state,
                    "pre_compatibility_qc_pass": qc_pass,
                    "technical_reason_code": technical_reason,
                    "compatible_path_ids": list(path_ids),
                    "compatible_path_ids_key": ";".join(path_ids),
                    "final_fate": final_fate,
                    "molecule_count": int(molecule_count),
                }
            )
            _update_audit_groups(
                audit_groups,
                split=str(split),
                library_id=str(library_id),
                donor_id=str(donor_id),
                gene_id=gene_id,
                cell_state=str(cell_state),
                qc_pass=bool(qc_pass),
                final_fate=str(final_fate),
                molecule_count=int(molecule_count),
            )
        for cell_id, counts in sorted(accumulator.cell_counts.items()):
            cell_gene_writer.append(
                {
                    "cell_id": cell_id,
                    "target_gene_id": gene_id,
                    "split": accumulator.cell_splits[cell_id],
                    "captured_gene_assigned_mass": int(counts["captured"]),
                    "pre_compatibility_mass": int(counts["pre_qc"]),
                    "technical_qc_failure_mass": int(counts["technical"]),
                    "empty_compatible_mass": int(counts[EMPTY_FATE]),
                    "proper_subset_compatible_mass": int(counts[INFORMATIVE_FATE]),
                    "full_set_compatible_mass": int(counts[FULL_FATE]),
                }
            )

    stop = False
    with pysam.AlignmentFile(bam_path, "rb", index_filename=str(bam_index), threads=8) as bam:
        for chrom in chosen_chromosomes:
            if chrom not in set(bam.references):
                raise ValueError(f"BAM misses candidate chromosome: {chrom}")
            end_heap = [
                (
                    max(
                        catalogs[gene_id].end_0based_exclusive,
                        authoritative_gene_loci.get(
                            gene_id,
                            (
                                catalogs[gene_id].chrom,
                                catalogs[gene_id].start_0based,
                                catalogs[gene_id].end_0based_exclusive,
                            ),
                        )[2],
                    ),
                    gene_id,
                )
                for gene_id in genes_by_chrom.get(chrom, [])
            ]
            heapq.heapify(end_heap)
            for read in bam.fetch(chrom):
                run_counts["bam_records_visited"] += 1
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    run_counts["non_primary_records_skipped"] += 1
                    continue
                if (
                    max_primary_records is not None
                    and run_counts["primary_records_visited"] >= max_primary_records
                ):
                    stop = True
                    break
                run_counts["primary_records_visited"] += 1
                current_start = int(read.reference_start)
                while end_heap and end_heap[0][0] <= current_start:
                    _, finished_gene = heapq.heappop(end_heap)
                    flush_gene(finished_gene)
                tx_tag = str(read.get_tag("TX")) if read.has_tag("TX") else ""
                gene_id = authoritative_tx_gene.get(tx_tag, "")
                if gene_id not in candidate_set:
                    run_counts["primary_records_outside_candidate_genes"] += 1
                    continue
                catalog = catalogs[gene_id]
                if catalog.chrom != chrom:
                    raise ValueError(
                        "TX-assigned primary alignment conflicts with the frozen gene "
                        f"chromosome: gene={gene_id} expected={catalog.chrom} "
                        f"observed={chrom} read={read.query_name}"
                    )
                if gene_id in flushed:
                    raise ValueError(
                        "TX-assigned primary alignment starts beyond its frozen gene locus: "
                        f"gene={gene_id} read={read.query_name} start={current_start}"
                    )
                run_counts["candidate_gene_primary_records"] += 1
                parsed = parse_read_name(str(read.query_name))
                xe_tag = str(read.get_tag("XE")).lower() if read.has_tag("XE") else ""
                library_id = str(read.get_tag("TS")) if read.has_tag("TS") else ""
                cell_key = (xe_tag, library_id, parsed["barcode"])
                resolved = cell_lookup.get(cell_key)
                if resolved is None:
                    run_counts["candidate_records_unresolved_cell"] += 1
                    continue
                cell_id, split = resolved
                if split == "test":
                    run_counts["test_primary_records_seen_not_materialized"] += 1
                    continue
                if split not in {"train", "val"}:
                    raise ValueError(f"invalid frozen split: {split}")
                evidence = parse_alignment_evidence(read)
                if not any(
                    _overlap_bp(
                        block,
                        (catalog.start_0based, catalog.end_0based_exclusive),
                    )
                    > 0
                    for block in evidence.covered_blocks
                ):
                    run_counts["candidate_alignments_outside_frozen_path_span"] += 1
                result = evaluate_alignment_compatibility(
                    evidence,
                    catalog,
                    policy,
                    mapq=int(read.mapping_quality),
                    alignment_strand="-" if read.is_reverse else "+",
                    is_primary=True,
                    sa_tag_present=read.has_tag("SA"),
                    read_name_parse_status=parsed["parse_status"],
                )
                cell_state = (
                    str(read.get_tag("XM")) if read.has_tag("XM") else "not_available"
                )
                accumulator = accumulators.setdefault(gene_id, GeneAccumulator())
                _register_unique_cell_gene_umi(
                    accumulator,
                    cell_id=cell_id,
                    gene_id=gene_id,
                    umi=parsed["umi"],
                )
                molecule_row = {
                    "molecule_id": str(read.query_name),
                    "read_uuid": parsed["read_uuid"],
                    "cell_id": cell_id,
                    "cell_barcode": parsed["barcode"],
                    "umi": parsed["umi"],
                    "library_id": library_id,
                    "donor_id": xe_tag,
                    "reporting_cell_state": cell_state,
                    "target_gene_id": gene_id,
                    "split": split,
                    "chrom": chrom,
                    "alignment_start_0based": int(read.reference_start),
                    "alignment_end_0based_exclusive": int(read.reference_end or read.reference_start),
                    "alignment_strand": "-" if read.is_reverse else "+",
                    "mapq": int(read.mapping_quality),
                    "tx_tag": tx_tag,
                    "cigarstring": str(read.cigarstring or ""),
                    "pre_compatibility_qc_pass": result.pre_compatibility_qc_pass,
                    "technical_reason_code": result.technical_reason_code,
                    "compatible_path_ids": list(result.compatible_path_ids),
                    "final_fate": result.final_fate,
                    "ir_alignment_supported_count": result.ir_alignment_supported_count,
                    "ir_evidence_censored_count": result.ir_evidence_censored_count,
                    "multi_intron_unspliced_pattern": result.multi_intron_unspliced_pattern,
                    "ir_biogenesis_context": result.ir_biogenesis_context,
                    "internal_priming_status": "not_available_from_frozen_bam",
                    "genomic_dna_contamination_status": "not_available_from_frozen_bam",
                    "protocol_mature_transcript_evidence_status": "not_available_from_frozen_bam",
                    "molecule_count": 1,
                }
                molecule_writer.append(molecule_row)
                run_counts["materialized_train_validation_molecules"] += 1
                support[gene_id][f"{split}:captured"] += 1
                support[gene_id][f"{split}:{result.final_fate}"] += 1
                if result.pre_compatibility_qc_pass:
                    support[gene_id][f"{split}:pre_qc"] += 1
                ec_key = (
                    cell_id,
                    split,
                    library_id,
                    xe_tag,
                    cell_state,
                    result.pre_compatibility_qc_pass,
                    result.technical_reason_code,
                    result.compatible_path_ids,
                    result.final_fate,
                )
                accumulator.ec_counts[ec_key] += 1
                cell_counter = accumulator.cell_counts[cell_id]
                previous_split = accumulator.cell_splits.setdefault(cell_id, split)
                if previous_split != split:
                    raise AssertionError(
                        f"frozen cell appears in multiple splits: {cell_id}"
                    )
                cell_counter["captured"] += 1
                if result.pre_compatibility_qc_pass:
                    cell_counter["pre_qc"] += 1
                    cell_counter[result.final_fate] += 1
                else:
                    cell_counter["technical"] += 1
            while end_heap:
                _, finished_gene = heapq.heappop(end_heap)
                flush_gene(finished_gene)
            if stop:
                break
    for gene_id in sorted(set(accumulators)):
        flush_gene(gene_id)
    molecule_writer.close()
    ec_writer.close()
    cell_gene_writer.close()

    support_rows = _candidate_support_rows(candidate_ids, selected, support)
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_parquet(destination / "candidate_support_status.parquet", index=False)
    g_fit = support_frame.loc[
        support_frame["train_positive_informative_ec_mass"].gt(0),
        [
            "target_gene_id",
            "DTU_score",
            "top_DTU_gene",
            "train_positive_informative_ec_mass",
            "support_status",
        ],
    ].sort_values("target_gene_id", kind="mergesort")
    partial = max_primary_records is not None or set(chosen_chromosomes) != {
        catalog.chrom for catalog in catalogs.values()
    }
    g_fit_name = "G_fit_candidate_partial.tsv" if partial else "G_fit.tsv"
    g_fit.to_csv(destination / g_fit_name, sep="\t", index=False)

    audit = _long_read_audit_frame(audit_groups, candidate_ids)
    audit = _augment_long_read_audit_with_ir(
        audit,
        [destination / "molecule_fates.parquet"],
        raw.get("upstream_alignment_and_molecule_provenance_audit", {}),
    )
    audit.to_parquet(destination / "LongReadCompatibilityAudit.parquet", index=False)
    split_conservation = _split_conservation_from_support(support_frame)
    alignment_identity = {
        "bam": _path_identity(bam_path),
        "bam_index": _path_identity(bam_index),
    }
    reference_identity = {
        "fasta": _path_identity(reference_fasta),
        "fasta_index": _path_identity(reference_fai),
        "authoritative_gtf": _path_identity(authoritative_gtf),
        "matrix_matched_gtf": _path_identity(matrix_gtf),
    }
    matrix_observation_input_identity = {
        "transcript_matrix": _path_identity(ont_matrix),
        "transcript_axis": _path_identity(ont_transcripts),
        "barcode_axis": _path_identity(ont_barcodes),
    }
    quantifier = raw.get("matrix_quantifier_provenance", {})
    missing_quantifier = sorted(
        field
        for field in (
            "software_identity",
            "config_identity",
            "umi_read_collapse_policy_identity",
            "multi_mapping_policy_identity",
            "hard_assignment_count_semantics",
        )
        if not isinstance(quantifier, Mapping)
        or str(quantifier.get(field, "")).startswith("MISSING")
        or not str(quantifier.get(field, ""))
    )
    compatibility_complete = not partial
    observation_process = _observation_process_config(raw)
    reconciliation_status = "PENDING_NUMERICAL_MATRIX_JOIN"
    reconciliation_complete = False
    observation_process_admitted = False
    for key in (
        "bam_records_visited",
        "non_primary_records_skipped",
        "primary_records_visited",
        "primary_records_outside_candidate_genes",
        "candidate_gene_primary_records",
        "candidate_alignments_outside_frozen_path_span",
        "candidate_records_unresolved_cell",
        "test_primary_records_seen_not_materialized",
        "materialized_train_validation_molecules",
        "duplicate_cell_gene_umi_primary_records",
    ):
        run_counts.setdefault(key, 0)
    manifest = {
        "schema_version": "fabric.compatibility_artifact_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "fabric.compatibility_artifact",
        "command": " ".join(map(str, sys.argv)),
        "code_version": {
            "producer_version": "fabric.compatibility_artifact.v1",
            "git_head": _git_head(config_path.parent),
            "source_snapshot": "producer_source_snapshot.py",
            "config_snapshot": "producer_config_snapshot.yaml",
        },
        "build_scope": "real_small_scale_test" if partial else "complete_train_validation",
        "alignment_identity": alignment_identity,
        "reference_identity": reference_identity,
        "matrix_observation_input_identity": matrix_observation_input_identity,
        "legal_path_catalog_identity": path_identity,
        "cell_split_identity": summary["split"]["manifest_sha256"],
        "cell_split_row_identity": split_row_identity,
        "qc_policy": policy.to_dict(),
        "compatibility_policy": {
            **policy.to_dict(),
            "candidate_space": "frozen_matrix_structural_paths_only",
            "tx_tag_use": "gene_assignment_only_not_compatible_path_assignment",
            "ir_censoring": "unsupported_intronic_alignment_removed_before_Ck_rebuild",
        },
        "model_isoform_universe": "resolved_ont_matrix_structural_paths_only",
        "matrix_structural_path_count": len(path_table),
        "processed_chromosomes": list(chosen_chromosomes),
        "candidate_gene_ids": list(candidate_ids),
        "candidate_support_status": support_rows,
        "split_conservation": split_conservation,
        "train_policy_identity": policy_identity,
        "validation_policy_identity": policy_identity,
        "test_exposure": "not_materialized_before_checkpoint",
        "compatible_test_row_exposure": observation_process[
            "compatible_test_row_exposure"
        ],
        "matrix_test_count_exposure": observation_process[
            "matrix_test_count_exposure"
        ],
        "test_predictions_or_metrics_computed": False,
        "test_rows_written": False,
        "training_authorized_or_started": False,
        "run_counts": {key: int(value) for key, value in sorted(run_counts.items())},
        "record_mass_semantics": (
            "one_unique_primary_BAM_record_per_cell_gene_UMI_in_observed_scope"
            if int(run_counts["duplicate_cell_gene_umi_primary_records"]) == 0
            else "primary_BAM_record_mass_with_duplicate_cell_gene_UMI_records_detected"
        ),
        "informative_gene_ids": g_fit["target_gene_id"].astype(str).tolist(),
        "G_fit_artifact": g_fit_name,
        "G_fit_freeze_status": "NOT_FROZEN_PARTIAL" if partial else "FROZEN_FROM_TRAIN_ONLY",
        "compatibility_validation_status": (
            "PARTIAL_REAL_SCALE_TEST" if partial else "COMPLETE_MASS_VALIDATED"
        ),
        "observation_process_status": (
            "ADMITTED" if observation_process_admitted else "PENDING_OBSERVATION_PROCESS_AUDIT"
        ),
        "validation_status": (
            "PARTIAL_REAL_SCALE_TEST"
            if partial
            else (
                "ADMITTED"
                if observation_process_admitted
                else "COMPATIBILITY_COMPLETE_OBSERVATION_PROCESS_PENDING"
            )
        ),
        "artifact_complete": compatibility_complete,
        "admission_pass": observation_process_admitted,
    }
    (destination / "CompatibilityArtifactManifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    observation_reasons = []
    if partial:
        observation_reasons.append("compatible_read_rebuild_is_partial_real_scale_test")
    if missing_quantifier:
        observation_reasons.append(
            "missing_matrix_quantifier_provenance:" + ",".join(missing_quantifier)
        )
    if not reconciliation_complete:
        observation_reasons.append("numerical_matrix_cell_gene_reconciliation_pending")
    observation_audit = {
        "schema_version": "fabric.ont_observation_process_audit.v1",
        "status": "PENDING_OBSERVATION_PROCESS_AUDIT" if observation_reasons else "ADMITTED",
        "comparison_name": observation_process["comparison_name"],
        "compatible_test_row_exposure": observation_process[
            "compatible_test_row_exposure"
        ],
        "matrix_test_count_exposure": observation_process[
            "matrix_test_count_exposure"
        ],
        "test_predictions_or_metrics_computed": False,
        "reasons": observation_reasons,
        "matrix_quantifier_provenance": quantifier,
        "matrix_observation_input_identity": matrix_observation_input_identity,
        "upstream_alignment_and_molecule_provenance_audit": raw.get(
            "upstream_alignment_and_molecule_provenance_audit", {}
        ),
        "missing_provenance_fields": missing_quantifier,
        "minimum_external_delivery_required": raw.get(
            "minimum_external_delivery_required", {}
        ),
        "compatible_artifact_provenance": {
            "software_identity": "producer_source_snapshot.py",
            "config_identity": "producer_config_snapshot.yaml",
            "reference_identity": reference_identity,
            "path_identity": path_identity,
            "split_identity": summary["split"]["manifest_sha256"],
            "molecule_identity": "one_primary_BAM_record_after_upstream_filtering",
            "barcode_identity": split_row_identity,
            "qc_policy_identity": policy_identity,
            "assignment_policy_identity": policy_identity,
        },
        "matrix_count_semantics_verified_same_population": observation_process[
            "matrix_count_semantics_verified_same_population"
        ],
        "same_population_reason": observation_process["same_population_reason"],
        "per_cell_gene_compatibility_counts": "compatibility_cell_gene_counts.parquet",
        "matrix_cell_gene_reconciliation": reconciliation_status,
        "training_authorized_or_started": False,
    }
    (destination / "OntObservationProcessAudit.json").write_text(
        json.dumps(observation_audit, indent=2, ensure_ascii=False) + "\n"
    )
    return {
        "output_dir": str(destination),
        "structural_candidate_count": len(candidate_ids),
        "matrix_structural_path_count": len(path_table),
        "G_fit_count": len(g_fit),
        "run_counts": manifest["run_counts"],
        "compatibility_validation_status": manifest["validation_status"],
        "observation_process_status": observation_audit["status"],
    }


def merge_compatible_ec_shards(
    config_path: str | Path,
    shard_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Merge exact single-chromosome shards without reinterpreting any row."""

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("compatibility producer config must contain a mapping")
    inputs = raw["inputs"]
    if not isinstance(inputs, Mapping):
        raise TypeError("compatibility producer inputs must contain a mapping")
    selection_dir = Path(str(inputs["selection_dir"])).resolve()
    _, catalogs, path_identity = build_path_catalog(selection_dir)
    expected_chromosomes = tuple(
        sorted({catalog.chrom for catalog in catalogs.values()}, key=_chrom_key)
    )
    shard_root = Path(shard_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")

    manifests: dict[str, dict[str, object]] = {}
    shard_dirs: dict[str, Path] = {}
    for chrom in expected_chromosomes:
        shard_dir = shard_root / chrom
        manifest_path = shard_dir / "CompatibilityArtifactManifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing chromosome shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("processed_chromosomes") != [chrom]:
            raise ValueError(f"shard chromosome identity mismatch: {manifest_path}")
        if manifest.get("build_scope") != "real_small_scale_test":
            raise ValueError(f"unexpected shard build scope: {manifest_path}")
        manifests[chrom] = manifest
        shard_dirs[chrom] = shard_dir
    _validate_shard_contracts(manifests, path_identity)

    destination.mkdir(parents=True, exist_ok=True)
    _copy_equal_shard_snapshot(
        shard_dirs, expected_chromosomes, "producer_source_snapshot.py", destination
    )
    _copy_equal_shard_snapshot(
        shard_dirs, expected_chromosomes, "producer_config_snapshot.yaml", destination
    )
    component_sources = {
        "molecule_fates": "molecule_fates.parquet",
        "compatible_ec": "compatible_ec.parquet",
        "compatibility_cell_gene_counts": "compatibility_cell_gene_counts.parquet",
    }
    for component, source_name in component_sources.items():
        component_dir = destination / component
        component_dir.mkdir()
        for chrom in expected_chromosomes:
            source = shard_dirs[chrom] / source_name
            if not source.exists():
                raise FileNotFoundError(source)
            (component_dir / f"part-{chrom}.parquet").hardlink_to(source)
    (destination / "legal_structural_paths.parquet").hardlink_to(
        shard_dirs[expected_chromosomes[0]] / "legal_structural_paths.parquet"
    )

    support_frame = _merge_candidate_support(
        [
            pd.read_parquet(shard_dirs[chrom] / "candidate_support_status.parquet")
            for chrom in expected_chromosomes
        ]
    )
    if len(support_frame) != 17_706:
        raise AssertionError("merged candidate support does not cover all 17,706 genes")
    support_frame.to_parquet(destination / "candidate_support_status.parquet", index=False)
    g_fit = support_frame.loc[
        support_frame["train_positive_informative_ec_mass"].gt(0),
        [
            "target_gene_id",
            "DTU_score",
            "top_DTU_gene",
            "train_positive_informative_ec_mass",
            "support_status",
        ],
    ].sort_values("target_gene_id", kind="mergesort")
    g_fit.to_csv(destination / "G_fit.tsv", sep="\t", index=False)

    audit = _merge_long_read_audits(
        [
            pd.read_parquet(shard_dirs[chrom] / "LongReadCompatibilityAudit.parquet")
            for chrom in expected_chromosomes
        ]
    )
    audit = _augment_long_read_audit_with_ir(
        audit,
        [shard_dirs[chrom] / "molecule_fates.parquet" for chrom in expected_chromosomes],
        raw.get("upstream_alignment_and_molecule_provenance_audit", {}),
    )
    audit.to_parquet(destination / "LongReadCompatibilityAudit.parquet", index=False)
    split_conservation = _split_conservation_from_support(support_frame)

    first = manifests[expected_chromosomes[0]]
    run_counts: Counter[str] = Counter()
    for chrom in expected_chromosomes:
        run_counts.update(
            {
                str(key): int(value)
                for key, value in manifests[chrom]["run_counts"].items()
            }
        )
    quantifier = raw.get("matrix_quantifier_provenance", {})
    missing_quantifier = _missing_quantifier_fields(quantifier)
    observation_process = _observation_process_config(raw)
    reconciliation_status = "PENDING_NUMERICAL_MATRIX_JOIN"
    reconciliation_complete = False
    observation_process_admitted = False
    alignment_identity = {
        "bam": _path_identity(Path(str(inputs["bam"])).resolve()),
        "bam_index": _path_identity(Path(str(inputs["bam_index"])).resolve()),
    }
    matrix_observation_input_identity = {
        "transcript_matrix": _path_identity(
            Path(str(inputs["ont_transcript_matrix"])).resolve()
        ),
        "transcript_axis": _path_identity(
            Path(str(inputs["ont_transcripts"])).resolve()
        ),
        "barcode_axis": _path_identity(
            Path(str(inputs["ont_barcodes"])).resolve()
        ),
    }
    support_rows = support_frame.astype(object).where(
        pd.notna(support_frame), None
    ).to_dict("records")
    merged_manifest = {
        **first,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(map(str, sys.argv)),
        "code_version": {
            "producer_version": "fabric.compatibility_artifact.v1",
            "git_head": _git_head(config_path.parent),
            "source_snapshot": "producer_source_snapshot.py",
            "config_snapshot": "producer_config_snapshot.yaml",
        },
        "build_scope": "complete_train_validation",
        "alignment_identity": alignment_identity,
        "matrix_observation_input_identity": matrix_observation_input_identity,
        "processed_chromosomes": list(expected_chromosomes),
        "legal_path_catalog_identity": path_identity,
        "candidate_support_status": support_rows,
        "split_conservation": split_conservation,
        "test_exposure": "not_materialized_before_checkpoint",
        "compatible_test_row_exposure": observation_process[
            "compatible_test_row_exposure"
        ],
        "matrix_test_count_exposure": observation_process[
            "matrix_test_count_exposure"
        ],
        "test_predictions_or_metrics_computed": False,
        "test_rows_written": False,
        "run_counts": {key: int(value) for key, value in sorted(run_counts.items())},
        "record_mass_semantics": (
            "one_unique_primary_BAM_record_per_cell_gene_UMI_in_observed_scope"
            if int(run_counts["duplicate_cell_gene_umi_primary_records"]) == 0
            else "primary_BAM_record_mass_with_duplicate_cell_gene_UMI_records_detected"
        ),
        "informative_gene_ids": g_fit["target_gene_id"].astype(str).tolist(),
        "G_fit_artifact": "G_fit.tsv",
        "G_fit_freeze_status": "FROZEN_FROM_TRAIN_ONLY",
        "compatibility_validation_status": "COMPLETE_MASS_VALIDATED",
        "observation_process_status": (
            "ADMITTED"
            if observation_process_admitted
            else "PENDING_OBSERVATION_PROCESS_AUDIT"
        ),
        "validation_status": (
            "ADMITTED"
            if observation_process_admitted
            else "COMPATIBILITY_COMPLETE_OBSERVATION_PROCESS_PENDING"
        ),
        "artifact_complete": True,
        "admission_pass": observation_process_admitted,
        "artifact_components": {
            "molecule_fates": "molecule_fates/",
            "compatible_ec": "compatible_ec/",
            "compatibility_cell_gene_counts": "compatibility_cell_gene_counts/",
            "legal_structural_paths": "legal_structural_paths.parquet",
            "candidate_support_status": "candidate_support_status.parquet",
            "G_fit": "G_fit.tsv",
            "long_read_compatibility_audit": "LongReadCompatibilityAudit.parquet",
            "ont_observation_process_audit": "OntObservationProcessAudit.json",
        },
        "chromosome_shards": [
            {
                "chromosome": chrom,
                "manifest_path": str(
                    shard_dirs[chrom] / "CompatibilityArtifactManifest.json"
                ),
            }
            for chrom in expected_chromosomes
        ],
        "training_authorized_or_started": False,
    }
    (destination / "CompatibilityArtifactManifest.json").write_text(
        json.dumps(merged_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    first_observation = json.loads(
        (shard_dirs[expected_chromosomes[0]] / "OntObservationProcessAudit.json").read_text()
    )
    observation_reasons = [
        reason
        for reason in first_observation.get("reasons", [])
        if reason != "compatible_read_rebuild_is_partial_real_scale_test"
        and not str(reason).startswith("missing_matrix_quantifier_provenance:")
        and reason != "numerical_matrix_cell_gene_reconciliation_pending"
    ]
    if missing_quantifier:
        observation_reasons.append(
            "missing_matrix_quantifier_provenance:" + ",".join(missing_quantifier)
        )
    if not reconciliation_complete:
        observation_reasons.append("numerical_matrix_cell_gene_reconciliation_pending")
    merged_observation = {
        **first_observation,
        "status": "PENDING_OBSERVATION_PROCESS_AUDIT" if observation_reasons else "ADMITTED",
        "reasons": observation_reasons,
        "comparison_name": observation_process["comparison_name"],
        "compatible_test_row_exposure": observation_process[
            "compatible_test_row_exposure"
        ],
        "matrix_test_count_exposure": observation_process[
            "matrix_test_count_exposure"
        ],
        "test_predictions_or_metrics_computed": False,
        "matrix_quantifier_provenance": quantifier,
        "upstream_alignment_and_molecule_provenance_audit": raw.get(
            "upstream_alignment_and_molecule_provenance_audit", {}
        ),
        "missing_provenance_fields": missing_quantifier,
        "minimum_external_delivery_required": raw.get(
            "minimum_external_delivery_required", {}
        ),
        "matrix_observation_input_identity": matrix_observation_input_identity,
        "compatible_artifact_provenance": {
            **first_observation["compatible_artifact_provenance"],
            "software_identity": "producer_source_snapshot.py",
            "config_identity": "producer_config_snapshot.yaml",
            "path_identity": path_identity,
        },
        "matrix_count_semantics_verified_same_population": observation_process[
            "matrix_count_semantics_verified_same_population"
        ],
        "same_population_reason": observation_process["same_population_reason"],
        "per_cell_gene_compatibility_counts": "compatibility_cell_gene_counts/",
        "matrix_cell_gene_reconciliation": reconciliation_status,
        "training_authorized_or_started": False,
    }
    (destination / "OntObservationProcessAudit.json").write_text(
        json.dumps(merged_observation, indent=2, ensure_ascii=False) + "\n"
    )
    return {
        "output_dir": str(destination),
        "structural_candidate_count": len(support_frame),
        "matrix_structural_path_count": int(first["matrix_structural_path_count"]),
        "G_fit_count": len(g_fit),
        "processed_chromosomes": list(expected_chromosomes),
        "run_counts": merged_manifest["run_counts"],
        "compatibility_validation_status": merged_manifest[
            "compatibility_validation_status"
        ],
        "observation_process_status": merged_observation["status"],
    }


def refresh_observation_process_audit(
    config_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Refresh provenance/audit metadata without rebuilding compatible EC rows."""

    config_path = Path(config_path).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("compatibility producer config must contain a mapping")
    manifest_path = artifact_dir / "CompatibilityArtifactManifest.json"
    observation_path = artifact_dir / "OntObservationProcessAudit.json"
    long_read_path = artifact_dir / "LongReadCompatibilityAudit.parquet"
    for path in (manifest_path, observation_path, long_read_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text())
    observation = json.loads(observation_path.read_text())
    if manifest.get("artifact_complete") is not True:
        raise ValueError("observation audit refresh requires a complete artifact")
    if manifest.get("training_authorized_or_started") is not False:
        raise ValueError("observation audit refresh cannot modify a training artifact")

    quantifier = raw.get("matrix_quantifier_provenance", {})
    upstream_provenance = raw.get(
        "upstream_alignment_and_molecule_provenance_audit", {}
    )
    if not isinstance(upstream_provenance, Mapping):
        raise TypeError("upstream provenance audit must contain a mapping")
    missing_quantifier = _missing_quantifier_fields(quantifier)
    process = _observation_process_config(raw)
    reconciliation_status = str(process["matrix_cell_gene_reconciliation"])
    reconciliation_complete = reconciliation_status in {
        "CROSS_PIPELINE_RECONCILED",
        "SAME_POPULATION_RECONCILED",
    }
    existing_reconciliation = observation.get(
        "matrix_cell_gene_reconciliation_artifact"
    )
    if reconciliation_complete and not isinstance(existing_reconciliation, Mapping):
        raise ValueError(
            "reconciled observation status requires a reconciliation artifact"
        )
    if isinstance(existing_reconciliation, Mapping) and str(
        existing_reconciliation.get("status", "")
    ) != reconciliation_status:
        raise ValueError("reconciliation artifact status differs from config")
    reasons = []
    if missing_quantifier:
        reasons.append(
            "missing_matrix_quantifier_provenance:" + ",".join(missing_quantifier)
        )
    if not reconciliation_complete:
        reasons.append("numerical_matrix_cell_gene_reconciliation_pending")
    admitted = not reasons

    provenance_source = Path(str(quantifier.get("provenance_source", ""))).resolve()
    if not provenance_source.is_file():
        raise FileNotFoundError(
            f"matrix quantifier provenance source is missing: {provenance_source}"
        )
    audit_source_snapshot = artifact_dir / "observation_audit_source_snapshot.py"
    audit_config_snapshot = artifact_dir / "observation_audit_config_snapshot.yaml"
    shutil.copy2(Path(__file__).resolve(), audit_source_snapshot)
    shutil.copy2(config_path, audit_config_snapshot)
    reconciliation = existing_reconciliation
    if isinstance(reconciliation, Mapping):
        reconciliation_source_snapshot = (
            artifact_dir / "matrix_reconciliation_source_snapshot.py"
        )
        reconciliation_config_snapshot = (
            artifact_dir / "matrix_reconciliation_config_snapshot.yaml"
        )
        shutil.copy2(Path(__file__).resolve(), reconciliation_source_snapshot)
        shutil.copy2(config_path, reconciliation_config_snapshot)
        reconciliation = {
            **reconciliation,
            "software_identity": reconciliation_source_snapshot.name,
            "config_identity": reconciliation_config_snapshot.name,
        }

    long_read = pd.read_parquet(long_read_path)
    long_read = _apply_protocol_qc_provenance(
        long_read,
        upstream_provenance,
    )
    long_read_tmp = artifact_dir / ".LongReadCompatibilityAudit.parquet.tmp"
    long_read.to_parquet(long_read_tmp, index=False)
    long_read_tmp.replace(long_read_path)

    revision = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_identity": "observation_audit_source_snapshot.py",
        "config_identity": "observation_audit_config_snapshot.yaml",
        "manuscript_methods_identity": _path_identity(provenance_source),
        "manuscript_methods_sections": quantifier.get("provenance_sections", []),
        "user_confirmed_defaults": {
            "blaze_source_modification_scope": quantifier.get(
                "blaze_source_modification_scope"
            ),
            "internal_priming_qc": upstream_provenance.get(
                "internal_priming_qc_provenance"
            ),
            "genomic_dna_contamination_qc": upstream_provenance.get(
                "genomic_dna_contamination_qc_provenance"
            ),
            "protocol_mature_transcript_qc": upstream_provenance.get(
                "protocol_mature_transcript_qc_provenance"
            ),
        },
    }
    observation.update(
        {
            "status": "ADMITTED" if admitted else "PENDING_OBSERVATION_PROCESS_AUDIT",
            "comparison_name": process["comparison_name"],
            "compatible_test_row_exposure": process[
                "compatible_test_row_exposure"
            ],
            "matrix_test_count_exposure": process["matrix_test_count_exposure"],
            "test_predictions_or_metrics_computed": False,
            "reasons": reasons,
            "matrix_quantifier_provenance": quantifier,
            "upstream_alignment_and_molecule_provenance_audit": upstream_provenance,
            "missing_provenance_fields": missing_quantifier,
            "minimum_external_delivery_required": raw.get(
                "minimum_external_delivery_required", {}
            ),
            "matrix_count_semantics_verified_same_population": process[
                "matrix_count_semantics_verified_same_population"
            ],
            "same_population_reason": process["same_population_reason"],
            "matrix_cell_gene_reconciliation": reconciliation_status,
            **(
                {"matrix_cell_gene_reconciliation_artifact": reconciliation}
                if isinstance(reconciliation, Mapping)
                else {}
            ),
            "observation_audit_revision": revision,
            "training_authorized_or_started": False,
        }
    )
    manifest.update(
        {
            "observation_process_status": observation["status"],
            "validation_status": (
                "ADMITTED"
                if admitted
                else "COMPATIBILITY_COMPLETE_OBSERVATION_PROCESS_PENDING"
            ),
            "admission_pass": admitted,
            "compatible_test_row_exposure": process[
                "compatible_test_row_exposure"
            ],
            "matrix_test_count_exposure": process["matrix_test_count_exposure"],
            "test_predictions_or_metrics_computed": False,
            "observation_audit_revision": revision,
            **(
                {"matrix_cell_gene_reconciliation_artifact": reconciliation}
                if isinstance(reconciliation, Mapping)
                else {}
            ),
            "training_authorized_or_started": False,
        }
    )
    manifest.setdefault("artifact_components", {}).update(
        {
            "observation_audit_source_snapshot": audit_source_snapshot.name,
            "observation_audit_config_snapshot": audit_config_snapshot.name,
        }
    )
    if isinstance(reconciliation, Mapping):
        manifest["artifact_components"].update(
            {
                "matrix_reconciliation_source_snapshot": (
                    "matrix_reconciliation_source_snapshot.py"
                ),
                "matrix_reconciliation_config_snapshot": (
                    "matrix_reconciliation_config_snapshot.yaml"
                ),
            }
        )
    observation_tmp = artifact_dir / ".OntObservationProcessAudit.json.tmp"
    manifest_tmp = artifact_dir / ".CompatibilityArtifactManifest.json.tmp"
    observation_tmp.write_text(
        json.dumps(observation, indent=2, ensure_ascii=False) + "\n"
    )
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    observation_tmp.replace(observation_path)
    manifest_tmp.replace(manifest_path)
    return {
        "artifact_dir": str(artifact_dir),
        "missing_provenance_fields": missing_quantifier,
        "matrix_cell_gene_reconciliation": reconciliation_status,
        "observation_process_status": observation["status"],
        "admission_pass": admitted,
    }


def reconcile_ont_matrix_with_compatibility(
    config_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Join frozen ONT hard counts to the train/validation cell-gene scope."""

    config_path = Path(config_path).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("compatibility producer config must contain a mapping")
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("compatibility producer inputs must contain a mapping")
    selection_dir = Path(str(inputs["selection_dir"])).resolve()
    matrix_path = Path(str(inputs["ont_transcript_matrix"])).resolve()
    transcript_axis_path = Path(str(inputs["ont_transcripts"])).resolve()
    barcode_axis_path = Path(str(inputs["ont_barcodes"])).resolve()
    for path in (
        selection_dir,
        matrix_path,
        transcript_axis_path,
        barcode_axis_path,
        artifact_dir / "CompatibilityArtifactManifest.json",
        artifact_dir / "OntObservationProcessAudit.json",
        artifact_dir / "legal_structural_paths.parquet",
        artifact_dir / "compatibility_cell_gene_counts",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest_path = artifact_dir / "CompatibilityArtifactManifest.json"
    observation_path = artifact_dir / "OntObservationProcessAudit.json"
    manifest = json.loads(manifest_path.read_text())
    observation = json.loads(observation_path.read_text())
    if manifest.get("artifact_complete") is not True:
        raise ValueError("matrix reconciliation requires a complete compatibility artifact")
    if manifest.get("compatibility_validation_status") != "COMPLETE_MASS_VALIDATED":
        raise ValueError("matrix reconciliation requires validated compatible mass")
    if manifest.get("training_authorized_or_started") is not False:
        raise ValueError("matrix reconciliation cannot modify a training artifact")

    crosswalk = pd.read_parquet(selection_dir / "transcript_crosswalk_audit.parquet")
    cells = pd.read_parquet(selection_dir / "matrix_cell_index.parquet")
    paths = pd.read_parquet(artifact_dir / "legal_structural_paths.parquet")
    selected = pd.read_csv(
        selection_dir / "selected_ont_training_gene_catalog.tsv", sep="\t"
    )
    transcript_axis = pd.read_csv(
        transcript_axis_path, sep="\t", header=None, dtype=str
    )[0]
    barcode_axis = pd.read_csv(barcode_axis_path, sep="\t", header=None, dtype=str)[0]
    _validate_reconciliation_axes(
        crosswalk=crosswalk,
        cells=cells,
        paths=paths,
        selected=selected,
        transcript_axis=transcript_axis,
        barcode_axis=barcode_axis,
        manifest=manifest,
    )

    matrix = mmread(matrix_path)
    if not sparse.issparse(matrix):
        raise TypeError("ONT MatrixMarket input must be sparse")
    if matrix.shape != (len(crosswalk), len(cells)):
        raise ValueError(
            f"ONT matrix shape differs from frozen axes: {matrix.shape} != "
            f"{(len(crosswalk), len(cells))}"
        )
    coordinate = matrix.tocoo(copy=False)
    if coordinate.data.dtype.kind not in "iu":
        raise TypeError("ONT transcript matrix counts must be integer")
    if np.any(coordinate.data <= 0):
        raise ValueError("ONT transcript matrix stores non-positive coordinates")
    coordinate_nnz = int(coordinate.nnz)
    matrix = coordinate.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if int(matrix.nnz) != coordinate_nnz:
        raise ValueError("ONT MatrixMarket contains duplicate transcript-cell coordinates")

    candidate_ids = tuple(map(str, manifest["candidate_gene_ids"]))
    gene_index = {gene_id: index for index, gene_id in enumerate(candidate_ids)}
    paths = paths.sort_values("matrix_row_0based", kind="mergesort").reset_index(drop=True)
    candidate_rows = paths["matrix_row_0based"].to_numpy(dtype=np.int64)
    path_gene_index = paths["gene_id"].astype(str).map(gene_index).to_numpy(dtype=np.int64)
    candidate_matrix = matrix[candidate_rows].tocsr()
    incidence = sparse.csr_matrix(
        (
            np.ones(len(paths), dtype=np.int64),
            (path_gene_index, np.arange(len(paths), dtype=np.int64)),
        ),
        shape=(len(candidate_ids), len(paths)),
    )
    matrix_gene_counts = (incidence @ candidate_matrix).tocsr()
    positive_candidate_matrix = candidate_matrix.copy()
    positive_candidate_matrix.data = np.ones(
        positive_candidate_matrix.nnz, dtype=np.int32
    )
    positive_incidence = incidence.astype(np.int32)
    matrix_positive_transcripts = (
        positive_incidence @ positive_candidate_matrix
    ).tocsr()
    matrix_gene_counts.sort_indices()
    matrix_positive_transcripts.sort_indices()
    if not np.array_equal(matrix_gene_counts.indptr, matrix_positive_transcripts.indptr):
        raise AssertionError("matrix count and positive-transcript sparsity differ")
    if not np.array_equal(matrix_gene_counts.indices, matrix_positive_transcripts.indices):
        raise AssertionError("matrix count and positive-transcript cell identities differ")

    train_columns = cells.loc[cells["split"].eq("train"), "matrix_column_0based"].to_numpy(
        dtype=np.int64
    )
    observed_train_mass = np.asarray(
        matrix_gene_counts[:, train_columns].sum(axis=1)
    ).reshape(-1).astype(np.int64)
    selected_train_mass = (
        selected.set_index("gene_id")
        .loc[list(candidate_ids), "train_total_raw_count"]
        .to_numpy(dtype=np.int64)
    )
    if not np.array_equal(observed_train_mass, selected_train_mass):
        raise AssertionError("candidate train matrix mass differs from frozen gene selection")

    cell_index = dict(
        zip(
            cells["cell_id"].astype(str),
            cells["matrix_column_0based"].astype(np.int64),
            strict=True,
        )
    )
    cell_metadata = cells.set_index("cell_id")[["rna_embryo_id", "matrix_barcode", "split"]]
    output_dir = artifact_dir / "ont_matrix_compatibility_cell_gene_reconciliation"
    if output_dir.exists():
        raise FileExistsError(f"reconciliation output already exists: {output_dir}")
    output_dir.mkdir()
    summary_partials: list[pd.DataFrame] = []
    total_scope_rows = 0
    compatibility_root = artifact_dir / "compatibility_cell_gene_counts"
    for source in sorted(compatibility_root.glob("part-*.parquet"), key=lambda p: _chrom_key(p.stem[5:])):
        writer = pq.ParquetWriter(
            output_dir / source.name,
            RECONCILIATION_SCHEMA,
            compression="snappy",
        )
        try:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=500_000):
                frame = batch.to_pandas()
                frame = frame.loc[frame["pre_compatibility_mass"].gt(0)].copy()
                if frame.empty:
                    continue
                if set(frame["split"].astype(str)) - {"train", "val"}:
                    raise ValueError("test or invalid split entered reconciliation scope")
                if not np.array_equal(
                    frame["captured_gene_assigned_mass"].to_numpy(dtype=np.int64),
                    frame["pre_compatibility_mass"].to_numpy(dtype=np.int64)
                    + frame["technical_qc_failure_mass"].to_numpy(dtype=np.int64),
                ):
                    raise AssertionError("captured compatibility mass is not conserved")
                if not np.array_equal(
                    frame["pre_compatibility_mass"].to_numpy(dtype=np.int64),
                    frame["empty_compatible_mass"].to_numpy(dtype=np.int64)
                    + frame["proper_subset_compatible_mass"].to_numpy(dtype=np.int64)
                    + frame["full_set_compatible_mass"].to_numpy(dtype=np.int64),
                ):
                    raise AssertionError("pre-QC compatibility fate mass is not conserved")
                row_gene_index = frame["target_gene_id"].astype(str).map(gene_index)
                row_cell_index = frame["cell_id"].astype(str).map(cell_index)
                if row_gene_index.isna().any() or row_cell_index.isna().any():
                    raise ValueError("compatibility row is outside frozen gene/cell axes")
                matrix_count = _sparse_pair_values(
                    matrix_gene_counts,
                    row_gene_index.to_numpy(dtype=np.int64),
                    row_cell_index.to_numpy(dtype=np.int64),
                ).astype(np.int64)
                positive_transcripts = _sparse_pair_values(
                    matrix_positive_transcripts,
                    row_gene_index.to_numpy(dtype=np.int64),
                    row_cell_index.to_numpy(dtype=np.int64),
                ).astype(np.int32)
                metadata = cell_metadata.loc[frame["cell_id"].astype(str)]
                if not np.array_equal(
                    frame["split"].astype(str).to_numpy(),
                    metadata["split"].astype(str).to_numpy(),
                ):
                    raise AssertionError("compatibility split differs from matrix cell axis")
                scope_fate = np.where(
                    matrix_count == 0,
                    "ont_count_total_zero",
                    np.where(
                        positive_transcripts < 2,
                        "fewer_than_two_positive_matrix_transcripts",
                        "eligible",
                    ),
                )
                pre_mass = frame["pre_compatibility_mass"].to_numpy(dtype=np.int64)
                relation = np.where(
                    matrix_count == 0,
                    "matrix_zero",
                    np.where(
                        matrix_count < pre_mass,
                        "matrix_less_than_compatible",
                        np.where(
                            matrix_count == pre_mass,
                            "matrix_equals_compatible",
                            "matrix_greater_than_compatible",
                        ),
                    ),
                )
                reconciled = pd.DataFrame(
                    {
                        "cell_id": frame["cell_id"].astype(str).to_numpy(),
                        "target_gene_id": frame["target_gene_id"].astype(str).to_numpy(),
                        "split": frame["split"].astype(str).to_numpy(),
                        "rna_embryo_id": metadata["rna_embryo_id"].astype(str).to_numpy(),
                        "matrix_library_prefix": metadata["matrix_barcode"]
                        .astype(str)
                        .str.rsplit("_", n=1)
                        .str[0]
                        .to_numpy(),
                        "matrix_mapped_count": matrix_count,
                        "matrix_positive_transcript_count": positive_transcripts,
                        "captured_gene_assigned_mass": frame[
                            "captured_gene_assigned_mass"
                        ].to_numpy(dtype=np.int64),
                        "pre_compatibility_mass": pre_mass,
                        "technical_qc_failure_mass": frame[
                            "technical_qc_failure_mass"
                        ].to_numpy(dtype=np.int64),
                        "empty_compatible_mass": frame["empty_compatible_mass"].to_numpy(
                            dtype=np.int64
                        ),
                        "proper_subset_compatible_mass": frame[
                            "proper_subset_compatible_mass"
                        ].to_numpy(dtype=np.int64),
                        "full_set_compatible_mass": frame[
                            "full_set_compatible_mass"
                        ].to_numpy(dtype=np.int64),
                        "other_explicit_fate_mass": np.zeros(len(frame), dtype=np.int64),
                        "matrix_scope_fate": scope_fate,
                        "matrix_pre_compatibility_relation": relation,
                        "matrix_minus_pre_compatibility_mass": matrix_count - pre_mass,
                    }
                )
                writer.write_table(
                    pa.Table.from_pandas(
                        reconciled,
                        schema=RECONCILIATION_SCHEMA,
                        preserve_index=False,
                    )
                )
                summary_partials.extend(_reconciliation_summary_partials(reconciled))
                total_scope_rows += len(reconciled)
        finally:
            writer.close()

    summary = _combine_reconciliation_summaries(summary_partials)
    summary_path = artifact_dir / "OntMatrixCompatibilityReconciliationSummary.parquet"
    summary.to_parquet(summary_path, index=False)
    global_row = summary.loc[summary["stratum_type"].eq("global")]
    if len(global_row) != 1:
        raise AssertionError("reconciliation summary must have exactly one global row")
    global_record = global_row.iloc[0].to_dict()
    if int(global_record["candidate_cell_gene_count"]) != total_scope_rows:
        raise AssertionError("reconciliation summary row count differs from output")
    if not bool(global_record["mass_conservation_pass"]):
        raise AssertionError("global matrix/compatibility reconciliation is not conserved")

    reconciliation_identity = {
        "status": "CROSS_PIPELINE_RECONCILED",
        "comparison_name": "same_library_cross_pipeline_ont_matrix_agreement",
        "cell_gene_scope": (
            "train_validation_unique_cell_gene_with_positive_pre_compatibility_mass"
        ),
        "same_observation_population": False,
        "compatible_test_row_exposure": "not_materialized_before_checkpoint",
        "matrix_test_count_exposure": "previously_materialized_held_out_test",
        "test_predictions_or_metrics_computed": False,
        "reason": (
            "transcript_matrix_removes_multi_transcript_reads_while_compatible_EC_"
            "retains_ambiguous_reads_as_compatible_sets"
        ),
        "cell_gene_artifact": output_dir.name + "/",
        "summary_artifact": summary_path.name,
        "candidate_cell_gene_count": total_scope_rows,
        "matrix_coordinate_nnz": coordinate_nnz,
        "matrix_total_integer_mass": int(matrix.sum()),
        "candidate_scope_global_summary": {
            key: _json_scalar(value)
            for key, value in global_record.items()
            if key not in {"stratum_type", "stratum_values"}
        },
    }
    observation.update(
        {
            "status": "ADMITTED",
            "comparison_name": reconciliation_identity["comparison_name"],
            "compatible_test_row_exposure": reconciliation_identity[
                "compatible_test_row_exposure"
            ],
            "matrix_test_count_exposure": reconciliation_identity[
                "matrix_test_count_exposure"
            ],
            "test_predictions_or_metrics_computed": False,
            "reasons": [],
            "matrix_count_semantics_verified_same_population": False,
            "same_population_reason": reconciliation_identity["reason"],
            "matrix_cell_gene_reconciliation": "CROSS_PIPELINE_RECONCILED",
            "matrix_cell_gene_reconciliation_artifact": reconciliation_identity,
            "training_authorized_or_started": False,
        }
    )
    manifest.update(
        {
            "observation_process_status": "ADMITTED",
            "validation_status": "ADMITTED",
            "admission_pass": True,
            "compatible_test_row_exposure": reconciliation_identity[
                "compatible_test_row_exposure"
            ],
            "matrix_test_count_exposure": reconciliation_identity[
                "matrix_test_count_exposure"
            ],
            "test_predictions_or_metrics_computed": False,
            "matrix_cell_gene_reconciliation_artifact": reconciliation_identity,
            "training_authorized_or_started": False,
        }
    )
    manifest.setdefault("artifact_components", {}).update(
        {
            "ont_matrix_compatibility_cell_gene_reconciliation": output_dir.name + "/",
            "ont_matrix_compatibility_reconciliation_summary": summary_path.name,
        }
    )
    observation_tmp = artifact_dir / ".OntObservationProcessAudit.json.tmp"
    manifest_tmp = artifact_dir / ".CompatibilityArtifactManifest.json.tmp"
    observation_tmp.write_text(
        json.dumps(observation, indent=2, ensure_ascii=False) + "\n"
    )
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    observation_tmp.replace(observation_path)
    manifest_tmp.replace(manifest_path)
    return reconciliation_identity


def _validate_reconciliation_axes(
    *,
    crosswalk: pd.DataFrame,
    cells: pd.DataFrame,
    paths: pd.DataFrame,
    selected: pd.DataFrame,
    transcript_axis: pd.Series,
    barcode_axis: pd.Series,
    manifest: Mapping[str, object],
) -> None:
    if len(crosswalk) != 101_067:
        raise ValueError("complete ONT transcript crosswalk must have 101,067 rows")
    if len(cells) != 217_933:
        raise ValueError("complete ONT cell axis must have 217,933 rows")
    for frame, identity_column, name in (
        (crosswalk, "matrix_row_0based", "matrix transcript row"),
        (crosswalk, "matrix_transcript_name", "matrix transcript name"),
        (crosswalk, "resolved_transcript_id", "resolved transcript"),
        (cells, "matrix_column_0based", "matrix cell column"),
        (cells, "matrix_barcode", "matrix barcode"),
        (cells, "cell_id", "canonical cell"),
    ):
        if frame[identity_column].isna().any() or frame[identity_column].duplicated().any():
            raise ValueError(f"{name} identity is missing or duplicated")
    ordered_crosswalk = crosswalk.sort_values("matrix_row_0based", kind="mergesort")
    ordered_cells = cells.sort_values("matrix_column_0based", kind="mergesort")
    if not np.array_equal(
        ordered_crosswalk["matrix_row_0based"].to_numpy(dtype=np.int64),
        np.arange(len(crosswalk), dtype=np.int64),
    ):
        raise ValueError("matrix transcript rows are not a complete zero-based axis")
    if not np.array_equal(
        ordered_cells["matrix_column_0based"].to_numpy(dtype=np.int64),
        np.arange(len(cells), dtype=np.int64),
    ):
        raise ValueError("matrix cell columns are not a complete zero-based axis")
    if transcript_axis.astype(str).tolist() != ordered_crosswalk[
        "matrix_transcript_name"
    ].astype(str).tolist():
        raise ValueError("transcript sidecar order differs from the frozen crosswalk")
    if barcode_axis.astype(str).tolist() != ordered_cells["matrix_barcode"].astype(str).tolist():
        raise ValueError("barcode sidecar order differs from the frozen cell axis")
    candidate_ids = tuple(map(str, manifest["candidate_gene_ids"]))
    if len(candidate_ids) != 17_706 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("manifest candidate gene identity is not the frozen 17,706 set")
    if len(paths) != 90_672:
        raise ValueError("candidate matrix structural path axis must have 90,672 rows")
    if paths["matrix_row_0based"].duplicated().any() or paths["path_id"].duplicated().any():
        raise ValueError("candidate matrix path identity is duplicated")
    if set(paths["gene_id"].astype(str)) != set(candidate_ids):
        raise ValueError("candidate path genes differ from manifest candidate genes")
    if set(selected["gene_id"].astype(str)) != set(candidate_ids):
        raise ValueError("selected gene catalog differs from manifest candidate genes")
    subset = ordered_crosswalk.set_index("matrix_row_0based").loc[
        paths["matrix_row_0based"].to_numpy(dtype=np.int64)
    ]
    if not np.array_equal(
        subset["resolved_transcript_id"].astype(str).to_numpy(),
        paths["path_id"].astype(str).to_numpy(),
    ):
        raise ValueError("candidate matrix transcript-to-path mapping is not bijective")
    if not np.array_equal(
        subset["gene_id"].astype(str).to_numpy(),
        paths["gene_id"].astype(str).to_numpy(),
    ):
        raise ValueError("candidate matrix transcript-to-gene mapping differs")


def _sparse_pair_values(
    matrix: sparse.csr_matrix,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
) -> np.ndarray:
    if row_indices.shape != column_indices.shape:
        raise ValueError("sparse pair row/column index shapes differ")
    result = np.zeros(len(row_indices), dtype=matrix.data.dtype)
    order = np.argsort(row_indices, kind="stable")
    sorted_rows = row_indices[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_rows[1:] != sorted_rows[:-1], True]
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        row = int(sorted_rows[start])
        output_positions = order[start:end]
        requested_columns = column_indices[output_positions]
        matrix_start = int(matrix.indptr[row])
        matrix_end = int(matrix.indptr[row + 1])
        stored_columns = matrix.indices[matrix_start:matrix_end]
        stored_values = matrix.data[matrix_start:matrix_end]
        positions = np.searchsorted(stored_columns, requested_columns)
        in_range = positions < len(stored_columns)
        matched = np.zeros(len(positions), dtype=bool)
        matched[in_range] = (
            stored_columns[positions[in_range]] == requested_columns[in_range]
        )
        result[output_positions[matched]] = stored_values[positions[matched]]
    return result


def _reconciliation_summary_partials(frame: pd.DataFrame) -> list[pd.DataFrame]:
    metrics = pd.DataFrame(
        {
            "candidate_cell_gene_count": np.ones(len(frame), dtype=np.int64),
            "matrix_mapped_count_mass": frame["matrix_mapped_count"].to_numpy(
                dtype=np.int64
            ),
            "captured_gene_assigned_mass": frame[
                "captured_gene_assigned_mass"
            ].to_numpy(dtype=np.int64),
            "pre_compatibility_mass": frame["pre_compatibility_mass"].to_numpy(
                dtype=np.int64
            ),
            "technical_qc_failure_mass": frame[
                "technical_qc_failure_mass"
            ].to_numpy(dtype=np.int64),
            "empty_compatible_mass": frame["empty_compatible_mass"].to_numpy(
                dtype=np.int64
            ),
            "proper_subset_compatible_mass": frame[
                "proper_subset_compatible_mass"
            ].to_numpy(dtype=np.int64),
            "full_set_compatible_mass": frame["full_set_compatible_mass"].to_numpy(
                dtype=np.int64
            ),
            "other_explicit_fate_mass": frame[
                "other_explicit_fate_mass"
            ].to_numpy(dtype=np.int64),
        }
    )
    for fate in (
        "ont_count_total_zero",
        "fewer_than_two_positive_matrix_transcripts",
        "eligible",
    ):
        mask = frame["matrix_scope_fate"].eq(fate).to_numpy()
        metrics[f"{fate}_cell_gene_count"] = mask.astype(np.int64)
        metrics[f"{fate}_matrix_count_mass"] = (
            frame["matrix_mapped_count"].to_numpy(dtype=np.int64) * mask
        )
    for relation in (
        "matrix_zero",
        "matrix_less_than_compatible",
        "matrix_equals_compatible",
        "matrix_greater_than_compatible",
    ):
        metrics[f"{relation}_cell_gene_count"] = frame[
            "matrix_pre_compatibility_relation"
        ].eq(relation).to_numpy(dtype=np.int64)
    working = pd.concat(
        [
            frame[
                [
                    "split",
                    "rna_embryo_id",
                    "matrix_library_prefix",
                    "target_gene_id",
                ]
            ].reset_index(drop=True),
            metrics,
        ],
        axis=1,
    )
    partials = []
    strata = (
        ("global", ()),
        ("split", ("split",)),
        ("split_donor", ("split", "rna_embryo_id")),
        ("split_library", ("split", "matrix_library_prefix")),
        ("split_gene", ("split", "target_gene_id")),
    )
    metric_columns = list(metrics.columns)
    for stratum_type, columns in strata:
        if columns:
            grouped = working.groupby(list(columns), sort=False, observed=True)[
                metric_columns
            ].sum().reset_index()
            grouped["stratum_values"] = grouped.apply(
                lambda row: [str(row[column]) for column in columns], axis=1
            )
            grouped = grouped[["stratum_values", *metric_columns]]
        else:
            grouped = pd.DataFrame(
                [
                    {
                        "stratum_values": ["all"],
                        **{
                            column: int(metrics[column].sum())
                            for column in metric_columns
                        },
                    }
                ]
            )
        grouped["stratum_type"] = stratum_type
        partials.append(grouped)
    return partials


def _combine_reconciliation_summaries(partials: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(partials, ignore_index=True)
    combined["stratum_key"] = combined["stratum_values"].map(tuple)
    metric_columns = [
        column
        for column in combined.columns
        if column not in {"stratum_type", "stratum_values", "stratum_key"}
    ]
    result = combined.groupby(
        ["stratum_type", "stratum_key"], sort=True, as_index=False
    )[metric_columns].sum()
    result["mass_conservation_pass"] = (
        result["captured_gene_assigned_mass"]
        == result["pre_compatibility_mass"]
        + result["technical_qc_failure_mass"]
    ) & (
        result["pre_compatibility_mass"]
        == result["empty_compatible_mass"]
        + result["proper_subset_compatible_mass"]
        + result["full_set_compatible_mass"]
        + result["other_explicit_fate_mass"]
    ) & (
        result["candidate_cell_gene_count"]
        == result["ont_count_total_zero_cell_gene_count"]
        + result["fewer_than_two_positive_matrix_transcripts_cell_gene_count"]
        + result["eligible_cell_gene_count"]
    ) & (
        result["matrix_mapped_count_mass"]
        == result["ont_count_total_zero_matrix_count_mass"]
        + result["fewer_than_two_positive_matrix_transcripts_matrix_count_mass"]
        + result["eligible_matrix_count_mass"]
    ) & (
        result["candidate_cell_gene_count"]
        == result["matrix_zero_cell_gene_count"]
        + result["matrix_less_than_compatible_cell_gene_count"]
        + result["matrix_equals_compatible_cell_gene_count"]
        + result["matrix_greater_than_compatible_cell_gene_count"]
    )
    result["stratum_values"] = result.pop("stratum_key").map(list)
    return result[["stratum_type", "stratum_values", *metric_columns, "mass_conservation_pass"]]


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _validate_shard_contracts(
    manifests: Mapping[str, Mapping[str, object]], path_identity: str
) -> None:
    identity_fields = (
        "code_version",
        "alignment_identity",
        "reference_identity",
        "cell_split_identity",
        "cell_split_row_identity",
        "compatibility_policy",
        "train_policy_identity",
        "validation_policy_identity",
        "model_isoform_universe",
        "matrix_structural_path_count",
    )
    values = list(manifests.values())
    first = values[0]
    if first.get("legal_path_catalog_identity") != path_identity:
        raise ValueError("shard legal path identity differs from frozen selection")
    for manifest in values[1:]:
        if manifest.get("legal_path_catalog_identity") != path_identity:
            raise ValueError("shard legal path identity differs from frozen selection")
        for field in identity_fields:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"chromosome shards disagree on {field}")
        if manifest.get("test_rows_written") is not False:
            raise ValueError("a chromosome shard materialized formal-test rows")


def _copy_equal_shard_snapshot(
    shard_dirs: Mapping[str, Path],
    chromosomes: Sequence[str],
    name: str,
    destination: Path,
) -> None:
    first_path = shard_dirs[chromosomes[0]] / name
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    first_content = first_path.read_bytes()
    for chrom in chromosomes[1:]:
        candidate = shard_dirs[chrom] / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.read_bytes() != first_content:
            raise ValueError(f"chromosome shard snapshot content differs: {name}")
    shutil.copy2(first_path, destination / name)


def _merge_candidate_support(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    count_columns = [
        "train_captured_gene_assigned_mass",
        "train_pre_compatibility_qc_pass_mass",
        "train_empty_compatible_mass",
        "train_positive_informative_ec_mass",
        "train_full_set_compatible_mass",
        "validation_captured_gene_assigned_mass",
        "validation_pre_compatibility_qc_pass_mass",
        "validation_empty_compatible_mass",
        "validation_positive_informative_ec_mass",
        "validation_full_set_compatible_mass",
    ]
    combined = pd.concat(frames, ignore_index=True)
    for column in count_columns:
        if not pd.api.types.is_integer_dtype(combined[column]):
            raise TypeError(f"candidate support count is not integer: {column}")
    metadata_columns = ["DTU_score", "top_DTU_gene", "G_fit_rule"]
    for column in metadata_columns:
        if combined.groupby("target_gene_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"candidate metadata differs across shards: {column}")
    metadata = combined.groupby("target_gene_id", sort=True)[metadata_columns].first()
    counts = combined.groupby("target_gene_id", sort=True)[count_columns].sum()
    result = metadata.join(counts).reset_index()
    result["support_status"] = result["train_positive_informative_ec_mass"].map(
        lambda value: (
            "likelihood_fit_train_positive_informative_mass"
            if int(value) > 0
            else "graph_only_zero_train_informative_mass"
        )
    )
    ordered_columns = [
        "target_gene_id",
        "support_status",
        *count_columns,
        "DTU_score",
        "top_DTU_gene",
        "G_fit_rule",
    ]
    return result[ordered_columns]


def _merge_long_read_audits(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    count_columns = [
        "captured_gene_assigned_row_count",
        "captured_gene_assigned_molecule_mass",
        "pre_compatibility_qc_pass_row_count",
        "pre_compatibility_qc_pass_molecule_mass",
        "technical_qc_failure_row_count",
        "technical_qc_failure_molecule_mass",
        "terminal_row_count",
        "terminal_molecule_mass",
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["stratum_key"] = combined["stratum_values"].map(tuple)
    grouped = (
        combined.groupby(
            ["stratum_type", "stratum_key", "terminal_fate"],
            sort=True,
            as_index=False,
        )[count_columns]
        .sum()
    )
    denominator = grouped["pre_compatibility_qc_pass_molecule_mass"]
    grouped["terminal_fraction"] = grouped["terminal_molecule_mass"].div(
        denominator.where(denominator.gt(0))
    )
    grouped["fraction_status"] = denominator.map(
        lambda value: "estimated" if int(value) > 0 else "not_estimable"
    )
    grouped["mass_conservation_pass"] = False
    for _, indexes in grouped.groupby(
        ["stratum_type", "stratum_key"], sort=False, dropna=False
    ).groups.items():
        index_list = list(indexes)
        pre_qc_values = grouped.loc[
            index_list, "pre_compatibility_qc_pass_molecule_mass"
        ].unique()
        if len(pre_qc_values) != 1:
            raise AssertionError("merged audit denominator differs across terminal fates")
        conserved = int(grouped.loc[index_list, "terminal_molecule_mass"].sum()) == int(
            pre_qc_values[0]
        )
        if not conserved:
            raise AssertionError("merged long-read compatibility mass is not conserved")
        grouped.loc[index_list, "mass_conservation_pass"] = True
    grouped = grouped.sort_values(
        ["stratum_type", "stratum_key", "terminal_fate"], kind="mergesort"
    ).reset_index(drop=True)
    grouped["stratum_values"] = grouped.pop("stratum_key").map(list)
    ordered_columns = [
        "stratum_type",
        "stratum_values",
        "captured_gene_assigned_row_count",
        "captured_gene_assigned_molecule_mass",
        "pre_compatibility_qc_pass_row_count",
        "pre_compatibility_qc_pass_molecule_mass",
        "technical_qc_failure_row_count",
        "technical_qc_failure_molecule_mass",
        "terminal_fate",
        "terminal_row_count",
        "terminal_molecule_mass",
        "terminal_fraction",
        "fraction_status",
        "mass_conservation_pass",
    ]
    return grouped[ordered_columns]


def _missing_quantifier_fields(quantifier: object) -> list[str]:
    return sorted(
        field
        for field in (
            "software_identity",
            "config_identity",
            "umi_read_collapse_policy_identity",
            "multi_mapping_policy_identity",
            "hard_assignment_count_semantics",
        )
        if not isinstance(quantifier, Mapping)
        or str(quantifier.get(field, "")).startswith("MISSING")
        or not str(quantifier.get(field, ""))
    )


def _observation_process_config(raw: Mapping[str, object]) -> dict[str, object]:
    value = raw.get("observation_process")
    if not isinstance(value, Mapping):
        raise ValueError("observation_process config must contain a mapping")
    required = {
        "comparison_name",
        "compatible_test_row_exposure",
        "matrix_test_count_exposure",
        "test_predictions_or_metrics_computed",
        "matrix_count_semantics_verified_same_population",
        "same_population_reason",
        "matrix_cell_gene_reconciliation",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            "observation_process config misses fields: " + ",".join(missing)
        )
    status = str(value["matrix_cell_gene_reconciliation"])
    allowed = {
        "PENDING_NUMERICAL_MATRIX_JOIN",
        "CROSS_PIPELINE_RECONCILED",
        "SAME_POPULATION_RECONCILED",
    }
    if status not in allowed:
        raise ValueError(f"invalid matrix cell-gene reconciliation status: {status}")
    same_population = value["matrix_count_semantics_verified_same_population"]
    if not isinstance(same_population, bool):
        raise TypeError(
            "matrix_count_semantics_verified_same_population must be boolean"
        )
    if same_population != (status == "SAME_POPULATION_RECONCILED"):
        raise ValueError("same-population flag conflicts with reconciliation status")
    if not str(value["comparison_name"]).strip():
        raise ValueError("observation comparison name must be nonempty")
    if not str(value["same_population_reason"]).strip():
        raise ValueError("same-population reason must be nonempty")
    if value["compatible_test_row_exposure"] != "not_materialized_before_checkpoint":
        raise ValueError("compatible test rows must remain unmaterialized")
    if value["matrix_test_count_exposure"] != "previously_materialized_held_out_test":
        raise ValueError("matrix test count exposure marker is invalid")
    if value["test_predictions_or_metrics_computed"] is not False:
        raise ValueError("observation preparation cannot compute test predictions/metrics")
    return {field: value[field] for field in required}


IR_AUDIT_COUNT_COLUMNS = (
    "ir_alignment_supported_row_count",
    "ir_alignment_supported_molecule_mass",
    "ir_alignment_supported_opportunity_count",
    "ir_evidence_censored_row_count",
    "ir_evidence_censored_molecule_mass",
    "ir_evidence_censored_opportunity_count",
    "multi_intron_unspliced_row_count",
    "multi_intron_unspliced_molecule_mass",
    "processed_context_supported_row_count",
    "processed_context_supported_molecule_mass",
    "mature_vs_nascent_unresolved_row_count",
    "mature_vs_nascent_unresolved_molecule_mass",
)


def _augment_long_read_audit_with_ir(
    audit: pd.DataFrame,
    molecule_paths: Sequence[Path],
    upstream_provenance: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Attach contract-required RI/protocol strata from canonical molecule rows."""

    ir = _summarize_ir_molecule_files(molecule_paths)
    result = audit.copy()
    existing_ir = [column for column in IR_AUDIT_COUNT_COLUMNS if column in result]
    if existing_ir:
        result = result.drop(columns=existing_ir)
    result["stratum_key"] = result["stratum_values"].map(tuple)
    if not ir.empty:
        result = result.merge(
            ir,
            on=["stratum_type", "stratum_key"],
            how="left",
            validate="many_to_one",
        )
    for column in IR_AUDIT_COUNT_COLUMNS:
        if column not in result:
            result[column] = 0
        result[column] = result[column].fillna(0).astype(np.int64)
    denominator = result["pre_compatibility_qc_pass_molecule_mass"].astype(np.int64)
    fraction_metrics = {
        "ir_alignment_supported_fraction": "ir_alignment_supported_molecule_mass",
        "ir_evidence_censored_fraction": "ir_evidence_censored_molecule_mass",
        "multi_intron_unspliced_fraction": "multi_intron_unspliced_molecule_mass",
        "processed_context_supported_fraction": "processed_context_supported_molecule_mass",
        "mature_vs_nascent_unresolved_fraction": "mature_vs_nascent_unresolved_molecule_mass",
    }
    for fraction_column, mass_column in fraction_metrics.items():
        result[fraction_column] = result[mass_column].div(
            denominator.where(denominator.gt(0))
        )
    result["ir_fraction_status"] = denominator.map(
        lambda value: "estimated" if int(value) > 0 else "not_estimable"
    )
    result = _apply_protocol_qc_provenance(result, upstream_provenance)
    return result.drop(columns="stratum_key")


def _apply_protocol_qc_provenance(
    audit: pd.DataFrame,
    upstream_provenance: Mapping[str, object] | None,
) -> pd.DataFrame:
    """Report absent protocol QC as missing or explicitly not performed."""

    result = audit.copy()
    provenance = upstream_provenance or {}
    fields = {
        "internal_priming": "internal_priming_qc_provenance",
        "genomic_dna_contamination": "genomic_dna_contamination_qc_provenance",
        "protocol_mature_transcript_evidence": (
            "protocol_mature_transcript_qc_provenance"
        ),
    }
    for prefix, source_field in fields.items():
        declared = str(provenance.get(source_field, "MISSING"))
        if declared.startswith("NOT_PERFORMED"):
            assessment_status = "not_performed_upstream"
            fraction_status = "not_applicable_qc_not_performed"
        elif declared.startswith("MISSING") or not declared:
            assessment_status = "not_available_from_frozen_bam"
            fraction_status = "not_estimable_missing_upstream_flag"
        else:
            raise ValueError(
                f"{source_field} declares performed QC but no per-molecule flag is available"
            )
        result[f"{prefix}_assessment_status"] = assessment_status
        result[f"{prefix}_assessed_molecule_mass"] = 0
        result[f"{prefix}_positive_molecule_mass"] = 0
        result[f"{prefix}_positive_fraction"] = np.nan
        result[f"{prefix}_fraction_status"] = fraction_status
    return result


def _summarize_ir_molecule_files(molecule_paths: Sequence[Path]) -> pd.DataFrame:
    columns = [
        "split",
        "library_id",
        "donor_id",
        "target_gene_id",
        "reporting_cell_state",
        "pre_compatibility_qc_pass",
        "molecule_count",
        "ir_alignment_supported_count",
        "ir_evidence_censored_count",
        "multi_intron_unspliced_pattern",
        "ir_biogenesis_context",
        "internal_priming_status",
        "genomic_dna_contamination_status",
        "protocol_mature_transcript_evidence_status",
    ]
    strata = (
        ("global", ()),
        ("split", ("split",)),
        ("split_library", ("split", "library_id")),
        ("split_donor", ("split", "donor_id")),
        ("split_gene", ("split", "target_gene_id")),
        ("split_cell_state", ("split", "reporting_cell_state")),
    )
    partials: list[pd.DataFrame] = []
    for molecule_path in molecule_paths:
        frame = pd.read_parquet(molecule_path, columns=columns)
        for status_column in (
            "internal_priming_status",
            "genomic_dna_contamination_status",
            "protocol_mature_transcript_evidence_status",
        ):
            observed = set(frame[status_column].astype(str))
            if observed - {"not_available_from_frozen_bam"}:
                raise ValueError(
                    f"unsupported protocol QC status in {status_column}: {sorted(observed)}"
                )
        qc = frame.loc[frame["pre_compatibility_qc_pass"].astype(bool)].copy()
        del frame
        if qc.empty:
            continue
        mass = qc["molecule_count"].astype(np.int64)
        supported = qc["ir_alignment_supported_count"].astype(np.int64).gt(0)
        censored = qc["ir_evidence_censored_count"].astype(np.int64).gt(0)
        multi = qc["multi_intron_unspliced_pattern"].astype(bool)
        context = qc["ir_biogenesis_context"].astype(str)
        metric_values = {
            "ir_alignment_supported_row_count": supported.astype(np.int64),
            "ir_alignment_supported_molecule_mass": mass * supported,
            "ir_alignment_supported_opportunity_count": (
                mass * qc["ir_alignment_supported_count"].astype(np.int64)
            ),
            "ir_evidence_censored_row_count": censored.astype(np.int64),
            "ir_evidence_censored_molecule_mass": mass * censored,
            "ir_evidence_censored_opportunity_count": (
                mass * qc["ir_evidence_censored_count"].astype(np.int64)
            ),
            "multi_intron_unspliced_row_count": multi.astype(np.int64),
            "multi_intron_unspliced_molecule_mass": mass * multi,
            "processed_context_supported_row_count": context.eq(
                "processed_context_supported"
            ).astype(np.int64),
            "processed_context_supported_molecule_mass": mass
            * context.eq("processed_context_supported"),
            "mature_vs_nascent_unresolved_row_count": context.eq(
                "mature_vs_nascent_unresolved"
            ).astype(np.int64),
            "mature_vs_nascent_unresolved_molecule_mass": mass
            * context.eq("mature_vs_nascent_unresolved"),
        }
        for name, values in metric_values.items():
            qc[name] = values.astype(np.int64)
        for stratum_type, group_columns in strata:
            if group_columns:
                grouped = (
                    qc.groupby(list(group_columns), sort=False, observed=True)[
                        list(IR_AUDIT_COUNT_COLUMNS)
                    ]
                    .sum()
                    .reset_index()
                )
                grouped["stratum_key"] = grouped.apply(
                    lambda row: tuple(str(row[column]) for column in group_columns),
                    axis=1,
                )
                grouped = grouped[["stratum_key", *IR_AUDIT_COUNT_COLUMNS]]
            else:
                grouped = pd.DataFrame(
                    [
                        {
                            "stratum_key": ("all",),
                            **{
                                column: int(qc[column].sum())
                                for column in IR_AUDIT_COUNT_COLUMNS
                            },
                        }
                    ]
                )
            grouped["stratum_type"] = stratum_type
            partials.append(grouped)
        del qc
    if not partials:
        return pd.DataFrame(
            columns=["stratum_type", "stratum_key", *IR_AUDIT_COUNT_COLUMNS]
        )
    combined = pd.concat(partials, ignore_index=True)
    return (
        combined.groupby(["stratum_type", "stratum_key"], sort=True, as_index=False)[
            list(IR_AUDIT_COUNT_COLUMNS)
        ]
        .sum()
    )


def _read_transcript_annotation(
    gtf: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, int, int]]]:
    transcript_to_gene: dict[str, str] = {}
    gene_loci: dict[str, tuple[str, int, int]] = {}
    attr_re = re.compile(r'([^\s;]+)\s+"([^"]*)"')
    with gtf.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"invalid GTF row at {gtf}:{line_number}")
            if fields[2] != "transcript":
                continue
            attrs = dict(attr_re.findall(fields[8]))
            transcript_id = attrs.get("transcript_id", "")
            gene_id = attrs.get("gene_id", "")
            if not transcript_id or not gene_id:
                raise ValueError(f"GTF transcript misses identity at {gtf}:{line_number}")
            previous = transcript_to_gene.setdefault(transcript_id, gene_id)
            if previous != gene_id:
                raise ValueError(f"GTF transcript maps to multiple genes: {transcript_id}")
            chrom = str(fields[0])
            start = int(fields[3]) - 1
            end = int(fields[4])
            previous_locus = gene_loci.get(gene_id)
            if previous_locus is None:
                gene_loci[gene_id] = (chrom, start, end)
            else:
                if previous_locus[0] != chrom:
                    raise ValueError(f"GTF gene spans multiple chromosomes: {gene_id}")
                gene_loci[gene_id] = (
                    chrom,
                    min(previous_locus[1], start),
                    max(previous_locus[2], end),
                )
    return transcript_to_gene, gene_loci


def _read_transcript_to_gene(gtf: Path) -> dict[str, str]:
    transcript_to_gene, _ = _read_transcript_annotation(gtf)
    return transcript_to_gene


def _candidate_support_rows(
    candidate_ids: Sequence[str],
    selected: pd.DataFrame,
    support: Mapping[str, Counter[str]],
) -> list[dict[str, object]]:
    rows = []
    for gene_id in candidate_ids:
        counts = support[gene_id]
        train_informative = int(counts[f"train:{INFORMATIVE_FATE}"])
        rows.append(
            {
                "target_gene_id": gene_id,
                "support_status": (
                    "likelihood_fit_train_positive_informative_mass"
                    if train_informative > 0
                    else "graph_only_zero_train_informative_mass"
                ),
                "train_captured_gene_assigned_mass": int(counts["train:captured"]),
                "train_pre_compatibility_qc_pass_mass": int(counts["train:pre_qc"]),
                "train_empty_compatible_mass": int(counts[f"train:{EMPTY_FATE}"]),
                "train_positive_informative_ec_mass": train_informative,
                "train_full_set_compatible_mass": int(counts[f"train:{FULL_FATE}"]),
                "validation_captured_gene_assigned_mass": int(counts["val:captured"]),
                "validation_pre_compatibility_qc_pass_mass": int(counts["val:pre_qc"]),
                "validation_empty_compatible_mass": int(counts[f"val:{EMPTY_FATE}"]),
                "validation_positive_informative_ec_mass": int(
                    counts[f"val:{INFORMATIVE_FATE}"]
                ),
                "validation_full_set_compatible_mass": int(counts[f"val:{FULL_FATE}"]),
                "DTU_score": (
                    None
                    if pd.isna(selected.loc[gene_id, "DTU_score"])
                    else float(selected.loc[gene_id, "DTU_score"])
                ),
                "top_DTU_gene": bool(selected.loc[gene_id, "top_DTU_gene"]),
                "G_fit_rule": "train_positive_informative_ec_mass_gt_0",
            }
        )
    return rows


def _update_audit_groups(
    groups: dict[tuple[str, ...], Counter[str]],
    *,
    split: str,
    library_id: str,
    donor_id: str,
    gene_id: str,
    cell_state: str,
    qc_pass: bool,
    final_fate: str,
    molecule_count: int,
) -> None:
    keys = (
        ("global", "all"),
        ("split", split),
        ("split_library", split, library_id),
        ("split_donor", split, donor_id),
        ("split_gene", split, gene_id),
        ("split_cell_state", split, cell_state),
    )
    for key in keys:
        counts = groups[key]
        counts["captured_rows"] += 1
        counts["captured_mass"] += molecule_count
        if qc_pass:
            counts["pre_qc_rows"] += 1
            counts["pre_qc_mass"] += molecule_count
            counts[f"{final_fate}:rows"] += 1
            counts[f"{final_fate}:mass"] += molecule_count
        else:
            counts["technical_failure_rows"] += 1
            counts["technical_failure_mass"] += molecule_count


def _long_read_audit_frame(
    groups: Mapping[tuple[str, ...], Counter[str]],
    candidate_ids: Sequence[str],
) -> pd.DataFrame:
    mutable = {key: Counter(value) for key, value in groups.items()}
    for split in ("train", "val"):
        for gene_id in candidate_ids:
            mutable.setdefault(("split_gene", split, gene_id), Counter())
    rows = []
    for key in sorted(mutable):
        counts = mutable[key]
        denominator = int(counts["pre_qc_mass"])
        terminal_mass = sum(int(counts[f"{fate}:mass"]) for fate in COMPATIBILITY_FATES)
        if terminal_mass != denominator:
            raise AssertionError(f"compatibility mass is not conserved for stratum {key}")
        for fate in COMPATIBILITY_FATES:
            rows.append(
                {
                    "stratum_type": key[0],
                    "stratum_values": list(key[1:]),
                    "captured_gene_assigned_row_count": int(counts["captured_rows"]),
                    "captured_gene_assigned_molecule_mass": int(counts["captured_mass"]),
                    "pre_compatibility_qc_pass_row_count": int(counts["pre_qc_rows"]),
                    "pre_compatibility_qc_pass_molecule_mass": denominator,
                    "technical_qc_failure_row_count": int(counts["technical_failure_rows"]),
                    "technical_qc_failure_molecule_mass": int(counts["technical_failure_mass"]),
                    "terminal_fate": fate,
                    "terminal_row_count": int(counts[f"{fate}:rows"]),
                    "terminal_molecule_mass": int(counts[f"{fate}:mass"]),
                    "terminal_fraction": (
                        float(counts[f"{fate}:mass"]) / denominator
                        if denominator > 0
                        else None
                    ),
                    "fraction_status": "estimated" if denominator > 0 else "not_estimable",
                    "mass_conservation_pass": terminal_mass == denominator,
                }
            )
    return pd.DataFrame(rows)


def _split_conservation_from_support(
    support: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    for split, prefix in (("train", "train"), ("val", "validation")):
        captured = int(support[f"{prefix}_captured_gene_assigned_mass"].sum())
        pre_qc = int(support[f"{prefix}_pre_compatibility_qc_pass_mass"].sum())
        empty = int(support[f"{prefix}_empty_compatible_mass"].sum())
        informative = int(support[f"{prefix}_positive_informative_ec_mass"].sum())
        full = int(support[f"{prefix}_full_set_compatible_mass"].sum())
        rows.append(
            {
                "split": split,
                "captured_molecule_mass": captured,
                "pre_qc_pass_molecule_mass": pre_qc,
                "technical_qc_failure_molecule_mass": captured - pre_qc,
                "empty_compatible_molecule_mass": empty,
                "proper_subset_compatible_molecule_mass": informative,
                "full_set_compatible_molecule_mass": full,
                "other_explicit_fate_molecule_mass": 0,
            }
        )
        if empty + informative + full != pre_qc:
            raise AssertionError(f"split compatibility mass is not conserved: {split}")
    return rows


def _path_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _git_head(workdir: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "not_available"


def _chrom_key(value: str) -> tuple[int, object]:
    match = re.fullmatch(r"chr(\d+)", value)
    if match:
        return (0, int(match.group(1)))
    return (1, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-primary-records", type=int)
    parser.add_argument("--chromosome", action="append", dest="chromosomes")
    parser.add_argument(
        "--merge-shards",
        type=Path,
        help="merge complete single-chromosome outputs under this directory",
    )
    parser.add_argument(
        "--refresh-observation-audit",
        type=Path,
        help="refresh provenance fields in an existing complete artifact",
    )
    parser.add_argument(
        "--reconcile-ont-matrix",
        type=Path,
        help="join the frozen ONT hard-count matrix to compatible cell-gene mass",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.reconcile_ont_matrix is not None:
        if args.refresh_observation_audit is not None or args.merge_shards is not None:
            raise ValueError("matrix reconciliation cannot be combined with other modes")
        if args.output_dir is not None:
            raise ValueError("matrix reconciliation writes into the selected artifact")
        if args.max_primary_records is not None or args.chromosomes is not None:
            raise ValueError(
                "matrix reconciliation does not accept record/chromosome filters"
            )
        result = reconcile_ont_matrix_with_compatibility(
            args.config, args.reconcile_ont_matrix
        )
    elif args.refresh_observation_audit is not None:
        if args.merge_shards is not None or args.output_dir is not None:
            raise ValueError(
                "observation-audit refresh does not accept merge/output options"
            )
        if args.max_primary_records is not None or args.chromosomes is not None:
            raise ValueError(
                "observation-audit refresh does not accept record/chromosome filters"
            )
        result = refresh_observation_process_audit(
            args.config, args.refresh_observation_audit
        )
    elif args.merge_shards is not None:
        if args.output_dir is None:
            raise ValueError("--output-dir is required with --merge-shards")
        if args.max_primary_records is not None or args.chromosomes is not None:
            raise ValueError("merge mode does not accept record/chromosome filters")
        result = merge_compatible_ec_shards(
            args.config, args.merge_shards, args.output_dir
        )
    else:
        result = build_compatible_ec_artifact(
            args.config,
            output_dir=args.output_dir,
            max_primary_records=args.max_primary_records,
            chromosomes=args.chromosomes,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
