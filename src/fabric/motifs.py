"""FABRIC V2 physical motif events and graph-anchor routing.

This module owns only split-neutral catalog semantics: motif parsing/scanning,
physical-hit collapse, graph-anchor routes, the per-anchor evidence-class cap,
and structural route audits.  Cell observations, train-only gates, interaction
support, and model tensors are built in :mod:`fabric.dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DNA_ORIENTATIONS = ("same_transcript", "opposite_transcript")
RNA_ORIENTATIONS = ("transcribed",)
FACTOR_IDENTITY_KINDS = (
    "unique",
    "factor_equivalence_group",
    "accessibility_only",
)
CAP_EVIDENCE_CLASSES = ("motif_anchored", "accessibility_only")
EVENT_KINDS = (
    "TSS_PROXIMAL",
    "SPLICE_SITE_PROXIMAL",
    "EXON_CONTAINED",
    "DNA_INTRAGENIC",
    "PAS_PROXIMAL",
    "MULTI_ANCHOR",
)
GEOMETRY_KINDS = ("site_window", "edge_contained")


PHYSICAL_EVENT_COLUMNS = (
    "event_id",
    "target_gene_id",
    "factor_entity_id",
    "factor_identity_kind",
    "cap_evidence_class",
    "candidate_factor_ids",
    "activity_entity_id",
    "activity_gene_ids",
    "activity_proxy_rule",
    "modality",
    "event_kind",
    "motif_id",
    "motif_equivalence_family_id",
    "source_motif_ids",
    "chromosome",
    "start",
    "end",
    "strand",
    "source_hit_coordinates",
    "motif_score",
    "orientation",
    "peak_id",
    "peak_support",
    "gate_key_id",
    "is_self_factor",
    "source_valid",
    "has_retained_route",
    "gate_key_active",
    "model_active",
    "admission_reasons",
)

EVENT_ROUTE_COLUMNS = (
    "route_id",
    "event_id",
    "target_gene_id",
    "modality",
    "anchor_region_id",
    "anchor_site_id",
    "edge_id",
    "route_weight",
    "region_type",
    "anchor_type",
    "transcript_oriented_side",
    "signed_distance_bp",
    "edge_relative_position",
    "distance_to_5prime_boundary_bp",
    "distance_to_3prime_boundary_bp",
    "geometry_kind",
)


@dataclass(frozen=True)
class PWM:
    motif_id: str
    name: str
    probabilities: np.ndarray

    @property
    def width(self) -> int:
        return int(self.probabilities.shape[0])


def transcript_relative_interval(
    position: int,
    rel_start: int,
    rel_end: int,
    strand: str,
) -> tuple[int, int]:
    """Map one transcript-oriented half-open window to genomic coordinates."""

    if rel_end <= rel_start:
        raise ValueError("transcript-relative interval must have positive width")
    if strand == "+":
        return position + rel_start, position + rel_end
    if strand == "-":
        return position - rel_end, position - rel_start
    raise ValueError("transcript-relative interval strand must be + or -")


@dataclass(frozen=True)
class FactorCatalogResult:
    """Frozen factor/entity identities and motif-to-entity mapping."""

    factors: pd.DataFrame
    motif_mapping: pd.DataFrame
    excluded_motifs: pd.DataFrame


@dataclass(frozen=True)
class PhysicalEventCatalog:
    """One split-neutral V2 event catalog with both audit and production routes."""

    physical_events: pd.DataFrame
    candidate_routes: pd.DataFrame
    event_routes: pd.DataFrame
    cap_audit: pd.DataFrame
    route_degree_cap_audit: pd.DataFrame
    catalog_burden: pd.DataFrame


def parse_meme_motifs(path: str | Path) -> dict[str, PWM]:
    """Parse a frozen MEME probability-matrix library in source order."""

    lines = Path(path).read_text().splitlines()
    motifs: dict[str, PWM] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("MOTIF "):
            index += 1
            continue
        fields = line.split(maxsplit=2)
        motif_id = fields[1]
        name = fields[2] if len(fields) > 2 else motif_id
        index += 1
        while index < len(lines) and "letter-probability matrix:" not in lines[index]:
            index += 1
        if index == len(lines):
            raise ValueError(f"MEME motif {motif_id} has no probability matrix")
        width_match = re.search(r"\bw=\s*(\d+)", lines[index])
        if width_match is None:
            raise ValueError(f"MEME motif {motif_id} has no matrix width")
        width = int(width_match.group(1))
        matrix: list[list[float]] = []
        for row in lines[index + 1 : index + 1 + width]:
            values = [float(value) for value in row.split()]
            if len(values) != 4:
                raise ValueError(f"MEME motif {motif_id} is not a four-letter PWM")
            matrix.append(values)
        probabilities = np.asarray(matrix, dtype=np.float64)
        _validate_pwm(motif_id, probabilities)
        if motif_id in motifs:
            raise ValueError(f"duplicate MEME motif ID: {motif_id}")
        motifs[motif_id] = PWM(motif_id, name, probabilities)
        index += width + 1
    if not motifs:
        raise ValueError("MEME library contains no motifs")
    return motifs


def parse_cisbp_motifs(
    directory: str | Path, *, motif_ids: Sequence[str] | None = None
) -> dict[str, PWM]:
    """Parse CisBP-RNA A/C/G/U tables, mapping U to the internal T column."""

    root = Path(directory)
    selected = None if motif_ids is None else {str(value) for value in motif_ids}
    motifs: dict[str, PWM] = {}
    for path in sorted(root.glob("*.txt")):
        motif_id = path.stem
        if selected is not None and motif_id not in selected:
            continue
        frame = pd.read_csv(path, sep="\t")
        if list(frame.columns)[:5] != ["Pos", "A", "C", "G", "U"]:
            raise ValueError(f"CisBP PWM has unexpected columns: {path}")
        probabilities = frame[["A", "C", "G", "U"]].to_numpy(dtype=np.float64)
        _validate_pwm(motif_id, probabilities)
        motifs[motif_id] = PWM(motif_id, motif_id, probabilities)
    if selected is not None:
        missing = sorted(selected - set(motifs))
        if missing:
            raise FileNotFoundError(f"CisBP PWM files are absent: {missing[:10]}")
    return motifs


def build_factor_catalog(
    motif_mapping: pd.DataFrame,
    *,
    frozen_rna_gene_axis: Sequence[str] | None = None,
) -> FactorCatalogResult:
    """Validate the V2 motif/entity mapping without choosing an expressed owner.

    ``motif_mapping`` has one row per ``(modality, motif_id)`` and must declare
    ``factor_identity_kind``, ``factor_entity_id``, ``candidate_factor_ids``,
    ``activity_entity_id``, and ``activity_gene_ids``.  A group remains a group;
    no cell-specific expression is read here.
    """

    required = {
        "modality",
        "motif_id",
        "factor_identity_kind",
        "factor_entity_id",
        "candidate_factor_ids",
        "activity_entity_id",
        "activity_gene_ids",
        "activity_proxy_rule",
    }
    _require_columns(motif_mapping, required, "factor mapping")
    mapping = motif_mapping.copy().reset_index(drop=True)
    mapping["modality"] = mapping["modality"].astype(str).str.upper()
    if not set(mapping["modality"]).issubset({"DNA", "RNA"}):
        raise ValueError("factor mapping modality must be DNA or RNA")
    if mapping.duplicated(["modality", "motif_id"]).any():
        raise ValueError("one motif may map to only one frozen factor/entity")
    if "motif_equivalence_family_id" not in mapping:
        mapping["motif_equivalence_family_id"] = mapping["motif_id"].astype(str)
    else:
        mapping["motif_equivalence_family_id"] = [
            _nullable_text(family_id) or str(motif_id)
            for motif_id, family_id in zip(
                mapping["motif_id"],
                mapping["motif_equivalence_family_id"],
                strict=True,
            )
        ]

    axis = None if frozen_rna_gene_axis is None else tuple(map(str, frozen_rna_gene_axis))
    if axis is not None and len(axis) != len(set(axis)):
        raise ValueError("frozen RNA source gene axis must be unique")
    axis_set = set(axis or ())
    rows: list[dict[str, object]] = []
    for row in mapping.itertuples(index=False):
        kind = str(row.factor_identity_kind)
        if kind not in {"unique", "factor_equivalence_group"}:
            raise ValueError("motif mappings require unique or factor-equivalence identity")
        entity = _required_text(row.factor_entity_id, "factor_entity_id")
        activity_entity = _required_text(row.activity_entity_id, "activity_entity_id")
        if entity != activity_entity:
            raise ValueError("motif factor_entity_id must equal activity_entity_id")
        factors = _string_list(row.candidate_factor_ids, "candidate_factor_ids")
        genes = _string_list(row.activity_gene_ids, "activity_gene_ids")
        if not factors or len(factors) != len(set(factors)):
            raise ValueError("candidate_factor_ids must be non-empty and unique")
        if not genes or len(genes) != len(set(genes)):
            raise ValueError("activity_gene_ids must be non-empty and unique")
        if kind == "unique" and factors != [entity]:
            raise ValueError("unique factor candidate_factor_ids must contain only itself")
        if kind == "factor_equivalence_group" and len(factors) < 2:
            raise ValueError("factor-equivalence groups require at least two candidates")
        invalid_members = [] if axis is None else sorted(set(genes) - axis_set)
        source_valid = not invalid_members
        rows.append(
            {
                "activity_entity_id": activity_entity,
                "factor_entity_id": entity,
                "factor_identity_kind": kind,
                "candidate_factor_ids": factors,
                "activity_gene_ids": genes,
                "activity_proxy_rule": str(row.activity_proxy_rule),
                "source_valid": source_valid,
                "source_failure_reasons": (
                    [] if source_valid else ["invalid_activity_axis"]
                ),
            }
        )
    entities = pd.DataFrame(rows).drop_duplicates("activity_entity_id")
    for entity_id, group in pd.DataFrame(rows).groupby("activity_entity_id", sort=False):
        for column in (
            "factor_entity_id",
            "factor_identity_kind",
            "candidate_factor_ids",
            "activity_gene_ids",
            "activity_proxy_rule",
        ):
            if len({_canonical_json(value) for value in group[column]}) != 1:
                raise ValueError(f"activity entity {entity_id} has inconsistent {column}")
    mapping = mapping.merge(
        entities[["activity_entity_id", "source_valid", "source_failure_reasons"]],
        on="activity_entity_id",
        how="left",
        validate="many_to_one",
    )
    return FactorCatalogResult(
        factors=entities.sort_values("activity_entity_id", kind="mergesort").reset_index(drop=True),
        motif_mapping=mapping.sort_values(
            ["modality", "factor_entity_id", "motif_id"], kind="mergesort"
        ).reset_index(drop=True),
        excluded_motifs=pd.DataFrame(columns=["modality", "motif_id", "reason"]),
    )


def scan_pwm(
    sequence: str,
    motif: PWM,
    *,
    minimum_relative_score: float,
    reverse_strand: bool,
) -> list[tuple[int, str, float]]:
    """Return fixed motif hits as ``(offset, source strand, relative score)``."""

    if not 0 <= minimum_relative_score <= 1:
        raise ValueError("minimum_relative_score must lie in [0, 1]")
    encoded = np.fromiter(
        ("ACGT".find(base) for base in sequence.upper().replace("U", "T")),
        dtype=np.int8,
    )
    hits = _scan_encoded(encoded, motif.probabilities, minimum_relative_score, "+")
    if reverse_strand:
        hits.extend(
            _scan_encoded(
                encoded,
                motif.probabilities[::-1, ::-1],
                minimum_relative_score,
                "-",
            )
        )
    return sorted(hits, key=lambda value: (value[0], value[1], -value[2]))


def scan_motif_regions(
    regions: pd.DataFrame,
    motifs: Mapping[str, PWM],
    motif_mapping: pd.DataFrame,
    *,
    modality: str,
    minimum_relative_score: float,
) -> pd.DataFrame:
    """Scan explicit split-neutral windows and emit uncollapsed physical hits.

    Regions are provenance windows, not event identities.  If two overlapping
    windows expose the same genomic hit, their rows retain distinct
    ``source_window_id`` values and are collapsed before routing.
    """

    required = {
        "target_gene_id",
        "source_window_id",
        "chromosome",
        "window_start",
        "window_end",
        "strand",
        "sequence",
    }
    _require_columns(regions, required, "motif scan regions")
    modality = str(modality).upper()
    if modality not in {"DNA", "RNA"}:
        raise ValueError("modality must be DNA or RNA")
    mapping = motif_mapping.loc[
        motif_mapping["modality"].astype(str).str.upper() == modality
    ]
    required_mapping = {
        "motif_id",
        "factor_entity_id",
        "factor_identity_kind",
        "candidate_factor_ids",
        "activity_entity_id",
        "activity_gene_ids",
        "activity_proxy_rule",
    }
    _require_columns(mapping, required_mapping, "motif mapping")
    if mapping["motif_id"].astype(str).duplicated().any():
        raise ValueError("motif scan mapping contains duplicate motif IDs")
    rows: list[dict[str, object]] = []
    for region in regions.itertuples(index=False):
        start0, end0 = int(region.window_start), int(region.window_end)
        if start0 < 0 or end0 <= start0 or len(str(region.sequence)) != end0 - start0:
            raise ValueError("motif scan window coordinates and sequence length differ")
        gene_strand = str(region.strand)
        if gene_strand not in {"+", "-"}:
            raise ValueError("motif scan region strand must be '+' or '-'")
        for identity in mapping.itertuples(index=False):
            motif_id = str(identity.motif_id)
            family_id = _nullable_text(
                getattr(identity, "motif_equivalence_family_id", None)
            ) or motif_id
            if motif_id not in motifs:
                raise ValueError(f"factor mapping references absent motif {motif_id}")
            motif = motifs[motif_id]
            for offset, hit_strand, score in scan_pwm(
                str(region.sequence),
                motif,
                minimum_relative_score=minimum_relative_score,
                reverse_strand=modality == "DNA",
            ):
                if modality == "DNA" or gene_strand == "+":
                    hit_start = start0 + offset
                else:
                    hit_start = end0 - offset - motif.width
                hit_end = hit_start + motif.width
                if modality == "RNA":
                    orientation = "transcribed"
                else:
                    orientation = (
                        "same_transcript"
                        if (hit_strand == "+") == (gene_strand == "+")
                        else "opposite_transcript"
                    )
                rows.append(
                    {
                        "source_hit_id": (
                            f"{region.source_window_id}|{motif_id}|{offset}|{hit_strand}"
                        ),
                        "source_window_id": str(region.source_window_id),
                        "target_gene_id": str(region.target_gene_id),
                        "factor_entity_id": str(identity.factor_entity_id),
                        "factor_identity_kind": str(identity.factor_identity_kind),
                        "candidate_factor_ids": _string_list(
                            identity.candidate_factor_ids, "candidate_factor_ids"
                        ),
                        "activity_entity_id": str(identity.activity_entity_id),
                        "activity_gene_ids": _string_list(
                            identity.activity_gene_ids, "activity_gene_ids"
                        ),
                        "activity_proxy_rule": str(identity.activity_proxy_rule),
                        "modality": modality,
                        "motif_id": motif_id,
                        "motif_equivalence_family_id": family_id,
                        "chromosome": str(region.chromosome),
                        "start": hit_start,
                        "end": hit_end,
                        "strand": gene_strand,
                        "motif_score": float(score),
                        "calibrated_motif_quality": float(
                            getattr(identity, "calibrated_motif_quality", np.nan)
                        ),
                        "orientation": orientation,
                        "peak_id": _nullable_text(getattr(region, "peak_id", None)),
                        "peak_support": float(getattr(region, "peak_support", 0.0)),
                        "source_priority": int(getattr(identity, "source_priority", 0)),
                        "source_local_rank": float(
                            getattr(identity, "source_local_rank", np.nan)
                        ),
                        "source_valid": bool(getattr(identity, "source_valid", True)),
                    }
                )
    scanned = pd.DataFrame(rows)
    if scanned.empty:
        return scanned
    # A PWM scan score is a within-PWM sequence score, not a calibrated
    # cross-PWM binding-strength scale.  It remains provenance unless the
    # mapping explicitly supplies ``calibrated_motif_quality``.  Otherwise a
    # deterministic source-local rank is fitted without looking at outcomes.
    missing_rank = ~np.isfinite(scanned["source_local_rank"].to_numpy(dtype=float))
    if bool(missing_rank.any()):
        rank_groups = [
            "target_gene_id",
            "modality",
            "motif_id",
            "source_window_id",
        ]
        derived = scanned.groupby(rank_groups, sort=False)["motif_score"].rank(
            method="first", ascending=False
        )
        scanned.loc[missing_rank, "source_local_rank"] = derived.loc[missing_rank]
    if modality == "DNA":
        scanned = assign_unique_peak_to_dna_hits(scanned)
    return scanned.sort_values("source_hit_id", kind="mergesort").reset_index(drop=True)


def assign_unique_peak_to_dna_hits(source_hits: pd.DataFrame) -> pd.DataFrame:
    """Assign each factor-specific genomic hit to exactly one frozen peak.

    Overlapping scan windows may expose the same genomic source hit.  Peak
    ownership is chosen before physical collapse by descending peak support and
    stable peak ID; all source-window IDs remain as provenance on the retained
    row.  This operation never duplicates a hit across overlapping peaks.
    """

    if source_hits.empty:
        return source_hits.copy()
    required = {
        "target_gene_id",
        "modality",
        "factor_entity_id",
        "motif_equivalence_family_id",
        "motif_id",
        "chromosome",
        "start",
        "end",
        "strand",
        "orientation",
        "peak_id",
        "peak_support",
        "source_hit_id",
        "source_window_id",
    }
    _require_columns(source_hits, required, "DNA source-hit peak assignment")
    if set(source_hits["modality"].astype(str)) != {"DNA"}:
        raise ValueError("unique peak assignment accepts only DNA motif hits")
    if source_hits["peak_id"].map(_nullable_text).isna().any():
        raise ValueError("factor-specific DNA hits require a frozen peak assignment")
    identity = [
        "target_gene_id",
        "factor_entity_id",
        "motif_equivalence_family_id",
        "motif_id",
        "chromosome",
        "start",
        "end",
        "strand",
        "orientation",
    ]
    rows: list[pd.Series] = []
    for _, group in source_hits.groupby(identity, sort=True, dropna=False):
        ranked = group.assign(_stable_peak_id=group["peak_id"].astype(str)).sort_values(
            ["peak_support", "_stable_peak_id", "source_hit_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        selected = ranked.iloc[0].drop(labels=["_stable_peak_id"]).copy()
        selected["source_window_ids"] = sorted(set(group["source_window_id"].astype(str)))
        selected["overlapping_peak_ids"] = sorted(set(group["peak_id"].astype(str)))
        selected["source_hit_id"] = "dna-hit:" + hashlib.sha256(
            _canonical_json(
                {
                    **{column: selected[column] for column in identity},
                    "peak_id": selected["peak_id"],
                }
            ).encode("utf-8")
        ).hexdigest()[:24]
        rows.append(selected)
    result = pd.DataFrame(rows)
    if result["source_hit_id"].astype(str).duplicated().any():
        raise RuntimeError("DNA unique-peak assignment produced duplicate source hits")
    return result.reset_index(drop=True)


def accessibility_only_hits(peaks: pd.DataFrame, *, target_gene_id: str) -> pd.DataFrame:
    """Create explicit accessibility-only source hits from uniquely assigned peaks."""

    _require_columns(
        peaks,
        {"peak_id", "chromosome", "start", "end", "strand", "peak_support"},
        "accessibility-only peaks",
    )
    support = peaks["peak_support"].to_numpy(dtype=np.float64)
    if not np.isfinite(support).all() or bool((support < 0).any()):
        raise ValueError("accessibility-only peak support must be finite and non-negative")
    rows: list[dict[str, object]] = []
    for peak in peaks.itertuples(index=False):
        rows.append(
            {
                "source_hit_id": f"OPEN|{target_gene_id}|{peak.peak_id}",
                "source_window_id": f"peak:{peak.peak_id}",
                "target_gene_id": str(target_gene_id),
                "factor_entity_id": None,
                "factor_identity_kind": "accessibility_only",
                "candidate_factor_ids": [],
                "activity_entity_id": None,
                "activity_gene_ids": [],
                "activity_proxy_rule": None,
                "modality": "DNA",
                "motif_id": None,
                "motif_equivalence_family_id": None,
                "chromosome": str(peak.chromosome),
                "start": int(peak.start),
                "end": int(peak.end),
                "strand": str(peak.strand),
                "motif_score": np.nan,
                "calibrated_motif_quality": np.nan,
                "orientation": None,
                "peak_id": str(peak.peak_id),
                "peak_support": float(peak.peak_support),
                "source_priority": 0,
                "source_local_rank": np.nan,
                "source_valid": bool(getattr(peak, "source_valid", True)),
            }
        )
    return pd.DataFrame(rows)


def collapse_physical_events(
    source_hits: pd.DataFrame,
    *,
    minimum_overlap_bp: int = 1,
    minimum_reciprocal_overlap: float = 0.0,
) -> pd.DataFrame:
    """Collapse equivalent overlapping PWM hits by connected components.

    The overlap graph is transitive by construction; chained overlaps therefore
    form one component.  Non-equivalent motif families and different peak IDs
    never share a component.
    """

    if minimum_overlap_bp <= 0 or not 0 <= minimum_reciprocal_overlap <= 1:
        raise ValueError("physical-collapse overlap thresholds are invalid")
    required = {
        "source_hit_id",
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
        "source_local_rank",
        "source_priority",
        "orientation",
        "peak_id",
        "peak_support",
        "source_valid",
    }
    _require_columns(source_hits, required, "physical event source hits")
    if source_hits.empty:
        return _empty_frame(PHYSICAL_EVENT_COLUMNS)
    hits = source_hits.copy().reset_index(drop=True)
    if hits["source_hit_id"].astype(str).duplicated().any():
        raise ValueError("source_hit_id values must be unique")
    intervals = hits[["start", "end"]].to_numpy(dtype=np.int64)
    if bool((intervals[:, 0] < 0).any()) or bool((intervals[:, 1] <= intervals[:, 0]).any()):
        raise ValueError("physical source hits require positive half-open intervals")
    if not set(hits["modality"].astype(str)).issubset({"DNA", "RNA"}):
        raise ValueError("physical source-hit modality must be DNA or RNA")
    if not set(hits["strand"].astype(str)).issubset({"+", "-"}):
        raise ValueError("physical source-hit strand must be '+' or '-'")
    peak_support = hits["peak_support"].to_numpy(dtype=np.float64)
    if not np.isfinite(peak_support).all() or bool((peak_support < 0).any()):
        raise ValueError("physical source-hit peak support must be finite and non-negative")
    motif_rows = hits["factor_identity_kind"].astype(str) != "accessibility_only"
    if bool(motif_rows.any()):
        priorities = pd.to_numeric(hits.loc[motif_rows, "source_priority"], errors="coerce")
        if not np.isfinite(priorities).all() or not np.equal(priorities, np.floor(priorities)).all():
            raise ValueError("motif source_priority must be a finite integer")
        calibration = pd.to_numeric(
            hits.loc[motif_rows, "calibrated_motif_quality"], errors="coerce"
        )
        ranks = pd.to_numeric(hits.loc[motif_rows, "source_local_rank"], errors="coerce")
        if bool((~np.isfinite(calibration.to_numpy()) & ~np.isfinite(ranks.to_numpy())).any()):
            raise ValueError(
                "each motif source hit requires calibrated quality or source-local rank"
            )

    motif_mask = hits["factor_identity_kind"].astype(str) != "accessibility_only"
    motif_hits = hits.loc[motif_mask].copy()
    open_hits = hits.loc[~motif_mask].copy()
    components: list[pd.DataFrame] = []
    motif_bucket = [
        "target_gene_id",
        "modality",
        "factor_entity_id",
        "motif_equivalence_family_id",
        "chromosome",
        "strand",
        "peak_id",
    ]
    for _, bucket in motif_hits.groupby(motif_bucket, sort=True, dropna=False):
        bucket = bucket.sort_values(
            ["start", "end", "motif_id", "source_hit_id"], kind="mergesort"
        ).reset_index(drop=True)
        parent = list(range(len(bucket)))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        starts = bucket["start"].to_numpy(dtype=np.int64)
        ends = bucket["end"].to_numpy(dtype=np.int64)
        for left in range(len(bucket)):
            for right in range(left + 1, len(bucket)):
                if starts[right] >= ends[left]:
                    break
                overlap = min(ends[left], ends[right]) - max(starts[left], starts[right])
                reciprocal = min(
                    overlap / (ends[left] - starts[left]),
                    overlap / (ends[right] - starts[right]),
                )
                if overlap >= minimum_overlap_bp and reciprocal >= minimum_reciprocal_overlap:
                    union(left, right)
        labels = [find(index) for index in range(len(bucket))]
        for label in sorted(set(labels)):
            components.append(bucket.loc[np.asarray(labels) == label])

    # Accessibility-only identity includes the exact peak and interval.  Exact
    # duplicates are one physical record; overlapping distinct peaks remain distinct.
    open_key = [
        "target_gene_id",
        "chromosome",
        "start",
        "end",
        "strand",
        "peak_id",
    ]
    for _, component in open_hits.groupby(open_key, sort=True, dropna=False):
        components.append(component)

    event_rows: list[dict[str, object]] = []
    for component in components:
        component = component.copy()
        first = component.iloc[0]
        kind = str(first["factor_identity_kind"])
        _validate_component_identity(component)
        if kind == "accessibility_only":
            representative = component.sort_values(
                ["peak_support", "start", "end", "source_hit_id"],
                ascending=[False, True, True, True],
                kind="mergesort",
            ).iloc[0]
        else:
            representative = _representative_motif_hit(component)
        canonical_start = int(representative["start"])
        canonical_end = int(representative["end"])
        motif_id = None if kind == "accessibility_only" else str(representative["motif_id"])
        family_id = (
            None
            if kind == "accessibility_only"
            else str(representative["motif_equivalence_family_id"])
        )
        identity = {
            "target_gene_id": str(representative["target_gene_id"]),
            "modality": str(representative["modality"]),
            "factor_entity_id": _nullable_text(representative["factor_entity_id"]),
            "motif_equivalence_family_id": family_id,
            "chromosome": str(representative["chromosome"]),
            "start": canonical_start,
            "end": canonical_end,
            "strand": str(representative["strand"]),
            "peak_id": _nullable_text(representative["peak_id"]),
            "factor_identity_kind": kind,
        }
        event_id = "event:" + hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:24]
        source_motifs = (
            []
            if kind == "accessibility_only"
            else sorted(set(component["motif_id"].astype(str)))
        )
        source_coordinates = []
        for source_row in component.sort_values(
            "source_hit_id", kind="mergesort"
        ).itertuples(index=False):
            source_window_id = _nullable_text(
                getattr(source_row, "source_window_id", None)
            )
            source_window_ids = getattr(source_row, "source_window_ids", None)
            overlapping_peak_ids = getattr(source_row, "overlapping_peak_ids", None)
            source_coordinates.append(
                {
                    "source_hit_id": str(source_row.source_hit_id),
                    "source_window_id": source_window_id,
                    "source_window_ids": (
                        [source_window_id]
                        if _is_missing(source_window_ids) and source_window_id is not None
                        else _string_list(source_window_ids, "source_window_ids")
                    ),
                    "overlapping_peak_ids": (
                        []
                        if _is_missing(overlapping_peak_ids)
                        else _string_list(overlapping_peak_ids, "overlapping_peak_ids")
                    ),
                    "chromosome": str(source_row.chromosome),
                    "start": int(source_row.start),
                    "end": int(source_row.end),
                }
            )
        candidate_factors = _string_list(
            representative["candidate_factor_ids"], "candidate_factor_ids"
        )
        activity_genes = _string_list(
            representative["activity_gene_ids"], "activity_gene_ids"
        )
        if kind != "accessibility_only" and not activity_genes:
            raise ValueError("motif/group events require non-empty activity_gene_ids")
        source_valid = bool(component["source_valid"].astype(bool).all())
        row = {
            "event_id": event_id,
            "target_gene_id": identity["target_gene_id"],
            "factor_entity_id": identity["factor_entity_id"],
            "factor_identity_kind": kind,
            "cap_evidence_class": (
                "accessibility_only" if kind == "accessibility_only" else "motif_anchored"
            ),
            "candidate_factor_ids": candidate_factors,
            "activity_entity_id": _nullable_text(representative["activity_entity_id"]),
            "activity_gene_ids": activity_genes,
            "activity_proxy_rule": _nullable_text(representative["activity_proxy_rule"]),
            "modality": identity["modality"],
            "event_kind": None,
            "motif_id": motif_id,
            "motif_equivalence_family_id": family_id,
            "source_motif_ids": source_motifs,
            "chromosome": identity["chromosome"],
            "start": canonical_start,
            "end": canonical_end,
            "strand": identity["strand"],
            "source_hit_coordinates": source_coordinates,
            "motif_score": (
                np.nan if kind == "accessibility_only" else float(representative["motif_score"])
            ),
            "calibrated_motif_quality": float(
                representative.get("calibrated_motif_quality", np.nan)
            ),
            "orientation": (
                None if kind == "accessibility_only" else str(representative["orientation"])
            ),
            "peak_id": identity["peak_id"],
            "peak_support": float(representative["peak_support"]),
            "gate_key_id": None,
            "is_self_factor": (
                None
                if kind == "accessibility_only"
                else identity["target_gene_id"] in activity_genes
            ),
            "source_valid": source_valid,
            "has_retained_route": False,
            "gate_key_active": False,
            "model_active": False,
            "admission_reasons": ([] if source_valid else ["invalid_source"]),
            # Sorting provenance used only by the static cap.
            "source_priority": int(representative.get("source_priority", 0)),
            "source_local_rank": float(representative.get("source_local_rank", np.nan)),
        }
        _validate_event_identity_row(row)
        event_rows.append(row)
    result = pd.DataFrame(event_rows).sort_values("event_id", kind="mergesort")
    if result["event_id"].duplicated().any():
        raise ValueError("physical-event identity collision remained after collapse")
    return result.reset_index(drop=True)


def build_graph_anchor_regions(
    graph: object,
    *,
    modality: str,
    site_flanks: Mapping[str, tuple[int, int]],
    maximum_short_exon_bp: int,
    contig_lengths: Mapping[str, int],
    gene_bounds: tuple[int, int],
) -> pd.DataFrame:
    """Materialize graph-defined site windows and eligible edge intervals.

    Site flank tuples are transcript-oriented ``(upstream_bp, downstream_bp)``.
    The graph itself is authoritative for which processing edges carry a site.
    """

    modality = str(modality).upper()
    if modality not in {"DNA", "RNA"}:
        raise ValueError("anchor modality must be DNA or RNA")
    if maximum_short_exon_bp <= 0:
        raise ValueError("maximum_short_exon_bp must be positive")
    nodes = graph.nodes.copy()
    edges = graph.edges.copy()
    gene_id = str(graph.gene_id)
    strands = set(nodes["strand"].astype(str)) | set(edges["strand"].astype(str))
    if len(strands) != 1:
        raise ValueError("gene graph must have one strand for anchor routing")
    strand = next(iter(strands))
    chroms = set(nodes["chrom"].astype(str)) | set(edges["chrom"].astype(str))
    if len(chroms) != 1:
        raise ValueError("gene graph must lie on one contig for anchor routing")
    chrom = next(iter(chroms))
    if chrom not in contig_lengths or int(contig_lengths[chrom]) <= 0:
        raise ValueError(f"reference contig length is absent or invalid for {chrom}")
    gene_start, gene_end = map(int, gene_bounds)
    contig_end = int(contig_lengths[chrom])
    if gene_start < 0 or gene_end <= gene_start or gene_end > contig_end:
        raise ValueError("gene contract bounds are invalid for the reference contig")
    edge_by_src: dict[str, list[str]] = {}
    edge_by_dst: dict[str, list[str]] = {}
    for edge in edges.itertuples(index=False):
        edge_by_src.setdefault(str(edge.src_node_id), []).append(str(edge.edge_id))
        edge_by_dst.setdefault(str(edge.dst_node_id), []).append(str(edge.edge_id))
    rows: list[dict[str, object]] = []
    for node in nodes.itertuples(index=False):
        node_type = str(node.node_type)
        if node_type not in site_flanks:
            continue
        upstream, downstream = site_flanks[node_type]
        if upstream < 0 or downstream < 0 or upstream + downstream <= 0:
            raise ValueError(f"invalid site flank for {node_type}")
        anchor = int(node.pos_0based)
        raw_start, raw_end = transcript_relative_interval(
            anchor, -upstream, downstream, strand
        )
        start = max(0, raw_start)
        end = min(contig_end, raw_end)
        if end <= start:
            raise ValueError("clipped site window became empty")
        if node_type in {"TSS", "donor"}:
            carriers = sorted(edge_by_src.get(str(node.node_id), []))
        else:
            carriers = sorted(edge_by_dst.get(str(node.node_id), []))
        for edge_id in carriers:
            rows.append(
                {
                    "target_gene_id": gene_id,
                    "modality": modality,
                    "anchor_region_id": f"{gene_id}:site:{node.node_id}",
                    "anchor_site_id": str(node.node_id),
                    "edge_id": edge_id,
                    "chromosome": str(node.chrom),
                    "strand": strand,
                    "region_start": start,
                    "region_end": end,
                    "raw_region_start": raw_start,
                    "raw_region_end": raw_end,
                    "region_clipped": bool(start != raw_start or end != raw_end),
                    "anchor_position": anchor,
                    "region_type": _site_region_type(node_type),
                    "anchor_type": node_type,
                    "geometry_kind": "site_window",
                }
            )
    for edge in edges.itertuples(index=False):
        start, end = int(edge.start_0based), int(edge.end_0based_exclusive)
        if end <= start:
            continue
        if start < gene_start or end > gene_end or start < 0 or end > contig_end:
            raise ValueError("processing edge extends beyond frozen gene/reference bounds")
        edge_type = str(edge.edge_type)
        if modality == "RNA" and not (
            edge_type == "EXON_CONTINUATION" and end - start <= maximum_short_exon_bp
        ):
            continue
        rows.append(
            {
                "target_gene_id": gene_id,
                "modality": modality,
                "anchor_region_id": f"{gene_id}:edge:{edge.edge_id}",
                "anchor_site_id": None,
                "edge_id": str(edge.edge_id),
                "chromosome": str(edge.chrom),
                "strand": strand,
                "region_start": start,
                "region_end": end,
                "raw_region_start": start,
                "raw_region_end": end,
                "region_clipped": False,
                "anchor_position": np.nan,
                "region_type": (
                    "exon" if edge_type == "EXON_CONTINUATION" else "intragenic"
                ),
                "anchor_type": "EDGE",
                "geometry_kind": "edge_contained",
            }
        )
    anchors = pd.DataFrame(rows)
    if anchors.empty:
        return anchors
    key = ["anchor_region_id", "anchor_site_id", "edge_id"]
    if anchors.duplicated(key).any():
        raise ValueError("graph-anchor route carriers are not unique")
    return anchors.sort_values(key, kind="mergesort", na_position="first").reset_index(
        drop=True
    )


def build_candidate_routes(
    physical_events: pd.DataFrame, anchor_regions: pd.DataFrame
) -> pd.DataFrame:
    """Resolve physical events to all legal graph anchors without nearest fallback."""

    _require_columns(physical_events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    anchor_required = {
        "target_gene_id",
        "modality",
        "anchor_region_id",
        "anchor_site_id",
        "edge_id",
        "chromosome",
        "strand",
        "region_start",
        "region_end",
        "anchor_position",
        "region_type",
        "anchor_type",
        "geometry_kind",
    }
    _require_columns(anchor_regions, anchor_required, "anchor region table")
    rows: list[dict[str, object]] = []
    grouped_events = {
        key: group
        for key, group in physical_events.groupby(
            ["target_gene_id", "modality", "chromosome", "strand"],
            sort=False,
        )
    }
    for anchor in anchor_regions.itertuples(index=False):
        key = (
            str(anchor.target_gene_id),
            str(anchor.modality),
            str(anchor.chromosome),
            str(anchor.strand),
        )
        events = grouped_events.get(key)
        if events is None:
            continue
        events = events.loc[events["source_valid"].astype(bool)]
        if events.empty:
            continue
        region_start, region_end = int(anchor.region_start), int(anchor.region_end)
        geometry = str(anchor.geometry_kind)
        for event in events.itertuples(index=False):
            center = (int(event.start) + int(event.end)) / 2.0
            if geometry == "site_window":
                if not (int(event.start) < region_end and int(event.end) > region_start):
                    continue
                anchor_position = int(anchor.anchor_position)
                if int(event.start) < anchor_position < int(event.end):
                    side = "OVERLAP_ANCHOR"
                    signed_distance = np.nan
                else:
                    signed = center - anchor_position
                    if str(anchor.strand) == "-":
                        signed = -signed
                    side = "UPSTREAM" if signed < 0 else "DOWNSTREAM"
                    signed_distance = signed
                relative = np.nan
                d5 = np.nan
                d3 = np.nan
            elif geometry == "edge_contained":
                if not (region_start <= center < region_end):
                    continue
                span = region_end - region_start
                if span <= 0:
                    raise ValueError("edge-contained anchor requires positive span")
                d5 = (
                    center - region_start
                    if str(anchor.strand) == "+"
                    else region_end - center
                )
                d3 = span - d5
                relative = d5 / span
                if not 0 <= relative <= 1:
                    raise RuntimeError("edge-relative position left [0,1]")
                side = "WITHIN_EDGE"
                signed_distance = np.nan
            else:
                raise ValueError(f"unknown route geometry kind: {geometry}")
            route_identity = {
                "event_id": str(event.event_id),
                "anchor_region_id": str(anchor.anchor_region_id),
                "anchor_site_id": _nullable_text(anchor.anchor_site_id),
                "edge_id": str(anchor.edge_id),
            }
            rows.append(
                {
                    "route_id": "route:"
                    + hashlib.sha256(
                        _canonical_json(route_identity).encode("utf-8")
                    ).hexdigest()[:24],
                    **route_identity,
                    "target_gene_id": str(event.target_gene_id),
                    "modality": str(event.modality),
                    "route_weight": np.nan,
                    "region_type": str(anchor.region_type),
                    "anchor_type": str(anchor.anchor_type),
                    "transcript_oriented_side": side,
                    "signed_distance_bp": float(signed_distance),
                    "edge_relative_position": float(relative),
                    "distance_to_5prime_boundary_bp": float(d5),
                    "distance_to_3prime_boundary_bp": float(d3),
                    "geometry_kind": geometry,
                }
            )
    routes = pd.DataFrame(rows, columns=EVENT_ROUTE_COLUMNS)
    if routes.empty:
        return routes
    key = ["event_id", "anchor_region_id", "anchor_site_id", "edge_id"]
    if routes.duplicated(key).any() or routes["route_id"].duplicated().any():
        raise ValueError("candidate EventRouteTable primary key is not unique")
    return routes.sort_values("route_id", kind="mergesort").reset_index(drop=True)


def cap_and_finalize_routes(
    physical_events: pd.DataFrame,
    candidate_routes: pd.DataFrame,
    *,
    events_per_bucket_cap: int = 16,
    gate_admission: pd.DataFrame | None = None,
) -> PhysicalEventCatalog:
    """Apply the V2 per-anchor/evidence-class cap and production normalization."""

    if events_per_bucket_cap <= 0:
        raise ValueError("events_per_bucket_cap must be positive")
    events = physical_events.copy().reset_index(drop=True)
    _require_columns(events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("PhysicalEventTable event_id values must be unique")
    routes = candidate_routes.copy().reset_index(drop=True)
    _require_columns(routes, set(EVENT_ROUTE_COLUMNS), "candidate EventRouteTable")
    unknown = sorted(set(routes["event_id"].astype(str)) - set(events["event_id"].astype(str)))
    if unknown:
        raise ValueError(f"candidate routes reference unknown events: {unknown[:5]}")
    metadata = events.set_index("event_id")
    valid_event_ids = set(events.loc[events["source_valid"].astype(bool), "event_id"].astype(str))
    invalid_route_mask = ~routes["event_id"].astype(str).isin(valid_event_ids)
    invalid_routes = routes.loc[invalid_route_mask].copy()
    routes_for_cap = routes.loc[~invalid_route_mask].copy()
    if routes_for_cap.empty:
        decisions = routes.assign(
            cap_evidence_class=routes["event_id"].map(metadata["cap_evidence_class"]),
            cap_bucket_id=None,
            cap_rank=-1,
            cap_selected=False,
            cap_reason=np.where(invalid_route_mask, "invalid_source", "no_candidate_route"),
        )
        cap_audit = pd.DataFrame()
        retained = routes.copy()
        retained = retained.iloc[:0]
    else:
        valid_decisions, cap_audit = _cap_route_decisions(
            routes_for_cap, metadata, events_per_bucket_cap
        )
        if invalid_routes.empty:
            decisions = valid_decisions
        else:
            invalid_routes = invalid_routes.assign(
                cap_evidence_class=invalid_routes["event_id"].map(
                    metadata["cap_evidence_class"]
                ),
                cap_bucket_id=None,
                cap_rank=-1,
                cap_selected=False,
                cap_reason="invalid_source",
            )
            decisions = pd.concat([valid_decisions, invalid_routes], ignore_index=True)
            decisions = decisions.sort_values("route_id", kind="mergesort").reset_index(drop=True)
        retained = decisions.loc[decisions["cap_selected"]].copy()
        retained = retained[list(EVENT_ROUTE_COLUMNS)]
        post_counts = retained.groupby("event_id", sort=False).size()
        retained["route_weight"] = retained["event_id"].map(
            {event_id: 1.0 / int(count) for event_id, count in post_counts.items()}
        )
        retained = retained.sort_values("route_id", kind="mergesort").reset_index(drop=True)
        sums = retained.groupby("event_id", sort=False)["route_weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-12, rtol=0):
            raise RuntimeError("production route weights do not sum to one per event")

    retained_ids = set(retained["event_id"].astype(str))
    events["has_retained_route"] = events["event_id"].astype(str).isin(retained_ids)
    event_kind = _event_kind_by_event(retained)
    events["event_kind"] = events["event_id"].astype(str).map(event_kind)
    events.loc[~events["has_retained_route"], "event_kind"] = None
    if gate_admission is not None:
        _require_columns(
            gate_admission,
            {"gate_key_id", "gate_key_active"},
            "GateAdmissionManifest",
        )
        if gate_admission["gate_key_id"].astype(str).duplicated().any():
            raise ValueError("GateAdmissionManifest gate keys must be unique")
        active_by_key = gate_admission.set_index("gate_key_id")["gate_key_active"].astype(bool)
        missing = sorted(
            set(events["gate_key_id"].dropna().astype(str)) - set(active_by_key.index.astype(str))
        )
        if missing:
            raise ValueError(f"events reference missing gate admission keys: {missing[:5]}")
        events["gate_key_active"] = events["gate_key_id"].map(active_by_key).fillna(False)
    events["model_active"] = (
        events["source_valid"].astype(bool)
        & events["has_retained_route"].astype(bool)
        & events["gate_key_active"].astype(bool)
    )
    events["admission_reasons"] = [
        _event_admission_reasons(row)
        for row in events.itertuples(index=False)
    ]
    for row in events.to_dict("records"):
        _validate_event_identity_row(row)

    valid_decisions = decisions.loc[decisions["event_id"].astype(str).isin(valid_event_ids)]
    audit_catalog = build_route_degree_cap_audit(
        events, valid_decisions, retained, audit_population="catalog"
    )
    active_ids = set(events.loc[events["model_active"], "event_id"].astype(str))
    audit_model = build_route_degree_cap_audit(
        events,
        valid_decisions.loc[valid_decisions["event_id"].astype(str).isin(active_ids)],
        retained.loc[retained["event_id"].astype(str).isin(active_ids)],
        audit_population="model_input",
    )
    route_audit = pd.concat([audit_catalog, audit_model], ignore_index=True)
    catalog_burden = build_route_burden(
        events.loc[events["source_valid"].astype(bool)],
        retained,
        valid_decisions,
        cap_audit,
        audit_population="catalog"
    )
    model_burden = build_route_burden(
        events.loc[events["model_active"]],
        retained.loc[retained["event_id"].astype(str).isin(active_ids)],
        valid_decisions.loc[valid_decisions["event_id"].astype(str).isin(active_ids)],
        cap_audit,
        audit_population="model_input",
    )
    burden = pd.concat([catalog_burden, model_burden], ignore_index=True)
    return PhysicalEventCatalog(
        physical_events=events,
        candidate_routes=decisions,
        event_routes=retained,
        cap_audit=cap_audit,
        route_degree_cap_audit=route_audit,
        catalog_burden=burden,
    )


def build_route_degree_cap_audit(
    physical_events: pd.DataFrame,
    candidate_routes: pd.DataFrame,
    event_routes: pd.DataFrame,
    *,
    audit_population: str,
) -> pd.DataFrame:
    """Reproduce pre/post route multiplicity and cap-renormalization equations."""

    if audit_population not in {"catalog", "model_input"}:
        raise ValueError("audit_population must be catalog or model_input")
    events = physical_events.set_index("event_id", drop=False)
    pre_by_event = {
        str(key): frame for key, frame in candidate_routes.groupby("event_id", sort=False)
    }
    post_by_event = {
        str(key): frame for key, frame in event_routes.groupby("event_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for event_id in sorted(pre_by_event):
        pre = pre_by_event[event_id]
        post = post_by_event.get(event_id, pre.iloc[:0])
        d_pre, d_post = len(pre), len(post)
        if d_pre <= 0:
            continue
        anchors = sorted(set(pre["anchor_region_id"].astype(str)))
        pre_edge_counts = pre.groupby("edge_id", sort=True).size().astype(int).to_dict()
        post_edge_counts = post.groupby("edge_id", sort=True).size().astype(int).to_dict()
        dropped = 1.0 - d_post / d_pre
        renorm = np.nan if d_post == 0 else d_pre / d_post
        pre_anchor_count = pre["anchor_region_id"].astype(str).nunique()
        post_anchor_count = post["anchor_region_id"].astype(str).nunique()
        drop_rows = pre
        if "cap_selected" in drop_rows:
            drop_rows = drop_rows.loc[~drop_rows["cap_selected"].astype(bool)]
        drop_buckets = (
            sorted(set(drop_rows["cap_bucket_id"].astype(str)))
            if "cap_bucket_id" in drop_rows
            else []
        )
        for anchor_id in anchors:
            pre_anchor = pre.loc[pre["anchor_region_id"].astype(str) == anchor_id]
            post_anchor = post.loc[post["anchor_region_id"].astype(str) == anchor_id]
            n_pre, n_post = len(pre_anchor), len(post_anchor)
            m_pre = n_pre / d_pre
            m_rawret = n_post / d_pre
            m_post = np.nan if d_post == 0 else n_post / d_post
            cap_loss = m_pre - m_rawret
            renorm_gain = np.nan if d_post == 0 else m_post - m_rawret
            if d_post and not np.isclose(
                m_post - m_pre, renorm_gain - cap_loss, atol=1e-12, rtol=0
            ):
                raise RuntimeError("route cap anchor-mass decomposition failed")
            rows.append(
                {
                    "audit_population": audit_population,
                    "event_id": event_id,
                    "target_gene_id": str(events.loc[event_id, "target_gene_id"]),
                    "modality": str(events.loc[event_id, "modality"]),
                    "anchor_region_id": anchor_id,
                    "anchor_scope": "single_anchor" if pre_anchor_count == 1 else "multi_anchor",
                    "D_pre": int(d_pre),
                    "D_post": int(d_post),
                    "A_pre": int(pre_anchor_count),
                    "A_post": int(post_anchor_count),
                    "n_anchor_pre": int(n_pre),
                    "n_anchor_post": int(n_post),
                    "d_anchor_pre": int(pre_anchor["edge_id"].astype(str).nunique()),
                    "d_anchor_post": int(post_anchor["edge_id"].astype(str).nunique()),
                    "m_pre": float(m_pre),
                    "m_rawret": float(m_rawret),
                    "m_post": float(m_post),
                    "cap_loss": float(cap_loss),
                    "renorm_gain": float(renorm_gain),
                    "per_edge_route_counts_pre": pre_edge_counts,
                    "per_edge_route_counts_post": post_edge_counts,
                    "dropped_route_mass": float(dropped),
                    "renormalization_factor": float(renorm),
                    "dropped_cap_bucket_ids": drop_buckets,
                    "drop_reasons": ([] if not drop_buckets else ["anchor_bucket_cap"]),
                    "external_only_coupling": bool(
                        d_post > 0 and n_pre == n_post and d_post < d_pre and m_post > m_pre
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_route_burden(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    candidate_routes: pd.DataFrame,
    cap_audit: pd.DataFrame,
    *,
    audit_population: str,
) -> pd.DataFrame:
    """Summarize catalog or model-input burden without changing route weights."""

    if event_routes.empty:
        return pd.DataFrame()
    event_meta = physical_events.set_index("event_id")
    joined = event_routes.copy()
    joined["gate_key_id"] = joined["event_id"].map(event_meta["gate_key_id"])
    saturated = set()
    if not cap_audit.empty:
        saturated = set(
            cap_audit.loc[cap_audit["cap_saturated"], "cap_bucket_id"].astype(str)
        )
    bucket_by_route: dict[str, str] = {}
    anchor_by_route: dict[str, str] = {}
    if not candidate_routes.empty:
        _require_columns(
            candidate_routes,
            {"route_id", "cap_bucket_id"},
            "candidate routes for burden audit",
        )
        retained_candidates = candidate_routes.loc[
            candidate_routes["route_id"].astype(str).isin(event_routes["route_id"].astype(str))
        ]
        bucket_by_route = {
            str(row.route_id): str(row.cap_bucket_id)
            for row in retained_candidates.itertuples(index=False)
        }
        anchor_by_route = {
            str(row.route_id): str(row.anchor_region_id)
            for row in retained_candidates.itertuples(index=False)
        }
    rows: list[dict[str, object]] = []
    for (gene_id, modality, edge_id), group in joined.groupby(
        ["target_gene_id", "modality", "edge_id"], sort=True
    ):
        by_gate = group.groupby("gate_key_id", dropna=False)["route_weight"].sum()
        b_gate = float(np.sqrt(np.square(by_gate.to_numpy(dtype=np.float64)).sum()))
        anchor_ids = set(group["anchor_region_id"].astype(str))
        rows.append(
            {
                "audit_population": audit_population,
                "target_gene_id": str(gene_id),
                "modality": str(modality),
                "edge_token_id": str(edge_id),
                "distinct_physical_event_count": int(group["event_id"].nunique()),
                "distinct_active_gate_key_count": (
                    int(group["gate_key_id"].dropna().astype(str).nunique())
                    if audit_population == "model_input"
                    else np.nan
                ),
                "distinct_anchor_group_count": len(anchor_ids),
                "saturated_anchor_group_count": len(
                    {
                        anchor_by_route[route_id]
                        for route_id in group["route_id"].astype(str)
                        if bucket_by_route.get(route_id) in saturated
                    }
                ),
                "saturated_cap_bucket_count": len(
                    {
                        bucket_by_route[route_id]
                        for route_id in group["route_id"].astype(str)
                        if bucket_by_route.get(route_id) in saturated
                    }
                ),
                "route_l1_mass": float(group["route_weight"].abs().sum()),
                "B_gate": b_gate if audit_population == "model_input" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_physical_event_catalog(
    physical_events: pd.DataFrame, event_routes: pd.DataFrame
) -> None:
    """Validate identity, activation, and production-routing invariants."""

    _require_columns(physical_events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    _require_columns(event_routes, set(EVENT_ROUTE_COLUMNS), "EventRouteTable")
    if physical_events["event_id"].astype(str).duplicated().any():
        raise ValueError("PhysicalEventTable primary key is not unique")
    route_key = ["event_id", "anchor_region_id", "anchor_site_id", "edge_id"]
    if event_routes.duplicated(route_key).any():
        raise ValueError("EventRouteTable primary key is not unique")
    known = set(physical_events["event_id"].astype(str))
    if not set(event_routes["event_id"].astype(str)).issubset(known):
        raise ValueError("EventRouteTable references an unknown physical event")
    if len(event_routes):
        weights = event_routes["route_weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or bool((weights <= 0).any()):
            raise ValueError("production route weights must be finite and positive")
        sums = event_routes.groupby("event_id", sort=False)["route_weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-12, rtol=0):
            raise ValueError("production route weights must sum to one per event")
    routed = set(event_routes["event_id"].astype(str))
    if not np.array_equal(
        physical_events["has_retained_route"].astype(bool).to_numpy(),
        physical_events["event_id"].astype(str).isin(routed).to_numpy(),
    ):
        raise ValueError("has_retained_route differs from EventRouteTable")
    expected_active = (
        physical_events["source_valid"].astype(bool)
        & physical_events["has_retained_route"].astype(bool)
        & physical_events["gate_key_active"].astype(bool)
    )
    if not np.array_equal(expected_active, physical_events["model_active"].astype(bool)):
        raise ValueError("model_active differs from its required conjunction")
    for row in physical_events.to_dict("records"):
        _validate_event_identity_row(row)


def _cap_route_decisions(
    routes: pd.DataFrame, event_meta: pd.DataFrame, cap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = routes.copy()
    decisions["cap_evidence_class"] = decisions["event_id"].map(
        event_meta["cap_evidence_class"]
    )
    bucket_columns = [
        "target_gene_id",
        "modality",
        "cap_evidence_class",
        "anchor_region_id",
        "region_type",
        "anchor_type",
    ]
    decisions["cap_bucket_id"] = [
        "cap:" + hashlib.sha256(_canonical_json(dict(zip(bucket_columns, values))).encode()).hexdigest()[:20]
        for values in decisions[bucket_columns].itertuples(index=False, name=None)
    ]
    decisions["cap_rank"] = -1
    decisions["cap_selected"] = False
    decisions["cap_reason"] = "anchor_bucket_cap"
    audit_rows: list[dict[str, object]] = []
    for bucket_id, group in decisions.groupby("cap_bucket_id", sort=True):
        event_ids = sorted(set(group["event_id"].astype(str)))
        ranked_rows: list[dict[str, object]] = []
        for event_id in event_ids:
            event = event_meta.loc[event_id]
            event_routes = group.loc[group["event_id"].astype(str) == event_id]
            geometry = str(event_routes.iloc[0]["geometry_kind"])
            if geometry == "site_window":
                sides = event_routes["transcript_oriented_side"].astype(str)
                finite_distance = pd.to_numeric(
                    event_routes.loc[sides != "OVERLAP_ANCHOR", "signed_distance_bp"],
                    errors="coerce",
                ).abs()
                proximity = (
                    0.0
                    if bool((sides == "OVERLAP_ANCHOR").any())
                    else float(finite_distance.min())
                )
                if not np.isfinite(proximity):
                    raise ValueError("site-window cap route has no valid geometry proximity")
            else:
                proximity = float(
                    np.minimum(
                        event_routes["distance_to_5prime_boundary_bp"].to_numpy(float),
                        event_routes["distance_to_3prime_boundary_bp"].to_numpy(float),
                    ).min()
                )
            calibrated_quality = float(event.get("calibrated_motif_quality", np.nan))
            source_rank = float(event.get("source_local_rank", np.nan))
            evidence_class = str(event["cap_evidence_class"])
            if evidence_class == "accessibility_only":
                sort_key = (-float(event["peak_support"]), proximity, event_id)
                boundary_quality = float(event["peak_support"])
            elif np.isfinite(calibrated_quality):
                if str(event["modality"]) == "DNA":
                    sort_key = (
                        -calibrated_quality,
                        -float(event["peak_support"]),
                        proximity,
                        event_id,
                    )
                else:
                    # RNA has no ATAC support term in its cap ranking.
                    sort_key = (-calibrated_quality, proximity, event_id)
                boundary_quality = calibrated_quality
            else:
                if not np.isfinite(source_rank):
                    raise ValueError(
                        "uncalibrated motif cap requires a frozen source_local_rank"
                    )
                sort_key = (
                    source_rank,
                    int(event.get("source_priority", 0)),
                    proximity,
                    event_id,
                )
                boundary_quality = -source_rank
            ranked_rows.append(
                {"event_id": event_id, "sort_key": sort_key, "boundary_quality": boundary_quality}
            )
        ranked_rows.sort(key=lambda row: row["sort_key"])
        selected = {row["event_id"] for row in ranked_rows[:cap]}
        rank_by_event = {row["event_id"]: rank + 1 for rank, row in enumerate(ranked_rows)}
        index = group.index
        decisions.loc[index, "cap_rank"] = group["event_id"].astype(str).map(rank_by_event)
        decisions.loc[index, "cap_selected"] = group["event_id"].astype(str).isin(selected)
        decisions.loc[index, "cap_reason"] = np.where(
            decisions.loc[index, "cap_selected"], "retained", "anchor_bucket_cap"
        )
        bucket_values = group.iloc[0]
        collapse_count = int(
            sum(
                max(0, len(event_meta.loc[event_id, "source_hit_coordinates"]) - 1)
                for event_id in event_ids
            )
        )
        audit_rows.append(
            {
                "cap_bucket_id": bucket_id,
                **{column: bucket_values[column] for column in bucket_columns},
                "candidate_event_count": len(event_ids),
                "selected_event_count": len(selected),
                "cap_saturated": len(event_ids) > cap,
                "motif_equivalence_family_collapse_count": collapse_count,
                "boundary_quality": (
                    ranked_rows[min(cap, len(ranked_rows)) - 1]["boundary_quality"]
                    if ranked_rows
                    else np.nan
                ),
            }
        )
    decisions["cap_rank"] = decisions["cap_rank"].astype(np.int64)
    return decisions, pd.DataFrame(audit_rows)


def _event_kind_by_event(routes: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for event_id, group in routes.groupby("event_id", sort=False):
        kinds: set[str] = set()
        for row in group.itertuples(index=False):
            if row.geometry_kind == "edge_contained":
                kinds.add(
                    "EXON_CONTAINED" if row.region_type == "exon" else "DNA_INTRAGENIC"
                )
            elif row.anchor_type == "TSS":
                kinds.add("TSS_PROXIMAL")
            elif row.anchor_type in {"donor", "acceptor"}:
                kinds.add("SPLICE_SITE_PROXIMAL")
            elif row.anchor_type == "PAS":
                kinds.add("PAS_PROXIMAL")
            else:
                raise ValueError(f"route has unknown reporting anchor type {row.anchor_type}")
        result[str(event_id)] = next(iter(kinds)) if len(kinds) == 1 else "MULTI_ANCHOR"
    return result


def _event_admission_reasons(row: object) -> list[str]:
    reasons: list[str] = []
    if not bool(row.source_valid):
        reasons.append("invalid_source")
    if not bool(row.has_retained_route):
        reasons.append("no_retained_route")
    if not bool(row.gate_key_active):
        reasons.append("inactive_gate_key")
    return reasons


def _validate_component_identity(component: pd.DataFrame) -> None:
    columns = (
        "target_gene_id",
        "modality",
        "factor_entity_id",
        "factor_identity_kind",
        "candidate_factor_ids",
        "activity_entity_id",
        "activity_gene_ids",
        "activity_proxy_rule",
        "motif_equivalence_family_id",
        "chromosome",
        "strand",
        "peak_id",
    )
    for column in columns:
        if len({_canonical_json(value) for value in component[column]}) != 1:
            raise ValueError(f"physical collapse component mixes {column}")


def _representative_motif_hit(component: pd.DataFrame) -> pd.Series:
    ranked = component.copy()
    calibrated = pd.to_numeric(
        ranked.get("calibrated_motif_quality", np.nan), errors="coerce"
    )
    source_rank = pd.to_numeric(ranked.get("source_local_rank", np.nan), errors="coerce")
    if np.isfinite(np.asarray(calibrated, dtype=float)).any():
        ranked["_quality_missing"] = ~np.isfinite(np.asarray(calibrated, dtype=float))
        ranked["_quality"] = -pd.Series(calibrated, index=ranked.index).fillna(-np.inf)
    elif np.isfinite(np.asarray(source_rank, dtype=float)).any():
        ranked["_quality_missing"] = source_rank.isna()
        ranked["_quality"] = source_rank.fillna(np.inf)
    else:
        raise ValueError("motif representative selection needs score or source-local rank")
    ranked["source_priority"] = pd.to_numeric(
        ranked.get("source_priority", 0), errors="raise"
    )
    return ranked.sort_values(
        [
            "_quality_missing",
            "_quality",
            "source_priority",
            "start",
            "end",
            "motif_id",
            "source_hit_id",
        ],
        kind="mergesort",
    ).iloc[0]


def _validate_event_identity_row(row: Mapping[str, object]) -> None:
    kind = str(row["factor_identity_kind"])
    if kind not in FACTOR_IDENTITY_KINDS:
        raise ValueError(f"unknown factor_identity_kind: {kind}")
    factors = _string_list(row["candidate_factor_ids"], "candidate_factor_ids")
    genes = _string_list(row["activity_gene_ids"], "activity_gene_ids")
    factor_entity = _nullable_text(row["factor_entity_id"])
    activity_entity = _nullable_text(row["activity_entity_id"])
    peak_support = float(row["peak_support"])
    if not np.isfinite(peak_support) or peak_support < 0:
        raise ValueError("physical event peak support must be finite and non-negative")
    if kind == "accessibility_only":
        null_fields = (
            factor_entity,
            activity_entity,
            _nullable_text(row["motif_id"]),
            _nullable_text(row["motif_equivalence_family_id"]),
            _nullable_text(row["orientation"]),
        )
        if any(value is not None for value in null_fields) or factors or genes:
            raise ValueError("accessibility-only event has factor/motif identity")
        if row["cap_evidence_class"] != "accessibility_only" or row["modality"] != "DNA":
            raise ValueError("accessibility-only events require DNA accessibility cap class")
    else:
        if factor_entity is None or factor_entity != activity_entity:
            raise ValueError("motif event factor and activity entity must be identical")
        if not genes:
            raise ValueError("motif/group events require non-empty activity_gene_ids")
        if kind == "unique" and factors != [factor_entity]:
            raise ValueError("unique event candidate factors differ from factor entity")
        if kind == "factor_equivalence_group" and len(factors) < 2:
            raise ValueError("factor-equivalence event has fewer than two candidates")
        motif_id = _required_text(row["motif_id"], "motif_id")
        source_motifs = _string_list(row["source_motif_ids"], "source_motif_ids")
        if motif_id not in source_motifs:
            raise ValueError("representative motif is absent from source_motif_ids")
        if row["cap_evidence_class"] != "motif_anchored":
            raise ValueError("motif event has the wrong cap evidence class")
    event_kind = _nullable_text(row["event_kind"])
    if event_kind is not None and event_kind not in EVENT_KINDS:
        raise ValueError(f"unknown event_kind: {event_kind}")


def _site_region_type(node_type: str) -> str:
    if node_type == "TSS":
        return "promoter"
    if node_type in {"donor", "acceptor"}:
        return "splice_site"
    if node_type == "PAS":
        return "pas"
    raise ValueError(f"unknown processing-site node type: {node_type}")


def _scan_encoded(
    encoded: np.ndarray,
    probabilities: np.ndarray,
    minimum_relative_score: float,
    orientation: str,
) -> list[tuple[int, str, float]]:
    width = probabilities.shape[0]
    if len(encoded) < width:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(encoded, width)
    valid = (windows >= 0).all(axis=1)
    clipped = windows.clip(min=0)
    log_odds = np.log((probabilities + 1e-6) / 0.25)
    scores = log_odds[np.arange(width)[None, :], clipped].sum(axis=1)
    minimum = float(log_odds.min(axis=1).sum())
    maximum = float(log_odds.max(axis=1).sum())
    if maximum <= minimum:
        raise ValueError("PWM has no sequence discrimination")
    relative = (scores - minimum) / (maximum - minimum)
    selected = np.flatnonzero(valid & (relative >= minimum_relative_score))
    return [(int(index), orientation, float(relative[index])) for index in selected]


def _validate_pwm(motif_id: str, probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or len(probabilities) == 0:
        raise ValueError(f"motif {motif_id} must have shape [width,4]")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError(f"motif {motif_id} has invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"motif {motif_id} rows do not sum to one")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} misses required columns: {missing}")


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


def _canonical_json(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        value = None
    if isinstance(value, (list, tuple, set, np.ndarray)):
        value = [_jsonable(item) for item in value]
    elif isinstance(value, dict):
        value = {str(key): _jsonable(item) for key, item in value.items()}
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is pd.NA:
        return None
    return value


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TypeError(f"{label} must be a sequence, not a scalar string") from exc
        value = parsed
    if value is None or value is pd.NA:
        return []
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        raise TypeError(f"{label} must be a sequence")
    result = [str(item) for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} entries must be unique and non-empty")
    return result


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def _nullable_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return None if text in {"", "NA", "nan", "None"} else text


def _required_text(value: object, label: str) -> str:
    result = _nullable_text(value)
    if result is None:
        raise ValueError(f"{label} must be explicit")
    return result
