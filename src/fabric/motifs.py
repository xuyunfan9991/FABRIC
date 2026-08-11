"""Unified factor identities and fixed DNA/RNA motif-event candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .annotation import ReferenceSequence
from .choices import ChoiceCatalog
from .graph import GeneGraph


DNA_ORIENTATIONS = ("same_transcript", "opposite_transcript")
RNA_ORIENTATIONS = ("transcribed",)
CHOICE_SCOPES = ("internal", "tss", "pas", "full_length")


@dataclass(frozen=True)
class PWM:
    motif_id: str
    name: str
    probabilities: np.ndarray

    @property
    def width(self) -> int:
        return int(self.probabilities.shape[0])


@dataclass(frozen=True)
class FactorCatalogResult:
    factors: pd.DataFrame
    motif_mapping: pd.DataFrame
    excluded_motifs: pd.DataFrame


def parse_meme_motifs(path: str | Path) -> dict[str, PWM]:
    """Parse the frozen JASPAR MEME probability matrices."""

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
        matrix = []
        for row in lines[index + 1 : index + 1 + width]:
            values = [float(value) for value in row.split()]
            if len(values) != 4:
                raise ValueError(f"MEME motif {motif_id} is not a four-letter PWM")
            matrix.append(values)
        probabilities = np.asarray(matrix, dtype=np.float64)
        _validate_pwm(motif_id, probabilities)
        if motif_id in motifs:
            raise ValueError(f"duplicate MEME motif ID: {motif_id}")
        motifs[motif_id] = PWM(
            motif_id=motif_id, name=name, probabilities=probabilities
        )
        index += width + 1
    if not motifs:
        raise ValueError("MEME library contains no motifs")
    return motifs


def parse_cisbp_motifs(
    directory: str | Path, *, motif_ids: Sequence[str] | None = None
) -> dict[str, PWM]:
    """Parse CisBP-RNA A/C/G/U tables, converting U to the internal T column."""

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
        motifs[motif_id] = PWM(
            motif_id=motif_id,
            name=motif_id,
            probabilities=probabilities,
        )
    if selected is not None:
        missing = sorted(selected - set(motifs))
        if missing:
            raise FileNotFoundError(f"CisBP PWM files are absent: {missing[:10]}")
    return motifs


def build_factor_catalog(
    jaspar_index_path: str | Path,
    cisbp_gene_map_path: str | Path,
    *,
    gene_symbol_to_id: Mapping[str, str],
    explicit_mapping: pd.DataFrame | None = None,
) -> FactorCatalogResult:
    """Unify DNA and RNA motifs only where activity identity is explicit.

    Ambiguous family/composite motifs are excluded unless ``explicit_mapping``
    supplies one factor/group identity and one activity gene.  No motif is
    copied across multiple family members.
    """

    jaspar = pd.read_csv(jaspar_index_path, sep="\t", dtype=str)
    cisbp = pd.read_csv(cisbp_gene_map_path, sep="\t", dtype=str)
    required_jaspar = {"motif_id", "tf_name"}
    required_cisbp = {"motif_id", "rbp_gene", "gene_id"}
    if required_jaspar - set(jaspar) or required_cisbp - set(cisbp):
        raise ValueError("motif indices miss required factor identity columns")
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    if explicit_mapping is not None:
        required = {
            "modality",
            "motif_id",
            "factor_id",
            "factor_name",
            "activity_gene_id",
            "factor_group_id",
        }
        if required - set(explicit_mapping):
            raise ValueError("explicit factor mapping misses required columns")
        for row in explicit_mapping.itertuples(index=False):
            key = (str(row.modality).upper(), str(row.motif_id))
            if key in overrides:
                raise ValueError(f"duplicate explicit motif mapping: {key}")
            overrides[key] = {
                "factor_id": str(row.factor_id),
                "factor_name": str(row.factor_name),
                "activity_gene_id": str(row.activity_gene_id).split(".", 1)[0],
                "factor_group_id": str(row.factor_group_id),
            }

    mapped: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    symbol_map = {
        str(key).upper(): str(value).split(".", 1)[0]
        for key, value in gene_symbol_to_id.items()
    }
    for row in jaspar.itertuples(index=False):
        motif_id, name = str(row.motif_id), str(row.tf_name)
        override = overrides.get(("DNA", motif_id))
        components = [value.strip().upper() for value in name.split("::")]
        if override is not None:
            identity = override
        elif len(components) == 1 and components[0] in symbol_map:
            gene_id = symbol_map[components[0]]
            identity = {
                "factor_id": gene_id,
                "factor_name": name,
                "activity_gene_id": gene_id,
                "factor_group_id": gene_id,
            }
        else:
            excluded.append(
                {
                    "modality": "DNA",
                    "motif_id": motif_id,
                    "name": name,
                    "reason": "ambiguous_or_unmapped_factor",
                }
            )
            continue
        mapped.append({"modality": "DNA", "motif_id": motif_id, **identity})

    for motif_id, group in cisbp.groupby("motif_id", sort=True):
        override = overrides.get(("RNA", str(motif_id)))
        gene_ids = sorted({str(value).split(".", 1)[0] for value in group["gene_id"]})
        names = sorted(set(group["rbp_gene"].astype(str)))
        if override is not None:
            identity = override
        elif len(gene_ids) == 1:
            identity = {
                "factor_id": gene_ids[0],
                "factor_name": names[0],
                "activity_gene_id": gene_ids[0],
                "factor_group_id": gene_ids[0],
            }
        else:
            excluded.append(
                {
                    "modality": "RNA",
                    "motif_id": str(motif_id),
                    "name": "::".join(names),
                    "reason": "ambiguous_factor_group",
                }
            )
            continue
        mapped.append({"modality": "RNA", "motif_id": str(motif_id), **identity})
    mapping = pd.DataFrame(mapped).sort_values(
        ["factor_id", "modality", "motif_id"], kind="mergesort"
    )
    if mapping.empty:
        raise ValueError("factor catalog maps zero motifs to activity genes")
    conflict = mapping.groupby(["modality", "motif_id"])["factor_id"].nunique()
    if bool((conflict > 1).any()):
        raise ValueError("one motif maps to multiple factor identities")
    factor_rows: list[dict[str, object]] = []
    for factor_id, group in mapping.groupby("factor_id", sort=True):
        activity_ids = set(group["activity_gene_id"].astype(str))
        group_ids = set(group["factor_group_id"].astype(str))
        if len(activity_ids) != 1 or len(group_ids) != 1:
            raise ValueError(
                f"factor {factor_id} has inconsistent activity/group identity"
            )
        dna = sorted(group.loc[group["modality"] == "DNA", "motif_id"].astype(str))
        rna = sorted(group.loc[group["modality"] == "RNA", "motif_id"].astype(str))
        names = sorted(set(group["factor_name"].astype(str)))
        factor_rows.append(
            {
                "factor_id": str(factor_id),
                "factor_name": names[0],
                "activity_gene_id": next(iter(activity_ids)),
                "factor_group_id": next(iter(group_ids)),
                "has_dna_motif": bool(dna),
                "has_rna_motif": bool(rna),
                "canonical_label": names[0],
                "dna_motif_ids": dna,
                "rna_motif_ids": rna,
            }
        )
    return FactorCatalogResult(
        factors=pd.DataFrame(factor_rows),
        motif_mapping=mapping.reset_index(drop=True),
        excluded_motifs=pd.DataFrame(excluded),
    )


def scan_pwm(
    sequence: str,
    motif: PWM,
    *,
    minimum_relative_score: float,
    reverse_strand: bool,
) -> list[tuple[int, str, float]]:
    """Return fixed motif hits as (offset, orientation, relative score)."""

    if not 0 <= minimum_relative_score <= 1:
        raise ValueError("minimum_relative_score must lie in [0, 1]")
    sequence = sequence.upper().replace("U", "T")
    encoded = np.fromiter(("ACGT".find(base) for base in sequence), dtype=np.int8)
    hits = _scan_encoded(encoded, motif.probabilities, minimum_relative_score, "+")
    if reverse_strand:
        reverse_pwm = motif.probabilities[::-1, ::-1]
        hits.extend(_scan_encoded(encoded, reverse_pwm, minimum_relative_score, "-"))
    return sorted(hits, key=lambda value: (value[0], value[1], -value[2]))


def scan_motif_regions(
    regions: pd.DataFrame,
    motifs: Mapping[str, PWM],
    motif_mapping: pd.DataFrame,
    *,
    modality: str,
    minimum_relative_score: float,
) -> pd.DataFrame:
    """Scan fixed sequence regions; this function has no cell or label input."""

    modality = modality.upper()
    if modality not in {"DNA", "RNA"}:
        raise ValueError("modality must be DNA or RNA")
    required = {
        "gene_id",
        "choice_id",
        "alternative_id",
        "chrom",
        "start_0based",
        "end_0based",
        "strand",
        "anchor_0based",
        "region_type",
        "sequence",
    }
    if required - set(regions):
        raise ValueError("motif regions miss required geometry columns")
    mapping = motif_mapping.loc[motif_mapping["modality"].astype(str) == modality]
    mapping_by_motif = {
        str(row.motif_id): row for row in mapping.itertuples(index=False)
    }
    raw_events: list[dict[str, object]] = []
    for region in regions.itertuples(index=False):
        strand = str(region.strand)
        for motif_id, identity in mapping_by_motif.items():
            motif = motifs.get(motif_id)
            if motif is None:
                raise ValueError(f"factor mapping references an absent PWM: {motif_id}")
            for offset, hit_strand, score in scan_pwm(
                str(region.sequence),
                motif,
                minimum_relative_score=minimum_relative_score,
                reverse_strand=modality == "DNA",
            ):
                # DNA regions always contain forward genomic sequence, so the
                # scan offset is a direct genomic offset regardless of the gene
                # strand.  RNA regions contain transcript-oriented sequence and
                # therefore need reverse coordinate projection on negative genes.
                if modality == "DNA" or strand == "+":
                    start = int(region.start_0based) + offset
                else:
                    start = int(region.end_0based) - offset - motif.width
                end = start + motif.width
                center = (start + end) / 2.0
                signed_distance = (center - float(region.anchor_0based)) * (
                    1 if strand == "+" else -1
                )
                if modality == "RNA":
                    orientation = "transcribed"
                else:
                    genomic_same = hit_strand == "+"
                    orientation = (
                        "same_transcript"
                        if genomic_same == (strand == "+")
                        else "opposite_transcript"
                    )
                raw_events.append(
                    {
                        "modality": modality,
                        "gene_id": str(region.gene_id),
                        "choice_id": str(region.choice_id),
                        "alternative_id": str(region.alternative_id),
                        "factor_id": str(identity.factor_id),
                        "factor_group_id": str(identity.factor_group_id),
                        "motif_id": motif_id,
                        "chrom": str(region.chrom),
                        "start_0based": start,
                        "end_0based": end,
                        "orientation": orientation,
                        "anchor_0based": int(region.anchor_0based),
                        "signed_distance_bp": float(signed_distance),
                        "region_type": str(region.region_type),
                        "peak_id": str(getattr(region, "peak_id", "")),
                        "peak_support": float(getattr(region, "peak_support", 0.0)),
                        "motif_score": float(score),
                    }
                )
    if not raw_events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "modality",
                "gene_id",
                "choice_id",
                "relation_alternative_ids",
                "factor_id",
                "factor_group_id",
                "motif_id",
                "chrom",
                "start_0based",
                "end_0based",
                "orientation",
                "anchor_0based",
                "signed_distance_bp",
                "region_type",
                "peak_id",
                "peak_support",
                "motif_score",
            ]
        )
    frame = pd.DataFrame(raw_events)
    identity_columns = [
        "modality",
        "gene_id",
        "choice_id",
        "factor_id",
        "factor_group_id",
        "motif_id",
        "chrom",
        "start_0based",
        "end_0based",
        "orientation",
        "anchor_0based",
        "signed_distance_bp",
        "region_type",
        "peak_id",
        "peak_support",
        "motif_score",
    ]
    rows = []
    for _, group in frame.groupby(identity_columns, sort=True, dropna=False):
        row = group.iloc[0][identity_columns].to_dict()
        relations = sorted(set(group["alternative_id"].astype(str)))
        row["relation_alternative_ids"] = relations
        row["event_id"] = "event|" + "|".join(
            f"{column}={_event_identity_value(row[column])}"
            for column in identity_columns
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("event_id", kind="mergesort")
        .reset_index(drop=True)
    )


def _event_identity_value(value: object) -> str:
    """Render one MotifEvent identity field as stable, readable text."""

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric == 0:
            return "0"
        return format(numeric, ".17g")
    return str(value)


def cap_motif_events(
    events: pd.DataFrame, *, events_per_choice_cap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the one static, cell-independent candidate cap and report saturation."""

    if events_per_choice_cap <= 0:
        raise ValueError("events_per_choice_cap must be positive")
    selected_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for (modality, choice_id), group in events.groupby(
        ["modality", "choice_id"], sort=True
    ):
        ranked = group.assign(
            absolute_signed_distance=group["signed_distance_bp"].abs()
        ).sort_values(
            [
                "motif_score",
                "absolute_signed_distance",
                "peak_support",
                "region_type",
                "event_id",
            ],
            ascending=[False, True, False, True, True],
            kind="mergesort",
        )
        selected = ranked.head(events_per_choice_cap).drop(
            columns="absolute_signed_distance"
        )
        selected_parts.append(selected)
        audit_rows.append(
            {
                "modality": modality,
                "choice_id": choice_id,
                "candidate_event_count": int(len(group)),
                "selected_event_count": int(len(selected)),
                "cap_saturated": bool(len(group) > events_per_choice_cap),
                "boundary_rank_motif_score": (
                    float(selected.iloc[-1]["motif_score"]) if len(selected) else np.nan
                ),
            }
        )
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else events.copy()
    )
    return selected, pd.DataFrame(audit_rows)


def build_rna_choice_regions(
    graph: GeneGraph,
    catalog: ChoiceCatalog,
    reference: ReferenceSequence,
    *,
    window_bp: int,
) -> pd.DataFrame:
    """Build transcript-oriented fixed windows around alternative processing sites."""

    if window_bp <= 0:
        raise ValueError("RNA motif window must be positive")
    edge_rows = graph.edges.reset_index(drop=True)
    node_positions = graph.nodes.set_index(graph.nodes["node_id"].astype(str))[
        "pos_0based"
    ].to_dict()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for choice in catalog.choices:
        anchors = [
            int(node_positions[choice.entry_node_id]),
            int(node_positions[choice.exit_node_id]),
        ]
        for alternative in choice.alternatives:
            for edge_index in alternative.edge_indices:
                edge = edge_rows.iloc[edge_index]
                if str(edge.edge_type) in {"START", "END"}:
                    continue
                edge_start = int(edge.start_0based)
                edge_end = int(edge.end_0based_exclusive)
                if edge_end <= edge_start:
                    continue
                endpoint_windows = {
                    (edge_start, min(edge_end, edge_start + window_bp)),
                    (max(edge_start, edge_end - window_bp), edge_end),
                }
                region_type = (
                    "exon" if str(edge.edge_type) == "EXON_CONTINUATION" else "intron"
                )
                for start, end in sorted(endpoint_windows):
                    key = (
                        choice.choice_id,
                        alternative.alternative_id,
                        start,
                        end,
                        region_type,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    midpoint = (start + end) / 2.0
                    anchor = min(
                        anchors, key=lambda value: (abs(value - midpoint), value)
                    )
                    rows.append(
                        {
                            "gene_id": graph.gene_id,
                            "choice_id": choice.choice_id,
                            "alternative_id": alternative.alternative_id,
                            "chrom": str(edge.chrom),
                            "start_0based": start,
                            "end_0based": end,
                            "strand": str(edge.strand),
                            "anchor_0based": anchor,
                            "region_type": region_type,
                            "sequence": reference.fetch(
                                str(edge.chrom), start, end, str(edge.strand)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_dna_peak_regions(
    graph: GeneGraph,
    catalog: ChoiceCatalog,
    peaks: pd.DataFrame,
    reference: ReferenceSequence,
    *,
    window_bp: int,
) -> pd.DataFrame:
    """Relate full-RNA consensus peaks to fixed alternative processing-site windows."""

    required = {"chrom", "start_0based", "end_0based", "peak_id", "peak_support"}
    if required - set(peaks):
        raise ValueError("peak table misses DNA candidate columns")
    if window_bp <= 0:
        raise ValueError("DNA motif window must be positive")
    if peaks["peak_id"].astype(str).duplicated().any():
        raise ValueError("DNA peak IDs must be unique")
    peak_index: dict[str, tuple[pd.DataFrame, np.ndarray, int]] = {}
    for chrom, frame in peaks.groupby(peaks["chrom"].astype(str), sort=False):
        ordered = frame.sort_values("start_0based", kind="mergesort").reset_index(
            drop=True
        )
        starts = ordered["start_0based"].to_numpy(dtype=np.int64)
        ends = ordered["end_0based"].to_numpy(dtype=np.int64)
        if bool((starts < 0).any()) or bool((ends <= starts).any()):
            raise ValueError(f"peak table contains an invalid interval on {chrom}")
        peak_index[str(chrom)] = (ordered, starts, int((ends - starts).max()))
    nodes = graph.nodes.set_index(graph.nodes["node_id"].astype(str))
    rows: list[dict[str, object]] = []
    for choice in catalog.choices:
        for alternative in choice.alternatives:
            # Entry/exit geometry belongs to every alternative.  Shared hits
            # are merged later into a multi-alternative relation; omitting the
            # boundaries here would incorrectly assign them only to a direct
            # (no-internal-node) alternative.
            sites = (
                choice.entry_node_id,
                *alternative.node_ids[1:-1],
                choice.exit_node_id,
            )
            for node_id in sites:
                node = nodes.loc[node_id]
                position = int(node.pos_0based)
                chrom = str(node.chrom)
                if chrom not in peak_index:
                    continue
                chrom_peaks, starts, maximum_width = peak_index[chrom]
                left = int(
                    np.searchsorted(
                        starts, position - window_bp - maximum_width, side="left"
                    )
                )
                right = int(np.searchsorted(starts, position + window_bp, side="left"))
                candidates = chrom_peaks.iloc[left:right]
                candidates = candidates.loc[
                    candidates["end_0based"].to_numpy(dtype=np.int64)
                    > position - window_bp
                ]
                for peak in candidates.itertuples(index=False):
                    rows.append(
                        {
                            "gene_id": graph.gene_id,
                            "choice_id": choice.choice_id,
                            "alternative_id": alternative.alternative_id,
                            "chrom": str(peak.chrom),
                            "start_0based": int(peak.start_0based),
                            "end_0based": int(peak.end_0based),
                            "strand": str(node.strand),
                            "anchor_0based": position,
                            "region_type": "peak",
                            "peak_id": str(peak.peak_id),
                            "peak_support": float(peak.peak_support),
                            # DNA scanning uses the forward genomic sequence; hit
                            # orientation is converted relative to transcription.
                            "sequence": reference.fetch(
                                str(peak.chrom),
                                int(peak.start_0based),
                                int(peak.end_0based),
                                "+",
                            ),
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                *required,
                "gene_id",
                "choice_id",
                "alternative_id",
                "strand",
                "anchor_0based",
                "region_type",
                "sequence",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(
        ["choice_id", "alternative_id", "peak_id", "anchor_0based"]
    )


def fixed_event_feature_matrix(
    events: pd.DataFrame,
    *,
    modality: str,
    factor_order: Sequence[str],
    region_order: Sequence[str],
    distance_scale_bp: float,
) -> np.ndarray:
    """Create the explicit non-learned event vector used by one scorer."""

    modality = modality.upper()
    orientations = DNA_ORIENTATIONS if modality == "DNA" else RNA_ORIENTATIONS
    if distance_scale_bp <= 0:
        raise ValueError("event distance scale must be positive")
    factors = tuple(str(value) for value in factor_order)
    regions = tuple(str(value) for value in region_order)
    feature_dim = (
        len(factors)
        + 1  # motif score
        + len(orientations)
        + 1  # signed distance
        + len(regions)
        + int(modality == "DNA")  # peak support
    )
    rows: list[list[float]] = []
    for event in events.itertuples(index=False):
        factor_id = str(event.factor_id)
        orientation = str(event.orientation)
        region_type = str(event.region_type)
        if (
            factor_id not in factors
            or orientation not in orientations
            or region_type not in regions
        ):
            raise ValueError(
                "event categorical identity is absent from the fixed feature order"
            )
        feature = [float(factor_id == value) for value in factors]
        feature.append(float(event.motif_score))
        feature.extend(float(orientation == value) for value in orientations)
        feature.append(float(event.signed_distance_bp) / distance_scale_bp)
        feature.extend(float(region_type == value) for value in regions)
        if modality == "DNA":
            feature.append(float(np.log1p(float(event.peak_support))))
        rows.append(feature)
    result = np.asarray(rows, dtype=np.float32).reshape(len(rows), feature_dim)
    if result.size and not np.isfinite(result).all():
        raise ValueError("fixed event feature matrix contains non-finite values")
    return result


def event_relation_matrix(
    events: pd.DataFrame, alternative_order: Sequence[str]
) -> np.ndarray:
    alternative_index = {
        str(value): index for index, value in enumerate(alternative_order)
    }
    relation = np.zeros((len(events), len(alternative_order)), dtype=np.float32)
    for event_index, values in enumerate(events["relation_alternative_ids"]):
        for alternative_id in values:
            if str(alternative_id) not in alternative_index:
                raise ValueError(
                    f"event relation references an unknown alternative: {alternative_id}"
                )
            relation[event_index, alternative_index[str(alternative_id)]] = 1.0
    return relation


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
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 4
        or len(probabilities) == 0
    ):
        raise ValueError(f"motif {motif_id} must have shape [width,4]")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError(f"motif {motif_id} has invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"motif {motif_id} rows do not sum to one")
