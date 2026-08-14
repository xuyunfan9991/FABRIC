"""Explicit, split-neutral CIS features for FABRIC V2.

The architecture contract fixes the biological fields but does not prescribe
sequence-scanner algorithms.  This module therefore treats scanner outputs as
a strict, audited input table.  It derives structure and geometry directly
from the structural-edge catalog, validates sequence-score applicability, and
fits normalization only on unique train-admitted structural edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


EDGE_TYPES = ("EXON_CONTINUATION", "SPLICE", "RETAINED_INTRON")
SITE_TYPES = ("TSS", "donor", "acceptor", "PAS")

STRUCTURAL_EDGE_COLUMNS = (
    "edge_id",
    "target_gene_id",
    "edge_type",
    "src_node_type",
    "dst_node_type",
    "span_bp",
    "length_bp",
    "relative_edge_pos",
    "annotation_confidence",
    "edge_prior_score",
)

STRUCTURE_FEATURES = (
    *(f"edge_type__{value}" for value in EDGE_TYPES),
    *(f"src_site_type__{value}" for value in SITE_TYPES),
    *(f"dst_site_type__{value}" for value in SITE_TYPES),
)

GEOMETRY_FEATURES = (
    "log1p_span_bp",
    "log1p_length_bp",
    "relative_edge_pos",
    "annotation_confidence",
    "edge_prior_score",
)

SEQUENCE_FEATURES = (
    "edge_gc_fraction",
    "donor_strength",
    "acceptor_strength",
    "branchpoint_score",
    "polypyrimidine_tract_score",
    "tss_core_promoter_score",
    "polya_hexamer_score",
    "pas_downstream_u_gu_fraction",
)

SEQUENCE_MASKS = tuple(f"{feature}_available" for feature in SEQUENCE_FEATURES)
CONTINUOUS_FEATURES = (*GEOMETRY_FEATURES, *SEQUENCE_FEATURES)
RAW_FEATURE_ORDER = (*STRUCTURE_FEATURES, *CONTINUOUS_FEATURES, *SEQUENCE_MASKS)

GEOMETRY_TRANSFORMS = (
    ("log1p_span_bp", "log1p_bp"),
    ("log1p_length_bp", "log1p_bp"),
    ("relative_edge_pos", "identity"),
    ("annotation_confidence", "identity"),
    ("edge_prior_score", "identity"),
)


@dataclass(frozen=True)
class CISSequenceFeatureSpec:
    """Frozen provenance for one precomputed, final-form sequence score."""

    feature_name: str
    scanner_name: str
    scanner_version: str
    sequence_window: str
    fixed_transform: str

    def __post_init__(self) -> None:
        for field_name in (
            "feature_name",
            "scanner_name",
            "scanner_version",
            "sequence_window",
            "fixed_transform",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"CIS sequence specification has empty {field_name}")


@dataclass(frozen=True)
class CISNormalizationPolicy:
    """The only normalization policy admitted by the V2 explicit CIS block."""

    numerical_tolerance: float = 1.0e-8
    population: str = "train_admitted_unique_structural_edges"
    weighting: str = "each_edge_once_no_cell_molecule_path_transcript_weights"
    center: str = "availability_masked_mean"
    scale: str = "availability_masked_population_sd_ddof0"
    inapplicable_after_normalization: str = "numeric_zero"
    categorical_policy: str = "one_hot_and_availability_masks_not_standardized"
    constant_feature_policy: str = "remove_when_scale_lte_numerical_tolerance"

    def __post_init__(self) -> None:
        if not np.isfinite(self.numerical_tolerance) or self.numerical_tolerance <= 0:
            raise ValueError("CIS numerical_tolerance must be finite and positive")
        required = {
            "population": "train_admitted_unique_structural_edges",
            "weighting": "each_edge_once_no_cell_molecule_path_transcript_weights",
            "center": "availability_masked_mean",
            "scale": "availability_masked_population_sd_ddof0",
            "inapplicable_after_normalization": "numeric_zero",
            "categorical_policy": (
                "one_hot_and_availability_masks_not_standardized"
            ),
            "constant_feature_policy": (
                "remove_when_scale_lte_numerical_tolerance"
            ),
        }
        for field_name, expected in required.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"CIS normalization {field_name} must be {expected!r}"
                )


@dataclass(frozen=True)
class CISFeatureManifest:
    """Frozen schema and provenance for the explicit V2 CIS feature block."""

    reference_build: str
    strand_convention: str
    sequence_features: tuple[CISSequenceFeatureSpec, ...]
    normalization: CISNormalizationPolicy
    output_order: tuple[str, ...] = RAW_FEATURE_ORDER
    geometry_transforms: tuple[tuple[str, str], ...] = GEOMETRY_TRANSFORMS
    schema_version: str = "FABRIC_V2_EXPLICIT_CIS_V1"
    split_neutral: bool = True
    alphagenome_status: str = "DEFERRED_CIS_EXTENSION_DISABLED"

    def __post_init__(self) -> None:
        if self.schema_version != "FABRIC_V2_EXPLICIT_CIS_V1":
            raise ValueError("unsupported CISFeatureManifest schema_version")
        if not self.reference_build.strip() or not self.strand_convention.strip():
            raise ValueError("CIS reference build and strand convention are required")
        if not self.split_neutral:
            raise ValueError("explicit CIS scanners and fields must be split-neutral")
        if self.alphagenome_status != "DEFERRED_CIS_EXTENSION_DISABLED":
            raise ValueError("AlphaGenome is not enabled in the current V2 CIS block")
        names = tuple(spec.feature_name for spec in self.sequence_features)
        if names != SEQUENCE_FEATURES:
            raise ValueError(
                "CIS sequence feature specifications must exactly follow the "
                "frozen V2 sequence field order"
            )
        if self.geometry_transforms != GEOMETRY_TRANSFORMS:
            raise ValueError("CIS geometry transforms differ from the V2 contract")
        if self.output_order != RAW_FEATURE_ORDER:
            raise ValueError("CIS output order differs from the V2 contract")

    @property
    def identity(self) -> str:
        return _stable_identity(asdict(self))

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["cis_feature_manifest_identity"] = self.identity
        return record


@dataclass(frozen=True)
class RawCISFeatureTable:
    """An edge-aligned explicit CIS table before train-only normalization."""

    edge_ids: tuple[str, ...]
    target_gene_ids: tuple[str, ...]
    column_names: tuple[str, ...]
    values: np.ndarray
    cis_feature_manifest_identity: str

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values.copy(), columns=self.column_names)
        frame.insert(0, "target_gene_id", self.target_gene_ids)
        frame.insert(0, "edge_id", self.edge_ids)
        return frame


@dataclass(frozen=True)
class CISNormalizationStatistic:
    feature_name: str
    availability_mask_name: str | None
    available_unique_edge_count: int
    mean: float
    standard_deviation: float
    status: str


@dataclass(frozen=True)
class FittedCISNormalization:
    """Frozen train-catalog statistics and the resulting model column order."""

    cis_feature_manifest_identity: str
    train_admitted_gene_ids: tuple[str, ...]
    train_edge_ids: tuple[str, ...]
    statistics: tuple[CISNormalizationStatistic, ...]
    model_output_order: tuple[str, ...]
    numerical_tolerance: float
    normalization_population: str = "train_admitted_unique_structural_edges"
    weighting: str = "each_edge_once_no_cell_molecule_path_transcript_weights"

    @property
    def identity(self) -> str:
        return _stable_identity(asdict(self))

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["cis_normalization_identity"] = self.identity
        return record


@dataclass(frozen=True)
class NormalizedCISFeatureTable:
    edge_ids: tuple[str, ...]
    target_gene_ids: tuple[str, ...]
    column_names: tuple[str, ...]
    values: np.ndarray
    cis_feature_manifest_identity: str
    cis_normalization_identity: str

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values.copy(), columns=self.column_names)
        frame.insert(0, "target_gene_id", self.target_gene_ids)
        frame.insert(0, "edge_id", self.edge_ids)
        return frame


def build_explicit_cis_table(
    structural_edges: pd.DataFrame,
    precomputed_sequence_scores: pd.DataFrame,
    *,
    manifest: CISFeatureManifest,
) -> RawCISFeatureTable:
    """Build the fixed explicit CIS table from edges and audited scanner scores.

    Sequence values are interpreted as the final transformed scanner outputs
    declared in ``manifest``.  Score rows are joined by ``edge_id`` rather than
    by row position, and missing, extra, or duplicate IDs fail closed.
    """

    _require_columns(structural_edges, STRUCTURAL_EDGE_COLUMNS, "structural edges")
    _require_columns(
        precomputed_sequence_scores,
        ("edge_id", *SEQUENCE_FEATURES, *SEQUENCE_MASKS),
        "precomputed CIS sequence scores",
    )
    edge_ids = _validated_identifiers(structural_edges, "edge_id", "structural edges")
    target_gene_ids = _validated_identifiers(
        structural_edges, "target_gene_id", "structural edges"
    )
    score_edge_ids = _validated_identifiers(
        precomputed_sequence_scores,
        "edge_id",
        "precomputed CIS sequence scores",
    )
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("structural edge catalog must contain each edge_id once")
    if len(set(score_edge_ids)) != len(score_edge_ids):
        raise ValueError("precomputed CIS sequence scores contain duplicate edge_id")
    missing = sorted(set(edge_ids) - set(score_edge_ids))
    extra = sorted(set(score_edge_ids) - set(edge_ids))
    if missing or extra:
        raise ValueError(
            "precomputed CIS edge_id set does not exactly match the structural "
            f"catalog; missing={missing}, extra={extra}"
        )

    edges = structural_edges.reset_index(drop=True)
    scores = (
        precomputed_sequence_scores.set_index(
            precomputed_sequence_scores["edge_id"].astype(str), drop=False
        )
        .loc[list(edge_ids)]
        .reset_index(drop=True)
    )
    edge_type = edges["edge_type"].astype(str)
    src_type = edges["src_node_type"].astype(str)
    dst_type = edges["dst_node_type"].astype(str)
    _require_levels(edge_type, EDGE_TYPES, "edge_type")
    _require_levels(src_type, SITE_TYPES, "src_node_type")
    _require_levels(dst_type, SITE_TYPES, "dst_node_type")

    columns: dict[str, np.ndarray] = {}
    for value in EDGE_TYPES:
        columns[f"edge_type__{value}"] = (edge_type == value).to_numpy(np.float32)
    for value in SITE_TYPES:
        columns[f"src_site_type__{value}"] = (src_type == value).to_numpy(np.float32)
    for value in SITE_TYPES:
        columns[f"dst_site_type__{value}"] = (dst_type == value).to_numpy(np.float32)

    span = _finite_numeric(edges, "span_bp", "structural edges")
    length = _finite_numeric(edges, "length_bp", "structural edges")
    if bool((span < 0).any()) or bool((length < 0).any()):
        raise ValueError("structural edge span_bp and length_bp must be non-negative")
    columns["log1p_span_bp"] = np.log1p(span)
    columns["log1p_length_bp"] = np.log1p(length)
    for feature in GEOMETRY_FEATURES[2:]:
        columns[feature] = _finite_numeric(edges, feature, "structural edges")

    applicability = _sequence_applicability(src_type, dst_type)
    for feature in SEQUENCE_FEATURES:
        mask_name = f"{feature}_available"
        values = _finite_numeric(scores, feature, "precomputed CIS sequence scores")
        available = _binary_mask(scores, mask_name)
        expected = applicability[feature]
        invalid_available = available & ~expected
        if bool(invalid_available.any()):
            differing = [edge_ids[index] for index in np.flatnonzero(invalid_available)]
            raise ValueError(
                f"{mask_name} is true without endpoint applicability for edge_id "
                f"{differing}"
            )
        if bool((values[~available] != 0.0).any()):
            raise ValueError(
                f"inapplicable {feature} values must be numeric zero with mask zero"
            )
        if feature in {"edge_gc_fraction", "pas_downstream_u_gu_fraction"} and (
            bool((values < 0.0).any()) or bool((values > 1.0).any())
        ):
            raise ValueError(f"{feature} must be within [0, 1]")
        columns[feature] = values
        columns[mask_name] = available.astype(np.float64)

    values = np.column_stack([columns[name] for name in manifest.output_order])
    if not np.isfinite(values).all():
        raise ValueError("explicit CIS feature table contains non-finite values")
    return RawCISFeatureTable(
        edge_ids=edge_ids,
        target_gene_ids=target_gene_ids,
        column_names=manifest.output_order,
        values=values.astype(np.float32),
        cis_feature_manifest_identity=manifest.identity,
    )


def fit_cis_normalization(
    raw_features: RawCISFeatureTable,
    *,
    train_admitted_gene_ids: Sequence[str],
    manifest: CISFeatureManifest,
) -> FittedCISNormalization:
    """Fit CIS statistics on each unique train-admitted edge exactly once."""

    _validate_raw_table(raw_features, manifest)
    train_genes = tuple(sorted(set(map(str, train_admitted_gene_ids))))
    if not train_genes or any(not value for value in train_genes):
        raise ValueError("train_admitted_gene_ids must contain non-empty IDs")
    known_genes = set(raw_features.target_gene_ids)
    unknown_genes = sorted(set(train_genes) - known_genes)
    if unknown_genes:
        raise ValueError(f"train-admitted genes have no structural edges: {unknown_genes}")
    train_row = np.asarray(
        [gene_id in set(train_genes) for gene_id in raw_features.target_gene_ids],
        dtype=bool,
    )
    if not bool(train_row.any()):
        raise ValueError("CIS normalization population is empty")
    column_index = {name: index for index, name in enumerate(raw_features.column_names)}
    statistics: list[CISNormalizationStatistic] = []
    dropped: set[str] = set()
    for feature in CONTINUOUS_FEATURES:
        availability_name = (
            f"{feature}_available" if feature in SEQUENCE_FEATURES else None
        )
        available = train_row.copy()
        if availability_name is not None:
            available &= raw_features.values[:, column_index[availability_name]] == 1.0
        observed = raw_features.values[available, column_index[feature]].astype(np.float64)
        if not len(observed):
            mean = 0.0
            scale = 0.0
            status = "constant_cis_feature_no_available_edge"
            dropped.add(feature)
        else:
            if not np.isfinite(observed).all():
                raise ValueError(f"train CIS feature {feature} contains non-finite values")
            mean = float(observed.mean())
            scale = float(np.sqrt(np.mean(np.square(observed - mean))))
            if scale <= manifest.normalization.numerical_tolerance:
                status = "constant_cis_feature"
                dropped.add(feature)
            else:
                status = "retained"
        statistics.append(
            CISNormalizationStatistic(
                feature_name=feature,
                availability_mask_name=availability_name,
                available_unique_edge_count=int(available.sum()),
                mean=mean,
                standard_deviation=scale,
                status=status,
            )
        )
    candidate_output_order = tuple(name for name in RAW_FEATURE_ORDER if name not in dropped)
    candidate_values = _normalized_cis_values(
        raw_features,
        column_names=candidate_output_order,
        statistics=statistics,
        numerical_tolerance=manifest.normalization.numerical_tolerance,
    )
    output_order = _select_full_rank_cis_columns(
        candidate_values[train_row],
        candidate_output_order,
        numerical_tolerance=manifest.normalization.numerical_tolerance,
    )
    train_edge_ids = tuple(
        sorted(
            edge_id
            for edge_id, is_train in zip(raw_features.edge_ids, train_row)
            if is_train
        )
    )
    return FittedCISNormalization(
        cis_feature_manifest_identity=manifest.identity,
        train_admitted_gene_ids=train_genes,
        train_edge_ids=train_edge_ids,
        statistics=tuple(statistics),
        model_output_order=output_order,
        numerical_tolerance=manifest.normalization.numerical_tolerance,
    )


def _normalized_cis_values(
    raw_features: RawCISFeatureTable,
    *,
    column_names: Sequence[str],
    statistics: Sequence[CISNormalizationStatistic],
    numerical_tolerance: float,
) -> np.ndarray:
    column_index = {name: index for index, name in enumerate(raw_features.column_names)}
    statistic_by_name = {row.feature_name: row for row in statistics}
    output: list[np.ndarray] = []
    for feature in column_names:
        values = raw_features.values[:, column_index[feature]].astype(np.float64)
        if feature in CONTINUOUS_FEATURES:
            statistic = statistic_by_name[feature]
            if statistic.status != "retained" or (
                statistic.standard_deviation <= numerical_tolerance
            ):
                raise RuntimeError("non-retained CIS feature entered the normalized design")
            if statistic.availability_mask_name is None:
                available = np.ones(len(values), dtype=bool)
            else:
                available = (
                    raw_features.values[
                        :, column_index[statistic.availability_mask_name]
                    ]
                    == 1.0
                )
            normalized = np.zeros(len(values), dtype=np.float64)
            normalized[available] = (
                values[available] - statistic.mean
            ) / statistic.standard_deviation
            values = normalized
        output.append(values)
    return np.column_stack(output)


def _select_full_rank_cis_columns(
    values: np.ndarray,
    column_names: Sequence[str],
    *,
    numerical_tolerance: float,
) -> tuple[str, ...]:
    """Keep a stable full-rank CIS design in the presence of the model bias."""

    matrix = np.asarray(values, dtype=np.float64)
    names = tuple(map(str, column_names))
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not len(matrix):
        raise ValueError("CIS rank closure requires a nonempty aligned matrix")
    retained: list[int] = []
    intercept = np.ones(len(matrix), dtype=np.float64)
    orthonormal = [intercept / np.linalg.norm(intercept)]
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        residual = vector.copy()
        # A second pass prevents a nearly dependent column from borrowing
        # numerical rank from accumulated roundoff.
        for _ in range(2):
            for basis in orthonormal:
                residual -= np.dot(basis, residual) * basis
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > numerical_tolerance * max(float(np.linalg.norm(vector)), 1.0):
            retained.append(column)
            orthonormal.append(residual / residual_norm)
    return tuple(names[index] for index in retained)


def apply_cis_normalization(
    raw_features: RawCISFeatureTable,
    *,
    normalization: FittedCISNormalization,
    manifest: CISFeatureManifest,
    expected_edge_ids: Sequence[str] | None = None,
) -> NormalizedCISFeatureTable:
    """Apply frozen train statistics without clipping held-out feature values."""

    _validate_raw_table(raw_features, manifest)
    if normalization.cis_feature_manifest_identity != manifest.identity:
        raise ValueError("CIS normalization was fitted under a different feature manifest")
    if normalization.normalization_population != manifest.normalization.population:
        raise ValueError("CIS normalization population differs from the feature manifest")
    if normalization.weighting != manifest.normalization.weighting:
        raise ValueError("CIS normalization weighting differs from the feature manifest")
    if normalization.numerical_tolerance != manifest.normalization.numerical_tolerance:
        raise ValueError("CIS normalization tolerance differs from the feature manifest")
    if expected_edge_ids is not None and tuple(map(str, expected_edge_ids)) != raw_features.edge_ids:
        raise ValueError("CIS edge order does not exactly match the requested graph edge axis")

    column_index = {name: index for index, name in enumerate(raw_features.column_names)}
    statistics = {row.feature_name: row for row in normalization.statistics}
    output: list[np.ndarray] = []
    for feature in normalization.model_output_order:
        values = raw_features.values[:, column_index[feature]].astype(np.float64)
        if feature in CONTINUOUS_FEATURES:
            statistic = statistics[feature]
            if statistic.status != "retained":
                raise RuntimeError("constant CIS feature entered the frozen output order")
            if statistic.standard_deviation <= normalization.numerical_tolerance:
                raise RuntimeError("retained CIS feature has a constant frozen scale")
            if statistic.availability_mask_name is None:
                available = np.ones(len(values), dtype=bool)
            else:
                available = (
                    raw_features.values[
                        :, column_index[statistic.availability_mask_name]
                    ]
                    == 1.0
                )
            transformed = np.zeros(len(values), dtype=np.float64)
            transformed[available] = (
                values[available] - statistic.mean
            ) / statistic.standard_deviation
            values = transformed
        output.append(values)
    matrix = np.column_stack(output).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("normalized CIS feature table contains non-finite values")
    return NormalizedCISFeatureTable(
        edge_ids=raw_features.edge_ids,
        target_gene_ids=raw_features.target_gene_ids,
        column_names=normalization.model_output_order,
        values=matrix,
        cis_feature_manifest_identity=manifest.identity,
        cis_normalization_identity=normalization.identity,
    )


def _sequence_applicability(
    src_type: pd.Series, dst_type: pd.Series
) -> dict[str, np.ndarray]:
    has_donor = ((src_type == "donor") | (dst_type == "donor")).to_numpy()
    has_acceptor = ((src_type == "acceptor") | (dst_type == "acceptor")).to_numpy()
    has_tss = ((src_type == "TSS") | (dst_type == "TSS")).to_numpy()
    has_pas = ((src_type == "PAS") | (dst_type == "PAS")).to_numpy()
    return {
        "edge_gc_fraction": np.ones(len(src_type), dtype=bool),
        "donor_strength": has_donor,
        "acceptor_strength": has_acceptor,
        "branchpoint_score": has_acceptor,
        "polypyrimidine_tract_score": has_acceptor,
        "tss_core_promoter_score": has_tss,
        "polya_hexamer_score": has_pas,
        "pas_downstream_u_gu_fraction": has_pas,
    }


def _validate_raw_table(
    raw_features: RawCISFeatureTable, manifest: CISFeatureManifest
) -> None:
    if raw_features.cis_feature_manifest_identity != manifest.identity:
        raise ValueError("raw CIS table was built under a different feature manifest")
    if raw_features.column_names != manifest.output_order:
        raise ValueError("raw CIS columns differ from the frozen feature manifest")
    n_edges = len(raw_features.edge_ids)
    if n_edges == 0:
        raise ValueError("raw CIS feature table is empty")
    if len(set(raw_features.edge_ids)) != n_edges:
        raise ValueError("raw CIS feature table contains duplicate edge_id")
    if len(raw_features.target_gene_ids) != n_edges:
        raise ValueError("raw CIS target_gene_id axis length mismatch")
    if raw_features.values.shape != (n_edges, len(raw_features.column_names)):
        raise ValueError("raw CIS matrix shape does not match its edge/feature axes")
    if not np.isfinite(raw_features.values).all():
        raise ValueError("raw CIS feature table contains non-finite values")
    column_index = {name: index for index, name in enumerate(raw_features.column_names)}
    for prefix, levels in (
        ("edge_type", EDGE_TYPES),
        ("src_site_type", SITE_TYPES),
        ("dst_site_type", SITE_TYPES),
    ):
        one_hot = np.column_stack(
            [raw_features.values[:, column_index[f"{prefix}__{level}"]] for level in levels]
        )
        if not np.isin(one_hot, (0.0, 1.0)).all() or not np.all(
            one_hot.sum(axis=1) == 1.0
        ):
            raise ValueError(f"raw CIS {prefix} fields must form an exact one-hot")

    src_donor = raw_features.values[:, column_index["src_site_type__donor"]] == 1.0
    dst_donor = raw_features.values[:, column_index["dst_site_type__donor"]] == 1.0
    src_acceptor = (
        raw_features.values[:, column_index["src_site_type__acceptor"]] == 1.0
    )
    dst_acceptor = (
        raw_features.values[:, column_index["dst_site_type__acceptor"]] == 1.0
    )
    src_tss = raw_features.values[:, column_index["src_site_type__TSS"]] == 1.0
    dst_tss = raw_features.values[:, column_index["dst_site_type__TSS"]] == 1.0
    src_pas = raw_features.values[:, column_index["src_site_type__PAS"]] == 1.0
    dst_pas = raw_features.values[:, column_index["dst_site_type__PAS"]] == 1.0
    expected_availability = {
        "edge_gc_fraction": np.ones(n_edges, dtype=bool),
        "donor_strength": src_donor | dst_donor,
        "acceptor_strength": src_acceptor | dst_acceptor,
        "branchpoint_score": src_acceptor | dst_acceptor,
        "polypyrimidine_tract_score": src_acceptor | dst_acceptor,
        "tss_core_promoter_score": src_tss | dst_tss,
        "polya_hexamer_score": src_pas | dst_pas,
        "pas_downstream_u_gu_fraction": src_pas | dst_pas,
    }
    for feature in SEQUENCE_FEATURES:
        values = raw_features.values[:, column_index[feature]]
        raw_mask = raw_features.values[:, column_index[f"{feature}_available"]]
        if not np.isin(raw_mask, (0.0, 1.0)).all():
            raise ValueError(f"raw CIS {feature}_available must contain only 0/1")
        available = raw_mask == 1.0
        if not np.array_equal(available, expected_availability[feature]):
            raise ValueError(
                f"raw CIS {feature}_available does not match endpoint applicability"
            )
        if bool((values[~available] != 0.0).any()):
            raise ValueError(
                f"raw CIS inapplicable {feature} must be numeric zero with mask zero"
            )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _validated_identifiers(
    frame: pd.DataFrame, column: str, label: str
) -> tuple[str, ...]:
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"{label} {column} contains null values")
    result = tuple(values.astype(str))
    if any(not value.strip() for value in result):
        raise ValueError(f"{label} {column} contains empty values")
    return result


def _require_levels(values: pd.Series, allowed: Sequence[str], label: str) -> None:
    observed = set(values)
    invalid = sorted(observed - set(allowed))
    if invalid:
        raise ValueError(f"{label} contains unsupported values: {invalid}")


def _finite_numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    values = frame[column]
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise ValueError(f"{label} {column} must be a numeric, non-boolean column")
    result = values.to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} {column} contains non-finite values")
    return result


def _binary_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if not (is_bool_dtype(values.dtype) or is_numeric_dtype(values.dtype)):
        raise ValueError(f"{column} must be boolean or numeric binary")
    result = values.to_numpy(dtype=np.float64)
    if not np.isfinite(result).all() or not np.isin(result, (0.0, 1.0)).all():
        raise ValueError(f"{column} must contain only 0/1 values")
    return result.astype(bool)


def _stable_identity(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
