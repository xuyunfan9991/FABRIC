"""FABRIC V2 cell context, fixed route design, and data admission.

The public functions in this module are deliberately table-first.  Every axis
has stable IDs, every train-derived quantity requires an explicit train mask,
and audit-only records never silently enter model tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .motifs import EVENT_ROUTE_COLUMNS, PHYSICAL_EVENT_COLUMNS


INFORMATIVE_FATE = "likelihood_informative"
EMPTY_FATE = "no_matrix_isoform_compatible"
FULL_FATE = "matrix_catalog_compatible_uninformative"
COMPATIBILITY_FATES = (EMPTY_FATE, INFORMATIVE_FATE, FULL_FATE)


@dataclass(frozen=True)
class ActivityContext:
    cell_ids: tuple[str, ...]
    activity_entity_ids: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray
    library_size: np.ndarray


@dataclass(frozen=True)
class ATACMappingContext:
    cell_ids: tuple[str, ...]
    peak_ids: tuple[str, ...]
    accessibility: sparse.csr_matrix
    mapping_valid: np.ndarray
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class GateRawSignals:
    cell_ids: tuple[str, ...]
    gate_key_ids: tuple[str, ...]
    raw: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class GateValues:
    cell_ids: tuple[str, ...]
    gate_key_ids: tuple[str, ...]
    raw: np.ndarray
    standardized_residual: np.ndarray
    gate: np.ndarray
    observed: np.ndarray
    out_of_train_range: np.ndarray
    out_of_train_quantile_support: np.ndarray


@dataclass(frozen=True)
class GateCollinearityAudit:
    pairs: pd.DataFrame
    correlated_sets: pd.DataFrame


@dataclass(frozen=True)
class RouteBaseDesign:
    route_ids: tuple[str, ...]
    values: np.ndarray
    column_names: tuple[str, ...]
    manifest: Mapping[str, object]
    route_context: pd.DataFrame


@dataclass(frozen=True)
class InteractionDesign:
    route_ids: tuple[str, ...]
    values_by_modality: Mapping[str, np.ndarray]
    active_mask_by_modality: Mapping[str, np.ndarray]
    route_indices_by_modality: Mapping[str, np.ndarray]
    raw_support: pd.DataFrame
    manifest: Mapping[str, object]
    raw_contrasts: pd.DataFrame


@dataclass(frozen=True)
class ProductionModalityTensors:
    cell_ids: tuple[str, ...]
    target_gene_id: str
    modality: str
    ordered_edge_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    gate_key_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    route_event_index: np.ndarray
    route_edge_index: np.ndarray
    route_weight: np.ndarray
    route_base_features: np.ndarray | sparse.spmatrix
    route_interaction_features: np.ndarray | sparse.spmatrix
    interaction_active_mask: np.ndarray
    event_gate_key_index: np.ndarray
    gate: np.ndarray


@dataclass(frozen=True)
class CompatibilityArtifactValidation:
    status: str
    reasons: tuple[str, ...]
    informative_gene_ids: tuple[str, ...]
    audit: pd.DataFrame
    candidate_gene_ids: tuple[str, ...] = ()
    legal_path_catalog_identity: str = ""
    cell_split_identity: str = ""
    test_exposure: str = ""
    model_isoform_universe: str = ""
    matrix_structural_path_count: int = 0


def build_compatibility_admission_record(
    validation: CompatibilityArtifactValidation,
    *,
    input_manifest_id: str,
    compatibility_artifact_id: str,
) -> dict[str, object]:
    """Create the narrow runtime record only from a strict admitted validation."""

    for name, value in (
        ("input_manifest_id", input_manifest_id),
        ("compatibility_artifact_id", compatibility_artifact_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"compatibility admission {name} must be nonempty")
    if validation.status != "ADMITTED" or validation.reasons:
        raise RuntimeError("rejected compatibility validation cannot create admission")
    candidates = _unique_ids(
        validation.candidate_gene_ids, "validated structural candidate"
    )
    informative = validation.informative_gene_ids
    if not informative or not set(informative).issubset(candidates):
        raise RuntimeError("validated informative gene axis is empty or outside candidates")
    for name in ("legal_path_catalog_identity", "cell_split_identity"):
        value = getattr(validation, name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"validated compatibility {name} is empty")
    if validation.model_isoform_universe != (
        "resolved_ont_matrix_structural_paths_only"
    ):
        raise RuntimeError("validated compatibility path universe is not matrix-only")
    if type(validation.matrix_structural_path_count) is not int or (
        validation.matrix_structural_path_count < 1
    ):
        raise RuntimeError("validated matrix structural-path count is invalid")
    exposure = validation.test_exposure
    if exposure not in {
        "not_materialized_before_checkpoint",
        "previously_materialized",
    }:
        raise RuntimeError("validated compatibility exposure marker is invalid")
    return {
        "admission_pass": True,
        "validation_status": validation.status,
        "input_manifest_id": input_manifest_id,
        "compatibility_artifact_id": compatibility_artifact_id,
        "structural_candidate_count": len(candidates),
        "model_isoform_universe": validation.model_isoform_universe,
        "matrix_structural_path_count": validation.matrix_structural_path_count,
        "informative_gene_ids": list(informative),
        "legal_path_catalog_identity": validation.legal_path_catalog_identity,
        "cell_split_identity": validation.cell_split_identity,
        "test_exposure": exposure,
    }


@dataclass(frozen=True)
class IRPolicy:
    minimum_mapq: float
    minimum_exon_aligned_bp_each_side: int
    minimum_intron_aligned_bp_each_side: int


@dataclass(frozen=True)
class OntObservationProcessAudit:
    status: str
    comparison_name: str
    audit: pd.DataFrame
    reasons: tuple[str, ...]


def build_ont_observation_admission_record(
    audit: OntObservationProcessAudit,
    *,
    matrix_identity: str,
    crosswalk_identity: str,
    path_identity: str,
    split_identity: str,
    metric_schema_version: str,
) -> dict[str, object]:
    """Create the narrow runtime record only from an admitted ONT audit."""

    if audit.status != "ADMITTED":
        raise RuntimeError("pending ONT observation audit cannot create admission")
    identities = {
        "matrix_identity": matrix_identity,
        "crosswalk_identity": crosswalk_identity,
        "path_identity": path_identity,
        "split_identity": split_identity,
        "metric_schema_version": metric_schema_version,
    }
    if any(not isinstance(value, str) or not value.strip() for value in identities.values()):
        raise ValueError("ONT observation admission identities must be nonempty")
    return {
        "admission_pass": True,
        "status": audit.status,
        "comparison_name": audit.comparison_name,
        **identities,
    }


@dataclass(frozen=True)
class V2GeneAssembly:
    gene_id: str
    model_input: object
    compatible_path_indices: torch.Tensor
    compatible_path_mask: torch.Tensor
    row_cell_index: torch.Tensor
    molecule_count: torch.Tensor
    informative_row_mask: torch.Tensor
    cell_ids: tuple[str, ...]
    cell_split: tuple[str, ...]
    path_ids: tuple[str, ...]


def normalize_log1p_counts(
    counts: sparse.spmatrix | np.ndarray, *, target_sum: float = 10_000.0
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Apply full-library CP10K-log1p without changing the gene denominator."""

    if target_sum <= 0:
        raise ValueError("normalization target_sum must be positive")
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or bool((matrix.data < 0).any())
    ):
        raise ValueError("count matrix must contain finite non-negative values")
    library_size = np.asarray(matrix.sum(axis=1)).reshape(-1)
    normalized = matrix.copy()
    positive = library_size > 0
    if bool(positive.any()):
        scale = np.zeros(len(library_size), dtype=np.float64)
        scale[positive] = target_sum / library_size[positive]
        normalized = sparse.diags(scale) @ normalized
        normalized = normalized.tocsr()
        normalized.data = np.log1p(normalized.data)
    return normalized.astype(np.float32), library_size.astype(np.float64)


def compute_activity_entities(
    raw_counts: sparse.spmatrix | np.ndarray,
    *,
    cell_ids: Sequence[str],
    frozen_gene_axis: Sequence[str],
    entity_table: pd.DataFrame,
    rna_observation_valid: Sequence[bool] | None = None,
    target_sum: float = 10_000.0,
) -> ActivityContext:
    """Build unique/group activity after raw-count summation and one CP10K-log1p."""

    genes = tuple(map(str, frozen_gene_axis))
    cells = _unique_ids(cell_ids, "RNA cell")
    if len(genes) != len(set(genes)) or any(not value for value in genes):
        raise ValueError("frozen RNA source gene axis must be unique and non-empty")
    counts = sparse.csr_matrix(raw_counts, dtype=np.float64)
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("raw RNA count shape differs from cell/full-gene axes")
    if counts.data.size and (
        not np.isfinite(counts.data).all() or bool((counts.data < 0).any())
    ):
        raise ValueError("raw RNA counts must be finite and non-negative")
    required = {"activity_entity_id", "activity_gene_ids", "source_valid"}
    _require_columns(entity_table, required, "activity entity table")
    entities = entity_table.sort_values("activity_entity_id", kind="mergesort").reset_index(
        drop=True
    )
    if entities["activity_entity_id"].astype(str).duplicated().any():
        raise ValueError("activity entity IDs must be unique")
    gene_index = {value: index for index, value in enumerate(genes)}
    library = np.asarray(counts.sum(axis=1)).reshape(-1)
    valid_rna = (
        np.ones(len(cells), dtype=bool)
        if rna_observation_valid is None
        else np.asarray(rna_observation_valid, dtype=bool)
    )
    if valid_rna.shape != (len(cells),):
        raise ValueError("RNA observation-valid mask must match cell axis")
    valid_rna &= library > 0
    values = np.zeros((len(cells), len(entities)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    for column, entity in enumerate(entities.itertuples(index=False)):
        member_ids = _string_list(entity.activity_gene_ids, "activity_gene_ids")
        if not member_ids:
            raise ValueError("activity entities require at least one source gene")
        missing = sorted(set(member_ids) - set(gene_index))
        source_valid = bool(entity.source_valid) and not missing
        if bool(entity.source_valid) and missing:
            raise ValueError(
                f"activity entity {entity.activity_entity_id} is marked valid but "
                f"members are absent: {missing}"
            )
        if not source_valid:
            continue
        raw_sum = np.asarray(counts[:, [gene_index[value] for value in member_ids]].sum(axis=1)).reshape(-1)
        values[valid_rna, column] = np.log1p(
            target_sum * raw_sum[valid_rna] / library[valid_rna]
        ).astype(np.float32)
        # Sparse zero is observed zero when the full RNA observation is valid.
        observed[valid_rna, column] = True
    return ActivityContext(
        cell_ids=cells,
        activity_entity_ids=tuple(entities["activity_entity_id"].astype(str)),
        values=values,
        observed=observed,
        library_size=library,
    )


def assess_atac_mapping(
    neighbors: pd.DataFrame,
    *,
    target_cell_ids: Sequence[str],
    expected_k: int,
    maximum_distance: float,
) -> pd.DataFrame:
    """Evaluate absolute mapping/QC rules; ESS-like metadata is diagnostic only."""

    cells = _unique_ids(target_cell_ids, "target RNA cell")
    if expected_k <= 0 or maximum_distance < 0 or not np.isfinite(maximum_distance):
        raise ValueError("ATAC mapping K/distance policy is invalid")
    required = {
        "cell_id",
        "neighbor_atac_cell_id",
        "neighbor_weight",
        "distance",
        "rna_qc_pass",
        "atac_qc_pass",
        "pairing_valid",
        "neighborhood_consistency_status",
    }
    _require_columns(neighbors, required, "ATAC neighbor table")
    extra = sorted(set(neighbors["cell_id"].astype(str)) - set(cells))
    if extra:
        raise ValueError(f"ATAC neighbor table contains non-target cells: {extra[:5]}")
    rows: list[dict[str, object]] = []
    for cell_id in cells:
        group = neighbors.loc[neighbors["cell_id"].astype(str) == cell_id].copy()
        reasons: list[str] = []
        if group.empty:
            rows.append(
                {
                    "cell_id": cell_id,
                    "mapping_valid": False,
                    "mapping_failure_reasons": ["no_legal_neighbor"],
                    "neighbor_count": 0,
                    "ess_atac": np.nan,
                    "evenness_atac": np.nan,
                    "coverage_atac": 0.0,
                    "maximum_neighbor_weight": np.nan,
                    "nearest_distance": np.nan,
                    "weighted_mean_distance": np.nan,
                    "neighborhood_consistency_status": "not_estimable",
                }
            )
            continue
        if len(group) > expected_k:
            reasons.append("neighbor_count_exceeds_policy")
        if group["neighbor_atac_cell_id"].astype(str).duplicated().any():
            reasons.append("duplicate_neighbor")
        weights = group["neighbor_weight"].to_numpy(dtype=np.float64)
        distances = group["distance"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or bool((weights < 0).any()) or not np.isclose(
            weights.sum(), 1.0, atol=1e-8, rtol=0
        ):
            reasons.append("invalid_neighbor_weights")
        if not np.isfinite(distances).all() or bool((distances > maximum_distance).any()):
            reasons.append("distance_outside_admissible_range")
        if not bool(group["rna_qc_pass"].astype(bool).all()):
            reasons.append("rna_qc_failure")
        if not bool(group["atac_qc_pass"].astype(bool).all()):
            reasons.append("atac_qc_failure")
        if not bool(group["pairing_valid"].astype(bool).all()):
            reasons.append("pairing_policy_failure")
        statuses = set(group["neighborhood_consistency_status"].astype(str))
        consistency = next(iter(statuses)) if len(statuses) == 1 else "inconsistent_metadata"
        if len(group) == 1 and consistency == "not_estimable":
            pass
        elif consistency != "pass":
            reasons.append("neighborhood_consistency_failure")
        if np.isfinite(weights).all() and bool((weights >= 0).all()) and weights.sum() > 0:
            normalized_weights = weights / weights.sum()
            ess = 1.0 / float(np.square(normalized_weights).sum())
            evenness = ess / len(group)
            maximum_weight = float(normalized_weights.max())
            weighted_distance = float(np.sum(normalized_weights * distances))
        else:
            ess = evenness = maximum_weight = weighted_distance = np.nan
        rows.append(
            {
                "cell_id": cell_id,
                "mapping_valid": not reasons,
                "mapping_failure_reasons": reasons,
                "neighbor_count": len(group),
                "ess_atac": ess,
                "evenness_atac": evenness,
                "coverage_atac": len(group) / expected_k,
                "maximum_neighbor_weight": maximum_weight,
                "nearest_distance": float(np.min(distances)) if np.isfinite(distances).all() else np.nan,
                "weighted_mean_distance": weighted_distance,
                "neighborhood_consistency_status": consistency,
            }
        )
    return pd.DataFrame(rows)


def map_atac_accessibility(
    raw_atac_counts: sparse.spmatrix | np.ndarray,
    *,
    atac_cell_ids: Sequence[str],
    peak_ids: Sequence[str],
    target_cell_ids: Sequence[str],
    neighbors: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    target_sum: float = 10_000.0,
) -> ATACMappingContext:
    """Normalize each ATAC cell first, then map with frozen neighbor weights."""

    atac_ids = _unique_ids(atac_cell_ids, "ATAC cell")
    peaks = _unique_ids(peak_ids, "ATAC peak")
    targets = _unique_ids(target_cell_ids, "target RNA cell")
    counts = sparse.csr_matrix(raw_atac_counts)
    if counts.shape != (len(atac_ids), len(peaks)):
        raise ValueError("ATAC count shape differs from cell/peak axes")
    normalized, library = normalize_log1p_counts(counts, target_sum=target_sum)
    _require_columns(mapping_audit, {"cell_id", "mapping_valid"}, "ATAC mapping audit")
    if mapping_audit["cell_id"].astype(str).duplicated().any():
        raise ValueError("ATAC mapping audit has duplicate target cells")
    audit = mapping_audit.set_index(mapping_audit["cell_id"].astype(str)).reindex(targets)
    if audit["mapping_valid"].isna().any():
        raise ValueError("ATAC mapping audit does not cover the complete target axis")
    mapping_valid = audit["mapping_valid"].to_numpy(dtype=bool)
    _require_columns(
        neighbors,
        {"cell_id", "neighbor_atac_cell_id", "neighbor_weight"},
        "ATAC neighbor table",
    )
    weights_raw = neighbors["neighbor_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights_raw).all() or bool((weights_raw < 0).any()):
        raise ValueError("ATAC neighbor weights must be finite and non-negative")
    if neighbors.duplicated(["cell_id", "neighbor_atac_cell_id"]).any():
        raise ValueError("ATAC neighbor identities must be unique per target cell")
    for cell_id, group in neighbors.groupby("cell_id", sort=False):
        if str(cell_id) not in set(targets):
            raise ValueError("ATAC neighbors contain a cell outside the frozen target axis")
        target_row = targets.index(str(cell_id))
        if mapping_valid[target_row] and not np.isclose(
            group["neighbor_weight"].astype(float).sum(), 1.0, atol=1e-8, rtol=0
        ):
            raise ValueError("valid ATAC mapping neighbor weights must sum to one")
    target_index = {value: index for index, value in enumerate(targets)}
    atac_index = {value: index for index, value in enumerate(atac_ids)}
    weight_rows: list[int] = []
    weight_cols: list[int] = []
    weight_values: list[float] = []
    for row in neighbors.itertuples(index=False):
        cell_id, neighbor_id = str(row.cell_id), str(row.neighbor_atac_cell_id)
        if cell_id not in target_index or neighbor_id not in atac_index:
            raise ValueError("ATAC neighbor identity is absent from a frozen axis")
        if not mapping_valid[target_index[cell_id]]:
            continue
        if library[atac_index[neighbor_id]] <= 0:
            raise ValueError("valid ATAC mapping references a zero-library ATAC cell")
        weight_rows.append(target_index[cell_id])
        weight_cols.append(atac_index[neighbor_id])
        weight_values.append(float(row.neighbor_weight))
    weights = sparse.csr_matrix(
        (weight_values, (weight_rows, weight_cols)),
        shape=(len(targets), len(atac_ids)),
    )
    if bool(mapping_valid.any()):
        sums = np.asarray(weights.sum(axis=1)).reshape(-1)
        if not np.allclose(sums[mapping_valid], 1.0, atol=1e-8, rtol=0):
            raise ValueError("valid mapped cells require neighbor weights summing to one")
    accessibility = (weights @ normalized).tocsr().astype(np.float32)
    return ATACMappingContext(
        cell_ids=targets,
        peak_ids=peaks,
        accessibility=accessibility,
        mapping_valid=mapping_valid,
        diagnostics=audit.reset_index(drop=True),
    )


def build_gate_keys(physical_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign one stable shared dynamic key to every physical event."""

    _require_columns(physical_events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    events = physical_events.copy()
    rows: dict[str, dict[str, object]] = {}
    key_ids: list[str] = []
    for event in events.itertuples(index=False):
        gene = str(event.target_gene_id)
        modality = str(event.modality)
        kind = str(event.factor_identity_kind)
        if modality == "RNA":
            channel = "RNA"
            identity = (gene, str(event.activity_entity_id))
        elif kind == "accessibility_only":
            channel = "Open"
            identity = (gene, str(event.peak_id))
        else:
            channel = "DNA"
            identity = (gene, str(event.activity_entity_id), str(event.peak_id))
        gate_key_id = "gate:" + hashlib.sha256(
            json.dumps([channel, *identity], separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        key_ids.append(gate_key_id)
        row = {
            "gate_key_id": gate_key_id,
            "target_gene_id": gene,
            "channel": channel,
            "activity_entity_id": (
                None if channel == "Open" else str(event.activity_entity_id)
            ),
            "peak_id": None if channel == "RNA" else str(event.peak_id),
        }
        if gate_key_id in rows and rows[gate_key_id] != row:
            raise RuntimeError("stable gate-key hash collision")
        rows[gate_key_id] = row
    events["gate_key_id"] = key_ids
    keys = pd.DataFrame(rows.values()).sort_values("gate_key_id", kind="mergesort")
    return events, keys.reset_index(drop=True)


def build_raw_gate_signals(
    gate_keys: pd.DataFrame,
    *,
    activity: ActivityContext,
    atac: ATACMappingContext,
) -> GateRawSignals:
    """Construct RNA, factor-specific DNA, and Open raw signals and masks."""

    _require_columns(
        gate_keys,
        {"gate_key_id", "channel", "activity_entity_id", "peak_id"},
        "gate key table",
    )
    if activity.cell_ids != atac.cell_ids:
        raise ValueError("RNA activity and mapped ATAC cell axes differ")
    keys = gate_keys.sort_values("gate_key_id", kind="mergesort").reset_index(drop=True)
    if keys["gate_key_id"].astype(str).duplicated().any():
        raise ValueError("gate-key IDs must be unique")
    entity_index = {value: index for index, value in enumerate(activity.activity_entity_ids)}
    peak_index = {value: index for index, value in enumerate(atac.peak_ids)}
    raw = np.zeros((len(activity.cell_ids), len(keys)), dtype=np.float32)
    observed = np.zeros_like(raw, dtype=bool)
    for column, key in enumerate(keys.itertuples(index=False)):
        channel = str(key.channel)
        if channel == "RNA":
            entity = str(key.activity_entity_id)
            if entity not in entity_index:
                raise ValueError(f"RNA gate activity entity is absent: {entity}")
            index = entity_index[entity]
            raw[:, column] = activity.values[:, index]
            observed[:, column] = activity.observed[:, index]
        elif channel == "Open":
            peak = str(key.peak_id)
            if peak not in peak_index:
                raise ValueError(f"Open gate peak is absent: {peak}")
            raw[:, column] = atac.accessibility[:, peak_index[peak]].toarray().reshape(-1)
            observed[:, column] = atac.mapping_valid
        elif channel == "DNA":
            entity, peak = str(key.activity_entity_id), str(key.peak_id)
            if entity not in entity_index or peak not in peak_index:
                raise ValueError("DNA gate entity or peak is absent from context axes")
            factor = activity.values[:, entity_index[entity]]
            access = atac.accessibility[:, peak_index[peak]].toarray().reshape(-1)
            # Contractual product-before-centering in the non-negative space.
            raw[:, column] = factor * access
            observed[:, column] = (
                activity.observed[:, entity_index[entity]] & atac.mapping_valid
            )
        else:
            raise ValueError(f"unknown gate channel: {channel}")
    return GateRawSignals(
        cell_ids=activity.cell_ids,
        gate_key_ids=tuple(keys["gate_key_id"].astype(str)),
        raw=raw,
        observed=observed,
    )


def fit_gate_admission(
    raw_signals: GateRawSignals,
    gate_keys: pd.DataFrame,
    *,
    train_mask: Sequence[bool],
    informative_molecule_mass: np.ndarray,
    thresholds_by_channel: Mapping[str, Mapping[str, float]],
    support_quantiles: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Fit train-only molecule-weighted gate baselines, scales, and admission."""

    keys = gate_keys.sort_values("gate_key_id", kind="mergesort").reset_index(drop=True)
    if tuple(keys["gate_key_id"].astype(str)) != raw_signals.gate_key_ids:
        raise ValueError("raw gate-key axis differs from GateKeyTable")
    train = np.asarray(train_mask, dtype=bool)
    mass = np.asarray(informative_molecule_mass, dtype=np.float64)
    expected = raw_signals.raw.shape
    if train.shape != (expected[0],):
        raise ValueError("gate train mask must have one value per cell")
    if mass.shape != expected:
        raise ValueError("informative molecule mass must have shape [cell, gate_key]")
    if not np.isfinite(mass).all() or bool((mass < 0).any()):
        raise ValueError("informative molecule mass must be finite and non-negative")
    if not (0 <= support_quantiles[0] < support_quantiles[1] <= 1):
        raise ValueError("gate support quantiles are invalid")
    rows: list[dict[str, object]] = []
    for column, key in enumerate(keys.itertuples(index=False)):
        channel = str(key.channel)
        if channel not in thresholds_by_channel:
            raise ValueError(f"gate thresholds are absent for channel {channel}")
        thresholds = thresholds_by_channel[channel]
        required_thresholds = {
            "minimum_valid_cells",
            "minimum_effective_cells",
            "minimum_informative_molecules",
            "minimum_standard_deviation",
        }
        missing = required_thresholds - set(thresholds)
        if missing:
            raise ValueError(f"gate thresholds for {channel} miss {sorted(missing)}")
        threshold_values = {
            name: float(thresholds[name]) for name in required_thresholds
        }
        if not np.isfinite(list(threshold_values.values())).all() or any(
            value < 0 for value in threshold_values.values()
        ):
            raise ValueError(f"gate thresholds for {channel} must be finite and non-negative")
        if not threshold_values["minimum_valid_cells"].is_integer():
            raise ValueError("minimum_valid_cells must be an integer count")
        valid = train & raw_signals.observed[:, column] & (mass[:, column] > 0)
        weights = mass[valid, column]
        values = raw_signals.raw[valid, column].astype(np.float64)
        n_valid = int(valid.sum())
        weight_sum = float(weights.sum())
        n_eff = (
            0.0
            if not len(weights) or float(np.square(weights).sum()) == 0
            else weight_sum**2 / float(np.square(weights).sum())
        )
        if weight_sum > 0:
            mean = float(np.dot(weights, values) / weight_sum)
            variance = float(np.dot(weights, np.square(values - mean)) / weight_sum)
            standard_deviation = float(np.sqrt(max(variance, 0.0)))
            raw_min, raw_max = float(values.min()), float(values.max())
            lower = _weighted_quantile(values, weights, support_quantiles[0])
            upper = _weighted_quantile(values, weights, support_quantiles[1])
        else:
            mean = standard_deviation = raw_min = raw_max = lower = upper = np.nan
        reasons: list[str] = []
        if weight_sum == 0:
            reasons.append("no_train_observation")
        if n_valid < float(thresholds["minimum_valid_cells"]):
            reasons.append("insufficient_valid_cells")
        if n_eff < float(thresholds["minimum_effective_cells"]):
            reasons.append("insufficient_gate_effective_cells")
        if weight_sum < float(thresholds["minimum_informative_molecules"]):
            reasons.append("insufficient_informative_molecules")
        sd_floor = max(float(thresholds["minimum_standard_deviation"]), 1.0e-8)
        if not np.isfinite(standard_deviation) or standard_deviation <= sd_floor:
            reasons.append("insufficient_train_variation")
        rows.append(
            {
                "gate_key_id": str(key.gate_key_id),
                "target_gene_id": str(key.target_gene_id),
                "channel": channel,
                "n_valid_cells": n_valid,
                "n_eff_gate": n_eff,
                "informative_molecule_mass": weight_sum,
                "train_mean": mean,
                "train_standard_deviation": standard_deviation,
                "train_raw_minimum": raw_min,
                "train_raw_maximum": raw_max,
                "train_lower_weighted_quantile": lower,
                "train_upper_weighted_quantile": upper,
                "support_quantile_probabilities": list(support_quantiles),
                "minimum_valid_cells": float(thresholds["minimum_valid_cells"]),
                "minimum_effective_cells": float(thresholds["minimum_effective_cells"]),
                "minimum_informative_molecules": float(
                    thresholds["minimum_informative_molecules"]
                ),
                "minimum_standard_deviation": float(
                    thresholds["minimum_standard_deviation"]
                ),
                "gate_key_active": not reasons,
                "failure_reasons": reasons,
            }
        )
    return pd.DataFrame(rows)


def transform_gates(
    raw_signals: GateRawSignals, gate_admission: pd.DataFrame
) -> GateValues:
    """Apply frozen train mean/scale without clipping or re-estimation."""

    required = {
        "gate_key_id",
        "train_mean",
        "train_standard_deviation",
        "train_raw_minimum",
        "train_raw_maximum",
        "train_lower_weighted_quantile",
        "train_upper_weighted_quantile",
        "gate_key_active",
    }
    _require_columns(gate_admission, required, "GateAdmissionManifest")
    manifest = gate_admission.set_index(gate_admission["gate_key_id"].astype(str))
    if manifest.index.duplicated().any() or set(manifest.index) != set(
        raw_signals.gate_key_ids
    ):
        raise ValueError("GateAdmissionManifest differs from the raw gate-key axis")
    manifest = manifest.loc[list(raw_signals.gate_key_ids)]
    z = np.zeros_like(raw_signals.raw, dtype=np.float32)
    gate = np.zeros_like(raw_signals.raw, dtype=np.float32)
    outside_range = np.zeros_like(raw_signals.observed, dtype=bool)
    outside_quantile = np.zeros_like(raw_signals.observed, dtype=bool)
    for column, row in enumerate(manifest.itertuples(index=False)):
        if not bool(row.gate_key_active):
            continue
        mean, sd = float(row.train_mean), float(row.train_standard_deviation)
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 1e-8:
            raise ValueError("active gate has an invalid frozen baseline or scale")
        observed = raw_signals.observed[:, column]
        z[observed, column] = (
            (raw_signals.raw[observed, column].astype(np.float64) - mean) / sd
        ).astype(np.float32)
        gate[observed, column] = z[observed, column]
        values = raw_signals.raw[:, column]
        outside_range[:, column] = observed & (
            (values < float(row.train_raw_minimum))
            | (values > float(row.train_raw_maximum))
        )
        outside_quantile[:, column] = observed & (
            (values < float(row.train_lower_weighted_quantile))
            | (values > float(row.train_upper_weighted_quantile))
        )
    return GateValues(
        cell_ids=raw_signals.cell_ids,
        gate_key_ids=raw_signals.gate_key_ids,
        raw=raw_signals.raw.copy(),
        standardized_residual=z,
        gate=gate,
        observed=raw_signals.observed.copy(),
        out_of_train_range=outside_range,
        out_of_train_quantile_support=outside_quantile,
    )


def build_gate_collinearity_audit(
    gate_values: GateValues,
    gate_keys: pd.DataFrame,
    *,
    train_mask: Sequence[bool],
    informative_molecule_mass_by_gene: pd.DataFrame,
    minimum_joint_effective_cells: float,
    absolute_correlation_threshold: float,
) -> GateCollinearityAudit:
    """Audit active gate pairs only on common valid, informative train cells."""

    if minimum_joint_effective_cells <= 0 or not 0 < absolute_correlation_threshold <= 1:
        raise ValueError("gate-collinearity thresholds are invalid")
    _require_columns(
        gate_keys,
        {"gate_key_id", "target_gene_id", "channel", "gate_key_active"},
        "gate keys for collinearity audit",
    )
    all_keys = gate_keys.set_index(gate_keys["gate_key_id"].astype(str))
    keys = all_keys.loc[all_keys["gate_key_active"].astype(bool)].copy()
    if not set(keys.index).issubset(set(gate_values.gate_key_ids)):
        raise ValueError("gate-collinearity key table differs from gate tensor axis")
    train = np.asarray(train_mask, dtype=bool)
    if train.shape != (len(gate_values.cell_ids),):
        raise ValueError("gate-collinearity train mask differs from cell axis")
    _require_columns(
        informative_molecule_mass_by_gene,
        {"cell_id", "target_gene_id", "informative_molecule_mass"},
        "cell-gene informative mass",
    )
    mass_lookup = {
        (str(row.cell_id), str(row.target_gene_id)): float(row.informative_molecule_mass)
        for row in informative_molecule_mass_by_gene.itertuples(index=False)
    }
    index_by_key = {value: index for index, value in enumerate(gate_values.gate_key_ids)}
    pair_rows: list[dict[str, object]] = []
    high_edges: list[tuple[str, str]] = []
    for gene_id, gene_keys in keys.groupby("target_gene_id", sort=True):
        key_ids = sorted(gene_keys.index.astype(str))
        weights = np.asarray(
            [mass_lookup.get((cell_id, str(gene_id)), 0.0) for cell_id in gate_values.cell_ids],
            dtype=np.float64,
        )
        for left_pos, left_id in enumerate(key_ids):
            for right_id in key_ids[left_pos + 1 :]:
                left, right = index_by_key[left_id], index_by_key[right_id]
                common = (
                    train
                    & gate_values.observed[:, left]
                    & gate_values.observed[:, right]
                    & (weights > 0)
                )
                joint_weights = weights[common]
                n_eff = (
                    0.0
                    if not len(joint_weights)
                    else float(joint_weights.sum() ** 2 / np.square(joint_weights).sum())
                )
                correlation = np.nan
                status = "evidence_separation_not_estimable"
                collinearity_kind = "insufficient_joint_support"
                correlation_sign = "not_estimable"
                not_estimable_reason: str | None = "joint_support_insufficient"
                if n_eff >= minimum_joint_effective_cells:
                    x = gate_values.gate[common, left].astype(np.float64)
                    y = gate_values.gate[common, right].astype(np.float64)
                    x_mean = float(np.dot(joint_weights, x) / joint_weights.sum())
                    y_mean = float(np.dot(joint_weights, y) / joint_weights.sum())
                    x_center = x - x_mean
                    y_center = y - y_mean
                    denominator = np.sqrt(
                        np.dot(joint_weights, np.square(x_center))
                        * np.dot(joint_weights, np.square(y_center))
                    )
                    if denominator > 0:
                        not_estimable_reason = None
                        correlation = float(
                            np.dot(joint_weights, x_center * y_center) / denominator
                        )
                        correlation_sign = "negative" if correlation < 0 else "positive"
                        if abs(correlation) >= absolute_correlation_threshold:
                            status = "correlated_evidence"
                            collinearity_kind = (
                                "perfect_collinearity"
                                if np.isclose(abs(correlation), 1.0, atol=1.0e-12, rtol=0)
                                else "near_collinearity"
                            )
                            high_edges.append((left_id, right_id))
                        else:
                            status = "no_high_pairwise_collinearity_detected"
                            collinearity_kind = "below_frozen_threshold"
                    else:
                        collinearity_kind = "common_weighted_variance_zero"
                        not_estimable_reason = "zero_common_variance"
                pair_rows.append(
                    {
                        "target_gene_id": str(gene_id),
                        "left_gate_key_id": left_id,
                        "right_gate_key_id": right_id,
                        "left_channel": str(keys.loc[left_id, "channel"]),
                        "right_channel": str(keys.loc[right_id, "channel"]),
                        "joint_valid_cell_count": int(common.sum()),
                        "joint_effective_cell_count": n_eff,
                        "weighted_pearson_correlation": correlation,
                        "collinearity_kind": collinearity_kind,
                        "correlation_sign": correlation_sign,
                        "not_estimable_reason": not_estimable_reason,
                        "status": status,
                    }
                )
    set_rows: list[dict[str, object]] = []
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left, right in high_edges:
        union(left, right)
    groups: dict[str, list[str]] = {}
    for key_id in sorted(parent):
        groups.setdefault(find(key_id), []).append(key_id)
    for members in groups.values():
        set_id = "correlated:" + hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        set_rows.append(
            {
                "correlated_evidence_set_id": set_id,
                "target_gene_id": str(keys.loc[members[0], "target_gene_id"]),
                "member_gate_key_ids": members,
                "member_count": len(members),
                "pairwise_edges": [
                    [left, right]
                    for left, right in high_edges
                    if left in members and right in members
                ],
            }
        )
    return GateCollinearityAudit(
        pairs=pd.DataFrame(pair_rows), correlated_sets=pd.DataFrame(set_rows)
    )


def build_event_feature_manifest(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    *,
    distance_bin_boundaries: Mapping[str, Sequence[float]],
    scientific_context_pairs: Mapping[
        str, Mapping[str, Sequence[Sequence[str]]]
    ],
    motif_score_in_model: bool = False,
    motif_score_calibration_identity: str | None = None,
    orientation_interaction_policy: Mapping[str, bool] | None = None,
    numeric_rank_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Freeze the split-neutral V2 event/route feature vocabulary.

    The manifest is built only from the physical catalog and retained routing
    table.  It therefore cannot depend on compatible-path outcomes or on a
    train/validation split.  Absolute coordinates and biological IDs are
    retained in provenance tables, never in numeric feature columns.
    """

    _require_columns(physical_events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    _require_columns(event_routes, set(EVENT_ROUTE_COLUMNS), "EventRouteTable")
    if numeric_rank_tolerance <= 0:
        raise ValueError("numeric rank tolerance must be positive")
    if motif_score_in_model and not motif_score_calibration_identity:
        raise ValueError(
            "motif scores require a frozen train-independent calibration identity"
        )
    unknown_events = sorted(
        set(event_routes["event_id"].astype(str))
        - set(physical_events["event_id"].astype(str))
    )
    if unknown_events:
        raise ValueError(f"routes reference unknown events: {unknown_events[:5]}")
    if event_routes["route_id"].astype(str).duplicated().any():
        raise ValueError("EventRouteTable route IDs must be unique")

    orientation_policy = {"DNA": False, "RNA": False}
    if orientation_interaction_policy is not None:
        unknown = set(orientation_interaction_policy) - {"DNA", "RNA"}
        if unknown:
            raise ValueError(f"unknown orientation interaction channels: {sorted(unknown)}")
        orientation_policy.update(
            {key: bool(value) for key, value in orientation_interaction_policy.items()}
        )

    joined = event_routes.merge(
        physical_events,
        on=["event_id", "target_gene_id", "modality"],
        how="left",
        validate="many_to_one",
        suffixes=("_route", "_event"),
    )
    joined = _route_context_levels(joined, distance_bin_boundaries)
    modalities: dict[str, object] = {}
    for modality in ("DNA", "RNA"):
        subset = joined.loc[joined["modality"].astype(str) == modality]
        if subset.empty:
            modalities[modality] = {
                "factor_vocabulary": [],
                "interaction_factor_vocabulary": [],
                "base_categorical_fields": {},
                "context_fields": {},
                "padded_interaction_width": 0,
            }
            continue
        factor_levels = sorted(
            {
                (
                    "OPEN_ONLY"
                    if str(row.factor_identity_kind) == "accessibility_only"
                    else str(row.factor_entity_id)
                )
                for row in subset.itertuples(index=False)
            }
        )
        interaction_factors = [value for value in factor_levels if value != "OPEN_ONLY"]
        categorical_fields: dict[str, object] = {}
        for field in (
            "orientation",
            "geometry_kind",
            "region_type",
            "anchor_type",
            "transcript_oriented_side",
            "distance_bin",
        ):
            levels = sorted(
                {
                    str(value)
                    for value in subset[field]
                    if not _is_missing(value) and str(value) != "NA"
                }
            )
            categorical_fields[field] = {
                "raw_levels": levels,
                "reference_level": levels[0] if levels else None,
                "coding": "stable_reference",
            }
        context_fields = (
            ["region_type", "anchor_type", "distance_bin"]
            if modality == "DNA"
            else [
                "region_type",
                "transcript_oriented_side",
                "anchor_type",
                "distance_bin",
            ]
        )
        if orientation_policy[modality]:
            context_fields.append("orientation")
        context_manifest: dict[str, object] = {}
        for field in context_fields:
            levels = list(categorical_fields[field]["raw_levels"])
            declared_pairs = [
                [str(pair[0]), str(pair[1])]
                for pair in scientific_context_pairs.get(modality, {}).get(field, [])
            ]
            for pair in declared_pairs:
                if len(pair) != 2 or pair[0] == pair[1]:
                    raise ValueError("scientific context pairs require two distinct levels")
                if not set(pair).issubset(levels):
                    raise ValueError(
                        f"scientific context pair {modality}/{field}/{pair} "
                        "is outside the frozen raw vocabulary"
                    )
            p_max = max(0, (len(interaction_factors) - 1) * (len(levels) - 1))
            context_manifest[field] = {
                "raw_levels": levels,
                "scientific_context_pairs": declared_pairs,
                "p_max": p_max,
            }
        modalities[modality] = {
            "factor_vocabulary": factor_levels,
            "interaction_factor_vocabulary": interaction_factors,
            "base_categorical_fields": categorical_fields,
            "context_fields": context_manifest,
            "padded_interaction_width": int(
                sum(value["p_max"] for value in context_manifest.values())
            ),
        }

    manifest = {
        "schema_version": "FABRIC_V2_EVENT_FEATURE_MANIFEST_V1",
        "catalog_scope": "split_neutral",
        "factor_coding": "complete_one_hot_bias_free",
        "other_categorical_coding": "stable_reference",
        "event_fields": [
            "factor_identity",
            "calibrated_motif_score" if motif_score_in_model else None,
            "orientation",
            "dna_log1p_peak_support",
        ],
        "route_fields": [
            "geometry_kind",
            "region_type",
            "anchor_type",
            "transcript_oriented_side",
            "distance_bin",
            "signed_distance_bp_scaled",
            "edge_relative_position",
            "log1p_distance_to_5prime_boundary_bp",
            "log1p_distance_to_3prime_boundary_bp",
            "availability_masks",
        ],
        "prohibited_numeric_fields": [
            "event_id",
            "route_id",
            "target_gene_id",
            "edge_id",
            "peak_id",
            "motif_id",
            "motif_equivalence_family_id",
            "chromosome",
            "start",
            "end",
        ],
        "motif_score_in_model": bool(motif_score_in_model),
        "motif_score_calibration_identity": motif_score_calibration_identity,
        "orientation_interaction_policy": orientation_policy,
        "distance_bin_boundaries": {
            str(key): [float(value) for value in values]
            for key, values in distance_bin_boundaries.items()
        },
        "numeric_rank_audit": {
            "algorithm": "numpy_svd_deterministic_column_order",
            "tolerance": float(numeric_rank_tolerance),
            "column_scaling": "none",
        },
        "modalities": modalities,
    }
    manifest["event_feature_manifest_identity"] = _stable_identity(manifest)
    return manifest


def encode_base_route_features(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    manifest: Mapping[str, object],
) -> RouteBaseDesign:
    """Encode the fixed bias-free base design using a frozen manifest."""

    if manifest.get("schema_version") != "FABRIC_V2_EVENT_FEATURE_MANIFEST_V1":
        raise ValueError("unsupported EventFeatureManifest schema")
    _require_columns(physical_events, set(PHYSICAL_EVENT_COLUMNS), "PhysicalEventTable")
    _require_columns(event_routes, set(EVENT_ROUTE_COLUMNS), "EventRouteTable")
    routes = event_routes.sort_values("route_id", kind="mergesort").reset_index(drop=True)
    joined = routes.merge(
        physical_events,
        on=["event_id", "target_gene_id", "modality"],
        how="left",
        validate="many_to_one",
        suffixes=("_route", "_event"),
        indicator=True,
    )
    if bool((joined["_merge"] != "both").any()):
        raise ValueError("EventRouteTable contains an event absent from PhysicalEventTable")
    joined = joined.drop(columns="_merge")
    joined = _route_context_levels(joined, manifest["distance_bin_boundaries"])
    factor_identity = np.where(
        joined["factor_identity_kind"].astype(str) == "accessibility_only",
        "OPEN_ONLY",
        joined["factor_entity_id"].astype(str),
    )
    joined["interaction_factor_id"] = factor_identity

    modality_blocks: list[np.ndarray] = []
    column_names: list[str] = []
    # Keep one audit matrix with explicit modality prefixes.  Production later
    # selects only the matching prefix because W_D and W_R are independent.
    for modality in ("DNA", "RNA"):
        config = manifest["modalities"][modality]
        selector = joined["modality"].astype(str).to_numpy() == modality
        for factor in config["factor_vocabulary"]:
            modality_blocks.append(
                (selector & (factor_identity == factor)).astype(np.float64)[:, None]
            )
            column_names.append(f"{modality}:factor={factor}")
        for field, coding in config["base_categorical_fields"].items():
            reference = coding["reference_level"]
            for level in coding["raw_levels"]:
                if level == reference:
                    continue
                modality_blocks.append(
                    (
                        selector
                        & (joined[field].astype(str).to_numpy() == str(level))
                    ).astype(np.float64)[:, None]
                )
                column_names.append(f"{modality}:{field}={level}")
        continuous: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        if bool(manifest["motif_score_in_model"]):
            if "calibrated_motif_quality" not in joined:
                raise ValueError("calibrated motif score is absent from event catalog")
            score = joined["calibrated_motif_quality"].to_numpy(dtype=np.float64)
            factor_specific = joined["factor_identity_kind"].astype(str).to_numpy() != "accessibility_only"
            if not np.isfinite(score[selector & factor_specific]).all():
                raise ValueError("model motif scores must be fully calibrated and finite")
            continuous.append(("calibrated_motif_score", np.nan_to_num(score), None))
        peak_support = joined["peak_support"].to_numpy(dtype=np.float64)
        if modality == "DNA":
            if not np.isfinite(peak_support[selector]).all() or bool((peak_support[selector] < 0).any()):
                raise ValueError("DNA peak support must be finite and non-negative")
            continuous.append(("log1p_peak_support", np.log1p(np.nan_to_num(peak_support)), None))
        signed = joined["signed_distance_bp"].to_numpy(dtype=np.float64)
        if np.isinf(signed).any():
            raise ValueError("signed route distance may be finite or NaN, never infinite")
        signed_available = np.isfinite(signed)
        edge_relative = joined["edge_relative_position"].to_numpy(dtype=np.float64)
        d5_values = joined["distance_to_5prime_boundary_bp"].to_numpy(dtype=np.float64)
        d3_values = joined["distance_to_3prime_boundary_bp"].to_numpy(dtype=np.float64)
        if np.isinf(edge_relative).any() or np.isinf(d5_values).any() or np.isinf(d3_values).any():
            raise ValueError("route geometry values may be finite or NaN, never infinite")
        boundaries = list(manifest["distance_bin_boundaries"].get(modality, []))
        scale = max([abs(float(value)) for value in boundaries] + [1.0])
        continuous.extend(
            [
                ("signed_distance_bp_scaled", np.nan_to_num(signed / scale), signed_available),
                (
                    "edge_relative_position",
                    np.nan_to_num(edge_relative),
                    np.isfinite(edge_relative),
                ),
                (
                    "log1p_distance_to_5prime_boundary_bp",
                    np.log1p(
                        np.nan_to_num(
                            d5_values
                        )
                    ),
                    np.isfinite(
                        d5_values
                    ),
                ),
                (
                    "log1p_distance_to_3prime_boundary_bp",
                    np.log1p(
                        np.nan_to_num(
                            d3_values
                        )
                    ),
                    np.isfinite(
                        d3_values
                    ),
                ),
            ]
        )
        for name, values, available in continuous:
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite numeric base feature: {name}")
            encoded = selector.astype(np.float64) * values
            if bool(np.any(encoded != 0)):
                modality_blocks.append(encoded[:, None])
                column_names.append(f"{modality}:{name}")
            # A constant availability mask either adds no information or is
            # exactly represented by the complete factor baseline.  Keep its
            # semantic definition in the manifest but do not materialize a
            # zero/dependent numeric column.
            if available is not None and bool(available[selector].any()) and not bool(
                available[selector].all()
            ):
                modality_blocks.append((selector & available).astype(np.float64)[:, None])
                column_names.append(f"{modality}:{name}:available")

    values = (
        np.concatenate(modality_blocks, axis=1)
        if modality_blocks
        else np.zeros((len(joined), 0), dtype=np.float64)
    )
    if not np.isfinite(values).all():
        raise ValueError("base route design contains non-finite values")
    # A globally zero column may be valid only when a modality has no routes;
    # such columns are never materialized because the vocabulary is empty.
    zero = np.flatnonzero(np.all(values == 0, axis=0))
    if len(zero):
        raise ValueError(
            "base route design contains zero columns: "
            + ", ".join(column_names[index] for index in zero[:8])
        )
    duplicates = _exact_duplicate_columns(values)
    if duplicates:
        left, right = duplicates[0]
        raise ValueError(
            f"base route design contains exact duplicate columns: "
            f"{column_names[left]} and {column_names[right]}"
        )
    rank_audit: dict[str, object] = {}
    tolerance = float(manifest["numeric_rank_audit"]["tolerance"])
    for modality in ("DNA", "RNA"):
        rows = joined["modality"].astype(str).to_numpy() == modality
        columns = np.asarray(
            [name.startswith(f"{modality}:") for name in column_names], dtype=bool
        )
        matrix = values[np.ix_(rows, columns)]
        rank, singular_values = _numeric_rank(matrix, tolerance)
        rank_audit[modality] = {
            "row_count": int(matrix.shape[0]),
            "column_count": int(matrix.shape[1]),
            "rank": int(rank),
            "singular_values": singular_values.tolist(),
            "base_full_column_rank": bool(rank == matrix.shape[1]),
        }
        if matrix.shape[1] and rank != matrix.shape[1]:
            raise ValueError(f"{modality} base route design is rank deficient")

    route_context = joined[
        [
            "route_id",
            "event_id",
            "target_gene_id",
            "modality",
            "gate_key_id",
            "factor_identity_kind",
            "interaction_factor_id",
            "region_type",
            "anchor_type",
            "transcript_oriented_side",
            "distance_bin",
            "orientation",
        ]
    ].copy()
    output_manifest = dict(manifest)
    output_manifest["base_column_names"] = column_names
    output_manifest["base_rank_audit"] = rank_audit
    output_manifest["encoded_design_identity"] = _stable_array_identity(values)
    return RouteBaseDesign(
        route_ids=tuple(joined["route_id"].astype(str)),
        values=values.astype(np.float32),
        column_names=tuple(column_names),
        manifest=output_manifest,
        route_context=route_context,
    )


def measure_raw_interaction_support(
    route_base: RouteBaseDesign,
    physical_events: pd.DataFrame,
    gate_values: GateValues,
    *,
    train_mask: Sequence[bool],
    informative_molecule_mass_by_gene: pd.DataFrame,
    thresholds_by_channel: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Measure train-only support in reference-independent raw biological cells.

    Informative molecule mass is counted at most once for each ``(cell,gene)``
    raw-cell membership, even when multiple routes/events/gates instantiate the
    same biological cell.
    """

    _require_columns(
        physical_events,
        {"event_id", "model_active"},
        "PhysicalEventTable for interaction support",
    )
    active = set(
        physical_events.loc[physical_events["model_active"].astype(bool), "event_id"].astype(str)
    )
    train = np.asarray(train_mask, dtype=bool)
    if train.shape != (len(gate_values.cell_ids),):
        raise ValueError("interaction-support train mask differs from gate cell axis")
    _require_columns(
        informative_molecule_mass_by_gene,
        {"cell_id", "target_gene_id", "informative_molecule_mass"},
        "cell-gene informative mass",
    )
    if informative_molecule_mass_by_gene.duplicated(["cell_id", "target_gene_id"]).any():
        raise ValueError("cell-gene informative mass must have unique identities")
    masses = informative_molecule_mass_by_gene.copy()
    masses["informative_molecule_mass"] = masses["informative_molecule_mass"].astype(float)
    if not np.isfinite(masses["informative_molecule_mass"]).all() or bool(
        (masses["informative_molecule_mass"] < 0).any()
    ):
        raise ValueError("cell-gene informative mass must be finite and non-negative")
    mass_lookup = {
        (str(row.cell_id), str(row.target_gene_id)): float(row.informative_molecule_mass)
        for row in masses.itertuples(index=False)
    }
    gate_index = {value: index for index, value in enumerate(gate_values.gate_key_ids)}
    cell_index = {value: index for index, value in enumerate(gate_values.cell_ids)}
    context = route_base.route_context.loc[
        route_base.route_context["event_id"].astype(str).isin(active)
    ].copy()
    rows: list[dict[str, object]] = []
    for modality in ("DNA", "RNA"):
        if modality not in thresholds_by_channel:
            raise ValueError(f"interaction thresholds are absent for channel {modality}")
        thresholds = thresholds_by_channel[modality]
        required = {
            "minimum_distinct_events",
            "minimum_distinct_genes",
            "minimum_distinct_gate_keys",
            "minimum_informative_molecules",
        }
        if required - set(thresholds):
            raise ValueError(
                f"interaction thresholds for {modality} miss "
                f"{sorted(required - set(thresholds))}"
            )
        threshold_values = {name: float(thresholds[name]) for name in required}
        if not np.isfinite(list(threshold_values.values())).all() or any(
            value < 0 for value in threshold_values.values()
        ):
            raise ValueError(
                f"interaction thresholds for {modality} must be finite and non-negative"
            )
        for count_name in (
            "minimum_distinct_events",
            "minimum_distinct_genes",
            "minimum_distinct_gate_keys",
        ):
            if not threshold_values[count_name].is_integer():
                raise ValueError(f"{count_name} must be an integer count")
        config = route_base.manifest["modalities"][modality]
        factors = list(config["interaction_factor_vocabulary"])
        modality_context = context.loc[context["modality"].astype(str) == modality]
        for field, field_config in config["context_fields"].items():
            for factor, level in itertools.product(factors, field_config["raw_levels"]):
                selected = modality_context.loc[
                    (modality_context["interaction_factor_id"].astype(str) == str(factor))
                    & (modality_context[field].astype(str) == str(level))
                ]
                event_ids = sorted(set(selected["event_id"].astype(str)))
                gene_ids = sorted(set(selected["target_gene_id"].astype(str)))
                gate_ids = sorted(set(selected["gate_key_id"].dropna().astype(str)))
                missing_gate_ids = sorted(set(gate_ids) - set(gate_index))
                if missing_gate_ids:
                    raise ValueError(
                        f"active interaction cells reference missing gates: {missing_gate_ids[:5]}"
                    )
                informative_mass = 0.0
                supported_cell_gene_pairs = 0
                for gene_id, gene_rows in selected.groupby("target_gene_id", sort=False):
                    gene_gate_ids = sorted(
                        set(gene_rows["gate_key_id"].dropna().astype(str))
                    )
                    gene_gate_columns = [gate_index[value] for value in gene_gate_ids]
                    for cell_id, cell_column in cell_index.items():
                        if not train[cell_column]:
                            continue
                        mass = mass_lookup.get((cell_id, str(gene_id)), 0.0)
                        if mass <= 0:
                            continue
                        observed = bool(
                            gene_gate_columns
                            and gate_values.observed[cell_column, gene_gate_columns].any()
                        )
                        if observed:
                            informative_mass += mass
                            supported_cell_gene_pairs += 1
                failures: list[str] = []
                if len(event_ids) < float(thresholds["minimum_distinct_events"]):
                    failures.append("insufficient_distinct_events")
                if len(gene_ids) < float(thresholds["minimum_distinct_genes"]):
                    failures.append("insufficient_distinct_genes")
                if len(gate_ids) < float(thresholds["minimum_distinct_gate_keys"]):
                    failures.append("insufficient_distinct_gate_keys")
                if informative_mass < float(thresholds["minimum_informative_molecules"]):
                    failures.append("insufficient_informative_molecules")
                rows.append(
                    {
                        "modality": modality,
                        "context_field": field,
                        "factor_entity_id": str(factor),
                        "context_level": str(level),
                        "distinct_physical_event_count": len(event_ids),
                        "distinct_target_gene_count": len(gene_ids),
                        "distinct_active_gate_key_count": len(gate_ids),
                        "informative_molecule_mass": float(informative_mass),
                        "supported_cell_gene_pair_count": supported_cell_gene_pairs,
                        "event_ids": event_ids,
                        "target_gene_ids": gene_ids,
                        "gate_key_ids": gate_ids,
                        "raw_cell_supported": not failures,
                        "support_failure_reasons": failures,
                        "support_population": "train_only_after_cap_gate_event_admission",
                    }
                )
    result = pd.DataFrame(rows)
    result.attrs["train_cell_axis_identity"] = _stable_identity(
        [cell_id for cell_id, selected in zip(gate_values.cell_ids, train) if selected]
    )
    result.attrs["support_identity"] = _stable_identity(result.to_dict("records"))
    return result


def build_canonical_interaction_design(
    route_base: RouteBaseDesign,
    raw_support: pd.DataFrame,
) -> InteractionDesign:
    """Build the canonical supported-rectangle basis and raw claim table."""

    _require_columns(
        raw_support,
        {
            "modality",
            "context_field",
            "factor_entity_id",
            "context_level",
            "raw_cell_supported",
        },
        "raw interaction support",
    )
    if raw_support.duplicated(
        ["modality", "context_field", "factor_entity_id", "context_level"]
    ).any():
        raise ValueError("raw interaction support has duplicate biological cells")
    route_context = route_base.route_context.reset_index(drop=True)
    if tuple(route_context["route_id"].astype(str)) != route_base.route_ids:
        raise ValueError("route context order differs from base route order")
    tolerance = float(route_base.manifest["numeric_rank_audit"]["tolerance"])
    values_by_modality: dict[str, np.ndarray] = {}
    masks_by_modality: dict[str, np.ndarray] = {}
    route_indices_by_modality: dict[str, np.ndarray] = {}
    modality_manifests: dict[str, object] = {}
    contrast_rows: list[dict[str, object]] = []

    support_lookup = {
        (
            str(row.modality),
            str(row.context_field),
            str(row.factor_entity_id),
            str(row.context_level),
        ): bool(row.raw_cell_supported)
        for row in raw_support.itertuples(index=False)
    }

    for modality in ("DNA", "RNA"):
        config = route_base.manifest["modalities"][modality]
        factors = list(config["interaction_factor_vocabulary"])
        row_indices = np.flatnonzero(
            route_context["modality"].astype(str).to_numpy() == modality
        )
        route_indices_by_modality[modality] = row_indices.astype(np.int64)
        modality_context = route_context.iloc[row_indices].reset_index(drop=True)
        base_columns = np.asarray(
            [name.startswith(f"{modality}:") for name in route_base.column_names],
            dtype=bool,
        )
        base_matrix = route_base.values[np.ix_(row_indices, base_columns)].astype(np.float64)
        base_rank, base_singular = _numeric_rank(base_matrix, tolerance)
        if base_rank != base_matrix.shape[1]:
            raise ValueError(f"{modality} base block is not full rank")

        padded_width = int(config["padded_interaction_width"])
        padded = np.zeros((len(row_indices), padded_width), dtype=np.float64)
        active_mask = np.zeros(padded_width, dtype=bool)
        field_manifests: dict[str, object] = {}
        supported_route_columns: dict[str, np.ndarray] = {}
        field_raw_active: dict[str, tuple[np.ndarray, list[tuple[str, str]]]] = {}
        candidate_padded_columns: list[tuple[str, int, int, np.ndarray]] = []
        offset = 0
        for field, field_config in config["context_fields"].items():
            levels = list(field_config["raw_levels"])
            raw_cells = [(factor, level) for factor in factors for level in levels]
            raw_index = {value: index for index, value in enumerate(raw_cells)}
            supported_cells = {
                (factor, level)
                for factor, level in raw_cells
                if support_lookup.get((modality, field, factor, level), False)
            }
            potential_rectangles: list[dict[str, object]] = []
            supported_vectors: list[np.ndarray] = []
            pivot_rectangles: list[dict[str, object]] = []
            exact_matrix = np.zeros((len(raw_cells), 0), dtype=np.int64)
            exact_rank = 0
            for factor_left, factor_right in itertools.combinations(factors, 2):
                for level_left, level_right in itertools.combinations(levels, 2):
                    rectangle = {
                        "factor_pair": [factor_left, factor_right],
                        "context_pair": [level_left, level_right],
                    }
                    four_cells = {
                        (factor_left, level_left),
                        (factor_left, level_right),
                        (factor_right, level_left),
                        (factor_right, level_right),
                    }
                    four_corner = four_cells.issubset(supported_cells)
                    rectangle["four_corner_supported"] = four_corner
                    rectangle["selected_as_canonical_pivot"] = False
                    if four_corner:
                        vector = np.zeros(len(raw_cells), dtype=np.int64)
                        vector[raw_index[(factor_left, level_left)]] = 1
                        vector[raw_index[(factor_left, level_right)]] = -1
                        vector[raw_index[(factor_right, level_left)]] = -1
                        vector[raw_index[(factor_right, level_right)]] = 1
                        trial = np.column_stack([exact_matrix, vector])
                        trial_rank = _exact_rank(trial)
                        if trial_rank > exact_rank:
                            rectangle["selected_as_canonical_pivot"] = True
                            exact_matrix = trial
                            exact_rank = trial_rank
                            supported_vectors.append(vector)
                            pivot_rectangles.append(dict(rectangle))
                    potential_rectangles.append(rectangle)
            support_matrix = (
                np.column_stack(supported_vectors).astype(np.int64)
                if supported_vectors
                else np.zeros((len(raw_cells), 0), dtype=np.int64)
            )
            p_max = int(field_config["p_max"])
            if support_matrix.shape[1] > p_max:
                raise RuntimeError("canonical support rank exceeds fixed p_max")
            route_support = np.zeros((len(row_indices), support_matrix.shape[1]), dtype=np.float64)
            for route_row, route in enumerate(modality_context.itertuples(index=False)):
                raw_cell = (str(route.interaction_factor_id), str(getattr(route, field)))
                if raw_cell[0] == "OPEN_ONLY" or raw_cell[1] == "NA":
                    continue
                if raw_cell not in raw_index:
                    raise ValueError(
                        f"route raw interaction cell is outside manifest: {modality}/{field}/{raw_cell}"
                    )
                route_support[route_row] = support_matrix[raw_index[raw_cell]]
            supported_route_columns[field] = route_support
            field_raw_active[field] = (support_matrix, raw_cells)
            for support_column in range(support_matrix.shape[1]):
                candidate_padded_columns.append(
                    (
                        field,
                        support_column,
                        offset + support_column,
                        route_support[:, support_column],
                    )
                )
                padded[:, offset + support_column] = route_support[:, support_column]
            field_manifests[field] = {
                "raw_cell_order": [[factor, level] for factor, level in raw_cells],
                "supported_raw_cells": [list(value) for value in sorted(supported_cells)],
                "candidate_rectangles": potential_rectangles,
                "canonical_pivot_rectangles": pivot_rectangles,
                "H_support": support_matrix.tolist(),
                "N_raw_rectangles_potential": int(
                    math.comb(len(factors), 2) * math.comb(len(levels), 2)
                    if len(factors) >= 2 and len(levels) >= 2
                    else 0
                ),
                "N_four_corner_supported": int(
                    sum(row["four_corner_supported"] for row in potential_rectangles)
                ),
                "N_support_span": int(support_matrix.shape[1]),
                "N_rank_retained": 0,
                "N_padded": p_max,
                "padded_offset": offset,
                "active_support_column_indices": [],
                "column_closure_reasons": [],
                "basis_coverage": (
                    "not_applicable_no_supported_rectangle"
                    if support_matrix.shape[1] == 0
                    else "zero"
                ),
            }
            offset += p_max
        if offset != padded_width:
            raise RuntimeError("interaction padded segment widths do not close")

        current = base_matrix.copy()
        current_rank = base_rank
        for field, support_column, padded_column, route_column in candidate_padded_columns:
            reason = "retained"
            if not bool(np.any(route_column != 0)):
                reason = "zero_on_final_route_design"
            else:
                trial = np.column_stack([current, route_column])
                trial_rank, _ = _numeric_rank(trial, tolerance)
                if trial_rank > current_rank:
                    active_mask[padded_column] = True
                    current = trial
                    current_rank = trial_rank
                    field_manifests[field]["active_support_column_indices"].append(
                        support_column
                    )
                else:
                    reason = "combined_design_rank_redundant"
            field_manifests[field]["column_closure_reasons"].append(reason)
        for field, details in field_manifests.items():
            retained_count = len(details["active_support_column_indices"])
            support_count = int(details["N_support_span"])
            details["N_rank_retained"] = retained_count
            if support_count:
                details["basis_coverage"] = (
                    "full"
                    if retained_count == support_count
                    else "partial" if retained_count else "zero"
                )
            support_matrix, raw_cells = field_raw_active[field]
            active_columns = details["active_support_column_indices"]
            details["H_active"] = support_matrix[:, active_columns].tolist()

        active_design = np.column_stack([base_matrix, padded[:, active_mask]])
        duplicate_columns = _exact_duplicate_columns(active_design)
        active_rank, singular = _numeric_rank(active_design, tolerance)
        if duplicate_columns or active_rank != active_design.shape[1]:
            raise RuntimeError("final base/interaction design failed rank closure")
        values_by_modality[modality] = padded.astype(np.float32)
        masks_by_modality[modality] = active_mask
        modality_manifests[modality] = {
            "padded_width": padded_width,
            "active_width": int(active_mask.sum()),
            "fields": field_manifests,
            "combined_rank_audit": {
                "base_rank": base_rank,
                "base_singular_values": base_singular.tolist(),
                "final_rank": active_rank,
                "final_column_count": int(active_design.shape[1]),
                "singular_values": singular.tolist(),
                "zero_column_count": 0,
                "exact_duplicate_count": 0,
                "tolerance": tolerance,
            },
        }

        contrast_rows.extend(
            _raw_interaction_contrasts(
                modality=modality,
                config=config,
                modality_context=modality_context,
                base_matrix=base_matrix,
                supported_route_columns=supported_route_columns,
                field_raw_active=field_raw_active,
                field_manifests=field_manifests,
                supported_lookup=support_lookup,
                tolerance=tolerance,
            )
        )

    support_manifest = {
        "schema_version": "FABRIC_V2_INTERACTION_SUPPORT_MANIFEST_V1",
        "support_population": "train_only_after_cap_gate_event_admission",
        "event_feature_manifest_identity": route_base.manifest.get(
            "event_feature_manifest_identity"
        ),
        "raw_support_identity": raw_support.attrs.get(
            "support_identity", _stable_identity(raw_support.to_dict("records"))
        ),
        "modalities": modality_manifests,
        "validation_test_may_activate_columns": False,
    }
    support_manifest["interaction_support_manifest_identity"] = _stable_identity(
        support_manifest
    )
    return InteractionDesign(
        route_ids=route_base.route_ids,
        values_by_modality=values_by_modality,
        active_mask_by_modality=masks_by_modality,
        route_indices_by_modality=route_indices_by_modality,
        raw_support=raw_support.copy(),
        manifest=support_manifest,
        raw_contrasts=pd.DataFrame(contrast_rows),
    )


def build_production_modality_tensors(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    route_base: RouteBaseDesign,
    interaction: InteractionDesign,
    gate_values: GateValues,
    *,
    target_gene_id: str,
    modality: str,
    ordered_edge_ids: Sequence[str],
) -> ProductionModalityTensors:
    """Materialize one gene/modality production tensor block after admission."""

    modality = str(modality).upper()
    if modality not in {"DNA", "RNA"}:
        raise ValueError("production modality must be DNA or RNA")
    edges = _unique_ids(ordered_edge_ids, "gene edge")
    events = physical_events.loc[
        (physical_events["target_gene_id"].astype(str) == str(target_gene_id))
        & (physical_events["modality"].astype(str) == modality)
        & physical_events["model_active"].astype(bool)
    ].sort_values("event_id", kind="mergesort")
    event_ids = tuple(events["event_id"].astype(str))
    if not event_ids:
        interaction_width = int(
            route_base.manifest["modalities"][modality]["padded_interaction_width"]
        )
        base_width = sum(
            name.startswith(f"{modality}:") for name in route_base.column_names
        )
        return ProductionModalityTensors(
            cell_ids=gate_values.cell_ids,
            target_gene_id=str(target_gene_id),
            modality=modality,
            ordered_edge_ids=edges,
            event_ids=(),
            gate_key_ids=(),
            route_ids=(),
            route_event_index=np.zeros(0, dtype=np.int64),
            route_edge_index=np.zeros(0, dtype=np.int64),
            route_weight=np.zeros(0, dtype=np.float32),
            route_base_features=np.zeros((0, base_width), dtype=np.float32),
            route_interaction_features=np.zeros((0, interaction_width), dtype=np.float32),
            interaction_active_mask=interaction.active_mask_by_modality[modality].copy(),
            event_gate_key_index=np.zeros(0, dtype=np.int64),
            gate=np.zeros((len(gate_values.cell_ids), 0), dtype=np.float32),
        )
    if events["gate_key_id"].isna().any():
        raise ValueError("model-active events require gate keys")
    event_index = {value: index for index, value in enumerate(event_ids)}
    routes = event_routes.loc[
        event_routes["event_id"].astype(str).isin(event_ids)
    ].sort_values("route_id", kind="mergesort")
    if set(routes["event_id"].astype(str)) != set(event_ids):
        raise ValueError("every model-active event must retain at least one route")
    route_ids = tuple(routes["route_id"].astype(str))
    base_index = {value: index for index, value in enumerate(route_base.route_ids)}
    missing_base = sorted(set(route_ids) - set(base_index))
    if missing_base:
        raise ValueError(f"production routes are absent from base design: {missing_base[:5]}")
    modality_route_ids = [
        route_base.route_ids[index]
        for index in interaction.route_indices_by_modality[modality]
    ]
    interaction_index = {value: index for index, value in enumerate(modality_route_ids)}
    missing_interaction = sorted(set(route_ids) - set(interaction_index))
    if missing_interaction:
        raise ValueError(
            f"production routes are absent from interaction design: {missing_interaction[:5]}"
        )
    edge_index = {value: index for index, value in enumerate(edges)}
    missing_edges = sorted(set(routes["edge_id"].astype(str)) - set(edge_index))
    if missing_edges:
        raise ValueError(f"production routes reference absent gene edges: {missing_edges[:5]}")
    weights = routes["route_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("production route weights must be finite and positive")
    sums = routes.groupby("event_id", sort=False)["route_weight"].sum()
    if not np.allclose(sums.to_numpy(float), 1.0, atol=1e-12, rtol=0):
        raise ValueError("production route weights must sum to one per event")

    gate_key_ids = tuple(sorted(set(events["gate_key_id"].astype(str))))
    source_gate_index = {value: index for index, value in enumerate(gate_values.gate_key_ids)}
    missing_gates = sorted(set(gate_key_ids) - set(source_gate_index))
    if missing_gates:
        raise ValueError(f"production events reference absent gate values: {missing_gates[:5]}")
    local_gate_index = {value: index for index, value in enumerate(gate_key_ids)}
    base_columns = np.asarray(
        [name.startswith(f"{modality}:") for name in route_base.column_names], dtype=bool
    )
    base_features = np.asarray(
        [route_base.values[base_index[value], base_columns] for value in route_ids],
        dtype=np.float32,
    )
    expected_base_width = sum(
        name.startswith(f"{modality}:") for name in route_base.column_names
    )
    if base_features.shape != (len(route_ids), expected_base_width) or not np.isfinite(
        base_features
    ).all():
        raise ValueError("production modality base design violates its frozen column axis")
    return ProductionModalityTensors(
        cell_ids=gate_values.cell_ids,
        target_gene_id=str(target_gene_id),
        modality=modality,
        ordered_edge_ids=edges,
        event_ids=event_ids,
        gate_key_ids=gate_key_ids,
        route_ids=route_ids,
        route_event_index=np.asarray(
            [event_index[str(value)] for value in routes["event_id"]], dtype=np.int64
        ),
        route_edge_index=np.asarray(
            [edge_index[str(value)] for value in routes["edge_id"]], dtype=np.int64
        ),
        route_weight=weights.astype(np.float32),
        route_base_features=base_features,
        route_interaction_features=np.asarray(
            [
                interaction.values_by_modality[modality][interaction_index[value]]
                for value in route_ids
            ],
            dtype=np.float32,
        ),
        interaction_active_mask=interaction.active_mask_by_modality[modality].copy(),
        event_gate_key_index=np.asarray(
            [local_gate_index[str(value)] for value in events["gate_key_id"]],
            dtype=np.int64,
        ),
        gate=gate_values.gate[
            :, [source_gate_index[value] for value in gate_key_ids]
        ].astype(np.float32, copy=True),
    )


def build_model_injection_equivalence_index(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    route_base: RouteBaseDesign,
    interaction: InteractionDesign,
    *,
    ordered_edge_ids_by_gene: Mapping[str, Sequence[str]],
    physical_overlap_minimum_bp: int = 1,
    physical_overlap_minimum_reciprocal: float = 0.0,
) -> pd.DataFrame:
    """Group model-active events by their exact frozen per-edge injection signature."""

    if physical_overlap_minimum_bp <= 0 or not 0 <= physical_overlap_minimum_reciprocal <= 1:
        raise ValueError("physical collapse audit overlap rule is invalid")
    active_events = physical_events.loc[
        physical_events["model_active"].astype(bool)
    ].sort_values("event_id", kind="mergesort")
    route_by_id = event_routes.sort_values("route_id", kind="mergesort")
    base_index = {value: index for index, value in enumerate(route_base.route_ids)}
    interaction_index_by_modality: dict[str, dict[str, int]] = {}
    interaction_values_by_modality: dict[
        str, np.ndarray | sparse.csr_matrix
    ] = {}
    for modality in ("DNA", "RNA"):
        ids = [
            route_base.route_ids[index]
            for index in interaction.route_indices_by_modality[modality]
        ]
        interaction_index_by_modality[modality] = {
            value: index for index, value in enumerate(ids)
        }
        values = interaction.values_by_modality[modality]
        interaction_values_by_modality[modality] = (
            values.tocsr() if sparse.issparse(values) else np.asarray(values)
        )
    signatures: dict[tuple[object, ...], list[str]] = {}
    materialized: dict[str, dict[str, object]] = {}
    event_meta = active_events.set_index(active_events["event_id"].astype(str))
    for event in active_events.itertuples(index=False):
        event_id = str(event.event_id)
        gene_id = str(event.target_gene_id)
        modality = str(event.modality)
        if gene_id not in ordered_edge_ids_by_gene:
            raise ValueError(f"complete stable edge axis is absent for gene {gene_id}")
        edges = _unique_ids(ordered_edge_ids_by_gene[gene_id], f"{gene_id} edge")
        event_rows = route_by_id.loc[
            route_by_id["event_id"].astype(str) == event_id
        ].sort_values("route_id", kind="mergesort")
        if event_rows.empty:
            raise ValueError("model-active event has no retained production route")
        base_columns = np.asarray(
            [name.startswith(f"{modality}:") for name in route_base.column_names],
            dtype=bool,
        )
        beta: dict[str, dict[int, float]] = {edge: {} for edge in edges}
        interaction_values = interaction_values_by_modality[modality]
        int_width = int(interaction_values.shape[1])
        iota: dict[str, dict[int, float]] = {edge: {} for edge in edges}
        edge_set = set(edges)
        modality_interaction_index = interaction_index_by_modality[modality]
        active_mask = interaction.active_mask_by_modality[modality].astype(bool)
        if active_mask.shape != (int_width,):
            raise ValueError("interaction active mask differs from padded width")

        def accumulate(
            target: dict[int, float],
            indices: np.ndarray,
            values: np.ndarray,
            weight: float,
        ) -> None:
            for column, raw_value in zip(indices, values, strict=True):
                value = target.get(int(column), 0.0) + weight * float(raw_value)
                if value == 0:
                    target.pop(int(column), None)
                else:
                    target[int(column)] = value

        for route in event_rows.itertuples(index=False):
            route_id = str(route.route_id)
            edge_id = str(route.edge_id)
            if edge_id not in edge_set:
                raise ValueError("event route is outside the complete stable edge axis")
            if route_id not in base_index or route_id not in modality_interaction_index:
                raise ValueError("event route is absent from frozen encoded designs")
            weight = float(route.route_weight)
            base_row = route_base.values[
                base_index[route_id], base_columns
            ].astype(np.float64)
            base_nonzero = np.flatnonzero(base_row)
            accumulate(
                beta[edge_id],
                base_nonzero,
                base_row[base_nonzero],
                weight,
            )
            interaction_row_index = modality_interaction_index[route_id]
            if sparse.issparse(interaction_values):
                interaction_row = interaction_values.getrow(interaction_row_index)
                keep = active_mask[interaction_row.indices] & (
                    interaction_row.data != 0
                )
                interaction_indices = interaction_row.indices[keep]
                interaction_data = interaction_row.data[keep]
            else:
                interaction_row = np.asarray(
                    interaction_values[interaction_row_index], dtype=np.float64
                )
                interaction_indices = np.flatnonzero(active_mask & (interaction_row != 0))
                interaction_data = interaction_row[interaction_indices]
            accumulate(
                iota[edge_id],
                interaction_indices,
                interaction_data,
                weight,
            )

        def canonical_sparse(
            values: Mapping[int, float],
        ) -> tuple[tuple[int, float], ...]:
            return tuple(
                (int(column), 0.0 if value == 0 else float(value))
                for column, value in sorted(values.items())
                if value != 0
            )

        beta_values = [canonical_sparse(beta[edge]) for edge in edges]
        iota_values = [canonical_sparse(iota[edge]) for edge in edges]
        signature = (
            gene_id,
            modality,
            str(event.gate_key_id),
            tuple(
                (edge, tuple(beta_value), tuple(iota_value))
                for edge, beta_value, iota_value in zip(edges, beta_values, iota_values)
            ),
        )
        signatures.setdefault(signature, []).append(event_id)
        materialized[event_id] = {
            "target_gene_id": gene_id,
            "modality": modality,
            "gate_key_id": str(event.gate_key_id),
            "ordered_edge_ids": list(edges),
            "base_width": int(base_columns.sum()),
            "interaction_padded_width": int_width,
            "per_edge_beta_base_sparse": [
                [
                    {"column_index": column, "value": value}
                    for column, value in vector
                ]
                for vector in beta_values
            ],
            "per_edge_iota_masked_interaction_sparse": [
                [
                    {"column_index": column, "value": value}
                    for column, value in vector
                ]
                for vector in iota_values
            ],
            "member_route_ids": list(event_rows["route_id"].astype(str)),
            "anchor_region_ids": sorted(set(event_rows["anchor_region_id"].astype(str))),
        }
    rows: list[dict[str, object]] = []
    for signature, members_unsorted in sorted(
        signatures.items(), key=lambda value: repr(value[0])
    ):
        members = sorted(members_unsorted)
        first = materialized[members[0]]
        member_rows = event_meta.loc[members]
        collapse_error = _should_have_physically_collapsed(
            member_rows,
            minimum_overlap_bp=physical_overlap_minimum_bp,
            minimum_reciprocal_overlap=physical_overlap_minimum_reciprocal,
        )
        group_id = "injection:" + hashlib.sha256(
            _canonical_json(
                {
                    "target_gene_id": first["target_gene_id"],
                    "modality": first["modality"],
                    "gate_key_id": first["gate_key_id"],
                    "members": members,
                    "signature": [
                        [
                            edge,
                            first["per_edge_beta_base_sparse"][index],
                            first[
                                "per_edge_iota_masked_interaction_sparse"
                            ][index],
                        ]
                        for index, edge in enumerate(first["ordered_edge_ids"])
                    ],
                }
            ).encode("utf-8")
        ).hexdigest()[:24]
        all_routes = sorted(
            {
                route_id
                for event_id in members
                for route_id in materialized[event_id]["member_route_ids"]
            }
        )
        all_anchors = sorted(
            {
                anchor
                for event_id in members
                for anchor in materialized[event_id]["anchor_region_ids"]
            }
        )
        rows.append(
            {
                "model_injection_group_id": group_id,
                "target_gene_id": first["target_gene_id"],
                "modality": first["modality"],
                "gate_key_id": first["gate_key_id"],
                "member_event_ids": members,
                "member_count": len(members),
                "ordered_edge_ids": first["ordered_edge_ids"],
                "base_width": first["base_width"],
                "interaction_padded_width": first["interaction_padded_width"],
                "per_edge_beta_base_sparse": first[
                    "per_edge_beta_base_sparse"
                ],
                "per_edge_iota_masked_interaction_sparse": first[
                    "per_edge_iota_masked_interaction_sparse"
                ],
                "member_route_ids": all_routes,
                "anchor_region_ids": all_anchors,
                "footprint_relation": _footprint_relation(member_rows),
                "motif_family_relation": _motif_family_relation(member_rows),
                "physical_collapse_status": (
                    "should_have_collapsed_error"
                    if collapse_error
                    else "correctly_distinct"
                ),
                "attribution_policy": (
                    "singleton_event_primary"
                    if len(members) == 1
                    else "exact_injection_set_primary"
                ),
                "signature_comparison": "exact_no_tolerance_no_rounding",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and bool(
        (result["physical_collapse_status"] == "should_have_collapsed_error").any()
    ):
        raise ValueError(
            "ModelInjectionEquivalenceIndex found physical rows that should have collapsed"
        )
    return result


def assemble_gene_cell_model_input(
    graph: object,
    *,
    cell_split: pd.DataFrame,
    normalized_cis_edges: pd.DataFrame,
    cis_feature_names: Sequence[str],
    dna: ProductionModalityTensors,
    rna: ProductionModalityTensors,
    compatibility_rows: pd.DataFrame,
) -> V2GeneAssembly:
    """Bind one gene's frozen graph, cell contexts, and compatible-EC rows.

    This is the sole V2 data-to-model assembly boundary.  It does not discover
    paths, routes, cells, or compatible sets and never intersects axes to make
    an input fit: any identity mismatch fails explicitly.
    """

    from .model import GeneCellModelInput, RoutedModalityInput

    required_graph = (
        "gene_id",
        "edge_ids",
        "path_ids",
        "local_edge_index",
        "path_edge_incidence",
        "path_first_edge_indices",
        "path_last_edge_indices",
        "path_log_edge_count",
    )
    missing_graph = [name for name in required_graph if not hasattr(graph, name)]
    if missing_graph:
        raise ValueError(f"gene graph misses V2 model fields: {missing_graph}")
    gene_id = str(graph.gene_id)
    edges = tuple(map(str, graph.edge_ids))
    paths = tuple(map(str, graph.path_ids))
    if not edges or not paths or len(edges) != len(set(edges)) or len(paths) != len(set(paths)):
        raise ValueError("gene graph requires non-empty unique edge/path axes")
    _require_columns(cell_split, {"cell_id", "split"}, "gene cell split")
    if cell_split["cell_id"].astype(str).duplicated().any():
        raise ValueError("gene cell split has duplicate cell IDs")
    cells = tuple(cell_split["cell_id"].astype(str))
    splits = tuple(cell_split["split"].astype(str))
    if not cells or not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("gene cell split is empty or has invalid split labels")
    if dna.cell_ids != cells or rna.cell_ids != cells:
        raise ValueError("DNA/RNA gate cell axes differ from the frozen gene cell axis")
    for label, values in (("DNA", dna), ("RNA", rna)):
        if values.target_gene_id != gene_id or values.modality != label:
            raise ValueError(f"{label} production tensors belong to another gene/modality")
        if values.ordered_edge_ids != edges:
            raise ValueError(f"{label} production tensor edge axis differs from the graph")

    feature_names = tuple(map(str, cis_feature_names))
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("CIS feature axis must be non-empty and unique")
    _require_columns(normalized_cis_edges, {"edge_id"} | set(feature_names), "normalized CIS edges")
    if normalized_cis_edges["edge_id"].astype(str).duplicated().any():
        raise ValueError("normalized CIS edges have duplicate edge IDs")
    cis = normalized_cis_edges.set_index(normalized_cis_edges["edge_id"].astype(str))
    if set(cis.index) != set(edges):
        raise ValueError("normalized CIS edge axis differs from the complete gene edge axis")
    cis_values = cis.loc[list(edges), list(feature_names)].to_numpy(dtype=np.float32)
    if not np.isfinite(cis_values).all():
        raise ValueError("normalized CIS model features must be finite")

    def routed(values: ProductionModalityTensors) -> RoutedModalityInput:
        route_event_index = torch.from_numpy(values.route_event_index.copy()).long()
        route_edge_index = torch.from_numpy(values.route_edge_index.copy()).long()
        route_weight = torch.from_numpy(values.route_weight.copy()).float()
        route_base = _route_feature_matrix_to_torch(values.route_base_features)
        route_interaction = _route_feature_matrix_to_torch(
            values.route_interaction_features
        )
        active_mask = torch.from_numpy(values.interaction_active_mask.copy()).bool()
        event_gate_index = torch.from_numpy(values.event_gate_key_index.copy()).long()
        gate = torch.from_numpy(values.gate.copy()).float()
        expected_routes = len(values.route_ids)
        if not (
            route_event_index.shape == route_edge_index.shape == route_weight.shape == (expected_routes,)
            and route_base.shape[0] == expected_routes
            and route_interaction.shape[0] == expected_routes
            and active_mask.shape == (route_interaction.shape[1],)
            and event_gate_index.shape == (len(values.event_ids),)
            and gate.shape == (len(cells), len(values.gate_key_ids))
        ):
            raise ValueError("production modality tensor axes are inconsistent")
        if expected_routes:
            if bool((route_event_index < 0).any()) or bool(
                (route_event_index >= len(values.event_ids)).any()
            ):
                raise ValueError("production route event index is out of range")
            if bool((route_edge_index < 0).any()) or bool(
                (route_edge_index >= len(edges)).any()
            ):
                raise ValueError("production route edge index is out of range")
        if event_gate_index.numel() and (
            bool((event_gate_index < 0).any())
            or bool((event_gate_index >= len(values.gate_key_ids)).any())
        ):
            raise ValueError("production event gate-key index is out of range")
        for tensor in (route_weight, route_base, route_interaction, gate):
            checked = (
                tensor
                if tensor.layout == torch.strided
                else tensor.to_sparse_coo().coalesce().values()
            )
            if not bool(torch.isfinite(checked).all()):
                raise ValueError("production modality tensor contains non-finite values")
        return RoutedModalityInput(
            route_event_index=route_event_index,
            route_edge_index=route_edge_index,
            route_weight=route_weight,
            route_base_features=route_base,
            route_interaction_features=route_interaction,
            interaction_active_mask=active_mask,
            event_gate_key_index=event_gate_index,
            gate=gate,
        )

    incidence = sparse.coo_matrix(graph.path_edge_incidence)
    indices = torch.tensor(
        np.vstack([incidence.row, incidence.col]), dtype=torch.long
    )
    path_incidence = torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(incidence.data.astype(np.float32)),
        size=incidence.shape,
    ).coalesce()
    if incidence.shape != (len(paths), len(edges)):
        raise ValueError("gene path-edge incidence differs from frozen axes")
    model_input = GeneCellModelInput(
        cis_features=torch.from_numpy(cis_values),
        local_edge_index=torch.from_numpy(
            np.asarray(graph.local_edge_index, dtype=np.int64).copy()
        ).long(),
        dna=routed(dna),
        rna=routed(rna),
        path_edge_incidence=path_incidence,
        path_first_edge_index=torch.tensor(
            graph.path_first_edge_indices, dtype=torch.long
        ),
        path_last_edge_index=torch.tensor(
            graph.path_last_edge_indices, dtype=torch.long
        ),
        log_edge_count=torch.tensor(graph.path_log_edge_count, dtype=torch.float32),
    )

    required_ec = {
        "cell_id",
        "target_gene_id",
        "split",
        "molecule_count",
        "compatible_path_ids",
        "pre_compatibility_qc_pass",
        "final_fate",
    }
    _require_columns(compatibility_rows, required_ec, "gene compatible EC rows")
    ec = compatibility_rows.copy().reset_index(drop=True)
    if "compatibility_class_id" in ec and ec["compatibility_class_id"].astype(str).duplicated().any():
        raise ValueError("compatible EC rows have duplicate compatibility_class_id")
    aggregate_identity = ec.apply(
        lambda row: _canonical_json(
            [
                str(row["cell_id"]),
                str(row["target_gene_id"]),
                str(row["split"]),
                bool(row["pre_compatibility_qc_pass"]),
                _string_list(row["compatible_path_ids"], "compatible_path_ids"),
                str(row["final_fate"]),
            ]
        ),
        axis=1,
    )
    if aggregate_identity.duplicated().any():
        raise ValueError("compatible EC rows contain an unaggregated duplicate identity")
    if set(ec["target_gene_id"].astype(str)) != {gene_id}:
        raise ValueError("compatible EC rows must belong to exactly the assembled gene")
    extra_cells = sorted(set(ec["cell_id"].astype(str)) - set(cells))
    if extra_cells:
        raise ValueError(f"compatible EC cells are absent from the frozen cell axis: {extra_cells[:5]}")
    split_lookup = dict(zip(cells, splits, strict=True))
    if any(
        str(row.split) != split_lookup[str(row.cell_id)]
        for row in ec.itertuples(index=False)
    ):
        raise ValueError("compatible EC split differs from the authoritative cell split")
    path_index = {value: index for index, value in enumerate(paths)}
    compatible_lists: list[list[int]] = []
    for value in ec["compatible_path_ids"]:
        ids = _string_list(value, "compatible_path_ids")
        if not _is_ordered_unique_subset(ids, paths):
            raise ValueError("compatible path IDs differ from the frozen path-axis order")
        compatible_lists.append([path_index[path_id] for path_id in ids])
    mass = ec["molecule_count"].to_numpy()
    if any(not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in mass):
        raise ValueError("compatible EC molecule mass must be positive integer")
    width = max(1, max((len(values) for values in compatible_lists), default=0))
    compatible_indices = np.zeros((len(ec), width), dtype=np.int64)
    compatible_mask = np.zeros((len(ec), width), dtype=bool)
    for row_index, values in enumerate(compatible_lists):
        compatible_indices[row_index, : len(values)] = values
        compatible_mask[row_index, : len(values)] = True
    fates = ec["final_fate"].astype(str).to_numpy()
    expected_informative = (
        ec["pre_compatibility_qc_pass"].astype(bool).to_numpy()
        & (np.asarray([len(values) for values in compatible_lists]) > 0)
        & (np.asarray([len(values) for values in compatible_lists]) < len(paths))
    )
    if not np.array_equal(fates == INFORMATIVE_FATE, expected_informative):
        raise ValueError("compatible EC final fate differs from the unique K^inf definition")
    cell_index = {value: index for index, value in enumerate(cells)}
    return V2GeneAssembly(
        gene_id=gene_id,
        model_input=model_input,
        compatible_path_indices=torch.from_numpy(compatible_indices).long(),
        compatible_path_mask=torch.from_numpy(compatible_mask).bool(),
        row_cell_index=torch.tensor(
            [cell_index[str(value)] for value in ec["cell_id"]], dtype=torch.long
        ),
        molecule_count=torch.from_numpy(mass.astype(np.float32)),
        informative_row_mask=torch.from_numpy(expected_informative).bool(),
        cell_ids=cells,
        cell_split=splits,
        path_ids=paths,
    )


def _route_feature_matrix_to_torch(
    values: np.ndarray | sparse.spmatrix,
) -> torch.Tensor:
    """Preserve the contracted sparse route design at the runtime boundary."""

    if sparse.issparse(values):
        matrix = sparse.coo_matrix(values, dtype=np.float32)
        indices = torch.from_numpy(
            np.vstack([matrix.row, matrix.col]).astype(np.int64, copy=False)
        ).long()
        return torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(matrix.data.astype(np.float32, copy=False)),
            size=matrix.shape,
        ).coalesce()
    array = np.asarray(values, dtype=np.float32)
    return torch.from_numpy(array.copy()).float()


def validate_compatibility_artifact(
    manifest: Mapping[str, object],
    ec_rows: pd.DataFrame,
    *,
    legal_paths_by_gene: Mapping[str, Sequence[str]],
    expected_candidate_gene_ids: Sequence[str],
    expected_candidate_gene_count: int,
) -> CompatibilityArtifactValidation:
    """Fail-closed validation of the external frozen compatible-EC delivery."""

    required_manifest = {
        "producer",
        "command",
        "code_version",
        "alignment_identity",
        "reference_identity",
        "matrix_observation_input_identity",
        "legal_path_catalog_identity",
        "cell_split_identity",
        "qc_policy",
        "compatibility_policy",
        "model_isoform_universe",
        "matrix_structural_path_count",
        "candidate_gene_ids",
        "candidate_support_status",
        "split_conservation",
        "train_policy_identity",
        "validation_policy_identity",
        "test_exposure",
        "run_counts",
        "artifact_complete",
        "G_fit_freeze_status",
        "test_rows_written",
        "training_authorized_or_started",
    }
    reasons: list[str] = []
    missing_manifest = sorted(required_manifest - set(manifest))
    if missing_manifest:
        reasons.append("missing_manifest_fields:" + ",".join(missing_manifest))
    candidates = _unique_ids(expected_candidate_gene_ids, "expected structural candidate")
    if len(candidates) != expected_candidate_gene_count:
        reasons.append("expected_candidate_count_contract_mismatch")
    manifest_candidates: tuple[str, ...] = ()
    if "candidate_gene_ids" in manifest:
        try:
            manifest_candidates = _unique_ids(
                manifest["candidate_gene_ids"], "manifest structural candidate"
            )
        except (TypeError, ValueError) as error:
            reasons.append(f"invalid_candidate_gene_ids:{error}")
    if len(manifest_candidates) == 7_198:
        reasons.append("historical_7198_artifact_forbidden")
    if set(manifest_candidates) != set(candidates):
        reasons.append("candidate_gene_identity_mismatch")
    nonempty_identity_policy_fields = (
        "producer",
        "command",
        "code_version",
        "alignment_identity",
        "reference_identity",
        "matrix_observation_input_identity",
        "legal_path_catalog_identity",
        "cell_split_identity",
        "qc_policy",
        "compatibility_policy",
        "train_policy_identity",
        "validation_policy_identity",
    )
    for field in nonempty_identity_policy_fields:
        value = manifest.get(field)
        if value is None or value == "" or value == {} or value == []:
            reasons.append(f"empty_manifest_identity_or_policy:{field}")
    if manifest.get("train_policy_identity") != manifest.get(
        "validation_policy_identity"
    ):
        reasons.append("train_validation_policy_drift")
    if manifest.get("artifact_complete") is not True:
        reasons.append("partial_compatibility_artifact_not_admissible")
    if manifest.get("G_fit_freeze_status") != "FROZEN_FROM_TRAIN_ONLY":
        reasons.append("G_fit_not_frozen_from_complete_train_only_mass")
    if manifest.get("test_rows_written") is not False:
        reasons.append("test_rows_written_before_checkpoint")
    if manifest.get("training_authorized_or_started") is not False:
        reasons.append("compatibility_build_must_not_start_or_authorize_training")
    run_counts = manifest.get("run_counts")
    if not isinstance(run_counts, Mapping):
        reasons.append("invalid_compatibility_run_counts")
    else:
        duplicate_count = run_counts.get(
            "duplicate_cell_gene_umi_primary_records"
        )
        if (
            isinstance(duplicate_count, bool)
            or not isinstance(duplicate_count, (int, np.integer))
            or int(duplicate_count) != 0
        ):
            reasons.append("duplicate_cell_gene_umi_primary_records_detected")
    if manifest.get("model_isoform_universe") != (
        "resolved_ont_matrix_structural_paths_only"
    ):
        reasons.append("model_isoform_universe_is_not_matrix_only")
    if not manifest.get("producer") or not manifest.get("command"):
        reasons.append("producer_invocation_not_frozen")
    exposure = manifest.get("test_exposure")
    if exposure not in {
        "not_materialized_before_checkpoint",
        "previously_materialized",
    }:
        reasons.append("invalid_test_exposure_marker")

    support_status = manifest.get("candidate_support_status", [])
    try:
        support_frame = pd.DataFrame(support_status)
        _require_columns(
            support_frame,
            {"target_gene_id", "support_status"},
            "candidate support status",
        )
        if support_frame["target_gene_id"].astype(str).duplicated().any():
            reasons.append("duplicate_candidate_support_status")
        if set(support_frame["target_gene_id"].astype(str)) != set(candidates):
            reasons.append("candidate_support_status_incomplete")
        status_values = support_frame["support_status"]
        if status_values.isna().any() or any(
            not str(value).strip() for value in status_values
        ):
            reasons.append("empty_candidate_support_status")
    except (TypeError, ValueError) as error:
        reasons.append(f"invalid_candidate_support_status:{error}")

    required_rows = {
        "cell_id",
        "target_gene_id",
        "split",
        "molecule_count",
        "pre_compatibility_qc_pass",
        "compatible_path_ids",
        "final_fate",
        "technical_reason_code",
    }
    if required_rows - set(ec_rows):
        reasons.append("missing_ec_fields:" + ",".join(sorted(required_rows - set(ec_rows))))
        return CompatibilityArtifactValidation(
            status="REJECTED",
            reasons=tuple(reasons),
            informative_gene_ids=(),
            audit=pd.DataFrame(),
            candidate_gene_ids=candidates,
            legal_path_catalog_identity=str(
                manifest.get("legal_path_catalog_identity") or ""
            ),
            cell_split_identity=str(manifest.get("cell_split_identity") or ""),
            test_exposure=str(manifest.get("test_exposure") or ""),
            model_isoform_universe=str(
                manifest.get("model_isoform_universe") or ""
            ),
            matrix_structural_path_count=(
                int(manifest["matrix_structural_path_count"])
                if type(manifest.get("matrix_structural_path_count")) is int
                else 0
            ),
        )
    rows = ec_rows.copy().reset_index(drop=True)
    invalid_splits = sorted(set(rows["split"].astype(str)) - {"train", "val", "test"})
    if invalid_splits:
        reasons.append("invalid_ec_split_labels:" + ",".join(invalid_splits))
    split_count_by_cell = rows.groupby(rows["cell_id"].astype(str))["split"].nunique()
    if bool((split_count_by_cell > 1).any()):
        reasons.append("cell_split_conflict")
    row_audit: list[dict[str, object]] = []
    legal_paths = {str(gene): tuple(map(str, paths)) for gene, paths in legal_paths_by_gene.items()}
    if set(candidates) != set(legal_paths):
        reasons.append("legal_path_candidate_identity_mismatch")
    matrix_path_count = sum(len(path_ids) for path_ids in legal_paths.values())
    if (
        type(manifest.get("matrix_structural_path_count")) is not int
        or manifest["matrix_structural_path_count"] != matrix_path_count
    ):
        reasons.append("matrix_structural_path_count_mismatch")
    for gene_id, path_ids in legal_paths.items():
        if not path_ids:
            reasons.append(f"empty_legal_path_catalog:{gene_id}")
        elif any(not path_id.strip() for path_id in path_ids):
            reasons.append(f"empty_legal_path_identity:{gene_id}")
        elif len(set(path_ids)) != len(path_ids):
            reasons.append(f"duplicate_legal_path_identity:{gene_id}")
    for row_index, row in rows.iterrows():
        gene_id = str(row["target_gene_id"])
        mass = row["molecule_count"]
        row_reasons: list[str] = []
        if gene_id not in set(candidates):
            row_reasons.append("gene_outside_candidate_catalog")
        if not isinstance(mass, (int, np.integer)) or int(mass) <= 0:
            row_reasons.append("non_positive_or_non_integer_ec_mass")
        paths = _string_list(row["compatible_path_ids"], "compatible_path_ids")
        gene_paths = legal_paths.get(gene_id, ())
        if not _is_ordered_unique_subset(paths, gene_paths):
            row_reasons.append("compatible_path_ids_not_in_frozen_axis_order")
        if not set(paths).issubset(gene_paths):
            row_reasons.append("compatible_path_outside_frozen_catalog")
        qc_pass = bool(row["pre_compatibility_qc_pass"])
        fate = str(row["final_fate"])
        if not qc_pass:
            expected_fate = "pre_compatibility_technical_qc_failure"
            if not str(row["technical_reason_code"]):
                row_reasons.append("technical_failure_requires_reason_code")
        elif not paths:
            expected_fate = EMPTY_FATE
        elif set(paths) == set(gene_paths) and gene_paths:
            expected_fate = FULL_FATE
        else:
            expected_fate = INFORMATIVE_FATE
        if fate != expected_fate:
            row_reasons.append("fate_not_reproducible")
        if qc_pass and str(row["technical_reason_code"]) not in {"", "NA", "None"}:
            row_reasons.append("technical_reason_code_on_qc_pass_row")
        if row_reasons:
            reasons.append(f"ec_row_{row_index}:" + ",".join(row_reasons))
        row_audit.append(
            {
                "row_index": row_index,
                "cell_id": str(row["cell_id"]),
                "target_gene_id": gene_id,
                "split": str(row["split"]),
                "molecule_count": int(mass) if isinstance(mass, (int, np.integer)) else mass,
                "reproduced_fate": expected_fate,
                "declared_fate": fate,
                "row_valid": not row_reasons,
                "failure_reasons": row_reasons,
            }
        )
    if not rows.empty:
        ec_identity = rows.apply(
            lambda row: _canonical_json(
                [
                    str(row["cell_id"]),
                    str(row["target_gene_id"]),
                    str(row["split"]),
                    bool(row["pre_compatibility_qc_pass"]),
                    _string_list(row["compatible_path_ids"], "compatible_path_ids"),
                    str(row["final_fate"]),
                    str(row["technical_reason_code"]),
                ]
            ),
            axis=1,
        )
        if ec_identity.duplicated().any():
            reasons.append("duplicate_aggregated_ec_identity")
    if exposure == "not_materialized_before_checkpoint" and bool(
        (rows["split"].astype(str) == "test").any()
    ):
        reasons.append("test_rows_materialized_despite_unexposed_marker")

    actual_split_conservation = _compatibility_split_conservation(rows)
    try:
        declared_split_conservation = pd.DataFrame(manifest.get("split_conservation", []))
        required_conservation_columns = set(actual_split_conservation.columns)
        _require_columns(
            declared_split_conservation,
            required_conservation_columns,
            "manifest split conservation",
        )
        if declared_split_conservation["split"].astype(str).duplicated().any():
            reasons.append("duplicate_manifest_split_conservation")
        declared = declared_split_conservation.sort_values("split", kind="mergesort")[
            list(actual_split_conservation.columns)
        ].reset_index(drop=True)
        actual = actual_split_conservation.sort_values("split", kind="mergesort").reset_index(
            drop=True
        )
        if declared["split"].astype(str).tolist() != actual["split"].astype(str).tolist():
            reasons.append("split_conservation_split_identity_mismatch")
        else:
            numeric_columns = [column for column in actual if column != "split"]
            if not np.array_equal(
                declared[numeric_columns].to_numpy(dtype=np.int64),
                actual[numeric_columns].to_numpy(dtype=np.int64),
            ):
                reasons.append("split_conservation_totals_mismatch")
    except (TypeError, ValueError) as error:
        reasons.append(f"invalid_split_conservation:{error}")

    informative_gene_ids = tuple(
        sorted(
            set(
                rows.loc[
                    (rows["split"].astype(str) == "train")
                    & (rows["final_fate"].astype(str) == INFORMATIVE_FATE),
                    "target_gene_id",
                ].astype(str)
            )
        )
    )
    audit = build_long_read_compatibility_audit(
        rows,
        legal_paths_by_gene=legal_paths,
        model_admitted_gene_ids=candidates,
        strata=("split", "target_gene_id"),
    )
    if not _compatibility_conservation_pass(audit):
        reasons.append("long_read_compatibility_mass_not_conserved")
    return CompatibilityArtifactValidation(
        status="ADMITTED" if not reasons else "REJECTED",
        reasons=tuple(reasons),
        informative_gene_ids=informative_gene_ids if not reasons else (),
        audit=audit,
        candidate_gene_ids=candidates,
        legal_path_catalog_identity=str(
            manifest.get("legal_path_catalog_identity") or ""
        ),
        cell_split_identity=str(manifest.get("cell_split_identity") or ""),
        test_exposure=str(manifest.get("test_exposure") or ""),
        model_isoform_universe=str(
            manifest.get("model_isoform_universe") or ""
        ),
        matrix_structural_path_count=matrix_path_count,
    )


def require_real_v2_compatibility_admission(
    validation: CompatibilityArtifactValidation,
    *,
    expected_candidate_gene_count: int = 17_706,
    manifest_candidate_gene_count: int,
) -> None:
    """Guard the real-cohort boundary; toy/fixture validation remains allowed."""

    if manifest_candidate_gene_count != expected_candidate_gene_count:
        raise RuntimeError(
            f"formal V2 input requires exactly {expected_candidate_gene_count:,} "
            "explicit structural candidates"
        )
    if validation.status != "ADMITTED":
        raise RuntimeError(
            "compatible-EC artifact is not admitted: " + "; ".join(validation.reasons)
        )


def build_long_read_compatibility_audit(
    ec_rows: pd.DataFrame,
    *,
    legal_paths_by_gene: Mapping[str, Sequence[str]],
    model_admitted_gene_ids: Sequence[str],
    strata: Sequence[str] = ("split", "library_id", "donor_id", "target_gene_id", "cell_state"),
    stratum_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the fixed captured/QC/fate compatibility waterfall and fractions."""

    _require_columns(
        ec_rows,
        {
            "target_gene_id",
            "molecule_count",
            "pre_compatibility_qc_pass",
            "final_fate",
        }
        | set(strata),
        "compatible EC rows",
    )
    admitted = set(map(str, model_admitted_gene_ids))
    legal = {str(gene): tuple(paths) for gene, paths in legal_paths_by_gene.items()}
    rows = ec_rows.loc[
        ec_rows["target_gene_id"].astype(str).isin(admitted)
        & ec_rows["target_gene_id"].astype(str).map(lambda value: bool(legal.get(value)))
        & (ec_rows["molecule_count"].astype(float) > 0)
    ].copy()
    terminal = set(COMPATIBILITY_FATES)
    audit_rows: list[dict[str, object]] = []
    groupby_key = list(strata)
    grouped = rows.groupby(groupby_key, dropna=False, sort=True)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        qc = group.loc[group["pre_compatibility_qc_pass"].astype(bool)]
        unexpected = sorted(set(qc["final_fate"].astype(str)) - terminal)
        if unexpected:
            raise ValueError(f"QC-pass compatibility rows have unknown fates: {unexpected}")
        denominator_mass = float(qc["molecule_count"].astype(float).sum())
        denominator_rows = len(qc)
        fate_masses = {
            fate: float(
                qc.loc[qc["final_fate"].astype(str) == fate, "molecule_count"]
                .astype(float)
                .sum()
            )
            for fate in COMPATIBILITY_FATES
        }
        if not np.isclose(
            sum(fate_masses.values()), denominator_mass, atol=1e-8, rtol=0
        ):
            raise ValueError("compatibility terminal fates do not conserve QC-pass mass")
        base = {column: value for column, value in zip(strata, key_tuple)}
        for fate in COMPATIBILITY_FATES:
            fate_rows = qc.loc[qc["final_fate"].astype(str) == fate]
            audit_rows.append(
                {
                    **base,
                    "audit_population": "model_admitted_nonempty_catalog_positive_mass",
                    "captured_gene_assigned_row_count": len(group),
                    "captured_gene_assigned_molecule_mass": float(
                        group["molecule_count"].astype(float).sum()
                    ),
                    "pre_compatibility_qc_pass_row_count": denominator_rows,
                    "pre_compatibility_qc_pass_molecule_mass": denominator_mass,
                    "terminal_fate": fate,
                    "terminal_row_count": len(fate_rows),
                    "terminal_molecule_mass": fate_masses[fate],
                    "terminal_fraction": (
                        fate_masses[fate] / denominator_mass
                        if denominator_mass > 0
                        else np.nan
                    ),
                    "fraction_status": "estimable" if denominator_mass > 0 else "not_estimable",
                    "mass_conservation_pass": True,
                }
            )
    result = pd.DataFrame(audit_rows)
    if stratum_universe is not None:
        _require_columns(stratum_universe, set(strata), "compatibility stratum universe")
        universe = stratum_universe.drop_duplicates(list(strata))
        terminals = pd.DataFrame({"terminal_fate": list(COMPATIBILITY_FATES)})
        universe["_key"] = 1
        terminals["_key"] = 1
        complete = universe.merge(terminals, on="_key").drop(columns="_key")
        result = complete.merge(result, on=[*strata, "terminal_fate"], how="left")
        count_columns = [column for column in result if column.endswith("_count")]
        mass_columns = [column for column in result if column.endswith("_mass")]
        result[count_columns + mass_columns] = result[count_columns + mass_columns].fillna(0)
        missing = result["fraction_status"].isna()
        result.loc[missing, "fraction_status"] = "not_estimable"
        result.loc[missing, "mass_conservation_pass"] = True
    return result


def classify_retained_intron_evidence(
    molecule_intron_rows: pd.DataFrame, *, policy: IRPolicy
) -> pd.DataFrame:
    """Apply the frozen two-boundary retained-intron evidence policy."""

    if (
        policy.minimum_mapq < 0
        or policy.minimum_exon_aligned_bp_each_side <= 0
        or policy.minimum_intron_aligned_bp_each_side <= 0
    ):
        raise ValueError("retained-intron evidence policy thresholds are invalid")
    required = {
        "molecule_id",
        "target_gene_id",
        "intron_id",
        "is_primary",
        "is_chimeric",
        "mapq",
        "left_exon_aligned_bp",
        "left_intron_aligned_bp",
        "right_intron_aligned_bp",
        "right_exon_aligned_bp",
        "supports_excising_junction",
        "other_canonical_splice_junction_count",
        "unspliced_intron_count",
        "internal_priming_flag",
        "genomic_dna_contamination_flag",
        "protocol_mature_transcript_evidence",
    }
    _require_columns(molecule_intron_rows, required, "retained-intron evidence rows")
    if molecule_intron_rows.duplicated(["molecule_id", "intron_id"]).any():
        raise ValueError("retained-intron evidence rows require unique molecule/intron IDs")
    output = molecule_intron_rows.copy()
    qc = (
        output["is_primary"].astype(bool)
        & ~output["is_chimeric"].astype(bool)
        & (output["mapq"].astype(float) >= policy.minimum_mapq)
    )
    left_boundary = (
        (output["left_exon_aligned_bp"].astype(int) >= policy.minimum_exon_aligned_bp_each_side)
        & (output["left_intron_aligned_bp"].astype(int) >= policy.minimum_intron_aligned_bp_each_side)
    )
    right_boundary = (
        (output["right_intron_aligned_bp"].astype(int) >= policy.minimum_intron_aligned_bp_each_side)
        & (output["right_exon_aligned_bp"].astype(int) >= policy.minimum_exon_aligned_bp_each_side)
    )
    bilateral = left_boundary & right_boundary
    output["left_boundary_supported"] = qc & left_boundary
    output["right_boundary_supported"] = qc & right_boundary
    output["bilateral_boundary_supported"] = qc & bilateral
    output["single_boundary_only"] = qc & (left_boundary ^ right_boundary)
    output["intron_only_coverage"] = (
        qc
        & ~left_boundary
        & ~right_boundary
        & (
            (output["left_intron_aligned_bp"].astype(int) > 0)
            | (output["right_intron_aligned_bp"].astype(int) > 0)
        )
    )
    output["excising_junction_present"] = output[
        "supports_excising_junction"
    ].astype(bool)
    output["IR_alignment_supported"] = (
        qc & bilateral & ~output["supports_excising_junction"].astype(bool)
    )
    output["IR_evidence_censored"] = qc & ~output["IR_alignment_supported"]
    output["IR_evidence_class"] = np.select(
        [
            ~qc,
            qc & output["supports_excising_junction"].astype(bool),
            output["IR_alignment_supported"].astype(bool),
            output["single_boundary_only"].astype(bool),
            output["intron_only_coverage"].astype(bool),
        ],
        [
            "pre_ir_qc_failure",
            "excising_junction",
            "bilateral_boundary_supported",
            "single_boundary_only",
            "intron_only",
        ],
        default="no_boundary_evidence",
    )
    output["multi_intron_unspliced_pattern"] = (
        output["unspliced_intron_count"].astype(int) > 1
    )
    output["IR_biogenesis_context"] = np.where(
        output["other_canonical_splice_junction_count"].astype(int) > 0,
        "processed_context_supported",
        "mature_vs_nascent_unresolved",
    )
    output["IR_interpretation_scope"] = np.select(
        [
            output["IR_alignment_supported"].astype(bool)
            & output["protocol_mature_transcript_evidence"].astype(bool),
            output["IR_alignment_supported"].astype(bool)
            & (output["IR_biogenesis_context"] == "processed_context_supported"),
            output["IR_alignment_supported"].astype(bool),
        ],
        [
            "protocol_supported_mature_RI",
            "processed_context_RI_compatible",
            "mature_vs_nascent_unresolved_RI_compatible",
        ],
        default="not_IR_positive",
    )
    output["ir_policy_identity"] = _stable_identity(
        {
            "minimum_mapq": policy.minimum_mapq,
            "minimum_exon_aligned_bp_each_side": policy.minimum_exon_aligned_bp_each_side,
            "minimum_intron_aligned_bp_each_side": policy.minimum_intron_aligned_bp_each_side,
        }
    )
    return output


def rebuild_compatible_sets_after_ir_censoring(
    molecule_rows: pd.DataFrame,
    *,
    legal_paths_by_gene: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Remove censored IR-positive evidence, then rebuild an honest compatible set."""

    required = {
        "molecule_id",
        "target_gene_id",
        "accepted_non_ir_compatible_path_ids",
        "ir_supported_path_ids",
        "IR_alignment_supported",
        "IR_evidence_censored",
    }
    _require_columns(molecule_rows, required, "IR-compatible molecule rows")
    output = molecule_rows.copy()
    rebuilt: list[list[str]] = []
    fates: list[str] = []
    for row in output.itertuples(index=False):
        gene_id = str(row.target_gene_id)
        legal = list(_unique_ids(legal_paths_by_gene.get(gene_id, ()), f"{gene_id} legal path"))
        if not legal:
            raise ValueError(f"retained-intron molecule references absent legal paths: {gene_id}")
        non_ir = _string_list(
            row.accepted_non_ir_compatible_path_ids,
            "accepted_non_ir_compatible_path_ids",
        )
        ir_paths = _string_list(row.ir_supported_path_ids, "ir_supported_path_ids")
        if not _is_ordered_unique_subset(non_ir, legal) or not _is_ordered_unique_subset(
            ir_paths, legal
        ):
            raise ValueError("IR evidence references a path outside the frozen legal catalog")
        if bool(row.IR_alignment_supported):
            selected = set(non_ir) | set(ir_paths)
            paths = [path_id for path_id in legal if path_id in selected]
        elif bool(row.IR_evidence_censored):
            selected = set(non_ir)
            paths = [path_id for path_id in legal if path_id in selected] if non_ir else legal
        else:
            selected = set(non_ir)
            paths = [path_id for path_id in legal if path_id in selected] if non_ir else legal
        if not paths:
            paths = legal
        rebuilt.append(paths)
        fates.append(
            FULL_FATE if set(paths) == set(legal) else INFORMATIVE_FATE
        )
    output["compatible_path_ids_after_ir_policy"] = rebuilt
    output["final_fate_after_ir_policy"] = fates
    return output


def build_ir_library_audit(
    classified_rows: pd.DataFrame,
    *,
    library_columns: Sequence[str] = ("library_id", "donor_id"),
) -> pd.DataFrame:
    """Report the required IR evidence/biogenesis flags by library and donor."""

    flags = {
        "IR_alignment_supported",
        "IR_evidence_censored",
        "multi_intron_unspliced_pattern",
        "internal_priming_flag",
        "genomic_dna_contamination_flag",
        "IR_biogenesis_context",
    }
    _require_columns(classified_rows, set(library_columns) | flags, "IR classified rows")
    rows: list[dict[str, object]] = []
    for key, group in classified_rows.groupby(list(library_columns), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = {name: value for name, value in zip(library_columns, key_tuple)}
        denominator = len(group)
        row: dict[str, object] = {**base, "molecule_intron_row_count": denominator}
        for flag in (
            "IR_alignment_supported",
            "IR_evidence_censored",
            "multi_intron_unspliced_pattern",
            "internal_priming_flag",
            "genomic_dna_contamination_flag",
        ):
            count = int(group[flag].astype(bool).sum())
            row[f"{flag}_count"] = count
            row[f"{flag}_fraction"] = count / denominator if denominator else np.nan
        for status in ("processed_context_supported", "mature_vs_nascent_unresolved"):
            count = int((group["IR_biogenesis_context"].astype(str) == status).sum())
            row[f"{status}_count"] = count
            row[f"{status}_fraction"] = count / denominator if denominator else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_rna_window_coverage_audit(reference_sites: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen eligible-site RNA-window structural/active waterfall."""

    required = {
        "reference_site_id",
        "factor_entity_id",
        "reference_uniquely_mappable",
        "eligible_gene",
        "factor_entity_mappable",
        "exonic_in_any_legal_transcript",
        "inside_allowed_splice_window",
        "inside_allowed_tss_pas_window",
        "inside_allowed_rna_window",
        "has_legal_route",
        "retained_after_cap",
        "model_active",
        "denominator_kind",
        "assay",
        "biosample_context",
        "reference_build_identity",
    }
    _require_columns(reference_sites, required, "RNA reference-site coverage table")
    if reference_sites["reference_site_id"].astype(str).duplicated().any():
        raise ValueError("RNA reference coverage denominator sites must be unique")
    allowed_kinds = {
        "reference_experimentally_supported_site_coverage",
        "motif_candidate_coverage",
    }
    if not set(reference_sites["denominator_kind"].astype(str)).issubset(allowed_kinds):
        raise ValueError("RNA coverage denominator kind is not contractually named")
    eligible = reference_sites.loc[
        reference_sites["reference_uniquely_mappable"].astype(bool)
        & reference_sites["eligible_gene"].astype(bool)
        & reference_sites["factor_entity_mappable"].astype(bool)
    ].copy()
    eligible["region_class"] = np.select(
        [
            eligible["exonic_in_any_legal_transcript"].astype(bool),
            eligible["inside_allowed_splice_window"].astype(bool),
            eligible["inside_allowed_tss_pas_window"].astype(bool),
        ],
        [
            "exonic",
            "splice_proximal_intronic",
            "other_allowed_site_proximal",
        ],
        default="deep_intronic",
    )
    rows: list[dict[str, object]] = []
    group_columns = [
        "factor_entity_id",
        "region_class",
        "denominator_kind",
        "assay",
        "biosample_context",
        "reference_build_identity",
    ]
    for key, group in eligible.groupby(group_columns, dropna=False, sort=True):
        base = {name: value for name, value in zip(group_columns, key)}
        inside = group["inside_allowed_rna_window"].astype(bool)
        legal = inside & group["has_legal_route"].astype(bool)
        retained = legal & group["retained_after_cap"].astype(bool)
        active = retained & group["model_active"].astype(bool)
        rows.append(
            {
                **base,
                "eligible_frozen_reference_site_count": len(group),
                "inside_allowed_rna_window_count": int(inside.sum()),
                "has_legal_route_count": int(legal.sum()),
                "retained_after_cap_count": int(retained.sum()),
                "model_active_count": int(active.sum()),
                "structural_waterfall_scope": "split_neutral_through_retained_after_cap",
                "model_active_scope": "train_derived_gate_admission_suffix",
                "denominator_provenance_complete": bool(
                    group[
                        ["assay", "biosample_context", "reference_build_identity"]
                    ]
                    .notna()
                    .all()
                    .all()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_ont_observation_process_audit(
    *,
    matrix_manifest: Mapping[str, object],
    compatibility_manifest: Mapping[str, object],
    conservation_rows: pd.DataFrame,
    compatible_rebuild_performed: bool,
) -> OntObservationProcessAudit:
    """Audit matrix/compatible-read provenance and admission without relabeling scope."""

    identity_fields = (
        "software_identity",
        "config_identity",
        "reference_identity",
        "gtf_identity",
        "feature_identity",
        "barcode_identity",
        "qc_policy_identity",
        "assignment_policy_identity",
    )
    blocking_reasons: list[str] = []
    scope_reasons: list[str] = []
    rows: list[dict[str, object]] = []
    same_population_proven = True
    for field in identity_fields:
        matrix_value = matrix_manifest.get(field)
        compatibility_value = compatibility_manifest.get(field)
        present = bool(matrix_value) and bool(compatibility_value)
        match = bool(present and matrix_value == compatibility_value)
        rows.append(
            {
                "audit_component": "provenance_identity",
                "field": field,
                "matrix_value": matrix_value,
                "compatibility_value": compatibility_value,
                "exact_match": match,
            }
        )
        if not present:
            same_population_proven = False
            blocking_reasons.append(f"missing_provenance_identity:{field}")
        elif not match:
            same_population_proven = False
            scope_reasons.append(f"cross_pipeline_identity_difference:{field}")
    required_conservation = {
        "split",
        "cell_id",
        "target_gene_id",
        "matrix_count",
        "pre_compatibility_mass",
        "empty_compatible_mass",
        "proper_subset_compatible_mass",
        "full_set_compatible_mass",
        "other_explicit_fate_mass",
        "matrix_compatible_overlap_mass",
        "matrix_only_mass",
        "compatibility_only_mass",
    }
    missing = sorted(required_conservation - set(conservation_rows))
    conservation_pass = False
    if missing:
        blocking_reasons.append("missing_conservation_fields:" + ",".join(missing))
    else:
        conservation = conservation_rows.copy()
        if conservation.duplicated(["split", "cell_id", "target_gene_id"]).any():
            blocking_reasons.append("duplicate_cell_gene_conservation_identity")
        components = conservation[
            [
                "empty_compatible_mass",
                "proper_subset_compatible_mass",
                "full_set_compatible_mass",
                "other_explicit_fate_mass",
            ]
        ].to_numpy(dtype=np.float64)
        pre = conservation["pre_compatibility_mass"].to_numpy(dtype=np.float64)
        matrix = conservation["matrix_count"].to_numpy(dtype=np.float64)
        overlap = conservation["matrix_compatible_overlap_mass"].to_numpy(dtype=np.float64)
        matrix_only = conservation["matrix_only_mass"].to_numpy(dtype=np.float64)
        compatibility_only = conservation["compatibility_only_mass"].to_numpy(dtype=np.float64)
        conservation_pass = bool(
            np.isfinite(components).all()
            and np.isfinite(pre).all()
            and (components >= 0).all()
            and (pre >= 0).all()
            and np.allclose(components.sum(axis=1), pre, atol=1e-8, rtol=0)
            and np.isfinite(matrix).all()
            and np.isfinite(overlap).all()
            and np.isfinite(matrix_only).all()
            and np.isfinite(compatibility_only).all()
            and (matrix >= 0).all()
            and (overlap >= 0).all()
            and (matrix_only >= 0).all()
            and (compatibility_only >= 0).all()
            and np.allclose(matrix, overlap + matrix_only, atol=1e-8, rtol=0)
            and np.allclose(pre, overlap + compatibility_only, atol=1e-8, rtol=0)
        )
        if not conservation_pass:
            blocking_reasons.append("observation_process_mass_or_overlap_not_conserved")
        if same_population_proven and bool(
            (matrix_only > 0).any() or (compatibility_only > 0).any()
        ):
            same_population_proven = False
            scope_reasons.append("declared_identity_has_nonoverlapping_molecule_mass")
        for record in conservation.to_dict("records"):
            rows.append(
                {
                    "audit_component": "cell_gene_conservation",
                    **record,
                    "conservation_pass": bool(
                        np.isclose(
                            float(record["pre_compatibility_mass"]),
                            sum(
                                float(record[column])
                                for column in (
                                    "empty_compatible_mass",
                                    "proper_subset_compatible_mass",
                                    "full_set_compatible_mass",
                                    "other_explicit_fate_mass",
                                )
                            ),
                            atol=1e-8,
                            rtol=0,
                        )
                    )
                    and bool(
                        np.isclose(
                            float(record["matrix_count"]),
                            float(record["matrix_compatible_overlap_mass"])
                            + float(record["matrix_only_mass"]),
                            atol=1e-8,
                            rtol=0,
                        )
                    )
                    and bool(
                        np.isclose(
                            float(record["pre_compatibility_mass"]),
                            float(record["matrix_compatible_overlap_mass"])
                            + float(record["compatibility_only_mass"]),
                            atol=1e-8,
                            rtol=0,
                        )
                    ),
                }
            )
    if not compatible_rebuild_performed:
        blocking_reasons.insert(0, "compatible_read_rebuild_not_performed")
    comparison_name = (
        "same_observation_process_ont_matrix_agreement"
        if same_population_proven
        else "same_library_cross_pipeline_ont_matrix_agreement"
    )
    status = (
        "ADMITTED"
        if compatible_rebuild_performed and conservation_pass and not blocking_reasons
        else "PENDING_OBSERVATION_PROCESS_AUDIT"
    )
    return OntObservationProcessAudit(
        status=status,
        comparison_name=comparison_name,
        audit=pd.DataFrame(rows),
        reasons=tuple([*blocking_reasons, *scope_reasons]),
    )


def audit_high_dtu_provenance(
    gene_metadata: pd.DataFrame,
    *,
    sensitivity_enabled: bool,
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    """Freeze a train-side-only high-DTU sensitivity provenance record."""

    required = {
        "target_gene_id",
        "metadata_split_scope",
        "high_dtu_gene",
        "dtu_source_identity",
    }
    _require_columns(gene_metadata, required, "high-DTU provenance metadata")
    if gene_metadata["target_gene_id"].astype(str).duplicated().any():
        raise ValueError("high-DTU provenance requires unique gene IDs")
    scopes = set(gene_metadata["metadata_split_scope"].astype(str))
    if scopes != {"train_only"}:
        raise ValueError("high-DTU selection metadata must be derived only from train scope")
    if gene_metadata["dtu_source_identity"].isna().any():
        raise ValueError("high-DTU source identity must be complete")
    enabled = bool(sensitivity_enabled)
    manifest = {
        "schema_version": "FABRIC_V2_HIGH_DTU_PROVENANCE_V1",
        "sensitivity_enabled": enabled,
        "selection_config": dict(selection_config),
        "metadata_split_scope": "train_only",
        "gene_count": len(gene_metadata),
        "high_dtu_gene_count": int(gene_metadata["high_dtu_gene"].astype(bool).sum()),
        "dtu_source_identities": sorted(
            set(gene_metadata["dtu_source_identity"].astype(str))
        ),
        "primary_sampling_or_loss_effect": "none",
        "formal_claim_scope": "sensitivity_only" if enabled else "disabled_zero_effect",
    }
    if not enabled and any(
        float(selection_config.get(key, default)) != default
        for key, default in (
            ("high_dtu_sampling_multiplier", 1.0),
            ("high_dtu_loss_multiplier", 1.0),
        )
    ):
        raise ValueError("disabled high-DTU sensitivity must have zero sampling/loss effect")
    manifest["high_dtu_provenance_identity"] = _stable_identity(manifest)
    return manifest


def _raw_interaction_contrasts(
    *,
    modality: str,
    config: Mapping[str, object],
    modality_context: pd.DataFrame,
    base_matrix: np.ndarray,
    supported_route_columns: Mapping[str, np.ndarray],
    field_raw_active: Mapping[str, tuple[np.ndarray, list[tuple[str, str]]]],
    field_manifests: Mapping[str, Mapping[str, object]],
    supported_lookup: Mapping[tuple[str, str, str, str], bool],
    tolerance: float,
) -> list[dict[str, object]]:
    factors = list(config["interaction_factor_vocabulary"])
    rows: list[dict[str, object]] = []
    for field, field_config in config["context_fields"].items():
        support_matrix, raw_cells = field_raw_active[field]
        raw_index = {value: index for index, value in enumerate(raw_cells)}
        active_columns = list(field_manifests[field]["active_support_column_indices"])
        h_active = support_matrix[:, active_columns]
        other_blocks = [base_matrix]
        for other_field, values in supported_route_columns.items():
            if other_field != field and values.shape[1]:
                other_blocks.append(values)
        z_minus = np.column_stack(other_blocks)
        z_rank, _ = _numeric_rank(z_minus, tolerance)
        for factor in factors:
            for level_left, level_right in field_config["scientific_context_pairs"]:
                contrast_id = "raw-contrast:" + hashlib.sha256(
                    _canonical_json(
                        [modality, field, factor, level_left, level_right]
                    ).encode("utf-8")
                ).hexdigest()[:24]
                focal_supported = all(
                    supported_lookup.get((modality, field, factor, level), False)
                    for level in (level_left, level_right)
                )
                comparators = [
                    comparator
                    for comparator in factors
                    if comparator != factor
                    and all(
                        supported_lookup.get(
                            (modality, field, comparator, level), False
                        )
                        for level in (level_left, level_right)
                    )
                ] if focal_supported else []
                if not focal_supported:
                    q_status = "unsupported_focal_arms"
                elif not comparators:
                    q_status = "within_factor_only"
                else:
                    q_status = None
                comparator_rows: list[dict[str, object]] = []
                for comparator in comparators:
                    vector = np.zeros(len(raw_cells), dtype=np.int64)
                    vector[raw_index[(factor, level_left)]] = 1
                    vector[raw_index[(factor, level_right)]] = -1
                    vector[raw_index[(comparator, level_left)]] = -1
                    vector[raw_index[(comparator, level_right)]] = 1
                    rank_left = _exact_rank(h_active)
                    rank_right = _exact_rank(np.column_stack([h_active, vector]))
                    in_span = rank_left == rank_right
                    lifted = np.zeros(len(modality_context), dtype=np.float64)
                    for route_index, route in enumerate(modality_context.itertuples(index=False)):
                        raw_cell = (
                            str(route.interaction_factor_id),
                            str(getattr(route, field)),
                        )
                        if raw_cell in raw_index:
                            lifted[route_index] = vector[raw_index[raw_cell]]
                    alias_rank, _ = _numeric_rank(
                        np.column_stack([z_minus, lifted]), tolerance
                    )
                    separable = alias_rank > z_rank
                    aliased_fields = []
                    if not separable:
                        aliased_fields = [
                            other_field
                            for other_field, block in supported_route_columns.items()
                            if other_field != field and block.shape[1]
                        ]
                    comparator_rows.append(
                        {
                            "comparator_id": comparator,
                            "contrast_in_active_span": in_span,
                            "raw_active_rank": rank_left,
                            "raw_active_plus_contrast_rank": rank_right,
                            "cross_field_context_separable": separable,
                            "aliased_field_ids": aliased_fields,
                        }
                    )
                passing = [
                    row["comparator_id"]
                    for row in comparator_rows
                    if row["contrast_in_active_span"]
                    and row["cross_field_context_separable"]
                ]
                if comparator_rows:
                    if passing:
                        q_status = "factor_specific_grammar_estimable"
                    elif any(
                        not row["cross_field_context_separable"]
                        for row in comparator_rows
                    ):
                        q_status = "cross_field_context_not_separable"
                    else:
                        q_status = "raw_contrast_not_in_active_span"
                summary = {
                    "raw_interaction_contrast_id": contrast_id,
                    "row_kind": "q_summary",
                    "modality": modality,
                    "context_field": field,
                    "factor_entity_id": factor,
                    "context_level_a": level_left,
                    "context_level_b": level_right,
                    "comparator_id": None,
                    "raw_support_status": (
                        "four_corner_covered"
                        if comparator_rows
                        else q_status
                    ),
                    "contrast_in_active_span": None,
                    "cross_field_context_separable": None,
                    "aliased_field_ids": sorted(
                        {
                            value
                            for row in comparator_rows
                            for value in row["aliased_field_ids"]
                        }
                    ),
                    "raw_interaction_claim_status": q_status,
                    "comparator_ids_passing_claim_gate": passing,
                }
                rows.append(summary)
                for comparator_row in comparator_rows:
                    rows.append(
                        {
                            **summary,
                            "row_kind": "comparator",
                            "comparator_id": comparator_row["comparator_id"],
                            "contrast_in_active_span": comparator_row[
                                "contrast_in_active_span"
                            ],
                            "raw_active_rank": comparator_row["raw_active_rank"],
                            "raw_active_plus_contrast_rank": comparator_row[
                                "raw_active_plus_contrast_rank"
                            ],
                            "cross_field_context_separable": comparator_row[
                                "cross_field_context_separable"
                            ],
                            "aliased_field_ids": comparator_row["aliased_field_ids"],
                        }
                    )
    return rows


def _route_context_levels(
    frame: pd.DataFrame, distance_bin_boundaries: Mapping[str, Sequence[float]]
) -> pd.DataFrame:
    result = frame.copy()
    labels: list[str] = []
    for row in result.itertuples(index=False):
        modality = str(row.modality)
        boundaries = [float(value) for value in distance_bin_boundaries.get(modality, ())]
        if boundaries != sorted(set(boundaries)) or any(
            not np.isfinite(value) or value < 0 for value in boundaries
        ):
            raise ValueError(f"distance-bin boundaries for {modality} are invalid")
        if str(row.geometry_kind) == "site_window":
            signed_distance = float(row.signed_distance_bp)
            if np.isinf(signed_distance):
                raise ValueError("signed route distance may be finite or NaN, never infinite")
            if not np.isfinite(signed_distance):
                if str(row.transcript_oriented_side) != "OVERLAP_ANCHOR":
                    raise ValueError("non-overlap site route requires signed distance")
                labels.append("NA")
                continue
            distance = abs(signed_distance)
        elif str(row.geometry_kind) == "edge_contained":
            distance = min(
                float(row.distance_to_5prime_boundary_bp),
                float(row.distance_to_3prime_boundary_bp),
            )
        else:
            raise ValueError(f"unknown route geometry kind: {row.geometry_kind}")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("route distance-bin value is absent or invalid")
        position = int(np.searchsorted(boundaries, distance, side="right"))
        lower = "0" if position == 0 else _format_number(boundaries[position - 1])
        upper = "inf" if position == len(boundaries) else _format_number(boundaries[position])
        labels.append(f"[{lower},{upper}{']' if upper == 'inf' else ')'}")
    result["distance_bin"] = labels
    return result


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))


def _exact_rank(matrix: np.ndarray) -> int:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("exact-rank input must be a matrix")
    if not array.size:
        return 0
    work = [
        [Fraction(value.item() if hasattr(value, "item") else value) for value in row]
        for row in array
    ]
    row_count, column_count = len(work), len(work[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, row_count) if work[row][column] != 0), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(row_count):
            if row == rank or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                left - factor * right
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def _numeric_rank(matrix: np.ndarray, tolerance: float) -> tuple[int, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("numeric-rank input must be a matrix")
    if min(matrix.shape, default=0) == 0:
        return 0, np.zeros(0, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0, singular
    return int(np.sum(singular > tolerance * singular[0])), singular


def _exact_duplicate_columns(matrix: np.ndarray) -> list[tuple[int, int]]:
    array = np.asarray(matrix)
    duplicates: list[tuple[int, int]] = []
    for left in range(array.shape[1]):
        for right in range(left + 1, array.shape[1]):
            if np.array_equal(array[:, left], array[:, right]):
                duplicates.append((left, right))
    return duplicates


def _canonical_zero_vector(values: np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float64).copy()
    vector[vector == 0] = 0.0
    return vector.tolist()


def _footprint_relation(events: pd.DataFrame) -> str:
    intervals = [
        (str(row.chromosome), int(row.start), int(row.end), str(row.strand))
        for row in events.itertuples(index=False)
    ]
    if len(set(intervals)) == 1:
        return "identical_interval"
    if all(
        left[0] == right[0]
        and left[3] == right[3]
        and min(left[2], right[2]) > max(left[1], right[1])
        for left, right in itertools.combinations(intervals, 2)
    ):
        return "overlapping_under_frozen_rule"
    return "mixed_or_disjoint"


def _motif_family_relation(events: pd.DataFrame) -> str:
    kinds = set(events["factor_identity_kind"].astype(str))
    if kinds == {"accessibility_only"}:
        return "accessibility_only"
    if "accessibility_only" in kinds:
        return "mixed_or_NA"
    families = set(events["motif_equivalence_family_id"].astype(str))
    return (
        "same_motif_equivalence_family"
        if len(families) == 1
        else "distinct_non_equivalent_families"
    )


def _should_have_physically_collapsed(
    events: pd.DataFrame, *, minimum_overlap_bp: int, minimum_reciprocal_overlap: float
) -> bool:
    if len(events) < 2:
        return False
    for left, right in itertools.combinations(events.to_dict("records"), 2):
        collapse_fields = (
            "target_gene_id",
            "modality",
            "factor_entity_id",
            "motif_equivalence_family_id",
            "chromosome",
            "strand",
            "peak_id",
        )
        if any(_canonical_json(left[field]) != _canonical_json(right[field]) for field in collapse_fields):
            continue
        overlap = min(int(left["end"]), int(right["end"])) - max(
            int(left["start"]), int(right["start"])
        )
        if overlap < minimum_overlap_bp:
            continue
        reciprocal = min(
            overlap / (int(left["end"]) - int(left["start"])),
            overlap / (int(right["end"]) - int(right["start"])),
        )
        if reciprocal >= minimum_reciprocal_overlap:
            return True
    return False


def _compatibility_conservation_pass(audit: pd.DataFrame) -> bool:
    return audit.empty or bool(audit["mass_conservation_pass"].astype(bool).all())


def _compatibility_split_conservation(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "captured_molecule_mass",
        "pre_qc_pass_molecule_mass",
        "technical_qc_failure_molecule_mass",
        "empty_compatible_molecule_mass",
        "proper_subset_compatible_molecule_mass",
        "full_set_compatible_molecule_mass",
        "other_explicit_fate_molecule_mass",
    ]
    output: list[dict[str, object]] = []
    for split, group in rows.groupby("split", sort=True):
        mass = group["molecule_count"].astype(np.int64)
        qc = group["pre_compatibility_qc_pass"].astype(bool)
        fates = group["final_fate"].astype(str)
        known_qc_fates = {EMPTY_FATE, INFORMATIVE_FATE, FULL_FATE}
        output.append(
            {
                "split": str(split),
                "captured_molecule_mass": int(mass.sum()),
                "pre_qc_pass_molecule_mass": int(mass.loc[qc].sum()),
                "technical_qc_failure_molecule_mass": int(mass.loc[~qc].sum()),
                "empty_compatible_molecule_mass": int(mass.loc[qc & (fates == EMPTY_FATE)].sum()),
                "proper_subset_compatible_molecule_mass": int(
                    mass.loc[qc & (fates == INFORMATIVE_FATE)].sum()
                ),
                "full_set_compatible_molecule_mass": int(
                    mass.loc[qc & (fates == FULL_FATE)].sum()
                ),
                "other_explicit_fate_molecule_mass": int(
                    mass.loc[qc & ~fates.isin(known_qc_fates)].sum()
                ),
            }
        )
    return pd.DataFrame(output, columns=columns)


def _is_ordered_unique_subset(values: Sequence[str], frozen_axis: Sequence[str]) -> bool:
    items = list(map(str, values))
    axis = list(map(str, frozen_axis))
    if len(axis) != len(set(axis)) or len(items) != len(set(items)):
        return False
    position = {value: index for index, value in enumerate(axis)}
    if any(value not in position for value in items):
        return False
    indices = [position[value] for value in items]
    return indices == sorted(indices)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * sorted_weights.sum()
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(sorted_values[index])


def _unique_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if any(not value or value in {"None", "nan"} for value in result):
        raise ValueError(f"{label} IDs must be non-empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} IDs must be unique")
    return result


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of IDs")
    result = [str(item) for item in value]
    if any(not item or item in {"None", "nan"} for item in result):
        raise ValueError(f"{label} contains an empty ID")
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} misses required columns: {missing}")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _stable_identity(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_array_identity(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode("ascii")
        + _canonical_json(list(array.shape)).encode("ascii")
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
