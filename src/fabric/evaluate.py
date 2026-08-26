"""FABRIC V2 reporting, counterfactuals, and held-out diagnostics.

This module never trains or mutates a model.  Every operation either resolves a
frozen reporting set, evaluates fixed predictions, or fits the explicitly
validation-only state-residual diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import json
from math import ceil, comb
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .choices import (
    PathIdentifiabilityIndex,
    aggregate_group_log_probabilities,
    aggregate_group_probabilities,
    alternative_relative_log_mass,
)
from .dataset import (
    ATACMappingContext,
    ActivityContext,
    GateValues,
    build_raw_gate_signals,
    compute_activity_entities,
    transform_gates,
)
from .model import FABRICOutput, FABRICV2Model, GeneCellModelInput


PRIMARY_TRAINING_CONDITIONS = (
    "cis",
    "cis_dna",
    "cis_rna",
    "full",
    "full_additive_edge",
)
RUNTIME_CONDITIONS = ("full", "atac", "rbp")
PRIMARY_INJECTION_SCOPES = {"singleton_supported", "set_supported"}


@dataclass(frozen=True)
class EvidenceSelectorResolution:
    selector_id: str
    selector_kind: str
    route_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    route_count: int
    complete_model_injection_group_ids: tuple[str, ...]
    partial_model_injection_group_ids: tuple[str, ...]
    model_injection_scope: str


@dataclass(frozen=True)
class OntMatrixIdentity:
    matrix_rows: pd.DataFrame
    matrix_cells: pd.DataFrame
    crosswalk: pd.DataFrame
    model_paths: pd.DataFrame
    status: str = "PASS"


@dataclass(frozen=True)
class OntMatrixScope:
    candidates: pd.DataFrame
    conservation: pd.DataFrame


@dataclass(frozen=True)
class OntMatrixAgreement:
    records: pd.DataFrame
    summary: pd.DataFrame
    compatible_set_records: pd.DataFrame
    compatible_set_summary: pd.DataFrame
    numerical_status: str


@dataclass(frozen=True)
class ValidationMonitorBundle:
    """Frozen validation-only inputs and provenance for the sealed monitor."""

    ont_identity: OntMatrixIdentity
    ont_scope: OntMatrixScope
    compatible_validation_ec_rows: pd.DataFrame
    matrix_identity: str
    crosswalk_identity: str
    path_identity: str
    split_identity: str
    observation_process_status: str
    comparison_name: str = "same_library_cross_pipeline_ont_matrix_agreement"
    metric_schema_version: str = "fabric_v2_epoch_core_metrics_v1"
    model_output_dtype: str = "float32"
    aggregation_dtype: str = "float64"
    probability_export_dtype: str = "float64"
    numerical_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class OntMatrixKlTarget:
    """Frozen validation-only ONT counts on the exact G_fit path/cell axes."""

    counts: sparse.csr_matrix  # [global G_fit path, validation cell]
    path_ids: tuple[str, ...]
    path_gene_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    expected_cell_gene_keys: np.ndarray
    matrix_identity: str
    path_identity: str
    split_identity: str
    scope_policy: str

    def __post_init__(self) -> None:
        if not sparse.isspmatrix_csr(self.counts):
            raise TypeError("ONT KL target counts must be scipy CSR")
        if self.counts.shape != (len(self.path_ids), len(self.cell_ids)):
            raise ValueError("ONT KL target count axes differ from path/cell identities")
        if len(self.path_gene_ids) != len(self.path_ids):
            raise ValueError("ONT KL target gene and path axes differ")
        if (
            not self.path_ids
            or not self.cell_ids
            or len(set(self.path_ids)) != len(self.path_ids)
            or len(set(self.cell_ids)) != len(self.cell_ids)
            or any(not value for value in (*self.path_ids, *self.cell_ids))
        ):
            raise ValueError("ONT KL target path/cell identities must be unique and nonempty")
        if any(not value for value in self.path_gene_ids):
            raise ValueError("ONT KL target gene identities must be nonempty")
        if self.expected_cell_gene_keys.ndim != 1 or len(self.expected_cell_gene_keys) == 0:
            raise ValueError(
                "ONT KL expected validation cell-gene keys must be one-dimensional and nonempty"
            )
        if not np.issubdtype(self.expected_cell_gene_keys.dtype, np.integer):
            raise TypeError("ONT KL expected validation cell-gene keys must be integers")
        gene_count = len(dict.fromkeys(self.path_gene_ids))
        if (
            bool((self.expected_cell_gene_keys < 0).any())
            or bool(
                (
                    self.expected_cell_gene_keys
                    >= gene_count * len(self.cell_ids)
                ).any()
            )
            or bool(
                (
                    self.expected_cell_gene_keys[1:]
                    <= self.expected_cell_gene_keys[:-1]
                ).any()
            )
        ):
            raise ValueError(
                "ONT KL expected validation cell-gene keys must be ordered unique in-range identities"
            )
        values = self.counts.data
        if (
            not np.isfinite(values).all()
            or bool((values < 0).any())
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("ONT KL target counts must be finite non-negative integers")
        if self.scope_policy != (
            "likelihood_informative_validation_cell_gene_with_at_least_two_"
            "positive_ont_paths"
        ):
            raise ValueError("ONT KL target scope policy differs")
        for name in ("matrix_identity", "path_identity", "split_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ONT KL target {name} must be explicit")

    @classmethod
    def load(cls, root: str | Path) -> "OntMatrixKlTarget":
        root = Path(root)
        manifest = json.loads((root / "OntMatrixKlTargetManifest.json").read_text())
        if manifest.get("schema_version") != "fabric.ont_validation_kl_target.v2":
            raise ValueError("unsupported ONT validation KL target schema")
        if manifest.get("test_cells_or_counts_included") is not False:
            raise ValueError("ONT validation KL target must exclude test cells and counts")
        path_axis = pd.read_parquet(root / manifest["path_axis"])
        cell_axis = pd.read_parquet(root / manifest["cell_axis"])
        expected_axis = pd.read_parquet(
            root / manifest["expected_cell_gene_axis"],
            columns=[
                "expected_instance_order_0based",
                "gene_order_0based",
                "cell_order_0based",
            ],
        )
        _require_columns(
            path_axis,
            ("path_order_0based", "path_id", "gene_id"),
            "ONT KL path axis",
        )
        _require_columns(
            cell_axis,
            ("cell_order_0based", "cell_id", "split"),
            "ONT KL cell axis",
        )
        _require_columns(
            expected_axis,
            (
                "expected_instance_order_0based",
                "gene_order_0based",
                "cell_order_0based",
            ),
            "ONT KL expected cell-gene axis",
        )
        if path_axis["path_order_0based"].tolist() != list(range(len(path_axis))):
            raise ValueError("ONT KL path order is not contiguous")
        if cell_axis["cell_order_0based"].tolist() != list(range(len(cell_axis))):
            raise ValueError("ONT KL cell order is not contiguous")
        if not cell_axis["split"].astype(str).eq("val").all():
            raise ValueError("ONT KL target contains non-validation cells")
        if not np.array_equal(
            expected_axis["expected_instance_order_0based"].to_numpy(dtype=np.int64),
            np.arange(len(expected_axis), dtype=np.int64),
        ) or len(expected_axis) != manifest.get("expected_cell_gene_count"):
            raise ValueError("ONT KL expected cell-gene order or count differs")
        expected_keys = (
            expected_axis["gene_order_0based"].to_numpy(dtype=np.int64)
            * len(cell_axis)
            + expected_axis["cell_order_0based"].to_numpy(dtype=np.int64)
        )
        counts = sparse.load_npz(root / manifest["counts"]).tocsr()
        if (
            list(counts.shape) != manifest.get("counts_shape")
            or counts.nnz != manifest.get("counts_nnz")
        ):
            raise ValueError("ONT KL target sparse count identity differs")
        return cls(
            counts=counts,
            path_ids=tuple(path_axis["path_id"].astype(str)),
            path_gene_ids=tuple(path_axis["gene_id"].astype(str)),
            cell_ids=tuple(cell_axis["cell_id"].astype(str)),
            expected_cell_gene_keys=expected_keys,
            matrix_identity=str(manifest["matrix_identity"]),
            path_identity=str(manifest["path_identity"]),
            split_identity=str(manifest["split_identity"]),
            scope_policy=str(manifest["scope_policy"]),
        )


@dataclass(frozen=True)
class OntMatrixKlResult:
    """The one per-epoch ONT distribution-agreement metric plus denominators."""

    ont_matrix_kl_count_weighted: float
    eligible_cell_gene_count: int
    eligible_ont_count: int
    zero_total_cell_gene_count: int
    fewer_than_two_positive_paths_cell_gene_count: int


def compute_validation_ont_matrix_kl(
    snapshot: object,
    target: OntMatrixKlTarget,
) -> OntMatrixKlResult:
    """Compute exact log-space count-weighted KL without another model forward."""

    if getattr(snapshot, "split", None) != "val":
        raise ValueError("ONT matrix KL accepts validation snapshots only")
    predictions = getattr(snapshot, "predictions", None)
    if not isinstance(predictions, tuple) or not predictions:
        raise ValueError("validation snapshot has no predictions for ONT matrix KL")
    path_position = {value: index for index, value in enumerate(target.path_ids)}
    cell_position = {value: index for index, value in enumerate(target.cell_ids)}
    gene_paths: dict[str, tuple[str, ...]] = {}
    gene_position: dict[str, int] = {}
    for gene_id, path_id in zip(target.path_gene_ids, target.path_ids, strict=True):
        gene_position.setdefault(gene_id, len(gene_position))
        gene_paths.setdefault(gene_id, tuple())
        gene_paths[gene_id] = (*gene_paths[gene_id], path_id)

    weighted_kl_sum = 0.0
    eligible_ont_count = 0
    eligible_cell_gene_count = 0
    zero_total_count = 0
    fewer_than_two_count = 0
    observed_expected = np.zeros(len(target.expected_cell_gene_keys), dtype=bool)
    for prediction in predictions:
        gene_id = str(getattr(prediction, "gene_id"))
        path_ids = tuple(map(str, getattr(prediction, "path_ids")))
        if gene_id not in gene_paths or path_ids != gene_paths[gene_id]:
            raise ValueError(f"ONT KL path axis differs for gene {gene_id}")
        cell_ids = tuple(map(str, getattr(prediction, "cell_ids")))
        missing_cells = [value for value in cell_ids if value not in cell_position]
        if missing_cells:
            raise ValueError(
                f"ONT KL target misses validation cells for {gene_id}: {missing_cells[:5]}"
            )
        logits = getattr(prediction, "path_logits").detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        if logits.shape != (len(cell_ids), len(path_ids)) or not np.isfinite(logits).all():
            raise ValueError("ONT KL logits are non-finite or differ from cell/path axes")
        row_index = [path_position[value] for value in path_ids]
        column_index = [cell_position[value] for value in cell_ids]
        counts = target.counts[row_index][:, column_index].toarray().T.astype(
            np.float64, copy=False
        )
        for cell_id, cell_counts, cell_logits in zip(
            cell_ids, counts, logits, strict=True
        ):
            key = gene_position[gene_id] * len(target.cell_ids) + cell_position[cell_id]
            expected_position = int(
                np.searchsorted(target.expected_cell_gene_keys, key, side="left")
            )
            if (
                expected_position == len(target.expected_cell_gene_keys)
                or int(target.expected_cell_gene_keys[expected_position]) != key
            ):
                raise ValueError(
                    "ONT KL validation prediction scope differs from the frozen "
                    f"cell-gene set: unexpected={(cell_id, gene_id)}"
                )
            if observed_expected[expected_position]:
                raise ValueError("ONT KL validation cell-gene instance is duplicated")
            observed_expected[expected_position] = True
            total = int(cell_counts.sum())
            if total == 0:
                zero_total_count += 1
                continue
            if int(np.count_nonzero(cell_counts)) < 2:
                fewer_than_two_count += 1
                continue
            q = cell_counts / total
            positive = q > 0
            log_probability = cell_logits - logsumexp(cell_logits)
            kl = float(
                np.sum(q[positive] * (np.log(q[positive]) - log_probability[positive]))
            )
            if not np.isfinite(kl) or kl < -1.0e-10:
                raise FloatingPointError("ONT matrix KL is non-finite or materially negative")
            weighted_kl_sum += total * max(0.0, kl)
            eligible_ont_count += total
            eligible_cell_gene_count += 1
    if not bool(observed_expected.all()):
        gene_ids = tuple(gene_position)
        missing_keys = target.expected_cell_gene_keys[~observed_expected][:5]
        missing = [
            (
                target.cell_ids[int(key) % len(target.cell_ids)],
                gene_ids[int(key) // len(target.cell_ids)],
            )
            for key in missing_keys
        ]
        raise ValueError(
            "ONT KL validation prediction scope differs from the frozen cell-gene set: "
            f"missing={missing}"
        )
    if eligible_ont_count <= 0:
        raise ValueError("ONT matrix KL has zero eligible validation count mass")
    return OntMatrixKlResult(
        ont_matrix_kl_count_weighted=weighted_kl_sum / eligible_ont_count,
        eligible_cell_gene_count=eligible_cell_gene_count,
        eligible_ont_count=eligible_ont_count,
        zero_total_cell_gene_count=zero_total_count,
        fewer_than_two_positive_paths_cell_gene_count=fewer_than_two_count,
    )


@dataclass(frozen=True)
class PairingPermutationManifest:
    assignments: pd.DataFrame
    strata: pd.DataFrame
    strata_fields: tuple[str, ...]
    null_kind: str
    repetitions: int
    seed: int
    minimum_stratum_cells: int


@dataclass(frozen=True)
class FrozenRidgeDiagnostic:
    feature_columns: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    include_state: bool
    state_pc_columns: tuple[str, ...]
    category_levels: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class StateResidualDiagnostics:
    nuisance: FrozenRidgeDiagnostic
    state: FrozenRidgeDiagnostic


@dataclass(frozen=True)
class AttributionSeedSummary:
    records: pd.DataFrame
    effect_rank_correlations: pd.DataFrame


@dataclass(frozen=True)
class PerturbedGateContext:
    """One explicitly rebuilt dynamic context at a declared input layer."""

    perturbation_kind: str
    cell_id: str
    input_kind: str
    input_id: str
    input_value: float
    activity: ActivityContext
    atac: ATACMappingContext
    gates: GateValues
    affected_gate_key_ids: tuple[str, ...]
    affected_event_ids: tuple[str, ...]
    gate_audit: pd.DataFrame
    support_status: str


@dataclass(frozen=True)
class PerturbationResponseCurve:
    gate_records: pd.DataFrame
    response_records: pd.DataFrame


@dataclass(frozen=True)
class EventDensityTables:
    prediction: pd.DataFrame
    attribution: pd.DataFrame


@dataclass(frozen=True)
class SupportStratifiedSensitivity:
    assignments: pd.DataFrame
    per_seed: pd.DataFrame
    across_seed: pd.DataFrame


@dataclass(frozen=True)
class ArchitectureComparison:
    per_seed: pd.DataFrame
    claim_summary: pd.DataFrame


@dataclass(frozen=True)
class CounterfactualAttribution:
    path_records: pd.DataFrame
    group_records: pd.DataFrame
    alternative_records: pd.DataFrame
    intermediate_audit: pd.DataFrame
    named_intermediate_deltas: Mapping[str, torch.Tensor]
    full_output: FABRICOutput
    counterfactual_output: FABRICOutput


@dataclass(frozen=True)
class PathScaleAuditResult:
    records: pd.DataFrame
    representation_collisions: pd.DataFrame
    output: FABRICOutput


class OntEpochMonitor:
    """A sealed, reporting-only callback invoked once per completed epoch."""

    def __init__(self, compute: Callable[[torch.nn.Module], Mapping[str, object]], *, enabled: bool = True):
        self.compute = compute
        self.enabled = bool(enabled)
        self._records: list[dict[str, object]] = []
        self._last_epoch = 0

    def record_completed_epoch(self, epoch: int, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        if int(epoch) != self._last_epoch + 1:
            raise ValueError("ONT monitor must run exactly once for each completed epoch")
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = dict(self.compute(model))
        model.train(was_training)
        self._records.append({"completed_epoch": int(epoch), **values})
        self._last_epoch = int(epoch)

    def read_for_post_selection_reporting(
        self,
        *,
        selection_and_reporting_rules_frozen: bool,
    ) -> pd.DataFrame:
        """Expose records only after every selection/reporting rule is frozen.

        The method deliberately has no selection-oriented read path: sealed ONT
        monitor fields are reporting outputs and can never become model-selection
        inputs.
        """

        if not selection_and_reporting_rules_frozen:
            raise RuntimeError(
                "sealed ONT monitor records are available only for post-selection reporting"
            )
        return pd.DataFrame(self._records).copy()


def compute_validation_monitor_record(
    snapshot: object,
    bundle: ValidationMonitorBundle,
) -> dict[str, float | int | str | bool]:
    """Compute all validation monitor fields from one frozen prediction traversal.

    ``snapshot`` is intentionally duck-typed to avoid a train/evaluate import
    cycle.  It must obey ``train.ValidationSnapshot`` and its nested prediction
    contract.
    """

    if getattr(snapshot, "split", None) != "val":
        raise ValueError("sealed ONT monitor accepts validation snapshots only")
    for name in (
        "matrix_identity",
        "crosswalk_identity",
        "path_identity",
        "split_identity",
        "observation_process_status",
        "comparison_name",
        "metric_schema_version",
        "model_output_dtype",
        "aggregation_dtype",
        "probability_export_dtype",
    ):
        value = getattr(bundle, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"validation monitor bundle has empty {name}")
    if bundle.comparison_name != "same_library_cross_pipeline_ont_matrix_agreement":
        raise ValueError("V2 monitor comparison name must preserve cross_pipeline semantics")
    if bundle.aggregation_dtype != "float64" or bundle.probability_export_dtype != "float64":
        raise ValueError("current V2 monitor aggregation and probability export use float64")
    if not np.isfinite(bundle.numerical_tolerance) or bundle.numerical_tolerance <= 0:
        raise ValueError("validation monitor numerical tolerance must be positive and finite")
    ec_rows = bundle.compatible_validation_ec_rows.copy()
    _require_columns(
        ec_rows,
        ("cell_id", "gene_id", "split", "compatible_path_ids", "molecule_count"),
        "validation monitor EC rows",
    )
    if not ec_rows["split"].astype(str).eq("val").all():
        raise ValueError("validation monitor EC bundle contains non-validation rows")
    if not bundle.ont_scope.candidates["split"].astype(str).eq("val").all():
        raise ValueError("validation ONT scope contains non-validation candidates")

    path_logits, snapshot_ec, observed_dtype = _flatten_validation_snapshot(snapshot)
    if observed_dtype != bundle.model_output_dtype:
        raise ValueError(
            "validation snapshot model-output dtype differs from frozen monitor bundle"
        )
    _validate_snapshot_path_axes(path_logits, bundle.ont_identity)
    _validate_snapshot_ec_identity(
        snapshot_ec, ec_rows, numerical_tolerance=bundle.numerical_tolerance
    )
    predicted_instances = set(
        zip(path_logits["cell_id"].astype(str), path_logits["gene_id"].astype(str))
    )
    eligible = bundle.ont_scope.candidates.loc[
        bundle.ont_scope.candidates["scope_status"].eq("eligible")
    ]
    missing_eligible = sorted(
        set(zip(eligible["cell_id"].astype(str), eligible["gene_id"].astype(str)))
        - predicted_instances
    )
    if missing_eligible:
        raise ValueError(
            f"validation snapshot misses frozen eligible ONT instances: {missing_eligible[:5]}"
        )

    agreement = compute_ont_matrix_agreement(
        bundle.ont_identity,
        bundle.ont_scope,
        path_logits,
    )
    summary = agreement.summary.iloc[0].to_dict()
    ont_kl = summary.get("ont_matrix_kl_count_weighted")
    if (
        isinstance(ont_kl, bool)
        or not isinstance(ont_kl, (int, float))
        or not np.isfinite(ont_kl)
        or ont_kl < 0
    ):
        raise FloatingPointError("validation ONT matrix KL is not finite and non-negative")
    conservation = bundle.ont_scope.conservation
    if len(conservation) != 1 or str(conservation.iloc[0]["split"]) != "val":
        raise ValueError("validation ONT scope conservation must contain one val row")
    scope = conservation.iloc[0]
    fields: dict[str, object] = {
        "metric_schema": bundle.metric_schema_version,
        "matrix_identity": bundle.matrix_identity,
        "crosswalk_identity": bundle.crosswalk_identity,
        "path_identity": bundle.path_identity,
        "split_identity": bundle.split_identity,
        "observation_process_status": bundle.observation_process_status,
        "comparison_name": bundle.comparison_name,
        "model_output_dtype": bundle.model_output_dtype,
        "aggregation_dtype": bundle.aggregation_dtype,
        "probability_export_dtype": bundle.probability_export_dtype,
        "numerical_tolerance": float(bundle.numerical_tolerance),
        "validation_compatible_path_nll": float(getattr(snapshot, "nll")),
        "ont_matrix_kl_count_weighted": float(ont_kl),
        "validation_informative_molecule_mass": float(
            getattr(snapshot, "informative_molecule_mass")
        ),
        "prediction_gene_count": int(path_logits["gene_id"].nunique()),
        "prediction_cell_gene_count": int(
            path_logits[["cell_id", "gene_id"]].drop_duplicates().shape[0]
        ),
        "same_validation_prediction_traversal": True,
        "sealed": True,
        "selection_eligible": False,
        "ont_eligible_cell_gene_count": int(scope["eligible_count"]),
        "ont_eligible_count_denominator": int(scope["eligible_raw_count_mass"]),
        "ont_zero_total_cell_gene_count": int(scope["zero_total_count"]),
        "ont_fewer_than_two_positive_paths_cell_gene_count": int(
            scope["fewer_than_two_positive_count"]
        ),
    }
    return {key: _json_scalar(value) for key, value in fields.items()}


def resolve_evidence_selector(
    selector_kind: str,
    selector_id: str,
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    *,
    model_injection_index: pd.DataFrame | None = None,
    correlated_gate_sets: pd.DataFrame | None = None,
) -> EvidenceSelectorResolution:
    """Resolve one primitive or reporting-derived selector to an exact route set."""

    _require_columns(physical_events, ("event_id", "factor_entity_id", "gate_key_id"), "physical events")
    _require_columns(event_routes, ("route_id", "event_id", "anchor_region_id"), "event routes")
    _require_unique(physical_events, "event_id", "physical event")
    _require_unique(event_routes, "route_id", "event route")
    unknown_events = sorted(set(event_routes["event_id"].astype(str)) - set(physical_events["event_id"].astype(str)))
    if unknown_events:
        raise ValueError(f"routes reference unknown physical events: {unknown_events[:5]}")
    event_table = physical_events.copy()
    event_table["event_id"] = event_table["event_id"].astype(str)
    routes = event_routes.copy()
    routes["route_id"] = routes["route_id"].astype(str)
    routes["event_id"] = routes["event_id"].astype(str)

    if selector_kind == "event":
        if selector_id not in set(event_table["event_id"]):
            raise KeyError(f"unknown event selector {selector_id}")
        selected = routes.loc[routes["event_id"].eq(selector_id)]
    elif selector_kind == "factor":
        event_ids = set(
            event_table.loc[event_table["factor_entity_id"].astype(str).eq(str(selector_id)), "event_id"]
        )
        if not event_ids:
            raise KeyError(f"unknown factor/group selector {selector_id}")
        selected = routes.loc[routes["event_id"].isin(event_ids)]
    elif selector_kind == "anchor_region":
        selected = routes.loc[routes["anchor_region_id"].astype(str).eq(str(selector_id))]
        if selected.empty:
            raise KeyError(f"unknown anchor-region selector {selector_id}")
    elif selector_kind == "correlated_evidence_set":
        if correlated_gate_sets is None:
            raise ValueError("correlated_evidence_set selector requires its frozen table")
        _require_columns(
            correlated_gate_sets,
            ("correlated_evidence_set_id", "member_gate_key_ids"),
            "correlated gate sets",
        )
        rows = correlated_gate_sets.loc[
            correlated_gate_sets["correlated_evidence_set_id"].astype(str).eq(str(selector_id))
        ]
        if len(rows) != 1:
            raise KeyError(f"correlated evidence set {selector_id} is absent or duplicated")
        gate_keys = set(map(str, _as_list(rows.iloc[0]["member_gate_key_ids"])))
        event_ids = set(event_table.loc[event_table["gate_key_id"].astype(str).isin(gate_keys), "event_id"])
        selected = routes.loc[routes["event_id"].isin(event_ids)]
    elif selector_kind == "model_injection_group":
        if model_injection_index is None:
            raise ValueError("model injection selector requires its frozen index")
        _require_columns(
            model_injection_index,
            ("model_injection_group_id", "member_event_ids", "member_route_ids"),
            "model injection index",
        )
        rows = model_injection_index.loc[
            model_injection_index["model_injection_group_id"].astype(str).eq(str(selector_id))
        ]
        if len(rows) != 1:
            raise KeyError(f"model injection group {selector_id} is absent or duplicated")
        route_ids = set(map(str, _as_list(rows.iloc[0]["member_route_ids"])))
        selected = routes.loc[routes["route_id"].isin(route_ids)]
        if set(selected["route_id"]) != route_ids:
            raise ValueError("model injection group references routes outside EventRouteTable")
    else:
        raise ValueError(f"unsupported evidence selector kind: {selector_kind}")
    if selected.empty:
        raise ValueError(f"selector {selector_kind}:{selector_id} resolves to no routes")

    selected_route_ids = tuple(sorted(selected["route_id"].astype(str)))
    complete, partial, scope = classify_model_injection_coverage(
        selected_route_ids, model_injection_index
    )
    return EvidenceSelectorResolution(
        selector_id=str(selector_id),
        selector_kind=selector_kind,
        route_ids=selected_route_ids,
        event_ids=tuple(sorted(set(selected["event_id"].astype(str)))),
        route_count=len(selected_route_ids),
        complete_model_injection_group_ids=complete,
        partial_model_injection_group_ids=partial,
        model_injection_scope=scope,
    )


def classify_model_injection_coverage(
    selected_route_ids: Sequence[str],
    model_injection_index: pd.DataFrame | None,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Classify full versus partial exact-injection-class coverage."""

    if model_injection_index is None:
        return (), (), "not_evaluated_missing_index"
    if model_injection_index.empty:
        if selected_route_ids:
            raise ValueError(
                "selected production routes are absent from the empty model injection index"
            )
        return (), (), "singleton_supported"
    _require_columns(
        model_injection_index,
        ("model_injection_group_id", "member_count", "member_route_ids"),
        "model injection index",
    )
    selected = set(map(str, selected_route_ids))
    indexed_routes: set[str] = set()
    for value in model_injection_index["member_route_ids"]:
        routes = set(map(str, _as_list(value)))
        overlap = indexed_routes & routes
        if overlap:
            raise ValueError(
                f"model injection index is not a route partition: {sorted(overlap)[:5]}"
            )
        indexed_routes.update(routes)
    absent = sorted(selected - indexed_routes)
    if absent:
        raise ValueError(
            f"selected production routes are absent from model injection index: {absent[:5]}"
        )
    complete: list[str] = []
    partial: list[str] = []
    complete_sizes: list[int] = []
    for row in model_injection_index.itertuples(index=False):
        routes = set(map(str, _as_list(row.member_route_ids)))
        overlap = selected & routes
        if not overlap:
            continue
        if overlap == routes:
            complete.append(str(row.model_injection_group_id))
            complete_sizes.append(int(row.member_count))
        else:
            partial.append(str(row.model_injection_group_id))
    if partial:
        scope = "partial_model_injection_group"
    elif any(size > 1 for size in complete_sizes):
        scope = "set_supported"
    else:
        scope = "singleton_supported"
    return tuple(sorted(complete)), tuple(sorted(partial)), scope


def neutralize_routed_terms(
    route_edge_terms: np.ndarray | torch.Tensor,
    route_ids: Sequence[str],
    selector: EvidenceSelectorResolution,
) -> np.ndarray | torch.Tensor:
    """Zero selected route rows before event aggregation, leaving gates unchanged."""

    if route_edge_terms.shape[-2] != len(route_ids):
        raise ValueError("route-term tensor and route ID axes differ")
    route_index = {str(route_id): index for index, route_id in enumerate(route_ids)}
    unknown = sorted(set(selector.route_ids) - set(route_index))
    if unknown:
        raise ValueError(f"selector routes are absent from tensor axis: {unknown[:5]}")
    selected = [route_index[route_id] for route_id in selector.route_ids]
    if isinstance(route_edge_terms, torch.Tensor):
        result = route_edge_terms.clone()
        result[..., selected, :] = 0
        return result
    result = np.asarray(route_edge_terms).copy()
    result[..., selected, :] = 0
    return result


def run_evidence_counterfactual(
    model: FABRICV2Model,
    model_input: GeneCellModelInput,
    selector: EvidenceSelectorResolution,
    *,
    gene_id: str,
    cell_ids: Sequence[str],
    path_ids: Sequence[str],
    dna_route_ids: Sequence[str],
    rna_route_ids: Sequence[str],
    dna_event_ids: Sequence[str],
    rna_event_ids: Sequence[str],
    dna_gate_observed: torch.Tensor,
    rna_gate_observed: torch.Tensor,
    path_identifiability_index: PathIdentifiabilityIndex | None = None,
    alternative_contrasts: pd.DataFrame | None = None,
    condition: str = "full",
) -> CounterfactualAttribution:
    """Forward one exact route-union neutralization through the same model.

    The full and counterfactual passes share parameters and all non-selected
    inputs.  Frozen route weights are never renormalized.  Missing observation
    masks suppress scientific estimands rather than turning an unobserved
    context into a reported zero effect.
    """

    cells = tuple(map(str, cell_ids))
    paths = tuple(map(str, path_ids))
    if len(cells) != len(set(cells)) or len(paths) != len(set(paths)):
        raise ValueError("counterfactual cell and path axes must be unique")
    if len(cells) != model_input.dna.gate.shape[0] or len(cells) != model_input.rna.gate.shape[0]:
        raise ValueError("counterfactual cell axis differs from model input")
    if len(paths) != model_input.path_edge_incidence.shape[0]:
        raise ValueError("counterfactual path axis differs from model input")
    if model_input.dna.route_keep_mask is not None or model_input.rna.route_keep_mask is not None:
        raise ValueError("full attribution input must not already contain a route counterfactual")

    dna_routes = _unique_axis(dna_route_ids, model_input.dna.route_event_index.numel(), "DNA route")
    rna_routes = _unique_axis(rna_route_ids, model_input.rna.route_event_index.numel(), "RNA route")
    overlap = set(dna_routes) & set(rna_routes)
    if overlap:
        raise ValueError(f"route IDs occur in both modality axes: {sorted(overlap)[:5]}")
    selected = set(selector.route_ids)
    absent = sorted(selected - set(dna_routes) - set(rna_routes))
    if absent:
        raise ValueError(f"selector routes are absent from model input axes: {absent[:5]}")
    dna_events = _unique_axis(
        dna_event_ids, model_input.dna.event_gate_key_index.numel(), "DNA event"
    )
    rna_events = _unique_axis(
        rna_event_ids, model_input.rna.event_gate_key_index.numel(), "RNA event"
    )
    event_overlap = set(dna_events) & set(rna_events)
    if event_overlap:
        raise ValueError(f"event IDs occur in both modality axes: {sorted(event_overlap)[:5]}")
    for observed, routed, label in (
        (dna_gate_observed, model_input.dna, "DNA"),
        (rna_gate_observed, model_input.rna, "RNA"),
    ):
        if observed.dtype != torch.bool or observed.shape != routed.gate.shape:
            raise ValueError(f"{label} observed mask must be boolean and match its gate tensor")
        if observed.device != routed.gate.device:
            raise ValueError(f"{label} observed mask and gate must share a device")

    dna_selected = torch.tensor(
        [route_id in selected for route_id in dna_routes],
        dtype=torch.bool,
        device=model_input.dna.route_event_index.device,
    )
    rna_selected = torch.tensor(
        [route_id in selected for route_id in rna_routes],
        dtype=torch.bool,
        device=model_input.rna.route_event_index.device,
    )
    selected_event_ids: set[str] = set()
    selected_context_observed = torch.ones(
        len(cells), dtype=torch.bool, device=model_input.dna.gate.device
    )
    selected_modalities: set[str] = set()
    for route_mask, routed, event_axis, observed, modality in (
        (dna_selected, model_input.dna, dna_events, dna_gate_observed, "DNA"),
        (rna_selected, model_input.rna, rna_events, rna_gate_observed, "RNA"),
    ):
        if not bool(route_mask.any()):
            continue
        selected_modalities.add(modality)
        event_indices = torch.unique(routed.route_event_index[route_mask], sorted=True)
        selected_event_ids.update(event_axis[index] for index in event_indices.tolist())
        gate_columns = routed.event_gate_key_index[event_indices]
        selected_context_observed &= observed[:, gate_columns].all(dim=1).to(
            selected_context_observed.device
        )
    if selected_event_ids != set(selector.event_ids):
        raise ValueError(
            "selector event IDs differ from events reached by its exact model route union"
        )

    counterfactual_input = replace(
        model_input,
        dna=model_input.dna.with_route_keep_mask(~dna_selected),
        rna=model_input.rna.with_route_keep_mask(~rna_selected),
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        full_output = model(model_input, condition=condition)
        counter_output = model(counterfactual_input, condition=condition)
    model.train(was_training)
    named_deltas = {
        "a_DNA": full_output.dna_aggregate - counter_output.dna_aggregate,
        "a_RNA": full_output.rna_aggregate - counter_output.rna_aggregate,
        "y": full_output.joint_projected - counter_output.joint_projected,
        "y_hat": full_output.normalized_tokens - counter_output.normalized_tokens,
        "H": full_output.edge_states - counter_output.edge_states,
        "path_logits": full_output.path_logits - counter_output.path_logits,
    }
    for name, tensor in named_deltas.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"counterfactual named intermediate {name} is non-finite")

    enabled_modalities = {
        "cis": set(),
        "cis_dna": {"DNA"},
        "cis_rna": {"RNA"},
        "full": {"DNA", "RNA"},
    }
    if condition not in enabled_modalities:
        raise ValueError("counterfactual condition must be one primary modality condition")
    modality_applicable = selected_modalities.issubset(enabled_modalities[condition])
    full_logp = torch.log_softmax(full_output.path_logits, dim=-1)
    counter_logp = torch.log_softmax(counter_output.path_logits, dim=-1)
    effects = gauge_invariant_counterfactual_outputs(
        full_output.path_logits, counter_output.path_logits
    )
    base_records = []
    for cell_index, cell_id in enumerate(cells):
        if not modality_applicable:
            status = "not_applicable_modality_absent"
        elif not bool(selected_context_observed[cell_index]):
            status = "missing_context_not_estimable"
        elif selector.model_injection_scope == "partial_model_injection_group":
            status = "partial_model_injection_group"
        else:
            status = "estimable_model_counterfactual"
        base_records.append(
            {
                "cell_id": cell_id,
                "gene_id": str(gene_id),
                "selector_kind": selector.selector_kind,
                "selector_id": selector.selector_id,
                "route_ids": list(selector.route_ids),
                "route_count": selector.route_count,
                "event_ids": list(selector.event_ids),
                "model_injection_scope": selector.model_injection_scope,
                "attribution_status": status,
                "required_dynamic_context_observed": bool(
                    selected_context_observed[cell_index]
                ),
                "primary_mechanism_summary_eligible": (
                    status == "estimable_model_counterfactual"
                    and selector.model_injection_scope in PRIMARY_INJECTION_SCOPES
                ),
            }
        )

    path_rows: list[dict[str, object]] = []
    for cell_index, base in enumerate(base_records):
        estimable = base["attribution_status"] in {
            "estimable_model_counterfactual",
            "partial_model_injection_group",
        }
        for path_index, path_id in enumerate(paths):
            path_rows.append(
                {
                    **base,
                    "path_id": path_id,
                    "full_log_path_probability": (
                        float(full_logp[cell_index, path_index]) if estimable else np.nan
                    ),
                    "counterfactual_log_path_probability": (
                        float(counter_logp[cell_index, path_index]) if estimable else np.nan
                    ),
                    "delta_log_path_probability": (
                        float(effects["delta_log_path_probability"][cell_index, path_index])
                        if estimable
                        else np.nan
                    ),
                    "delta_path_probability": (
                        float(effects["delta_path_probability"][cell_index, path_index])
                        if estimable
                        else np.nan
                    ),
                    "centered_delta_path_logit": (
                        float(effects["centered_delta_path_logit"][cell_index, path_index])
                        if estimable
                        else np.nan
                    ),
                }
            )

    group_rows: list[dict[str, object]] = []
    if path_identifiability_index is not None:
        gene_groups = path_identifiability_index.groups.loc[
            path_identifiability_index.groups["gene_id"].astype(str).eq(str(gene_id))
        ].sort_values("observational_group_index", kind="mergesort")
        if gene_groups.empty:
            raise ValueError("gene is absent from PathIdentifiabilityIndex")
        full_group = aggregate_group_probabilities(
            full_logp.exp(), path_identifiability_index, str(gene_id)
        )
        counter_group = aggregate_group_probabilities(
            counter_logp.exp(), path_identifiability_index, str(gene_id)
        )
        full_group_log = aggregate_group_log_probabilities(
            full_logp, path_identifiability_index, str(gene_id)
        )
        counter_group_log = aggregate_group_log_probabilities(
            counter_logp, path_identifiability_index, str(gene_id)
        )
        for cell_index, base in enumerate(base_records):
            estimable = base["attribution_status"] in {
                "estimable_model_counterfactual",
                "partial_model_injection_group",
            }
            for group_index, group in enumerate(gene_groups.itertuples(index=False)):
                full_value = full_group[cell_index, group_index]
                counter_value = counter_group[cell_index, group_index]
                group_rows.append(
                    {
                        **base,
                        "observational_group_id": str(group.observational_group_id),
                        "member_path_ids": list(group.member_path_ids),
                        "full_group_probability": float(full_value) if estimable else np.nan,
                        "counterfactual_group_probability": (
                            float(counter_value) if estimable else np.nan
                        ),
                        "delta_group_probability": (
                            float(full_value - counter_value) if estimable else np.nan
                        ),
                        "delta_log_group_probability": (
                            float(
                                full_group_log[cell_index, group_index]
                                - counter_group_log[cell_index, group_index]
                            )
                            if estimable
                            else np.nan
                        ),
                    }
                )

    alternative_rows: list[dict[str, object]] = []
    if alternative_contrasts is not None:
        required = (
            "contrast_id",
            "gene_id",
            "choice_id",
            "contrast_kind",
            "context_signature",
            "numerator_path_ids",
            "denominator_path_ids",
            "cohort_reportable",
        )
        _require_columns(alternative_contrasts, required, "alternative contrasts")
        contrasts = alternative_contrasts.loc[
            alternative_contrasts["gene_id"].astype(str).eq(str(gene_id))
        ]
        if contrasts["contrast_id"].astype(str).duplicated().any():
            raise ValueError("alternative contrast IDs must be unique within a gene")
        for contrast in contrasts.itertuples(index=False):
            full_rho = alternative_relative_log_mass(
                full_output.path_logits,
                paths,
                contrast.numerator_path_ids,
                contrast.denominator_path_ids,
            )
            counter_rho = alternative_relative_log_mass(
                counter_output.path_logits,
                paths,
                contrast.numerator_path_ids,
                contrast.denominator_path_ids,
            )
            for cell_index, base in enumerate(base_records):
                estimable = (
                    base["attribution_status"]
                    in {"estimable_model_counterfactual", "partial_model_injection_group"}
                    and bool(contrast.cohort_reportable)
                )
                alternative_rows.append(
                    {
                        **base,
                        "contrast_id": str(contrast.contrast_id),
                        "choice_id": str(contrast.choice_id),
                        "contrast_kind": str(contrast.contrast_kind),
                        "context_signature": contrast.context_signature,
                        "cohort_reportable": bool(contrast.cohort_reportable),
                        "full_relative_log_mass": (
                            float(full_rho[cell_index]) if estimable else np.nan
                        ),
                        "counterfactual_relative_log_mass": (
                            float(counter_rho[cell_index]) if estimable else np.nan
                        ),
                        "delta_relative_log_mass": (
                            float(full_rho[cell_index] - counter_rho[cell_index])
                            if estimable
                            else np.nan
                        ),
                    }
                )

    audit_rows = []
    for cell_index, base in enumerate(base_records):
        audit_rows.append(
            {
                **base,
                **{
                    f"delta_{name}_frobenius_norm": float(
                        torch.linalg.vector_norm(tensor[cell_index])
                    )
                    for name, tensor in named_deltas.items()
                },
                "all_named_intermediates_finite": True,
                "route_weights_renormalized": False,
                "same_frozen_model_parameters": True,
            }
        )
    return CounterfactualAttribution(
        path_records=pd.DataFrame(path_rows),
        group_records=pd.DataFrame(group_rows),
        alternative_records=pd.DataFrame(alternative_rows),
        intermediate_audit=pd.DataFrame(audit_rows),
        named_intermediate_deltas={
            key: value.detach().clone() for key, value in named_deltas.items()
        },
        full_output=full_output,
        counterfactual_output=counter_output,
    )


def rebuild_source_proxy_perturbation(
    *,
    cell_id: str,
    source_kind: str,
    source_id: str,
    source_value: float,
    activity: ActivityContext,
    atac: ATACMappingContext,
    gate_keys: pd.DataFrame,
    gate_admission: pd.DataFrame,
    physical_events: pd.DataFrame,
) -> PerturbedGateContext:
    """Rebuild all affected gates from a declared source-proxy intervention."""

    value = _nonnegative_finite_scalar(source_value, "source proxy value")
    cell_index = _context_cell_index(activity, atac, cell_id)
    keys = _validated_gate_key_table(gate_keys)
    if source_kind == "activity_proxy":
        entity_index = {
            value: index for index, value in enumerate(activity.activity_entity_ids)
        }
        if source_id not in entity_index:
            raise ValueError(f"activity source proxy is absent: {source_id}")
        values = np.asarray(activity.values).copy()
        values[cell_index, entity_index[source_id]] = value
        perturbed_activity = replace(activity, values=values)
        perturbed_atac = atac
        affected = keys["activity_entity_id"].astype("string").eq(str(source_id))
    elif source_kind == "mapped_accessibility":
        peak_index = {value: index for index, value in enumerate(atac.peak_ids)}
        if source_id not in peak_index:
            raise ValueError(f"mapped ATAC source proxy is absent: {source_id}")
        accessibility = sparse.csr_matrix(atac.accessibility).tolil(copy=True)
        accessibility[cell_index, peak_index[source_id]] = value
        perturbed_atac = replace(atac, accessibility=accessibility.tocsr())
        perturbed_activity = activity
        affected = keys["peak_id"].astype("string").eq(str(source_id))
    else:
        raise ValueError(
            "source_kind must be 'activity_proxy' or 'mapped_accessibility'"
        )
    affected_keys = tuple(keys.loc[affected, "gate_key_id"].astype(str))
    if not affected_keys:
        raise ValueError("source proxy has no affected gate keys")
    return _materialize_perturbed_gate_context(
        perturbation_kind="source_proxy_perturbation",
        cell_id=str(cell_id),
        input_kind=source_kind,
        input_id=str(source_id),
        input_value=value,
        activity=perturbed_activity,
        atac=perturbed_atac,
        gate_keys=keys,
        gate_admission=gate_admission,
        physical_events=physical_events,
        affected_gate_key_ids=affected_keys,
        extra_audit_fields={
            "library_denominator_semantics": "not_applicable_source_proxy",
        },
    )


def rebuild_member_count_perturbation(
    *,
    cell_id: str,
    member_gene_id: str,
    member_count: int,
    raw_rna_counts: sparse.spmatrix | np.ndarray,
    frozen_gene_axis: Sequence[str],
    entity_table: pd.DataFrame,
    activity: ActivityContext,
    atac: ATACMappingContext,
    gate_keys: pd.DataFrame,
    gate_admission: pd.DataFrame,
    physical_events: pd.DataFrame,
    target_sum: float = 10_000.0,
) -> PerturbedGateContext:
    """Recompute unique/group proxies while fixing the observed RNA denominator."""

    count = _nonnegative_finite_scalar(member_count, "member count")
    if count != np.floor(count):
        raise ValueError("member count must be an integer raw count")
    if target_sum <= 0 or not np.isfinite(target_sum):
        raise ValueError("activity normalization target_sum must be positive")
    cell_index = _context_cell_index(activity, atac, cell_id)
    cells = tuple(map(str, activity.cell_ids))
    genes = tuple(map(str, frozen_gene_axis))
    if len(genes) != len(set(genes)) or str(member_gene_id) not in genes:
        raise ValueError("member gene is absent from a unique frozen RNA gene axis")
    counts = sparse.csr_matrix(raw_rna_counts, dtype=np.float64)
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("raw RNA count shape differs from activity cell/gene axes")
    if counts.data.size and (
        not np.isfinite(counts.data).all() or bool((counts.data < 0).any())
    ):
        raise ValueError("raw RNA counts must be finite and non-negative")
    observed_library = np.asarray(counts.sum(axis=1)).reshape(-1)
    if not np.allclose(
        observed_library,
        np.asarray(activity.library_size, dtype=np.float64),
        atol=1.0e-8,
        rtol=0,
    ):
        raise ValueError("raw RNA counts differ from the frozen activity denominator")
    denominator = float(activity.library_size[cell_index])
    if denominator <= 0:
        raise ValueError("member-count perturbation requires a positive library denominator")

    _require_columns(
        entity_table,
        ("activity_entity_id", "activity_gene_ids", "source_valid"),
        "activity entity table",
    )
    entities = entity_table.sort_values(
        "activity_entity_id", kind="mergesort"
    ).reset_index(drop=True)
    if tuple(entities["activity_entity_id"].astype(str)) != tuple(
        activity.activity_entity_ids
    ):
        raise ValueError("activity entity table differs from the frozen activity axis")
    gene_index = {value: index for index, value in enumerate(genes)}
    member_column = gene_index[str(member_gene_id)]
    original_count = float(counts[cell_index, member_column])
    values = np.asarray(activity.values).copy()
    affected_entities: list[str] = []
    for entity_column, entity in enumerate(entities.itertuples(index=False)):
        members = tuple(map(str, _as_list(entity.activity_gene_ids)))
        if not members:
            raise ValueError("activity entities require at least one source gene")
        missing = sorted(set(members) - set(gene_index))
        if bool(entity.source_valid) and missing:
            raise ValueError(
                f"valid activity entity has members absent from RNA axis: {missing[:5]}"
            )
        if str(member_gene_id) not in members:
            continue
        affected_entities.append(str(entity.activity_entity_id))
        if not bool(entity.source_valid) or not bool(
            activity.observed[cell_index, entity_column]
        ):
            continue
        member_columns = [gene_index[value] for value in members]
        raw_sum = float(counts[cell_index, member_columns].sum())
        perturbed_sum = raw_sum - original_count + count
        if perturbed_sum < 0:
            raise RuntimeError("member-count recomputation produced a negative group sum")
        values[cell_index, entity_column] = np.float32(
            np.log1p(target_sum * perturbed_sum / denominator)
        )
    if not affected_entities:
        raise ValueError("member gene belongs to no activity entity")
    perturbed_activity = replace(activity, values=values)
    keys = _validated_gate_key_table(gate_keys)
    affected = keys["activity_entity_id"].astype("string").isin(affected_entities)
    affected_keys = tuple(keys.loc[affected, "gate_key_id"].astype(str))
    if not affected_keys:
        raise ValueError("member-count perturbation has no affected gate keys")
    return _materialize_perturbed_gate_context(
        perturbation_kind="member_count_perturbation",
        cell_id=str(cell_id),
        input_kind="raw_member_count_fixed_library_denominator",
        input_id=str(member_gene_id),
        input_value=count,
        activity=perturbed_activity,
        atac=atac,
        gate_keys=keys,
        gate_admission=gate_admission,
        physical_events=physical_events,
        affected_gate_key_ids=affected_keys,
        extra_audit_fields={
            "original_member_count": original_count,
            "fixed_library_denominator": denominator,
            "library_denominator_semantics": "fixed_observed_library_denominator",
        },
    )


def rebuild_observed_library_context(
    *,
    cell_id: str,
    raw_rna_counts: sparse.spmatrix | np.ndarray,
    frozen_gene_axis: Sequence[str],
    entity_table: pd.DataFrame,
    rna_observation_valid: Sequence[bool],
    atac: ATACMappingContext,
    gate_keys: pd.DataFrame,
    gate_admission: pd.DataFrame,
    physical_events: pd.DataFrame,
    target_sum: float = 10_000.0,
) -> PerturbedGateContext:
    """Build a measured library context using that library's full denominator."""

    activity = compute_activity_entities(
        raw_rna_counts,
        cell_ids=atac.cell_ids,
        frozen_gene_axis=frozen_gene_axis,
        entity_table=entity_table,
        rna_observation_valid=rna_observation_valid,
        target_sum=target_sum,
    )
    cell_index = _context_cell_index(activity, atac, cell_id)
    keys = _validated_gate_key_table(gate_keys)
    affected_keys = tuple(
        keys.loc[keys["activity_entity_id"].notna(), "gate_key_id"].astype(str)
    )
    if not affected_keys:
        raise ValueError("observed RNA library context has no activity-dependent gates")
    denominator = float(activity.library_size[cell_index])
    return _materialize_perturbed_gate_context(
        perturbation_kind="observed_library_context",
        cell_id=str(cell_id),
        input_kind="complete_observed_rna_library",
        input_id="complete_observed_rna_library",
        input_value=denominator,
        activity=activity,
        atac=atac,
        gate_keys=keys,
        gate_admission=gate_admission,
        physical_events=physical_events,
        affected_gate_key_ids=affected_keys,
        extra_audit_fields={
            "fixed_library_denominator": denominator,
            "library_denominator_semantics": "new_observed_library_denominator",
        },
    )


def evaluate_perturbation_response_curve(
    points: Sequence[PerturbedGateContext],
    *,
    predictor: Callable[[PerturbedGateContext], pd.DataFrame],
    expected_outputs: pd.DataFrame,
) -> PerturbationResponseCurve:
    """Forward rebuilt points and enforce the complete frozen output manifest."""

    if not points:
        raise ValueError("response curve requires at least one rebuilt input point")
    _require_columns(
        expected_outputs,
        ("output_kind", "output_id"),
        "response output manifest",
    )
    expected = expected_outputs[["output_kind", "output_id"]].copy()
    expected["output_kind"] = expected["output_kind"].astype(str)
    expected["output_id"] = expected["output_id"].astype(str)
    if expected.empty or expected.duplicated(["output_kind", "output_id"]).any():
        raise ValueError("response output manifest must contain unique non-empty outputs")
    allowed_kinds = {
        "matched_context_relative_log_mass",
        "marginal_relative_log_mass",
        "path_log_probability",
        "path_probability",
        "group_log_probability",
        "group_probability",
    }
    if not set(expected["output_kind"]).issubset(allowed_kinds):
        raise ValueError("response output manifest contains an unsupported estimand")
    scope = (
        points[0].cell_id,
        points[0].perturbation_kind,
        points[0].input_kind,
        points[0].input_id,
    )
    input_values = [float(point.input_value) for point in points]
    if len(input_values) != len(set(input_values)):
        raise ValueError("response curve input values must be unique")
    expected_keys = set(map(tuple, expected.to_numpy()))
    gate_frames: list[pd.DataFrame] = []
    response_frames: list[pd.DataFrame] = []
    for point_index, point in enumerate(points):
        if (
            point.cell_id,
            point.perturbation_kind,
            point.input_kind,
            point.input_id,
        ) != scope:
            raise ValueError("response curve points differ in cell or intervention scope")
        gate_frame = point.gate_audit.copy()
        gate_frame.insert(0, "response_point_index", point_index)
        gate_frames.append(gate_frame)

        predicted = predictor(point).copy()
        _require_columns(
            predicted,
            ("output_kind", "output_id", "value"),
            "response prediction",
        )
        predicted["output_kind"] = predicted["output_kind"].astype(str)
        predicted["output_id"] = predicted["output_id"].astype(str)
        if predicted.duplicated(["output_kind", "output_id"]).any():
            raise ValueError("response prediction repeats a declared output")
        actual_keys = set(
            map(tuple, predicted[["output_kind", "output_id"]].to_numpy())
        )
        if actual_keys != expected_keys:
            raise ValueError(
                "response prediction does not exactly cover the frozen output manifest"
            )
        output_values = predicted["value"].to_numpy(dtype=np.float64)
        if not np.isfinite(output_values).all():
            raise ValueError("response prediction contains non-finite outputs")
        probability = predicted["output_kind"].isin(
            ("path_probability", "group_probability")
        ).to_numpy()
        if bool(
            ((output_values[probability] < 0) | (output_values[probability] > 1)).any()
        ):
            raise ValueError("response probability output lies outside [0, 1]")
        predicted.insert(0, "response_point_index", point_index)
        predicted["cell_id"] = point.cell_id
        predicted["perturbation_kind"] = point.perturbation_kind
        predicted["input_kind"] = point.input_kind
        predicted["input_id"] = point.input_id
        predicted["input_value"] = point.input_value
        predicted["affected_gate_key_ids"] = [
            list(point.affected_gate_key_ids)
        ] * len(predicted)
        predicted["affected_event_ids"] = [list(point.affected_event_ids)] * len(
            predicted
        )
        predicted["support_status"] = point.support_status
        predicted["primary_supported_claim_allowed"] = (
            point.support_status == "supported_model_counterfactual"
        )
        predicted["claim_semantics"] = (
            "fixed_model_prediction_sensitivity_not_causal_or_kinetic"
        )
        response_frames.append(predicted)
    return PerturbationResponseCurve(
        gate_records=pd.concat(gate_frames, ignore_index=True),
        response_records=pd.concat(response_frames, ignore_index=True),
    )


def gauge_invariant_counterfactual_outputs(
    full_logits: np.ndarray | torch.Tensor,
    counterfactual_logits: np.ndarray | torch.Tensor,
    *,
    observational_group_indices: Sequence[Sequence[int]] | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Return path/group probability effects and the sole permitted logit gauge."""

    if full_logits.shape != counterfactual_logits.shape:
        raise ValueError("full and counterfactual logits have different shapes")
    if isinstance(full_logits, torch.Tensor):
        full_log_probability = torch.log_softmax(full_logits, dim=-1)
        counter_log_probability = torch.log_softmax(counterfactual_logits, dim=-1)
        full_probability = full_log_probability.exp()
        counter_probability = counter_log_probability.exp()
        delta_logits = full_logits - counterfactual_logits
        centered = delta_logits - delta_logits.mean(dim=-1, keepdim=True)
    else:
        full_values = np.asarray(full_logits, dtype=np.float64)
        counter_values = np.asarray(counterfactual_logits, dtype=np.float64)
        full_log_probability = full_values - logsumexp(full_values, axis=-1, keepdims=True)
        counter_log_probability = counter_values - logsumexp(counter_values, axis=-1, keepdims=True)
        full_probability = np.exp(full_log_probability)
        counter_probability = np.exp(counter_log_probability)
        delta_logits = full_values - counter_values
        centered = delta_logits - delta_logits.mean(axis=-1, keepdims=True)
    result: dict[str, np.ndarray | torch.Tensor] = {
        "delta_log_path_probability": full_log_probability - counter_log_probability,
        "delta_path_probability": full_probability - counter_probability,
        "centered_delta_path_logit": centered,
    }
    if observational_group_indices is not None:
        if any(not group for group in observational_group_indices):
            raise ValueError("observational groups must be non-empty")
        if isinstance(full_logits, torch.Tensor):
            full_group_log = torch.stack(
                [torch.logsumexp(full_log_probability[..., list(group)], dim=-1) for group in observational_group_indices], dim=-1
            )
            counter_group_log = torch.stack(
                [torch.logsumexp(counter_log_probability[..., list(group)], dim=-1) for group in observational_group_indices], dim=-1
            )
        else:
            full_group_log = np.stack(
                [logsumexp(full_log_probability[..., list(group)], axis=-1) for group in observational_group_indices], axis=-1
            )
            counter_group_log = np.stack(
                [logsumexp(counter_log_probability[..., list(group)], axis=-1) for group in observational_group_indices], axis=-1
            )
        result["delta_log_group_probability"] = full_group_log - counter_group_log
        result["delta_group_probability"] = (
            full_group_log.exp() - counter_group_log.exp()
            if isinstance(full_group_log, torch.Tensor)
            else np.exp(full_group_log) - np.exp(counter_group_log)
        )
    return result


def validate_ont_matrix_identity(
    matrix_rows: pd.DataFrame,
    matrix_cells: pd.DataFrame,
    crosswalk: pd.DataFrame,
    model_paths: pd.DataFrame,
) -> OntMatrixIdentity:
    """Fail closed on every static matrix/transcript/path identity error."""

    _require_columns(matrix_rows, ("matrix_row_id", "transcript_id", "gene_id"), "ONT matrix rows")
    _require_columns(matrix_cells, ("cell_id",), "ONT matrix cells")
    _require_columns(crosswalk, ("matrix_row_id", "transcript_id", "gene_id", "path_id"), "ONT crosswalk")
    _require_columns(
        model_paths, ("gene_id", "path_id", "transcript_aliases"), "model paths"
    )
    for frame, column, label in (
        (matrix_rows, "matrix_row_id", "matrix row"),
        (matrix_rows, "transcript_id", "matrix transcript"),
        (matrix_cells, "cell_id", "matrix cell"),
        (crosswalk, "matrix_row_id", "crosswalk matrix row"),
        (crosswalk, "transcript_id", "crosswalk transcript"),
    ):
        _require_unique(frame, column, label)
    _require_unique(model_paths.assign(_key=model_paths["gene_id"].astype(str) + "\0" + model_paths["path_id"].astype(str)), "_key", "model gene/path")

    rows = matrix_rows.copy()
    cells = matrix_cells.copy()
    mapping = crosswalk.copy()
    paths = model_paths.copy()
    for frame, columns in (
        (rows, ("matrix_row_id", "transcript_id", "gene_id")),
        (mapping, ("matrix_row_id", "transcript_id", "gene_id", "path_id")),
        (paths, ("gene_id", "path_id")),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)
    cells["cell_id"] = cells["cell_id"].astype(str)
    alias_owner: dict[tuple[str, str], str] = {}
    for row in paths.itertuples(index=False):
        aliases = list(map(str, _as_list(row.transcript_aliases)))
        if not aliases or len(aliases) != len(set(aliases)) or any(not alias for alias in aliases):
            raise ValueError("each structural path requires unique non-empty transcript aliases")
        for alias in aliases:
            key = (str(row.gene_id), alias)
            if key in alias_owner:
                raise ValueError(
                    "a transcript alias belongs to multiple structural paths"
                )
            alias_owner[key] = str(row.path_id)
    joined = rows.merge(mapping, on="matrix_row_id", how="outer", suffixes=("_matrix", "_crosswalk"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("every ONT matrix row must resolve exactly once in the crosswalk")
    if not joined["transcript_id_matrix"].eq(joined["transcript_id_crosswalk"]).all() or not joined["gene_id_matrix"].eq(joined["gene_id_crosswalk"]).all():
        raise ValueError("ONT crosswalk transcript/gene identity differs from matrix rows")
    if mapping.duplicated(["gene_id", "path_id"]).any():
        raise ValueError("two ONT matrix transcripts map to one structural path")
    path_keys = set(zip(paths["gene_id"], paths["path_id"]))
    mapped_path_keys = {
        (row.gene_id, row.path_id) for row in mapping.itertuples(index=False)
    }
    missing_paths = sorted(mapped_path_keys - path_keys)
    if missing_paths:
        raise ValueError(f"ONT transcripts map to paths absent from live model: {missing_paths[:5]}")
    extra_paths = sorted(path_keys - mapped_path_keys)
    if extra_paths:
        raise ValueError(
            "live model contains paths absent from the ONT matrix isoform axis: "
            f"{extra_paths[:5]}"
        )
    for row in mapping.itertuples(index=False):
        owner = alias_owner.get((str(row.gene_id), str(row.transcript_id)))
        if owner is None:
            raise ValueError(
                "ONT crosswalk transcript is absent from model transcript_aliases"
            )
        if owner != str(row.path_id):
            raise ValueError(
                "ONT crosswalk transcript maps to the wrong structural path alias"
            )
    mapped_aliases = set(zip(mapping["gene_id"], mapping["transcript_id"]))
    extra_aliases = sorted(set(alias_owner) - mapped_aliases)
    if extra_aliases:
        raise ValueError(
            "model transcript_aliases contain isoforms absent from the ONT matrix: "
            f"{extra_aliases[:5]}"
        )
    return OntMatrixIdentity(
        matrix_rows=rows.sort_values("matrix_row_id", kind="mergesort").reset_index(drop=True),
        matrix_cells=cells.sort_values("cell_id", kind="mergesort").reset_index(drop=True),
        crosswalk=mapping.sort_values("matrix_row_id", kind="mergesort").reset_index(drop=True),
        model_paths=paths.sort_values(["gene_id", "path_id"], kind="mergesort").reset_index(drop=True),
    )


def build_ont_matrix_scope(
    identity: OntMatrixIdentity,
    likelihood_candidates: pd.DataFrame,
    matrix_counts: pd.DataFrame,
) -> OntMatrixScope:
    """Freeze candidate cell-genes and the two mutually exclusive exclusions."""

    _require_columns(likelihood_candidates, ("cell_id", "gene_id", "split"), "likelihood candidates")
    _require_columns(matrix_counts, ("cell_id", "matrix_row_id", "count"), "ONT matrix counts")
    candidates = likelihood_candidates[["cell_id", "gene_id", "split"]].copy()
    for column in ("cell_id", "gene_id", "split"):
        candidates[column] = candidates[column].astype(str)
    split_counts = candidates.groupby(["cell_id", "gene_id"], sort=False)["split"].nunique()
    if bool((split_counts > 1).any()):
        raise ValueError("a likelihood candidate cell-gene is assigned to conflicting splits")
    if candidates.duplicated(["cell_id", "gene_id"]).any():
        raise ValueError("likelihood candidate cell-gene coordinates are duplicated")
    candidates = candidates.sort_values(
        ["split", "cell_id", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)
    counts = matrix_counts.copy()
    counts["cell_id"] = counts["cell_id"].astype(str)
    counts["matrix_row_id"] = counts["matrix_row_id"].astype(str)
    values = counts["count"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or bool((values < 0).any()) or not np.equal(values, np.floor(values)).all():
        raise ValueError("ONT matrix counts must be finite non-negative integers")
    if counts.duplicated(["cell_id", "matrix_row_id"]).any():
        raise ValueError("ONT matrix count coordinates are duplicated")
    unknown_cells = sorted(set(counts["cell_id"]) - set(identity.matrix_cells["cell_id"]))
    unknown_rows = sorted(set(counts["matrix_row_id"]) - set(identity.matrix_rows["matrix_row_id"]))
    if unknown_cells or unknown_rows:
        raise ValueError("ONT matrix counts reference cells or rows outside frozen identity")
    missing_candidate_cells = sorted(set(candidates["cell_id"]) - set(identity.matrix_cells["cell_id"]))
    if missing_candidate_cells:
        raise ValueError(f"likelihood candidate cells are absent from ONT matrix: {missing_candidate_cells[:5]}")

    count_lookup = counts.set_index(["cell_id", "matrix_row_id"])["count"].to_dict()
    records: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        gene_rows = identity.crosswalk.loc[identity.crosswalk["gene_id"].eq(candidate.gene_id)].sort_values("transcript_id", kind="mergesort")
        if gene_rows.empty:
            raise ValueError(f"candidate gene {candidate.gene_id} has no ONT matrix rows")
        transcript_ids = gene_rows["transcript_id"].tolist()
        path_ids = gene_rows["path_id"].tolist()
        row_ids = gene_rows["matrix_row_id"].tolist()
        gene_counts = [int(count_lookup.get((candidate.cell_id, row_id), 0)) for row_id in row_ids]
        total = int(sum(gene_counts))
        positive = int(sum(value > 0 for value in gene_counts))
        exclusion = (
            "ont_count_total_zero"
            if total == 0
            else "fewer_than_two_positive_matrix_transcripts"
            if positive < 2
            else "eligible"
        )
        records.append(
            {
                "cell_id": candidate.cell_id,
                "gene_id": candidate.gene_id,
                "split": candidate.split,
                "scope_status": exclusion,
                "ont_count_total": total,
                "positive_matrix_transcript_count": positive,
                "matrix_row_ids": row_ids,
                "transcript_ids": transcript_ids,
                "path_ids": path_ids,
                "counts": gene_counts,
            }
        )
    table = pd.DataFrame(records)
    conservation: list[dict[str, object]] = []
    for split, rows in table.groupby("split", sort=True):
        branch_counts = rows["scope_status"].value_counts().to_dict()
        branch_mass = rows.groupby("scope_status")["ont_count_total"].sum().to_dict()
        conservation.append(
            {
                "split": split,
                "candidate_cell_gene_count": len(rows),
                "zero_total_count": int(branch_counts.get("ont_count_total_zero", 0)),
                "fewer_than_two_positive_count": int(branch_counts.get("fewer_than_two_positive_matrix_transcripts", 0)),
                "eligible_count": int(branch_counts.get("eligible", 0)),
                "candidate_raw_count_mass": int(rows["ont_count_total"].sum()),
                "zero_total_raw_count_mass": int(branch_mass.get("ont_count_total_zero", 0)),
                "fewer_than_two_positive_raw_count_mass": int(branch_mass.get("fewer_than_two_positive_matrix_transcripts", 0)),
                "eligible_raw_count_mass": int(branch_mass.get("eligible", 0)),
                "cell_gene_conservation_pass": len(rows) == sum(branch_counts.values()),
                "raw_count_conservation_pass": int(rows["ont_count_total"].sum()) == int(sum(branch_mass.values())),
            }
        )
    return OntMatrixScope(table, pd.DataFrame(conservation))


def _lexicographic_top_winner(
    labels: Sequence[str], values: Sequence[float]
) -> tuple[str, set[str]]:
    """Return the lexicographically first stable ID among exact top ties."""

    label_array = np.asarray(labels, dtype=str)
    value_array = np.asarray(values)
    if len(label_array) == 0 or label_array.shape != value_array.shape:
        raise ValueError("top-1 labels and values require the same nonempty axis")
    tied = set(label_array[np.flatnonzero(value_array == value_array.max())])
    return min(tied), tied


def compute_ont_matrix_agreement(
    identity: OntMatrixIdentity,
    scope: OntMatrixScope,
    path_logits: pd.DataFrame,
    *,
    compatible_ec_rows: pd.DataFrame | None = None,
) -> OntMatrixAgreement:
    """Compare matrix counts and probabilities on the identical isoform axis."""

    _require_columns(path_logits, ("cell_id", "gene_id", "path_id", "logit"), "path logits")
    predictions = path_logits.copy()
    for column in ("cell_id", "gene_id", "path_id"):
        predictions[column] = predictions[column].astype(str)
    if predictions.duplicated(["cell_id", "gene_id", "path_id"]).any():
        raise ValueError("path predictions contain duplicate cell/gene/path rows")
    if not np.isfinite(predictions["logit"].to_numpy(dtype=np.float64)).all():
        summary = pd.DataFrame([{"numerical_status": "numerically_invalid"}])
        return OntMatrixAgreement(pd.DataFrame(), summary, pd.DataFrame(), pd.DataFrame(), "numerically_invalid")

    eligible = scope.candidates.loc[scope.candidates["scope_status"].eq("eligible")]
    records: list[dict[str, object]] = []
    for candidate in eligible.itertuples(index=False):
        expected_paths = identity.model_paths.loc[
            identity.model_paths["gene_id"].eq(candidate.gene_id), "path_id"
        ].astype(str).tolist()
        observed = predictions.loc[
            predictions["cell_id"].eq(candidate.cell_id)
            & predictions["gene_id"].eq(candidate.gene_id)
        ]
        if set(observed["path_id"]) != set(expected_paths) or len(observed) != len(expected_paths):
            raise ValueError("each matrix candidate requires the complete live model path axis")
        observed = observed.set_index("path_id").loc[expected_paths]
        logits = observed["logit"].to_numpy(dtype=np.float64)
        log_probability = logits - logsumexp(logits)
        log_by_path = dict(zip(expected_paths, log_probability, strict=True))
        transcript_ids = list(map(str, candidate.transcript_ids))
        mapped_paths = list(map(str, candidate.path_ids))
        order = np.argsort(np.asarray(transcript_ids), kind="stable")
        transcript_ids = [transcript_ids[index] for index in order]
        mapped_paths = [mapped_paths[index] for index in order]
        counts = np.asarray(candidate.counts, dtype=np.float64)[order]
        q = counts / counts.sum()
        matrix_log_probability = np.asarray([log_by_path[path] for path in mapped_paths])
        positive = q > 0
        entropy = float(-np.sum(q[positive] * np.log(q[positive])))
        matrix_ce = float(-np.sum(q[positive] * matrix_log_probability[positive]))
        matrix_kl = matrix_ce - entropy
        prism_probability = np.exp(matrix_log_probability)
        prism_ce = float(-np.sum(q[positive] * np.log(np.maximum(prism_probability[positive], 1.0e-12))))
        prism_d = prism_ce - entropy

        observed_winner, observed_top = _lexicographic_top_winner(
            transcript_ids, counts
        )
        predicted_winner, predicted_top = _lexicographic_top_winner(
            transcript_ids, matrix_log_probability
        )
        records.append(
            {
                "cell_id": candidate.cell_id,
                "gene_id": candidate.gene_id,
                "split": candidate.split,
                "ont_count_total": int(candidate.ont_count_total),
                "ont_top1_hit": float(predicted_winner == observed_winner),
                "ont_top1_tie_aware_hit": float(predicted_winner in observed_top),
                "ont_unique_top1_hit": float(predicted_winner == observed_winner) if len(observed_top) == 1 else np.nan,
                "observed_top_tie": len(observed_top) > 1,
                "predicted_top_tie": len(predicted_top) > 1,
                "ont_matrix_cross_entropy": matrix_ce,
                "ont_matrix_kl": matrix_kl,
                "ont_cross_entropy_prism_clamped": prism_ce,
                "ont_kl_prism_clamped": prism_d,
                "ont_entropy": entropy,
            }
        )
    record_table = pd.DataFrame(records)
    summary = _aggregate_ont_records(record_table)
    compatible_records = pd.DataFrame()
    compatible_summary = pd.DataFrame()
    if compatible_ec_rows is not None:
        compatible_records, compatible_summary = compute_compatible_set_diagnostics(
            predictions, compatible_ec_rows
        )
    return OntMatrixAgreement(
        record_table,
        summary,
        compatible_records,
        compatible_summary,
        "valid",
    )


def compute_compatible_set_diagnostics(
    path_logits: pd.DataFrame,
    ec_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy-like diagnostics that retain ambiguous compatible sets."""

    _require_columns(path_logits, ("cell_id", "gene_id", "path_id", "logit"), "path logits")
    _require_columns(ec_rows, ("cell_id", "gene_id", "compatible_path_ids", "molecule_count"), "compatible EC rows")
    prediction_groups = {
        key: group.sort_values("path_id", kind="mergesort")
        for key, group in path_logits.assign(
            cell_id=path_logits["cell_id"].astype(str),
            gene_id=path_logits["gene_id"].astype(str),
            path_id=path_logits["path_id"].astype(str),
        ).groupby(["cell_id", "gene_id"], sort=False)
    }
    rows: list[dict[str, object]] = []
    for ec in ec_rows.itertuples(index=False):
        key = (str(ec.cell_id), str(ec.gene_id))
        if key not in prediction_groups:
            raise ValueError(f"EC row has no fixed prediction for {key}")
        prediction = prediction_groups[key]
        path_ids = prediction["path_id"].tolist()
        compatible_ids = list(map(str, _as_list(ec.compatible_path_ids)))
        if len(compatible_ids) != len(set(compatible_ids)):
            raise ValueError("compatible EC path IDs must not contain duplicates")
        compatible = set(compatible_ids)
        molecule_mass = float(ec.molecule_count)
        if not np.isfinite(molecule_mass) or molecule_mass <= 0:
            raise ValueError(
                "compatible EC molecule_count must be finite and strictly positive"
            )
        if not compatible.issubset(path_ids):
            raise ValueError("compatible EC contains path outside prediction axis")
        if not compatible or compatible == set(path_ids):
            raise ValueError(
                "compatible-set diagnostics accept likelihood-informative proper subsets only"
            )
        logits = prediction["logit"].to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("compatible-set diagnostics require finite logits")
        logp = logits - logsumexp(logits)
        order = sorted(range(len(path_ids)), key=lambda index: (-logits[index], path_ids[index]))
        top1 = path_ids[order[0]]
        k = min(5, len(path_ids))
        topk = {path_ids[index] for index in order[:k]}
        c = len(compatible)
        y = len(path_ids)
        chance_topk = 1.0 - (comb(y - c, k) / comb(y, k) if y - c >= k else 0.0)
        rows.append(
            {
                "cell_id": key[0],
                "gene_id": key[1],
                "compatible_path_count": c,
                "legal_path_count": y,
                "molecule_count": molecule_mass,
                "top1_in_C": float(top1 in compatible),
                "top5_in_C": float(bool(topk & compatible)),
                "singleton_EC_top1_hit": float(top1 in compatible) if c == 1 else np.nan,
                "posterior_mass_in_C": float(sum(np.exp(logp[index]) for index, path in enumerate(path_ids) if path in compatible)),
                "top1_chance_baseline": c / y,
                "top5_intersection_chance_baseline": chance_topk,
            }
        )
    table = pd.DataFrame(rows)
    summaries = []
    if len(table):
        for (compatible_count, legal_count), group in table.groupby(["compatible_path_count", "legal_path_count"], sort=True):
            for metric in ("top1_in_C", "top5_in_C", "singleton_EC_top1_hit", "posterior_mass_in_C", "top1_chance_baseline", "top5_intersection_chance_baseline"):
                valid = group[metric].notna()
                summaries.append(
                    {
                        "compatible_path_count": compatible_count,
                        "legal_path_count": legal_count,
                        "metric": metric,
                        "ec_row_denominator": int(valid.sum()),
                        "molecule_denominator": float(group.loc[valid, "molecule_count"].sum()),
                        "ec_row_macro": float(group.loc[valid, metric].mean()) if valid.any() else np.nan,
                        "molecule_weighted": _weighted_mean(group.loc[valid, metric], group.loc[valid, "molecule_count"]) if valid.any() else np.nan,
                    }
                )
    return table, pd.DataFrame(summaries)


def build_pairing_permutation_assignments(
    cell_metadata: pd.DataFrame,
    *,
    strata_fields: Sequence[str],
    seed: int,
    repetitions: int = 100,
    minimum_stratum_cells: int = 20,
    null_kind: str = "factor_activity",
) -> PairingPermutationManifest:
    """Freeze joint row permutations within legal coarse or strict strata."""

    _require_columns(cell_metadata, ("cell_id", *strata_fields), "cell metadata")
    _require_unique(cell_metadata, "cell_id", "permutation cell")
    if repetitions <= 0 or minimum_stratum_cells <= 1:
        raise ValueError("permutation repetitions and minimum stratum size must be positive")
    metadata = cell_metadata.copy()
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    rng = np.random.default_rng(seed)
    assignment_rows: list[dict[str, object]] = []
    strata_rows: list[dict[str, object]] = []
    grouped = metadata.groupby(list(strata_fields), sort=True, dropna=False)
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        stratum_id = "|".join(f"{field}={value}" for field, value in zip(strata_fields, key_values, strict=True))
        cells = np.asarray(sorted(group["cell_id"].astype(str)), dtype=object)
        estimable = len(cells) >= minimum_stratum_cells
        strata_rows.append(
            {
                "stratum_id": stratum_id,
                **dict(zip(strata_fields, key_values, strict=True)),
                "cell_count": len(cells),
                "status": "estimable" if estimable else "not_estimable",
            }
        )
        if not estimable:
            continue
        for permutation_index in range(repetitions):
            source = rng.permutation(cells)
            for target_cell, source_cell in zip(cells, source, strict=True):
                assignment_rows.append(
                    {
                        "null_kind": null_kind,
                        "permutation_index": permutation_index,
                        "stratum_id": stratum_id,
                        "target_cell_id": str(target_cell),
                        "source_cell_id": str(source_cell),
                    }
                )
    return PairingPermutationManifest(
        pd.DataFrame(assignment_rows),
        pd.DataFrame(strata_rows),
        tuple(strata_fields),
        null_kind,
        repetitions,
        int(seed),
        minimum_stratum_cells,
    )


def apply_joint_cell_permutation(
    values: pd.DataFrame,
    manifest: PairingPermutationManifest,
    *,
    permutation_index: int,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Move an entire raw evidence row and its masks with one assignment."""

    _require_columns(values, ("cell_id", *value_columns), "permuted values")
    _require_unique(values, "cell_id", "permuted value cell")
    assignment = manifest.assignments.loc[
        manifest.assignments["permutation_index"].eq(int(permutation_index))
    ]
    if assignment.empty:
        raise KeyError(f"permutation {permutation_index} has no estimable assignments")
    source = values[["cell_id", *value_columns]].copy()
    source["cell_id"] = source["cell_id"].astype(str)
    source = source.rename(columns={"cell_id": "source_cell_id"})
    result = assignment.merge(source, on="source_cell_id", how="left", validate="many_to_one")
    if result[list(value_columns)].isna().all(axis=1).any():
        raise ValueError("permutation assignment references a cell without raw evidence")
    return result.rename(columns={"target_cell_id": "cell_id"})


def aggregate_pairing_null(
    per_seed_statistics: pd.DataFrame,
    *,
    seed_ids: Sequence[int],
    repetitions: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate each permutation across seeds only after per-seed statistics."""

    required = (
        "seed",
        "null_kind",
        "permutation_index",
        "nll_permuted",
        "nll_paired",
        "t_attr_permuted",
        "t_attr_paired",
    )
    _require_columns(per_seed_statistics, required, "per-seed null statistics")
    seeds = tuple(map(int, seed_ids))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("null aggregation requires nonempty unique command seeds")
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for null_kind, group in per_seed_statistics.groupby("null_kind", sort=True):
        expected = {(seed, b) for seed in seeds for b in range(repetitions)}
        observed = set(zip(group["seed"].astype(int), group["permutation_index"].astype(int)))
        if observed != expected or len(group) != len(expected):
            raise ValueError("each null kind requires every frozen seed × permutation exactly once")
        paired_attr_by_seed = group.groupby("seed")["t_attr_paired"].nunique(dropna=False)
        paired_nll_by_seed = group.groupby("seed")["nll_paired"].nunique(dropna=False)
        if not paired_attr_by_seed.eq(1).all() or not paired_nll_by_seed.eq(1).all():
            raise ValueError("paired statistics must be fixed within each seed")
        for permutation_index, permutation in group.groupby("permutation_index", sort=True):
            t_nll = permutation["nll_permuted"].to_numpy(float) - permutation["nll_paired"].to_numpy(float)
            rows.append(
                {
                    "null_kind": null_kind,
                    "permutation_index": int(permutation_index),
                    "T_NLL": float(np.median(t_nll)),
                    "T_attr": float(np.median(permutation["t_attr_permuted"].to_numpy(float))),
                }
            )
        aggregate = pd.DataFrame(rows).loc[lambda frame: frame["null_kind"].eq(null_kind)]
        paired_attr = float(np.median(group.groupby("seed")["t_attr_paired"].first().to_numpy(float)))
        if not np.isfinite(paired_attr) or paired_attr == 0 or not np.isfinite(aggregate[["T_NLL", "T_attr"]]).all().all():
            status = "not_estimable"
            gate = False
        else:
            positive_count = int(aggregate["T_NLL"].gt(0).sum())
            required_positive_count = ceil(0.95 * repetitions)
            empirical_95 = float(np.quantile(aggregate["T_attr"], 0.95, method="higher"))
            gate = (
                positive_count >= required_positive_count
                and paired_attr > empirical_95
            )
            status = "pass" if gate else "fail"
        summaries.append(
            {
                "null_kind": null_kind,
                "status": status,
                "claim_admission_pass": gate,
                "positive_T_NLL_count": int(aggregate["T_NLL"].gt(0).sum()),
                "required_positive_T_NLL_count": ceil(0.95 * repetitions),
                "paired_T_attr": paired_attr,
                "null_T_NLL_median": float(aggregate["T_NLL"].median()),
                "null_T_NLL_iqr": _iqr(aggregate["T_NLL"]),
                "null_T_attr_median": float(aggregate["T_attr"].median()),
                "null_T_attr_iqr": _iqr(aggregate["T_attr"]),
                "null_T_attr_empirical_95th": float(np.quantile(aggregate["T_attr"], 0.95, method="higher")),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def compute_compatible_score_residuals(
    path_probabilities: pd.DataFrame,
    ec_rows: pd.DataFrame,
    index: PathIdentifiabilityIndex,
) -> pd.DataFrame:
    """Reproduce the held-out compatible-likelihood score residual exactly."""

    _require_columns(path_probabilities, ("cell_id", "gene_id", "path_id", "probability"), "path probabilities")
    _require_columns(ec_rows, ("cell_id", "gene_id", "compatible_path_ids", "molecule_count"), "EC rows")
    predictions = path_probabilities.copy()
    for column in ("cell_id", "gene_id", "path_id"):
        predictions[column] = predictions[column].astype(str)
    if predictions.duplicated(["cell_id", "gene_id", "path_id"]).any():
        raise ValueError("path probability records are duplicated")
    values = predictions["probability"].to_numpy(float)
    if not np.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("state residual requires finite strictly positive path probabilities")
    records: list[dict[str, object]] = []
    for (cell_id, gene_id), cell_ec in ec_rows.groupby(["cell_id", "gene_id"], sort=True):
        gene = index.genes.loc[index.genes["gene_id"].eq(str(gene_id))]
        if len(gene) != 1:
            raise ValueError(f"state residual gene {gene_id} is absent from identifiability index")
        gene = gene.iloc[0]
        path_ids = list(gene["path_ids"])
        prediction = predictions.loc[
            predictions["cell_id"].eq(str(cell_id)) & predictions["gene_id"].eq(str(gene_id))
        ].set_index("path_id")
        if set(prediction.index) != set(path_ids):
            raise ValueError("state residual prediction path axis is incomplete")
        probability = prediction.loc[path_ids, "probability"].to_numpy(float)
        if not np.isclose(probability.sum(), 1.0, atol=1.0e-8, rtol=0.0):
            raise ValueError("state residual path probabilities must sum to one")
        informative = []
        for row in cell_ec.itertuples(index=False):
            compatible_ids = list(map(str, _as_list(row.compatible_path_ids)))
            if len(compatible_ids) != len(set(compatible_ids)):
                raise ValueError(
                    "state residual compatible path IDs must not contain duplicates"
                )
            unknown = sorted(set(compatible_ids) - set(path_ids))
            if unknown:
                raise ValueError(
                    f"state residual compatible paths are absent from path axis: {unknown[:5]}"
                )
            mass = float(row.molecule_count)
            if not np.isfinite(mass) or mass < 0:
                raise ValueError(
                    "state residual molecule_count must be finite and non-negative"
                )
            if mass > 0 and compatible_ids and len(set(compatible_ids)) < len(path_ids):
                informative.append((compatible_ids, mass))
        molecule_mass = sum(mass for _, mass in informative)
        if molecule_mass <= 0:
            continue
        path_position = {path_id: position for position, path_id in enumerate(path_ids)}
        groups = index.groups.loc[
            index.groups["gene_id"].eq(str(gene_id)) & index.groups["cohort_contrast_separable"]
        ]
        for group in groups.itertuples(index=False):
            members = set(map(str, group.member_path_ids))
            group_probability = float(sum(probability[path_position[path]] for path in members))
            numerator = 0.0
            for compatible_ids, mass in informative:
                compatible = set(compatible_ids)
                denominator_probability = float(sum(probability[path_position[path]] for path in compatible))
                if denominator_probability <= 0:
                    raise ValueError("compatible probability denominator must be strictly positive")
                intersection_probability = float(
                    sum(probability[path_position[path]] for path in compatible & members)
                )
                numerator += mass * (intersection_probability / denominator_probability - group_probability)
            records.append(
                {
                    "cell_id": str(cell_id),
                    "gene_id": str(gene_id),
                    "observational_group_id": group.observational_group_id,
                    "gene_group_id": f"{gene_id}|{group.observational_group_id}",
                    "score_residual": numerator / molecule_mass,
                    "informative_molecule_mass": molecule_mass,
                    "positive_ec_row_count": len(informative),
                }
            )
    return pd.DataFrame(records)


def fit_state_residual_diagnostics(
    validation_rows: pd.DataFrame,
    *,
    state_pc_columns: Sequence[str],
    alpha_grid: Sequence[float],
    cv_folds: int = 3,
    seed: int = 0,
) -> StateResidualDiagnostics:
    """Fit the nuisance and added-state ridge diagnostics on validation only."""

    required = (
        "score_residual",
        "informative_molecule_mass",
        "gene_group_id",
        "positive_ec_row_count",
        "donor",
        "stage",
        "developmental_system",
        "cell_type",
        *state_pc_columns,
    )
    _require_columns(validation_rows, required, "validation residual rows")
    if len(validation_rows) < cv_folds or cv_folds < 2:
        raise ValueError("state-residual ridge CV requires at least cv_folds rows")
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(value < 0 or not np.isfinite(value) for value in grid):
        raise ValueError("ridge alpha grid must be finite and non-negative")
    nuisance = _fit_frozen_ridge(
        validation_rows, False, tuple(state_pc_columns), grid, cv_folds, seed
    )
    state = _fit_frozen_ridge(
        validation_rows, True, tuple(state_pc_columns), grid, cv_folds, seed
    )
    return StateResidualDiagnostics(nuisance, state)


def evaluate_state_residual_gate(
    diagnostics: StateResidualDiagnostics,
    test_rows: pd.DataFrame,
    *,
    threshold: float = 0.05,
    high_dtu_column: str | None = "high_dtu",
) -> pd.DataFrame:
    """Evaluate frozen test delta-R2 overall and, when available, high-DTU."""

    if threshold != 0.05:
        raise ValueError("V2 state-residual threshold is frozen at 0.05")
    _require_columns(test_rows, ("score_residual", "informative_molecule_mass"), "test residual rows")
    nuisance_prediction = _predict_frozen_ridge(diagnostics.nuisance, test_rows)
    state_prediction = _predict_frozen_ridge(diagnostics.state, test_rows)
    strata = [("overall", np.ones(len(test_rows), dtype=bool))]
    if high_dtu_column is not None and high_dtu_column in test_rows:
        strata.append(("high_DTU", test_rows[high_dtu_column].astype(bool).to_numpy()))
    rows = []
    y = test_rows["score_residual"].to_numpy(float)
    weights = test_rows["informative_molecule_mass"].to_numpy(float)
    for stratum, mask in strata:
        if not mask.any() or weights[mask].sum() <= 0:
            rows.append({"stratum": stratum, "status": "not_estimable", "delta_R2_state": np.nan, "cell_state_mechanism_claim_allowed": False})
            continue
        nuisance_r2 = _weighted_r2(y[mask], nuisance_prediction[mask], weights[mask])
        state_r2 = _weighted_r2(y[mask], state_prediction[mask], weights[mask])
        if not np.isfinite(nuisance_r2) or not np.isfinite(state_r2):
            rows.append(
                {
                    "stratum": stratum,
                    "status": "not_estimable",
                    "nuisance_R2": nuisance_r2,
                    "state_R2": state_r2,
                    "delta_R2_state": np.nan,
                    "threshold": threshold,
                    "cell_state_mechanism_claim_allowed": False,
                }
            )
            continue
        delta = state_r2 - nuisance_r2
        rows.append(
            {
                "stratum": stratum,
                "status": "pass" if delta <= threshold else "fail",
                "nuisance_R2": nuisance_r2,
                "state_R2": state_r2,
                "delta_R2_state": delta,
                "threshold": threshold,
                "cell_state_mechanism_claim_allowed": bool(delta <= threshold),
            }
        )
    return pd.DataFrame(rows)


def validate_training_run_manifest(
    manifest: pd.DataFrame,
    *,
    expected_conditions: Sequence[str] = RUNTIME_CONDITIONS,
) -> tuple[int, ...]:
    """Validate independently submitted single-seed, single-condition records."""

    _require_columns(manifest, ("seed", "condition"), "TrainingRunManifest")
    if manifest.duplicated(["seed", "condition"]).any():
        raise ValueError("TrainingRunManifest repeats a seed/condition")
    if manifest.empty:
        raise ValueError("TrainingRunManifest must contain at least one command")
    if any(type(value) is not int for value in manifest["seed"].tolist()):
        raise TypeError("TrainingRunManifest seeds must be integer command arguments")
    seeds = tuple(sorted(manifest["seed"].unique()))
    expected = set(map(str, expected_conditions))
    observed = set(manifest["condition"].astype(str))
    if not observed <= expected:
        raise ValueError("TrainingRunManifest contains an unknown runtime condition")
    return seeds


def summarize_attribution_seeds(
    per_seed_records: pd.DataFrame,
    *,
    seed_ids: Sequence[int],
    record_columns: Sequence[str],
    value_column: str,
    epsilon_num: float,
    effect_floor: float,
    maximum_dispersion: float,
    interaction_support_column: str | None = None,
) -> AttributionSeedSummary:
    """Compute median/IQR/sign/DQ and pairwise effect-rank correlations."""

    _require_columns(per_seed_records, ("seed", value_column, *record_columns), "per-seed attribution")
    seeds = tuple(map(int, seed_ids))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("attribution summary requires exactly three frozen seeds")
    if min(epsilon_num, effect_floor, maximum_dispersion) < 0:
        raise ValueError("attribution stability thresholds must be non-negative")
    records = per_seed_records.copy()
    if "model_injection_scope" in records:
        records["primary_summary_eligible"] = records["model_injection_scope"].isin(PRIMARY_INJECTION_SCOPES)
    else:
        records["primary_summary_eligible"] = True
    summary_rows: list[dict[str, object]] = []
    grouper = list(record_columns) if len(record_columns) > 1 else record_columns[0]
    for key, group in records.groupby(grouper, sort=True, dropna=False):
        if set(group["seed"].astype(int)) != set(seeds) or len(group) != 3:
            raise ValueError("every attribution record requires exactly the three frozen seeds")
        values = group.set_index(group["seed"].astype(int)).loc[list(seeds), value_column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite attribution invalidates the record")
        median = float(np.median(values))
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        positive = int(np.sum(values > epsilon_num))
        negative = int(np.sum(values < -epsilon_num))
        sign_agreement = max(positive, negative) / 3.0
        dispersion = iqr / max(abs(median), effect_floor)
        direction_status = "stable_direction" if sign_agreement == 1.0 else "seed_unstable"
        magnitude_status = (
            "below_effect_reporting_floor"
            if abs(median) < effect_floor
            else "magnitude_seed_unstable"
            if dispersion > maximum_dispersion
            else "stable_magnitude"
        )
        key_values = key if isinstance(key, tuple) else (key,)
        summary_rows.append(
            {
                **dict(zip(record_columns, key_values, strict=True)),
                "per_seed_values": values.tolist(),
                "across_seed_median": median,
                "across_seed_iqr": iqr,
                "sign_agreement": sign_agreement,
                "D_Q": dispersion,
                "direction_status": direction_status,
                "magnitude_status": magnitude_status,
                "primary_summary_eligible": bool(group["primary_summary_eligible"].all()),
            }
        )
    rank_rows: list[dict[str, object]] = []
    primary = records.loc[records["primary_summary_eligible"]].copy()
    record_key = primary[list(record_columns)].astype(str).agg("\0".join, axis=1) if len(primary) else pd.Series(dtype=str)
    primary = primary.assign(_record_key=record_key)
    strata = [("all", primary)]
    if interaction_support_column is not None:
        _require_columns(primary, (interaction_support_column,), "interaction-stratified attribution")
        strata.extend(
            (str(value), stratum_rows)
            for value, stratum_rows in primary.groupby(
                interaction_support_column, sort=True
            )
        )
    for stratum, stratum_rows in strata:
        if stratum != "all":
            stratum_rows = primary.loc[primary[interaction_support_column].astype(str).eq(stratum)]
        pivot = stratum_rows.pivot(index="_record_key", columns="seed", values=value_column)
        for left, right in combinations(seeds, 2):
            common = pivot[[left, right]].dropna() if left in pivot and right in pivot else pd.DataFrame()
            if (
                len(common) >= 2
                and common[left].nunique() > 1
                and common[right].nunique() > 1
            ):
                statistic = float(spearmanr(common[left], common[right]).statistic)
            else:
                statistic = np.nan
            rank_rows.append({"interaction_support_stratum": stratum, "seed_left": left, "seed_right": right, "record_count": len(common), "spearman_r": statistic})
    return AttributionSeedSummary(pd.DataFrame(summary_rows), pd.DataFrame(rank_rows))


def summarize_between_state_effects(
    per_cell_records: pd.DataFrame,
    *,
    seed_ids: Sequence[int],
    state_pairs: Sequence[tuple[str, str]],
    record_columns: Sequence[str],
    value_column: str,
    minimum_state_cells: int,
    epsilon_num: float,
    effect_floor: float,
    maximum_dispersion: float,
) -> pd.DataFrame:
    """Compare equal-cell-weight state medians on a seed-invariant cell set."""

    required = ("seed", "cell_id", "reporting_state", value_column, *record_columns)
    _require_columns(per_cell_records, required, "between-state attribution")
    seeds = tuple(map(int, seed_ids))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("between-state summary requires three frozen seeds")
    rows = per_cell_records.copy()
    if "eligible" in rows:
        rows = rows.loc[rows["eligible"].astype(bool)]
    results: list[dict[str, object]] = []
    grouper = list(record_columns) if len(record_columns) > 1 else record_columns[0]
    for key, group in rows.groupby(grouper, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        for state_a, state_b in state_pairs:
            state_cells: dict[str, set[str]] = {}
            invariant = True
            for state in (state_a, state_b):
                by_seed = [
                    set(group.loc[group["seed"].eq(seed) & group["reporting_state"].eq(state), "cell_id"].astype(str))
                    for seed in seeds
                ]
                invariant &= all(value == by_seed[0] for value in by_seed[1:])
                state_cells[state] = by_seed[0]
            base = {**dict(zip(record_columns, key_values, strict=True)), "state_a": state_a, "state_b": state_b, "state_a_cell_count": len(state_cells[state_a]), "state_b_cell_count": len(state_cells[state_b])}
            if not invariant:
                raise ValueError("between-state eligible cell IDs must be identical across seeds")
            if min(len(state_cells[state_a]), len(state_cells[state_b])) < minimum_state_cells:
                results.append({**base, "status": "state_contrast_not_estimable", "per_seed_contrasts": None})
                continue
            contrasts = []
            theta_a = []
            theta_b = []
            for seed in seeds:
                a = group.loc[group["seed"].eq(seed) & group["reporting_state"].eq(state_a), value_column].to_numpy(float)
                b = group.loc[group["seed"].eq(seed) & group["reporting_state"].eq(state_b), value_column].to_numpy(float)
                theta_a.append(float(np.median(a)))
                theta_b.append(float(np.median(b)))
                contrasts.append(theta_a[-1] - theta_b[-1])
            median = float(np.median(contrasts))
            direction = all(value > epsilon_num for value in contrasts) or all(value < -epsilon_num for value in contrasts)
            effect_pass = abs(median) >= effect_floor
            dispersion = _iqr(contrasts) / max(abs(median), effect_floor)
            magnitude_pass = dispersion <= maximum_dispersion
            status = (
                "direction_not_stable"
                if not direction
                else "below_state_effect_reporting_floor"
                if not effect_pass
                else "magnitude_seed_unstable"
                if not magnitude_pass
                else "stable_state_difference"
            )
            results.append(
                {
                    **base,
                    "theta_state_a_per_seed": theta_a,
                    "theta_state_b_per_seed": theta_b,
                    "per_seed_contrasts": contrasts,
                    "across_seed_median_contrast": median,
                    "D_state_Q": dispersion,
                    "direction_pass": direction,
                    "effect_floor_pass": effect_pass,
                    "magnitude_pass": magnitude_pass,
                    "status": status,
                }
            )
    return pd.DataFrame(results)


def summarize_predictive_seeds(
    metrics: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Optimization-repeat summaries: mean, sample SD, and range, never CI."""

    _require_columns(metrics, ("seed", *group_columns, *metric_columns), "seed metrics")
    rows = []
    grouper = list(group_columns) if len(group_columns) > 1 else group_columns[0]
    for key, group in metrics.groupby(grouper, sort=True, dropna=False):
        if group["seed"].nunique() != 3 or len(group) != 3:
            raise ValueError("predictive seed summary requires exactly three values per group")
        key_values = key if isinstance(key, tuple) else (key,)
        for metric in metric_columns:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    **dict(zip(group_columns, key_values, strict=True)),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "range_min": float(np.min(values)),
                    "range_max": float(np.max(values)),
                    "uncertainty_semantics": "optimization_repeat_not_biological_confidence_interval",
                }
            )
    return pd.DataFrame(rows)


def explanation_manifest_coverage(
    eligible_records: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    selection_rule: str,
) -> pd.DataFrame:
    """Validate outcome-blind selection and report fixed-denominator coverage."""

    if not selection_rule.strip():
        raise ValueError("explanation manifest requires a non-empty predeclared rule")
    _require_columns(eligible_records, key_columns, "eligible explanation records")
    _require_columns(manifest, key_columns, "explanation manifest")
    forbidden_tokens = ("test_effect", "effect_magnitude", "delta_rho", "prediction_correct", "test_nll", "support_tier", "direct_cell_supported")
    forbidden = [column for column in manifest if any(token in column.lower() for token in forbidden_tokens)]
    if forbidden:
        raise ValueError(f"explanation manifest contains held-out outcome/support fields: {forbidden}")
    eligible = eligible_records.copy()
    selected = manifest[list(key_columns)].drop_duplicates()
    if len(selected.merge(eligible[list(key_columns)].drop_duplicates(), on=list(key_columns), how="left", indicator=True).query("_merge != 'both'")):
        raise ValueError("explanation manifest selects records outside the eligible universe")
    joined = eligible.merge(selected.assign(_selected=True), on=list(key_columns), how="left")
    joined["_selected"] = joined["_selected"].eq(True)
    rows = []
    scopes = [("all", joined)]
    if "split" in joined:
        scopes.extend((str(split), group) for split, group in joined.groupby("split", sort=True))
    for scope, group in scopes:
        denominator = len(group)
        count = int(group["_selected"].sum())
        rows.append(
            {
                "scope": scope,
                "eligible_denominator": denominator,
                "manifest_selected_count": count,
                "selection_coverage": count / denominator if denominator else np.nan,
                "status": "estimable" if denominator else "not_estimable",
                "selection_rule": selection_rule,
            }
        )
    return pd.DataFrame(rows)


def build_path_scale_audit(
    model: FABRICV2Model,
    model_input: GeneCellModelInput,
    *,
    gene_id: str,
    cell_ids: Sequence[str],
    path_ids: Sequence[str],
    compatible_path_indices: torch.Tensor,
    compatible_path_mask: torch.Tensor,
    row_cell_index: torch.Tensor,
    molecule_count: torch.Tensor,
    alternative_contrasts: pd.DataFrame,
    representation_collision_relative_tolerance: float,
    condition: str = "full",
) -> PathScaleAuditResult:
    """Construct PathScaleAudit rows directly from one differentiable forward."""

    cells = _unique_axis(cell_ids, model_input.dna.gate.shape[0], "path-scale cell")
    paths = _unique_axis(
        path_ids, model_input.path_edge_incidence.shape[0], "path-scale path"
    )
    if model.readout_kind != "path_context":
        raise ValueError("PathScaleAudit requires the PathContextReadout model")
    if condition not in {"cis", "cis_dna", "cis_rna", "full"}:
        raise ValueError("PathScaleAudit condition is not a primary modality condition")
    if (
        isinstance(representation_collision_relative_tolerance, bool)
        or not isinstance(representation_collision_relative_tolerance, (int, float))
        or not np.isfinite(representation_collision_relative_tolerance)
        or representation_collision_relative_tolerance <= 0
    ):
        raise ValueError(
            "representation-collision tolerance must be finite and positive"
        )
    if compatible_path_indices.shape != compatible_path_mask.shape:
        raise ValueError("PathScaleAudit compatible indices/mask shapes differ")
    if compatible_path_indices.ndim != 2 or compatible_path_mask.dtype != torch.bool:
        raise ValueError("PathScaleAudit compatible tensors require padded long/bool rows")
    row_count = compatible_path_indices.shape[0]
    if row_cell_index.shape != (row_count,) or molecule_count.shape != (row_count,):
        raise ValueError("PathScaleAudit EC rows, cells, and molecule mass are misaligned")
    if compatible_path_indices.dtype != torch.long or row_cell_index.dtype != torch.long:
        raise TypeError("PathScaleAudit compatible and cell indices must use torch.long")
    if bool((row_cell_index < 0).any()) or bool((row_cell_index >= len(cells)).any()):
        raise IndexError("PathScaleAudit EC row cell index is out of range")
    masses = molecule_count.detach().to(dtype=torch.float64)
    if not torch.isfinite(masses).all() or bool((masses <= 0).any()):
        raise ValueError("PathScaleAudit informative molecule mass must be finite and positive")
    if not compatible_path_mask.any(dim=1).all():
        raise ValueError("PathScaleAudit accepts informative non-empty EC rows only")
    valid_indices = compatible_path_indices[compatible_path_mask]
    if bool((valid_indices < 0).any()) or bool((valid_indices >= len(paths)).any()):
        raise IndexError("PathScaleAudit compatible path index is out of range")
    for row in range(row_count):
        values = compatible_path_indices[row, compatible_path_mask[row]].tolist()
        if len(values) != len(set(values)):
            raise ValueError("PathScaleAudit compatible sets cannot repeat paths")

    _require_columns(
        alternative_contrasts,
        ("contrast_id", "gene_id", "numerator_path_ids", "denominator_path_ids"),
        "PathScaleAudit alternative contrasts",
    )
    contrasts = alternative_contrasts.loc[
        alternative_contrasts["gene_id"].astype(str).eq(str(gene_id))
    ]
    if contrasts.empty or contrasts["contrast_id"].astype(str).duplicated().any():
        raise ValueError("PathScaleAudit requires unique local contrasts for the gene")

    incidence = model_input.path_edge_incidence.to_dense().to(dtype=torch.float64)
    if incidence.shape[0] != len(paths) or not torch.logical_or(
        incidence.eq(0), incidence.eq(1)
    ).all():
        raise ValueError("PathScaleAudit requires binary structural path incidence")
    catalog_mean = incidence.mean(dim=0)
    variable_edge_count = int(((catalog_mean > 0) & (catalog_mean < 1)).sum())
    centered_energy = float((catalog_mean * (1.0 - catalog_mean)).sum())
    path_edge_counts = incidence.sum(dim=1)

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        output = model(model_input, condition=condition)
        if (
            output.path_residual is None
            or output.path_first_layer_preactivation is None
            or output.path_vector is None
        ):
            raise RuntimeError("PathContextReadout did not expose its named path intermediates")
        named = (
            output.path_logits,
            output.joint_projected,
            output.normalized_tokens,
            output.edge_states,
            output.path_residual,
            output.path_first_layer_preactivation,
        )
        if not all(torch.isfinite(value).all() for value in named):
            raise FloatingPointError("PathScaleAudit forward contains non-finite values")
        log_probability = torch.log_softmax(output.path_logits, dim=-1)
        probability = log_probability.exp()
        entropy = -(probability * log_probability).sum(dim=-1)
        posterior_mass = torch.zeros(row_count, dtype=probability.dtype, device=probability.device)
        for row in range(row_count):
            selected = compatible_path_indices[row, compatible_path_mask[row]].to(
                probability.device
            )
            posterior_mass[row] = probability[row_cell_index[row], selected].sum()
        gradient_norms: list[list[float]] = [[] for _ in cells]
        for contrast in contrasts.itertuples(index=False):
            rho = alternative_relative_log_mass(
                output.path_logits,
                paths,
                contrast.numerator_path_ids,
                contrast.denominator_path_ids,
            )
            for cell_index in range(len(cells)):
                gradient = torch.autograd.grad(
                    rho[cell_index],
                    output.path_residual,
                    retain_graph=True,
                    create_graph=False,
                )[0][cell_index]
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "PathScaleAudit local contrast gradient is non-finite"
                    )
                gradient_norms[cell_index].append(
                    float(torch.linalg.vector_norm(gradient))
                )
    model.train(was_training)

    rows: list[dict[str, object]] = []
    zeta_norm = torch.linalg.vector_norm(output.path_residual.detach(), dim=-1)
    preactivation_norm = torch.linalg.vector_norm(
        output.path_first_layer_preactivation.detach(), dim=-1
    )
    token_rms = torch.sqrt(output.edge_states.detach().square().mean(dim=(1, 2)))
    logits = output.path_logits.detach()
    for cell_index, cell_id in enumerate(cells):
        cell_rows = row_cell_index.eq(cell_index)
        weights = masses[cell_rows].cpu().numpy()
        compatible_mass = posterior_mass.detach()[cell_rows].double().cpu().numpy()
        if not len(weights):
            calibration = np.nan
            informative_mass = 0.0
        else:
            calibration = float(np.dot(weights, compatible_mass) / weights.sum())
            informative_mass = float(weights.sum())
        norms = zeta_norm[cell_index].cpu().numpy()
        relative = (
            float(np.median(norms) / float(token_rms[cell_index]))
            if float(token_rms[cell_index]) > 0
            else np.nan
        )
        rows.append(
            {
                "cell_id": cell_id,
                "gene_id": str(gene_id),
                "condition": condition,
                "centered_path_incidence_energy_D_g": centered_energy,
                "variable_edge_count_V_g": variable_edge_count,
                "legal_path_count": len(paths),
                "path_edge_count_min": int(path_edge_counts.min()),
                "path_edge_count_max": int(path_edge_counts.max()),
                "zeta_norm_median": float(np.median(norms)),
                "zeta_norm_q95": float(np.quantile(norms, 0.95)),
                "zeta_norm_max": float(np.max(norms)),
                "relative_token_rms": relative,
                "path_mlp_preactivation_norm": float(
                    preactivation_norm[cell_index].mean()
                ),
                "path_logit_sd": float(logits[cell_index].double().std(unbiased=False)),
                "path_logit_range": float(
                    logits[cell_index].max() - logits[cell_index].min()
                ),
                "softmax_entropy": float(entropy[cell_index]),
                "compatible_set_posterior_mass_molecule_weighted": calibration,
                "informative_molecule_mass": informative_mass,
                "local_contrast_gradient_norm": float(
                    np.max(gradient_norms[cell_index])
                ),
                "local_contrast_gradient_norm_median": float(
                    np.median(gradient_norms[cell_index])
                ),
                "prediction_seed_stability": np.nan,
                "attribution_seed_stability": np.nan,
                "seed_stability_stage": "requires_three_checkpoint_join",
                "finite_output_pass": True,
                "production_path_scaling": "unscaled_gene_centered_residual_sum",
            }
        )
    path_vectors = output.path_vector.detach().double()
    membership_count = torch.zeros(
        row_count, len(paths), dtype=torch.int64, device=compatible_path_mask.device
    )
    membership_count.scatter_add_(
        1,
        compatible_path_indices.masked_fill(~compatible_path_mask, 0),
        compatible_path_mask.to(torch.int64),
    )
    membership = membership_count > 0
    collision_rows: list[dict[str, object]] = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            # Only path pairs distinguished by at least one admitted
            # compatible row are supervision-relevant collision candidates.
            if torch.equal(membership[:, left], membership[:, right]):
                continue
            left_vector = path_vectors[:, left]
            right_vector = path_vectors[:, right]
            difference = torch.linalg.vector_norm(left_vector - right_vector, dim=1)
            denominator = torch.maximum(
                torch.maximum(
                    torch.linalg.vector_norm(left_vector, dim=1),
                    torch.linalg.vector_norm(right_vector, dim=1),
                ),
                torch.ones_like(difference),
            )
            relative = difference / denominator
            exact = torch.all(left_vector == right_vector, dim=1)
            near = relative <= float(representation_collision_relative_tolerance)
            exact_count = int(exact.sum())
            near_count = int(near.sum())
            if exact_count == len(cells):
                status = "systematic_exact_representation_collision"
            elif near_count == len(cells):
                status = "systematic_near_representation_collision"
            elif near_count:
                status = "cell_specific_near_representation_collision"
            else:
                status = "no_representation_collision_detected"
            collision_rows.append(
                {
                    "gene_id": str(gene_id),
                    "path_id_a": paths[left],
                    "path_id_b": paths[right],
                    "supervision_distinguishable": True,
                    "cell_count": len(cells),
                    "exact_collision_cell_count": exact_count,
                    "near_collision_cell_count": near_count,
                    "relative_distance_min": float(relative.min()),
                    "relative_distance_median": float(relative.median()),
                    "relative_distance_max": float(relative.max()),
                    "relative_tolerance": float(
                        representation_collision_relative_tolerance
                    ),
                    "status": status,
                }
            )
    collision_columns = (
        "gene_id",
        "path_id_a",
        "path_id_b",
        "supervision_distinguishable",
        "cell_count",
        "exact_collision_cell_count",
        "near_collision_cell_count",
        "relative_distance_min",
        "relative_distance_median",
        "relative_distance_max",
        "relative_tolerance",
        "status",
    )
    return PathScaleAuditResult(
        records=pd.DataFrame(rows),
        representation_collisions=pd.DataFrame(
            collision_rows, columns=collision_columns
        ),
        output=output,
    )


def build_event_density_token_audit(
    output: FABRICOutput,
    route_burden: pd.DataFrame,
    *,
    gene_id: str,
    cell_ids: Sequence[str],
    edge_token_ids: Sequence[str],
) -> pd.DataFrame:
    """Join direct named token norms to catalog/model-input route burden."""

    cells = _unique_axis(cell_ids, output.path_logits.shape[0], "event-density cell")
    edges = _unique_axis(edge_token_ids, output.edge_states.shape[1], "event-density edge token")
    required = (
        "audit_population",
        "target_gene_id",
        "modality",
        "edge_token_id",
        "distinct_physical_event_count",
        "distinct_active_gate_key_count",
        "saturated_anchor_group_count",
        "saturated_cap_bucket_count",
        "route_l1_mass",
        "B_gate",
    )
    _require_columns(route_burden, required, "route burden audit")
    burden = route_burden.loc[
        route_burden["target_gene_id"].astype(str).eq(str(gene_id))
    ].copy()
    allowed_population = {"catalog", "model_input"}
    if set(burden["audit_population"].astype(str)) - allowed_population:
        raise ValueError("route burden contains an unknown audit population")
    if burden.duplicated(["audit_population", "modality", "edge_token_id"]).any():
        raise ValueError("route burden repeats a population/modality/edge token")
    unknown_edges = sorted(set(burden["edge_token_id"].astype(str)) - set(edges))
    if unknown_edges:
        raise ValueError(f"route burden references unknown edge tokens: {unknown_edges[:5]}")
    named = (
        output.dna_aggregate,
        output.rna_aggregate,
        output.joint_projected,
        output.normalized_tokens,
        output.edge_states,
    )
    if not all(torch.isfinite(value).all() for value in named):
        raise FloatingPointError("event-density named intermediates contain non-finite values")
    dynamic = torch.cat((output.dna_aggregate, output.rna_aggregate), dim=-1)
    dynamic_norm = torch.linalg.vector_norm(dynamic, dim=-1).detach().cpu().numpy()
    pre_norm = torch.linalg.vector_norm(
        output.joint_projected, dim=-1
    ).detach().cpu().numpy()
    post_norm = torch.linalg.vector_norm(
        output.normalized_tokens, dim=-1
    ).detach().cpu().numpy()
    edge_state_norm = torch.linalg.vector_norm(
        output.edge_states, dim=-1
    ).detach().cpu().numpy()

    aggregated: dict[tuple[str, str], dict[str, float | int]] = {}
    for (population, edge_id), group in burden.groupby(
        ["audit_population", "edge_token_id"], sort=True
    ):
        active_gate = group["distinct_active_gate_key_count"].dropna().to_numpy(float)
        b_gate = group["B_gate"].dropna().to_numpy(float)
        aggregated[(str(population), str(edge_id))] = {
            "distinct_physical_event_count": int(
                group["distinct_physical_event_count"].sum()
            ),
            "distinct_active_gate_key_count": (
                int(active_gate.sum()) if len(active_gate) else 0
            ),
            "saturated_anchor_group_count": int(
                group["saturated_anchor_group_count"].sum()
            ),
            "saturated_cap_bucket_count": int(
                group["saturated_cap_bucket_count"].sum()
            ),
            "route_l1_mass": float(group["route_l1_mass"].sum()),
            "B_gate": float(np.sqrt(np.square(b_gate).sum())) if len(b_gate) else np.nan,
        }
    rows: list[dict[str, object]] = []
    for cell_index, cell_id in enumerate(cells):
        for edge_index, edge_id in enumerate(edges):
            row: dict[str, object] = {
                "cell_id": cell_id,
                "gene_id": str(gene_id),
                "edge_token_id": edge_id,
                "dynamic_block_norm": float(dynamic_norm[cell_index, edge_index]),
                "pre_normalization_token_norm": float(pre_norm[cell_index, edge_index]),
                "post_normalization_token_norm": float(post_norm[cell_index, edge_index]),
                "contextual_edge_state_norm": float(
                    edge_state_norm[cell_index, edge_index]
                ),
                "all_named_intermediates_finite": True,
            }
            for population in ("catalog", "model_input"):
                values = aggregated.get((population, edge_id))
                explicit_status = "observed_route_burden" if values is not None else "no_routes_on_token"
                if values is None:
                    values = {
                        "distinct_physical_event_count": 0,
                        "distinct_active_gate_key_count": 0,
                        "saturated_anchor_group_count": 0,
                        "saturated_cap_bucket_count": 0,
                        "route_l1_mass": 0.0,
                        "B_gate": np.nan,
                    }
                for key, value in values.items():
                    row[f"{population}_{key}"] = value
                row[f"{population}_token_burden_status"] = explicit_status
            rows.append(row)
    return pd.DataFrame(rows)


def compare_architecture_readouts(
    ec_records: pd.DataFrame,
    *,
    epsilon_num: float,
    strata_columns: Sequence[str] = (),
    path_context_condition: str = "full",
    additive_condition: str = "full_additive_edge",
) -> ArchitectureComparison:
    """Compare the two readouts on exactly matched held-out EC rows per seed."""

    if epsilon_num < 0 or not np.isfinite(epsilon_num):
        raise ValueError("architecture numerical tolerance must be finite and non-negative")
    required = (
        "seed",
        "condition",
        "parameter_count",
        "cell_id",
        "gene_id",
        "ec_id",
        "path_axis_identity",
        "compatible_path_ids",
        "molecule_count",
        "nll_numerator",
        "calibration_error",
        *strata_columns,
    )
    _require_columns(ec_records, required, "architecture comparison EC records")
    records = ec_records.copy()
    records["condition"] = records["condition"].astype(str)
    expected_conditions = {path_context_condition, additive_condition}
    if set(records["condition"]) != expected_conditions:
        raise ValueError("architecture comparison contains conditions outside the frozen pair")
    seeds = tuple(sorted(map(int, records["seed"].unique())))
    if len(seeds) != 3:
        raise ValueError("architecture comparison requires exactly three frozen seeds")
    key_columns = ("cell_id", "gene_id", "ec_id")
    if records.duplicated(["seed", "condition", *key_columns]).any():
        raise ValueError("architecture comparison repeats a seed/condition EC row")
    molecule_count = records["molecule_count"].to_numpy(dtype=np.float64)
    numerator = records["nll_numerator"].to_numpy(dtype=np.float64)
    calibration = records["calibration_error"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(molecule_count).all()
        or bool((molecule_count <= 0).any())
        or not np.isfinite(numerator).all()
        or bool((numerator < 0).any())
        or not np.isfinite(calibration).all()
    ):
        raise ValueError("architecture comparison contains invalid weights or metrics")
    parameter_counts: dict[str, int] = {}
    for condition, group in records.groupby("condition", sort=True):
        values = group["parameter_count"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or bool((values <= 0).any())
            or not np.equal(values, np.floor(values)).all()
            or len(set(values)) != 1
        ):
            raise ValueError(
                "architecture parameter count must be one positive integer per condition"
            )
        parameter_counts[str(condition)] = int(values[0])
    _validate_matched_evaluation_scope(
        records,
        group_columns=("seed", "condition"),
        key_columns=key_columns,
        invariant_columns=(
            "path_axis_identity",
            "compatible_path_ids",
            "molecule_count",
            *strata_columns,
        ),
        label="architecture comparison",
    )

    axes: list[tuple[str, str | None]] = [("overall", None)]
    axes.extend((str(column), str(column)) for column in strata_columns)
    output_rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_rows = records.loc[records["seed"].astype(int).eq(seed)]
        for stratum_axis, column in axes:
            levels: Sequence[object] = ("overall",) if column is None else sorted(
                seed_rows[column].drop_duplicates().tolist(), key=str
            )
            for level in levels:
                selected = seed_rows if column is None else seed_rows.loc[
                    seed_rows[column].eq(level)
                ]
                by_condition: dict[str, dict[str, float | int]] = {}
                for condition, condition_rows in selected.groupby("condition", sort=True):
                    mass = float(condition_rows["molecule_count"].sum())
                    by_condition[str(condition)] = {
                        "nll": float(condition_rows["nll_numerator"].sum() / mass),
                        "calibration_macro": float(condition_rows["calibration_error"].mean()),
                        "calibration_molecule_weighted": _weighted_mean(
                            condition_rows["calibration_error"],
                            condition_rows["molecule_count"],
                        ),
                        "ec_row_count": len(condition_rows),
                        "molecule_denominator": mass,
                    }
                if set(by_condition) != expected_conditions:
                    raise RuntimeError("matched architecture stratum lost one condition")
                full = by_condition[path_context_condition]
                additive = by_condition[additive_condition]
                if (
                    full["ec_row_count"] != additive["ec_row_count"]
                    or full["molecule_denominator"] != additive["molecule_denominator"]
                ):
                    raise RuntimeError("matched architecture denominator changed after grouping")
                output_rows.append(
                    {
                        "seed": seed,
                        "stratum_axis": stratum_axis,
                        "stratum": str(level),
                        "path_context_nll": full["nll"],
                        "additive_edge_nll": additive["nll"],
                        "delta_nll_arch": additive["nll"] - full["nll"],
                        "path_context_calibration_macro": full["calibration_macro"],
                        "additive_edge_calibration_macro": additive["calibration_macro"],
                        "path_context_calibration_molecule_weighted": full[
                            "calibration_molecule_weighted"
                        ],
                        "additive_edge_calibration_molecule_weighted": additive[
                            "calibration_molecule_weighted"
                        ],
                        "ec_row_count": full["ec_row_count"],
                        "molecule_denominator": full["molecule_denominator"],
                    }
                )
    per_seed = pd.DataFrame(output_rows)
    overall = per_seed.loc[per_seed["stratum_axis"].eq("overall")]
    if set(overall["seed"].astype(int)) != set(seeds) or len(overall) != 3:
        raise RuntimeError("architecture comparison overall rows differ from frozen seeds")
    deltas = overall.sort_values("seed")["delta_nll_arch"].to_numpy(dtype=np.float64)
    claim = pd.DataFrame(
        [
            {
                "epsilon_arch": float(epsilon_num),
                "seed_ids": list(seeds),
                "delta_nll_arch_by_seed": deltas.tolist(),
                "path_context_parameter_count": parameter_counts[
                    path_context_condition
                ],
                "additive_edge_parameter_count": parameter_counts[
                    additive_condition
                ],
                "all_three_seeds_above_epsilon": bool((deltas > epsilon_num).all()),
                "consistent_empirical_predictive_gain_allowed": bool(
                    (deltas > epsilon_num).all()
                ),
                "comparison_scope": "same_test_EC_rows_weights_and_legal_path_axes",
            }
        ]
    )
    return ArchitectureComparison(per_seed=per_seed, claim_summary=claim)


def summarize_path_scale_strata(
    prediction_records: pd.DataFrame,
    path_scale_audit: pd.DataFrame,
    *,
    strata_columns: Sequence[str],
) -> pd.DataFrame:
    """Join frozen PathScaleAudit rows and report each complexity axis separately."""

    if not strata_columns:
        raise ValueError("path-scale reporting requires frozen strata columns")
    key_columns = ("seed", "condition", "cell_id", "gene_id")
    _require_columns(
        prediction_records,
        (
            *key_columns,
            "compatible_nll_numerator",
            "informative_molecule_mass",
            "compatible_calibration_error",
        ),
        "path-scale prediction records",
    )
    audit_metrics = (
        "zeta_norm_median",
        "zeta_norm_q95",
        "zeta_norm_max",
        "relative_token_rms",
        "path_mlp_preactivation_norm",
        "path_logit_sd",
        "path_logit_range",
        "softmax_entropy",
        "local_contrast_gradient_norm",
        "prediction_seed_stability",
        "attribution_seed_stability",
    )
    _require_columns(
        path_scale_audit,
        (*key_columns, *strata_columns, *audit_metrics),
        "PathScaleAudit",
    )
    if prediction_records.duplicated(list(key_columns)).any():
        raise ValueError("path-scale prediction records repeat a gene-cell/model row")
    if path_scale_audit.duplicated(list(key_columns)).any():
        raise ValueError("PathScaleAudit repeats a gene-cell/model row")
    prediction_keys = set(
        map(tuple, prediction_records[list(key_columns)].astype(str).to_numpy())
    )
    audit_keys = set(map(tuple, path_scale_audit[list(key_columns)].astype(str).to_numpy()))
    if prediction_keys != audit_keys:
        raise ValueError("PathScaleAudit and prediction records have different exact scopes")
    records = prediction_records.merge(
        path_scale_audit,
        on=list(key_columns),
        how="inner",
        validate="one_to_one",
    )
    mass = records["informative_molecule_mass"].to_numpy(dtype=np.float64)
    numerator = records["compatible_nll_numerator"].to_numpy(dtype=np.float64)
    calibration = records["compatible_calibration_error"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(mass).all()
        or bool((mass <= 0).any())
        or not np.isfinite(numerator).all()
        or bool((numerator < 0).any())
        or not np.isfinite(calibration).all()
    ):
        raise ValueError("path-scale prediction inputs contain non-finite/invalid values")
    for metric in audit_metrics:
        values = records[metric].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"PathScaleAudit {metric} contains an infinite value")
    _validate_frozen_unit_strata(
        records,
        unit_columns=("cell_id", "gene_id"),
        strata_columns=strata_columns,
        label="PathScaleAudit",
    )
    rows: list[dict[str, object]] = []
    for axis in strata_columns:
        for (condition, seed, stratum), group in records.groupby(
            ["condition", "seed", axis], sort=True, dropna=False
        ):
            denominator = float(group["informative_molecule_mass"].sum())
            row: dict[str, object] = {
                "condition": str(condition),
                "seed": int(seed),
                "stratum_axis": str(axis),
                "stratum": str(stratum),
                "eligible_cell_gene_count": len(group),
                "gene_count": int(group["gene_id"].astype(str).nunique()),
                "informative_molecule_denominator": denominator,
                "compatible_path_nll": float(
                    group["compatible_nll_numerator"].sum() / denominator
                ),
                "compatible_calibration_macro": float(
                    group["compatible_calibration_error"].mean()
                ),
                "compatible_calibration_molecule_weighted": _weighted_mean(
                    group["compatible_calibration_error"],
                    group["informative_molecule_mass"],
                ),
                "numerical_status": "valid",
            }
            for metric in audit_metrics:
                valid = group[metric].notna()
                row[f"{metric}_mean"] = (
                    float(group.loc[valid, metric].mean()) if valid.any() else np.nan
                )
                row[f"{metric}_denominator"] = int(valid.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_event_density_strata(
    prediction_records: pd.DataFrame,
    attribution_records: pd.DataFrame,
    frozen_gene_strata: pd.DataFrame,
    *,
    strata_columns: Sequence[str] = (
        "catalog_token_burden_stratum",
        "model_input_token_burden_stratum",
    ),
) -> EventDensityTables:
    """Report prediction once per gene-cell and attribution once per record."""

    if not strata_columns:
        raise ValueError("event-density reporting requires frozen burden strata")
    _require_columns(
        frozen_gene_strata,
        ("gene_id", *strata_columns),
        "frozen event-density gene strata",
    )
    if frozen_gene_strata["gene_id"].astype(str).duplicated().any():
        raise ValueError("event-density gene strata require one row per gene")
    prediction_key = ("seed", "condition", "cell_id", "gene_id")
    _require_columns(
        prediction_records,
        (
            *prediction_key,
            "compatible_nll_numerator",
            "informative_molecule_mass",
            "dynamic_block_norm",
            "pre_normalization_token_norm",
            "post_normalization_token_norm",
        ),
        "event-density prediction records",
    )
    if prediction_records.duplicated(list(prediction_key)).any():
        raise ValueError(
            "event-density prediction records would duplicate gene-cell performance"
        )
    prediction = _join_complete_gene_strata(
        prediction_records, frozen_gene_strata, strata_columns, "event-density prediction"
    )
    mass = prediction["informative_molecule_mass"].to_numpy(dtype=np.float64)
    numerator = prediction["compatible_nll_numerator"].to_numpy(dtype=np.float64)
    norm_columns = (
        "dynamic_block_norm",
        "pre_normalization_token_norm",
        "post_normalization_token_norm",
    )
    if (
        not np.isfinite(mass).all()
        or bool((mass <= 0).any())
        or not np.isfinite(numerator).all()
        or bool((numerator < 0).any())
        or not np.isfinite(prediction[list(norm_columns)].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("event-density prediction inputs contain invalid/non-finite values")
    prediction_rows: list[dict[str, object]] = []
    for axis in strata_columns:
        for (condition, seed, stratum), group in prediction.groupby(
            ["condition", "seed", axis], sort=True, dropna=False
        ):
            denominator = float(group["informative_molecule_mass"].sum())
            prediction_rows.append(
                {
                    "condition": str(condition),
                    "seed": int(seed),
                    "stratum_axis": str(axis),
                    "stratum": str(stratum),
                    "eligible_cell_gene_count": len(group),
                    "gene_count": int(group["gene_id"].astype(str).nunique()),
                    "informative_molecule_denominator": denominator,
                    "compatible_path_nll": float(
                        group["compatible_nll_numerator"].sum() / denominator
                    ),
                    **{
                        f"{column}_mean": float(group[column].mean())
                        for column in norm_columns
                    },
                    "prediction_unit": "unique_seed_condition_cell_gene",
                }
            )

    attribution_required = (
        "condition",
        "cell_id",
        "gene_id",
        "record_id",
        "across_seed_median",
        "across_seed_iqr",
        "sign_agreement",
        "D_Q",
        "direction_status",
        "magnitude_status",
        "primary_summary_eligible",
    )
    _require_columns(
        attribution_records, attribution_required, "event-density attribution records"
    )
    if attribution_records.duplicated(
        ["condition", "cell_id", "gene_id", "record_id"]
    ).any():
        raise ValueError("event-density attribution records repeat a frozen record")
    attribution = _join_complete_gene_strata(
        attribution_records,
        frozen_gene_strata,
        strata_columns,
        "event-density attribution",
    )
    numeric = attribution[
        ["across_seed_median", "across_seed_iqr", "sign_agreement", "D_Q"]
    ].to_numpy(dtype=np.float64)
    if numeric.size and not np.isfinite(numeric).all():
        raise ValueError("event-density attribution contains non-finite stability values")
    attribution_rows: list[dict[str, object]] = []
    primary = attribution.loc[attribution["primary_summary_eligible"].astype(bool)]
    for axis in strata_columns:
        for (condition, stratum), group in primary.groupby(
            ["condition", axis], sort=True, dropna=False
        ):
            attribution_rows.append(
                {
                    "condition": str(condition),
                    "stratum_axis": str(axis),
                    "stratum": str(stratum),
                    "attribution_record_count": len(group),
                    "cell_gene_count": int(
                        group[["cell_id", "gene_id"]].drop_duplicates().shape[0]
                    ),
                    "median_absolute_effect": float(
                        np.median(np.abs(group["across_seed_median"].to_numpy(float)))
                    ),
                    "median_across_seed_iqr": float(
                        np.median(group["across_seed_iqr"].to_numpy(float))
                    ),
                    "direction_stable_count": int(
                        group["direction_status"].eq("stable_direction").sum()
                    ),
                    "magnitude_stable_count": int(
                        group["magnitude_status"].eq("stable_magnitude").sum()
                    ),
                    "median_D_Q": float(np.median(group["D_Q"].to_numpy(float))),
                    "attribution_unit": "frozen_primary_attribution_record",
                }
            )
    return EventDensityTables(
        prediction=pd.DataFrame(prediction_rows),
        attribution=pd.DataFrame(attribution_rows),
    )


def build_train_support_bin_assignments(
    train_gene_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze the three one-dimensional §16.1 support bins from train only."""

    required = (
        "gene_id",
        "matrix_path_count",
        "train_ont_raw_count",
        "train_positive_cell_support",
    )
    _require_columns(train_gene_metadata, required, "train gene support metadata")
    metadata = train_gene_metadata[list(required)].copy()
    metadata["gene_id"] = metadata["gene_id"].astype(str)
    if metadata["gene_id"].duplicated().any() or metadata.empty:
        raise ValueError("train gene support metadata requires unique non-empty genes")
    values = metadata[list(required[1:])].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or bool((values <= 0).any())
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("support-bin inputs must be positive integer train metadata")
    if bool((metadata["matrix_path_count"].to_numpy(dtype=int) < 2).any()):
        raise ValueError("matrix path count lies outside frozen bins beginning at two")
    rows: list[dict[str, object]] = []
    path_labels = ("2", "3", "4-5", "6-10", "11-20", ">20")
    for row in metadata.itertuples(index=False):
        path_count = int(row.matrix_path_count)
        if path_count == 2:
            path_bin, path_order = path_labels[0], 0
        elif path_count == 3:
            path_bin, path_order = path_labels[1], 1
        elif path_count <= 5:
            path_bin, path_order = path_labels[2], 2
        elif path_count <= 10:
            path_bin, path_order = path_labels[3], 3
        elif path_count <= 20:
            path_bin, path_order = path_labels[4], 4
        else:
            path_bin, path_order = path_labels[5], 5
        rows.append(
            {
                "gene_id": str(row.gene_id),
                "stratifier": "matrix_path_count",
                "support_bin": path_bin,
                "bin_order": path_order,
                "train_support_value": path_count,
            }
        )
        for stratifier, raw_value in (
            ("train_ont_raw_count", row.train_ont_raw_count),
            ("train_positive_cell_support", row.train_positive_cell_support),
        ):
            value = int(raw_value)
            log_bin = int(np.floor(np.log2(value)))
            rows.append(
                {
                    "gene_id": str(row.gene_id),
                    "stratifier": stratifier,
                    "support_bin": f"floor_log2={log_bin}",
                    "bin_order": log_bin,
                    "train_support_value": value,
                }
            )
    assignments = pd.DataFrame(rows)
    assignments["assignment_source"] = "train_only_before_model_prediction"
    return assignments.sort_values(
        ["stratifier", "bin_order", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_support_stratified_sensitivity(
    matrix_agreement_records: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
) -> SupportStratifiedSensitivity:
    """Report high/non-high-DTU metrics in each frozen one-dimensional bin."""

    if not metric_columns:
        raise ValueError("support-stratified reporting requires at least one metric")
    required = (
        "condition",
        "seed",
        "cell_id",
        "gene_id",
        "path_axis_identity",
        "high_dtu",
        "ont_count_total",
        *metric_columns,
    )
    _require_columns(
        matrix_agreement_records, required, "support-stratified matrix agreement"
    )
    _require_columns(
        assignments,
        (
            "gene_id",
            "stratifier",
            "support_bin",
            "bin_order",
            "assignment_source",
        ),
        "support-bin assignments",
    )
    records = matrix_agreement_records.copy()
    records["gene_id"] = records["gene_id"].astype(str)
    records["cell_id"] = records["cell_id"].astype(str)
    records["condition"] = records["condition"].astype(str)
    if not records["path_axis_identity"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(
            "support-stratified path-axis identity must be an explicit non-empty string"
        )
    if records.duplicated(["condition", "seed", "cell_id", "gene_id"]).any():
        raise ValueError("support-stratified input repeats a model/seed cell-gene")
    if records["high_dtu"].isna().any() or not records["high_dtu"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("high-DTU stratum must be an explicit frozen boolean")
    seeds = tuple(sorted(map(int, records["seed"].unique())))
    if len(seeds) != 3:
        raise ValueError("support-stratified final tables require exactly three seeds")
    counts = records["ont_count_total"].to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or bool((counts <= 0).any()):
        raise ValueError("eligible ONT matrix records require positive finite raw counts")
    for metric in metric_columns:
        values = records[metric].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"support-stratified metric {metric} contains infinity")
    _validate_matched_evaluation_scope(
        records,
        group_columns=("condition", "seed"),
        key_columns=("cell_id", "gene_id"),
        invariant_columns=(
            "path_axis_identity",
            "high_dtu",
            "ont_count_total",
        ),
        label="support-stratified matrix agreement",
    )
    dtu_by_gene = records.groupby("gene_id", sort=False)["high_dtu"].nunique(
        dropna=False
    )
    if bool((dtu_by_gene > 1).any()):
        raise ValueError("high-DTU label changes within a frozen gene")
    frozen = assignments.copy()
    frozen["gene_id"] = frozen["gene_id"].astype(str)
    if frozen.duplicated(["gene_id", "stratifier"]).any():
        raise ValueError("a gene has duplicate support-bin assignment")
    expected_stratifiers = {
        "matrix_path_count",
        "train_ont_raw_count",
        "train_positive_cell_support",
    }
    if set(frozen["stratifier"].astype(str)) != expected_stratifiers:
        raise ValueError("support-bin assignments differ from the three frozen stratifiers")
    if not frozen["assignment_source"].eq("train_only_before_model_prediction").all():
        raise ValueError("support-bin assignment was not frozen from train before prediction")
    for stratifier, group in frozen.groupby("stratifier", sort=False):
        if set(group["gene_id"]) != set(records["gene_id"]):
            raise ValueError(
                f"support-bin assignment for {stratifier} differs from evaluation genes"
            )

    per_seed_rows: list[dict[str, object]] = []
    conditions = tuple(sorted(records["condition"].unique()))
    for stratifier, bin_rows in frozen.groupby("stratifier", sort=True):
        bins = (
            bin_rows[["support_bin", "bin_order"]]
            .drop_duplicates()
            .sort_values("bin_order", kind="mergesort")
        )
        mapping = bin_rows[["gene_id", "support_bin", "bin_order"]]
        joined = records.merge(mapping, on="gene_id", how="left", validate="many_to_one")
        for condition in conditions:
            for seed in seeds:
                condition_seed = joined.loc[
                    joined["condition"].eq(condition)
                    & joined["seed"].astype(int).eq(seed)
                ]
                for bin_row in bins.itertuples(index=False):
                    in_bin = condition_seed.loc[
                        condition_seed["support_bin"].eq(bin_row.support_bin)
                    ]
                    for dtu_label, dtu_value in (
                        ("high_DTU", True),
                        ("non_high_DTU", False),
                    ):
                        subgroup = in_bin.loc[in_bin["high_dtu"].astype(bool).eq(dtu_value)]
                        raw_denominator = float(subgroup["ont_count_total"].sum())
                        gene_count = int(subgroup["gene_id"].nunique())
                        cell_gene_count = len(subgroup)
                        for metric in metric_columns:
                            valid = subgroup[metric].notna()
                            for aggregation in ("cell_gene_macro", "count_weighted"):
                                if not valid.any() or raw_denominator <= 0:
                                    value = np.nan
                                    status = "not_estimable"
                                elif aggregation == "cell_gene_macro":
                                    value = float(subgroup.loc[valid, metric].mean())
                                    status = "estimable"
                                else:
                                    value = _weighted_mean(
                                        subgroup.loc[valid, metric],
                                        subgroup.loc[valid, "ont_count_total"],
                                    )
                                    status = "estimable"
                                per_seed_rows.append(
                                    {
                                        "condition": condition,
                                        "seed": seed,
                                        "stratifier": str(stratifier),
                                        "support_bin": str(bin_row.support_bin),
                                        "bin_order": int(bin_row.bin_order),
                                        "dtu_stratum": dtu_label,
                                        "metric": str(metric),
                                        "aggregation": aggregation,
                                        "value": value,
                                        "status": status,
                                        "gene_count": gene_count,
                                        "eligible_cell_gene_count": cell_gene_count,
                                        "raw_count_denominator": raw_denominator,
                                        "metric_cell_gene_denominator": int(valid.sum()),
                                        "dtu_stratum_interpretation": "frozen_gene_prior",
                                    }
                                )
    per_seed = pd.DataFrame(per_seed_rows)
    across_rows: list[dict[str, object]] = []
    group_columns = (
        "condition",
        "stratifier",
        "support_bin",
        "bin_order",
        "dtu_stratum",
        "metric",
        "aggregation",
    )
    for key, group in per_seed.groupby(list(group_columns), sort=True, dropna=False):
        if set(group["seed"].astype(int)) != set(seeds) or len(group) != 3:
            raise RuntimeError("support-stratified row lacks the three frozen seeds")
        for denominator_column in (
            "gene_count",
            "eligible_cell_gene_count",
            "raw_count_denominator",
            "metric_cell_gene_denominator",
        ):
            if group[denominator_column].nunique(dropna=False) != 1:
                raise ValueError(
                    "support-stratified denominator differs across matched seeds"
                )
        estimable = group["status"].eq("estimable").all()
        values = group["value"].to_numpy(dtype=np.float64)
        across_rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "status": "estimable" if estimable else "not_estimable",
                "mean_across_seeds": float(values.mean()) if estimable else np.nan,
                "sample_sd_across_seeds": (
                    float(values.std(ddof=1)) if estimable else np.nan
                ),
                "range_across_seeds": (
                    float(values.max() - values.min()) if estimable else np.nan
                ),
                "gene_count": int(group["gene_count"].iloc[0]),
                "eligible_cell_gene_count": int(
                    group["eligible_cell_gene_count"].iloc[0]
                ),
                "raw_count_denominator": float(
                    group["raw_count_denominator"].iloc[0]
                ),
                "uncertainty_semantics": (
                    "optimization_repeat_not_biological_confidence_interval"
                ),
                "dtu_stratum_interpretation": "frozen_gene_prior",
            }
        )
    return SupportStratifiedSensitivity(
        assignments=frozen.sort_values(
            ["stratifier", "bin_order", "gene_id"], kind="mergesort"
        ).reset_index(drop=True),
        per_seed=per_seed,
        across_seed=pd.DataFrame(across_rows),
    )


def _aggregate_ont_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame([{"numerical_status": "valid", "eligible_cell_gene_count": 0, "status": "not_estimable"}])
    metrics = {
        "ont_top1_acc": "ont_top1_hit",
        "ont_top1_acc_tie_aware": "ont_top1_tie_aware_hit",
        "ont_unique_top1_acc": "ont_unique_top1_hit",
        "ont_matrix_cross_entropy": "ont_matrix_cross_entropy",
        "ont_matrix_kl": "ont_matrix_kl",
        "ont_cross_entropy": "ont_cross_entropy_prism_clamped",
        "ont_kl": "ont_kl_prism_clamped",
    }
    result: dict[str, object] = {
        "numerical_status": "valid",
        "status": "estimable",
        "eligible_cell_gene_count": len(records),
        "eligible_ont_count": float(records["ont_count_total"].sum()),
        "observed_top_tie_count": int(records["observed_top_tie"].sum()),
        "observed_top_tie_fraction": float(records["observed_top_tie"].mean()),
        "predicted_top_tie_count": int(records["predicted_top_tie"].sum()),
        "predicted_top_tie_fraction": float(records["predicted_top_tie"].mean()),
        "prism_compatibility_status": "PRISM_CLAMPED_COMPATIBILITY_ONLY",
    }
    for output_name, column in metrics.items():
        valid = records[column].notna()
        result[f"{output_name}_macro"] = float(records.loc[valid, column].mean()) if valid.any() else np.nan
        result[f"{output_name}_count_weighted"] = _weighted_mean(records.loc[valid, column], records.loc[valid, "ont_count_total"]) if valid.any() else np.nan
        result[f"{output_name}_denominator"] = int(valid.sum())
    result.update(
        {
            "ont_top1_acc": result["ont_top1_acc_macro"],
            "ont_count_weighted_top1_acc": result[
                "ont_top1_acc_count_weighted"
            ],
            "ont_top1_acc_tie_aware": result[
                "ont_top1_acc_tie_aware_macro"
            ],
            "ont_count_weighted_top1_acc_tie_aware": result[
                "ont_top1_acc_tie_aware_count_weighted"
            ],
            "ont_unique_top1_acc": result["ont_unique_top1_acc_macro"],
            "ont_cross_entropy_weighted": result[
                "ont_cross_entropy_count_weighted"
            ],
            "ont_kl_weighted": result["ont_kl_count_weighted"],
        }
    )
    return pd.DataFrame([result])


def _aggregate_compatible_overall(records: pd.DataFrame) -> dict[str, object]:
    if records.empty:
        return {
            "compatible_ec_row_count": 0,
            "compatible_molecule_mass": 0.0,
            "top1_in_C_macro": np.nan,
            "top1_in_C_molecule_weighted": np.nan,
            "top5_in_C_macro": np.nan,
            "top5_in_C_molecule_weighted": np.nan,
            "singleton_EC_top1_hit_macro": np.nan,
            "singleton_EC_top1_hit_molecule_weighted": np.nan,
            "posterior_mass_in_C_macro": np.nan,
            "posterior_mass_in_C_molecule_weighted": np.nan,
        }
    output: dict[str, object] = {
        "compatible_ec_row_count": len(records),
        "compatible_molecule_mass": float(records["molecule_count"].sum()),
    }
    for metric in (
        "top1_in_C",
        "top5_in_C",
        "singleton_EC_top1_hit",
        "posterior_mass_in_C",
        "top1_chance_baseline",
        "top5_intersection_chance_baseline",
    ):
        valid = records[metric].notna()
        output[f"{metric}_macro"] = (
            float(records.loc[valid, metric].mean()) if valid.any() else np.nan
        )
        output[f"{metric}_molecule_weighted"] = (
            _weighted_mean(
                records.loc[valid, metric], records.loc[valid, "molecule_count"]
            )
            if valid.any()
            else np.nan
        )
        output[f"{metric}_row_denominator"] = int(valid.sum())
        output[f"{metric}_molecule_denominator"] = float(
            records.loc[valid, "molecule_count"].sum()
        )
    output["singleton_top1_acc"] = output["singleton_EC_top1_hit_macro"]
    return output


def _flatten_validation_snapshot(
    snapshot: object,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    predictions = getattr(snapshot, "predictions", None)
    if not isinstance(predictions, tuple) or not predictions:
        raise ValueError("validation snapshot has no frozen predictions")
    path_records: list[dict[str, object]] = []
    ec_records: list[dict[str, object]] = []
    dtypes: set[str] = set()
    observed_genes: set[str] = set()
    for prediction in predictions:
        gene_id = str(prediction.gene_id)
        if gene_id in observed_genes:
            raise ValueError("validation snapshot repeats a gene prediction")
        observed_genes.add(gene_id)
        cell_ids = tuple(map(str, prediction.cell_ids))
        path_ids = tuple(map(str, prediction.path_ids))
        if len(cell_ids) != len(set(cell_ids)) or len(path_ids) != len(set(path_ids)):
            raise ValueError("validation prediction cell and path IDs must be unique")
        logits = prediction.path_logits
        if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu":
            raise TypeError("validation path logits must be detached CPU tensors")
        if logits.shape != (len(cell_ids), len(path_ids)):
            raise ValueError("validation prediction logits and named axes differ")
        dtypes.add(str(logits.dtype).removeprefix("torch."))
        for cell_position, cell_id in enumerate(cell_ids):
            for path_position, path_id in enumerate(path_ids):
                path_records.append(
                    {
                        "cell_id": cell_id,
                        "gene_id": gene_id,
                        "path_id": path_id,
                        "logit": float(logits[cell_position, path_position]),
                    }
                )
        compatible = prediction.compatible_path_indices
        mask = prediction.compatible_path_mask
        row_cells = prediction.row_cell_index
        masses = prediction.molecule_count
        if not all(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            for value in (compatible, mask, row_cells, masses)
        ):
            raise TypeError("validation EC prediction tensors must be on CPU")
        if compatible.shape != mask.shape or compatible.ndim != 2:
            raise ValueError("validation compatible index/mask tensors differ")
        if row_cells.shape != masses.shape or row_cells.numel() != compatible.shape[0]:
            raise ValueError("validation EC row tensors have different axes")
        for row_index in range(compatible.shape[0]):
            cell_position = int(row_cells[row_index])
            if cell_position < 0 or cell_position >= len(cell_ids):
                raise ValueError("validation EC row_cell_index is out of range")
            selected_indices = compatible[row_index][mask[row_index]].tolist()
            if len(selected_indices) != len(set(map(int, selected_indices))) or any(
                int(index) < 0 or int(index) >= len(path_ids)
                for index in selected_indices
            ):
                raise ValueError(
                    "validation snapshot compatible indices are duplicate or out of range"
                )
            mass = float(masses[row_index])
            if (
                not np.isfinite(mass)
                or mass <= 0
                or not selected_indices
                or len(selected_indices) == len(path_ids)
            ):
                raise ValueError(
                    "validation snapshot must contain likelihood-informative EC rows only"
                )
            ec_records.append(
                {
                    "cell_id": cell_ids[cell_position],
                    "gene_id": gene_id,
                    "compatible_path_ids": [path_ids[int(index)] for index in selected_indices],
                    "molecule_count": mass,
                    "split": "val",
                }
            )
    if len(dtypes) != 1:
        raise ValueError("validation predictions use inconsistent model-output dtypes")
    return pd.DataFrame(path_records), pd.DataFrame(ec_records), dtypes.pop()


def _validate_snapshot_path_axes(
    path_logits: pd.DataFrame,
    identity: OntMatrixIdentity,
) -> None:
    for (cell_id, gene_id), rows in path_logits.groupby(
        ["cell_id", "gene_id"], sort=False
    ):
        expected = set(
            identity.model_paths.loc[
                identity.model_paths["gene_id"].eq(str(gene_id)), "path_id"
            ].astype(str)
        )
        if not expected:
            raise ValueError(
                f"validation prediction gene {gene_id} is absent from frozen path identity"
            )
        if set(rows["path_id"].astype(str)) != expected:
            raise ValueError(
                f"validation prediction path identity is incomplete for {(cell_id, gene_id)}"
            )


def _validate_snapshot_ec_identity(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    numerical_tolerance: float,
) -> None:
    def aggregate(frame: pd.DataFrame) -> dict[tuple[str, str, tuple[str, ...]], float]:
        records: dict[tuple[str, str, tuple[str, ...]], float] = {}
        for row in frame.itertuples(index=False):
            paths = list(map(str, _as_list(row.compatible_path_ids)))
            if len(paths) != len(set(paths)):
                raise ValueError(
                    "validation monitor EC paths must not contain duplicates"
                )
            mass = float(row.molecule_count)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(
                    "validation monitor EC molecule mass must be finite and positive"
                )
            key = (str(row.cell_id), str(row.gene_id), tuple(sorted(paths)))
            records[key] = records.get(key, 0.0) + mass
        return records

    observed_records = aggregate(observed)
    expected_records = aggregate(expected)
    if set(observed_records) != set(expected_records):
        raise ValueError(
            "validation snapshot compatible EC identities differ from frozen monitor bundle"
        )
    for key in observed_records:
        if not np.isclose(
            observed_records[key],
            expected_records[key],
            atol=numerical_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "validation snapshot compatible molecule mass differs from frozen bundle"
            )


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return "not_estimable"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"monitor field is not JSON-scalar: {type(value).__name__}")


def _nonnegative_finite_scalar(value: object, label: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return scalar


def _unique_axis(values: Sequence[str], expected_length: int, label: str) -> tuple[str, ...]:
    axis = tuple(map(str, values))
    if len(axis) != expected_length or len(axis) != len(set(axis)) or any(
        not value for value in axis
    ):
        raise ValueError(
            f"{label} axis must contain exactly {expected_length} unique non-empty IDs"
        )
    return axis


def _context_cell_index(
    activity: ActivityContext,
    atac: ATACMappingContext,
    cell_id: str,
) -> int:
    if tuple(map(str, activity.cell_ids)) != tuple(map(str, atac.cell_ids)):
        raise ValueError("activity and mapped ATAC contexts have different cell axes")
    if len(activity.cell_ids) != len(set(map(str, activity.cell_ids))):
        raise ValueError("dynamic context cell axis is not unique")
    index = {str(value): position for position, value in enumerate(activity.cell_ids)}
    if str(cell_id) not in index:
        raise ValueError(f"perturbation cell is absent from the dynamic context: {cell_id}")
    return index[str(cell_id)]


def _validated_gate_key_table(gate_keys: pd.DataFrame) -> pd.DataFrame:
    required = (
        "gate_key_id",
        "target_gene_id",
        "channel",
        "activity_entity_id",
        "peak_id",
    )
    _require_columns(gate_keys, required, "gate key table")
    keys = gate_keys[list(required)].copy()
    keys["gate_key_id"] = keys["gate_key_id"].astype(str)
    if keys["gate_key_id"].duplicated().any() or keys.empty:
        raise ValueError("gate key table requires unique non-empty gate keys")
    return keys.sort_values("gate_key_id", kind="mergesort").reset_index(drop=True)


def _materialize_perturbed_gate_context(
    *,
    perturbation_kind: str,
    cell_id: str,
    input_kind: str,
    input_id: str,
    input_value: float,
    activity: ActivityContext,
    atac: ATACMappingContext,
    gate_keys: pd.DataFrame,
    gate_admission: pd.DataFrame,
    physical_events: pd.DataFrame,
    affected_gate_key_ids: Sequence[str],
    extra_audit_fields: Mapping[str, object],
) -> PerturbedGateContext:
    raw = build_raw_gate_signals(gate_keys, activity=activity, atac=atac)
    gates = transform_gates(raw, gate_admission)
    cell_index = _context_cell_index(activity, atac, cell_id)
    key_index = {value: index for index, value in enumerate(gates.gate_key_ids)}
    affected_keys = tuple(sorted(set(map(str, affected_gate_key_ids))))
    missing = sorted(set(affected_keys) - set(key_index))
    if missing:
        raise ValueError(f"affected perturbation gate keys are absent: {missing[:5]}")
    _require_columns(
        physical_events,
        ("event_id", "gate_key_id"),
        "PhysicalEventTable for perturbation",
    )
    event_table = physical_events[["event_id", "gate_key_id"]].copy()
    if event_table["event_id"].astype(str).duplicated().any():
        raise ValueError("PhysicalEventTable has duplicate event IDs")
    event_table["gate_key_id"] = event_table["gate_key_id"].astype("string")
    affected_events = tuple(
        sorted(
            event_table.loc[
                event_table["gate_key_id"].isin(affected_keys), "event_id"
            ].astype(str)
        )
    )
    if not affected_events:
        raise ValueError("affected perturbation gate keys have no physical events")
    _require_columns(
        gate_admission,
        (
            "gate_key_id",
            "train_mean",
            "train_standard_deviation",
            "train_raw_minimum",
            "train_raw_maximum",
            "train_lower_weighted_quantile",
            "train_upper_weighted_quantile",
            "gate_key_active",
        ),
        "GateAdmissionManifest",
    )
    admission = gate_admission.copy()
    admission["gate_key_id"] = admission["gate_key_id"].astype(str)
    if admission["gate_key_id"].duplicated().any():
        raise ValueError("GateAdmissionManifest has duplicate gate keys")
    admission = admission.set_index("gate_key_id")
    if not set(affected_keys).issubset(admission.index):
        raise ValueError("affected gate keys are absent from GateAdmissionManifest")
    audit_rows: list[dict[str, object]] = []
    for gate_key_id in affected_keys:
        column = key_index[gate_key_id]
        manifest = admission.loc[gate_key_id]
        event_ids = tuple(
            sorted(
                event_table.loc[
                    event_table["gate_key_id"].eq(gate_key_id), "event_id"
                ].astype(str)
            )
        )
        observed = bool(gates.observed[cell_index, column])
        outside_range = bool(gates.out_of_train_range[cell_index, column])
        outside_quantile = bool(
            gates.out_of_train_quantile_support[cell_index, column]
        )
        active = bool(manifest.gate_key_active)
        context_status = (
            "observed"
            if observed and active
            else "missing_required_dynamic_context"
            if not observed
            else "train_inactive_gate"
        )
        audit_rows.append(
            {
                "perturbation_kind": perturbation_kind,
                "cell_id": cell_id,
                "input_kind": input_kind,
                "input_id": input_id,
                "input_value": input_value,
                "gate_key_id": gate_key_id,
                "event_ids": list(event_ids),
                "train_raw_minimum": float(manifest.train_raw_minimum),
                "train_raw_maximum": float(manifest.train_raw_maximum),
                "train_lower_weighted_quantile": float(
                    manifest.train_lower_weighted_quantile
                ),
                "train_upper_weighted_quantile": float(
                    manifest.train_upper_weighted_quantile
                ),
                "train_mean": float(manifest.train_mean),
                "train_standard_deviation": float(
                    manifest.train_standard_deviation
                ),
                "raw_signal_b": float(gates.raw[cell_index, column]),
                "standardized_residual_z": float(
                    gates.standardized_residual[cell_index, column]
                ),
                "final_gate_G": float(gates.gate[cell_index, column]),
                "observation_mask": observed,
                "gate_key_active": active,
                "out_of_train_range": outside_range,
                "out_of_train_quantile_support": outside_quantile,
                "dynamic_context_status": context_status,
                **extra_audit_fields,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if audit["out_of_train_range"].any() or audit[
        "out_of_train_quantile_support"
    ].any():
        support_status = "model_extrapolation"
    elif not audit["dynamic_context_status"].eq("observed").all():
        support_status = "missing_context_not_estimable"
    else:
        support_status = "supported_model_counterfactual"
    audit["support_status"] = support_status
    audit["primary_supported_claim_allowed"] = (
        support_status == "supported_model_counterfactual"
    )
    return PerturbedGateContext(
        perturbation_kind=perturbation_kind,
        cell_id=cell_id,
        input_kind=input_kind,
        input_id=input_id,
        input_value=float(input_value),
        activity=activity,
        atac=atac,
        gates=gates,
        affected_gate_key_ids=affected_keys,
        affected_event_ids=affected_events,
        gate_audit=audit,
        support_status=support_status,
    )


def _validate_matched_evaluation_scope(
    records: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    key_columns: Sequence[str],
    invariant_columns: Sequence[str],
    label: str,
) -> None:
    groups = list(records.groupby(list(group_columns), sort=True, dropna=False))
    if not groups:
        raise ValueError(f"{label} has no evaluation records")
    reference_key: set[tuple[object, ...]] | None = None
    reference_invariants: pd.DataFrame | None = None
    for _, group in groups:
        ordered = group.sort_values(list(key_columns), kind="mergesort")
        keys = set(map(tuple, ordered[list(key_columns)].astype(str).to_numpy()))
        invariant = ordered[
            [*key_columns, *invariant_columns]
        ].reset_index(drop=True)
        if reference_key is None:
            reference_key = keys
            reference_invariants = invariant
            continue
        if keys != reference_key:
            raise ValueError(f"{label} cell-gene/EC scope differs across models or seeds")
        try:
            pd.testing.assert_frame_equal(
                invariant,
                reference_invariants,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                f"{label} path axes, weights, denominators, or frozen strata differ"
            ) from error


def _validate_frozen_unit_strata(
    records: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    strata_columns: Sequence[str],
    label: str,
) -> None:
    for column in strata_columns:
        counts = records.groupby(list(unit_columns), dropna=False)[column].nunique(
            dropna=False
        )
        if bool((counts > 1).any()):
            raise ValueError(f"{label} stratum {column} changes within a frozen unit")


def _join_complete_gene_strata(
    records: pd.DataFrame,
    frozen_gene_strata: pd.DataFrame,
    strata_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    left = records.copy()
    left["gene_id"] = left["gene_id"].astype(str)
    right = frozen_gene_strata[["gene_id", *strata_columns]].copy()
    right["gene_id"] = right["gene_id"].astype(str)
    joined = left.merge(right, on="gene_id", how="left", validate="many_to_one")
    if joined[list(strata_columns)].isna().any().any():
        raise ValueError(f"{label} has genes absent from frozen strata")
    return joined


def _fit_frozen_ridge(rows, include_state, state_pc_columns, alpha_grid, cv_folds, seed):
    category_fields = ["gene_group_id", "donor"]
    if include_state:
        category_fields.extend(["stage", "developmental_system", "cell_type"])
    category_levels = tuple(
        (field, tuple(sorted(rows[field].astype(str).unique())))
        for field in category_fields
    )
    design = _state_design(
        rows, include_state, state_pc_columns, category_levels=category_levels
    )
    y = rows["score_residual"].to_numpy(float)
    weights = rows["informative_molecule_mass"].to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("ridge diagnostic requires finite residuals and positive weights")
    splits = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = []
    for grid_index, alpha in enumerate(alpha_grid):
        fold_loss = []
        for train_rows, validation_rows in splits.split(design):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(design.iloc[train_rows], y[train_rows], sample_weight=weights[train_rows])
            prediction = model.predict(design.iloc[validation_rows])
            fold_loss.append(np.average((y[validation_rows] - prediction) ** 2, weights=weights[validation_rows]))
        scores.append((float(np.mean(fold_loss)), grid_index, alpha))
    _, _, selected_alpha = min(scores)
    model = Ridge(alpha=selected_alpha, fit_intercept=True)
    model.fit(design, y, sample_weight=weights)
    return FrozenRidgeDiagnostic(
        tuple(design.columns),
        np.asarray(model.coef_, dtype=np.float64),
        float(model.intercept_),
        float(selected_alpha),
        include_state,
        state_pc_columns,
        category_levels,
    )


def _state_design(
    rows,
    include_state,
    state_pc_columns,
    *,
    category_levels=None,
):
    numeric = pd.DataFrame(
        {
            "log1p_informative_molecule_mass": np.log1p(rows["informative_molecule_mass"].to_numpy(float)),
            "positive_ec_row_count": rows["positive_ec_row_count"].to_numpy(float),
        },
        index=rows.index,
    )
    categories = ["gene_group_id", "donor"]
    if include_state:
        categories.extend(["stage", "developmental_system", "cell_type"])
        for column in state_pc_columns:
            numeric[column] = rows[column].to_numpy(float)
    if category_levels is None:
        category_levels = tuple(
            (field, tuple(sorted(rows[field].astype(str).unique())))
            for field in categories
        )
    level_lookup = {field: tuple(levels) for field, levels in category_levels}
    if set(level_lookup) != set(categories):
        raise ValueError("frozen ridge category fields differ from diagnostic design")
    categorical_values = pd.DataFrame(index=rows.index)
    for field in categories:
        observed = rows[field].astype(str)
        unknown = sorted(set(observed) - set(level_lookup[field]))
        if unknown:
            raise ValueError(
                f"held-out state-residual category is outside frozen vocabulary: "
                f"{field}={unknown[:5]}"
            )
        categorical_values[field] = pd.Categorical(
            observed, categories=level_lookup[field]
        )
    categorical = pd.get_dummies(
        categorical_values, prefix=categories, dtype=float
    )
    return pd.concat([numeric, categorical], axis=1).sort_index(axis=1)


def _predict_frozen_ridge(model: FrozenRidgeDiagnostic, rows: pd.DataFrame):
    design = _state_design(
        rows,
        model.include_state,
        model.state_pc_columns,
        category_levels=model.category_levels,
    )
    design = design.reindex(columns=model.feature_columns, fill_value=0.0)
    return design.to_numpy(float) @ model.coefficients + model.intercept


def _weighted_r2(y, prediction, weights):
    mean = np.average(y, weights=weights)
    denominator = np.sum(weights * (y - mean) ** 2)
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum(weights * (y - prediction) ** 2) / denominator)


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / np.sum(weights))


def _iqr(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _as_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return list(value)
    raise TypeError("frozen list-valued field must contain an explicit sequence")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} misses columns: {missing}")


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].astype(str).duplicated().any():
        raise ValueError(f"{label} {column} values must be unique")
