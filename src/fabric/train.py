"""FABRIC V2 training, paired ablations, and fail-closed execution admission."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import copy
import fcntl
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from .evaluate import OntMatrixKlTarget, compute_validation_ont_matrix_kl
from .likelihood import compatible_path_nll
from .model import (
    ARCHITECTURE_COMPARATOR,
    PRIMARY_ABLATIONS,
    FABRICV2Model,
    GeneCellModelInput,
    RoutedModalityInput,
)
from .source_identity import committed_source_identity


# ``python -m fabric.train`` executes this file as ``__main__``.  Bind the
# canonical module name before unpickling a locally prepared bundle so its
# dataclass identities remain exact; the installed ``fabric-train`` entry
# point already imports the canonical name directly.
if __name__ == "__main__":
    sys.modules.setdefault("fabric.train", sys.modules[__name__])


SPLITS = ("train", "val", "test")
FULL_COHORT_SCOPE = "full_cohort"
RUN_CONDITIONS = ("full", "atac", "rbp")
_MODEL_CONDITION = {"full": "full", "atac": "cis_dna", "rbp": "cis_rna"}


@dataclass(frozen=True)
class PreparedGene:
    """One V2 gene with unique cell instances and compatible EC supervision."""

    gene_id: str
    model_input: GeneCellModelInput
    compatible_path_indices: torch.Tensor  # padded long [K,W]
    compatible_path_mask: torch.Tensor  # bool [K,W]
    row_cell_index: torch.Tensor  # long [K]
    molecule_count: torch.Tensor  # float [K]
    informative_row_mask: torch.Tensor  # bool [K], exact K^inf definition
    cell_ids: tuple[str, ...]
    cell_split: tuple[str, ...]
    path_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainGeneCellSample:
    """One group-closed train sample for a single gene and epoch.

    ``selected_rows`` always contains every informative EC row belonging to
    ``selected_cells``.  ``inclusion_multiplier`` is the inverse uniform
    gene-cell inclusion probability, N_g / n_g, used with the frozen complete
    train molecule total to form the Horvitz-Thompson objective estimate.
    """

    selected_cells: torch.Tensor
    selected_rows: torch.Tensor
    available_cell_count: int
    selected_cell_count: int
    inclusion_multiplier: float


@dataclass(frozen=True)
class GeneBatchPlan:
    """Deterministic gene-local cell batches under the frozen GPU policy."""

    batches: tuple[torch.Tensor, ...]
    estimated_bytes: tuple[int, ...]
    per_cell_shape_elements: int
    per_compatible_row_shape_elements: int

    @property
    def maximum_estimated_bytes(self) -> int:
        return max(self.estimated_bytes)


@dataclass(frozen=True)
class PreparedDataset:
    genes: tuple[PreparedGene, ...]
    input_manifest_id: str
    compatibility_artifact_id: str
    informative_gene_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("input_manifest_id", "compatibility_artifact_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PreparedDataset {name} must be a nonempty identity")
        if self.informative_gene_ids is not None:
            values = self.informative_gene_ids
            if (
                not isinstance(values, tuple)
                or not values
                or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(
                    "PreparedDataset informative_gene_ids must be unique nonempty IDs"
                )


class BackedGeneSequence(Sequence[PreparedGene]):
    """One immutable ordered axis of per-gene torch shards.

    The index contains only relative shard paths and gene IDs.  A small LRU
    avoids repeatedly deserializing the current gene while keeping the
    17,600-gene dataset out of host RAM.
    """

    def __init__(
        self,
        root: str | Path,
        records: Sequence[Mapping[str, str]],
        *,
        expected_split_mass: Mapping[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.records = tuple(
            (str(row["gene_id"]), str(row["relative_path"])) for row in records
        )
        if not self.records or len({value[0] for value in self.records}) != len(
            self.records
        ):
            raise ValueError("backed gene index must contain unique nonempty records")
        self.expected_split_mass = {
            str(split): int(mass) for split, mass in (expected_split_mass or {}).items()
        }
        if any(mass <= 0 for mass in self.expected_split_mass.values()):
            raise ValueError("backed expected split mass must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int | slice
    ) -> PreparedGene | tuple[PreparedGene, ...]:
        if isinstance(index, slice):
            return tuple(self[value] for value in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._load(index)

    @lru_cache(maxsize=2)
    def _load(self, index: int) -> PreparedGene:
        gene_id, relative_path = self.records[index]
        shard = self.root / relative_path
        value = torch.load(shard, map_location="cpu", weights_only=False)
        if not isinstance(value, PreparedGene) or value.gene_id != gene_id:
            raise TypeError(f"backed gene shard identity differs: {shard}")
        _validate_genes((value,))
        return value


@dataclass(frozen=True)
class BackedPreparedDataset:
    """The sole production dataset container for the real full cohort."""

    genes: BackedGeneSequence
    input_manifest_id: str
    compatibility_artifact_id: str
    informative_gene_ids: tuple[str, ...]
    source_git_commit: str | None = None

    @classmethod
    def load(cls, root: str | Path) -> "BackedPreparedDataset":
        root = Path(root)
        manifest = json.loads((root / "PreparedDatasetManifest.json").read_text())
        if manifest.get("schema_version") != "fabric.backed_prepared_dataset.v1":
            raise ValueError("unsupported backed PreparedDataset manifest")
        records = manifest.get("gene_shards")
        if not isinstance(records, list):
            raise TypeError(
                "backed PreparedDataset gene_shards must be an ordered list"
            )
        axis = tuple(str(value) for value in manifest["informative_gene_ids"])
        if tuple(str(row["gene_id"]) for row in records) != axis:
            raise ValueError("backed PreparedDataset shard order differs from G_fit")
        expected_split_mass = {}
        for split, field_name in (
            ("train", "expected_train_informative_molecule_mass"),
            ("val", "expected_validation_informative_molecule_mass"),
        ):
            value = manifest.get(field_name)
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        f"backed PreparedDataset {field_name} must be positive integer"
                    )
                expected_split_mass[split] = value
        return cls(
            genes=BackedGeneSequence(
                root, records, expected_split_mass=expected_split_mass
            ),
            input_manifest_id=str(manifest["input_manifest_id"]),
            compatibility_artifact_id=str(manifest["compatibility_artifact_id"]),
            informative_gene_ids=axis,
            source_git_commit=(
                None
                if manifest.get("source_git_commit") is None
                else str(manifest["source_git_commit"])
            ),
        )


def prepared_gene_from_assembly(assembly: object) -> PreparedGene:
    """Bind the sole dataset V2 assembly type to the training contract."""

    from .dataset import V2GeneAssembly

    if not isinstance(assembly, V2GeneAssembly):
        raise TypeError("training input must be one dataset.V2GeneAssembly")
    if not isinstance(assembly.model_input, GeneCellModelInput):
        raise TypeError("V2GeneAssembly model_input must be GeneCellModelInput")
    gene = PreparedGene(
        gene_id=assembly.gene_id,
        model_input=assembly.model_input,
        compatible_path_indices=assembly.compatible_path_indices,
        compatible_path_mask=assembly.compatible_path_mask,
        row_cell_index=assembly.row_cell_index,
        molecule_count=assembly.molecule_count,
        informative_row_mask=assembly.informative_row_mask,
        cell_ids=assembly.cell_ids,
        cell_split=assembly.cell_split,
        path_ids=assembly.path_ids,
    )
    _validate_genes((gene,))
    return gene


@dataclass(frozen=True)
class TrainingRunManifest:
    """Identity of one independently executable train/validation run."""

    seed: int
    condition: str
    learning_rate: float
    lr_scheduler_name: str
    lr_scheduler_factor: float | None
    lr_scheduler_patience: int | None
    lr_scheduler_min_lr: float | None
    gradient_clip_norm: float
    lambda_base: float
    lambda_int: float
    max_epochs: int
    early_stopping_patience: int
    inputs_frozen: bool
    resources_frozen: bool
    primary_epoch_unit: str
    max_train_gene_cells_per_gene_per_epoch: int
    resample_train_gene_cells_each_epoch: bool
    selected_gene_cell_ec_rows: str
    sampling_estimator: str
    optimizer_step_unit: str
    gene_microbatch_gradient_accumulation: bool
    batch_policy: str
    target_gpu_allocated_bytes: int
    max_cells_per_gpu_batch: int
    prefetch_backed_gene_shards: bool
    compute_precision: str

    def validate(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("TrainingRunManifest seed must be an integer")
        if self.condition not in RUN_CONDITIONS:
            raise ValueError(f"run condition must be one of {RUN_CONDITIONS}")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "TrainingRunManifest learning rate must be positive and finite"
            )
        _validate_lr_scheduler_fields(
            name=self.lr_scheduler_name,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
            learning_rate=float(self.learning_rate),
        )
        if (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not np.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm < 0
        ):
            raise ValueError(
                "TrainingRunManifest gradient clip norm must be finite and non-negative"
            )
        if (
            isinstance(self.lambda_base, bool)
            or isinstance(self.lambda_int, bool)
            or not isinstance(self.lambda_base, (int, float))
            or not isinstance(self.lambda_int, (int, float))
            or not np.isfinite((self.lambda_base, self.lambda_int)).all()
        ):
            raise TypeError("TrainingRunManifest penalties must be finite numbers")
        if not (self.lambda_int > self.lambda_base >= 0):
            raise ValueError(
                "TrainingRunManifest requires lambda_int > lambda_base >= 0"
            )
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError("TrainingRunManifest max_epochs must be positive")
        if (
            type(self.early_stopping_patience) is not int
            or not 1 <= self.early_stopping_patience <= self.max_epochs
        ):
            raise ValueError(
                "TrainingRunManifest early-stopping patience must lie in [1, max_epochs]"
            )
        if (
            type(self.inputs_frozen) is not bool
            or type(self.resources_frozen) is not bool
        ):
            raise TypeError("TrainingRunManifest frozen fields must be booleans")
        if self.batch_policy != "gene_shape_adaptive_v1":
            raise ValueError("TrainingRunManifest batch policy differs")
        if (
            type(self.target_gpu_allocated_bytes) is not int
            or self.target_gpu_allocated_bytes < 1
        ):
            raise ValueError("TrainingRunManifest GPU target must be positive")
        if (
            type(self.max_cells_per_gpu_batch) is not int
            or self.max_cells_per_gpu_batch < 1
        ):
            raise ValueError("TrainingRunManifest GPU cell cap must be positive")
        if type(self.prefetch_backed_gene_shards) is not bool:
            raise TypeError("TrainingRunManifest prefetch field must be boolean")
        if self.compute_precision != "float32_highest":
            raise ValueError("TrainingRunManifest compute precision differs")
        _validate_train_sampling_manifest(self)


def _validate_train_sampling_manifest(manifest: object) -> None:
    if (
        getattr(manifest, "primary_epoch_unit", None)
        != "sampled_informative_gene_cell_horvitz_thompson"
    ):
        raise ValueError("V2 must sample complete informative gene-cell groups")
    cap = getattr(manifest, "max_train_gene_cells_per_gene_per_epoch", None)
    if type(cap) is not int or cap < 1:
        raise ValueError("V2 per-gene train gene-cell cap must be a positive integer")
    if getattr(manifest, "resample_train_gene_cells_each_epoch", None) is not True:
        raise ValueError("V2 train gene-cells must be resampled each epoch")
    if getattr(manifest, "selected_gene_cell_ec_rows", None) != "all_informative_rows":
        raise ValueError("selected V2 gene-cells must retain all informative EC rows")
    if (
        getattr(manifest, "sampling_estimator", None)
        != "horvitz_thompson_full_train_molecule_total"
    ):
        raise ValueError("V2 sampling estimator identity differs")
    if getattr(manifest, "optimizer_step_unit", None) != "train_positive_gene":
        raise ValueError("V2 optimizer step unit must be one train-positive gene")
    if getattr(manifest, "gene_microbatch_gradient_accumulation", None) is not True:
        raise ValueError(
            "V2 must accumulate all attention microbatches before the gene step"
        )


@dataclass(frozen=True)
class RouteDegreeCapAuditMeasurement:
    audit_seed: int
    family: str
    condition_id: str
    route_degree: int
    D_pre: int
    D_post: int
    planted_delta_rho: float
    matched_delta_rho: float
    recovery_error: float
    absolute_gate_gradient: float
    paired_output_drift: float
    paired_gradient_drift: float
    seed_dispersion: float
    route_l1_mass: float
    route_weight_identity_error: float
    event_frobenius_norm: float
    frobenius_identity_error: float
    cap_loss_norm: float
    cap_renormalization_gain_norm: float
    cap_decomposition_error: float
    delta_a_norm_median: float
    delta_a_norm_max: float
    delta_y_norm: float
    delta_normalized_token_norm: float
    delta_hidden_norm: float
    event_to_background_pre_layernorm_norm_ratio: float
    neutralization_identity_error: float
    joint_projection_identity_error: float
    reproducibility_error: float
    all_finite: bool


@dataclass(frozen=True)
class RouteDegreeCapAuditManifest:
    generator_identity: str
    planted_target_identity: str
    structural_route_audit_identity: str
    model_input_degree_gt2_event_count: int
    model_input_external_only_coupling_event_count: int
    audit_seeds: tuple[int, int, int]
    epsilon_syn: float
    recovery_tolerance: float
    seed_dispersion_tolerance: float
    degree_gradient_drift_tolerance: float
    cap_output_drift_tolerance: float
    cap_gradient_drift_tolerance: float
    measurements: tuple[RouteDegreeCapAuditMeasurement, ...]
    route_degree_catalog_applicable: bool = field(init=False)
    cap_coupling_catalog_applicable: bool = field(init=False)
    implementation_valid: bool = field(init=False)
    baseline_capability_pass: bool = field(init=False)
    route_degree_pass: bool = field(init=False)
    cap_coupling_pass: bool = field(init=False)
    failure_reasons: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_specification()
        object.__setattr__(
            self,
            "route_degree_catalog_applicable",
            self.model_input_degree_gt2_event_count > 0,
        )
        object.__setattr__(
            self,
            "cap_coupling_catalog_applicable",
            self.model_input_external_only_coupling_event_count > 0,
        )
        condition_shape = {
            "degree_2": ("single_anchor", 2, 2),
            "degree_4": ("single_anchor", 4, 4),
            "degree_8": ("single_anchor", 8, 8),
            "cap_4_plus_2": ("multi_anchor_cap", 8, 8),
            "cap_2": ("multi_anchor_cap", 8, 4),
            "cap_0": ("multi_anchor_cap", 8, 2),
        }
        expected = {
            (seed, family, condition)
            for seed in self.audit_seeds
            for family, conditions in (
                ("single_anchor", ("degree_2", "degree_4", "degree_8")),
                ("multi_anchor_cap", ("cap_4_plus_2", "cap_2", "cap_0")),
            )
            for condition in conditions
        }
        observed = {
            (row.audit_seed, row.family, row.condition_id) for row in self.measurements
        }
        if len(self.measurements) != 18 or observed != expected:
            raise ValueError(
                "route audit measurements must contain all 18 balanced seed-condition records"
            )
        for row in self.measurements:
            self._validate_measurement(row)
            if (row.family, row.D_pre, row.D_post) != condition_shape[row.condition_id]:
                raise ValueError(
                    "route audit condition differs from its frozen family and D_pre/D_post"
                )
            expected_recovery = abs(
                row.matched_delta_rho - row.planted_delta_rho
            ) / max(abs(row.planted_delta_rho), self.epsilon_syn)
            if not np.isclose(
                row.recovery_error, expected_recovery, atol=1e-12, rtol=0
            ):
                raise ValueError(
                    "route audit recovery_error differs from measured values"
                )

        for seed in self.audit_seeds:
            seed_rows = {
                row.condition_id: row
                for row in self.measurements
                if row.audit_seed == seed
            }
            for family_conditions in (
                ("degree_2", "degree_4", "degree_8"),
                ("cap_4_plus_2", "cap_2", "cap_0"),
            ):
                reference = seed_rows[family_conditions[0]]
                denominator = max(abs(reference.planted_delta_rho), self.epsilon_syn)
                gradient_denominator = max(
                    abs(reference.absolute_gate_gradient), self.epsilon_syn
                )
                for condition_id in family_conditions:
                    row = seed_rows[condition_id]
                    expected_output_drift = (
                        abs(row.matched_delta_rho - reference.matched_delta_rho)
                        / denominator
                    )
                    expected_gradient_drift = (
                        abs(
                            row.absolute_gate_gradient
                            - reference.absolute_gate_gradient
                        )
                        / gradient_denominator
                    )
                    if not np.isclose(
                        row.paired_output_drift,
                        expected_output_drift,
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError(
                            "route audit paired_output_drift differs from measured values"
                        )
                    if not np.isclose(
                        row.paired_gradient_drift,
                        expected_gradient_drift,
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError(
                            "route audit paired_gradient_drift differs from measured values"
                        )
        for condition_id in condition_shape:
            condition_rows = [
                row for row in self.measurements if row.condition_id == condition_id
            ]
            planted = {row.planted_delta_rho for row in condition_rows}
            if len(planted) != 1:
                raise ValueError(
                    "route audit planted target differs across audit seeds"
                )
            denominator = max(
                abs(condition_rows[0].planted_delta_rho), self.epsilon_syn
            )
            expected_dispersion = (
                max(row.matched_delta_rho for row in condition_rows)
                - min(row.matched_delta_rho for row in condition_rows)
            ) / denominator
            if any(
                not np.isclose(
                    row.seed_dispersion, expected_dispersion, atol=1e-12, rtol=0
                )
                for row in condition_rows
            ):
                raise ValueError(
                    "route audit seed_dispersion differs from measured values"
                )

        implementation = all(
            row.all_finite
            and row.route_l1_mass == 1.0
            and row.route_weight_identity_error <= self.epsilon_syn
            and row.neutralization_identity_error <= self.epsilon_syn
            and row.joint_projection_identity_error <= self.epsilon_syn
            and row.reproducibility_error <= self.epsilon_syn
            and row.cap_decomposition_error <= self.epsilon_syn
            and (
                row.family != "single_anchor"
                or row.frobenius_identity_error <= self.epsilon_syn
            )
            for row in self.measurements
        )
        baseline_rows = [
            row
            for row in self.measurements
            if row.family == "single_anchor" and row.condition_id == "degree_2"
        ]
        baseline = implementation and all(
            row.recovery_error <= self.recovery_tolerance
            and row.seed_dispersion <= self.seed_dispersion_tolerance
            for row in baseline_rows
        )
        degree_rows = [
            row
            for row in self.measurements
            if row.family == "single_anchor"
            and row.condition_id in {"degree_4", "degree_8"}
        ]
        degree = baseline and all(
            row.recovery_error <= self.recovery_tolerance
            and row.seed_dispersion <= self.seed_dispersion_tolerance
            and row.paired_gradient_drift <= self.degree_gradient_drift_tolerance
            for row in degree_rows
        )
        cap_rows = [
            row
            for row in self.measurements
            if row.family == "multi_anchor_cap"
            and row.condition_id in {"cap_2", "cap_0"}
        ]
        cap = baseline and all(
            row.recovery_error <= self.recovery_tolerance
            and row.seed_dispersion <= self.seed_dispersion_tolerance
            and row.paired_output_drift <= self.cap_output_drift_tolerance
            and row.paired_gradient_drift <= self.cap_gradient_drift_tolerance
            for row in cap_rows
        )
        object.__setattr__(self, "implementation_valid", bool(implementation))
        object.__setattr__(self, "baseline_capability_pass", bool(baseline))
        object.__setattr__(self, "route_degree_pass", bool(degree))
        object.__setattr__(self, "cap_coupling_pass", bool(cap))
        failure_reasons = []
        if not implementation:
            failure_reasons.append("implementation_invalid")
        if not baseline:
            failure_reasons.append("baseline_capability_failed")
        if self.route_degree_catalog_applicable and not degree:
            failure_reasons.append("route_degree_capability_failed")
        if self.cap_coupling_catalog_applicable and not cap:
            failure_reasons.append("cap_coupling_capability_failed")
        object.__setattr__(self, "failure_reasons", tuple(failure_reasons))

    def _validate_specification(self) -> None:
        if self.generator_identity != "fabric_v2_balanced_route_degree_cap_v1":
            raise ValueError("RouteDegreeCapAuditManifest generator identity differs")
        if self.planted_target_identity != "shared_matched_delta_rho_linear_gate_v1":
            raise ValueError(
                "RouteDegreeCapAuditManifest planted-target identity differs"
            )
        if (
            not isinstance(self.structural_route_audit_identity, str)
            or not self.structural_route_audit_identity.strip()
        ):
            raise ValueError("route audit structural identity must be nonempty")
        for name in (
            "model_input_degree_gt2_event_count",
            "model_input_external_only_coupling_event_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"route audit {name} must be a non-negative integer")
        if any(type(seed) is not int for seed in self.audit_seeds):
            raise TypeError("route audit seed IDs must be integers")
        if not isinstance(self.audit_seeds, tuple):
            raise TypeError(
                "route audit seeds must use the frozen tuple representation"
            )
        if len(self.audit_seeds) != 3 or len(set(self.audit_seeds)) != 3:
            raise ValueError("route audit requires exactly three unique seed IDs")
        tolerance_fields = {
            "epsilon_syn": self.epsilon_syn,
            "recovery_tolerance": self.recovery_tolerance,
            "seed_dispersion_tolerance": self.seed_dispersion_tolerance,
            "degree_gradient_drift_tolerance": self.degree_gradient_drift_tolerance,
            "cap_output_drift_tolerance": self.cap_output_drift_tolerance,
            "cap_gradient_drift_tolerance": self.cap_gradient_drift_tolerance,
        }
        for name, value in tolerance_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"route audit {name} must be finite and positive")

    def _validate_measurement(
        self, measurement: RouteDegreeCapAuditMeasurement
    ) -> None:
        if measurement.audit_seed not in self.audit_seeds:
            raise ValueError("route audit measurement uses an undeclared seed")
        if type(measurement.all_finite) is not bool:
            raise TypeError("route audit all_finite must be boolean")
        for name, value in asdict(measurement).items():
            if name in {
                "audit_seed",
                "family",
                "condition_id",
                "route_degree",
                "D_pre",
                "D_post",
                "all_finite",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or (
                    name not in {"planted_delta_rho", "matched_delta_rho"} and value < 0
                )
            ):
                raise ValueError(
                    f"route audit measured field {name} must be finite and non-negative"
                )
        if measurement.route_degree != measurement.D_post:
            raise ValueError("route audit route degree must equal production D_post")
        if measurement.D_pre < measurement.D_post or measurement.D_post < 1:
            raise ValueError("route audit pre/post route counts are invalid")

    def validate(self) -> None:
        """Recompute all admission fields; callers cannot supply pass flags."""

        self.__post_init__()

    def capability_checks_pass(self) -> bool:
        self.validate()
        return (
            self.implementation_valid
            and self.baseline_capability_pass
            and (not self.route_degree_catalog_applicable or self.route_degree_pass)
            and (not self.cap_coupling_catalog_applicable or self.cap_coupling_pass)
        )


@dataclass(frozen=True)
class RouteDegreeCapSyntheticConfig:
    """Frozen §16.2 balanced shared-model synthetic generator and gates."""

    generator_identity: str = "fabric_v2_balanced_route_degree_cap_v1"
    planted_target_identity: str = "shared_matched_delta_rho_linear_gate_v1"
    structural_route_audit_identity: str = "synthetic-model-input-route-audit-v1"
    model_input_degree_gt2_event_count: int = 1
    model_input_external_only_coupling_event_count: int = 1
    audit_seeds: tuple[int, int, int] = (701, 709, 719)
    gate_levels: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    planted_delta_rho: float = 1.0
    training_steps: int = 180
    learning_rate: float = 0.02
    epsilon_syn: float = 1.0e-6
    recovery_tolerance: float = 0.25
    seed_dispersion_tolerance: float = 0.20
    degree_gradient_drift_tolerance: float = 0.80
    cap_output_drift_tolerance: float = 0.25
    cap_gradient_drift_tolerance: float = 0.80

    def validate(self) -> None:
        if self.generator_identity != "fabric_v2_balanced_route_degree_cap_v1":
            raise ValueError("unknown route-degree/cap synthetic generator identity")
        if self.planted_target_identity != "shared_matched_delta_rho_linear_gate_v1":
            raise ValueError("unknown planted route-audit target identity")
        if (
            not isinstance(self.structural_route_audit_identity, str)
            or not self.structural_route_audit_identity.strip()
        ):
            raise ValueError("route synthetic structural audit identity is required")
        for name in (
            "model_input_degree_gt2_event_count",
            "model_input_external_only_coupling_event_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"route synthetic {name} must be a non-negative integer"
                )
        if len(self.audit_seeds) != 3 or len(set(self.audit_seeds)) != 3:
            raise ValueError(
                "route synthetic requires exactly three unique audit seeds"
            )
        if any(type(seed) is not int for seed in self.audit_seeds):
            raise TypeError("route synthetic audit seeds must be integers")
        if self.gate_levels != (-1.0, -0.5, 0.0, 0.5, 1.0):
            raise ValueError("balanced route synthetic gate grid is frozen")
        if self.planted_delta_rho <= 0:
            raise ValueError("route synthetic planted delta rho must be positive")
        if type(self.training_steps) is not int or self.training_steps < 1:
            raise ValueError("route synthetic training_steps must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                "route synthetic learning rate must be positive and finite"
            )
        for name in (
            "epsilon_syn",
            "recovery_tolerance",
            "seed_dispersion_tolerance",
            "degree_gradient_drift_tolerance",
            "cap_output_drift_tolerance",
            "cap_gradient_drift_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"route synthetic {name} must be positive and finite")


def bind_route_degree_cap_structural_audit(
    specification: RouteDegreeCapSyntheticConfig,
    structural_audit: pd.DataFrame,
    *,
    structural_route_audit_identity: str,
) -> RouteDegreeCapSyntheticConfig:
    """Derive the two catalog applicability states from the model-input audit."""

    required = {
        "audit_population",
        "event_id",
        "D_post",
        "external_only_coupling",
    }
    missing = sorted(required - set(structural_audit))
    if missing:
        raise ValueError(f"structural route audit misses columns: {missing}")
    if (
        not isinstance(structural_route_audit_identity, str)
        or not structural_route_audit_identity.strip()
    ):
        raise ValueError("structural route audit identity must be nonempty")
    model_rows = structural_audit.loc[
        structural_audit["audit_population"].astype(str).eq("model_input")
    ].copy()
    if model_rows.empty:
        degree_count = 0
        coupling_count = 0
    else:
        degree = model_rows.groupby("event_id", sort=False)["D_post"].max()
        coupling = model_rows.groupby("event_id", sort=False)[
            "external_only_coupling"
        ].any()
        degree_count = int((degree > 2).sum())
        coupling_count = int(coupling.sum())
    return replace(
        specification,
        structural_route_audit_identity=structural_route_audit_identity,
        model_input_degree_gt2_event_count=degree_count,
        model_input_external_only_coupling_event_count=coupling_count,
    )


@lru_cache(maxsize=4)
def run_route_degree_cap_synthetic(
    specification: RouteDegreeCapSyntheticConfig = RouteDegreeCapSyntheticConfig(),
) -> RouteDegreeCapAuditManifest:
    """Train and measure the contract's shared-model balanced route audit.

    The six admission states are derived in ``RouteDegreeCapAuditManifest``
    from the returned measurements; no caller supplies pass booleans.
    """

    specification.validate()
    raw_records: list[RouteDegreeCapAuditMeasurement] = []
    for seed in specification.audit_seeds:
        _seed_everything(seed)
        conditions = _route_synthetic_inputs(
            torch.tensor(specification.gate_levels, dtype=torch.float32)
        )
        model = FABRICV2Model(
            cis_dim=3,
            dna_base_dim=2,
            dna_interaction_dim=0,
            rna_base_dim=1,
            rna_interaction_dim=0,
            dynamic_dim=4,
            hidden_dim=8,
            attention_heads=2,
            path_hidden_dim=8,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=specification.learning_rate)
        target_gate = torch.tensor(specification.gate_levels, dtype=torch.float32)
        target = specification.planted_delta_rho * target_gate
        for _ in range(specification.training_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for inputs in conditions.values():
                logits = model(inputs, condition="full").path_logits
                losses.append(torch.mean(((logits[:, 0] - logits[:, 1]) - target) ** 2))
                local_outputs = []
                for gate_value in (-0.1, 0.1):
                    local_input = replace(
                        inputs,
                        dna=replace(
                            inputs.dna,
                            gate=torch.tensor([[gate_value]], dtype=torch.float32),
                        ),
                        rna=replace(inputs.rna, gate=torch.empty(1, 0)),
                    )
                    local_logits = model(local_input, condition="full").path_logits
                    local_outputs.append(local_logits[0, 0] - local_logits[0, 1])
                center_slope = (local_outputs[1] - local_outputs[0]) / 0.2
                losses.append(
                    0.25 * (center_slope - specification.planted_delta_rho) ** 2
                )
            torch.stack(losses).mean().backward()
            _assert_finite_gradients(model, require_all=False)
            optimizer.step()

        evaluated = {
            condition_id: _measure_route_synthetic_condition(
                model,
                condition_id,
                specification.planted_delta_rho,
                specification.epsilon_syn,
            )
            for condition_id in conditions
        }
        degree_reference = evaluated["degree_2"]
        cap_reference = evaluated["cap_4_plus_2"]
        cap_decomposition = _measure_cap_decomposition(model)
        for condition_id, values in evaluated.items():
            if condition_id.startswith("degree_"):
                family = "single_anchor"
                paired_output_drift = abs(
                    values["matched_delta_rho"] - degree_reference["matched_delta_rho"]
                ) / max(abs(specification.planted_delta_rho), specification.epsilon_syn)
                paired_gradient_drift = abs(
                    values["absolute_gate_gradient"]
                    - degree_reference["absolute_gate_gradient"]
                ) / max(
                    abs(degree_reference["absolute_gate_gradient"]),
                    specification.epsilon_syn,
                )
                cap_values = (0.0, 0.0, 0.0)
            else:
                family = "multi_anchor_cap"
                paired_output_drift = abs(
                    values["matched_delta_rho"] - cap_reference["matched_delta_rho"]
                ) / max(abs(specification.planted_delta_rho), specification.epsilon_syn)
                paired_gradient_drift = abs(
                    values["absolute_gate_gradient"]
                    - cap_reference["absolute_gate_gradient"]
                ) / max(
                    abs(cap_reference["absolute_gate_gradient"]),
                    specification.epsilon_syn,
                )
                cap_values = cap_decomposition[condition_id]
            raw_records.append(
                RouteDegreeCapAuditMeasurement(
                    audit_seed=seed,
                    family=family,
                    condition_id=condition_id,
                    route_degree=values["route_degree"],
                    D_pre=8 if family == "multi_anchor_cap" else values["route_degree"],
                    D_post=values["route_degree"],
                    planted_delta_rho=specification.planted_delta_rho,
                    matched_delta_rho=values["matched_delta_rho"],
                    recovery_error=abs(
                        values["matched_delta_rho"] - specification.planted_delta_rho
                    )
                    / max(
                        abs(specification.planted_delta_rho),
                        specification.epsilon_syn,
                    ),
                    absolute_gate_gradient=values["absolute_gate_gradient"],
                    paired_output_drift=paired_output_drift,
                    paired_gradient_drift=paired_gradient_drift,
                    seed_dispersion=0.0,
                    route_l1_mass=values["route_l1_mass"],
                    route_weight_identity_error=values["route_weight_identity_error"],
                    event_frobenius_norm=values["event_frobenius_norm"],
                    frobenius_identity_error=values["frobenius_identity_error"],
                    cap_loss_norm=cap_values[0],
                    cap_renormalization_gain_norm=cap_values[1],
                    cap_decomposition_error=cap_values[2],
                    delta_a_norm_median=values["delta_a_norm_median"],
                    delta_a_norm_max=values["delta_a_norm_max"],
                    delta_y_norm=values["delta_y_norm"],
                    delta_normalized_token_norm=values["delta_normalized_token_norm"],
                    delta_hidden_norm=values["delta_hidden_norm"],
                    event_to_background_pre_layernorm_norm_ratio=values[
                        "event_to_background_pre_layernorm_norm_ratio"
                    ],
                    neutralization_identity_error=values[
                        "neutralization_identity_error"
                    ],
                    joint_projection_identity_error=values[
                        "joint_projection_identity_error"
                    ],
                    reproducibility_error=values["reproducibility_error"],
                    all_finite=values["all_finite"],
                )
            )

    dispersions = {}
    for condition_id in {record.condition_id for record in raw_records}:
        values = [
            record.matched_delta_rho
            for record in raw_records
            if record.condition_id == condition_id
        ]
        dispersions[condition_id] = (max(values) - min(values)) / max(
            abs(specification.planted_delta_rho), specification.epsilon_syn
        )
    measurements = tuple(
        replace(record, seed_dispersion=dispersions[record.condition_id])
        for record in raw_records
    )
    return RouteDegreeCapAuditManifest(
        generator_identity=specification.generator_identity,
        planted_target_identity=specification.planted_target_identity,
        structural_route_audit_identity=specification.structural_route_audit_identity,
        model_input_degree_gt2_event_count=(
            specification.model_input_degree_gt2_event_count
        ),
        model_input_external_only_coupling_event_count=(
            specification.model_input_external_only_coupling_event_count
        ),
        audit_seeds=specification.audit_seeds,
        epsilon_syn=specification.epsilon_syn,
        recovery_tolerance=specification.recovery_tolerance,
        seed_dispersion_tolerance=specification.seed_dispersion_tolerance,
        degree_gradient_drift_tolerance=specification.degree_gradient_drift_tolerance,
        cap_output_drift_tolerance=specification.cap_output_drift_tolerance,
        cap_gradient_drift_tolerance=specification.cap_gradient_drift_tolerance,
        measurements=measurements,
    )


def _route_synthetic_inputs(
    gate_levels: torch.Tensor,
) -> dict[str, GeneCellModelInput]:
    if gate_levels.ndim != 1 or gate_levels.numel() < 1:
        raise ValueError("route synthetic gate levels must be a non-empty vector")
    edge_count = 12
    # Keep one fixed processing graph for every matched condition.  Every
    # processing edge belongs to at least one legal path: added single-anchor
    # routes enter paired nuisance branches, never a degree-specific skeleton.
    path_rows = (
        (0, 1, 2, 4, 6, 8, 10, 11),
        (0, 1, 3, 5, 7, 9, 10, 11),
    )
    incidence = torch.zeros(2, edge_count)
    for path_index, edges in enumerate(path_rows):
        incidence[path_index, list(edges)] = 1.0
    local_pairs = sorted(
        {
            pair
            for path in path_rows
            for left, right in zip(path[:-1], path[1:])
            for pair in ((left, right), (right, left))
        }
    )
    local_edge_index = torch.tensor(local_pairs, dtype=torch.long).T.contiguous()
    edge_axis = torch.linspace(-1.0, 1.0, edge_count)
    cis = torch.stack(
        (edge_axis, edge_axis.square(), torch.sin(torch.pi * edge_axis)), dim=1
    )
    batch_size = gate_levels.numel()
    empty_rna = RoutedModalityInput(
        route_event_index=torch.empty(0, dtype=torch.long),
        route_edge_index=torch.empty(0, dtype=torch.long),
        route_weight=torch.empty(0),
        route_base_features=torch.empty(0, 1),
        route_interaction_features=torch.empty(0, 0),
        interaction_active_mask=torch.empty(0, dtype=torch.bool),
        event_gate_key_index=torch.empty(0, dtype=torch.long),
        gate=torch.empty(batch_size, 0),
    )
    focal_routes = (2, 3)
    single_anchor_nuisance_pairs = ((4, 5), (6, 7), (8, 9))
    multi_anchor_four = (4, 5, 6, 7)
    multi_anchor_two = (8, 9)
    single_anchor_definitions = {
        "degree_2": (
            focal_routes,
            ((1.0, 0.0),) * 2,
        ),
        "degree_4": (
            focal_routes + single_anchor_nuisance_pairs[0],
            ((1.0, 0.0),) * 4,
        ),
        "degree_8": (
            focal_routes
            + tuple(edge for pair in single_anchor_nuisance_pairs for edge in pair),
            ((1.0, 0.0),) * 8,
        ),
    }
    # The multi-anchor family starts from the same 2+4+2 candidate catalog.
    # A toy per-anchor cap of one is applied to static competitor events that
    # are gate-inactive in the model.  Their only effect is whether the focal
    # four-route/two-route event survives production capping; dropped events
    # are absent before model injection, exactly as in production routing.
    static_competitor_load = {
        "cap_4_plus_2": (0, 0),
        "cap_2": (1, 0),
        "cap_0": (1, 1),
    }
    toy_event_cap = 1
    multi_anchor_definitions = {}
    for condition_id, (
        four_anchor_load,
        two_anchor_load,
    ) in static_competitor_load.items():
        retained = focal_routes
        if four_anchor_load < toy_event_cap:
            retained += multi_anchor_four
        if two_anchor_load < toy_event_cap:
            retained += multi_anchor_two
        multi_anchor_definitions[condition_id] = (
            retained,
            ((1.0, 0.0),) * len(retained),
        )
    route_definitions = {**single_anchor_definitions, **multi_anchor_definitions}
    if route_definitions["degree_2"][0] != focal_routes:
        raise AssertionError("single-anchor focal routes changed")
    if route_definitions["degree_4"][0][2:] != single_anchor_nuisance_pairs[0]:
        raise AssertionError("degree-4 routes are not a symmetric nuisance pair")
    if route_definitions["degree_8"][0][2:] != tuple(
        edge for pair in single_anchor_nuisance_pairs for edge in pair
    ):
        raise AssertionError("degree-8 routes are not symmetric nuisance pairs")
    if tuple(map(len, (focal_routes, multi_anchor_four, multi_anchor_two))) != (
        2,
        4,
        2,
    ):
        raise AssertionError("multi-anchor candidate family is not 2+4+2")
    if tuple(
        len(route_definitions[name][0]) for name in ("cap_4_plus_2", "cap_2", "cap_0")
    ) != (8, 4, 2):
        raise AssertionError("multi-anchor production degrees are not 8/4/2")
    if any(
        len(set(feature_rows)) != 1 for _, feature_rows in route_definitions.values()
    ):
        raise AssertionError("one physical event changed route feature direction")
    result = {}
    for condition_id, (route_edges, feature_rows) in route_definitions.items():
        degree = len(route_edges)
        dna = RoutedModalityInput(
            route_event_index=torch.zeros(degree, dtype=torch.long),
            route_edge_index=torch.tensor(route_edges, dtype=torch.long),
            route_weight=torch.full((degree,), 1.0 / degree),
            route_base_features=torch.tensor(feature_rows, dtype=torch.float32),
            route_interaction_features=torch.empty(degree, 0),
            interaction_active_mask=torch.empty(0, dtype=torch.bool),
            event_gate_key_index=torch.tensor([0], dtype=torch.long),
            gate=gate_levels[:, None].clone(),
        )
        result[condition_id] = GeneCellModelInput(
            cis_features=cis,
            local_edge_index=local_edge_index,
            dna=dna,
            rna=empty_rna,
            path_edge_incidence=incidence.to_sparse_coo().coalesce(),
            path_first_edge_index=torch.tensor([0, 0]),
            path_last_edge_index=torch.tensor([11, 11]),
            log_edge_count=torch.log1p(torch.tensor([8.0, 8.0])),
        )
    return result


def _measure_route_synthetic_condition(
    model: FABRICV2Model,
    condition_id: str,
    planted_delta_rho: float,
    epsilon_syn: float,
) -> dict[str, float | int | bool]:
    inputs = _route_synthetic_inputs(torch.tensor([1.0]))[condition_id]
    neutral_inputs = replace(
        inputs,
        dna=inputs.dna.with_event_keep_mask(torch.tensor([False])),
    )
    model.eval()
    with torch.no_grad():
        full = model(inputs, condition="full")
        neutral = model(neutral_inputs, condition="full")
        repeated = model(inputs, condition="full")
    delta_a = full.full_dna_aggregate - neutral.full_dna_aggregate
    delta_x = full.joint_input - neutral.joint_input
    delta_y = full.joint_projected - neutral.joint_projected
    delta_normalized = full.normalized_tokens - neutral.normalized_tokens
    delta_hidden = full.edge_states - neutral.edge_states
    matched_delta_rho = float(
        (full.path_logits[0, 0] - full.path_logits[0, 1])
        - (neutral.path_logits[0, 0] - neutral.path_logits[0, 1])
    )

    # The balanced corpus is centered at G=0; measure the local planted slope
    # there rather than at an endpoint where a nonlinear readout can be flat
    # while still interpolating all frozen target points.
    gate = torch.zeros(1, 1, requires_grad=True)
    gradient_inputs = replace(inputs, dna=replace(inputs.dna, gate=gate))
    gradient_logits = model(gradient_inputs, condition="full").path_logits
    gate_gradient = torch.autograd.grad(
        gradient_logits[0, 0] - gradient_logits[0, 1], gate
    )[0]

    projected = model.dna_aggregator.route_projection(inputs.dna)
    expected_a = projected.new_zeros(delta_a.shape[1:])
    expected_a.index_add_(
        0,
        inputs.dna.route_edge_index,
        inputs.dna.route_weight[:, None] * projected,
    )
    neutralization_error = float(torch.max(torch.abs(delta_a[0] - expected_a)))
    expected_y = torch.matmul(delta_x, model.joint_projection.weight.T)
    joint_projection_error = float(torch.max(torch.abs(delta_y - expected_y)))
    route_degree = inputs.dna.route_weight.numel()
    route_weight_identity_error = float(
        torch.max(
            torch.abs(
                inputs.dna.route_weight
                - torch.full_like(inputs.dna.route_weight, 1.0 / route_degree)
            )
        )
    )
    event_frobenius = float(torch.linalg.vector_norm(delta_a))
    if condition_id.startswith("degree_"):
        expected_frobenius = float(
            torch.linalg.vector_norm(projected[0]) / np.sqrt(route_degree)
        )
        frobenius_error = abs(event_frobenius - expected_frobenius)
    else:
        frobenius_error = 0.0
    active_edge_norms = torch.linalg.vector_norm(
        delta_a[0, torch.unique(inputs.dna.route_edge_index)], dim=1
    )
    background = float(torch.linalg.vector_norm(neutral.joint_projected))
    ratio = float(torch.linalg.vector_norm(delta_y)) / max(background, epsilon_syn)
    tensors = (
        full.path_logits,
        neutral.path_logits,
        delta_a,
        delta_y,
        delta_normalized,
        delta_hidden,
        gate_gradient,
    )
    return {
        "route_degree": route_degree,
        "matched_delta_rho": matched_delta_rho,
        "absolute_gate_gradient": float(torch.abs(gate_gradient).item()),
        "route_l1_mass": float(inputs.dna.route_weight.sum()),
        "route_weight_identity_error": route_weight_identity_error,
        "event_frobenius_norm": event_frobenius,
        "frobenius_identity_error": frobenius_error,
        "delta_a_norm_median": float(torch.median(active_edge_norms)),
        "delta_a_norm_max": float(torch.max(active_edge_norms)),
        "delta_y_norm": float(torch.linalg.vector_norm(delta_y)),
        "delta_normalized_token_norm": float(
            torch.linalg.vector_norm(delta_normalized)
        ),
        "delta_hidden_norm": float(torch.linalg.vector_norm(delta_hidden)),
        "event_to_background_pre_layernorm_norm_ratio": ratio,
        "neutralization_identity_error": neutralization_error,
        "joint_projection_identity_error": joint_projection_error,
        "reproducibility_error": float(
            torch.max(torch.abs(repeated.path_logits - full.path_logits))
        ),
        "all_finite": bool(all(torch.isfinite(value).all() for value in tensors)),
    }


def _measure_cap_decomposition(
    model: FABRICV2Model,
) -> dict[str, tuple[float, float, float]]:
    inputs = _route_synthetic_inputs(torch.tensor([1.0]))
    full_input = inputs["cap_4_plus_2"]
    with torch.no_grad():
        full_projection = model.dna_aggregator.route_projection(full_input.dna)
        full_aggregate = model.dna_aggregator(
            full_input.dna, full_input.cis_features.shape[0]
        )[0]
    result = {}
    for condition_id in ("cap_4_plus_2", "cap_2", "cap_0"):
        condition = inputs[condition_id]
        with torch.no_grad():
            actual = model.dna_aggregator(
                condition.dna, condition.cis_features.shape[0]
            )[0]
        raw_retained = full_aggregate.new_zeros(full_aggregate.shape)
        full_edge_to_index = {
            int(edge): index
            for index, edge in enumerate(full_input.dna.route_edge_index.tolist())
        }
        retained_indices = torch.tensor(
            [full_edge_to_index[int(edge)] for edge in condition.dna.route_edge_index],
            dtype=torch.long,
        )
        raw_retained.index_add_(
            0,
            full_input.dna.route_edge_index[retained_indices],
            torch.full(
                (len(retained_indices), 1),
                1.0 / 8.0,
                dtype=full_projection.dtype,
            )
            * full_projection[retained_indices],
        )
        cap_loss = full_aggregate - raw_retained
        renormalization_gain = actual - raw_retained
        residual = (actual - full_aggregate) - (renormalization_gain - cap_loss)
        result[condition_id] = (
            float(torch.linalg.vector_norm(cap_loss)),
            float(torch.linalg.vector_norm(renormalization_gain)),
            float(torch.max(torch.abs(residual))),
        )
    return result


@dataclass(frozen=True)
class ValidationPrediction:
    gene_id: str
    cell_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    path_logits: torch.Tensor  # detached CPU [B,P]
    compatible_path_indices: torch.Tensor  # detached CPU [K,W]
    compatible_path_mask: torch.Tensor
    row_cell_index: torch.Tensor
    molecule_count: torch.Tensor


@dataclass(frozen=True)
class ValidationSnapshot:
    split: str
    weighted_nll_sum: float
    informative_molecule_mass: float
    predictions: tuple[ValidationPrediction, ...]

    @property
    def nll(self) -> float:
        return self.weighted_nll_sum / self.informative_molecule_mass


@dataclass(frozen=True)
class MonitorRecord:
    seed: int
    condition: str
    epoch: int
    fields: Mapping[str, float | int | str | bool]
    sealed: bool = True
    selection_eligible: bool = False


@dataclass(frozen=True)
class ConditionResult:
    model: FABRICV2Model
    history: pd.DataFrame
    monitor_records: tuple[MonitorRecord, ...]
    best_epoch: int
    best_validation_nll: float
    best_ont_matrix_kl_count_weighted: float | None
    validation_informative_molecule_mass: float
    optimizer_state_dict: Mapping[str, object]
    lr_scheduler_state_dict: Mapping[str, object] | None


@dataclass(frozen=True)
class TrainingRunResult:
    manifest: TrainingRunManifest
    result: ConditionResult
    metrics: pd.DataFrame


@dataclass(frozen=True)
class OptimizerTuningRun:
    """One train/validation-only dynamic-model fit in the §15.5 grid."""

    lambda_base: float
    lambda_int: float
    tuning_seed: int
    condition: str
    best_validation_nll: float


@dataclass(frozen=True)
class OptimizerGridSelection:
    """Complete ordered evidence needed to freeze one shrinkage pair."""

    grid_order: tuple[tuple[float, float], ...]
    tuning_seeds: tuple[int, ...]
    selection_conditions: tuple[str, str, str]
    runs: tuple[OptimizerTuningRun, ...]
    condition_mean_validation_nll: tuple[tuple[float, float, float], ...]
    aggregate_validation_nll: tuple[float, ...]
    selected_pair: tuple[float, float]

    def frozen_config_fields(self) -> dict[str, object]:
        """Return only the fields copied into the frozen optimizer config."""

        return {
            "selection_status": "FROZEN",
            "selected_pair": list(self.selected_pair),
            "aggregate_validation_nll": list(self.aggregate_validation_nll),
        }


EpochMonitor = Callable[
    [str, int, ValidationSnapshot], Mapping[str, float | int | str | bool]
]
EpochCheckpoint = Callable[[Mapping[str, object]], None]


def _validate_optional_optimizer_selection_metadata(
    optimizer: Mapping[str, object],
    *,
    lambda_base: float,
    lambda_int: float,
) -> None:
    fields = {
        "selection_conditions",
        "grid",
        "tuning_seeds",
        "selection_status",
        "selected_pair",
        "aggregate_validation_nll",
    }
    present = fields & set(optimizer)
    if not present:
        return
    if present != fields:
        raise ValueError(
            "optional optimizer selection metadata must be complete; missing="
            f"{sorted(fields - present)}"
        )
    conditions = tuple(optimizer["selection_conditions"])
    if conditions not in {("full",), ("cis_dna", "cis_rna", "full")}:
        raise ValueError("optimizer selection metadata has unknown model conditions")
    raw_grid = optimizer["grid"]
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("optimizer grid must be a nonempty ordered list")
    grid: list[tuple[float, float]] = []
    for raw_pair in raw_grid:
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_pair
            )
        ):
            raise TypeError("each optimizer grid entry must be two numeric penalties")
        pair = (float(raw_pair[0]), float(raw_pair[1]))
        if not np.isfinite(pair).all() or not (pair[1] > pair[0] >= 0):
            raise ValueError(f"invalid optimizer grid entry: {pair}")
        grid.append(pair)
    if len(set(grid)) != len(grid):
        raise ValueError("optimizer grid entries must be unique")
    tuning_seeds = optimizer["tuning_seeds"]
    if (
        not isinstance(tuning_seeds, list)
        or any(type(value) is not int for value in tuning_seeds)
        or len(set(tuning_seeds)) != len(tuning_seeds)
    ):
        raise ValueError("optimizer tuning seeds must be a unique integer list")
    status = optimizer["selection_status"]
    if status not in {
        "PENDING",
        "PREDECLARED_IMPLEMENTATION",
        "PREDECLARED_NO_TUNING",
        "FROZEN",
    }:
        raise ValueError("optimizer selection_status is not recognized")
    raw_selected_pair = optimizer["selected_pair"]
    if status == "PENDING":
        if raw_selected_pair is not None:
            raise ValueError("pending optimizer selection cannot name a selected pair")
    else:
        if not isinstance(raw_selected_pair, list) or len(raw_selected_pair) != 2:
            raise TypeError("selected_pair must be [lambda_base, lambda_int]")
        selected_pair = (float(raw_selected_pair[0]), float(raw_selected_pair[1]))
        if selected_pair != (float(lambda_base), float(lambda_int)):
            raise ValueError("selected_pair differs from the runtime penalties")
        if selected_pair not in grid:
            raise ValueError("selected_pair is absent from the optimizer grid")
    scores = optimizer["aggregate_validation_nll"]
    if status == "FROZEN":
        if not isinstance(scores, list) or len(scores) != len(grid):
            raise ValueError("frozen selection requires one score per grid pair")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in scores
        ):
            raise ValueError("optimizer aggregate validation scores must be finite")
        ranked = [
            (float(score), -pair[1], -pair[0], order, pair)
            for order, (pair, score) in enumerate(zip(grid, scores, strict=True))
        ]
        if min(ranked)[-1] != (float(lambda_base), float(lambda_int)):
            raise ValueError(
                "selected_pair does not follow the frozen aggregate scores"
            )
    elif scores is not None:
        raise ValueError("non-frozen optimizer selection cannot store aggregate scores")
    if status == "PREDECLARED_NO_TUNING" and (
        tuning_seeds or grid != [(float(lambda_base), float(lambda_int))]
    ):
        raise ValueError("PREDECLARED_NO_TUNING requires only the runtime pair")


def _validate_lr_scheduler_fields(
    *,
    name: object,
    factor: object,
    patience: object,
    min_lr: object,
    learning_rate: float,
) -> None:
    if name == "constant":
        if any(value is not None for value in (factor, patience, min_lr)):
            raise ValueError(
                "constant lr_scheduler cannot define factor, patience, or min_lr"
            )
        return
    if name != "reduce_on_plateau":
        raise ValueError(
            "optimizer.lr_scheduler.name must be constant or reduce_on_plateau"
        )
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not np.isfinite(factor)
        or not 0 < factor < 1
    ):
        raise ValueError("reduce_on_plateau factor must lie strictly between 0 and 1")
    if type(patience) is not int or patience < 0:
        raise ValueError("reduce_on_plateau patience must be a non-negative integer")
    if (
        isinstance(min_lr, bool)
        or not isinstance(min_lr, (int, float))
        or not np.isfinite(min_lr)
        or not 0 < min_lr <= learning_rate
    ):
        raise ValueError(
            "reduce_on_plateau min_lr must be positive and no greater than learning_rate"
        )


def _validate_optimizer_config(optimizer: object) -> None:
    if not isinstance(optimizer, Mapping):
        raise TypeError("optimizer config must be a mapping")
    if optimizer.get("family") != "AdamW":
        raise ValueError("FABRIC V2 optimizer family is AdamW")
    for name in ("learning_rate", "lambda_base", "lambda_int"):
        value = optimizer.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise TypeError(f"optimizer.{name} must be a finite number")
    learning_rate = float(optimizer["learning_rate"])
    if learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    scheduler = optimizer.get("lr_scheduler")
    if not isinstance(scheduler, Mapping):
        raise TypeError("optimizer.lr_scheduler must be a mapping")
    _validate_lr_scheduler_fields(
        name=scheduler.get("name"),
        factor=scheduler.get("factor"),
        patience=scheduler.get("patience"),
        min_lr=scheduler.get("min_lr"),
        learning_rate=learning_rate,
    )
    expected_scheduler_fields = (
        {"name"}
        if scheduler.get("name") == "constant"
        else {"name", "factor", "patience", "min_lr"}
    )
    if set(scheduler) != expected_scheduler_fields:
        raise ValueError(
            "optimizer.lr_scheduler fields differ from the selected scheduler"
        )
    gradient_clip_norm = optimizer.get("gradient_clip_norm")
    if (
        isinstance(gradient_clip_norm, bool)
        or not isinstance(gradient_clip_norm, (int, float))
        or not np.isfinite(gradient_clip_norm)
        or gradient_clip_norm < 0
    ):
        raise ValueError("optimizer.gradient_clip_norm must be finite and non-negative")
    lambda_base = float(optimizer["lambda_base"])
    lambda_int = float(optimizer["lambda_int"])
    if not (lambda_int > lambda_base >= 0):
        raise ValueError("optimizer requires lambda_int > lambda_base >= 0")
    _validate_optional_optimizer_selection_metadata(
        optimizer,
        lambda_base=lambda_base,
        lambda_int=lambda_int,
    )


def resolve_run_config(
    config: Mapping[str, object],
    *,
    learning_rate: float | None = None,
    lr_scheduler: str | None = None,
    lr_factor: float | None = None,
    lr_patience: int | None = None,
    min_lr: float | None = None,
    gradient_clip_norm: float | None = None,
    lambda_base: float | None = None,
    lambda_int: float | None = None,
    max_train_gene_cells_per_gene: int | None = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
) -> dict[str, object]:
    """Apply one command's explicit hyperparameter overrides and validate them."""

    resolved = copy.deepcopy(dict(config))
    optimizer = resolved["optimizer"]
    training = resolved["training"]
    if learning_rate is not None:
        optimizer["learning_rate"] = learning_rate
    if lambda_base is not None or lambda_int is not None:
        selection_fields = {
            "selection_conditions",
            "grid",
            "tuning_seeds",
            "selection_status",
            "selected_pair",
            "aggregate_validation_nll",
        }
        if selection_fields & set(optimizer):
            raise ValueError(
                "command penalty overrides cannot replace frozen optimizer-selection metadata"
            )
        if lambda_base is not None:
            optimizer["lambda_base"] = lambda_base
        if lambda_int is not None:
            optimizer["lambda_int"] = lambda_int

    current_scheduler = optimizer["lr_scheduler"]
    scheduler_name = lr_scheduler or current_scheduler["name"]
    scheduler_parameter_override = any(
        value is not None for value in (lr_factor, lr_patience, min_lr)
    )
    if scheduler_name == "constant":
        if scheduler_parameter_override:
            raise ValueError(
                "plateau scheduler parameters require --lr-scheduler reduce_on_plateau"
            )
        optimizer["lr_scheduler"] = {"name": "constant"}
    elif scheduler_name == "reduce_on_plateau":
        previous = (
            current_scheduler if current_scheduler["name"] == scheduler_name else {}
        )
        optimizer["lr_scheduler"] = {
            "name": scheduler_name,
            "factor": (
                lr_factor if lr_factor is not None else previous.get("factor", 0.5)
            ),
            "patience": (
                lr_patience if lr_patience is not None else previous.get("patience", 1)
            ),
            "min_lr": min_lr if min_lr is not None else previous.get("min_lr", 1.0e-5),
        }
    else:
        raise ValueError("lr_scheduler must be constant or reduce_on_plateau")
    if gradient_clip_norm is not None:
        optimizer["gradient_clip_norm"] = gradient_clip_norm
    if max_train_gene_cells_per_gene is not None:
        training["max_train_gene_cells_per_gene_per_epoch"] = (
            max_train_gene_cells_per_gene
        )
    if max_epochs is not None:
        training["max_epochs"] = max_epochs
    if early_stopping_patience is not None:
        training["early_stopping_patience"] = early_stopping_patience

    _validate_optimizer_config(optimizer)
    for name in (
        "max_train_gene_cells_per_gene_per_epoch",
        "max_epochs",
        "early_stopping_patience",
    ):
        if type(training.get(name)) is not int or training[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if training["early_stopping_patience"] > training["max_epochs"]:
        raise ValueError("early_stopping_patience cannot exceed max_epochs")
    return resolved


def load_config(path: str | Path) -> dict[str, object]:
    """Load and validate the single V2 configuration schema."""

    config = yaml.safe_load(Path(path).read_text())
    if (
        not isinstance(config, dict)
        or config.get("contract") != "FABRIC_ARCHITECTURE_V2"
    ):
        raise ValueError("training config is not bound to FABRIC_ARCHITECTURE_V2")
    execution = config.get("execution")
    if not isinstance(execution, dict) or not isinstance(execution.get("scope"), str):
        raise TypeError("execution.scope must be explicit")
    for flag in ("training_authorized", "final_test_authorized"):
        if type(execution.get(flag)) is not bool:
            raise TypeError(f"execution.{flag} must be an explicit boolean")
    if execution["scope"] not in {
        "toy",
        "fixture",
        FULL_COHORT_SCOPE,
    }:
        raise ValueError("execution.scope must be toy, fixture, or full_cohort")
    inputs = config.get("inputs")
    resources = config.get("resources")
    if not isinstance(inputs, dict) or type(inputs.get("frozen")) is not bool:
        raise TypeError("inputs.frozen must be an explicit boolean")
    if not isinstance(resources, dict) or type(resources.get("frozen")) is not bool:
        raise TypeError("resources.frozen must be an explicit boolean")
    if resources.get("batch_policy") != "gene_shape_adaptive_v1":
        raise ValueError("resources.batch_policy must be gene_shape_adaptive_v1")
    if resources.get("compute_precision") != "float32_highest":
        raise ValueError("resources.compute_precision must be float32_highest")
    if type(resources.get("prefetch_backed_gene_shards")) is not bool:
        raise TypeError("resources.prefetch_backed_gene_shards must be boolean")
    for name in (
        "target_gpu_allocated_bytes",
        "unmodeled_gpu_reserve_bytes",
        "max_cells_per_gpu_batch",
    ):
        value = resources.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"resources.{name} must be a positive integer")
    if (
        resources["unmodeled_gpu_reserve_bytes"]
        >= resources["target_gpu_allocated_bytes"]
    ):
        raise ValueError("unmodeled GPU reserve must be below the allocation target")
    for name in (
        "train_bytes_per_shape_element",
        "evaluation_bytes_per_shape_element",
    ):
        value = resources.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"resources.{name} must be positive and finite")
    gpu_total = resources.get("gpu_total_memory_bytes")
    if gpu_total is not None and (
        type(gpu_total) is not int
        or gpu_total < resources["target_gpu_allocated_bytes"]
    ):
        raise ValueError("GPU total memory must cover the adaptive allocation target")
    batching_tolerance = resources.get("batching_probability_tolerance")
    if (
        isinstance(batching_tolerance, bool)
        or not isinstance(batching_tolerance, (int, float))
        or not np.isfinite(batching_tolerance)
        or batching_tolerance <= 0
    ):
        raise ValueError(
            "resources.batching_probability_tolerance must be positive and finite"
        )
    training = config.get("training")
    if not isinstance(training, dict):
        raise TypeError("training config must be a mapping")
    if "seed" in training or "seeds" in training:
        raise ValueError("seed is selected per command, not stored in the config")
    if "condition" in training or "conditions" in training:
        raise ValueError("condition is selected per command, not stored in the config")
    for name in ("max_epochs", "early_stopping_patience"):
        if type(training.get(name)) is not int or training[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if training["early_stopping_patience"] > training["max_epochs"]:
        raise ValueError("early_stopping_patience cannot exceed max_epochs")
    if (
        training.get("primary_epoch_unit")
        != "sampled_informative_gene_cell_horvitz_thompson"
    ):
        raise ValueError("all V2 modes require grouped gene-cell sampling")
    cap = training.get("max_train_gene_cells_per_gene_per_epoch")
    if type(cap) is not int or cap < 1:
        raise ValueError("V2 per-gene train gene-cell cap must be a positive integer")
    if training.get("resample_train_gene_cells_each_epoch") is not True:
        raise ValueError("V2 must resample train gene-cells each epoch")
    if "sampling_seed" in training:
        raise ValueError(
            "the command seed also determines deterministic train sampling"
        )
    if training.get("selected_gene_cell_ec_rows") != "all_informative_rows":
        raise ValueError("selected V2 gene-cells must retain all informative EC rows")
    if (
        training.get("sampling_estimator")
        != "horvitz_thompson_full_train_molecule_total"
    ):
        raise ValueError("V2 sampling estimator identity differs")
    if training.get("optimizer_step_unit") != "train_positive_gene":
        raise ValueError("V2 optimizer step unit must be one train-positive gene")
    if training.get("gene_microbatch_gradient_accumulation") is not True:
        raise ValueError(
            "V2 must accumulate all attention microbatches before the gene step"
        )
    model = config["model"]
    if not isinstance(model, dict):
        raise TypeError("model config must be a mapping")
    for name in (
        "dynamic_projection_dim",
        "hidden_dim",
        "attention_heads",
        "graphgps_layers",
        "path_hidden_dim",
    ):
        if type(model.get(name)) is not int or model[name] < 1:
            raise ValueError(f"model.{name} must be a positive integer")
    if model["graphgps_layers"] != 1:
        raise ValueError("FABRIC V2 has exactly one GraphGPS block")
    if model["hidden_dim"] % model["attention_heads"]:
        raise ValueError("model hidden_dim must be divisible by attention_heads")
    _validate_optimizer_config(config["optimizer"])
    if bool(config.get("deferred_cis_extension", {}).get("enabled", False)):
        raise ValueError(
            "the deferred AlphaGenome CIS extension is not a V2 runtime input"
        )
    monitor = config.get("monitor")
    if not isinstance(monitor, dict) or type(monitor.get("enabled")) is not bool:
        raise TypeError("monitor.enabled must be an explicit boolean")
    if monitor.get("timing") != "once_after_each_completed_epoch":
        raise ValueError("per-epoch monitor timing is frozen by V2")
    if (
        monitor.get("selection_eligible") is not False
        or monitor.get("sealed") is not True
    ):
        raise ValueError("per-epoch monitor must be sealed and selection-ineligible")
    if monitor.get("metric") != "ont_matrix_kl_count_weighted":
        raise ValueError("V2 per-epoch ONT monitor has one count-weighted KL metric")
    if monitor.get("scope_policy") != (
        "likelihood_informative_validation_cell_gene_with_at_least_two_"
        "positive_ont_paths"
    ):
        raise ValueError("V2 ONT monitor scope policy differs")
    target_root = monitor.get("target_root")
    if monitor["enabled"] and (
        not isinstance(target_root, str) or not target_root.strip()
    ):
        raise ValueError("enabled ONT monitor requires one validation target root")
    if not monitor["enabled"] and target_root is not None:
        raise ValueError("disabled ONT monitor cannot name a validation target")
    if execution["scope"] == FULL_COHORT_SCOPE:
        if inputs["frozen"] is not True:
            raise ValueError("full_cohort inputs must be frozen")
        if config.get("status") in {
            "READY_AWAITING_TRAINING_AUTHORIZATION",
            "READY_TO_LAUNCH_FULL_COHORT_RUN",
        }:
            if resources["frozen"] is not True:
                raise ValueError("ready full_cohort resources must be frozen")
        elif resources["frozen"] is not False:
            raise ValueError(
                "full_cohort resources must remain unfrozen while readiness is stale"
            )
        if inputs.get("structural_candidate_count") != 17_706:
            raise ValueError("full_cohort must bind all 17,706 structural candidates")
        if inputs.get("g_fit_gene_count") != 17_600:
            raise ValueError("full_cohort must bind all 17,600 G_fit genes")
        if inputs.get("graph_only_gene_count") != 106:
            raise ValueError("full_cohort must preserve 106 graph-only genes")
        if inputs.get("matrix_structural_path_count") != 90_672:
            raise ValueError("full_cohort must bind all 90,672 structural paths")
        for name in ("input_manifest_id", "compatibility_artifact_id"):
            value = inputs.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"full_cohort inputs.{name} must be frozen")
        if inputs.get("test_compatible_rows") != "ABSENT_NOT_MATERIALIZED":
            raise ValueError("full_cohort requires absent test compatible rows")
    return config


def training_manifest_from_config(
    config: Mapping[str, object],
    *,
    seed: int,
    condition: str,
) -> TrainingRunManifest:
    training = config["training"]
    optimizer = config["optimizer"]
    scheduler = optimizer["lr_scheduler"]
    resources = config.get("resources", {})
    inputs = config.get("inputs", {})
    manifest = TrainingRunManifest(
        seed=seed,
        condition=condition,
        learning_rate=float(optimizer["learning_rate"]),
        lr_scheduler_name=str(scheduler["name"]),
        lr_scheduler_factor=(
            float(scheduler["factor"])
            if scheduler["name"] == "reduce_on_plateau"
            else None
        ),
        lr_scheduler_patience=(
            int(scheduler["patience"])
            if scheduler["name"] == "reduce_on_plateau"
            else None
        ),
        lr_scheduler_min_lr=(
            float(scheduler["min_lr"])
            if scheduler["name"] == "reduce_on_plateau"
            else None
        ),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        lambda_base=float(optimizer["lambda_base"]),
        lambda_int=float(optimizer["lambda_int"]),
        max_epochs=int(training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        inputs_frozen=inputs.get("frozen", False),
        resources_frozen=resources.get("frozen", False),
        primary_epoch_unit=str(training["primary_epoch_unit"]),
        max_train_gene_cells_per_gene_per_epoch=int(
            training["max_train_gene_cells_per_gene_per_epoch"]
        ),
        resample_train_gene_cells_each_epoch=training[
            "resample_train_gene_cells_each_epoch"
        ],
        selected_gene_cell_ec_rows=str(training["selected_gene_cell_ec_rows"]),
        sampling_estimator=str(training["sampling_estimator"]),
        optimizer_step_unit=str(training["optimizer_step_unit"]),
        gene_microbatch_gradient_accumulation=training[
            "gene_microbatch_gradient_accumulation"
        ],
        batch_policy=str(resources["batch_policy"]),
        target_gpu_allocated_bytes=int(resources["target_gpu_allocated_bytes"]),
        max_cells_per_gpu_batch=int(resources["max_cells_per_gpu_batch"]),
        prefetch_backed_gene_shards=resources["prefetch_backed_gene_shards"],
        compute_precision=str(resources["compute_precision"]),
    )
    manifest.validate()
    return manifest


def assert_execution_admitted(
    config: Mapping[str, object],
) -> None:
    """Reject a full-cohort launch before filesystem, model, or GPU use."""

    execution = config.get("execution", {})
    if execution.get("scope") in {"toy", "fixture"}:
        if execution.get("training_authorized") is not True:
            raise RuntimeError("local implementation run is not authorized")
        return
    if execution.get("scope") != FULL_COHORT_SCOPE:
        raise RuntimeError("unknown execution scope")
    if execution.get("training_authorized") is not True:
        raise RuntimeError("full-cohort training is not authorized")
    if config.get("status") != "READY_TO_LAUNCH_FULL_COHORT_RUN":
        raise RuntimeError("full-cohort runtime is not launch-ready")
    resources = config.get("resources", {})
    if resources.get("frozen") is not True:
        raise RuntimeError("full-cohort resources are not frozen")
    if resources.get("profile_status") != "FROZEN_REAL_FULL_SHAPE_PROFILE":
        raise RuntimeError("real full-shape profile is not frozen")
    if resources.get("validation_status") != "FROZEN_STRICT_REAL_VALIDATION":
        raise RuntimeError("strict real-dataset validation is not frozen")
    profile_path = resources.get("profile_artifact")
    validation_path = resources.get("validation_artifact")
    if not isinstance(profile_path, str) or not Path(profile_path).is_file():
        raise RuntimeError("resource profile artifact is absent")
    if not isinstance(validation_path, str) or not Path(validation_path).is_file():
        raise RuntimeError("strict validation artifact is absent")
    profile = json.loads(Path(profile_path).read_text())
    validation = json.loads(Path(validation_path).read_text())
    training = config["training"]
    if (
        profile.get("schema_version") != "fabric.real_full_shape_profile.v3"
        or profile.get("status") != "FROZEN_REAL_FULL_SHAPE_PROFILE"
        or profile.get("scope") != FULL_COHORT_SCOPE
        or profile.get("profiled_condition") != "full"
        or profile.get("batch_policy") != resources.get("batch_policy")
        or profile.get("compute_precision") != resources.get("compute_precision")
        or profile.get("prefetch_backed_gene_shards")
        is not resources.get("prefetch_backed_gene_shards")
        or profile.get("target_gpu_allocated_bytes")
        != resources.get("target_gpu_allocated_bytes")
        or profile.get("unmodeled_gpu_reserve_bytes")
        != resources.get("unmodeled_gpu_reserve_bytes")
        or profile.get("max_cells_per_gpu_batch")
        != resources.get("max_cells_per_gpu_batch")
        or profile.get("train_bytes_per_shape_element")
        != resources.get("train_bytes_per_shape_element")
        or profile.get("evaluation_bytes_per_shape_element")
        != resources.get("evaluation_bytes_per_shape_element")
        or profile.get("train_sampling_unit") != training["primary_epoch_unit"]
        or profile.get("max_train_gene_cells_per_gene_per_epoch")
        != training["max_train_gene_cells_per_gene_per_epoch"]
        or profile.get("resample_train_gene_cells_each_epoch")
        is not training["resample_train_gene_cells_each_epoch"]
        or profile.get("selected_gene_cell_ec_rows")
        != training["selected_gene_cell_ec_rows"]
        or profile.get("sampling_estimator") != training["sampling_estimator"]
        or profile.get("optimizer_step_unit") != training["optimizer_step_unit"]
        or profile.get("gene_microbatch_gradient_accumulation")
        is not training["gene_microbatch_gradient_accumulation"]
        or profile.get("projected_optimizer_step_count_per_epoch")
        != config["inputs"]["g_fit_gene_count"]
        or profile.get("epoch_evaluation_policy")
        != "one_complete_validation_no_complete_train"
        or profile.get("epoch_core_metrics")
        != ["validation_compatible_path_nll", "ont_matrix_kl_count_weighted"]
        or profile.get("checkpoint_selection_metric")
        != "validation_compatible_path_nll"
        or profile.get("reporting_only_metric") != "ont_matrix_kl_count_weighted"
        or profile.get("ont_validation_target_root") != config["monitor"]["target_root"]
        or profile.get("adaptive_memory_estimate_upper_bounds_all_profiled_batches")
        is not True
        or any(
            profile.get(name) is not False
            for name in (
                "optimizer_constructed",
                "optimizer_step_called",
                "checkpoint_written",
                "test_rows_or_test_statistics_read",
                "test_predictions_or_metrics_computed",
            )
        )
    ):
        raise RuntimeError("resource profile violates the launch boundary")
    if (
        validation.get("status") != "ADMITTED_FOR_PRELAUNCH"
        or validation.get("g_fit_gene_count") != 17_600
        or validation.get("test_compatible_row_count") != 0
        or validation.get("final_test_authorized") is not False
    ):
        raise RuntimeError("strict validation is not launch-admitted")
    return


def _validate_prepared_dataset_identity(
    prepared: PreparedDataset | BackedPreparedDataset,
    config: Mapping[str, object],
) -> None:
    """Bind the loaded full-cohort bundle to the configured frozen identities."""

    inputs = config.get("inputs", {})
    expected_input = inputs.get("input_manifest_id")
    expected_compatibility = inputs.get("compatibility_artifact_id")
    full_cohort = config.get("execution", {}).get("scope") == FULL_COHORT_SCOPE
    if expected_input is not None and prepared.input_manifest_id != expected_input:
        raise RuntimeError("PreparedDataset input_manifest_id differs from config")
    if (
        expected_compatibility is not None
        and prepared.compatibility_artifact_id != expected_compatibility
    ):
        raise RuntimeError(
            "PreparedDataset compatibility_artifact_id differs from config"
        )
    if full_cohort:
        if prepared.informative_gene_ids is None:
            raise RuntimeError(
                "full-cohort PreparedDataset requires the frozen G_fit gene axis"
            )
        if len(prepared.informative_gene_ids) != 17_600:
            raise RuntimeError(
                "full-cohort PreparedDataset must contain exactly 17,600 G_fit genes"
            )
        observed_axis = (
            tuple(value[0] for value in prepared.genes.records)
            if isinstance(prepared, BackedPreparedDataset)
            else tuple(gene.gene_id for gene in prepared.genes)
        )
        if observed_axis != prepared.informative_gene_ids:
            raise RuntimeError(
                "full-cohort PreparedDataset gene order differs from frozen G_fit"
            )
        if isinstance(prepared, BackedPreparedDataset):
            if prepared.genes.expected_split_mass.get("train", 0) <= 0:
                raise RuntimeError(
                    "full-cohort backed dataset lacks conserved train informative mass"
                )
        else:
            for gene in prepared.genes:
                if split_informative_molecule_mass((gene,), "train") <= 0:
                    raise RuntimeError(
                        f"G_fit gene has no train informative mass: {gene.gene_id}"
                    )


def select_lambda_pair(
    validation_scores: Mapping[tuple[float, float], Mapping[str, float]],
    grid_order: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Select one shrinkage pair from dynamic-model aggregate validation NLL."""

    if not grid_order:
        raise ValueError("optimizer lambda grid is empty")
    ranked: list[tuple[float, float, float, int, tuple[float, float]]] = []
    for order, raw_pair in enumerate(grid_order):
        pair = (float(raw_pair[0]), float(raw_pair[1]))
        base, interaction = pair
        if not (interaction > base >= 0):
            raise ValueError(f"invalid lambda pair: {pair}")
        if pair not in validation_scores:
            raise ValueError(f"lambda grid score is absent: {pair}")
        condition_scores = validation_scores[pair]
        required = ("cis_dna", "cis_rna", "full")
        if set(condition_scores) != set(required):
            raise ValueError("lambda selection uses exactly CIS+DNA, CIS+RNA, and Full")
        values = [float(condition_scores[name]) for name in required]
        if not np.isfinite(values).all():
            raise ValueError("lambda validation score is non-finite")
        aggregate = float(np.mean(values))
        # Lower NLL, then larger interaction, larger base, then original order.
        ranked.append((aggregate, -interaction, -base, order, pair))
    return min(ranked)[-1]


def tune_optimizer_grid(
    genes: Sequence[PreparedGene],
    config: Mapping[str, object],
    *,
    device: str | torch.device,
    monitor_callback: EpochMonitor | None = None,
) -> OptimizerGridSelection:
    """Run the complete train/validation-only §15.5 shrinkage selection.

    For every predeclared pair and tuning seed this independently retrains the
    three dynamic conditions.  It never evaluates the held-out test and is not
    called by the single-run CLI; it is a separate optional analysis utility.
    """
    # Backed shards were strictly validated for readiness and are checked
    # again one at a time by BackedGeneSequence._load.  Avoid a redundant
    # 17,600-shard pre-scan before the first real epoch.
    if not isinstance(genes, BackedGeneSequence):
        _validate_genes(genes)
    optimizer_config = config.get("optimizer", {})
    _validate_optimizer_config(optimizer_config)
    conditions = tuple(optimizer_config.get("selection_conditions", ()))
    if conditions != ("cis_dna", "cis_rna", "full"):
        raise ValueError("optimizer tuning requires exactly the three dynamic models")
    grid = tuple(
        (float(pair[0]), float(pair[1])) for pair in optimizer_config.get("grid", ())
    )
    tuning_seeds = tuple(optimizer_config.get("tuning_seeds", ()))
    if not grid or not tuning_seeds:
        raise ValueError("optimizer tuning requires a nonempty grid and tuning seeds")
    if any(type(seed) is not int for seed in tuning_seeds) or len(
        set(tuning_seeds)
    ) != len(tuning_seeds):
        raise ValueError("optimizer tuning seeds must be unique integer IDs")
    # Reuse the selection validator for grid constraints and deterministic
    # ordering without accepting any unexecuted candidate.
    for pair in grid:
        if not (pair[1] > pair[0] >= 0) or not np.isfinite(pair).all():
            raise ValueError(f"invalid optimizer grid entry: {pair}")
    if len(set(grid)) != len(grid):
        raise ValueError("optimizer grid entries must be unique")

    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for tuning device {device}")
    runs: list[OptimizerTuningRun] = []
    scores: dict[tuple[float, float], dict[str, float]] = {}
    condition_means: list[tuple[float, float, float]] = []
    aggregate_scores: list[float] = []
    for lambda_base, lambda_int in grid:
        per_condition: dict[str, list[float]] = {
            condition: [] for condition in conditions
        }
        candidate_config = copy.deepcopy(dict(config))
        candidate_config["optimizer"]["lambda_base"] = lambda_base
        candidate_config["optimizer"]["lambda_int"] = lambda_int
        for seed in tuning_seeds:
            models = build_paired_models(
                genes[0], candidate_config["model"], seed=seed, device=device_object
            )
            for condition in conditions:
                result = _fit_condition(
                    genes,
                    models[condition],
                    forward_condition=condition,
                    condition_name=condition,
                    seed=seed,
                    config=candidate_config,
                    monitor_callback=monitor_callback,
                )
                value = float(result.best_validation_nll)
                if not np.isfinite(value):
                    raise FloatingPointError(
                        "optimizer tuning produced non-finite validation NLL"
                    )
                per_condition[condition].append(value)
                runs.append(
                    OptimizerTuningRun(
                        lambda_base=lambda_base,
                        lambda_int=lambda_int,
                        tuning_seed=seed,
                        condition=condition,
                        best_validation_nll=value,
                    )
                )
        means = tuple(
            float(np.mean(per_condition[condition])) for condition in conditions
        )
        condition_means.append(means)
        aggregate_scores.append(float(np.mean(means)))
        scores[(lambda_base, lambda_int)] = dict(zip(conditions, means))
    selected = select_lambda_pair(scores, grid)
    expected_runs = len(grid) * len(tuning_seeds) * len(conditions)
    if len(runs) != expected_runs:
        raise AssertionError("optimizer grid did not execute every declared fit")
    return OptimizerGridSelection(
        grid_order=grid,
        tuning_seeds=tuning_seeds,
        selection_conditions=conditions,
        runs=tuple(runs),
        condition_mean_validation_nll=tuple(condition_means),
        aggregate_validation_nll=tuple(aggregate_scores),
        selected_pair=selected,
    )


def optimizer_parameter_groups(
    model: FABRICV2Model, *, lambda_base: float, lambda_int: float
) -> list[dict[str, object]]:
    """Build the complete mutually-exclusive AdamW grouping from V2 section 15.5."""

    if not (lambda_int > lambda_base >= 0):
        raise ValueError("optimizer requires lambda_int > lambda_base >= 0")
    named = dict(model.named_parameters())
    interaction_names = {
        name
        for name in (
            "dna_aggregator.interaction_projection.weight",
            "rna_aggregator.interaction_projection.weight",
        )
        if name in named
    }
    norm_parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.modules.normalization.LayerNorm):
            norm_parameter_ids.update(
                id(value) for value in module.parameters(recurse=False)
            )
    no_decay_names = {
        name
        for name, parameter in named.items()
        if name.endswith("bias") or id(parameter) in norm_parameter_ids
    }
    base_names = set(named) - interaction_names - no_decay_names
    groups = []
    for group_name, names, decay in (
        ("no_decay", no_decay_names, 0.0),
        ("base", base_names, float(lambda_base)),
        ("interaction", interaction_names, float(lambda_int)),
    ):
        ordered = tuple(name for name in named if name in names)
        groups.append(
            {
                "params": [named[name] for name in ordered],
                "weight_decay": decay,
                "group_name": group_name,
                "parameter_names": ordered,
            }
        )
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {
        id(parameter) for parameter in named.values()
    }:
        raise AssertionError("optimizer parameter groups are not complete and disjoint")
    return groups


def build_optimizer(
    model: FABRICV2Model,
    *,
    learning_rate: float,
    lambda_base: float,
    lambda_int: float,
) -> torch.optim.AdamW:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.AdamW(
        optimizer_parameter_groups(
            model, lambda_base=lambda_base, lambda_int=lambda_int
        ),
        lr=float(learning_rate),
    )


def build_lr_scheduler(
    optimizer: torch.optim.AdamW,
    config: Mapping[str, object],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(config["name"])
    if name == "constant":
        return None
    if name != "reduce_on_plateau":
        raise ValueError("lr scheduler must be constant or reduce_on_plateau")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["factor"]),
        patience=int(config["patience"]),
        threshold=0.0,
        threshold_mode="abs",
        min_lr=float(config["min_lr"]),
    )


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(rates) != 1:
        raise RuntimeError(
            "FABRIC optimizer parameter groups have different learning rates"
        )
    return rates.pop()


def build_paired_models(
    example: PreparedGene,
    model_config: Mapping[str, object],
    *,
    seed: int,
    device: str | torch.device,
) -> dict[str, FABRICV2Model]:
    """Create paired primary initialization and the one upstream-matched comparator."""

    _seed_everything(seed)
    spec = _model_spec(example, model_config)
    reference = FABRICV2Model(**spec, readout_kind="path_context")
    reference_state = copy.deepcopy(reference.state_dict())
    models: dict[str, FABRICV2Model] = {}
    for condition in PRIMARY_ABLATIONS:
        model = FABRICV2Model(**spec, readout_kind="path_context")
        model.load_state_dict(reference_state)
        models[condition] = model.to(device)
    comparator = FABRICV2Model(**spec, readout_kind="additive_edge")
    comparator_state = comparator.state_dict()
    for name, value in reference_state.items():
        if not name.startswith("readout."):
            comparator_state[name] = value.clone()
    comparator.load_state_dict(comparator_state)
    models[ARCHITECTURE_COMPARATOR] = comparator.to(device)
    return models


def evaluate_final_test(
    genes: Sequence[PreparedGene],
    run: TrainingRunResult,
    config: Mapping[str, object],
    *,
    checkpoints_frozen: bool,
    report_rules_frozen: bool,
) -> pd.DataFrame:
    """Run one explicitly authorized held-out evaluation after checkpoint freeze."""

    if not checkpoints_frozen or not report_rules_frozen:
        raise RuntimeError(
            "final test requires frozen checkpoints and frozen report rules"
        )
    execution = config.get("execution", {})
    if execution.get("final_test_authorized") is not True:
        raise RuntimeError("held-out test inference is not authorized")
    condition = run.manifest.condition
    snapshot = _evaluate_split(
        genes,
        run.result.model,
        condition=_MODEL_CONDITION[condition],
        split="test",
        model_config=config["model"],
        resources=config["resources"],
    )
    return pd.DataFrame(
        [
            {
                "seed": run.manifest.seed,
                "condition": condition,
                "split": "test",
                "compatible_path_nll": snapshot.nll,
                "informative_molecule_mass": snapshot.informative_molecule_mass,
                "test_exposure": config.get("inputs", {}).get(
                    "test_exposure", "unspecified"
                ),
                "execution_scope": config["execution"]["scope"],
            }
        ]
    )


def train_run(
    data: Sequence[PreparedGene] | PreparedDataset | BackedPreparedDataset,
    config: Mapping[str, object],
    *,
    seed: int,
    condition: str,
    device: str | torch.device,
    run_dir: str | Path | None = None,
    monitor_callback: EpochMonitor | None = None,
    resume_from: str | Path | None = None,
) -> TrainingRunResult:
    """Train one seed/condition, optionally resuming after a completed epoch."""

    manifest = training_manifest_from_config(config, seed=seed, condition=condition)
    assert_execution_admitted(config)
    prepared = (
        data if isinstance(data, (PreparedDataset, BackedPreparedDataset)) else None
    )
    if prepared is None:
        genes = tuple(data)
    else:
        _validate_prepared_dataset_identity(prepared, config)
        genes = prepared.genes
    if config["monitor"]["enabled"] is True and monitor_callback is None:
        target = OntMatrixKlTarget.load(config["monitor"]["target_root"])
        monitor_callback = partial(validation_ont_matrix_kl_monitor, target=target)
    if monitor_callback is not None and not callable(monitor_callback):
        raise TypeError("per-epoch monitor callback must be callable")
    if config["monitor"]["enabled"] is False and monitor_callback is not None:
        raise RuntimeError("a disabled per-epoch monitor cannot receive a callback")
    if not isinstance(genes, BackedGeneSequence):
        _validate_genes(genes)
    torch_device = torch.device(device)
    torch.set_float32_matmul_precision("highest")
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for training device {device}")
    if (
        torch_device.type == "cuda"
        and config["execution"]["scope"] == FULL_COHORT_SCOPE
    ):
        resources = config["resources"]
        actual_total = int(torch.cuda.get_device_properties(torch_device).total_memory)
        if actual_total != int(resources["gpu_total_memory_bytes"]):
            raise RuntimeError(
                "runtime GPU total memory differs from the frozen profile"
            )
        free_bytes = int(torch.cuda.mem_get_info(torch_device)[0])
        if free_bytes < int(resources["target_gpu_allocated_bytes"]):
            raise RuntimeError(
                "runtime GPU is shared or lacks the frozen adaptive-batch target memory"
            )
    if resume_from is not None and run_dir is None:
        raise ValueError("resume_from requires the original run_dir")
    run_path = Path(run_dir) if run_dir is not None else None
    resume_path = Path(resume_from) if resume_from is not None else None
    if run_path is not None:
        if resume_path is None:
            run_path.mkdir(parents=True, exist_ok=False)
        elif not run_path.is_dir():
            raise FileNotFoundError(
                "resume requires the existing original run directory: "
                f"{run_path}"
            )

    lock = _exclusive_run_lock(run_path) if run_path is not None else nullcontext()
    with lock:
        _seed_everything(seed)
        model = FABRICV2Model(
            **_model_spec(genes[0], config["model"]), readout_kind="path_context"
        ).to(torch_device)
        recovery_identity = _training_recovery_identity(
            manifest=manifest,
            config=config,
            genes=genes,
            prepared=prepared,
        )
        resume_checkpoint: Mapping[str, object] | None = None
        if run_path is not None:
            if resume_path is None:
                _write_initial_run_identity(run_path, manifest, config)
            else:
                expected_latest = run_path / "latest.pt"
                if resume_path.resolve() != expected_latest.resolve():
                    raise ValueError(
                        "resume_from must be the original run_dir/latest.pt; "
                        "best checkpoints cannot continue an epoch history"
                    )
                _validate_stored_run_identity(run_path, manifest, config)
                resume_checkpoint = _load_training_recovery_checkpoint(
                    resume_path, expected_identity=recovery_identity
                )

        checkpoint_callback = (
            None
            if run_path is None
            else partial(
                _persist_epoch_recovery,
                run_dir=run_path,
                run_identity=recovery_identity,
            )
        )
        result = _fit_condition(
            genes,
            model,
            forward_condition=_MODEL_CONDITION[condition],
            condition_name=condition,
            seed=seed,
            config=config,
            monitor_callback=monitor_callback,
            resume_checkpoint=resume_checkpoint,
            epoch_checkpoint_callback=checkpoint_callback,
            resume_validated_callback=(
                None
                if run_path is None
                else partial(_reconcile_recovery_artifacts, run_path)
            ),
        )
        metric_rows = [
            {
                "seed": seed,
                "condition": condition,
                "split": "val",
                "validation_compatible_path_nll": result.best_validation_nll,
                "ont_matrix_kl_count_weighted": (
                    result.best_ont_matrix_kl_count_weighted
                ),
                "informative_molecule_mass": (
                    result.validation_informative_molecule_mass
                ),
                "execution_scope": config["execution"]["scope"],
            }
        ]
        run = TrainingRunResult(
            manifest=manifest,
            result=result,
            metrics=pd.DataFrame(metric_rows),
        )
        if run_path is not None:
            _write_run(
                run,
                config,
                run_path,
                genes,
                prepared,
                recovery_identity=recovery_identity,
            )
        return run


def sample_train_gene_cells_for_epoch(
    gene: PreparedGene,
    train_rows: torch.Tensor,
    *,
    max_gene_cells: int,
    seed: int,
    epoch: int,
    gene_order_0based: int,
    gene_count: int,
) -> TrainGeneCellSample:
    """Uniformly sample complete train gene-cell groups for one epoch.

    The arithmetic derived seed uses only the frozen seed, epoch, and G_fit
    order.  It is independent of shuffled traversal order and requires no hash
    identity.  All informative EC rows of every selected cell are retained.
    """

    if type(max_gene_cells) is not int or max_gene_cells < 1:
        raise ValueError("max_gene_cells must be a positive integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("sampling epoch must be a positive integer")
    if (
        type(gene_order_0based) is not int
        or type(gene_count) is not int
        or gene_count < 1
        or not 0 <= gene_order_0based < gene_count
    ):
        raise ValueError("gene sampling order is outside the frozen G_fit axis")
    if train_rows.ndim != 1 or train_rows.dtype != torch.long:
        raise TypeError("train_rows must be one-dimensional torch.long indices")
    row_cells = gene.row_cell_index[train_rows]
    available_cells = torch.unique(row_cells, sorted=True)
    available_count = int(available_cells.numel())
    if available_count < 1:
        raise ValueError("cannot sample a gene without informative train gene-cells")
    selected_count = min(available_count, max_gene_cells)
    if selected_count == available_count:
        selected_cells = available_cells
    else:
        derived_seed = seed + (epoch - 1) * gene_count + gene_order_0based
        positions = sorted(
            random.Random(derived_seed).sample(range(available_count), selected_count)
        )
        selected_cells = available_cells[positions]
    selected_rows = train_rows[torch.isin(row_cells, selected_cells)]
    if not torch.equal(
        torch.unique(gene.row_cell_index[selected_rows], sorted=True), selected_cells
    ):
        raise AssertionError("selected EC rows do not reproduce the sampled cell axis")
    return TrainGeneCellSample(
        selected_cells=selected_cells,
        selected_rows=selected_rows,
        available_cell_count=available_count,
        selected_cell_count=selected_count,
        inclusion_multiplier=available_count / selected_count,
    )


def _fit_condition(
    genes: Sequence[PreparedGene],
    model: FABRICV2Model,
    *,
    forward_condition: str,
    condition_name: str,
    seed: int,
    config: Mapping[str, object],
    monitor_callback: EpochMonitor | None,
    resume_checkpoint: Mapping[str, object] | None = None,
    epoch_checkpoint_callback: EpochCheckpoint | None = None,
    resume_validated_callback: EpochCheckpoint | None = None,
) -> ConditionResult:
    training = config["training"]
    optimizer_config = config["optimizer"]
    optimizer = build_optimizer(
        model,
        learning_rate=float(optimizer_config["learning_rate"]),
        lambda_base=float(optimizer_config["lambda_base"]),
        lambda_int=float(optimizer_config["lambda_int"]),
    )
    lr_scheduler = build_lr_scheduler(optimizer, optimizer_config["lr_scheduler"])
    gradient_clip_norm = float(optimizer_config["gradient_clip_norm"])
    max_epochs = int(training["max_epochs"])
    patience = int(training.get("early_stopping_patience", max_epochs))
    total_train_mass = split_informative_molecule_mass(genes, "train")
    if total_train_mass <= 0:
        raise ValueError("training split has zero likelihood-informative molecule mass")
    train_positive_gene_count = len(genes)
    validation_mass = split_informative_molecule_mass(genes, "val")
    if validation_mass <= 0:
        raise ValueError(
            "validation split has zero likelihood-informative molecule mass"
        )
    rng = random.Random(seed)
    history_rows: list[dict[str, object]] = []
    monitor_records: list[MonitorRecord] = []
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    best_lr_scheduler_state: dict[str, object] | None = None
    best_ont_matrix_kl: float | None = None
    epochs_without_improvement = 0
    completed_epoch = 0

    if resume_checkpoint is not None:
        restored = _restore_fit_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=seed,
            condition_name=condition_name,
            max_epochs=max_epochs,
            patience=patience,
            monitor_enabled=monitor_callback is not None,
        )
        completed_epoch = restored["completed_epoch"]
        history_rows = restored["history_rows"]
        monitor_records = restored["monitor_records"]
        best_nll = restored["best_nll"]
        best_epoch = restored["best_epoch"]
        best_state = restored["best_state"]
        best_optimizer_state = restored["best_optimizer_state"]
        best_lr_scheduler_state = restored["best_lr_scheduler_state"]
        best_ont_matrix_kl = restored["best_ont_matrix_kl"]
        epochs_without_improvement = restored["epochs_without_improvement"]
        rng.setstate(resume_checkpoint["gene_order_rng_state"])
        _restore_rng_state(resume_checkpoint["global_rng_state"])
        if resume_validated_callback is not None:
            resume_validated_callback(resume_checkpoint)

    for epoch in range(completed_epoch + 1, max_epochs + 1):
        if epochs_without_improvement >= patience:
            break
        epoch_learning_rate = _optimizer_learning_rate(optimizer)
        model.train()
        order = list(range(len(genes)))
        rng.shuffle(order)
        visited_instances: set[tuple[str, str]] = set()
        available_train_instances = 0
        sampled_train_instances = 0
        maximum_sampling_multiplier = 1.0
        train_cell_batch_count = 0
        maximum_train_batch_cells = 0
        maximum_train_batch_estimated_bytes = 0
        optimizer_step_count = 0
        for gene_index, gene in _iter_gene_order(
            genes,
            order,
            prefetch=bool(config["resources"]["prefetch_backed_gene_shards"]),
        ):
            rows = rows_for_split(gene, "train")
            if rows.numel() == 0:
                raise ValueError(
                    f"G_fit gene {gene.gene_id} has zero informative train rows"
                )
            optimizer.zero_grad(set_to_none=True)
            sample = sample_train_gene_cells_for_epoch(
                gene,
                rows,
                max_gene_cells=int(training["max_train_gene_cells_per_gene_per_epoch"]),
                seed=seed,
                epoch=epoch,
                gene_order_0based=gene_index,
                gene_count=len(genes),
            )
            cells = sample.selected_cells
            rows = sample.selected_rows
            inclusion_multiplier = sample.inclusion_multiplier
            available_train_instances += sample.available_cell_count
            sampled_train_instances += sample.selected_cell_count
            maximum_sampling_multiplier = max(
                maximum_sampling_multiplier, inclusion_multiplier
            )
            batch_plan = _plan_gene_cell_batches(
                gene,
                cells,
                rows,
                model_config=config["model"],
                resources=config["resources"],
                phase="train",
            )
            maximum_train_batch_estimated_bytes = max(
                maximum_train_batch_estimated_bytes,
                batch_plan.maximum_estimated_bytes,
            )
            for cell_batch in batch_plan.batches:
                cell_mask = torch.isin(gene.row_cell_index[rows], cell_batch)
                batch_rows = rows[cell_mask]
                for cell in cell_batch.tolist():
                    key = (gene.gene_id, gene.cell_ids[cell])
                    if key in visited_instances:
                        raise AssertionError(
                            "a train gene-cell instance was sampled twice"
                        )
                    visited_instances.add(key)
                train_cell_batch_count += 1
                maximum_train_batch_cells = max(
                    maximum_train_batch_cells, len(cell_batch)
                )
                batch_input, row_cell_index = _subset_gene_cells(
                    gene, cell_batch, batch_rows, model
                )
                details = compatible_path_nll(
                    model(batch_input, condition=forward_condition).path_logits,
                    gene.compatible_path_indices[batch_rows].to(_model_device(model)),
                    gene.compatible_path_mask[batch_rows].to(_model_device(model)),
                    gene.molecule_count[batch_rows].to(_model_device(model)),
                    row_cell_index=row_cell_index,
                    return_details=True,
                )
                (
                    details.weighted_sum
                    * train_positive_gene_count
                    * inclusion_multiplier
                    / total_train_mass
                ).backward()
            _assert_finite_gradients(
                model,
                require_all=False,
                require_any=True,
            )
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            optimizer.step()
            optimizer_step_count += 1
        if len(visited_instances) != sampled_train_instances:
            raise AssertionError(
                "visited gene-cell count differs from the sampled plan"
            )
        if optimizer_step_count != train_positive_gene_count:
            raise AssertionError(
                "optimizer step count differs from the train-positive gene count"
            )

        validation_snapshot = _evaluate_split(
            genes,
            model,
            condition=forward_condition,
            split="val",
            model_config=config["model"],
            resources=config["resources"],
        )
        val_nll = validation_snapshot.nll
        ont_matrix_kl: float | None = None
        # The callback consumes predictions from this same validation traversal.
        # It cannot enter checkpoint selection or trigger another model forward.
        if monitor_callback is not None:
            rng_state = _capture_rng_state()
            fields = dict(monitor_callback(condition_name, epoch, validation_snapshot))
            _restore_rng_state(rng_state)
            _validate_monitor_fields(config, fields, validation_nll=val_nll)
            ont_matrix_kl = float(fields["ont_matrix_kl_count_weighted"])
            monitor_records.append(
                MonitorRecord(
                    seed=seed,
                    condition=condition_name,
                    epoch=epoch,
                    fields=fields,
                )
            )
        if lr_scheduler is not None:
            lr_scheduler.step(val_nll)
        next_epoch_learning_rate = _optimizer_learning_rate(optimizer)
        best_improved = val_nll < best_nll
        if best_improved:
            best_nll = val_nll
            best_ont_matrix_kl = ont_matrix_kl
            best_epoch = epoch
            best_state = _cpu_copy(model.state_dict())
            best_optimizer_state = _cpu_copy(optimizer.state_dict())
            best_lr_scheduler_state = (
                _cpu_copy(lr_scheduler.state_dict())
                if lr_scheduler is not None
                else None
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history_rows.append(
            {
                "condition": condition_name,
                "epoch": epoch,
                "epoch_learning_rate": epoch_learning_rate,
                "next_epoch_learning_rate": next_epoch_learning_rate,
                "lr_scheduler": optimizer_config["lr_scheduler"]["name"],
                "gradient_clip_norm": gradient_clip_norm,
                "validation_compatible_path_nll": val_nll,
                "ont_matrix_kl_count_weighted": ont_matrix_kl,
                "epoch_train_denominator": total_train_mass,
                "validation_informative_molecule_mass": validation_mass,
                "train_sampling_unit": training["primary_epoch_unit"],
                "available_train_instances": available_train_instances,
                "sampled_train_instances": sampled_train_instances,
                "visited_train_instances": len(visited_instances),
                "maximum_sampling_multiplier": maximum_sampling_multiplier,
                "train_cell_batch_count": train_cell_batch_count,
                "maximum_train_batch_cells": maximum_train_batch_cells,
                "maximum_train_batch_estimated_bytes": (
                    maximum_train_batch_estimated_bytes
                ),
                "optimizer_step_unit": training["optimizer_step_unit"],
                "optimizer_steps": optimizer_step_count,
                "train_positive_gene_count": train_positive_gene_count,
                "gene_microbatch_gradient_accumulation": training[
                    "gene_microbatch_gradient_accumulation"
                ],
                "uniform_gene_step_objective_multiplier": (train_positive_gene_count),
            }
        )

        if epoch_checkpoint_callback is not None:
            if best_state is None or best_optimizer_state is None:
                raise RuntimeError("completed epoch has no selected checkpoint")
            epoch_checkpoint_callback(
                {
                    "completed_epoch": epoch,
                    "model_state_dict": _cpu_copy(model.state_dict()),
                    "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                    "lr_scheduler_state_dict": (
                        _cpu_copy(lr_scheduler.state_dict())
                        if lr_scheduler is not None
                        else None
                    ),
                    "best_epoch": best_epoch,
                    "best_validation_nll": best_nll,
                    "best_ont_matrix_kl_count_weighted": best_ont_matrix_kl,
                    "best_model_state_dict": best_state,
                    "best_optimizer_state_dict": best_optimizer_state,
                    "best_lr_scheduler_state_dict": best_lr_scheduler_state,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history_rows": copy.deepcopy(history_rows),
                    "monitor_records": [
                        asdict(record) for record in monitor_records
                    ],
                    "gene_order_rng_state": rng.getstate(),
                    "global_rng_state": _capture_rng_state(),
                    "training_complete": (
                        epoch >= max_epochs
                        or epochs_without_improvement >= patience
                    ),
                    "best_improved_this_epoch": best_improved,
                    "held_out_test_evaluated": False,
                }
            )

        if epochs_without_improvement >= patience:
            break
    if best_state is None or best_optimizer_state is None:
        raise RuntimeError(f"condition {condition_name} produced no checkpoint")
    model.load_state_dict(best_state)
    return ConditionResult(
        model=model,
        history=pd.DataFrame(history_rows),
        monitor_records=tuple(monitor_records),
        best_epoch=best_epoch,
        best_validation_nll=best_nll,
        best_ont_matrix_kl_count_weighted=best_ont_matrix_kl,
        validation_informative_molecule_mass=validation_mass,
        optimizer_state_dict=best_optimizer_state,
        lr_scheduler_state_dict=best_lr_scheduler_state,
    )


def _validate_monitor_fields(
    config: Mapping[str, object],
    fields: Mapping[str, object],
    *,
    validation_nll: float,
) -> None:
    if fields.get("sealed") is not True:
        raise RuntimeError("per-epoch monitor record must remain sealed")
    if fields.get("selection_eligible") is not False:
        raise RuntimeError("per-epoch monitor fields cannot be selection eligible")
    if fields.get("metric_schema") != "fabric_v2_epoch_core_metrics_v1":
        raise RuntimeError("per-epoch monitor metric schema differs")
    if not np.isclose(
        float(fields.get("validation_compatible_path_nll", np.nan)),
        validation_nll,
        atol=0,
        rtol=0,
    ):
        raise RuntimeError(
            "monitor validation NLL differs from checkpoint selection NLL"
        )
    ont_kl = fields.get("ont_matrix_kl_count_weighted")
    if (
        isinstance(ont_kl, bool)
        or not isinstance(ont_kl, (int, float))
        or not np.isfinite(ont_kl)
        or ont_kl < 0
    ):
        raise RuntimeError("ONT matrix KL must be finite and non-negative")


def _evaluate_split(
    genes: Sequence[PreparedGene],
    model: FABRICV2Model,
    *,
    condition: str,
    split: str,
    model_config: Mapping[str, object],
    resources: Mapping[str, object],
) -> ValidationSnapshot:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    model.eval()
    total_weighted = 0.0
    total_mass = 0.0
    predictions: list[ValidationPrediction] = []
    with torch.no_grad():
        for _, gene in _iter_gene_order(
            genes,
            range(len(genes)),
            prefetch=bool(resources["prefetch_backed_gene_shards"]),
        ):
            rows = rows_for_split(gene, split)
            if rows.numel() == 0:
                continue
            cells = torch.unique(gene.row_cell_index[rows], sorted=True)
            batch_plan = _plan_gene_cell_batches(
                gene,
                cells,
                rows,
                model_config=model_config,
                resources=resources,
                phase="evaluation",
            )
            path_logits_parts: list[torch.Tensor] = []
            for cell_batch in batch_plan.batches:
                cell_mask = torch.isin(gene.row_cell_index[rows], cell_batch)
                batch_rows = rows[cell_mask]
                batch_input, row_cell_index = _subset_gene_cells(
                    gene, cell_batch, batch_rows, model
                )
                output = model(batch_input, condition=condition)
                details = compatible_path_nll(
                    output.path_logits,
                    gene.compatible_path_indices[batch_rows].to(_model_device(model)),
                    gene.compatible_path_mask[batch_rows].to(_model_device(model)),
                    gene.molecule_count[batch_rows].to(_model_device(model)),
                    row_cell_index=row_cell_index,
                    return_details=True,
                )
                total_weighted += float(details.weighted_sum)
                total_mass += float(details.molecule_mass)
                path_logits_parts.append(output.path_logits.detach().cpu())
            path_logits = torch.cat(path_logits_parts, dim=0)
            lookup = torch.full((len(gene.cell_ids),), -1, dtype=torch.long)
            lookup[cells] = torch.arange(cells.numel(), dtype=torch.long)
            combined_row_cell_index = lookup[gene.row_cell_index[rows].cpu()]
            if bool((combined_row_cell_index < 0).any()):
                raise AssertionError("evaluation EC row was not assigned to its cell")
            predictions.append(
                ValidationPrediction(
                    gene_id=gene.gene_id,
                    cell_ids=tuple(gene.cell_ids[index] for index in cells.tolist()),
                    path_ids=gene.path_ids,
                    path_logits=path_logits,
                    compatible_path_indices=gene.compatible_path_indices[rows].cpu(),
                    compatible_path_mask=gene.compatible_path_mask[rows].cpu(),
                    row_cell_index=combined_row_cell_index,
                    molecule_count=gene.molecule_count[rows].cpu(),
                )
            )
    if total_mass <= 0:
        raise ValueError(f"split {split} has zero likelihood-informative molecule mass")
    return ValidationSnapshot(
        split=split,
        weighted_nll_sum=total_weighted,
        informative_molecule_mass=total_mass,
        predictions=tuple(predictions),
    )


def _iter_gene_order(
    genes: Sequence[PreparedGene],
    order: Sequence[int] | range,
    *,
    prefetch: bool,
):
    """Yield genes in exact order while overlapping one backed-shard load.

    Only immutable CPU shard deserialization is moved to the worker.  Model
    execution, sampling, RNG use, gradient accumulation, and optimizer updates
    remain on the caller thread in the original order.
    """

    positions = iter(order)
    try:
        first_index = next(positions)
    except StopIteration:
        return
    if not prefetch or not isinstance(genes, BackedGeneSequence):
        yield first_index, genes[first_index]
        for index in positions:
            yield index, genes[index]
        return
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="fabric-gene-load"
    ) as pool:
        current_index = first_index
        current = pool.submit(genes.__getitem__, current_index)
        for next_index in positions:
            gene = current.result()
            following = pool.submit(genes.__getitem__, next_index)
            yield current_index, gene
            current_index = next_index
            current = following
        yield current_index, current.result()


def _gene_shape_components(
    *,
    edge_count: int,
    path_count: int,
    route_count: int,
    compatible_width: int,
    cis_dim: int,
    dynamic_dim: int,
    hidden_dim: int,
    path_hidden_dim: int,
) -> tuple[int, int, int]:
    """Return static, per-cell, and per-compatible-row shape elements.

    MultiheadAttention is called with ``need_weights=False`` and therefore does
    not retain an explicit ``B x heads x E x E`` probability tensor.  The live
    tensors that drive memory are instead route contributions, edge states,
    path readout states, and compatible-row indexing.  The phase-specific byte
    multipliers in the frozen resource policy cover autograd saved tensors,
    gradients, kernel workspaces, dtypes, and allocator overhead.
    """

    positive_counts = (
        edge_count,
        path_count,
        compatible_width,
        cis_dim,
        dynamic_dim,
        hidden_dim,
        path_hidden_dim,
    )
    if any(type(value) is not int or value < 1 for value in positive_counts):
        raise ValueError("gene-shape batching dimensions must be positive integers")
    if type(route_count) is not int or route_count < 0:
        raise ValueError("gene-shape route count must be a non-negative integer")
    static = (
        route_count * dynamic_dim
        + edge_count * (cis_dim + 2 * dynamic_dim + 2 * hidden_dim)
        + path_count * (hidden_dim + path_hidden_dim + 4)
    )
    per_cell = (
        route_count * (dynamic_dim + 2)
        + edge_count * (cis_dim + 4 * dynamic_dim + 10 * hidden_dim)
        + path_count * (3 * hidden_dim + path_hidden_dim + 4)
    )
    per_compatible_row = 2 * compatible_width + 8
    return static, per_cell, per_compatible_row


def _estimated_gene_batch_bytes(
    *,
    static_shape_elements: int,
    per_cell_shape_elements: int,
    per_compatible_row_shape_elements: int,
    cell_count: int,
    compatible_row_count: int,
    resources: Mapping[str, object],
    phase: str,
) -> int:
    if phase not in {"train", "evaluation"}:
        raise ValueError("adaptive batching phase must be train or evaluation")
    if type(cell_count) is not int or cell_count < 1:
        raise ValueError("adaptive batching requires at least one cell")
    if type(compatible_row_count) is not int or compatible_row_count < 1:
        raise ValueError("adaptive batching requires informative compatible rows")
    multiplier = float(resources[f"{phase}_bytes_per_shape_element"])
    modeled = multiplier * (
        static_shape_elements
        + cell_count * per_cell_shape_elements
        + compatible_row_count * per_compatible_row_shape_elements
    )
    return int(np.ceil(modeled)) + int(resources["unmodeled_gpu_reserve_bytes"])


def _plan_gene_cell_batches(
    gene: PreparedGene,
    cells: torch.Tensor,
    rows: torch.Tensor,
    *,
    model_config: Mapping[str, object],
    resources: Mapping[str, object],
    phase: str,
) -> GeneBatchPlan:
    """Pack complete gene-cell groups against a deterministic GPU target.

    The plan depends only on frozen tensor shapes and resource coefficients,
    never on transient free memory or an OOM retry.  This keeps the numerical
    execution identity reproducible while allowing large and small genes to
    use different cell microbatch sizes.
    """

    if cells.ndim != 1 or cells.dtype != torch.long or cells.numel() < 1:
        raise TypeError(
            "adaptive batching cells must be non-empty one-dimensional long"
        )
    if rows.ndim != 1 or rows.dtype != torch.long or rows.numel() < 1:
        raise TypeError("adaptive batching rows must be non-empty one-dimensional long")
    cell_values = cells.detach().cpu()
    if torch.unique(cell_values).numel() != cell_values.numel():
        raise ValueError("adaptive batching cells must be unique")
    row_cells = gene.row_cell_index[rows].detach().cpu()
    selected = torch.zeros(len(gene.cell_ids), dtype=torch.bool)
    selected[cell_values] = True
    if not bool(selected[row_cells].all()):
        raise ValueError("compatible rows escape the adaptive batching cell set")
    row_counts = torch.bincount(row_cells, minlength=len(gene.cell_ids))
    if bool((row_counts[cell_values] < 1).any()):
        raise ValueError("every adaptively batched cell must retain a compatible row")

    route_count = int(
        gene.model_input.dna.route_edge_index.numel()
        + gene.model_input.rna.route_edge_index.numel()
    )
    static, per_cell, per_row = _gene_shape_components(
        edge_count=int(gene.model_input.cis_features.shape[0]),
        path_count=len(gene.path_ids),
        route_count=route_count,
        compatible_width=int(gene.compatible_path_indices.shape[1]),
        cis_dim=int(gene.model_input.cis_features.shape[1]),
        dynamic_dim=int(model_config["dynamic_projection_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        path_hidden_dim=int(model_config["path_hidden_dim"]),
    )
    target = int(resources["target_gpu_allocated_bytes"])
    kernel_cell_limit = int(resources["max_cells_per_gpu_batch"])
    multiplier = float(resources[f"{phase}_bytes_per_shape_element"])
    capacity = int(
        np.floor(
            (target - int(resources["unmodeled_gpu_reserve_bytes"])) / multiplier
            - static
        )
    )
    if capacity < 1:
        raise RuntimeError(
            f"gene {gene.gene_id} static {phase} shape exceeds the frozen GPU target"
        )
    cell_costs = (
        torch.full(
            (len(cell_values),),
            per_cell,
            dtype=torch.long,
        )
        + row_counts[cell_values].to(torch.long) * per_row
    )
    cumulative_costs = torch.cumsum(cell_costs, dim=0)
    batches: list[torch.Tensor] = []
    estimated_bytes: list[int] = []
    start = 0
    while start < len(cell_values):
        consumed = int(cumulative_costs[start - 1]) if start else 0
        end = int(
            torch.searchsorted(
                cumulative_costs,
                torch.tensor(consumed + capacity, dtype=torch.long),
                right=True,
            )
        )
        end = min(end, start + kernel_cell_limit)
        if end == start:
            raise RuntimeError(
                f"gene {gene.gene_id} one-cell {phase} shape exceeds the frozen GPU target"
            )
        batch = cell_values[start:end]
        batch_rows = int(row_counts[batch].sum())
        batches.append(batch)
        estimated_bytes.append(
            _estimated_gene_batch_bytes(
                static_shape_elements=static,
                per_cell_shape_elements=per_cell,
                per_compatible_row_shape_elements=per_row,
                cell_count=len(batch),
                compatible_row_count=batch_rows,
                resources=resources,
                phase=phase,
            )
        )
        start = end
    if sum(len(batch) for batch in batches) != len(cell_values):
        raise AssertionError("adaptive gene batches do not conserve the cell axis")
    if max(estimated_bytes) > target:
        raise AssertionError("adaptive gene batch exceeds the frozen GPU target")
    return GeneBatchPlan(
        batches=tuple(batches),
        estimated_bytes=tuple(estimated_bytes),
        per_cell_shape_elements=per_cell,
        per_compatible_row_shape_elements=per_row,
    )


def rows_for_split(gene: PreparedGene, split: str) -> torch.Tensor:
    cell_split = np.asarray(gene.cell_split, dtype=object)
    row_cells = gene.row_cell_index.detach().cpu().numpy()
    split_mask = torch.from_numpy(cell_split[row_cells] == split)
    mask = split_mask & gene.informative_row_mask.cpu()
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def split_informative_molecule_mass(genes: Sequence[PreparedGene], split: str) -> float:
    if isinstance(genes, BackedGeneSequence) and split in genes.expected_split_mass:
        return float(genes.expected_split_mass[split])
    return float(
        sum(
            gene.molecule_count[rows_for_split(gene, split)].double().sum().item()
            for gene in genes
        )
    )


def make_toy_genes(*, seed: int = 7) -> list[PreparedGene]:
    """Build a deterministic V2 toy with ambiguous and full-path audit rows."""

    generator = torch.Generator().manual_seed(seed)
    edge_count = 7
    cis_dim = 6
    cell_count = 12
    path_rows = ((0, 1, 2, 3, 6), (0, 1, 4, 5, 6))
    incidence = torch.zeros((2, edge_count), dtype=torch.float32)
    for path_index, edges in enumerate(path_rows):
        incidence[path_index, list(edges)] = 1
    local_pairs = {
        pair
        for path in path_rows
        for left, right in zip(path[:-1], path[1:])
        for pair in ((left, right), (right, left))
    }
    local = torch.tensor(sorted(local_pairs), dtype=torch.long).T.contiguous()
    cis = torch.randn(edge_count, cis_dim, generator=generator)
    dna = _toy_modality(
        cell_count=cell_count,
        edge_indices=(2, 4, 3, 5),
        base_dim=5,
        interaction_dim=3,
        generator=generator,
        gate_shift=0.8,
    )
    rna = _toy_modality(
        cell_count=cell_count,
        edge_indices=(2, 4, 3, 5),
        base_dim=4,
        interaction_dim=2,
        generator=generator,
        gate_shift=-0.5,
    )
    model_input = GeneCellModelInput(
        cis_features=cis,
        local_edge_index=local,
        dna=dna,
        rna=rna,
        path_edge_incidence=incidence.to_sparse_coo().coalesce(),
        path_first_edge_index=torch.tensor([0, 0]),
        path_last_edge_index=torch.tensor([6, 6]),
        log_edge_count=torch.log1p(torch.tensor([5.0, 5.0])),
    )

    # Two informative rows per cell share one model forward; one all-path row
    # per cell remains present for the audit but is excluded from K^inf.
    row_count = 3 * cell_count
    compatible = torch.full((row_count, 2), -1, dtype=torch.long)
    compatible_mask = torch.zeros_like(compatible, dtype=torch.bool)
    molecule = torch.empty(row_count, dtype=torch.float32)
    row_cell = torch.empty(row_count, dtype=torch.long)
    informative = torch.zeros(row_count, dtype=torch.bool)
    for cell in range(cell_count):
        start = 3 * cell
        compatible[start, 0] = 0
        compatible[start + 1, 0] = 1
        compatible[start + 2] = torch.tensor([0, 1])
        compatible_mask[start : start + 2, 0] = True
        compatible_mask[start + 2] = True
        signal = dna.gate[cell, 0] + 0.6 * rna.gate[cell, 0]
        probability = float(torch.sigmoid(signal))
        molecule[start] = 1.0 + round(5.0 * probability)
        molecule[start + 1] = 1.0 + round(5.0 * (1.0 - probability))
        molecule[start + 2] = 50.0  # must never enter the NLL denominator
        row_cell[start : start + 3] = cell
        informative[start : start + 2] = True
    split = ("train",) * 6 + ("val",) * 3 + ("test",) * 3
    return [
        PreparedGene(
            gene_id="TOY_GENE",
            model_input=model_input,
            compatible_path_indices=compatible,
            compatible_path_mask=compatible_mask,
            row_cell_index=row_cell,
            molecule_count=molecule,
            informative_row_mask=informative,
            cell_ids=tuple(f"cell_{index:02d}" for index in range(cell_count)),
            cell_split=split,
            path_ids=("path_a", "path_b"),
        )
    ]


def _toy_modality(
    *,
    cell_count: int,
    edge_indices: tuple[int, ...],
    base_dim: int,
    interaction_dim: int,
    generator: torch.Generator,
    gate_shift: float,
) -> RoutedModalityInput:
    event_count = 2
    route_count = len(edge_indices)
    route_event = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    route_weight = torch.full((route_count,), 0.5)
    base = torch.randn(route_count, base_dim, generator=generator)
    interaction = torch.randn(route_count, interaction_dim, generator=generator)
    raw = torch.randn(cell_count, event_count, generator=generator)
    gate = raw - raw[:6].mean(dim=0, keepdim=True)
    gate[:, 0] += gate_shift
    return RoutedModalityInput(
        route_event_index=route_event,
        route_edge_index=torch.tensor(edge_indices, dtype=torch.long),
        route_weight=route_weight,
        route_base_features=base,
        route_interaction_features=interaction,
        interaction_active_mask=torch.ones(interaction_dim, dtype=torch.bool),
        event_gate_key_index=torch.tensor([0, 1]),
        gate=gate,
    )


def _model_spec(example: PreparedGene, config: Mapping[str, object]) -> dict[str, int]:
    return {
        "cis_dim": int(example.model_input.cis_features.shape[1]),
        "dna_base_dim": int(example.model_input.dna.route_base_features.shape[1]),
        "dna_interaction_dim": int(
            example.model_input.dna.route_interaction_features.shape[1]
        ),
        "rna_base_dim": int(example.model_input.rna.route_base_features.shape[1]),
        "rna_interaction_dim": int(
            example.model_input.rna.route_interaction_features.shape[1]
        ),
        "dynamic_dim": int(config["dynamic_projection_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "attention_heads": int(config["attention_heads"]),
        "path_hidden_dim": int(config["path_hidden_dim"]),
    }


def _subset_gene_cells(
    gene: PreparedGene,
    cells: torch.Tensor,
    rows: torch.Tensor,
    model: FABRICV2Model,
) -> tuple[GeneCellModelInput, torch.Tensor]:
    device = _model_device(model)
    cells = cells.detach().cpu().long()
    lookup = torch.full((len(gene.cell_ids),), -1, dtype=torch.long)
    lookup[cells] = torch.arange(cells.numel(), dtype=torch.long)
    remapped = lookup[gene.row_cell_index[rows].cpu()]
    if bool((remapped < 0).any()):
        raise AssertionError("an EC row was not assigned to its selected cell")
    source = gene.model_input
    subset = GeneCellModelInput(
        cis_features=source.cis_features.to(device),
        local_edge_index=source.local_edge_index.to(device),
        dna=_routed_to_device(source.dna, device, cells),
        rna=_routed_to_device(source.rna, device, cells),
        path_edge_incidence=source.path_edge_incidence.to(device),
        path_first_edge_index=source.path_first_edge_index.to(device),
        path_last_edge_index=source.path_last_edge_index.to(device),
        log_edge_count=source.log_edge_count.to(device),
    )
    return subset, remapped.to(device)


def _routed_to_device(
    routed: RoutedModalityInput, device: torch.device, cells: torch.Tensor
) -> RoutedModalityInput:
    return RoutedModalityInput(
        route_event_index=routed.route_event_index.to(device),
        route_edge_index=routed.route_edge_index.to(device),
        route_weight=routed.route_weight.to(device),
        route_base_features=routed.route_base_features.to(device),
        route_interaction_features=routed.route_interaction_features.to(device),
        interaction_active_mask=routed.interaction_active_mask.to(device),
        event_gate_key_index=routed.event_gate_key_index.to(device),
        gate=routed.gate[cells].to(device),
        event_keep_mask=(
            None
            if routed.event_keep_mask is None
            else routed.event_keep_mask.to(device)
        ),
        route_keep_mask=(
            None
            if routed.route_keep_mask is None
            else routed.route_keep_mask.to(device)
        ),
    )


def _validate_genes(genes: Sequence[PreparedGene]) -> None:
    if not genes:
        raise ValueError("training requires at least one prepared gene")
    gene_ids = [gene.gene_id for gene in genes]
    if any(not isinstance(value, str) or not value for value in gene_ids):
        raise TypeError("prepared gene IDs must be non-empty strings")
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("prepared gene IDs must be unique")
    first_dims = _input_dimensions(genes[0])
    for gene in genes:
        if _input_dimensions(gene) != first_dims:
            raise ValueError("prepared genes do not share frozen model feature axes")
        row_count = gene.compatible_path_indices.shape[0]
        if (
            gene.compatible_path_mask.shape != gene.compatible_path_indices.shape
            or gene.row_cell_index.shape != (row_count,)
            or gene.molecule_count.shape != (row_count,)
            or gene.informative_row_mask.shape != (row_count,)
        ):
            raise ValueError(f"gene {gene.gene_id} EC tensor axes differ")
        if gene.compatible_path_indices.dtype != torch.long:
            raise TypeError("compatible path indices must use torch.long")
        if (
            gene.compatible_path_mask.dtype != torch.bool
            or gene.informative_row_mask.dtype != torch.bool
        ):
            raise TypeError("compatible and informative masks must use bool dtype")
        cell_count = gene.model_input.dna.gate.shape[0]
        if (
            len(gene.cell_ids) != cell_count
            or len(set(gene.cell_ids)) != cell_count
            or any(not isinstance(value, str) or not value for value in gene.cell_ids)
            or len(gene.cell_split) != cell_count
            or set(gene.cell_split) - set(SPLITS)
        ):
            raise ValueError(f"gene {gene.gene_id} cell identity/split axis differs")
        if row_count and bool(
            ((gene.row_cell_index < 0) | (gene.row_cell_index >= cell_count)).any()
        ):
            raise IndexError("EC row references an unknown cell")
        if not torch.isfinite(gene.molecule_count).all() or bool(
            (gene.molecule_count <= 0).any()
        ):
            raise ValueError("EC molecule counts must be finite and positive")
        if not torch.equal(gene.molecule_count, gene.molecule_count.round()):
            raise ValueError("EC molecule counts must be integer molecule mass")
        path_count = len(gene.path_ids)
        incidence_path_count = gene.model_input.path_edge_incidence.shape[0]
        if (
            path_count < 1
            or path_count != incidence_path_count
            or len(set(gene.path_ids)) != path_count
            or any(not isinstance(value, str) or not value for value in gene.path_ids)
        ):
            raise ValueError(f"gene {gene.gene_id} path identity axis differs")
        # The real artifact has roughly 120 million EC rows.  Validate its
        # ordered padded representation as tensor operations rather than a
        # Python loop over rows; this is the same identity check exercised by
        # the small fixtures, but remains practical at full cohort shape.
        mask = gene.compatible_path_mask
        indices = gene.compatible_path_indices
        if mask.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("compatible-path mask is not a left-aligned prefix")
        active_indices = indices[mask]
        if active_indices.numel() and bool(
            ((active_indices < 0) | (active_indices >= path_count)).any()
        ):
            raise IndexError("compatible-path row references an unknown path")
        if mask.shape[1] > 1:
            adjacent = mask[:, 1:] & mask[:, :-1]
            differences = indices[:, 1:] - indices[:, :-1]
            if bool((adjacent & differences.eq(0)).any()):
                raise ValueError(
                    "compatible-path row contains duplicate path identities"
                )
            if bool((adjacent & differences.lt(0)).any()):
                raise ValueError("compatible-path row is not in frozen path order")
        observed_informative = (
            (gene.molecule_count > 0)
            & (gene.compatible_path_mask.sum(dim=1) > 0)
            & (gene.compatible_path_mask.sum(dim=1) < path_count)
        )
        if not torch.equal(observed_informative, gene.informative_row_mask):
            raise ValueError(
                "informative_row_mask differs from the frozen K^inf definition"
            )


def _input_dimensions(gene: PreparedGene) -> tuple[int, ...]:
    value = gene.model_input
    return (
        value.cis_features.shape[1],
        value.dna.route_base_features.shape[1],
        value.dna.route_interaction_features.shape[1],
        value.rna.route_base_features.shape[1],
        value.rna.route_interaction_features.shape[1],
    )


def _assert_finite_gradients(
    model: FABRICV2Model,
    *,
    require_all: bool,
    require_any: bool = False,
) -> None:
    gradient_seen = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            if require_all:
                raise RuntimeError(f"Full model parameter has no gradient: {name}")
            continue
        gradient_seen = True
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"model parameter has a non-finite gradient: {name}"
            )
    if require_any and not gradient_seen:
        raise RuntimeError("gene optimizer step has no parameter gradient")


def validation_ont_matrix_kl_monitor(
    condition: str,
    epoch: int,
    snapshot: ValidationSnapshot,
    *,
    target: OntMatrixKlTarget,
) -> Mapping[str, float | int | str | bool]:
    """Return the two epoch core values from one validation traversal."""

    result = compute_validation_ont_matrix_kl(snapshot, target)
    return {
        "metric_schema": "fabric_v2_epoch_core_metrics_v1",
        "condition": condition,
        "epoch": epoch,
        "validation_compatible_path_nll": snapshot.nll,
        "ont_matrix_kl_count_weighted": (result.ont_matrix_kl_count_weighted),
        "validation_informative_molecule_mass": snapshot.informative_molecule_mass,
        "ont_eligible_cell_gene_count": result.eligible_cell_gene_count,
        "ont_eligible_count_denominator": result.eligible_ont_count,
        "ont_zero_total_cell_gene_count": result.zero_total_cell_gene_count,
        "ont_fewer_than_two_positive_paths_cell_gene_count": (
            result.fewer_than_two_positive_paths_cell_gene_count
        ),
        "same_validation_prediction_traversal": True,
        "comparison_name": "same_library_cross_pipeline_ont_matrix_agreement",
        "matrix_identity": target.matrix_identity,
        "path_identity": target.path_identity,
        "split_identity": target.split_identity,
        "sealed": True,
        "selection_eligible": False,
    }


def _capture_rng_state() -> (
    tuple[object, tuple, torch.Tensor, list[torch.Tensor] | None]
):
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda


def _restore_rng_state(
    state: tuple[object, tuple, torch.Tensor, list[torch.Tensor] | None]
) -> None:
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _model_device(model: FABRICV2Model) -> torch.device:
    return next(model.parameters()).device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_copy(value: object) -> object:
    """Clone checkpoint state onto CPU without retaining extra GPU allocations."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


@contextmanager
def _exclusive_run_lock(run_dir: Path):
    """Prevent two trainers from mutating the same recovery history."""

    lock_path = run_dir / ".training.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another training process holds the run directory: {run_dir}"
            ) from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _training_recovery_identity(
    *,
    manifest: TrainingRunManifest,
    config: Mapping[str, object],
    genes: Sequence[PreparedGene],
    prepared: PreparedDataset | BackedPreparedDataset | None,
) -> dict[str, object]:
    if isinstance(genes, BackedGeneSequence):
        ordered_gene_ids = tuple(gene_id for gene_id, _ in genes.records)
        dataset_kind = "backed_prepared_dataset"
    else:
        ordered_gene_ids = tuple(gene.gene_id for gene in genes)
        dataset_kind = "prepared_dataset" if prepared is not None else "sequence"
    source_commit = _runtime_source_commit(
        require_clean=config["execution"]["scope"] == FULL_COHORT_SCOPE
    )
    if (
        config["execution"]["scope"] == FULL_COHORT_SCOPE
        and isinstance(prepared, BackedPreparedDataset)
        and prepared.source_git_commit != source_commit
    ):
        raise RuntimeError(
            "full-cohort prepared artifact source commit differs from the training source"
        )
    return {
        "training_run_manifest": asdict(manifest),
        "resolved_config": copy.deepcopy(dict(config)),
        "dataset_kind": dataset_kind,
        "input_manifest_id": (
            prepared.input_manifest_id if prepared is not None else None
        ),
        "compatibility_artifact_id": (
            prepared.compatibility_artifact_id if prepared is not None else None
        ),
        "source_git_commit": source_commit,
        "ordered_gene_ids": ordered_gene_ids,
        "model_spec": _model_spec(genes[0], config["model"]),
        "readout_kind": "path_context",
        "test_model_predictions_status": "NOT_COMPUTED_DURING_TRAINING",
    }


def _runtime_source_commit(*, require_clean: bool) -> str:
    return committed_source_identity(require_clean=require_clean)


def _write_initial_run_identity(
    run_dir: Path,
    manifest: TrainingRunManifest,
    config: Mapping[str, object],
) -> None:
    _atomic_write_text(
        run_dir / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False)
    )
    _atomic_write_text(
        run_dir / "training_run_manifest.json",
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
    )


def _validate_stored_run_identity(
    run_dir: Path,
    manifest: TrainingRunManifest,
    config: Mapping[str, object],
) -> None:
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "training_run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "resume run directory lacks config.yaml or training_run_manifest.json"
        )
    stored_config = yaml.safe_load(config_path.read_text())
    if stored_config != dict(config):
        raise ValueError("resume resolved config differs from the original run")
    stored_manifest = json.loads(manifest_path.read_text())
    if stored_manifest != asdict(manifest):
        raise ValueError("resume training manifest differs from the original run")


def _load_training_recovery_checkpoint(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must be a mapping")
    if checkpoint.get("schema_version") != "fabric.training_recovery_checkpoint.v1":
        raise ValueError("unsupported FABRIC training recovery checkpoint")
    observed_identity = checkpoint.get("run_identity")
    if not isinstance(observed_identity, Mapping):
        raise TypeError("resume checkpoint lacks a run identity")
    for identity_field, expected in expected_identity.items():
        if observed_identity.get(identity_field) != expected:
            raise ValueError(
                f"resume checkpoint {identity_field} identity differs"
            )
    required = {
        "completed_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "best_epoch",
        "best_validation_nll",
        "best_ont_matrix_kl_count_weighted",
        "best_model_state_dict",
        "best_optimizer_state_dict",
        "best_lr_scheduler_state_dict",
        "epochs_without_improvement",
        "history_rows",
        "monitor_records",
        "gene_order_rng_state",
        "global_rng_state",
        "training_complete",
        "held_out_test_evaluated",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"resume checkpoint fields are missing: {sorted(missing)}")
    if checkpoint["held_out_test_evaluated"] is not False:
        raise RuntimeError("training recovery checkpoint cannot contain test exposure")
    return checkpoint


def _restore_fit_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    model: FABRICV2Model,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    seed: int,
    condition_name: str,
    max_epochs: int,
    patience: int,
    monitor_enabled: bool,
) -> dict[str, object]:
    completed_epoch = checkpoint["completed_epoch"]
    best_epoch = checkpoint["best_epoch"]
    epochs_without_improvement = checkpoint["epochs_without_improvement"]
    if (
        type(completed_epoch) is not int
        or not 1 <= completed_epoch <= max_epochs
        or type(best_epoch) is not int
        or not 1 <= best_epoch <= completed_epoch
        or type(epochs_without_improvement) is not int
        or epochs_without_improvement < 0
    ):
        raise ValueError("resume checkpoint epoch state is invalid")
    if epochs_without_improvement != completed_epoch - best_epoch:
        raise ValueError("resume early-stopping counter differs from best epoch")

    history = checkpoint["history_rows"]
    monitor_rows = checkpoint["monitor_records"]
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError("resume history does not cover every completed epoch")
    if any(not isinstance(row, Mapping) for row in history):
        raise TypeError("resume history rows must be mappings")
    expected_epochs = list(range(1, completed_epoch + 1))
    if [row.get("epoch") for row in history] != expected_epochs or any(
        row.get("condition") != condition_name for row in history
    ):
        raise ValueError("resume history epoch or condition identity differs")
    validation_nlls = [
        float(row["validation_compatible_path_nll"]) for row in history
    ]
    if not np.isfinite(validation_nlls).all():
        raise ValueError("resume history contains non-finite validation NLL")
    observed_best_nll = float(checkpoint["best_validation_nll"])
    if (
        best_epoch != int(np.argmin(validation_nlls)) + 1
        or observed_best_nll != validation_nlls[best_epoch - 1]
    ):
        raise ValueError("resume best checkpoint differs from validation history")

    expected_monitor_count = completed_epoch if monitor_enabled else 0
    if not isinstance(monitor_rows, list) or len(monitor_rows) != expected_monitor_count:
        raise ValueError("resume monitor history differs from completed epochs")
    monitor_records: list[MonitorRecord] = []
    for expected_epoch, row in enumerate(monitor_rows, start=1):
        if (
            not isinstance(row, Mapping)
            or row.get("seed") != seed
            or row.get("condition") != condition_name
            or row.get("epoch") != expected_epoch
            or row.get("sealed") is not True
            or row.get("selection_eligible") is not False
            or not isinstance(row.get("fields"), Mapping)
        ):
            raise ValueError("resume monitor record identity differs")
        monitor_records.append(
            MonitorRecord(
                seed=seed,
                condition=condition_name,
                epoch=expected_epoch,
                fields=dict(row["fields"]),
                sealed=True,
                selection_eligible=False,
            )
        )
    observed_best_ont = checkpoint["best_ont_matrix_kl_count_weighted"]
    history_best_ont = history[best_epoch - 1]["ont_matrix_kl_count_weighted"]
    if monitor_enabled:
        if (
            isinstance(observed_best_ont, bool)
            or not isinstance(observed_best_ont, (int, float))
            or not np.isfinite(observed_best_ont)
            or float(observed_best_ont) != float(history_best_ont)
        ):
            raise ValueError("resume best ONT monitor differs from epoch history")
    elif observed_best_ont is not None or any(
        row["ont_matrix_kl_count_weighted"] is not None for row in history
    ):
        raise ValueError("resume contains ONT monitor values while monitoring is disabled")

    expected_training_complete = (
        completed_epoch >= max_epochs or epochs_without_improvement >= patience
    )
    if checkpoint["training_complete"] is not expected_training_complete:
        raise ValueError("resume terminal state differs from epoch controls")
    if not isinstance(checkpoint["gene_order_rng_state"], tuple):
        raise TypeError("resume gene-order RNG state is invalid")
    global_rng_state = checkpoint["global_rng_state"]
    if not isinstance(global_rng_state, tuple) or len(global_rng_state) != 4:
        raise TypeError("resume global RNG state is invalid")

    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    best_state = checkpoint["best_model_state_dict"]
    best_optimizer_state = checkpoint["best_optimizer_state_dict"]
    if not all(
        isinstance(value, Mapping)
        for value in (model_state, optimizer_state, best_state, best_optimizer_state)
    ):
        raise TypeError("resume model or optimizer state is invalid")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint["lr_scheduler_state_dict"]
    best_scheduler_state = checkpoint["best_lr_scheduler_state_dict"]
    if lr_scheduler is None:
        if scheduler_state is not None or best_scheduler_state is not None:
            raise ValueError("constant-LR resume contains scheduler state")
    else:
        if not isinstance(scheduler_state, Mapping) or not isinstance(
            best_scheduler_state, Mapping
        ):
            raise ValueError("scheduled-LR resume lacks scheduler state")
        lr_scheduler.load_state_dict(scheduler_state)

    return {
        "completed_epoch": completed_epoch,
        "history_rows": [dict(row) for row in history],
        "monitor_records": monitor_records,
        "best_nll": observed_best_nll,
        "best_epoch": best_epoch,
        "best_state": dict(best_state),
        "best_optimizer_state": dict(best_optimizer_state),
        "best_lr_scheduler_state": (
            dict(best_scheduler_state) if best_scheduler_state is not None else None
        ),
        "best_ont_matrix_kl": observed_best_ont,
        "epochs_without_improvement": epochs_without_improvement,
    }


def _best_checkpoint_from_recovery(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    manifest = checkpoint["run_identity"]["training_run_manifest"]
    return {
        "schema_version": "fabric.training_checkpoint.v2",
        "checkpoint_role": "best_validation",
        "run_identity": checkpoint["run_identity"],
        "seed": manifest["seed"],
        "condition": manifest["condition"],
        "completed_epoch": checkpoint["best_epoch"],
        "best_validation_nll": checkpoint["best_validation_nll"],
        "best_ont_matrix_kl_count_weighted": checkpoint[
            "best_ont_matrix_kl_count_weighted"
        ],
        "model_state_dict": checkpoint["best_model_state_dict"],
        "optimizer_state_dict": checkpoint["best_optimizer_state_dict"],
        "lr_scheduler_state_dict": checkpoint[
            "best_lr_scheduler_state_dict"
        ],
        "held_out_test_evaluated": False,
    }


def _write_recovery_history(
    run_dir: Path, checkpoint: Mapping[str, object]
) -> None:
    history = pd.DataFrame(checkpoint["history_rows"])
    _atomic_write_text(run_dir / "history.tsv", history.to_csv(sep="\t", index=False))
    monitor_text = "".join(
        json.dumps(dict(record), sort_keys=True) + "\n"
        for record in checkpoint["monitor_records"]
    )
    _atomic_write_text(run_dir / "sealed_validation_monitor.jsonl", monitor_text)


def _reconcile_recovery_artifacts(
    run_dir: Path, checkpoint: Mapping[str, object]
) -> None:
    _atomic_torch_save(_best_checkpoint_from_recovery(checkpoint), run_dir / "best.pt")
    _write_recovery_history(run_dir, checkpoint)


def _persist_epoch_recovery(
    state: Mapping[str, object],
    *,
    run_dir: Path,
    run_identity: Mapping[str, object],
) -> None:
    checkpoint = {
        "schema_version": "fabric.training_recovery_checkpoint.v1",
        "checkpoint_role": "latest_completed_epoch",
        "recovery_granularity": "next_epoch_after_completed_epoch",
        "run_identity": copy.deepcopy(dict(run_identity)),
        **dict(state),
    }
    # latest.pt is self-contained, including the selected best state.  Its
    # atomic replacement is therefore the sole recovery commit point.
    _atomic_torch_save(checkpoint, run_dir / "latest.pt")
    if checkpoint["best_improved_this_epoch"] is True:
        _atomic_torch_save(
            _best_checkpoint_from_recovery(checkpoint), run_dir / "best.pt"
        )
    _write_recovery_history(run_dir, checkpoint)


def _write_run(
    run: TrainingRunResult,
    config: Mapping[str, object],
    run_dir: Path,
    genes: Sequence[PreparedGene],
    prepared: PreparedDataset | BackedPreparedDataset | None,
    *,
    recovery_identity: Mapping[str, object],
) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError("run directory disappeared before finalization")
    if set(run.metrics["split"]) != {"val"} or len(run.metrics) != 1:
        raise AssertionError(
            "training artifact writer requires one validation metric row"
        )
    _atomic_write_text(
        run_dir / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False)
    )
    _atomic_write_text(
        run_dir / "training_run_manifest.json",
        json.dumps(asdict(run.manifest), indent=2, sort_keys=True),
    )
    _atomic_write_text(
        run_dir / "input_manifest.json",
        json.dumps(
            {
                "contract": "FABRIC_ARCHITECTURE_V2",
                "input_manifest_id": (
                    prepared.input_manifest_id if prepared is not None else None
                ),
                "compatibility_artifact_id": (
                    prepared.compatibility_artifact_id if prepared is not None else None
                ),
                "gene_ids": [gene.gene_id for gene in genes],
                "cell_gene_instance_count": sum(len(gene.cell_ids) for gene in genes),
                "structural_path_count": sum(len(gene.path_ids) for gene in genes),
                "execution_scope": config["execution"]["scope"],
                "test_exposure": config.get("inputs", {}).get(
                    "test_exposure", "unspecified"
                ),
                "test_model_predictions_status": "NOT_COMPUTED_DURING_TRAINING",
                "gene_shape_adaptive_batching": {
                    "batch_policy": config["resources"]["batch_policy"],
                    "compute_precision": config["resources"]["compute_precision"],
                    "prefetch_backed_gene_shards": config["resources"][
                        "prefetch_backed_gene_shards"
                    ],
                    "target_gpu_allocated_bytes": config["resources"][
                        "target_gpu_allocated_bytes"
                    ],
                    "unmodeled_gpu_reserve_bytes": config["resources"][
                        "unmodeled_gpu_reserve_bytes"
                    ],
                    "max_cells_per_gpu_batch": config["resources"][
                        "max_cells_per_gpu_batch"
                    ],
                    "train_bytes_per_shape_element": config["resources"][
                        "train_bytes_per_shape_element"
                    ],
                    "evaluation_bytes_per_shape_element": config["resources"][
                        "evaluation_bytes_per_shape_element"
                    ],
                    "scientific_estimand": "path_log_softmax_probability",
                    "probability_numerical_tolerance": config["resources"][
                        "batching_probability_tolerance"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )
    run.metrics.to_csv(run_dir / "metrics.tsv", sep="\t", index=False)
    final_best_checkpoint = {
        "schema_version": "fabric.training_checkpoint.v2",
        "checkpoint_role": "best_validation",
        "run_identity": copy.deepcopy(dict(recovery_identity)),
        "seed": run.manifest.seed,
        "condition": run.manifest.condition,
        "completed_epoch": run.result.best_epoch,
        "best_validation_nll": run.result.best_validation_nll,
        "best_ont_matrix_kl_count_weighted": (
            run.result.best_ont_matrix_kl_count_weighted
        ),
        "model_state_dict": _cpu_copy(run.result.model.state_dict()),
        "optimizer_state_dict": run.result.optimizer_state_dict,
        "lr_scheduler_state_dict": run.result.lr_scheduler_state_dict,
        "held_out_test_evaluated": False,
    }
    _atomic_torch_save(final_best_checkpoint, run_dir / "best.pt")
    _atomic_torch_save(final_best_checkpoint, run_dir / "checkpoint.pt")
    run.result.history.to_csv(run_dir / "history.tsv", sep="\t", index=False)
    monitor_rows = [asdict(record) for record in run.result.monitor_records]
    groups = optimizer_parameter_groups(
        run.result.model,
        lambda_base=float(config["optimizer"]["lambda_base"]),
        lambda_int=float(config["optimizer"]["lambda_int"]),
    )
    with (run_dir / "sealed_validation_monitor.jsonl").open("w") as handle:
        for record in monitor_rows:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    optimizer = config["optimizer"]
    (run_dir / "optimizer_manifest.json").write_text(
        json.dumps(
            {
                "contract": "FABRIC_ARCHITECTURE_V2_15_5",
                "family": optimizer["family"],
                "learning_rate": optimizer["learning_rate"],
                "lr_scheduler": optimizer["lr_scheduler"],
                "gradient_clip_norm": optimizer["gradient_clip_norm"],
                "lambda_base": optimizer["lambda_base"],
                "lambda_int": optimizer["lambda_int"],
                "explicit_additional_l2_penalty": False,
                "parameter_groups": [
                    {
                        "group_name": group["group_name"],
                        "weight_decay": group["weight_decay"],
                        "parameter_names": list(group["parameter_names"]),
                    }
                    for group in groups
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    (run_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "contract": "FABRIC_ARCHITECTURE_V2",
                "checkpoint_status": "FROZEN_AFTER_VALIDATION_SELECTION",
                "selection_metric": "validation_molecule_weighted_compatible_path_nll",
                "selection_rules_used_held_out_test": False,
                "final_test_authorized_at_training_time": config["execution"][
                    "final_test_authorized"
                ],
                "record": {
                    "seed": run.manifest.seed,
                    "condition": run.manifest.condition,
                    "model_condition": _MODEL_CONDITION[run.manifest.condition],
                    "readout_kind": run.result.model.readout_kind,
                    "checkpoint_relative_path": "checkpoint.pt",
                    "best_checkpoint_relative_path": "best.pt",
                    "latest_recovery_checkpoint_relative_path": "latest.pt",
                    "checkpoint_schema_version": "fabric.training_checkpoint.v2",
                    "latest_recovery_schema_version": (
                        "fabric.training_recovery_checkpoint.v1"
                    ),
                    "recovery_granularity": (
                        "next_epoch_after_completed_epoch"
                    ),
                    "optimizer_state_included": True,
                    "lr_scheduler_state_included": (
                        run.result.lr_scheduler_state_dict is not None
                    ),
                    "best_epoch": run.result.best_epoch,
                    "best_validation_molecule_weighted_compatible_path_nll": run.result.best_validation_nll,
                    "best_ont_matrix_kl_count_weighted_reporting_only": (
                        run.result.best_ont_matrix_kl_count_weighted
                    ),
                    "selection_split": "val",
                    "held_out_test_evaluated": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    expected_monitor_records = (
        len(run.result.history) if config["monitor"]["enabled"] else 0
    )
    if len(monitor_rows) != expected_monitor_records:
        raise AssertionError(
            "per-epoch monitor record count differs from completed epochs"
        )
    (run_dir / "monitor_manifest.json").write_text(
        json.dumps(
            {
                "contract": "FABRIC_ARCHITECTURE_V2_15_4",
                "enabled": config["monitor"]["enabled"],
                "timing": config["monitor"]["timing"],
                "sealed": config["monitor"]["sealed"],
                "selection_eligible": config["monitor"]["selection_eligible"],
                "core_metrics": [
                    "validation_compatible_path_nll",
                    "ont_matrix_kl_count_weighted",
                ],
                "checkpoint_selection_metric": ("validation_compatible_path_nll"),
                "reporting_only_metric": "ont_matrix_kl_count_weighted",
                "completed_epoch_count": expected_monitor_records,
                "record_count": len(monitor_rows),
                "all_records_sealed": all(
                    row["sealed"] is True for row in monitor_rows
                ),
                "any_record_selection_eligible": any(
                    row["selection_eligible"] is True for row in monitor_rows
                ),
                "held_out_test_model_predictions_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the single FABRIC V2 runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--resume-from",
        help=(
            "resume the same run from its atomic latest.pt after the most recent "
            "completed epoch"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--condition", required=True, choices=RUN_CONDITIONS)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-scheduler", choices=("constant", "reduce_on_plateau"))
    parser.add_argument("--lr-factor", type=float)
    parser.add_argument("--lr-patience", type=int)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--lambda-base", type=float)
    parser.add_argument("--lambda-int", type=float)
    parser.add_argument("--max-train-gene-cells-per-gene", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true")
    source.add_argument("--fixture")
    args = parser.parse_args(argv)
    config = resolve_run_config(
        load_config(args.config),
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        lambda_base=args.lambda_base,
        lambda_int=args.lambda_int,
        max_train_gene_cells_per_gene=args.max_train_gene_cells_per_gene,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )
    if config["execution"]["scope"] == "toy" and not args.toy:
        raise ValueError("toy execution requires --toy")
    if config["execution"]["scope"] != "toy" and args.toy:
        raise ValueError("non-toy execution requires --fixture")
    assert_execution_admitted(config)
    if args.toy:
        data: Sequence[PreparedGene] | PreparedDataset | BackedPreparedDataset = (
            make_toy_genes()
        )
    else:
        fixture_path = Path(args.fixture)
        prepared = (
            BackedPreparedDataset.load(fixture_path)
            if fixture_path.is_dir()
            else torch.load(fixture_path, map_location="cpu", weights_only=False)
        )
        if not isinstance(prepared, (PreparedDataset, BackedPreparedDataset)):
            raise TypeError(
                "fixture must contain one FABRIC V2 PreparedDataset or backed dataset"
            )
        data = prepared
    train_run(
        data,
        config,
        seed=args.seed,
        condition=args.condition,
        device=args.device,
        run_dir=args.run_dir,
        resume_from=args.resume_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackedGeneSequence",
    "BackedPreparedDataset",
    "FULL_COHORT_SCOPE",
    "RUN_CONDITIONS",
    "SPLITS",
    "ConditionResult",
    "EpochMonitor",
    "MonitorRecord",
    "OptimizerGridSelection",
    "OptimizerTuningRun",
    "PreparedDataset",
    "PreparedGene",
    "RouteDegreeCapAuditMeasurement",
    "RouteDegreeCapAuditManifest",
    "RouteDegreeCapSyntheticConfig",
    "TrainingRunManifest",
    "TrainingRunResult",
    "TrainGeneCellSample",
    "ValidationPrediction",
    "ValidationSnapshot",
    "assert_execution_admitted",
    "bind_route_degree_cap_structural_audit",
    "build_optimizer",
    "build_lr_scheduler",
    "build_paired_models",
    "validation_ont_matrix_kl_monitor",
    "evaluate_final_test",
    "load_config",
    "make_toy_genes",
    "optimizer_parameter_groups",
    "prepared_gene_from_assembly",
    "rows_for_split",
    "run_route_degree_cap_synthetic",
    "resolve_run_config",
    "sample_train_gene_cells_for_epoch",
    "select_lambda_pair",
    "split_informative_molecule_mass",
    "train_run",
    "tune_optimizer_grid",
    "training_manifest_from_config",
]
