"""Direct FABRIC V1 training hierarchy for CIS and the four fixed children."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .likelihood import brute_force_compatible_path_nll, compatible_path_nll
from .model import (
    AlternativeBatch,
    AugmentedPathReadout,
    CISOutput,
    EdgeGraphGPS,
    EventBatch,
    EventScorer,
    FrozenAlternativeState,
    GraphGPSBatch,
    PathReadoutInput,
    StateBatch,
    StateScorer,
    build_frozen_alternative_state,
    clone_from_state_parent,
    freeze_cis_parent,
)

if TYPE_CHECKING:
    from .dataset import ATACContext, FactorActivityContext, RNAStatePCA


VARIANTS = ("cis", "state", "state_dna", "state_rna", "state_dna_rna")
SPLITS = ("train", "val", "test")
NORMALIZED_SOURCE_ROLES = (
    "rna_counts",
    "full_rna_glue_embedding",
    "full_rna_consensus_peak_bed",
    "full_rna_atac_peak_counts",
    "compatibility_ec",
    "reference_fasta",
    "reference_fasta_index",
    "rna_gene_gtf",
    "dna_motif_library",
    "dna_motif_index",
    "rna_motif_directory",
    "rna_motif_gene_map",
)
PREPARATION_CONFIG_FIELDS = (
    "data.reuse_documented_cell_split",
    "data.target_sum_rna",
    "data.target_sum_atac",
    "data.state_pca_dim",
    "data.atac_neighbors.exact_stage",
    "data.atac_neighbors.stage_field",
    "data.atac_neighbors.donor_id_field",
    "data.atac_neighbors.donor_eligibility_rule",
    "data.atac_neighbors.k",
    "data.atac_neighbors.weighting",
    "data.atac_neighbors.temperature",
    "choices.rank_tolerance",
    "choices.minimum_informative_molecule_mass",
    "choices.minimum_alternative_support",
    "motifs.dna_window_bp",
    "motifs.rna_window_bp",
    "motifs.dna_minimum_relative_score",
    "motifs.rna_minimum_relative_score",
    "motifs.dna_events_per_choice_cap",
    "motifs.rna_events_per_choice_cap",
    "gates.minimum_valid_molecule_mass",
    "gates.minimum_weighted_variance",
)


@dataclass(frozen=True)
class PreparedGateBaseline:
    """Frozen train-only eligibility statistics for one State/event key axis."""

    mean: torch.Tensor
    valid_molecule_mass: torch.Tensor
    weighted_variance: torch.Tensor
    eligible: torch.Tensor
    dna_reliability_mass: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedGene:
    """One on-demand gene bundle; large source matrices stay outside it."""

    gene_id: str
    graph: GraphGPSBatch
    alternatives: AlternativeBatch
    path_edge_incidence: torch.Tensor
    path_choice_incidence: torch.Tensor
    alternative_eligible: torch.Tensor  # bool [A]
    state_features: torch.Tensor  # [cells, Z]
    dna_event_features: torch.Tensor  # [DNA events, U_DNA]
    dna_event_relation: torch.Tensor  # [DNA events, A]
    dna_event_choice_index: torch.Tensor  # [DNA events]
    dna_gate: torch.Tensor  # [cells, DNA events]
    rna_event_features: torch.Tensor  # [RNA events, U_RNA]
    rna_event_relation: torch.Tensor  # [RNA events, A]
    rna_event_choice_index: torch.Tensor  # [RNA events]
    rna_gate: torch.Tensor  # [cells, RNA events]
    compatible_path_indices: torch.Tensor  # padded [EC rows, width]
    compatible_path_mask: torch.Tensor  # [EC rows, width]
    row_cell_index: torch.Tensor  # [EC rows]
    molecule_count: torch.Tensor  # [EC rows]
    split: tuple[str, ...]
    identifiable_row_mask: torch.Tensor  # bool [EC rows]
    cell_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    dna_event_ids: tuple[str, ...]
    rna_event_ids: tuple[str, ...]
    graph_generation: str
    split_source: str
    dna_event_factor_ids: tuple[str, ...] = ()
    rna_event_factor_ids: tuple[str, ...] = ()
    dna_event_peak_ids: tuple[str, ...] = ()
    state_baseline: PreparedGateBaseline | None = None
    dna_baseline: PreparedGateBaseline | None = None
    rna_baseline: PreparedGateBaseline | None = None
    alternative_span: torch.Tensor | None = None
    dna_candidate_event_count: torch.Tensor | None = None
    dna_selected_event_count: torch.Tensor | None = None
    dna_cap_saturated: torch.Tensor | None = None
    dna_boundary_rank_motif_score: torch.Tensor | None = None
    rna_candidate_event_count: torch.Tensor | None = None
    rna_selected_event_count: torch.Tensor | None = None
    rna_cap_saturated: torch.Tensor | None = None
    rna_boundary_rank_motif_score: torch.Tensor | None = None
    normalized_source_paths: tuple[tuple[str, str], ...] = ()
    reviewed_factor_mapping: str | None = None
    atac_donor_eligibility_source: str | None = None
    peak_support_source: str | None = None
    preparation_config_source: str | None = None
    preparation_values: tuple[tuple[str, object], ...] = ()
    state_pca_fit_batch_size: int | None = None


@dataclass
class VariantModules:
    cis: EdgeGraphGPS
    state: StateScorer | None
    dna: EventScorer | None
    rna: EventScorer | None


@dataclass(frozen=True)
class HierarchyResult:
    modules: Mapping[str, VariantModules]
    metrics: pd.DataFrame
    admission: Mapping[str, float | bool]
    history: pd.DataFrame


@dataclass(frozen=True)
class PreparedDataset:
    """Minimal identity envelope for a prepared, per-gene tensor collection."""

    genes: tuple[PreparedGene, ...]
    target_gene_ids: tuple[str, ...]
    graph_generation: str
    split_source: str
    factor_mapping_reviewed: bool
    normalized_source_paths: tuple[tuple[str, str], ...] = ()
    reviewed_factor_mapping: str | None = None
    atac_donor_eligibility_source: str | None = None
    peak_support_source: str | None = None
    preparation_config_source: str | None = None
    preparation_values: tuple[tuple[str, object], ...] = ()
    state_pca: RNAStatePCA | None = None
    factor_context: FactorActivityContext | None = None
    atac_context: ATACContext | None = None


def prepare_dataset_identity(
    genes: Sequence[PreparedGene],
    *,
    factor_mapping_reviewed: bool,
    normalized_source_paths: Mapping[str, str | Path] | None = None,
    reviewed_factor_mapping: str | Path | None = None,
    atac_donor_eligibility_source: str | Path | None = None,
    peak_support_source: str | Path | None = None,
    preparation_config_source: str | Path | None = None,
    preparation_values: Mapping[str, object] | None = None,
    state_pca: RNAStatePCA | None = None,
    factor_context: FactorActivityContext | None = None,
    atac_context: ATACContext | None = None,
) -> PreparedDataset:
    """Bind genes to the normalized sources used to prepare their tensors."""

    if not genes:
        raise ValueError("prepared dataset requires at least one gene")
    genes = tuple(_gene_to_cpu(gene) for gene in genes)
    _validate_prepared_genes(genes)
    source_paths = {} if normalized_source_paths is None else normalized_source_paths
    missing_roles = sorted(set(NORMALIZED_SOURCE_ROLES) - set(source_paths))
    extra_roles = sorted(set(source_paths) - set(NORMALIZED_SOURCE_ROLES))
    if source_paths and (missing_roles or extra_roles):
        raise ValueError(
            "normalized source identity must contain exactly the V1 source roles; "
            f"missing={missing_roles}, extra={extra_roles}"
        )
    if factor_mapping_reviewed and not source_paths:
        raise ValueError(
            "a reviewed factor mapping cannot be asserted without normalized source "
            "identity"
        )
    if factor_mapping_reviewed and reviewed_factor_mapping is None:
        raise ValueError(
            "a reviewed factor mapping requires its reviewed mapping path identity"
        )
    normalized_paths = tuple(
        (role, str(Path(source_paths[role]).resolve()))
        for role in NORMALIZED_SOURCE_ROLES
        if source_paths
    )
    mapping_path = (
        None
        if reviewed_factor_mapping is None
        else str(Path(reviewed_factor_mapping).resolve())
    )
    donor_eligibility_path = (
        None
        if atac_donor_eligibility_source is None
        else str(Path(atac_donor_eligibility_source).resolve())
    )
    peak_support_path = (
        None
        if peak_support_source is None
        else str(Path(peak_support_source).resolve())
    )
    config_path = (
        None
        if preparation_config_source is None
        else str(Path(preparation_config_source).resolve())
    )
    normalized_preparation_values = _normalize_preparation_values(preparation_values)
    pca_fit_batch_size = None if state_pca is None else int(state_pca.fit_batch_size)
    if (config_path is None) != (not normalized_preparation_values):
        raise ValueError(
            "preparation config source and frozen preparation values are required "
            "together"
        )
    for gene in genes:
        if gene.normalized_source_paths not in ((), normalized_paths):
            raise ValueError(
                f"gene {gene.gene_id} is already bound to different normalized sources"
            )
        if gene.reviewed_factor_mapping not in (None, mapping_path):
            raise ValueError(
                f"gene {gene.gene_id} is already bound to a different factor mapping"
            )
        if gene.atac_donor_eligibility_source not in (None, donor_eligibility_path):
            raise ValueError(
                f"gene {gene.gene_id} is bound to different ATAC donor eligibility"
            )
        if gene.peak_support_source not in (None, peak_support_path):
            raise ValueError(
                f"gene {gene.gene_id} is bound to a different peak-support source"
            )
        if gene.preparation_config_source not in (None, config_path):
            raise ValueError(
                f"gene {gene.gene_id} is bound to a different preparation config"
            )
        if gene.preparation_values not in ((), normalized_preparation_values):
            raise ValueError(
                f"gene {gene.gene_id} is bound to different preparation values"
            )
        if gene.state_pca_fit_batch_size not in (None, pca_fit_batch_size):
            raise ValueError(
                f"gene {gene.gene_id} is bound to a different State PCA fit batch"
            )
    genes = tuple(
        replace(
            gene,
            normalized_source_paths=normalized_paths,
            reviewed_factor_mapping=mapping_path,
            atac_donor_eligibility_source=donor_eligibility_path,
            peak_support_source=peak_support_path,
            preparation_config_source=config_path,
            preparation_values=normalized_preparation_values,
            state_pca_fit_batch_size=pca_fit_batch_size,
        )
        for gene in genes
    )
    graph_generations = {gene.graph_generation for gene in genes}
    split_sources = {gene.split_source for gene in genes}
    if len(graph_generations) != 1 or len(split_sources) != 1:
        raise ValueError(
            "prepared dataset cannot mix graph generations or split sources"
        )
    return PreparedDataset(
        genes=genes,
        target_gene_ids=tuple(gene.gene_id for gene in genes),
        graph_generation=next(iter(graph_generations)),
        split_source=next(iter(split_sources)),
        factor_mapping_reviewed=bool(factor_mapping_reviewed),
        normalized_source_paths=normalized_paths,
        reviewed_factor_mapping=mapping_path,
        atac_donor_eligibility_source=donor_eligibility_path,
        peak_support_source=peak_support_path,
        preparation_config_source=config_path,
        preparation_values=normalized_preparation_values,
        state_pca=state_pca,
        factor_context=factor_context,
        atac_context=atac_context,
    )


def preparation_values_from_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Extract the fixed F1/F2 numerical and categorical preparation contract."""

    values: dict[str, object] = {}
    for field in PREPARATION_CONFIG_FIELDS:
        current: object = config
        for part in field.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ValueError(f"preparation config misses {field}")
            current = current[part]
        if current is None:
            raise ValueError(f"preparation config field is unresolved: {field}")
        values[field] = current
    return values


def _normalize_preparation_values(
    values: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    if values is None:
        return ()
    missing = sorted(set(PREPARATION_CONFIG_FIELDS) - set(values))
    extra = sorted(set(values) - set(PREPARATION_CONFIG_FIELDS))
    if missing or extra:
        raise ValueError(
            "preparation values must contain exactly the frozen fields; "
            f"missing={missing}, extra={extra}"
        )
    normalized: list[tuple[str, object]] = []
    for field in PREPARATION_CONFIG_FIELDS:
        value = values[field]
        if isinstance(value, bool | str | int):
            normalized.append((field, value))
        elif isinstance(value, float) and np.isfinite(value):
            normalized.append((field, value))
        else:
            raise TypeError(f"preparation value {field} must be a finite scalar")
    return tuple(normalized)


def load_config(path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("contract") != "FABRIC_ARCHITECTURE_V1":
        raise ValueError("training config is not bound to FABRIC_ARCHITECTURE_V1")
    variants = tuple(config["training"]["variants"])
    if variants != VARIANTS:
        raise ValueError(f"V1 variants must be exactly {VARIANTS}")
    model = config["model"]
    if "cis_layers" in model and model["cis_layers"] not in {1, None}:
        raise ValueError("FABRIC V1 has exactly one shallow GraphGPS block")
    if "cis_dropout" in model and model["cis_dropout"] not in {0, 0.0, None}:
        raise ValueError("FABRIC V1 does not define an additional dropout mechanism")
    return config


def assert_full7198_ready(
    config: Mapping[str, object], prepared: PreparedDataset | None = None
) -> None:
    """Fail before formal F4 while audited numerical fields remain unresolved."""

    if not bool(config["training"].get("formal_full7198_authorized", False)):
        raise RuntimeError("formal full7198 training is not authorized by this config")
    unresolved: list[str] = []

    def visit(prefix: str, value: object) -> None:
        if value is None:
            unresolved.append(prefix)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)

    visit("", config)
    if unresolved:
        raise RuntimeError(f"full7198 audit values remain unresolved: {unresolved}")
    if prepared is None:
        return
    if not prepared.factor_mapping_reviewed:
        raise RuntimeError("formal full7198 requires the reviewed factor/group mapping")
    source_paths = dict(prepared.normalized_source_paths)
    if (
        len(source_paths) != len(prepared.normalized_source_paths)
        or set(source_paths) != set(NORMALIZED_SOURCE_ROLES)
        or any(not isinstance(path, str) or not path for path in source_paths.values())
    ):
        raise RuntimeError(
            "formal full7198 requires the exact normalized source path identity"
        )
    if (
        not isinstance(prepared.reviewed_factor_mapping, str)
        or not prepared.reviewed_factor_mapping
    ):
        raise RuntimeError(
            "formal full7198 requires the reviewed factor mapping path identity"
        )
    auxiliary_sources = {
        "ATAC donor eligibility": prepared.atac_donor_eligibility_source,
        "peak support": prepared.peak_support_source,
    }
    missing_auxiliary = [
        label
        for label, path in auxiliary_sources.items()
        if not isinstance(path, str) or not path
    ]
    if missing_auxiliary:
        raise RuntimeError(
            "formal full7198 requires explicit auxiliary source identity: "
            f"{missing_auxiliary}"
        )
    if (
        not isinstance(prepared.preparation_config_source, str)
        or not prepared.preparation_config_source
        or not prepared.preparation_values
    ):
        raise RuntimeError(
            "formal full7198 requires the frozen preparation config identity"
        )
    for gene in prepared.genes:
        if gene.normalized_source_paths != prepared.normalized_source_paths:
            raise RuntimeError("formal bundle mixes normalized source identities")
        if gene.reviewed_factor_mapping != prepared.reviewed_factor_mapping:
            raise RuntimeError("formal bundle mixes reviewed factor mapping identities")
        if (
            gene.atac_donor_eligibility_source != prepared.atac_donor_eligibility_source
            or gene.peak_support_source != prepared.peak_support_source
        ):
            raise RuntimeError("formal bundle mixes auxiliary source identities")
        if (
            gene.preparation_config_source != prepared.preparation_config_source
            or gene.preparation_values != prepared.preparation_values
        ):
            raise RuntimeError("formal bundle mixes preparation config identities")
    from .annotation import load_external_inputs, resolve_and_validate_graph_generation
    import pyarrow.dataset as pads

    external = load_external_inputs(config["external_inputs"])
    mismatched_sources = [
        role
        for role in NORMALIZED_SOURCE_ROLES
        if Path(source_paths[role]).resolve() != external.path(role).resolve()
    ]
    if mismatched_sources:
        raise RuntimeError(
            "formal bundle normalized sources differ from external inputs: "
            f"{mismatched_sources}"
        )
    factor_identity = config.get("factor_identity")
    if not isinstance(factor_identity, Mapping) or not isinstance(
        factor_identity.get("reviewed_mapping"), str
    ):
        raise RuntimeError("formal config requires a reviewed factor mapping path")
    reviewed_mapping = Path(factor_identity["reviewed_mapping"]).resolve()
    if Path(prepared.reviewed_factor_mapping).resolve() != reviewed_mapping:
        raise RuntimeError(
            "formal bundle factor mapping differs from the reviewed config mapping"
        )
    if not reviewed_mapping.exists():
        raise RuntimeError("the reviewed factor mapping path does not exist")
    neighbor_config = config.get("data", {}).get("atac_neighbors", {})
    motif_config = config.get("motifs", {})
    configured_donor_source = neighbor_config.get("donor_eligibility_path")
    configured_peak_support = motif_config.get("peak_support_path")
    if not isinstance(configured_donor_source, str) or not isinstance(
        configured_peak_support, str
    ):
        raise RuntimeError(
            "formal config requires donor_eligibility_path and peak_support_path"
        )
    if (
        Path(prepared.atac_donor_eligibility_source).resolve()
        != Path(configured_donor_source).resolve()
        or Path(prepared.peak_support_source).resolve()
        != Path(configured_peak_support).resolve()
    ):
        raise RuntimeError("formal bundle auxiliary sources differ from config")
    if (
        not Path(configured_donor_source).exists()
        or not Path(configured_peak_support).exists()
    ):
        raise RuntimeError("formal auxiliary source path does not exist")
    expected_preparation = _normalize_preparation_values(
        preparation_values_from_config(config)
    )
    if prepared.preparation_values != expected_preparation:
        raise RuntimeError("formal bundle preparation values differ from config")
    from .dataset import validate_prepared_external_context

    try:
        validate_prepared_external_context(prepared)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"formal bundle context is incomplete: {error}") from error
    if len(prepared.genes) != 7_198 or len(prepared.target_gene_ids) != 7_198:
        raise RuntimeError("formal full7198 input must contain exactly 7,198 genes")
    observed = tuple(gene.gene_id for gene in prepared.genes)
    if observed != prepared.target_gene_ids or len(set(observed)) != 7_198:
        raise RuntimeError(
            "formal full7198 gene order/identity differs from preparation metadata"
        )
    frozen_generation = resolve_and_validate_graph_generation(external)
    path_table = frozen_generation / "outputs" / "graph" / "path_table.parquet"
    gene_axis = (
        pads.dataset(path_table, format="parquet")
        .to_table(columns=["gene_id"])
        .column("gene_id")
        .to_pylist()
    )
    expected_gene_ids = tuple(dict.fromkeys(map(str, gene_axis)))
    if prepared.target_gene_ids != expected_gene_ids:
        raise RuntimeError(
            "formal full7198 gene axis differs from the frozen graph generation"
        )
    diagnostic_panel = config.get("diagnostic_panel")
    panel_ids = (
        None
        if not isinstance(diagnostic_panel, Mapping)
        else diagnostic_panel.get("frozen_gene_ids")
    )
    if (
        not isinstance(panel_ids, Sequence)
        or isinstance(panel_ids, (str, bytes))
        or not panel_ids
        or len(set(map(str, panel_ids))) != len(panel_ids)
        or not set(map(str, panel_ids)).issubset(expected_gene_ids)
    ):
        raise RuntimeError(
            "formal config requires a unique frozen diagnostic panel within full7198"
        )
    if Path(prepared.graph_generation).resolve() != frozen_generation:
        raise RuntimeError(
            "formal bundle graph generation differs from external inputs"
        )
    if Path(prepared.split_source).resolve() != external.path("cell_split").resolve():
        raise RuntimeError("formal bundle split differs from external inputs")
    for gene in prepared.genes:
        if gene.graph_generation != prepared.graph_generation:
            raise RuntimeError("formal bundle mixes graph generations")
        if gene.split_source != prepared.split_source:
            raise RuntimeError("formal bundle mixes split identities")


def train_hierarchy(
    genes: Sequence[PreparedGene],
    config: Mapping[str, object],
    *,
    device: str | torch.device,
    run_dir: str | Path | None = None,
    _formal_identity_validated: bool = False,
) -> HierarchyResult:
    """Train the fixed parent hierarchy without sharing unimodal child weights."""

    if not genes:
        raise ValueError("training requires at least one prepared gene")
    _validate_prepared_genes(genes)
    training = config["training"]
    model_config = config["model"]
    if _formal_identity_validated and (
        int(config.get("target_gene_count", 0)) != 7_198
        or not bool(training.get("formal_full7198_authorized", False))
    ):
        raise ValueError(
            "formal result labeling requires an authorized full7198 configuration"
        )
    screening_evidence_only = not _formal_identity_validated
    seed = int(training["seed"])
    _seed_everything(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for training device {device}")
    # Large per-cell context remains on CPU.  _loss_for_rows transfers only the
    # current gene's static graph and the current EC batch/cells to the device.
    genes_cpu = tuple(_gene_to_cpu(gene) for gene in genes)

    edge_feature_dim = int(genes_cpu[0].graph.edge_features.shape[1])
    cis = EdgeGraphGPS(
        edge_feature_dim=edge_feature_dim,
        hidden_dim=int(model_config["cis_hidden_dim"]),
        attention_heads=int(model_config["cis_heads"]),
    ).to(torch_device)
    readout = AugmentedPathReadout(
        length_penalty=float(model_config["path_length_prior_weight"])
    ).to(torch_device)
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    max_epochs = int(training["max_epochs"])
    batch_rows = int(training.get("ec_batch_rows", 512))

    history_rows: list[dict[str, object]] = []
    _fit_variant(
        variant="cis",
        genes=genes_cpu,
        modules=VariantModules(cis=cis, state=None, dna=None, rna=None),
        frozen=None,
        readout=readout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_rows=batch_rows,
        seed=seed,
        history_rows=history_rows,
    )
    freeze_cis_parent(cis)
    cis_cache = _freeze_cis_outputs(cis, genes_cpu)
    b0 = {gene.gene_id: fit_b0_path_logits(gene) for gene in genes_cpu}
    b0_val = _evaluate_fixed_logits(genes_cpu, b0, split="val")
    cis_modules = VariantModules(cis=cis, state=None, dna=None, rna=None)
    cis_val = evaluate_variant_nll(
        genes_cpu, cis_modules, cis_cache, readout, split="val"
    )
    if not any(gene.split for gene in genes_cpu):
        raise ValueError("likelihood exactness requires at least one supervised gene")
    toy_parity_error = frozen_toy_likelihood_parity_error()
    real_parity_error = fixed_real_fixture_parity_error(
        config["admission"]["real_fixture_directory"]
    )
    parity_error = max(toy_parity_error, real_parity_error)
    improvement = b0_val - cis_val
    minimum_improvement = float(
        config["admission"]["minimum_b0_validation_improvement"]
    )
    admission = {
        "likelihood_exactness": bool(parity_error <= 1e-7),
        "likelihood_parity_absolute_error": parity_error,
        "toy_likelihood_parity_absolute_error": toy_parity_error,
        "real_fixture_row_likelihood_parity_absolute_error": real_parity_error,
        "b0_validation_nll": b0_val,
        "cis_validation_nll": cis_val,
        "b0_improvement": improvement,
        "minimum_b0_validation_improvement": minimum_improvement,
        "passed": bool(parity_error <= 1e-7 and improvement >= minimum_improvement),
    }
    if not admission["passed"]:
        raise RuntimeError(f"CIS admission failed: {admission}")

    alternative_dim = next(
        cache[1].h_base.shape[1]
        for cache in cis_cache.values()
        if cache[1].h_base.shape[0] > 0
    )
    state_dim = int(genes_cpu[0].state_features.shape[1])
    state = StateScorer(
        state_dim=state_dim,
        alternative_dim=alternative_dim,
        rank=int(model_config["state_rank"]),
    ).to(torch_device)
    state_modules = VariantModules(cis=cis, state=state, dna=None, rna=None)
    _fit_variant(
        variant="state",
        genes=genes_cpu,
        modules=state_modules,
        frozen=cis_cache,
        readout=readout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_rows=batch_rows,
        seed=seed + 1,
        history_rows=history_rows,
    )

    modules: dict[str, VariantModules] = {
        "cis": cis_modules,
        "state": state_modules,
    }
    dna_dim = int(genes_cpu[0].dna_event_features.shape[1])
    rna_dim = int(genes_cpu[0].rna_event_features.shape[1])
    for offset, variant in enumerate(
        ("state_dna", "state_rna", "state_dna_rna"), start=2
    ):
        state_parent = clone_from_state_parent(state).to(torch_device)
        dna = (
            EventScorer(
                event_dim=dna_dim,
                alternative_dim=alternative_dim,
                rank=int(model_config["dna_rank"]),
            ).to(torch_device)
            if variant in {"state_dna", "state_dna_rna"}
            else None
        )
        rna = (
            EventScorer(
                event_dim=rna_dim,
                alternative_dim=alternative_dim,
                rank=int(model_config["rna_rank"]),
            ).to(torch_device)
            if variant in {"state_rna", "state_dna_rna"}
            else None
        )
        child = VariantModules(cis=cis, state=state_parent, dna=dna, rna=rna)
        _fit_variant(
            variant=variant,
            genes=genes_cpu,
            modules=child,
            frozen=cis_cache,
            readout=readout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_rows=batch_rows,
            seed=seed + offset,
            history_rows=history_rows,
        )
        modules[variant] = child

    metric_rows = []
    for variant in VARIANTS:
        for split in SPLITS:
            value = evaluate_variant_nll(
                genes_cpu, modules[variant], cis_cache, readout, split=split
            )
            metric_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "compatible_path_nll": value,
                    "screening_evidence_only": screening_evidence_only,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    history = pd.DataFrame(history_rows)
    result = HierarchyResult(
        modules=modules,
        metrics=metrics,
        admission=admission,
        history=history,
    )
    if run_dir is not None:
        run_path = Path(run_dir)
        _write_run(result, config, run_path, genes_cpu)
        from .evaluate import (
            evaluate_hierarchy,
            trained_scale_diagnostics,
            write_evaluation,
        )

        report = evaluate_hierarchy(
            result,
            genes_cpu,
            device=torch_device,
            path_length_prior_weight=float(model_config["path_length_prior_weight"]),
        )
        write_evaluation(report, run_path / "evaluation")
        diagnostic, correlations = trained_scale_diagnostics(
            result, genes_cpu, device=torch_device
        )
        diagnostic.to_csv(run_path / "rms_delta_diagnostic.tsv", sep="\t", index=False)
        correlations.to_csv(
            run_path / "rms_delta_correlations.tsv", sep="\t", index=False
        )
    return result


def train_paired_seeds(
    genes: Sequence[PreparedGene],
    config: Mapping[str, object],
    *,
    device: str | torch.device,
    run_dir: str | Path,
    formal_prepared: PreparedDataset | None = None,
) -> dict[int, HierarchyResult]:
    """Run the one frozen hierarchy independently for every paired F4 seed."""

    seeds = tuple(int(value) for value in config["training"]["seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("paired-seed execution requires at least two unique seeds")
    formal_requested = int(config.get("target_gene_count", 0)) == 7_198 and bool(
        config["training"].get("formal_full7198_authorized", False)
    )
    if formal_requested:
        if formal_prepared is None:
            raise RuntimeError(
                "formal full7198 result labeling requires the validated PreparedDataset"
            )
        assert_full7198_ready(config, formal_prepared)
        if tuple(gene.gene_id for gene in genes) != formal_prepared.target_gene_ids:
            raise RuntimeError("formal training genes differ from the validated bundle")
        genes = formal_prepared.genes
    elif formal_prepared is not None:
        raise ValueError(
            "formal_prepared is only valid for an authorized full7198 configuration"
        )
    root = Path(run_dir)
    results: dict[int, HierarchyResult] = {}
    metric_parts = []
    for seed in seeds:
        seed_config = copy.deepcopy(dict(config))
        seed_training = dict(seed_config["training"])
        seed_training.pop("seeds", None)
        seed_training["seed"] = seed
        seed_config["training"] = seed_training
        result = train_hierarchy(
            genes,
            seed_config,
            device=device,
            run_dir=root / f"seed_{seed}",
            _formal_identity_validated=formal_requested,
        )
        results[seed] = result
        metric_parts.append(result.metrics.assign(seed=seed))
    root.mkdir(parents=True, exist_ok=True)
    pd.concat(metric_parts, ignore_index=True).to_csv(
        root / "paired_seed_metrics.tsv", sep="\t", index=False
    )
    return results


def fit_b0_path_logits(gene: PreparedGene) -> torch.Tensor:
    """Train-only path-frequency baseline with uniform EC allocation and +1."""

    path_count = int(gene.path_edge_incidence.shape[0])
    counts = torch.ones(
        path_count, dtype=torch.float64, device=gene.molecule_count.device
    )
    for row, split in enumerate(gene.split):
        likelihood_informative = int(gene.compatible_path_mask[row].sum()) < path_count
        if split != "train" or not likelihood_informative:
            continue
        indices = gene.compatible_path_indices[row][gene.compatible_path_mask[row]]
        counts[indices] += gene.molecule_count[row].double() / len(indices)
    return (counts / counts.sum()).log().to(dtype=torch.float32)


def evaluate_variant_nll(
    genes: Sequence[PreparedGene],
    modules: VariantModules,
    frozen: Mapping[str, tuple[CISOutput, FrozenAlternativeState]],
    readout: AugmentedPathReadout,
    *,
    split: str,
    identifiable_only: bool = False,
    batch_rows: int = 4096,
) -> float:
    weighted_sum = 0.0
    molecule_mass = 0.0
    with torch.inference_mode():
        for gene in genes:
            rows = _rows_for_split(gene, split, identifiable_only=identifiable_only)
            if rows.numel() == 0:
                continue
            for start in range(0, len(rows), batch_rows):
                details = _loss_for_rows(
                    gene,
                    rows[start : start + batch_rows],
                    modules,
                    frozen.get(gene.gene_id),
                    readout,
                )
                weighted_sum += float(details.weighted_sum)
                molecule_mass += float(details.molecule_mass)
    if molecule_mass == 0:
        raise ValueError(f"split {split} has zero molecule mass")
    return weighted_sum / molecule_mass


def frozen_toy_likelihood_parity_error() -> float:
    """Enumerate fixed singleton and multi-path compatible sets for admission."""

    logits = torch.tensor([[1.2, -0.3, 0.7], [-0.4, 0.8, 0.1]], dtype=torch.float64)
    compatible = torch.tensor([[0, -1], [1, 2], [0, 2]], dtype=torch.long)
    mask = compatible >= 0
    row_cell_index = torch.tensor([0, 0, 1], dtype=torch.long)
    molecule_count = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    details = compatible_path_nll(
        logits,
        compatible,
        mask,
        molecule_count,
        row_cell_index=row_cell_index,
        return_details=True,
    )
    brute = brute_force_compatible_path_nll(
        logits,
        [[0], [1, 2], [0, 2]],
        molecule_count,
        row_cell_index=row_cell_index,
    )
    expected_rows = torch.tensor(
        [0.6041306053367284, 0.7908689178185055, 0.8124753168213709],
        dtype=torch.float64,
    )
    expected_mean = 0.7643244548235828
    return max(
        float((details.per_row_nll - expected_rows).abs().max()),
        abs(float(details.loss - brute)),
        abs(float(details.loss) - expected_mean),
    )


def fixed_real_fixture_parity_error(directory: str | Path) -> float:
    """Recompute the frozen real row-level likelihood reference independently."""

    from .graph import (
        load_graph_tables,
        normalize_compatibility_path_order,
        split_gene_graphs,
    )

    root = Path(directory)
    metadata = json.loads((root / "fixture.json").read_text())
    graphs = list(split_gene_graphs(load_graph_tables(root / "graph_generation")))
    if len(graphs) != 1 or graphs[0].gene_id != str(metadata["gene_id"]):
        raise ValueError("frozen real likelihood fixture graph identity differs")
    graph = graphs[0]
    rows = normalize_compatibility_path_order(
        pd.read_parquet(root / "compatibility_equivalence_classes.parquet"), graph
    )
    width = int(rows["compatible_path_count"].max())
    compatible = torch.full((len(rows), width), -1, dtype=torch.long)
    mask = torch.zeros_like(compatible, dtype=torch.bool)
    for row_index, values in enumerate(rows["compatible_path_indices"]):
        compatible[row_index, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row_index, : len(values)] = True
    cell_codes, unique_cells = pd.factorize(rows["cell_id"].astype(str), sort=True)
    reference = metadata["fixed_logit_likelihood_reference"]
    one_logit = torch.tensor(
        [reference["path_logits"][path_id] for path_id in graph.path_ids],
        dtype=torch.float64,
    )
    logits = one_logit.unsqueeze(0).repeat(len(unique_cells), 1)
    weights = torch.from_numpy(rows["molecule_count"].to_numpy(dtype=np.float64))
    row_cell_index = torch.from_numpy(cell_codes.astype(np.int64))
    details = compatible_path_nll(
        logits,
        compatible,
        mask,
        weights,
        row_cell_index=row_cell_index,
        return_details=True,
    )
    expected_rows = torch.tensor(
        [
            reference["row_nll"][str(values[0])]
            for values in rows["compatible_path_ids"]
        ],
        dtype=torch.float64,
    )
    brute = brute_force_compatible_path_nll(
        logits,
        [list(values) for values in rows["compatible_path_indices"]],
        weights,
        row_cell_index=row_cell_index,
    )
    errors = (
        float((details.per_row_nll - expected_rows).abs().max()),
        abs(float(details.loss - brute)),
        abs(float(details.loss) - float(reference["molecule_weighted_mean_nll"])),
    )
    return max(errors)


def make_toy_genes(*, seed: int = 7) -> list[PreparedGene]:
    """F0 end-to-end fixture with a two-alternative legal-path bubble."""

    generator = torch.Generator().manual_seed(seed)
    from .graph import edge_feature_matrix

    edge_count, cell_count = 7, 30
    edge_specs = (
        ("EXON_CONTINUATION", "TSS", "donor"),
        ("SPLICE", "donor", "acceptor"),
        ("EXON_CONTINUATION", "acceptor", "donor"),
        ("SPLICE", "donor", "acceptor"),
        ("EXON_CONTINUATION", "acceptor", "donor"),
        ("SPLICE", "donor", "acceptor"),
        ("EXON_CONTINUATION", "acceptor", "PAS"),
    )
    edge_rows = pd.DataFrame(
        [
            {
                "edge_type": edge_type,
                "src_node_type": src_type,
                "dst_node_type": dst_type,
                "span_bp": 100 + 10 * index,
                "length_bp": 100 + 10 * index
                if edge_type == "EXON_CONTINUATION"
                else 0,
                "relative_edge_pos": index / (edge_count - 1),
                "annotation_confidence": 1.0,
                "edge_prior_score": 0.0,
            }
            for index, (edge_type, src_type, dst_type) in enumerate(edge_specs)
        ]
    )
    edge_features = torch.from_numpy(edge_feature_matrix(edge_rows))
    path_edges = ((0, 1, 2, 3, 6), (0, 1, 4, 5, 6))
    local_pairs = sorted(
        {
            pair
            for path in path_edges
            for left, right in zip(path[:-1], path[1:])
            for pair in ((left, right), (right, left))
        }
    )
    chain = torch.tensor(local_pairs, dtype=torch.long).T
    graph = GraphGPSBatch(
        edge_features=edge_features,
        local_edge_index=chain.long(),
        edge_gene_index=torch.zeros(edge_count, dtype=torch.long),
    )
    alternatives = AlternativeBatch(
        edge_index=torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        edge_mask=torch.ones((2, 2), dtype=torch.bool),
        choice_index=torch.tensor([0, 0], dtype=torch.long),
        scope_index=torch.tensor([0, 0], dtype=torch.long),
    )
    path_edge = _sparse_tensor(
        torch.tensor(
            [[1, 1, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1]], dtype=torch.float32
        )
    )
    path_choice = _sparse_tensor(torch.eye(2))
    raw_state = torch.randn(cell_count, 4, generator=generator)
    raw_dna_gate = torch.rand(cell_count, 2, generator=generator)
    raw_rna_gate = torch.rand(cell_count, 2, generator=generator)
    train_rows = slice(0, 20)
    state_mean = raw_state[train_rows].mean(dim=0)
    dna_mean = raw_dna_gate[train_rows].mean(dim=0)
    rna_mean = raw_rna_gate[train_rows].mean(dim=0)
    state = raw_state - state_mean
    dna_gate = raw_dna_gate - dna_mean
    rna_gate = raw_rna_gate - rna_mean
    signal = 0.8 + 1.2 * state[:, 0] + 0.9 * dna_gate[:, 0] + 0.7 * rna_gate[:, 0]
    chosen = (signal < 0).long()
    compatible = chosen[:, None]
    split = tuple(["train"] * 20 + ["val"] * 5 + ["test"] * 5)
    dna_features = torch.randn(2, 6, generator=generator)
    rna_features = torch.randn(2, 5, generator=generator)
    relation = torch.eye(2)
    return [
        PreparedGene(
            gene_id="ENSG_TOY",
            graph=graph,
            alternatives=alternatives,
            path_edge_incidence=path_edge,
            path_choice_incidence=path_choice,
            alternative_eligible=torch.ones(2, dtype=torch.bool),
            state_features=state,
            dna_event_features=dna_features,
            dna_event_relation=relation,
            dna_event_choice_index=torch.zeros(2, dtype=torch.long),
            dna_gate=dna_gate,
            rna_event_features=rna_features,
            rna_event_relation=relation,
            rna_event_choice_index=torch.zeros(2, dtype=torch.long),
            rna_gate=rna_gate,
            compatible_path_indices=compatible,
            compatible_path_mask=torch.ones_like(compatible, dtype=torch.bool),
            row_cell_index=torch.arange(cell_count),
            molecule_count=torch.ones(cell_count),
            split=split,
            identifiable_row_mask=torch.ones(cell_count, dtype=torch.bool),
            cell_ids=tuple(f"toy_cell_{index:02d}" for index in range(cell_count)),
            path_ids=("toy_path_0", "toy_path_1"),
            dna_event_ids=("toy_dna_event_0", "toy_dna_event_1"),
            rna_event_ids=("toy_rna_event_0", "toy_rna_event_1"),
            graph_generation="toy",
            split_source="toy",
            dna_event_factor_ids=("toy_factor_0", "toy_factor_1"),
            rna_event_factor_ids=("toy_factor_0", "toy_factor_1"),
            dna_event_peak_ids=("toy_peak_0", "toy_peak_1"),
            state_baseline=PreparedGateBaseline(
                mean=state_mean,
                valid_molecule_mass=torch.full((4,), 20.0),
                weighted_variance=raw_state[train_rows].var(dim=0, unbiased=False),
                eligible=torch.ones(4, dtype=torch.bool),
            ),
            dna_baseline=PreparedGateBaseline(
                mean=dna_mean,
                valid_molecule_mass=torch.full((2,), 20.0),
                weighted_variance=raw_dna_gate[train_rows].var(dim=0, unbiased=False),
                eligible=torch.ones(2, dtype=torch.bool),
                dna_reliability_mass=torch.full((2,), 20.0),
            ),
            rna_baseline=PreparedGateBaseline(
                mean=rna_mean,
                valid_molecule_mass=torch.full((2,), 20.0),
                weighted_variance=raw_rna_gate[train_rows].var(dim=0, unbiased=False),
                eligible=torch.ones(2, dtype=torch.bool),
            ),
            alternative_span=torch.tensor([2.0]),
            dna_candidate_event_count=torch.tensor([2.0]),
            dna_selected_event_count=torch.tensor([2.0]),
            dna_cap_saturated=torch.tensor([0.0]),
            dna_boundary_rank_motif_score=torch.tensor([0.8]),
            rna_candidate_event_count=torch.tensor([2.0]),
            rna_selected_event_count=torch.tensor([2.0]),
            rna_cap_saturated=torch.tensor([0.0]),
            rna_boundary_rank_motif_score=torch.tensor([0.8]),
        )
    ]


def _fit_variant(
    *,
    variant: str,
    genes: Sequence[PreparedGene],
    modules: VariantModules,
    frozen: Mapping[str, tuple[CISOutput, FrozenAlternativeState]] | None,
    readout: AugmentedPathReadout,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    batch_rows: int,
    seed: int,
    history_rows: list[dict[str, object]],
) -> None:
    parameters = _trainable_parameters(variant, modules)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    best_val = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    rng = random.Random(seed)
    total_train_mass = sum(
        float(
            gene.molecule_count[
                _rows_for_split(gene, "train", identifiable_only=False)
            ].sum()
        )
        for gene in genes
    )
    if total_train_mass <= 0:
        raise ValueError("training split has zero molecule mass")
    for epoch in range(max_epochs):
        order = list(range(len(genes)))
        rng.shuffle(order)
        # Accumulating weighted sums before one epoch update is the exact
        # gradient of the contract's global molecule-weighted objective.  A
        # separate per-batch mean would incorrectly equalize low/high-mass ECs.
        optimizer.zero_grad(set_to_none=True)
        for gene_index in order:
            gene = genes[gene_index]
            rows = _rows_for_split(gene, "train", identifiable_only=False).tolist()
            rng.shuffle(rows)
            for start in range(0, len(rows), batch_rows):
                selected = torch.tensor(
                    rows[start : start + batch_rows],
                    dtype=torch.long,
                )
                details = _loss_for_rows(
                    gene,
                    selected,
                    modules,
                    None if frozen is None else frozen[gene.gene_id],
                    readout,
                )
                scaled_loss = details.weighted_sum / total_train_mass
                if scaled_loss.requires_grad:
                    scaled_loss.backward()
        optimizer.step()
        current_frozen = (
            frozen if frozen is not None else _freeze_cis_outputs(modules.cis, genes)
        )
        train_nll = evaluate_variant_nll(
            genes, modules, current_frozen, readout, split="train"
        )
        val_nll = evaluate_variant_nll(
            genes, modules, current_frozen, readout, split="val"
        )
        history_rows.append(
            {
                "variant": variant,
                "epoch": epoch + 1,
                "train_nll": train_nll,
                "val_nll": val_nll,
            }
        )
        if val_nll < best_val:
            best_val = val_nll
            best_state = _module_state(modules)
    if best_state is None:
        raise RuntimeError(f"variant {variant} produced no validation checkpoint")
    _restore_module_state(modules, best_state)


def _loss_for_rows(
    gene: PreparedGene,
    rows: torch.Tensor,
    modules: VariantModules,
    frozen: tuple[CISOutput, FrozenAlternativeState] | None,
    readout: AugmentedPathReadout,
):
    rows = rows.detach().cpu().long()
    device = next(modules.cis.parameters()).device
    row_cells = gene.row_cell_index[rows].cpu()
    cells, inverse = torch.unique(row_cells, sorted=True, return_inverse=True)
    graph = _graph_to_device(gene.graph, device)
    alternatives = _alternatives_to_device(gene.alternatives, device)
    if frozen is None:
        cis_output = modules.cis(graph)
        alternative_state = build_frozen_alternative_state(
            cis_output.edge_states, alternatives
        )
    else:
        cis_output, alternative_state = _frozen_to_device(frozen, device)
    batch_size = len(cells)
    alternative_count = int(gene.path_choice_incidence.shape[1])
    zeros = cis_output.edge_energy.new_zeros((batch_size, alternative_count))
    state_correction = zeros
    if modules.state is not None and alternative_count:
        state_correction = modules.state(
            StateBatch(gene.state_features[cells].to(device)), alternative_state
        ).correction
    eligible = gene.alternative_eligible.to(device=device, dtype=zeros.dtype)[None, :]
    state_correction = state_correction * eligible
    dna_correction = zeros
    if modules.dna is not None and alternative_count:
        dna_correction = (
            modules.dna(
                EventBatch(
                    features=gene.dna_event_features.to(device),
                    relation=gene.dna_event_relation.to(device),
                    event_choice_index=gene.dna_event_choice_index.to(device),
                    gate=gene.dna_gate[cells].to(device),
                ),
                alternative_state,
            ).correction
            * eligible
        )
    rna_correction = zeros
    if modules.rna is not None and alternative_count:
        rna_correction = (
            modules.rna(
                EventBatch(
                    features=gene.rna_event_features.to(device),
                    relation=gene.rna_event_relation.to(device),
                    event_choice_index=gene.rna_event_choice_index.to(device),
                    gate=gene.rna_gate[cells].to(device),
                ),
                alternative_state,
            ).correction
            * eligible
        )
    logits = readout(
        PathReadoutInput(
            edge_energy=cis_output.edge_energy,
            state_correction=state_correction,
            dna_correction=dna_correction,
            rna_correction=rna_correction,
            path_edge_incidence=gene.path_edge_incidence.to(device),
            path_choice_incidence=gene.path_choice_incidence.to(device),
        )
    ).total_logits
    return compatible_path_nll(
        logits,
        gene.compatible_path_indices[rows].to(device),
        gene.compatible_path_mask[rows].to(device),
        gene.molecule_count[rows].to(device),
        row_cell_index=inverse.to(device),
        return_details=True,
    )


def _evaluate_fixed_logits(
    genes: Sequence[PreparedGene], logits: Mapping[str, torch.Tensor], *, split: str
) -> float:
    weighted_sum = 0.0
    mass = 0.0
    for gene in genes:
        rows = _rows_for_split(gene, split, identifiable_only=False)
        for start in range(0, len(rows), 4096):
            selected = rows[start : start + 4096]
            cells, inverse = torch.unique(
                gene.row_cell_index[selected], sorted=True, return_inverse=True
            )
            repeated = logits[gene.gene_id].unsqueeze(0).expand(len(cells), -1)
            details = compatible_path_nll(
                repeated,
                gene.compatible_path_indices[selected],
                gene.compatible_path_mask[selected],
                gene.molecule_count[selected],
                row_cell_index=inverse,
                return_details=True,
            )
            weighted_sum += float(details.weighted_sum)
            mass += float(details.molecule_mass)
    return weighted_sum / mass


def _freeze_cis_outputs(
    cis: EdgeGraphGPS, genes: Sequence[PreparedGene]
) -> dict[str, tuple[CISOutput, FrozenAlternativeState]]:
    result = {}
    # no_grad keeps ordinary tensors that can be consumed by trainable child
    # projections; inference-mode tensors cannot be saved for their backward.
    device = next(cis.parameters()).device
    with torch.no_grad():
        for gene in genes:
            alternatives = _alternatives_to_device(gene.alternatives, device)
            output = cis(_graph_to_device(gene.graph, device))
            detached = CISOutput(
                edge_states=output.edge_states.detach().cpu().clone(),
                edge_energy=output.edge_energy.detach().cpu().clone(),
            )
            result[gene.gene_id] = (
                detached,
                build_frozen_alternative_state(
                    detached.edge_states,
                    _alternatives_to_device(alternatives, torch.device("cpu")),
                ),
            )
    return result


def _rows_for_split(
    gene: PreparedGene, split: str, *, identifiable_only: bool
) -> torch.Tensor:
    mask = torch.tensor(
        [value == split for value in gene.split],
        dtype=torch.bool,
        device=gene.molecule_count.device,
    )
    if identifiable_only:
        mask &= gene.identifiable_row_mask
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _trainable_parameters(
    variant: str, modules: VariantModules
) -> list[torch.nn.Parameter]:
    if variant == "cis":
        parameters = list(modules.cis.parameters())
    elif variant == "state":
        parameters = list(modules.state.parameters())
    elif variant == "state_dna":
        parameters = list(modules.dna.parameters())
    elif variant == "state_rna":
        parameters = list(modules.rna.parameters())
    elif variant == "state_dna_rna":
        parameters = [*modules.dna.parameters(), *modules.rna.parameters()]
    else:
        raise ValueError(f"unknown V1 variant: {variant}")
    return [parameter for parameter in parameters if parameter.requires_grad]


def _module_state(modules: VariantModules) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: copy.deepcopy(module.state_dict())
        for name, module in (
            ("cis", modules.cis),
            ("state", modules.state),
            ("dna", modules.dna),
            ("rna", modules.rna),
        )
        if module is not None
    }


def _restore_module_state(
    modules: VariantModules, states: Mapping[str, Mapping[str, torch.Tensor]]
) -> None:
    for name in ("cis", "state", "dna", "rna"):
        module = getattr(modules, name)
        if module is not None and name in states:
            module.load_state_dict(states[name])


def _validate_prepared_genes(genes: Sequence[PreparedGene]) -> None:
    gene_ids = [gene.gene_id for gene in genes]
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("prepared gene IDs must be unique")
    first = genes[0]
    dimensions = (
        first.graph.edge_features.shape[1],
        first.state_features.shape[1],
        first.dna_event_features.shape[1],
        first.rna_event_features.shape[1],
    )
    for gene in genes:
        if (
            gene.normalized_source_paths != first.normalized_source_paths
            or gene.reviewed_factor_mapping != first.reviewed_factor_mapping
            or gene.atac_donor_eligibility_source != first.atac_donor_eligibility_source
            or gene.peak_support_source != first.peak_support_source
            or gene.preparation_config_source != first.preparation_config_source
            or gene.preparation_values != first.preparation_values
            or gene.state_pca_fit_batch_size != first.state_pca_fit_batch_size
        ):
            raise ValueError("prepared genes do not share preparation identity")
        observed = (
            gene.graph.edge_features.shape[1],
            gene.state_features.shape[1],
            gene.dna_event_features.shape[1],
            gene.rna_event_features.shape[1],
        )
        if observed != dimensions:
            raise ValueError("prepared genes do not share fixed model feature axes")
        if gene.graph.edge_features.device.type != "cpu":
            raise ValueError("prepared gene tensors must remain on CPU before batching")
        row_count = len(gene.split)
        observed_splits = set(gene.split)
        if not observed_splits.issubset(SPLITS):
            raise ValueError(f"gene {gene.gene_id} has an invalid split")
        if not (
            gene.compatible_path_indices.shape[0]
            == gene.compatible_path_mask.shape[0]
            == gene.row_cell_index.numel()
            == gene.molecule_count.numel()
            == gene.identifiable_row_mask.numel()
            == row_count
        ):
            raise ValueError(f"gene {gene.gene_id} EC axes differ")
        path_count = len(gene.path_ids)
        alternative_count = gene.alternatives.choice_index.numel()
        if gene.path_edge_incidence.shape != (
            path_count,
            gene.graph.edge_features.shape[0],
        ) or gene.path_choice_incidence.shape != (path_count, alternative_count):
            raise ValueError(f"gene {gene.gene_id} path incidence axes differ")
        if (
            not gene.path_edge_incidence.is_sparse
            or not gene.path_choice_incidence.is_sparse
        ):
            raise TypeError("path incidence tensors must remain sparse")
        cell_count = gene.state_features.shape[0]
        if (
            len(gene.cell_ids) != cell_count
            or len(set(gene.cell_ids)) != cell_count
            or gene.dna_gate.shape[0] != cell_count
            or gene.rna_gate.shape[0] != cell_count
        ):
            raise ValueError(f"gene {gene.gene_id} cell/context identity axes differ")
        if len(gene.path_ids) != gene.path_edge_incidence.shape[0] or len(
            set(gene.path_ids)
        ) != len(gene.path_ids):
            raise ValueError(f"gene {gene.gene_id} path identity axis differs")
        if (
            len(gene.dna_event_ids) != gene.dna_event_features.shape[0]
            or len(gene.rna_event_ids) != gene.rna_event_features.shape[0]
        ):
            raise ValueError(f"gene {gene.gene_id} event identity axes differ")
        if len(set(gene.dna_event_ids)) != len(gene.dna_event_ids) or len(
            set(gene.rna_event_ids)
        ) != len(gene.rna_event_ids):
            raise ValueError(f"gene {gene.gene_id} event IDs are not unique")
        if gene.dna_gate.shape[1] != len(gene.dna_event_ids) or gene.rna_gate.shape[
            1
        ] != len(gene.rna_event_ids):
            raise ValueError(f"gene {gene.gene_id} event/gate axes differ")
        for label, ids, event_count in (
            ("DNA factor", gene.dna_event_factor_ids, len(gene.dna_event_ids)),
            ("RNA factor", gene.rna_event_factor_ids, len(gene.rna_event_ids)),
            ("DNA peak", gene.dna_event_peak_ids, len(gene.dna_event_ids)),
        ):
            if ids and (len(ids) != event_count or any(not value for value in ids)):
                raise ValueError(f"gene {gene.gene_id} {label} identity axis differs")
        if row_count and (
            int(gene.row_cell_index.min()) < 0
            or int(gene.row_cell_index.max()) >= cell_count
        ):
            raise ValueError(f"gene {gene.gene_id} EC row references an unknown cell")
        if not torch.isfinite(gene.molecule_count).all() or bool(
            (gene.molecule_count <= 0).any()
        ):
            raise ValueError(f"gene {gene.gene_id} EC molecule counts are invalid")
        valid_compatible = gene.compatible_path_indices[gene.compatible_path_mask]
        if not bool(gene.compatible_path_mask.any(dim=1).all()) or bool(
            ((valid_compatible < 0) | (valid_compatible >= path_count)).any()
        ):
            raise ValueError(f"gene {gene.gene_id} compatible path indices are invalid")
        cell_splits: dict[int, set[str]] = {}
        for row, split in enumerate(gene.split):
            cell_splits.setdefault(int(gene.row_cell_index[row]), set()).add(split)
        if any(len(values) != 1 for values in cell_splits.values()):
            raise ValueError(f"gene {gene.gene_id} assigns one cell to multiple splits")
        if not gene.graph_generation or not gene.split_source:
            raise ValueError(f"gene {gene.gene_id} lacks source identities")
        choice_index = gene.alternatives.choice_index.cpu().numpy()
        eligibility = gene.alternative_eligible.cpu().numpy().astype(bool)
        if len(eligibility) != len(choice_index):
            raise ValueError(
                f"gene {gene.gene_id} alternative eligibility axis differs"
            )
        for choice in np.unique(choice_index):
            if len(set(eligibility[choice_index == choice])) != 1:
                raise ValueError(
                    "all alternatives of one choice must share eligibility"
                )
        choice_count = int(choice_index.max()) + 1 if len(choice_index) else 0
        choice_audits = {
            "alternative span": gene.alternative_span,
            "DNA candidate event count": gene.dna_candidate_event_count,
            "DNA selected event count": gene.dna_selected_event_count,
            "DNA cap saturation": gene.dna_cap_saturated,
            "DNA boundary motif score": gene.dna_boundary_rank_motif_score,
            "RNA candidate event count": gene.rna_candidate_event_count,
            "RNA selected event count": gene.rna_selected_event_count,
            "RNA cap saturation": gene.rna_cap_saturated,
            "RNA boundary motif score": gene.rna_boundary_rank_motif_score,
        }
        present_choice_audits = {
            label: values
            for label, values in choice_audits.items()
            if values is not None
        }
        if present_choice_audits and len(present_choice_audits) != len(choice_audits):
            raise ValueError(f"gene {gene.gene_id} has an incomplete choice audit")
        if any(
            values.shape != (choice_count,) for values in present_choice_audits.values()
        ):
            raise ValueError(f"gene {gene.gene_id} choice audit axes differ")
        train_informative = any(
            split == "train" and int(gene.compatible_path_mask[row].sum()) < path_count
            for row, split in enumerate(gene.split)
        )
        if not train_informative and bool(eligibility.any()):
            raise ValueError(
                f"gene {gene.gene_id} has eligible choices without train path "
                "supervision"
            )
        for label, features, relation, event_choice, gate in (
            (
                "DNA",
                gene.dna_event_features,
                gene.dna_event_relation,
                gene.dna_event_choice_index,
                gene.dna_gate,
            ),
            (
                "RNA",
                gene.rna_event_features,
                gene.rna_event_relation,
                gene.rna_event_choice_index,
                gene.rna_gate,
            ),
        ):
            event_count = features.shape[0]
            if (
                relation.shape != (event_count, alternative_count)
                or gate.shape
                != (
                    cell_count,
                    event_count,
                )
                or event_choice.shape != (event_count,)
            ):
                raise ValueError(
                    f"gene {gene.gene_id} {label} event tensor axes differ"
                )
            if event_count and (
                alternative_count == 0
                or bool(
                    (
                        (event_choice < 0) | (event_choice > int(choice_index.max()))
                    ).any()
                )
            ):
                raise ValueError(f"gene {gene.gene_id} {label} event choice is invalid")
        for label, baseline, width, expect_reliability in (
            ("State", gene.state_baseline, gene.state_features.shape[1], False),
            ("DNA", gene.dna_baseline, len(gene.dna_event_ids), True),
            ("RNA", gene.rna_baseline, len(gene.rna_event_ids), False),
        ):
            if baseline is None:
                continue
            fields = (
                baseline.mean,
                baseline.valid_molecule_mass,
                baseline.weighted_variance,
                baseline.eligible,
            )
            if any(values.shape != (width,) for values in fields):
                raise ValueError(f"gene {gene.gene_id} {label} baseline axis differs")
            reliability = baseline.dna_reliability_mass
            if expect_reliability != (reliability is not None):
                raise ValueError(
                    f"gene {gene.gene_id} {label} reliability audit differs"
                )
            if reliability is not None and reliability.shape != (width,):
                raise ValueError(
                    f"gene {gene.gene_id} {label} reliability audit axis differs"
                )
            eligible_keys = baseline.eligible.to(dtype=torch.bool)
            if bool(eligible_keys.any()) and (
                not torch.isfinite(baseline.mean[eligible_keys]).all()
                or not torch.isfinite(baseline.weighted_variance[eligible_keys]).all()
                or bool((baseline.valid_molecule_mass[eligible_keys] <= 0).any())
            ):
                raise ValueError(
                    f"gene {gene.gene_id} {label} eligible baseline statistics are invalid"
                )


def _graph_to_device(graph: GraphGPSBatch, device: torch.device) -> GraphGPSBatch:
    return GraphGPSBatch(
        edge_features=graph.edge_features.to(device),
        local_edge_index=graph.local_edge_index.to(device),
        edge_gene_index=graph.edge_gene_index.to(device),
    )


def _alternatives_to_device(
    alternatives: AlternativeBatch, device: torch.device
) -> AlternativeBatch:
    return AlternativeBatch(
        edge_index=alternatives.edge_index.to(device),
        edge_mask=alternatives.edge_mask.to(device),
        choice_index=alternatives.choice_index.to(device),
        scope_index=alternatives.scope_index.to(device),
    )


def _frozen_to_device(
    frozen: tuple[CISOutput, FrozenAlternativeState], device: torch.device
) -> tuple[CISOutput, FrozenAlternativeState]:
    cis, alternatives = frozen
    return (
        CISOutput(
            edge_states=cis.edge_states.to(device),
            edge_energy=cis.edge_energy.to(device),
        ),
        FrozenAlternativeState(
            edge_states=alternatives.edge_states.to(device),
            h_base=alternatives.h_base.to(device),
            choice_index=alternatives.choice_index.to(device),
            scope_index=alternatives.scope_index.to(device),
        ),
    )


def _gene_to_cpu(gene: PreparedGene) -> PreparedGene:
    def baseline_to_cpu(
        baseline: PreparedGateBaseline | None,
    ) -> PreparedGateBaseline | None:
        if baseline is None:
            return None
        return PreparedGateBaseline(
            mean=baseline.mean.cpu(),
            valid_molecule_mass=baseline.valid_molecule_mass.cpu(),
            weighted_variance=baseline.weighted_variance.cpu(),
            eligible=baseline.eligible.cpu(),
            dna_reliability_mass=(
                None
                if baseline.dna_reliability_mass is None
                else baseline.dna_reliability_mass.cpu()
            ),
        )

    def optional_cpu(values: torch.Tensor | None) -> torch.Tensor | None:
        return None if values is None else values.cpu()

    return replace(
        gene,
        graph=GraphGPSBatch(
            edge_features=gene.graph.edge_features.cpu(),
            local_edge_index=gene.graph.local_edge_index.cpu(),
            edge_gene_index=gene.graph.edge_gene_index.cpu(),
        ),
        alternatives=_alternatives_to_device(gene.alternatives, torch.device("cpu")),
        path_edge_incidence=gene.path_edge_incidence.cpu(),
        path_choice_incidence=gene.path_choice_incidence.cpu(),
        alternative_eligible=gene.alternative_eligible.cpu(),
        state_features=gene.state_features.cpu(),
        dna_event_features=gene.dna_event_features.cpu(),
        dna_event_relation=gene.dna_event_relation.cpu(),
        dna_event_choice_index=gene.dna_event_choice_index.cpu(),
        dna_gate=gene.dna_gate.cpu(),
        rna_event_features=gene.rna_event_features.cpu(),
        rna_event_relation=gene.rna_event_relation.cpu(),
        rna_event_choice_index=gene.rna_event_choice_index.cpu(),
        rna_gate=gene.rna_gate.cpu(),
        compatible_path_indices=gene.compatible_path_indices.cpu(),
        compatible_path_mask=gene.compatible_path_mask.cpu(),
        row_cell_index=gene.row_cell_index.cpu(),
        molecule_count=gene.molecule_count.cpu(),
        identifiable_row_mask=gene.identifiable_row_mask.cpu(),
        state_baseline=baseline_to_cpu(gene.state_baseline),
        dna_baseline=baseline_to_cpu(gene.dna_baseline),
        rna_baseline=baseline_to_cpu(gene.rna_baseline),
        alternative_span=optional_cpu(gene.alternative_span),
        dna_candidate_event_count=optional_cpu(gene.dna_candidate_event_count),
        dna_selected_event_count=optional_cpu(gene.dna_selected_event_count),
        dna_cap_saturated=optional_cpu(gene.dna_cap_saturated),
        dna_boundary_rank_motif_score=optional_cpu(gene.dna_boundary_rank_motif_score),
        rna_candidate_event_count=optional_cpu(gene.rna_candidate_event_count),
        rna_selected_event_count=optional_cpu(gene.rna_selected_event_count),
        rna_cap_saturated=optional_cpu(gene.rna_cap_saturated),
        rna_boundary_rank_motif_score=optional_cpu(gene.rna_boundary_rank_motif_score),
    )


def _sparse_tensor(dense: torch.Tensor) -> torch.Tensor:
    return dense.to_sparse_coo().coalesce()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_run(
    result: HierarchyResult,
    config: Mapping[str, object],
    run_dir: Path,
    genes: Sequence[PreparedGene],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    (run_dir / "admission.json").write_text(
        json.dumps(result.admission, indent=2, sort_keys=True)
    )
    result.metrics.to_csv(run_dir / "metrics.tsv", sep="\t", index=False)
    result.history.to_csv(run_dir / "history.tsv", sep="\t", index=False)
    identity = {
        "contract": "FABRIC_ARCHITECTURE_V1",
        "gene_ids": [gene.gene_id for gene in genes],
        "graph_generations": sorted({gene.graph_generation for gene in genes}),
        "split_sources": sorted({gene.split_source for gene in genes}),
        "normalized_source_paths": dict(genes[0].normalized_source_paths),
        "reviewed_factor_mapping": genes[0].reviewed_factor_mapping,
        "atac_donor_eligibility_source": genes[0].atac_donor_eligibility_source,
        "peak_support_source": genes[0].peak_support_source,
        "preparation_config_source": genes[0].preparation_config_source,
        "preparation_values": dict(genes[0].preparation_values),
        "state_pca_fit_batch_size": genes[0].state_pca_fit_batch_size,
        "counts": {
            "genes": len(genes),
            "cells_summed_over_genes": sum(len(gene.cell_ids) for gene in genes),
            "legal_paths": sum(len(gene.path_ids) for gene in genes),
            "ec_rows": sum(len(gene.split) for gene in genes),
            "dna_events": sum(len(gene.dna_event_ids) for gene in genes),
            "rna_events": sum(len(gene.rna_event_ids) for gene in genes),
        },
    }
    (run_dir / "input_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True)
    )
    for variant, modules in result.modules.items():
        torch.save(_module_state(modules), run_dir / f"{variant}.pt")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the fixed FABRIC V1 hierarchy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true")
    source.add_argument("--prepared-bundle")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    formal_full7198_run = int(config.get("target_gene_count", 0)) == 7_198
    if formal_full7198_run:
        assert_full7198_ready(config)
    if args.toy:
        genes = make_toy_genes()
        prepared = None
    else:
        prepared = torch.load(
            args.prepared_bundle, map_location="cpu", weights_only=False
        )
        if not isinstance(prepared, PreparedDataset):
            raise TypeError("prepared bundle must be one FABRIC PreparedDataset")
        genes = prepared.genes
    if formal_full7198_run:
        if prepared is None:
            raise RuntimeError("formal full7198 training requires a prepared bundle")
        assert_full7198_ready(config, prepared)
    if "seeds" in config["training"]:
        train_paired_seeds(
            genes,
            config,
            device=args.device,
            run_dir=args.run_dir,
            formal_prepared=prepared if formal_full7198_run else None,
        )
    elif formal_full7198_run:
        raise RuntimeError("formal full7198 training requires paired seeds")
    else:
        train_hierarchy(genes, config, device=args.device, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
