"""FABRIC V1 primary metrics, fixed null, and logit-level diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .model import EventBatch, PathLogits
from .train import (
    HierarchyResult,
    PreparedGene,
    PreparedGateBaseline,
    PreparedDataset,
    VARIANTS,
    evaluate_variant_nll,
    train_hierarchy,
)
from .dataset import (
    FactorActivityContext,
    apply_centering,
    apply_dna_centering,
    fit_centering_baseline,
    fit_dna_centering_baseline,
    validate_prepared_external_context,
)


@dataclass(frozen=True)
class EvaluationReport:
    primary_metrics: pd.DataFrame
    paired_deltas: pd.DataFrame
    coverage: pd.DataFrame


@dataclass(frozen=True)
class GeneNullContext:
    """Gene/event alignment needed to rebuild gates after factor permutation."""

    gene_id: str
    cell_ids: tuple[str, ...]
    dna_event_factor_index: np.ndarray
    rna_event_factor_index: np.ndarray
    dna_accessibility: np.ndarray
    dna_accessibility_observed: np.ndarray
    dna_reliability: np.ndarray


def null_contexts_from_prepared(
    prepared: PreparedDataset, *, gene_ids: Sequence[str] | None = None
) -> tuple[FactorActivityContext, tuple[GeneNullContext, ...]]:
    """Recover the one fixed-null context from a normalized prepared bundle."""

    validate_prepared_external_context(prepared)
    factor = prepared.factor_context
    atac = prepared.atac_context
    assert factor is not None and atac is not None
    selected_ids = (
        prepared.target_gene_ids if gene_ids is None else tuple(map(str, gene_ids))
    )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("fixed null panel gene IDs must be unique")
    by_gene = {gene.gene_id: gene for gene in prepared.genes}
    missing = [gene_id for gene_id in selected_ids if gene_id not in by_gene]
    if missing:
        raise ValueError(
            f"fixed null panel genes are absent from bundle: {missing[:10]}"
        )
    cell_index = {cell_id: index for index, cell_id in enumerate(factor.cell_ids)}
    factor_index = {
        factor_id: index for index, factor_id in enumerate(factor.factor_ids)
    }
    peak_index = {peak_id: index for index, peak_id in enumerate(atac.peak_ids)}
    contexts: list[GeneNullContext] = []
    for gene_id in selected_ids:
        gene = by_gene[gene_id]
        cell_rows = np.asarray(
            [cell_index[value] for value in gene.cell_ids], dtype=np.int64
        )
        peak_columns = np.asarray(
            [peak_index[value] for value in gene.dna_event_peak_ids], dtype=np.int64
        )
        accessibility = atac.accessibility[cell_rows][:, peak_columns].toarray()
        observed = np.repeat(
            np.asarray(atac.observed, dtype=bool)[cell_rows, None],
            len(peak_columns),
            axis=1,
        )
        reliability = np.repeat(
            np.asarray(atac.reliability, dtype=np.float32)[cell_rows, None],
            len(peak_columns),
            axis=1,
        )
        contexts.append(
            GeneNullContext(
                gene_id=gene_id,
                cell_ids=gene.cell_ids,
                dna_event_factor_index=np.asarray(
                    [factor_index[value] for value in gene.dna_event_factor_ids],
                    dtype=np.int64,
                ),
                rna_event_factor_index=np.asarray(
                    [factor_index[value] for value in gene.rna_event_factor_ids],
                    dtype=np.int64,
                ),
                dna_accessibility=accessibility,
                dna_accessibility_observed=observed,
                dna_reliability=reliability,
            )
        )
    return factor, tuple(contexts)


def evaluate_hierarchy(
    result: HierarchyResult,
    genes: Sequence[PreparedGene],
    *,
    device: str | torch.device,
    path_length_prior_weight: float,
) -> EvaluationReport:
    """Report the two fixed scopes and paired deltas to the same parent."""

    torch_device = torch.device(device)
    genes_cpu = tuple(gene for gene in genes)
    cis = result.modules["cis"].cis
    from .train import (
        _freeze_cis_outputs,
    )  # one scientific implementation, not a compatibility layer
    from .model import AugmentedPathReadout

    frozen = _freeze_cis_outputs(cis, genes_cpu)
    readout = AugmentedPathReadout(length_penalty=path_length_prior_weight).to(
        torch_device
    )
    screening_values = set(result.metrics["screening_evidence_only"])
    if screening_values not in ({True}, {False}):
        raise ValueError("hierarchy metrics contain mixed screening semantics")
    screening_evidence_only = screening_values.pop()
    primary_rows: list[dict[str, object]] = []
    for scope, identifiable_only in (
        ("all_eligible_genes", False),
        ("choice_supervision_identifiable", True),
    ):
        for split in ("val", "test"):
            for variant in VARIANTS:
                nll = evaluate_variant_nll(
                    genes_cpu,
                    result.modules[variant],
                    frozen,
                    readout,
                    split=split,
                    identifiable_only=identifiable_only,
                )
                primary_rows.append(
                    {
                        "scope": scope,
                        "split": split,
                        "variant": variant,
                        "compatible_path_nll": nll,
                        "screening_evidence_only": screening_evidence_only,
                    }
                )
    primary = pd.DataFrame(primary_rows)
    comparisons = {
        "state": "cis",
        "state_dna": "state",
        "state_rna": "state",
        "state_dna_rna": "state",
    }
    delta_rows = []
    for (scope, split), group in primary.groupby(["scope", "split"], sort=True):
        nll = group.set_index("variant")["compatible_path_nll"]
        for child, parent in comparisons.items():
            delta_rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "child": child,
                    "parent": parent,
                    "paired_delta_nll_child_minus_parent": float(
                        nll[child] - nll[parent]
                    ),
                }
            )
    coverage_rows = []
    for split in ("val", "test"):
        total = 0.0
        identifiable = 0.0
        for gene in genes_cpu:
            split_mask = np.asarray([value == split for value in gene.split])
            weights = gene.molecule_count.detach().cpu().numpy()
            eligible = gene.identifiable_row_mask.detach().cpu().numpy().astype(bool)
            total += float(weights[split_mask].sum())
            identifiable += float(weights[split_mask & eligible].sum())
        coverage_rows.append(
            {
                "split": split,
                "supervision_identifiable_molecule_mass": identifiable,
                "all_molecule_mass": total,
                "supervision_identifiable_molecule_coverage": identifiable / total,
            }
        )
    return EvaluationReport(
        primary_metrics=primary,
        paired_deltas=pd.DataFrame(delta_rows),
        coverage=pd.DataFrame(coverage_rows),
    )


def permute_factor_activity_within_strata(
    activity: np.ndarray,
    stage: Sequence[str],
    developmental_system: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The single formal null: one joint cell permutation per stage×system stratum."""

    values = np.asarray(activity)
    stage_values = np.asarray(stage, dtype=str)
    system_values = np.asarray(developmental_system, dtype=str)
    if (
        values.ndim != 2
        or len(stage_values) != len(values)
        or len(system_values) != len(values)
    ):
        raise ValueError("activity and null-stratum cell axes differ")
    rng = np.random.default_rng(seed)
    source_index = np.arange(len(values), dtype=np.int64)
    strata = pd.DataFrame({"stage": stage_values, "system": system_values})
    for _, indices in strata.groupby(["stage", "system"], sort=True).groups.items():
        positions = np.asarray(sorted(indices), dtype=np.int64)
        source_index[positions] = rng.permutation(positions)
    return values[source_index].copy(), source_index


def rebuild_stage_system_factor_null(
    genes: Sequence[PreparedGene],
    factor_context: FactorActivityContext,
    gene_contexts: Sequence[GeneNullContext],
    *,
    seed: int,
    minimum_valid_molecule_mass: float,
    minimum_weighted_variance: float,
) -> tuple[tuple[PreparedGene, ...], np.ndarray]:
    """Permute factor rows once, then refit every train-only RNA/DNA baseline."""

    global_ids = tuple(str(value) for value in factor_context.cell_ids)
    if len(set(global_ids)) != len(global_ids):
        raise ValueError("formal-null global cell IDs must be unique")
    activity = np.asarray(factor_context.activity, dtype=np.float64)
    observed = np.asarray(factor_context.observed, dtype=bool)
    if activity.shape != observed.shape or activity.shape != (
        len(global_ids),
        len(factor_context.factor_ids),
    ):
        raise ValueError("formal-null factor activity/identity axes differ")
    if not np.isfinite(activity[observed]).all() or bool(
        (activity[observed] < 0).any()
    ):
        raise ValueError(
            "formal-null observed factor activity must be finite and non-negative"
        )
    permuted, source_index = permute_factor_activity_within_strata(
        activity,
        factor_context.stage,
        factor_context.developmental_system,
        seed=seed,
    )
    permuted_observed = observed[source_index]
    global_index = {value: index for index, value in enumerate(global_ids)}
    context_by_gene = {value.gene_id: value for value in gene_contexts}
    if len(context_by_gene) != len(gene_contexts):
        raise ValueError("formal-null gene contexts are not unique")

    rebuilt: list[PreparedGene] = []
    for gene in genes:
        if gene.gene_id not in context_by_gene:
            raise ValueError(f"formal-null context is absent for gene {gene.gene_id}")
        context = context_by_gene[gene.gene_id]
        if tuple(context.cell_ids) != tuple(gene.cell_ids):
            raise ValueError(f"formal-null cell order differs for gene {gene.gene_id}")
        missing = [value for value in gene.cell_ids if value not in global_index]
        if missing:
            raise ValueError(
                f"formal-null cells are absent from factor context: {missing[:10]}"
            )
        cell_rows = np.asarray(
            [global_index[value] for value in gene.cell_ids], dtype=np.int64
        )
        dna_factor_index = np.asarray(context.dna_event_factor_index, dtype=np.int64)
        rna_factor_index = np.asarray(context.rna_event_factor_index, dtype=np.int64)
        if len(dna_factor_index) != len(gene.dna_event_ids) or len(
            rna_factor_index
        ) != len(gene.rna_event_ids):
            raise ValueError(
                f"formal-null event/factor axes differ for gene {gene.gene_id}"
            )
        if bool(
            ((dna_factor_index < 0) | (dna_factor_index >= activity.shape[1])).any()
        ) or bool(
            ((rna_factor_index < 0) | (rna_factor_index >= activity.shape[1])).any()
        ):
            raise IndexError("formal-null event factor index is out of range")

        cell_mass = _train_likelihood_informative_cell_mass(gene)
        rna_values = permuted[cell_rows][:, rna_factor_index]
        rna_observed = permuted_observed[cell_rows][:, rna_factor_index]
        rna_baseline = fit_centering_baseline(
            rna_values,
            rna_observed,
            cell_mass,
            minimum_valid_mass=minimum_valid_molecule_mass,
            minimum_variance=minimum_weighted_variance,
        )
        rna_gate = apply_centering(rna_values, rna_observed, rna_baseline)

        dna_values = permuted[cell_rows][:, dna_factor_index]
        dna_factor_observed = permuted_observed[cell_rows][:, dna_factor_index]
        accessibility = np.asarray(context.dna_accessibility, dtype=np.float64)
        accessibility_observed = np.asarray(
            context.dna_accessibility_observed, dtype=bool
        )
        reliability = np.asarray(context.dna_reliability, dtype=np.float64)
        expected_dna_shape = (len(gene.cell_ids), len(gene.dna_event_ids))
        if not (
            accessibility.shape
            == accessibility_observed.shape
            == reliability.shape
            == expected_dna_shape
        ):
            raise ValueError(
                f"formal-null DNA context axes differ for gene {gene.gene_id}"
            )
        dna_baseline = fit_dna_centering_baseline(
            dna_values,
            dna_factor_observed,
            accessibility,
            accessibility_observed,
            reliability,
            cell_mass,
            minimum_valid_mass=minimum_valid_molecule_mass,
            minimum_variance=minimum_weighted_variance,
        )
        dna_gate = apply_dna_centering(
            dna_values,
            dna_factor_observed,
            accessibility,
            accessibility_observed,
            reliability,
            dna_baseline,
        )
        rebuilt.append(
            replace(
                gene,
                dna_gate=torch.from_numpy(dna_gate.astype(np.float32)),
                rna_gate=torch.from_numpy(rna_gate.astype(np.float32)),
                dna_baseline=_prepared_baseline(dna_baseline),
                rna_baseline=_prepared_baseline(rna_baseline),
            )
        )
    if set(context_by_gene) != {gene.gene_id for gene in genes}:
        raise ValueError("formal-null contexts contain genes outside the fixed panel")
    return tuple(rebuilt), source_index


def _train_likelihood_informative_cell_mass(gene: PreparedGene) -> np.ndarray:
    """Sum train EC mass per cell, excluding only all-path likelihood rows."""

    path_count = len(gene.path_ids)
    compatible_count = gene.compatible_path_mask.sum(dim=1).detach().cpu().numpy()
    train = np.asarray([value == "train" for value in gene.split], dtype=bool)
    informative = train & (compatible_count < path_count)
    row_cell_index = gene.row_cell_index.detach().cpu().numpy()
    molecule_count = gene.molecule_count.detach().cpu().numpy().astype(np.float64)
    cell_mass = np.zeros(len(gene.cell_ids), dtype=np.float64)
    np.add.at(
        cell_mass,
        row_cell_index[informative],
        molecule_count[informative],
    )
    return cell_mass


def _prepared_baseline(baseline) -> PreparedGateBaseline:
    return PreparedGateBaseline(
        mean=torch.from_numpy(np.asarray(baseline.mean, dtype=np.float32)),
        valid_molecule_mass=torch.from_numpy(
            np.asarray(baseline.valid_molecule_mass, dtype=np.float64)
        ),
        weighted_variance=torch.from_numpy(
            np.asarray(baseline.weighted_variance, dtype=np.float64)
        ),
        eligible=torch.from_numpy(np.asarray(baseline.eligible, dtype=bool)),
        dna_reliability_mass=(
            None
            if not hasattr(baseline, "dna_reliability_mass")
            else torch.from_numpy(
                np.asarray(baseline.dna_reliability_mass, dtype=np.float64)
            )
        ),
    )


def retrain_stage_system_factor_null(
    prepared: PreparedDataset,
    config: Mapping[str, object],
    *,
    seed: int,
    device: str | torch.device,
    run_dir: str | Path,
) -> HierarchyResult:
    """Rebuild and retrain the one pre-frozen stage-by-system factor null."""

    panel_config = config.get("diagnostic_panel")
    if not isinstance(panel_config, Mapping):
        raise ValueError("fixed null requires diagnostic_panel.frozen_gene_ids")
    panel = panel_config.get("frozen_gene_ids")
    if not isinstance(panel, Sequence) or isinstance(panel, (str, bytes)) or not panel:
        raise ValueError("fixed null panel gene IDs are unresolved")
    panel_ids = tuple(map(str, panel))
    factor_context, gene_contexts = null_contexts_from_prepared(
        prepared, gene_ids=panel_ids
    )
    genes_by_id = {gene.gene_id: gene for gene in prepared.genes}
    genes = tuple(genes_by_id[gene_id] for gene_id in panel_ids)
    gates = config["gates"]
    rebuilt, source_index = rebuild_stage_system_factor_null(
        genes,
        factor_context,
        gene_contexts,
        seed=seed,
        minimum_valid_molecule_mass=float(gates["minimum_valid_molecule_mass"]),
        minimum_weighted_variance=float(gates["minimum_weighted_variance"]),
    )
    null_config = dict(config)
    null_training = dict(config["training"])
    null_training["seed"] = int(seed)
    null_config["training"] = null_training
    result = train_hierarchy(rebuilt, null_config, device=device, run_dir=run_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"cell_id": factor_context.cell_ids, "permuted_source_row": source_index}
    ).to_csv(Path(run_dir) / "factor_permutation.tsv", sep="\t", index=False)
    return result


def mask_event(event_batch: EventBatch, event_index: int) -> EventBatch:
    """Set one centered gate to zero for counterfactual re-forward."""

    if event_index < 0 or event_index >= event_batch.gate.shape[1]:
        raise IndexError("event index is out of range")
    gate = event_batch.gate.clone()
    gate[:, event_index] = 0
    return EventBatch(
        features=event_batch.features,
        relation=event_batch.relation,
        event_choice_index=event_batch.event_choice_index,
        gate=gate,
    )


def probability_delta(
    with_event_logits: torch.Tensor,
    without_event_logits: torch.Tensor,
) -> torch.Tensor:
    """Probability attribution is defined only by re-forwarded path logits."""

    if with_event_logits.shape != without_event_logits.shape:
        raise ValueError("counterfactual path-logit shapes differ")
    return torch.softmax(with_event_logits, dim=1) - torch.softmax(
        without_event_logits, dim=1
    )


def branch_logit_table(path_logits: PathLogits) -> pd.DataFrame:
    """Flatten the exact additive logit decomposition for reporting."""

    components = {
        "edge": path_logits.edge_logits,
        "state": path_logits.state_logits,
        "dna": path_logits.dna_logits,
        "rna": path_logits.rna_logits,
        "length": path_logits.length_logits,
        "total": path_logits.total_logits,
    }
    rebuilt = sum(
        components[name] for name in ("edge", "state", "dna", "rna", "length")
    )
    torch.testing.assert_close(rebuilt, path_logits.total_logits, atol=1e-6, rtol=1e-6)
    rows = []
    for component, values in components.items():
        array = values.detach().cpu().numpy()
        for cell_index, path_index in np.ndindex(array.shape):
            rows.append(
                {
                    "cell_index": cell_index,
                    "path_index": path_index,
                    "component": component,
                    "logit": float(array[cell_index, path_index]),
                }
            )
    return pd.DataFrame(rows)


def correction_scale_diagnostic(
    correction: np.ndarray,
    alternative_choice_index: Sequence[int],
    *,
    event_count: Sequence[float],
    alternative_span: Sequence[float],
    cap_saturated: Sequence[float],
    audited_choice_indices: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F3 RMS(delta) audit; it reports scale dependence but never normalizes it."""

    values = np.asarray(correction, dtype=np.float64)
    choice_index = np.asarray(alternative_choice_index, dtype=np.int64)
    choice_count = int(choice_index.max()) + 1 if len(choice_index) else 0
    covariates = {
        "event_count": np.asarray(event_count, dtype=np.float64),
        "alternative_span": np.asarray(alternative_span, dtype=np.float64),
        "cap_saturated": np.asarray(cap_saturated, dtype=np.float64),
    }
    if values.ndim != 2 or values.shape[1] != len(choice_index):
        raise ValueError("correction and alternative-choice axes differ")
    if any(len(value) != choice_count for value in covariates.values()):
        raise ValueError("diagnostic covariates must have one value per choice")
    observed_choices = np.unique(choice_index)
    audited_choices = (
        observed_choices
        if audited_choice_indices is None
        else np.asarray(audited_choice_indices, dtype=np.int64)
    )
    if len(np.unique(audited_choices)) != len(audited_choices) or not set(
        audited_choices.tolist()
    ).issubset(observed_choices.tolist()):
        raise ValueError("audited choices must be unique observed choice indices")
    rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(values[:, choice_index == choice])))
            for choice in audited_choices
        ]
    )
    table = pd.DataFrame(
        {
            "choice_index": audited_choices,
            "rms_delta": rms,
            **{
                name: values_by_choice[audited_choices]
                for name, values_by_choice in covariates.items()
            },
        }
    )
    correlations = []
    for name in covariates:
        values_by_choice = table[name].to_numpy()
        statistic, pvalue = spearmanr(rms, values_by_choice)
        correlations.append(
            {
                "covariate": name,
                "spearman_r": float(statistic),
                "pvalue": float(pvalue),
            }
        )
    return table, pd.DataFrame(correlations)


def trained_scale_diagnostics(
    result: HierarchyResult,
    genes: Sequence[PreparedGene],
    *,
    device: str | torch.device,
    cell_batch_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed F3 RMS-vs-count/span/cap audit without rescaling logits."""

    from .train import _freeze_cis_outputs, _frozen_to_device

    torch_device = torch.device(device)
    audited_genes = tuple(gene for gene in genes if gene.cell_ids)
    frozen = _freeze_cis_outputs(result.modules["cis"].cis, audited_genes)
    diagnostic_parts: list[pd.DataFrame] = []
    correlation_parts: list[pd.DataFrame] = []
    for gene in audited_genes:
        choice_index = gene.alternatives.choice_index.cpu().numpy()
        if not len(choice_index):
            continue
        choice_count = int(choice_index.max()) + 1
        alternative_eligible = gene.alternative_eligible.cpu().numpy().astype(bool)
        if len(alternative_eligible) != len(choice_index):
            raise ValueError("F3 scale audit alternative eligibility axis differs")
        eligible_choices: list[int] = []
        for choice in np.unique(choice_index):
            choice_eligibility = np.unique(alternative_eligible[choice_index == choice])
            if len(choice_eligibility) != 1:
                raise ValueError(
                    "F3 scale audit eligibility differs within choice "
                    f"{choice} for gene {gene.gene_id}"
                )
            if bool(choice_eligibility[0]):
                eligible_choices.append(int(choice))
        if not eligible_choices:
            continue
        if gene.alternative_span is None:
            raise ValueError(
                f"F3 scale audit metadata is absent for prepared gene {gene.gene_id}"
            )
        alternative_span = gene.alternative_span.cpu().numpy()
        if len(alternative_span) != choice_count:
            raise ValueError("F3 scale audit choice covariate axes differ")
        _, alternatives = _frozen_to_device(frozen[gene.gene_id], torch_device)
        for (
            variant,
            modality,
            module,
            features,
            relation,
            event_choice,
            gate,
            selected_event_count,
            cap_saturated,
        ) in (
            (
                "state_dna",
                "DNA",
                result.modules["state_dna"].dna,
                gene.dna_event_features,
                gene.dna_event_relation,
                gene.dna_event_choice_index,
                gene.dna_gate,
                gene.dna_selected_event_count,
                gene.dna_cap_saturated,
            ),
            (
                "state_rna",
                "RNA",
                result.modules["state_rna"].rna,
                gene.rna_event_features,
                gene.rna_event_relation,
                gene.rna_event_choice_index,
                gene.rna_gate,
                gene.rna_selected_event_count,
                gene.rna_cap_saturated,
            ),
            (
                "state_dna_rna",
                "DNA",
                result.modules["state_dna_rna"].dna,
                gene.dna_event_features,
                gene.dna_event_relation,
                gene.dna_event_choice_index,
                gene.dna_gate,
                gene.dna_selected_event_count,
                gene.dna_cap_saturated,
            ),
            (
                "state_dna_rna",
                "RNA",
                result.modules["state_dna_rna"].rna,
                gene.rna_event_features,
                gene.rna_event_relation,
                gene.rna_event_choice_index,
                gene.rna_gate,
                gene.rna_selected_event_count,
                gene.rna_cap_saturated,
            ),
        ):
            if module is None or features.shape[0] == 0:
                continue
            if selected_event_count is None or cap_saturated is None:
                raise ValueError(
                    f"F3 scale audit {modality} metadata is absent for prepared "
                    f"gene {gene.gene_id}"
                )
            event_count = selected_event_count.cpu().numpy()
            cap_saturated_values = cap_saturated.cpu().numpy()
            if (
                len(event_count) != choice_count
                or len(cap_saturated_values) != choice_count
            ):
                raise ValueError("F3 scale audit choice covariate axes differ")
            observed_event_count = np.bincount(
                event_choice.cpu().numpy(), minlength=choice_count
            ).astype(np.float64)
            if not np.array_equal(event_count, observed_event_count):
                raise ValueError(
                    f"F3 scale audit {modality} selected event count differs from "
                    f"event-choice rows for gene {gene.gene_id}"
                )
            correction_parts = []
            with torch.inference_mode():
                for start in range(0, len(gene.cell_ids), cell_batch_size):
                    stop = min(start + cell_batch_size, len(gene.cell_ids))
                    output = module(
                        EventBatch(
                            features=features.to(torch_device),
                            relation=relation.to(torch_device),
                            event_choice_index=event_choice.to(torch_device),
                            gate=gate[start:stop].to(torch_device),
                        ),
                        alternatives,
                    )
                    correction_parts.append(output.correction.cpu().numpy())
            values, correlations = correction_scale_diagnostic(
                np.concatenate(correction_parts, axis=0),
                choice_index,
                event_count=event_count,
                alternative_span=alternative_span,
                cap_saturated=cap_saturated_values,
                audited_choice_indices=eligible_choices,
            )
            diagnostic_parts.append(
                values.assign(gene_id=gene.gene_id, variant=variant, modality=modality)
            )
            correlation_parts.append(
                correlations.assign(
                    gene_id=gene.gene_id, variant=variant, modality=modality
                )
            )
    diagnostic = (
        pd.concat(diagnostic_parts, ignore_index=True)
        if diagnostic_parts
        else pd.DataFrame(
            columns=[
                "choice_index",
                "rms_delta",
                "event_count",
                "alternative_span",
                "cap_saturated",
                "gene_id",
                "variant",
                "modality",
            ]
        )
    )
    correlations = (
        pd.concat(correlation_parts, ignore_index=True)
        if correlation_parts
        else pd.DataFrame(
            columns=[
                "covariate",
                "spearman_r",
                "pvalue",
                "gene_id",
                "variant",
                "modality",
            ]
        )
    )
    return diagnostic, correlations


def write_evaluation(report: EvaluationReport, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report.primary_metrics.to_csv(root / "primary_metrics.tsv", sep="\t", index=False)
    report.paired_deltas.to_csv(root / "paired_deltas.tsv", sep="\t", index=False)
    report.coverage.to_csv(root / "identifiable_coverage.tsv", sep="\t", index=False)
