"""Train-frozen path identifiability and reporting-only local alternatives.

Nothing in this module is a trainable model object.  The path index is built
only from likelihood-informative train EC patterns.  The alternative index is
split-neutral except for the explicitly labelled train-support columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .graph import GeneGraph


SVD_RELATIVE_TOLERANCE = 1.0e-8
SUPPORT_TIER_DIRECT = "direct_cell_supported"
SUPPORT_TIER_COHORT = "cohort_identifiable_model_prediction"
SUPPORT_TIER_UNIDENTIFIABLE = "supervision_unidentifiable_prediction"


@dataclass(frozen=True)
class PathIdentifiabilityIndex:
    """Frozen train-supervision partition and full compatibility operators."""

    genes: pd.DataFrame
    paths: pd.DataFrame
    groups: pd.DataFrame
    train_patterns: pd.DataFrame
    svd_relative_tolerance: float = SVD_RELATIVE_TOLERANCE
    minimum_exclusive_support: float = 1.0


@dataclass(frozen=True)
class AlternativeReportingIndex:
    """Split-neutral path subsets plus train-only local reportability."""

    choices: pd.DataFrame
    alternatives: pd.DataFrame
    path_membership: pd.DataFrame
    contrasts: pd.DataFrame
    svd_relative_tolerance: float = SVD_RELATIVE_TOLERANCE
    minimum_exclusive_support: float = 1.0


def build_path_identifiability_index(
    graphs: GeneGraph | Sequence[GeneGraph],
    ec_rows: pd.DataFrame,
    *,
    svd_relative_tolerance: float = SVD_RELATIVE_TOLERANCE,
    minimum_exclusive_support: float = 1.0,
) -> PathIdentifiabilityIndex:
    """Build observational groups from distinct informative train patterns.

    Empty and all-path compatible rows are audit-only and never enter this
    operator.  Repeated patterns are represented once and only their molecule
    mass is aggregated.
    """

    graph_rows = _as_graphs(graphs)
    if svd_relative_tolerance <= 0:
        raise ValueError("svd_relative_tolerance must be positive")
    if minimum_exclusive_support <= 0:
        raise ValueError("minimum_exclusive_support must be positive")
    _require_columns(ec_rows, ("gene_id", "split", "molecule_count"), "EC rows")
    train = ec_rows.loc[ec_rows["split"].astype(str).eq("train")].copy()
    masses = train["molecule_count"].to_numpy(dtype=np.float64)
    if not np.isfinite(masses).all() or bool((masses < 0).any()):
        raise ValueError("train EC molecule_count must be finite and non-negative")

    known_genes = {graph.gene_id for graph in graph_rows}
    unknown_genes = sorted(set(train["gene_id"].astype(str)) - known_genes)
    if unknown_genes:
        raise ValueError(f"train EC genes are absent from graph catalog: {unknown_genes[:5]}")

    gene_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    group_records: list[dict[str, object]] = []
    pattern_records: list[dict[str, object]] = []

    for graph in graph_rows:
        path_ids = tuple(map(str, graph.path_ids))
        if not path_ids:
            raise ValueError(f"gene {graph.gene_id} has no frozen legal paths")
        path_index = {path_id: index for index, path_id in enumerate(path_ids)}
        gene_train = train.loc[train["gene_id"].astype(str).eq(graph.gene_id)]
        patterns: dict[tuple[int, ...], dict[str, float | int]] = {}
        for row in gene_train.itertuples(index=False):
            compatible = _compatible_indices(row, path_index)
            mass = float(getattr(row, "molecule_count"))
            if mass <= 0 or not compatible or len(compatible) == len(path_ids):
                continue
            state = patterns.setdefault(compatible, {"row_count": 0, "molecule_mass": 0.0})
            state["row_count"] = int(state["row_count"]) + 1
            state["molecule_mass"] = float(state["molecule_mass"]) + mass

        ordered_patterns = sorted(patterns)
        compatibility = np.zeros((len(ordered_patterns), len(path_ids)), dtype=np.float64)
        for row_index, compatible in enumerate(ordered_patterns):
            compatibility[row_index, list(compatible)] = 1.0

        signature_to_group: dict[tuple[float, ...], int] = {}
        path_group_indices: list[int] = []
        group_members: list[list[int]] = []
        for column in range(len(path_ids)):
            signature = tuple(compatibility[:, column].tolist())
            if signature not in signature_to_group:
                signature_to_group[signature] = len(group_members)
                group_members.append([])
            group_index = signature_to_group[signature]
            group_members[group_index].append(column)
            path_group_indices.append(group_index)

        group_ids = tuple(
            f"obs-group:{graph.gene_id}:{group_index:04d}"
            for group_index in range(len(group_members))
        )
        collapsed = (
            compatibility[:, [members[0] for members in group_members]]
            if group_members
            else np.empty((len(ordered_patterns), 0), dtype=np.float64)
        )
        augmented = np.vstack([collapsed, np.ones((1, len(group_members)))])
        augmented_rank, singular_values, numerical_tolerance = _svd_rank(
            augmented, svd_relative_tolerance
        )
        cohort_separable = augmented_rank == len(group_members)

        exclusive_support: list[float] = []
        for members in group_members:
            member_set = set(members)
            support = sum(
                float(patterns[compatible]["molecule_mass"])
                for compatible in ordered_patterns
                if set(compatible).issubset(member_set)
            )
            exclusive_support.append(float(support))
        all_supported = all(
            support >= minimum_exclusive_support for support in exclusive_support
        )
        cohort_identifiable = bool(cohort_separable and all_supported)

        for pattern_index, compatible in enumerate(ordered_patterns):
            pattern_records.append(
                {
                    "gene_id": graph.gene_id,
                    "pattern_id": f"train-pattern:{graph.gene_id}:{pattern_index:04d}",
                    "compatible_path_ids": [path_ids[index] for index in compatible],
                    "compatible_path_indices": list(compatible),
                    "compatible_group_ids": sorted(
                        {group_ids[path_group_indices[index]] for index in compatible}
                    ),
                    "row_count": int(patterns[compatible]["row_count"]),
                    "molecule_mass": float(patterns[compatible]["molecule_mass"]),
                }
            )

        aliases = getattr(graph, "transcript_aliases", ())
        if aliases and len(aliases) != len(path_ids):
            raise ValueError(f"gene {graph.gene_id} path/alias axes differ")
        if not aliases:
            aliases = tuple((path_id,) for path_id in path_ids)
        for path_position, path_id in enumerate(path_ids):
            group_index = path_group_indices[path_position]
            path_records.append(
                {
                    "gene_id": graph.gene_id,
                    "path_id": path_id,
                    "path_index": path_position,
                    "transcript_aliases": list(map(str, aliases[path_position])),
                    "observational_group_id": group_ids[group_index],
                    "observational_group_index": group_index,
                    "observational_group_size": len(group_members[group_index]),
                    "within_group_prediction_status": (
                        "supervision_identifiable_singleton"
                        if len(group_members[group_index]) == 1
                        else "model_resolved_within_supervision_unidentifiable_group"
                    ),
                }
            )
        for group_index, members in enumerate(group_members):
            group_records.append(
                {
                    "gene_id": graph.gene_id,
                    "observational_group_id": group_ids[group_index],
                    "observational_group_index": group_index,
                    "member_path_ids": [path_ids[index] for index in members],
                    "member_count": len(members),
                    "train_exclusive_molecule_mass": exclusive_support[group_index],
                    "has_train_exclusive_support": bool(
                        exclusive_support[group_index] >= minimum_exclusive_support
                    ),
                    "cohort_contrast_separable": bool(cohort_separable),
                    "cohort_identifiable": cohort_identifiable,
                }
            )
        gene_records.append(
            {
                "gene_id": graph.gene_id,
                "path_ids": list(path_ids),
                "group_ids": list(group_ids),
                "path_group_indices": path_group_indices,
                "train_compatibility_matrix": compatibility.astype(int).tolist(),
                "collapsed_compatibility_matrix": collapsed.astype(int).tolist(),
                "augmented_operator": augmented.astype(int).tolist(),
                "singular_values": singular_values.tolist(),
                "svd_numerical_tolerance": numerical_tolerance,
                "augmented_rank": augmented_rank,
                "group_count": len(group_members),
                "cohort_contrast_separable": bool(cohort_separable),
                "all_groups_exclusive_supported": bool(all_supported),
                "cohort_identifiable": cohort_identifiable,
            }
        )

    return PathIdentifiabilityIndex(
        genes=pd.DataFrame(gene_records),
        paths=pd.DataFrame(path_records),
        groups=pd.DataFrame(group_records),
        train_patterns=pd.DataFrame(
            pattern_records,
            columns=(
                "gene_id",
                "pattern_id",
                "compatible_path_ids",
                "compatible_path_indices",
                "compatible_group_ids",
                "row_count",
                "molecule_mass",
            ),
        ),
        svd_relative_tolerance=svd_relative_tolerance,
        minimum_exclusive_support=minimum_exclusive_support,
    )


def classify_heldout_path_support(
    index: PathIdentifiabilityIndex,
    ec_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify held-out rows and cells without changing train-frozen groups."""

    _require_columns(
        ec_rows, ("cell_id", "gene_id", "split", "molecule_count"), "held-out EC rows"
    )
    if bool(ec_rows["split"].astype(str).eq("train").any()):
        raise ValueError("held-out classification cannot consume train rows")
    gene_lookup = index.genes.set_index("gene_id", drop=False)
    unknown = sorted(set(ec_rows["gene_id"].astype(str)) - set(gene_lookup.index))
    if unknown:
        raise ValueError(f"held-out EC genes are absent from frozen index: {unknown[:5]}")

    row_records: list[dict[str, object]] = []
    for source_row, row in ec_rows.reset_index(drop=False).iterrows():
        gene_id = str(row["gene_id"])
        gene = gene_lookup.loc[gene_id]
        path_ids = tuple(gene["path_ids"])
        path_index = {path_id: position for position, path_id in enumerate(path_ids)}
        compatible = _compatible_indices(row, path_index)
        mass = float(row["molecule_count"])
        if not np.isfinite(mass) or mass < 0:
            raise ValueError("held-out molecule_count must be finite and non-negative")
        informative = bool(mass > 0 and compatible and len(compatible) < len(path_ids))
        selected = set(compatible)
        path_group_indices = list(map(int, gene["path_group_indices"]))
        group_count = int(gene["group_count"])
        collapsed: list[int] = []
        split_groups: list[str] = []
        group_ids = tuple(gene["group_ids"])
        if informative:
            for group_index in range(group_count):
                members = {
                    path_position
                    for path_position, observed_group in enumerate(path_group_indices)
                    if observed_group == group_index
                }
                overlap = selected & members
                if overlap and overlap != members:
                    split_groups.append(group_ids[group_index])
                collapsed.append(int(overlap == members))
        group_constant = informative and not split_groups
        row_records.append(
            {
                "source_row_index": row["index"],
                "cell_id": str(row["cell_id"]),
                "gene_id": gene_id,
                "split": str(row["split"]),
                "molecule_count": mass,
                "compatible_path_ids": [path_ids[position] for position in compatible],
                "likelihood_informative": informative,
                "group_constant": bool(group_constant),
                "novel_split_group_row": bool(informative and split_groups),
                "split_observational_group_ids": split_groups,
                "collapsed_group_pattern": collapsed if group_constant else None,
            }
        )
    row_table = pd.DataFrame(row_records)

    cell_records: list[dict[str, object]] = []
    for (split, cell_id, gene_id), rows in row_table.groupby(
        ["split", "cell_id", "gene_id"], sort=True
    ):
        gene = gene_lookup.loc[gene_id]
        group_count = int(gene["group_count"])
        valid = rows.loc[rows["group_constant"]]
        unique_patterns = sorted(
            {tuple(map(int, values)) for values in valid["collapsed_group_pattern"]}
        )
        collapsed = np.asarray(unique_patterns, dtype=np.float64)
        if collapsed.size == 0:
            collapsed = np.empty((0, group_count), dtype=np.float64)
        augmented = np.vstack([collapsed, np.ones((1, group_count))])
        rank, singular_values, tolerance = _svd_rank(
            augmented, index.svd_relative_tolerance
        )
        support = []
        group_ids = tuple(gene["group_ids"])
        path_groups = list(map(int, gene["path_group_indices"]))
        for group_index in range(group_count):
            member_paths = {
                gene["path_ids"][position]
                for position, value in enumerate(path_groups)
                if value == group_index
            }
            group_support = sum(
                float(record.molecule_count)
                for record in valid.itertuples(index=False)
                if set(record.compatible_path_ids).issubset(member_paths)
            )
            support.append(float(group_support))
        direct = bool(
            gene["cohort_identifiable"]
            and rank == group_count
            and all(value >= index.minimum_exclusive_support for value in support)
        )
        tier = (
            SUPPORT_TIER_DIRECT
            if direct
            else SUPPORT_TIER_COHORT
            if bool(gene["cohort_identifiable"])
            else SUPPORT_TIER_UNIDENTIFIABLE
        )
        cell_records.append(
            {
                "split": split,
                "cell_id": cell_id,
                "gene_id": gene_id,
                "cell_augmented_operator": augmented.astype(int).tolist(),
                "cell_augmented_rank": rank,
                "cell_singular_values": singular_values.tolist(),
                "cell_svd_numerical_tolerance": tolerance,
                "group_ids": list(group_ids),
                "group_exclusive_molecule_mass": support,
                "has_novel_split_group_row": bool(rows["novel_split_group_row"].any()),
                "direct_cell_supported": direct,
                "support_tier": tier,
            }
        )
    return pd.DataFrame(cell_records), row_table


def aggregate_group_probabilities(
    path_probabilities: np.ndarray | torch.Tensor,
    index: PathIdentifiabilityIndex,
    gene_id: str,
) -> np.ndarray | torch.Tensor:
    """Sum path probabilities over frozen observational groups."""

    gene = _gene_row(index, gene_id)
    path_groups = list(map(int, gene["path_group_indices"]))
    if path_probabilities.shape[-1] != len(path_groups):
        raise ValueError("path probability axis differs from identifiability index")
    if isinstance(path_probabilities, torch.Tensor):
        columns = [
            path_probabilities[..., [i for i, group in enumerate(path_groups) if group == g]].sum(-1)
            for g in range(int(gene["group_count"]))
        ]
        return torch.stack(columns, dim=-1)
    values = np.asarray(path_probabilities)
    return np.stack(
        [
            values[..., [i for i, group in enumerate(path_groups) if group == g]].sum(-1)
            for g in range(int(gene["group_count"]))
        ],
        axis=-1,
    )


def aggregate_group_log_probabilities(
    path_log_probabilities: np.ndarray | torch.Tensor,
    index: PathIdentifiabilityIndex,
    gene_id: str,
) -> np.ndarray | torch.Tensor:
    """Log-sum-exp path log probabilities over frozen groups."""

    gene = _gene_row(index, gene_id)
    path_groups = list(map(int, gene["path_group_indices"]))
    if path_log_probabilities.shape[-1] != len(path_groups):
        raise ValueError("path log-probability axis differs from identifiability index")
    selections = [
        [i for i, group in enumerate(path_groups) if group == g]
        for g in range(int(gene["group_count"]))
    ]
    if isinstance(path_log_probabilities, torch.Tensor):
        return torch.stack(
            [torch.logsumexp(path_log_probabilities[..., selected], dim=-1) for selected in selections],
            dim=-1,
        )
    values = np.asarray(path_log_probabilities, dtype=np.float64)
    return np.stack([_numpy_logsumexp(values[..., selected], axis=-1) for selected in selections], axis=-1)


def build_alternative_reporting_index(
    graphs: GeneGraph | Sequence[GeneGraph],
    path_index: PathIdentifiabilityIndex,
) -> AlternativeReportingIndex:
    """Build TSS, internal and PAS reporting subsets without reading outcomes."""

    graph_rows = _as_graphs(graphs)
    index_genes = set(path_index.genes["gene_id"].astype(str))
    if {graph.gene_id for graph in graph_rows} != index_genes:
        raise ValueError("graph and PathIdentifiabilityIndex gene universes differ")

    choices: list[dict[str, object]] = []
    alternatives: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    for graph in graph_rows:
        graph_choices = _structural_reporting_choices(graph)
        choices.extend(graph_choices[0])
        alternatives.extend(graph_choices[1])
        membership.extend(graph_choices[2])

    choice_table = pd.DataFrame(
        choices,
        columns=(
            "choice_id",
            "gene_id",
            "choice_kind",
            "entry_node_id",
            "exit_node_id",
            "structurally_valid",
        ),
    )
    alternative_table = pd.DataFrame(
        alternatives,
        columns=(
            "choice_id",
            "gene_id",
            "choice_kind",
            "alternative_id",
            "endpoint_node_id",
            "edge_ids",
            "path_ids",
            "path_count",
        ),
    )
    membership_table = pd.DataFrame(
        membership,
        columns=(
            "choice_id",
            "gene_id",
            "choice_kind",
            "path_id",
            "alternative_id",
            "context_signature",
            "eligible_for_local_reporting",
        ),
    )
    contrast_records: list[dict[str, object]] = []
    for choice in choice_table.itertuples(index=False):
        choice_alternatives = alternative_table.loc[
            alternative_table["choice_id"].eq(choice.choice_id)
        ].sort_values("alternative_id", kind="mergesort")
        for left, right in combinations(choice_alternatives.itertuples(index=False), 2):
            contrast_records.append(
                _contrast_record(
                    path_index,
                    choice.gene_id,
                    choice.choice_id,
                    choice.choice_kind,
                    left.alternative_id,
                    right.alternative_id,
                    None,
                    left.path_ids,
                    right.path_ids,
                    "marginal",
                )
            )
            left_context = membership_table.loc[
                membership_table["alternative_id"].eq(left.alternative_id)
                & membership_table["eligible_for_local_reporting"]
            ].groupby("context_signature", sort=True)["path_id"].agg(list)
            right_context = membership_table.loc[
                membership_table["alternative_id"].eq(right.alternative_id)
                & membership_table["eligible_for_local_reporting"]
            ].groupby("context_signature", sort=True)["path_id"].agg(list)
            for context in sorted(set(left_context.index) & set(right_context.index)):
                contrast_records.append(
                    _contrast_record(
                        path_index,
                        choice.gene_id,
                        choice.choice_id,
                        choice.choice_kind,
                        left.alternative_id,
                        right.alternative_id,
                        str(context),
                        left_context.loc[context],
                        right_context.loc[context],
                        "matched_context",
                    )
                )
    contrasts = pd.DataFrame(
        contrast_records,
        columns=(
            "contrast_id",
            "gene_id",
            "choice_id",
            "choice_kind",
            "contrast_kind",
            "context_signature",
            "numerator_alternative_id",
            "denominator_alternative_id",
            "numerator_path_ids",
            "denominator_path_ids",
            "numerator_group_indicator",
            "denominator_group_indicator",
            "crossing_observational_group_ids",
            "numerator_rowspace_estimable",
            "denominator_rowspace_estimable",
            "numerator_train_exclusive_molecule_mass",
            "denominator_train_exclusive_molecule_mass",
            "cohort_local_contrast_separable",
            "cohort_reportable",
        ),
    )
    if len(choice_table):
        matched = contrasts.loc[contrasts["contrast_kind"].eq("matched_context")]
        counts = matched.groupby("choice_id", sort=False).size()
        reportable = matched.loc[matched["cohort_reportable"]].groupby("choice_id").size()
        choice_table["n_matched_context_candidates"] = (
            choice_table["choice_id"].map(counts).fillna(0).astype(int)
        )
        choice_table["has_matched_context_structure"] = choice_table[
            "n_matched_context_candidates"
        ].gt(0)
        choice_table["n_cohort_reportable_matched"] = (
            choice_table["choice_id"].map(reportable).fillna(0).astype(int)
        )
        choice_table["has_cohort_reportable_matched"] = choice_table[
            "n_cohort_reportable_matched"
        ].gt(0)
    return AlternativeReportingIndex(
        choices=choice_table,
        alternatives=alternative_table,
        path_membership=membership_table,
        contrasts=contrasts,
        svd_relative_tolerance=path_index.svd_relative_tolerance,
        minimum_exclusive_support=path_index.minimum_exclusive_support,
    )


def classify_heldout_alternative_support(
    reporting_index: AlternativeReportingIndex,
    path_index: PathIdentifiabilityIndex,
    ec_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Classify every held-out cell × frozen contrast on a fixed denominator."""

    cell_support, classified_rows = classify_heldout_path_support(path_index, ec_rows)
    records: list[dict[str, object]] = []
    for cell in cell_support.itertuples(index=False):
        candidates = reporting_index.contrasts.loc[
            reporting_index.contrasts["gene_id"].eq(cell.gene_id)
        ]
        cell_rows = classified_rows.loc[
            classified_rows["cell_id"].eq(cell.cell_id)
            & classified_rows["gene_id"].eq(cell.gene_id)
            & classified_rows["split"].eq(cell.split)
            & classified_rows["group_constant"]
        ]
        operator = np.asarray(cell.cell_augmented_operator, dtype=np.float64)
        for candidate in candidates.itertuples(index=False):
            numerator_estimable = _indicator_in_row_space(
                candidate.numerator_group_indicator, operator, path_index.svd_relative_tolerance
            ) if candidate.numerator_group_indicator is not None else False
            denominator_estimable = _indicator_in_row_space(
                candidate.denominator_group_indicator, operator, path_index.svd_relative_tolerance
            ) if candidate.denominator_group_indicator is not None else False
            numerator_paths = set(candidate.numerator_path_ids)
            denominator_paths = set(candidate.denominator_path_ids)
            numerator_support = sum(
                float(row.molecule_count)
                for row in cell_rows.itertuples(index=False)
                if set(row.compatible_path_ids).issubset(numerator_paths)
            )
            denominator_support = sum(
                float(row.molecule_count)
                for row in cell_rows.itertuples(index=False)
                if set(row.compatible_path_ids).issubset(denominator_paths)
            )
            direct = bool(
                candidate.cohort_reportable
                and numerator_estimable
                and denominator_estimable
                and numerator_support >= reporting_index.minimum_exclusive_support
                and denominator_support >= reporting_index.minimum_exclusive_support
            )
            tier = (
                SUPPORT_TIER_DIRECT
                if direct
                else SUPPORT_TIER_COHORT
                if candidate.cohort_reportable
                else SUPPORT_TIER_UNIDENTIFIABLE
            )
            records.append(
                {
                    "split": cell.split,
                    "cell_id": cell.cell_id,
                    "gene_id": cell.gene_id,
                    "choice_id": candidate.choice_id,
                    "choice_kind": candidate.choice_kind,
                    "contrast_id": candidate.contrast_id,
                    "contrast_kind": candidate.contrast_kind,
                    "context_signature": candidate.context_signature,
                    "cohort_reportable": bool(candidate.cohort_reportable),
                    "cell_numerator_rowspace_estimable": bool(numerator_estimable),
                    "cell_denominator_rowspace_estimable": bool(denominator_estimable),
                    "cell_numerator_exclusive_molecule_mass": numerator_support,
                    "cell_denominator_exclusive_molecule_mass": denominator_support,
                    "direct_cell_supported": direct,
                    "support_tier": tier,
                }
            )
    return pd.DataFrame(records)


def alternative_relative_log_mass(
    path_logits: np.ndarray | torch.Tensor,
    path_ids: Sequence[str],
    numerator_path_ids: Sequence[str],
    denominator_path_ids: Sequence[str],
) -> np.ndarray | torch.Tensor:
    """Gauge-invariant relative log mass for marginal or matched subsets."""

    axis = {str(path_id): index for index, path_id in enumerate(path_ids)}
    numerator = _resolve_subset_indices(numerator_path_ids, axis, "numerator")
    denominator = _resolve_subset_indices(denominator_path_ids, axis, "denominator")
    if path_logits.shape[-1] != len(axis):
        raise ValueError("path logit axis differs from path_ids")
    if isinstance(path_logits, torch.Tensor):
        return torch.logsumexp(path_logits[..., numerator], dim=-1) - torch.logsumexp(
            path_logits[..., denominator], dim=-1
        )
    values = np.asarray(path_logits, dtype=np.float64)
    return _numpy_logsumexp(values[..., numerator], axis=-1) - _numpy_logsumexp(
        values[..., denominator], axis=-1
    )


def centered_logit_change(
    full_logits: np.ndarray | torch.Tensor,
    counterfactual_logits: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Equal-structural-path gauge fixing required by the V2 contract."""

    if full_logits.shape != counterfactual_logits.shape:
        raise ValueError("full and counterfactual logit shapes differ")
    delta = full_logits - counterfactual_logits
    return delta - delta.mean(axis=-1, keepdims=True) if isinstance(delta, np.ndarray) else delta - delta.mean(dim=-1, keepdim=True)


def alternative_coverage_tables(
    reporting_index: AlternativeReportingIndex,
    heldout_records: pd.DataFrame,
    *,
    manifest_selected: pd.DataFrame | None = None,
    expected_splits: Sequence[str] = ("val", "test"),
) -> dict[str, pd.DataFrame]:
    """Return the fixed choice- and cell-record-level coverage waterfalls."""

    split_axis = tuple(map(str, expected_splits))
    if not split_axis or len(split_axis) != len(set(split_axis)):
        raise ValueError("alternative coverage expected_splits must be unique and non-empty")
    records = heldout_records.copy()
    matched = records["contrast_kind"].eq("matched_context") if len(records) else pd.Series(dtype=bool)
    records = records.loc[matched].copy() if len(records) else records
    records["manifest_selected"] = False
    if manifest_selected is not None and len(manifest_selected):
        keys = ["cell_id", "gene_id", "contrast_id"]
        _require_columns(manifest_selected, keys, "manifest selection")
        selected = manifest_selected[keys].drop_duplicates().assign(manifest_selected=True)
        records = records.drop(columns="manifest_selected").merge(selected, on=keys, how="left")
        records["manifest_selected"] = records["manifest_selected"].fillna(False).astype(bool)
    records["direct_and_manifest_selected"] = (
        records["direct_cell_supported"] & records["manifest_selected"]
        if len(records)
        else pd.Series(dtype=bool)
    )

    choices = reporting_index.choices.copy()
    direct_choices = set(records.loc[records["direct_cell_supported"], "choice_id"])
    selected_choices = set(records.loc[records["direct_and_manifest_selected"], "choice_id"])
    choices["has_direct_cell_supported_heldout_record"] = choices["choice_id"].isin(direct_choices)
    choices["has_direct_and_manifest_selected_heldout_record"] = choices["choice_id"].isin(selected_choices)

    levels = (
        ("all_structurally_valid_choices", pd.Series(True, index=choices.index)),
        ("has_two_arm_matched_context", choices["has_matched_context_structure"]),
        ("has_cohort_reportable_matched", choices["has_cohort_reportable_matched"]),
        ("has_direct_cell_supported_heldout_record", choices["has_direct_cell_supported_heldout_record"]),
        ("has_direct_and_manifest_selected_heldout_record", choices["has_direct_and_manifest_selected_heldout_record"]),
    )
    choice_summary: list[dict[str, object]] = []
    for scope in ("all", "tss", "internal", "pas"):
        scoped = choices if scope == "all" else choices.loc[choices["choice_kind"].eq(scope)]
        denominator = len(scoped)
        for level, mask in levels:
            selected_mask = mask.loc[scoped.index] if denominator else pd.Series(dtype=bool)
            count = int(selected_mask.sum()) if denominator else 0
            choice_summary.append(
                {
                    "choice_scope": scope,
                    "waterfall_level": level,
                    "choice_denominator": denominator,
                    "choice_count": count,
                    "choice_fraction": count / denominator if denominator else np.nan,
                    "status": "estimable" if denominator else "not_estimable",
                }
            )

    record_summary: list[dict[str, object]] = []
    observed_splits = set(records["split"].astype(str)) if len(records) else set()
    unexpected = sorted(observed_splits - set(split_axis))
    if unexpected:
        raise ValueError(f"alternative coverage contains unexpected splits: {unexpected}")
    for split in split_axis:
        split_rows = records.loc[records["split"].eq(split)]
        for scope in ("all", "tss", "internal", "pas"):
            scoped = split_rows if scope == "all" else split_rows.loc[split_rows["choice_kind"].eq(scope)]
            denominator = len(scoped)
            for level, column in (
                ("cohort_reportable", "cohort_reportable"),
                ("direct_cell_supported", "direct_cell_supported"),
                ("direct_and_manifest_selected", "direct_and_manifest_selected"),
            ):
                count = int(scoped[column].sum()) if denominator else 0
                record_summary.append(
                    {
                        "split": split,
                        "choice_scope": scope,
                        "waterfall_level": level,
                        "record_denominator": denominator,
                        "record_count": count,
                        "record_fraction": count / denominator if denominator else np.nan,
                        "status": "estimable" if denominator else "not_estimable",
                    }
                )
    return {
        "choice_records": choices,
        "choice_level": pd.DataFrame(choice_summary),
        "record_records": records,
        "record_level": pd.DataFrame(record_summary),
    }


def _structural_reporting_choices(
    graph: GeneGraph,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    path_ids = tuple(map(str, graph.path_ids))
    edge_sequences = tuple(
        tuple(graph.edge_ids[index] for index in sequence) for sequence in graph.path_edge_rows
    )
    path_metadata = graph.paths.set_index(graph.paths["path_id"].astype(str), drop=False)
    if set(path_metadata.index) != set(path_ids):
        raise ValueError(f"gene {graph.gene_id} reporting path metadata differs from graph axis")
    choices: list[dict[str, object]] = []
    alternatives: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []

    for kind, endpoint_column in (("tss", "tss_node_id"), ("pas", "pas_node_id")):
        endpoint_by_path = {
            path_id: str(path_metadata.loc[path_id, endpoint_column]) for path_id in path_ids
        }
        endpoints = sorted(set(endpoint_by_path.values()))
        if len(endpoints) < 2:
            continue
        choice_id = f"choice:{graph.gene_id}:{kind.upper()}"
        choices.append(
            {
                "choice_id": choice_id,
                "gene_id": graph.gene_id,
                "choice_kind": kind,
                "entry_node_id": None,
                "exit_node_id": None,
                "structurally_valid": True,
            }
        )
        alternative_by_endpoint = {
            endpoint: f"{choice_id}:alt:{endpoint}" for endpoint in endpoints
        }
        alt_by_path = {path_id: alternative_by_endpoint[endpoint_by_path[path_id]] for path_id in path_ids}
        contexts = _endpoint_contexts(path_ids, edge_sequences, alt_by_path, kind)
        for endpoint in endpoints:
            member_paths = [path for path in path_ids if endpoint_by_path[path] == endpoint]
            alternatives.append(
                {
                    "choice_id": choice_id,
                    "gene_id": graph.gene_id,
                    "choice_kind": kind,
                    "alternative_id": alternative_by_endpoint[endpoint],
                    "endpoint_node_id": endpoint,
                    "edge_ids": None,
                    "path_ids": member_paths,
                    "path_count": len(member_paths),
                }
            )
        for path_id in path_ids:
            membership.append(
                {
                    "choice_id": choice_id,
                    "gene_id": graph.gene_id,
                    "choice_kind": kind,
                    "path_id": path_id,
                    "alternative_id": alt_by_path[path_id],
                    "context_signature": contexts[path_id],
                    "eligible_for_local_reporting": not contexts[path_id].startswith("UNMATCHED:"),
                }
            )

    internal = _internal_choices(graph)
    choices.extend(internal[0])
    alternatives.extend(internal[1])
    membership.extend(internal[2])
    return choices, alternatives, membership


def _endpoint_contexts(
    path_ids: Sequence[str],
    sequences: Sequence[tuple[str, ...]],
    alternative_by_path: dict[str, str],
    kind: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path_index, path_id in enumerate(path_ids):
        sequence = sequences[path_index]
        candidates: Iterable[tuple[str, ...]]
        if kind == "tss":
            candidates = (sequence[start:] for start in range(len(sequence)))
        else:
            candidates = (sequence[:stop] for stop in range(len(sequence), 0, -1))
        selected: tuple[str, ...] | None = None
        for candidate in candidates:
            if any(
                alternative_by_path[other_path] != alternative_by_path[path_id]
                and (
                    sequences[other_index][-len(candidate) :] == candidate
                    if kind == "tss"
                    else sequences[other_index][: len(candidate)] == candidate
                )
                for other_index, other_path in enumerate(path_ids)
            ):
                selected = candidate
                break
        result[path_id] = (
            _signature(kind, selected) if selected else f"UNMATCHED:{kind}:{path_id}"
        )
    return result


def _internal_choices(
    graph: GeneGraph,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    edge_src = graph.edges["src_node_id"].astype(str).tolist()
    edge_dst = graph.edges["dst_node_id"].astype(str).tolist()
    outgoing: dict[str, set[int]] = {}
    incoming: dict[str, set[int]] = {}
    occurrences: dict[str, list[tuple[int, int]]] = {}
    for edge_index, (src, dst) in enumerate(zip(edge_src, edge_dst, strict=True)):
        outgoing.setdefault(src, set()).add(edge_index)
        incoming.setdefault(dst, set()).add(edge_index)
    for path_index, nodes in enumerate(graph.path_node_rows):
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"path {graph.path_ids[path_index]} repeats a node")
        for node_index, node in enumerate(nodes[:-1]):
            occurrences.setdefault(node, []).append((path_index, node_index))

    candidates: list[dict[str, object]] = []
    for entry in sorted(occurrences):
        path_occurrences = occurrences[entry]
        if len({graph.path_edge_rows[p][n] for p, n in path_occurrences}) < 2:
            continue
        common_exits = set.intersection(
            *[set(graph.path_node_rows[p][n + 1 :]) for p, n in path_occurrences]
        )
        if not common_exits:
            continue
        ranked = []
        for exit_node in common_exits:
            distances = [
                graph.path_node_rows[p].index(exit_node, n + 1) - n
                for p, n in path_occurrences
            ]
            ranked.append((max(distances), sum(distances), exit_node))
        _, _, exit_node = min(ranked)
        path_sequences: dict[int, tuple[int, ...]] = {}
        path_contexts: dict[int, str] = {}
        for path_position, entry_position in path_occurrences:
            nodes = graph.path_node_rows[path_position]
            exit_position = nodes.index(exit_node, entry_position + 1)
            path_sequences[path_position] = tuple(
                graph.path_edge_rows[path_position][entry_position:exit_position]
            )
            prefix = tuple(
                graph.edge_ids[index]
                for index in graph.path_edge_rows[path_position][:entry_position]
            )
            suffix = tuple(
                graph.edge_ids[index]
                for index in graph.path_edge_rows[path_position][exit_position:]
            )
            path_contexts[path_position] = _signature("internal", (prefix, suffix))
        sequences = sorted(
            set(path_sequences.values()),
            key=lambda row: tuple(graph.edge_ids[index] for index in row),
        )
        if len(sequences) < 2 or _candidate_rejection_reason(
            sequences, edge_src, edge_dst, outgoing, incoming
        ) is not None:
            continue
        candidates.append(
            {
                "entry": entry,
                "exit": exit_node,
                "sequences": sequences,
                "path_sequences": path_sequences,
                "path_contexts": path_contexts,
            }
        )

    admitted = []
    for candidate in candidates:
        candidate_edges = {edge for seq in candidate["sequences"] for edge in seq}
        conflict = False
        for other in candidates:
            if other is candidate:
                continue
            other_edges = {edge for seq in other["sequences"] for edge in seq}
            if candidate_edges & other_edges:
                conflict = True
                break
        if not conflict:
            admitted.append(candidate)

    choice_rows: list[dict[str, object]] = []
    alternative_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for candidate in sorted(admitted, key=lambda value: (value["entry"], value["exit"])):
        choice_id = f"choice:{graph.gene_id}:{candidate['entry']}->{candidate['exit']}"
        choice_rows.append(
            {
                "choice_id": choice_id,
                "gene_id": graph.gene_id,
                "choice_kind": "internal",
                "entry_node_id": candidate["entry"],
                "exit_node_id": candidate["exit"],
                "structurally_valid": True,
            }
        )
        alt_by_sequence = {}
        for alt_index, sequence in enumerate(candidate["sequences"]):
            alternative_id = f"{choice_id}:alt:{alt_index:03d}"
            alt_by_sequence[sequence] = alternative_id
            member_paths = [
                graph.path_ids[path_position]
                for path_position, observed in candidate["path_sequences"].items()
                if observed == sequence
            ]
            alternative_rows.append(
                {
                    "choice_id": choice_id,
                    "gene_id": graph.gene_id,
                    "choice_kind": "internal",
                    "alternative_id": alternative_id,
                    "endpoint_node_id": None,
                    "edge_ids": [graph.edge_ids[index] for index in sequence],
                    "path_ids": member_paths,
                    "path_count": len(member_paths),
                }
            )
        for path_position, path_id in enumerate(graph.path_ids):
            sequence = candidate["path_sequences"].get(path_position)
            membership_rows.append(
                {
                    "choice_id": choice_id,
                    "gene_id": graph.gene_id,
                    "choice_kind": "internal",
                    "path_id": path_id,
                    "alternative_id": alt_by_sequence.get(sequence),
                    "context_signature": candidate["path_contexts"].get(path_position),
                    "eligible_for_local_reporting": sequence is not None,
                }
            )
    return choice_rows, alternative_rows, membership_rows


def _candidate_rejection_reason(sequences, edge_src, edge_dst, outgoing, incoming):
    internal_sets = []
    for sequence in sequences:
        if not sequence:
            return "empty_alternative"
        nodes = [edge_src[sequence[0]], *[edge_dst[index] for index in sequence]]
        if any(edge_dst[a] != edge_src[b] for a, b in zip(sequence[:-1], sequence[1:])):
            return "discontinuous_alternative"
        internal = set(nodes[1:-1])
        if any(len(outgoing.get(node, ())) != 1 or len(incoming.get(node, ())) != 1 for node in internal):
            return "internal_branch_or_merge"
        internal_sets.append(internal)
    if any(internal_sets[a] & internal_sets[b] for a, b in combinations(range(len(internal_sets)), 2)):
        return "shared_internal_node"
    return None


def _contrast_record(
    index,
    gene_id,
    choice_id,
    choice_kind,
    left_id,
    right_id,
    context,
    left_paths,
    right_paths,
    kind,
):
    left_paths = sorted(map(str, left_paths))
    right_paths = sorted(map(str, right_paths))
    left_status = _subset_status(index, gene_id, left_paths)
    right_status = _subset_status(index, gene_id, right_paths)
    left_support = _train_subset_support(index, gene_id, set(left_paths))
    right_support = _train_subset_support(index, gene_id, set(right_paths))
    cohort = bool(
        left_status["defined"]
        and right_status["defined"]
        and left_status["rowspace_estimable"]
        and right_status["rowspace_estimable"]
        and left_support >= index.minimum_exclusive_support
        and right_support >= index.minimum_exclusive_support
    )
    canonical_pair = f"{left_id}__VS__{right_id}"
    context_id = "MARGINAL" if context is None else context
    return {
        "contrast_id": f"contrast:{choice_id}:{canonical_pair}:{context_id}",
        "gene_id": gene_id,
        "choice_id": choice_id,
        "choice_kind": choice_kind,
        "contrast_kind": kind,
        "context_signature": context,
        "numerator_alternative_id": left_id,
        "denominator_alternative_id": right_id,
        "numerator_path_ids": left_paths,
        "denominator_path_ids": right_paths,
        "numerator_group_indicator": left_status["indicator"],
        "denominator_group_indicator": right_status["indicator"],
        "crossing_observational_group_ids": sorted(
            set(left_status["crossing_groups"]) | set(right_status["crossing_groups"])
        ),
        "numerator_rowspace_estimable": bool(left_status["rowspace_estimable"]),
        "denominator_rowspace_estimable": bool(right_status["rowspace_estimable"]),
        "numerator_train_exclusive_molecule_mass": left_support,
        "denominator_train_exclusive_molecule_mass": right_support,
        "cohort_local_contrast_separable": bool(
            left_status["defined"]
            and right_status["defined"]
            and left_status["rowspace_estimable"]
            and right_status["rowspace_estimable"]
        ),
        "cohort_reportable": cohort,
    }


def _subset_status(index, gene_id, subset_path_ids):
    gene = _gene_row(index, gene_id)
    subset = set(subset_path_ids)
    groups = index.groups.loc[index.groups["gene_id"].eq(gene_id)].sort_values(
        "observational_group_index"
    )
    indicator = []
    crossing = []
    for group in groups.itertuples(index=False):
        members = set(group.member_path_ids)
        overlap = subset & members
        if overlap and overlap != members:
            crossing.append(group.observational_group_id)
        indicator.append(int(overlap == members))
    defined = not crossing
    operator = np.asarray(gene["augmented_operator"], dtype=np.float64)
    estimable = defined and _indicator_in_row_space(
        indicator, operator, index.svd_relative_tolerance
    )
    return {
        "defined": defined,
        "indicator": indicator if defined else None,
        "crossing_groups": crossing,
        "rowspace_estimable": bool(estimable),
    }


def _indicator_in_row_space(indicator, operator, tolerance):
    vector = np.asarray(indicator, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[1] != len(vector):
        raise ValueError("compatibility operator and group indicator axes differ")
    projection = np.linalg.pinv(operator, rcond=tolerance) @ operator @ vector
    residual = float(np.linalg.norm(vector - projection))
    return residual <= tolerance * max(1.0, float(np.linalg.norm(vector)))


def _train_subset_support(index, gene_id, subset):
    patterns = index.train_patterns.loc[index.train_patterns["gene_id"].eq(gene_id)]
    return float(
        sum(
            row.molecule_mass
            for row in patterns.itertuples(index=False)
            if set(row.compatible_path_ids).issubset(subset)
        )
    )


def _compatible_indices(row, path_index: dict[str, int]) -> tuple[int, ...]:
    if isinstance(row, pd.Series) and "compatible_path_ids" in row:
        values = row["compatible_path_ids"]
    elif hasattr(row, "compatible_path_ids"):
        values = row.compatible_path_ids
    else:
        values = None
    if values is not None:
        ids = [str(value) for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("compatible_path_ids contains duplicates")
        unknown = sorted(set(ids) - set(path_index))
        if unknown:
            raise ValueError(f"compatible paths are absent from graph: {unknown[:5]}")
        return tuple(sorted(path_index[value] for value in ids))
    if isinstance(row, pd.Series) and "compatible_path_indices" in row:
        values = row["compatible_path_indices"]
    elif hasattr(row, "compatible_path_indices"):
        values = row.compatible_path_indices
    else:
        values = None
    if values is not None:
        indices = [int(value) for value in values]
        if len(indices) != len(set(indices)) or any(
            value < 0 or value >= len(path_index) for value in indices
        ):
            raise ValueError("compatible_path_indices is duplicate or out of range")
        return tuple(sorted(indices))
    raise ValueError("EC rows require compatible_path_ids or compatible_path_indices")


def _svd_rank(matrix: np.ndarray, relative_tolerance: float):
    values = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    sigma_max = float(singular[0]) if len(singular) else 0.0
    tolerance = relative_tolerance * sigma_max
    rank = int(np.count_nonzero(singular > tolerance)) if sigma_max > 0 else 0
    return rank, singular, tolerance


def _gene_row(index: PathIdentifiabilityIndex, gene_id: str) -> pd.Series:
    rows = index.genes.loc[index.genes["gene_id"].astype(str).eq(str(gene_id))]
    if len(rows) != 1:
        raise KeyError(f"gene {gene_id} is absent or duplicated in identifiability index")
    return rows.iloc[0]


def _as_graphs(graphs: GeneGraph | Sequence[GeneGraph]) -> tuple[GeneGraph, ...]:
    values = (graphs,) if isinstance(graphs, GeneGraph) else tuple(graphs)
    if not values:
        raise ValueError("at least one graph is required")
    genes = [str(graph.gene_id) for graph in values]
    if len(genes) != len(set(genes)):
        raise ValueError("graph gene IDs must be unique")
    return values


def _resolve_subset_indices(values, axis, label):
    ids = list(map(str, values))
    if not ids:
        raise ValueError(f"{label} path subset must be non-empty")
    unknown = sorted(set(ids) - set(axis))
    if unknown:
        raise ValueError(f"{label} paths are absent from axis: {unknown[:5]}")
    return [axis[value] for value in ids]


def _signature(kind: str, value: object) -> str:
    return f"{kind}:" + json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _numpy_logsumexp(values: np.ndarray, axis: int):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} misses columns: {missing}")
