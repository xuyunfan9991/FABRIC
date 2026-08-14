from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fabric.choices import (
    SUPPORT_TIER_DIRECT,
    aggregate_group_log_probabilities,
    aggregate_group_probabilities,
    alternative_coverage_tables,
    alternative_relative_log_mass,
    build_alternative_reporting_index,
    build_path_identifiability_index,
    centered_logit_change,
    classify_heldout_alternative_support,
    classify_heldout_path_support,
)
from fabric.graph import build_gene_graph


def _ec(graph, rows):
    return pd.DataFrame(
        [
            {
                "cell_id": cell,
                "gene_id": graph.gene_id,
                "compatible_path_ids": paths,
                "compatible_path_count": len(paths),
                "molecule_count": mass,
                "split": split,
            }
            for cell, paths, mass, split in rows
        ]
    )


def _single_path_graph(graph):
    path_id = graph.path_ids[0]
    path_edges = graph.path_edges.loc[graph.path_edges["path_id"].astype(str).eq(path_id)]
    edge_ids = set(path_edges["edge_id"].astype(str))
    edges = graph.edges.loc[graph.edges["edge_id"].astype(str).isin(edge_ids)]
    node_ids = set(edges["src_node_id"].astype(str)) | set(edges["dst_node_id"].astype(str))
    nodes = graph.nodes.loc[graph.nodes["node_id"].astype(str).isin(node_ids)]
    paths = graph.paths.loc[graph.paths["path_id"].astype(str).eq(path_id)]
    return build_gene_graph(
        graph.gene_id,
        nodes=nodes,
        edges=edges,
        paths=paths,
        path_edges=path_edges,
    )


def test_train_patterns_define_observational_groups_and_duplicate_rows_only_add_mass(
    toy_gene_graph,
):
    rows = _ec(
        toy_gene_graph,
        [
            ("t0", ["p0"], 2, "train"),
            ("t1", ["p0"], 3, "train"),
            ("t2", ["p1"], 7, "train"),
            ("t3", ["p0", "p1"], 99, "train"),
            ("v0", ["p0"], 1000, "val"),
        ],
    )
    index = build_path_identifiability_index(toy_gene_graph, rows)

    gene = index.genes.iloc[0]
    assert gene["train_compatibility_matrix"] == [[1, 0], [0, 1]]
    assert gene["augmented_rank"] == gene["group_count"] == 2
    assert bool(gene["cohort_contrast_separable"])
    assert len(index.train_patterns) == 2
    assert sorted(index.train_patterns["molecule_mass"]) == [5.0, 7.0]
    assert sorted(index.groups["train_exclusive_molecule_mass"]) == [5.0, 7.0]


def test_identical_train_columns_form_one_group_and_split_group_row_cannot_upgrade_cell(
    toy_gene_graph,
):
    graph = toy_gene_graph
    # No informative train row distinguishes p0 from p1, so they form one group.
    train = _ec(graph, [("t0", ["p0", "p1"], 5, "train")])
    index = build_path_identifiability_index(graph, train)
    assert index.genes.iloc[0]["group_count"] == 1
    assert index.groups.iloc[0]["member_path_ids"] == ["p0", "p1"]

    heldout = _ec(graph, [("v0", ["p0"], 9, "val")])
    cells, classified = classify_heldout_path_support(index, heldout)
    assert bool(classified.iloc[0]["novel_split_group_row"])
    assert not bool(classified.iloc[0]["group_constant"])
    # The only train row was all-path and therefore audit-only: there is no
    # train exclusive support from which to claim cohort identifiability.
    assert cells.iloc[0]["support_tier"] == "supervision_unidentifiable_prediction"
    assert not bool(cells.iloc[0]["direct_cell_supported"])

    reporting = build_alternative_reporting_index(graph, index)
    assert not reporting.contrasts["cohort_reportable"].any()
    assert reporting.contrasts["crossing_observational_group_ids"].map(bool).all()


def test_heldout_direct_support_group_sums_and_reporting_coverage(toy_gene_graph):
    train = _ec(
        toy_gene_graph,
        [("t0", ["p0"], 3, "train"), ("t1", ["p1"], 4, "train")],
    )
    index = build_path_identifiability_index(toy_gene_graph, train)
    heldout = _ec(
        toy_gene_graph,
        [("v0", ["p0"], 2, "val"), ("v0", ["p1"], 5, "val")],
    )
    cells, _ = classify_heldout_path_support(index, heldout)
    assert cells.iloc[0]["support_tier"] == SUPPORT_TIER_DIRECT
    assert cells.iloc[0]["group_exclusive_molecule_mass"] == [2.0, 5.0]

    probabilities = torch.tensor([[0.2, 0.8]])
    torch.testing.assert_close(
        aggregate_group_probabilities(probabilities, index, toy_gene_graph.gene_id),
        probabilities,
    )
    torch.testing.assert_close(
        aggregate_group_log_probabilities(probabilities.log(), index, toy_gene_graph.gene_id).exp(),
        probabilities,
    )

    reporting = build_alternative_reporting_index(toy_gene_graph, index)
    assert reporting.choices["choice_kind"].tolist() == ["internal"]
    assert reporting.choices.iloc[0]["n_matched_context_candidates"] == 1
    support = classify_heldout_alternative_support(reporting, index, heldout)
    assert support["direct_cell_supported"].all()
    manifest = support.loc[
        support["contrast_kind"].eq("matched_context"),
        ["cell_id", "gene_id", "contrast_id"],
    ]
    coverage = alternative_coverage_tables(
        reporting, support, manifest_selected=manifest
    )
    final_choice = coverage["choice_level"].query(
        "choice_scope == 'internal' and waterfall_level == "
        "'has_direct_and_manifest_selected_heldout_record'"
    )
    assert final_choice.iloc[0]["choice_fraction"] == 1.0
    final_record = coverage["record_level"].query(
        "choice_scope == 'internal' and waterfall_level == "
        "'direct_and_manifest_selected'"
    )
    validation_record = final_record.query("split == 'val'").iloc[0]
    assert validation_record["record_denominator"] == 1
    assert validation_record["record_fraction"] == 1.0
    test_record = final_record.query("split == 'test'").iloc[0]
    assert test_record["record_denominator"] == 0
    assert test_record["status"] == "not_estimable"
    assert np.isnan(test_record["record_fraction"])


def test_marginal_and_matched_log_mass_are_independently_gauge_invariant(
    toy_gene_graph,
):
    index = build_path_identifiability_index(
        toy_gene_graph,
        _ec(
            toy_gene_graph,
            [("t0", ["p0"], 1, "train"), ("t1", ["p1"], 1, "train")],
        ),
    )
    reporting = build_alternative_reporting_index(toy_gene_graph, index)
    for contrast in reporting.contrasts.itertuples(index=False):
        full = torch.tensor([[1.5, -0.25]])
        counterfactual = torch.tensor([[0.2, 0.4]])
        full_rho = alternative_relative_log_mass(
            full,
            toy_gene_graph.path_ids,
            contrast.numerator_path_ids,
            contrast.denominator_path_ids,
        )
        shifted_full_rho = alternative_relative_log_mass(
            full + 100.0,
            toy_gene_graph.path_ids,
            contrast.numerator_path_ids,
            contrast.denominator_path_ids,
        )
        counter_rho = alternative_relative_log_mass(
            counterfactual,
            toy_gene_graph.path_ids,
            contrast.numerator_path_ids,
            contrast.denominator_path_ids,
        )
        shifted_counter_rho = alternative_relative_log_mass(
            counterfactual - 30.0,
            toy_gene_graph.path_ids,
            contrast.numerator_path_ids,
            contrast.denominator_path_ids,
        )
        torch.testing.assert_close(full_rho, shifted_full_rho)
        torch.testing.assert_close(counter_rho, shifted_counter_rho)
    centered = centered_logit_change(
        np.array([[2.0, 1.0]]), np.array([[0.0, 0.0]])
    )
    np.testing.assert_allclose(centered, [[0.5, -0.5]])


def test_single_path_catalog_has_stable_empty_reporting_schemas(toy_gene_graph):
    graph = _single_path_graph(toy_gene_graph)
    index = build_path_identifiability_index(
        graph,
        _ec(graph, [("t0", list(graph.path_ids), 3, "train")]),
    )
    reporting = build_alternative_reporting_index(graph, index)
    assert reporting.choices.empty
    assert reporting.alternatives.empty
    assert reporting.path_membership.empty
    assert reporting.contrasts.empty
    assert "choice_id" in reporting.choices
    assert "contrast_id" in reporting.contrasts
