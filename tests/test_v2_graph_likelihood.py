from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fabric.graph import build_gene_graph
from fabric.likelihood import (
    brute_force_compatible_path_nll,
    compatible_path_nll,
)


def _rebuild_with_alias(graph, *, duplicate_transcript_id: bool = False):
    paths = graph.paths.copy()
    alias = paths.iloc[1].copy()
    alias["path_id"] = "p_alias"
    alias["transcript_id"] = (
        str(paths.iloc[0]["transcript_id"])
        if duplicate_transcript_id
        else "tx_alias"
    )
    paths = pd.concat([paths, alias.to_frame().T], ignore_index=True)
    path_edges = graph.path_edges.copy()
    alias_edges = path_edges.loc[path_edges["path_id"].astype(str) == "p1"].copy()
    alias_edges["path_id"] = "p_alias"
    alias_edges["transcript_id"] = alias["transcript_id"]
    path_edges = pd.concat([path_edges, alias_edges], ignore_index=True)
    return build_gene_graph(
        graph.gene_id,
        nodes=graph.nodes,
        edges=graph.edges,
        paths=paths,
        path_edges=path_edges,
    )


def test_structural_paths_collapse_identical_transcripts_and_keep_aliases(
    toy_gene_graph,
):
    graph = _rebuild_with_alias(toy_gene_graph)

    assert graph.path_ids == ("p0", "p1")
    assert graph.transcript_aliases == (("tx_p0",), ("tx_alias", "tx_p1"))
    assert graph.paths.loc[1, "source_path_ids"] == ["p1", "p_alias"]
    assert graph.path_edge_incidence.shape[0] == 2
    np.testing.assert_array_equal(
        graph.path_edge_incidence.toarray(),
        toy_gene_graph.path_edge_incidence.toarray(),
    )


def test_transcript_identity_cannot_point_to_two_structural_paths(toy_gene_graph):
    with pytest.raises(ValueError, match="transcript transcript_id"):
        _rebuild_with_alias(toy_gene_graph, duplicate_transcript_id=True)


def test_path_edge_transcript_identity_drift_fails_before_structural_collapse(
    toy_gene_graph,
):
    path_edges = toy_gene_graph.path_edges.copy()
    path_edges.loc[path_edges["path_id"].astype(str).eq("p1"), "transcript_id"] = (
        "wrong_transcript"
    )
    with pytest.raises(ValueError, match="transcript identity differs"):
        build_gene_graph(
            toy_gene_graph.gene_id,
            nodes=toy_gene_graph.nodes,
            edges=toy_gene_graph.edges,
            paths=toy_gene_graph.paths,
            path_edges=path_edges,
        )


def test_alias_with_mismatched_declared_endpoint_fails_before_collapse(toy_gene_graph):
    paths = toy_gene_graph.paths.copy()
    alias = paths.iloc[1].copy()
    alias["path_id"] = "p_alias"
    alias["transcript_id"] = "tx_alias"
    alias["tss_node_id"] = "mismatched_tss"
    paths = pd.concat([paths, alias.to_frame().T], ignore_index=True)
    path_edges = toy_gene_graph.path_edges.copy()
    alias_edges = path_edges.loc[path_edges["path_id"].astype(str).eq("p1")].copy()
    alias_edges["path_id"] = "p_alias"
    alias_edges["transcript_id"] = "tx_alias"
    path_edges = pd.concat([path_edges, alias_edges], ignore_index=True)
    with pytest.raises(ValueError, match="declared TSS before collapse"):
        build_gene_graph(
            toy_gene_graph.gene_id,
            nodes=toy_gene_graph.nodes,
            edges=toy_gene_graph.edges,
            paths=paths,
            path_edges=path_edges,
        )


def test_extra_path_edge_identity_is_not_silently_dropped(toy_gene_graph):
    extra = toy_gene_graph.path_edges.iloc[[0]].copy()
    extra["path_id"] = "unknown_path"
    extra["transcript_id"] = "unknown_transcript"
    with pytest.raises(ValueError, match="path and path-edge table identities differ"):
        build_gene_graph(
            toy_gene_graph.gene_id,
            nodes=toy_gene_graph.nodes,
            edges=toy_gene_graph.edges,
            paths=toy_gene_graph.paths,
            path_edges=pd.concat([toy_gene_graph.path_edges, extra], ignore_index=True),
        )


def test_local_adjacency_contains_only_legal_consecutive_pairs_and_no_new_paths(
    toy_gene_graph,
):
    expected = {
        pair
        for path in toy_gene_graph.path_edge_rows
        for adjacent in zip(path[:-1], path[1:], strict=True)
        for pair in (adjacent, adjacent[::-1])
    }
    observed = set(map(tuple, toy_gene_graph.local_edge_index.T.tolist()))
    assert observed == expected
    assert toy_gene_graph.path_ids == ("p0", "p1")


def test_path_endpoint_and_complexity_audit_fields_are_catalog_exact(toy_gene_graph):
    graph = toy_gene_graph
    assert graph.path_first_edge_indices == tuple(row[0] for row in graph.path_edge_rows)
    assert graph.path_last_edge_indices == tuple(row[-1] for row in graph.path_edge_rows)
    np.testing.assert_allclose(graph.path_log_edge_count, np.log1p([5, 5]))
    assert graph.variable_edge_count == 4
    assert graph.centered_path_incidence_energy == pytest.approx(1.0)


def test_full_path_rows_are_audit_only_and_do_not_dilute_nll_denominator():
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    compatible = torch.tensor([[0, -1], [0, 1]])
    mask = compatible >= 0
    counts = torch.tensor([2.0, 100.0], dtype=torch.float64)

    details = compatible_path_nll(
        logits,
        compatible,
        mask,
        counts,
        row_cell_index=torch.tensor([0, 0]),
        return_details=True,
    )
    reference = brute_force_compatible_path_nll(
        logits,
        [[0], [0, 1]],
        counts,
        row_cell_index=torch.tensor([0, 0]),
    )

    expected = -torch.log_softmax(logits, dim=1)[0, 0]
    torch.testing.assert_close(details.loss, expected)
    torch.testing.assert_close(details.loss, reference)
    torch.testing.assert_close(details.molecule_mass, torch.tensor(2.0, dtype=torch.float64))
    assert details.informative_row_mask.tolist() == [True, False]
    details.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_compatible_set_duplicates_fail_instead_of_changing_probability_mass():
    with pytest.raises(ValueError, match="duplicate path indices"):
        compatible_path_nll(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0, 0]]),
            torch.ones((1, 2), dtype=torch.bool),
            torch.ones(1),
        )


def test_likelihood_rejects_nonfinite_mass_and_ignores_masked_padding_value():
    with pytest.raises(ValueError, match="molecule_count must be finite"):
        compatible_path_nll(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0]]),
            torch.ones((1, 1), dtype=torch.bool),
            torch.tensor([float("nan")]),
        )

    observed = compatible_path_nll(
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[99, 0]]),
        torch.tensor([[False, True]]),
        torch.ones(1),
    )
    expected = -torch.log_softmax(torch.tensor([[0.0, 1.0]]), dim=1)[0, 0]
    torch.testing.assert_close(observed, expected)


def test_likelihood_rejects_nonfinite_logits_before_softmax():
    with pytest.raises(ValueError, match="path_logits must be finite"):
        compatible_path_nll(
            torch.tensor([[0.0, float("inf")]]),
            torch.tensor([[0]]),
            torch.ones((1, 1), dtype=torch.bool),
            torch.ones(1),
        )
