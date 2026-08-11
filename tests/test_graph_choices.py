from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from fabric.choices import choice_identifiability, extract_elementary_choices
from fabric.graph import (
    GeneGraph,
    normalize_compatibility_path_order,
    validate_compatibility_rows,
)


def _choice_only_graph(
    gene_id: str, alternatives: tuple[tuple[str, ...], ...]
) -> GeneGraph:
    """Build a minimal graph whose legal paths are the supplied node sequences."""

    node_ids = tuple(dict.fromkeys(node for path in alternatives for node in path))
    node_types: dict[str, str] = {}
    for path in alternatives:
        for position, node_id in enumerate(path):
            node_type = "donor" if position % 2 == 0 else "acceptor"
            previous = node_types.setdefault(node_id, node_type)
            assert previous == node_type
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_type": [node_types[node_id] for node_id in node_ids],
        }
    )
    edge_pairs = tuple(
        dict.fromkeys(
            (left, right)
            for path in alternatives
            for left, right in zip(path[:-1], path[1:], strict=True)
        )
    )
    edge_ids = tuple(f"edge:{left}->{right}" for left, right in edge_pairs)
    edge_index = {pair: index for index, pair in enumerate(edge_pairs)}
    edges = pd.DataFrame(
        {
            "edge_id": edge_ids,
            "src_node_id": [left for left, _ in edge_pairs],
            "dst_node_id": [right for _, right in edge_pairs],
        }
    )
    path_edge_rows = tuple(
        tuple(edge_index[pair] for pair in zip(path[:-1], path[1:], strict=True))
        for path in alternatives
    )
    incidence_rows = [
        path_index
        for path_index, edge_row in enumerate(path_edge_rows)
        for _ in edge_row
    ]
    incidence_columns = [edge for edge_row in path_edge_rows for edge in edge_row]
    incidence = sparse.csr_matrix(
        (
            np.ones(len(incidence_rows), dtype=np.float32),
            (incidence_rows, incidence_columns),
        ),
        shape=(len(alternatives), len(edge_ids)),
    )
    return GeneGraph(
        gene_id=gene_id,
        nodes=nodes,
        edges=edges,
        paths=pd.DataFrame(),
        path_edges=pd.DataFrame(),
        edge_ids=edge_ids,
        path_ids=tuple(f"path:{index}" for index in range(len(alternatives))),
        path_edge_rows=path_edge_rows,
        path_node_rows=alternatives,
        path_edge_incidence=incidence,
        local_edge_index=np.empty((2, 0), dtype=np.int64),
        edge_features=np.empty((len(edge_ids), 0), dtype=np.float32),
    )


def test_elementary_choice_and_incidence_are_exact(toy_gene_graph):
    catalog = extract_elementary_choices(toy_gene_graph)
    assert len(catalog.choices) == 1
    choice = catalog.choices[0]
    assert choice.scope == "internal"
    assert len(choice.alternatives) == 2
    assert choice.path_to_alternative == (0, 1)
    np.testing.assert_array_equal(catalog.path_choice_incidence.toarray(), np.eye(2))


def test_unequal_length_exon_skipping_alternatives_share_one_exit():
    graph = _choice_only_graph(
        "ENSG_UNEQUAL",
        (
            ("entry", "included_acceptor", "included_donor", "exit"),
            ("entry", "exit"),
        ),
    )
    catalog = extract_elementary_choices(graph)

    assert len(catalog.choices) == 1
    choice = catalog.choices[0]
    assert (choice.entry_node_id, choice.exit_node_id) == ("entry", "exit")
    assert sorted(len(alternative.edge_ids) for alternative in choice.alternatives) == [
        1,
        3,
    ]
    assert sorted(choice.path_to_alternative) == [0, 1]


def test_staggered_arrival_multiway_alternatives_form_one_k4_choice():
    graph = _choice_only_graph(
        "ENSG_STAGGERED_K4",
        (
            ("entry", "long_1", "long_2", "long_3", "exit"),
            ("entry", "short_a", "exit"),
            ("entry", "short_b", "exit"),
            ("entry", "short_c", "exit"),
        ),
    )
    catalog = extract_elementary_choices(graph)

    assert len(catalog.choices) == 1
    choice = catalog.choices[0]
    assert len(choice.alternatives) == 4
    assert sorted(len(alternative.edge_ids) for alternative in choice.alternatives) == [
        2,
        2,
        2,
        4,
    ]
    assert sorted(choice.path_to_alternative) == [0, 1, 2, 3]


def test_structure_and_train_supervision_have_k_minus_one_rank(toy_gene_graph):
    catalog = extract_elementary_choices(toy_gene_graph)
    rows = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2"],
            "gene_id": [toy_gene_graph.gene_id] * 3,
            "compatible_path_ids": [["p0"], ["p1"], ["p0", "p1"]],
            "compatible_path_count": [1, 1, 2],
            "molecule_count": [3, 4, 20],
            "split": ["train", "train", "train"],
        }
    )
    audit = choice_identifiability(
        catalog,
        rows,
        rank_tolerance=1e-8,
        minimum_informative_molecule_mass=5,
        minimum_alternative_support=3,
    )
    assert audit.loc[0, "structural_rank"] == 1
    assert audit.loc[0, "supervision_rank"] == 1
    assert audit.loc[0, "informative_molecule_mass"] == 7
    # Each singleton EC informs both sides of the zero-sum binary contrast.
    assert audit.loc[0, "alternative_support"] == [7.0, 7.0]
    assert bool(audit.loc[0, "eligible"])


def test_choice_without_train_ec_is_explicitly_supervision_ineligible(
    toy_gene_graph,
):
    catalog = extract_elementary_choices(toy_gene_graph)
    empty_ec = pd.DataFrame(columns=["split", "molecule_count"])
    audit = choice_identifiability(
        catalog,
        empty_ec,
        rank_tolerance=1e-8,
        minimum_informative_molecule_mass=1,
        minimum_alternative_support=1,
    )

    assert audit.loc[0, "structural_rank"] == 1
    assert audit.loc[0, "supervision_rank"] == 0
    assert audit.loc[0, "informative_ec_count"] == 0
    assert audit.loc[0, "informative_molecule_mass"] == 0
    assert audit.loc[0, "alternative_support"] == [0.0, 0.0]
    assert not bool(audit.loc[0, "eligible"])


def test_external_ec_path_list_is_explicitly_reordered_to_path_table_axis(
    toy_gene_graph,
):
    rows = pd.DataFrame(
        {
            "cell_id": ["c0"],
            "gene_id": [toy_gene_graph.gene_id],
            "compatible_path_ids": [["p1", "p0"]],
            "compatible_path_count": [2],
            "molecule_count": [1],
            "split": ["train"],
        }
    )
    normalized = normalize_compatibility_path_order(rows, toy_gene_graph)
    assert normalized.loc[0, "source_compatible_path_ids"] == ["p1", "p0"]
    assert normalized.loc[0, "compatible_path_ids"] == ["p0", "p1"]
    assert normalized.loc[0, "compatible_path_indices"] == [0, 1]
    validate_compatibility_rows(normalized, toy_gene_graph)
