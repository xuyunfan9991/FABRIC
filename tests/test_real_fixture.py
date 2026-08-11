from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fabric.annotation import canonical_rna_cell_id, load_split_rows
from fabric.choices import choice_identifiability, extract_elementary_choices
from fabric.graph import load_graph_tables, split_gene_graphs
from fabric.likelihood import (
    brute_force_compatible_path_nll,
    compatible_path_nll,
)


FIXTURE = Path(__file__).parent / "fixtures" / "real"
GENE_ID = "ENSG00000275074"
PATH_RETAINED = "path:ENSG00000275074:ENST00000613958"
PATH_SPLICED = "path:ENSG00000275074:ENST00000611621"
CHOICE_ENTRY = "node:ENSG00000275074:donor:chr8:22108132:-"
CHOICE_EXIT = "node:ENSG00000275074:acceptor:chr8:22107895:-"


def _fixture_objects():
    tables = load_graph_tables(FIXTURE / "graph_generation")
    graphs = list(split_gene_graphs(tables))
    assert len(graphs) == 1
    ec = pd.read_parquet(FIXTURE / "compatibility_equivalence_classes.parquet")
    return tables, graphs[0], ec


def test_real_fixture_filters_import_scaffolding_and_rebinds_current_split():
    metadata = json.loads((FIXTURE / "fixture.json").read_text())
    assert metadata["gene_id"] == GENE_ID
    assert metadata["counts"] == {
        "ec_molecules": 66,
        "ec_rows": 66,
        "edges": 7,
        "excluded_ec_rows_outside_current_split": 31,
        "nodes": 7,
        "path_edges": 10,
        "paths": 2,
        "source_ec_molecules": 98,
        "source_ec_rows": 97,
        "source_edges": 10,
        "source_nodes": 9,
        "source_path_edges": 14,
        "source_paths": 2,
        "stale_split_mismatch_rows": 10,
        "test_rows": 5,
        "train_rows": 56,
        "val_rows": 5,
    }
    tables, graph, ec = _fixture_objects()
    assert set(tables.nodes["node_type"]) == {"TSS", "donor", "acceptor", "PAS"}
    assert not set(tables.edges["edge_type"]) & {"START", "END"}
    assert set(tables.edges["edge_id"]) == set(tables.path_edges["edge_id"])
    assert graph.path_ids == (PATH_RETAINED, PATH_SPLICED)
    assert graph.path_edge_incidence.shape == (2, 7)
    assert graph.path_edge_incidence.nnz == 10
    for path_id, rows in tables.path_edges.groupby("path_id", sort=False):
        ordered = rows.sort_values("edge_order", kind="mergesort")
        np.testing.assert_array_equal(ordered["edge_order"], np.arange(5))
        assert int(tables.paths.set_index("path_id").loc[path_id, "n_edges"]) == 5

    source_split = pd.read_parquet(FIXTURE / "split_rows.parquet")
    assert source_split["cell_id"].str.startswith("RNA__").all()
    split = load_split_rows(FIXTURE / "split_rows.parquet")
    assert set(split["cell_id"]) == set(ec["cell_id"])
    authority = split.set_index("cell_id")["split"]
    assert ec["cell_id"].map(authority).equals(ec["split"])
    assert int((ec["source_ec_split"] != ec["split"]).sum()) == 10
    assert ec["split"].value_counts().to_dict() == {"train": 56, "test": 5, "val": 5}
    assert ec.groupby("split")["molecule_count"].sum().to_dict() == {
        "test": 5,
        "train": 56,
        "val": 5,
    }
    assert [
        canonical_rna_cell_id(value) for value in source_split["cell_id"]
    ] == sorted(ec["cell_id"])


def test_real_fixture_negative_strand_choice_rank_support_and_exact_nll():
    metadata = json.loads((FIXTURE / "fixture.json").read_text())
    tables, graph, ec = _fixture_objects()
    node_position = tables.nodes.set_index("node_id")["pos_0based"]
    for _, rows in tables.path_edges.groupby("path_id", sort=False):
        ordered = rows.sort_values("edge_order", kind="mergesort")
        src = ordered["src_node_id"].map(node_position).to_numpy()
        dst = ordered["dst_node_id"].map(node_position).to_numpy()
        assert np.all(src > dst), (
            "negative-strand path order must run toward lower coordinates"
        )

    catalog = extract_elementary_choices(graph)
    assert len(catalog.choices) == 1
    choice = catalog.choices[0]
    assert (choice.entry_node_id, choice.exit_node_id) == (CHOICE_ENTRY, CHOICE_EXIT)
    assert choice.scope == "internal"
    assert choice.path_to_alternative == (0, 1)
    assert {
        graph.edges.iloc[alt.edge_indices[0]]["edge_type"]
        for alt in choice.alternatives
    } == {
        "RETAINED_INTRON",
        "SPLICE",
    }
    assert all(len(alt.edge_indices) == 1 for alt in choice.alternatives)
    np.testing.assert_array_equal(catalog.path_choice_incidence.toarray(), np.eye(2))

    audit = choice_identifiability(
        catalog,
        ec,
        rank_tolerance=1e-8,
        minimum_informative_molecule_mass=1,
        minimum_alternative_support=1,
    )
    assert audit.loc[0, "structural_rank"] == 1
    assert audit.loc[0, "supervision_rank"] == 1
    assert audit.loc[0, "informative_ec_count"] == 56
    assert audit.loc[0, "informative_molecule_mass"] == 56
    assert audit.loc[0, "alternative_support"] == [56.0, 56.0]
    assert bool(audit.loc[0, "eligible"])
    train_singleton_mass = (
        ec.loc[ec["split"] == "train"]
        .groupby("compatible_path_ids_key")["molecule_count"]
        .sum()
        .to_dict()
    )
    assert train_singleton_mass == {PATH_SPLICED: 40, PATH_RETAINED: 16}

    path_index = {path_id: index for index, path_id in enumerate(graph.path_ids)}
    compatible_lists = [
        [path_index[str(path_id)] for path_id in values]
        for values in ec["compatible_path_ids"]
    ]
    compatible = torch.tensor(compatible_lists, dtype=torch.long)
    mask = torch.ones_like(compatible, dtype=torch.bool)
    reference = metadata["fixed_logit_likelihood_reference"]
    logits = torch.tensor(
        [[reference["path_logits"][path_id] for path_id in graph.path_ids]],
        dtype=torch.float64,
    ).repeat(len(ec), 1)
    weights = torch.from_numpy(ec["molecule_count"].to_numpy(dtype=np.float64))
    row_cell_index = torch.arange(len(ec), dtype=torch.long)
    observed = compatible_path_nll(
        logits,
        compatible,
        mask,
        weights,
        row_cell_index=row_cell_index,
        return_details=True,
    )
    brute_force = brute_force_compatible_path_nll(
        logits,
        compatible_lists,
        weights,
        row_cell_index=row_cell_index,
    )
    torch.testing.assert_close(observed.loss, brute_force, atol=1e-12, rtol=1e-12)
    expected_rows = torch.tensor(
        [reference["row_nll"][str(values[0])] for values in ec["compatible_path_ids"]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        observed.per_row_nll, expected_rows, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        observed.loss,
        torch.tensor(reference["molecule_weighted_mean_nll"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert observed.informative_row_mask.all()
