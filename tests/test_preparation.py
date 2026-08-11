from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from fabric.choices import choice_identifiability, extract_elementary_choices
from fabric.dataset import (
    OrderedCellState,
    OrderedChoiceAudit,
    OrderedEventData,
    PreparationSources,
    prepare_gene,
)
from fabric.motifs import event_relation_matrix
from fabric.train import (
    NORMALIZED_SOURCE_ROLES,
    PreparedGene,
    _validate_prepared_genes,
    load_config,
    make_toy_genes,
    prepare_dataset_identity,
    preparation_values_from_config,
)


def _preparation_inputs(graph):
    catalog = extract_elementary_choices(graph)
    cell_ids = ("c2", "c0", "c1")
    state = OrderedCellState(
        cell_ids=cell_ids,
        values=np.asarray([[2.0, 0.2], [4.0, 0.4], [-3.0, -0.3]], dtype=np.float32),
    )
    ec_rows = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2"],
            "gene_id": [graph.gene_id] * 3,
            "compatible_path_ids": [["p0"], ["p1"], ["p0", "p1"]],
            "compatible_path_indices": [[0], [1], [0, 1]],
            "compatible_path_count": [1, 1, 2],
            "molecule_count": [3, 4, 5],
            "split": ["train", "train", "val"],
        }
    )
    identifiability = choice_identifiability(
        catalog,
        ec_rows,
        rank_tolerance=1e-8,
        minimum_informative_molecule_mass=5,
        minimum_alternative_support=3,
    )
    choice = catalog.choices[0]
    alternative_ids = tuple(
        alternative.alternative_id for alternative in choice.alternatives
    )

    def events(modality: str, feature_width: int) -> OrderedEventData:
        prefix = modality.lower()
        event_ids = (f"{prefix}_event_0", f"{prefix}_event_1")
        table = pd.DataFrame(
            {
                "event_id": event_ids,
                "modality": [modality, modality],
                "gene_id": [graph.gene_id, graph.gene_id],
                "choice_id": [choice.choice_id, choice.choice_id],
                "relation_alternative_ids": [
                    [alternative_ids[0]],
                    [alternative_ids[1]],
                ],
            }
        )
        relation = event_relation_matrix(table, alternative_ids)
        return OrderedEventData(
            events=table,
            feature_event_ids=event_ids,
            features=np.arange(
                len(event_ids) * feature_width, dtype=np.float32
            ).reshape(len(event_ids), feature_width),
            relation_event_ids=event_ids,
            relation_alternative_ids=alternative_ids,
            relation=relation,
            gate_cell_ids=cell_ids,
            gate_event_ids=event_ids,
            gate=np.asarray([[0.1, -0.1], [0.4, -0.4], [-0.3, 0.3]], dtype=np.float32),
        )

    return (
        catalog,
        ec_rows,
        state,
        events("DNA", 4),
        events("RNA", 3),
        identifiability,
    )


def _choice_audit(catalog) -> OrderedChoiceAudit:
    count = len(catalog.choices)
    return OrderedChoiceAudit(
        choice_ids=tuple(choice.choice_id for choice in catalog.choices),
        alternative_span=np.full(count, 2.0, dtype=np.float32),
        dna_candidate_event_count=np.full(count, 2, dtype=np.int64),
        dna_selected_event_count=np.full(count, 2, dtype=np.int64),
        dna_cap_saturated=np.zeros(count, dtype=bool),
        dna_boundary_rank_motif_score=np.full(count, 0.8, dtype=np.float32),
        rna_candidate_event_count=np.full(count, 2, dtype=np.int64),
        rna_selected_event_count=np.full(count, 2, dtype=np.int64),
        rna_cap_saturated=np.zeros(count, dtype=bool),
        rna_boundary_rank_motif_score=np.full(count, 0.8, dtype=np.float32),
    )


def test_prepare_gene_builds_id_bound_training_tensors(toy_gene_graph, tmp_path):
    catalog, ec_rows, state, dna, rna, identifiability = _preparation_inputs(
        toy_gene_graph
    )
    prepared = prepare_gene(
        toy_gene_graph,
        catalog,
        ec_rows,
        state=state,
        dna=dna,
        rna=rna,
        choice_identifiability=identifiability,
        choice_audit=_choice_audit(catalog),
        sources=PreparationSources(
            graph_generation="toy_graph_generation",
            split_source="toy_split_source",
        ),
    )

    assert isinstance(prepared, PreparedGene)
    assert prepared.cell_ids == ("c2", "c0", "c1")
    assert prepared.path_ids == ("p0", "p1")
    assert prepared.row_cell_index.tolist() == [1, 2, 0]
    assert prepared.compatible_path_indices.tolist() == [[0, -1], [1, -1], [0, 1]]
    assert prepared.compatible_path_mask.tolist() == [
        [True, False],
        [True, False],
        [True, True],
    ]
    assert prepared.identifiable_row_mask.tolist() == [True, True, False]
    assert prepared.alternative_eligible.tolist() == [True, True]
    assert prepared.dna_event_ids == ("dna_event_0", "dna_event_1")
    assert prepared.rna_event_ids == ("rna_event_0", "rna_event_1")
    torch.testing.assert_close(
        prepared.path_edge_incidence.to_dense(),
        torch.tensor(toy_gene_graph.path_edge_incidence.toarray()),
    )
    torch.testing.assert_close(
        prepared.path_choice_incidence.to_dense(),
        torch.eye(2),
    )
    torch.testing.assert_close(prepared.dna_event_relation, torch.eye(2))
    assert prepared.graph_generation == "toy_graph_generation"
    assert prepared.split_source == "toy_split_source"
    _validate_prepared_genes([prepared])
    dataset = prepare_dataset_identity([prepared], factor_mapping_reviewed=False)
    assert dataset.target_gene_ids == (toy_gene_graph.gene_id,)
    assert dataset.graph_generation == "toy_graph_generation"
    with pytest.raises(ValueError, match="without normalized source identity"):
        prepare_dataset_identity([prepared], factor_mapping_reviewed=True)

    source_paths = {role: tmp_path / role for role in NORMALIZED_SOURCE_ROLES}
    reviewed_mapping = tmp_path / "reviewed_factor_mapping.tsv"
    donor_eligibility = tmp_path / "atac_donor_eligibility.tsv"
    peak_support = tmp_path / "peak_support.tsv"
    preparation_config = tmp_path / "preparation.yaml"
    preparation_values = preparation_values_from_config(
        load_config("configs/fabric_v1_toy.yaml")
    )
    dataset = prepare_dataset_identity(
        [prepared],
        factor_mapping_reviewed=True,
        normalized_source_paths=source_paths,
        reviewed_factor_mapping=reviewed_mapping,
        atac_donor_eligibility_source=donor_eligibility,
        peak_support_source=peak_support,
        preparation_config_source=preparation_config,
        preparation_values=preparation_values,
    )
    assert dict(dataset.normalized_source_paths) == {
        role: str(path.resolve()) for role, path in source_paths.items()
    }
    assert dataset.reviewed_factor_mapping == str(reviewed_mapping.resolve())
    assert dataset.genes[0].normalized_source_paths == dataset.normalized_source_paths
    assert dataset.genes[0].reviewed_factor_mapping == dataset.reviewed_factor_mapping
    assert dataset.atac_donor_eligibility_source == str(donor_eligibility.resolve())
    assert dataset.peak_support_source == str(peak_support.resolve())
    assert dataset.preparation_config_source == str(preparation_config.resolve())
    assert dict(dataset.preparation_values) == preparation_values


def test_prepare_gene_rejects_same_shape_reordered_event_axis(toy_gene_graph):
    catalog, ec_rows, state, dna, rna, identifiability = _preparation_inputs(
        toy_gene_graph
    )
    reordered_gate_axis = replace(
        dna, gate_event_ids=tuple(reversed(dna.gate_event_ids))
    )
    with pytest.raises(ValueError, match="DNA gate event ID order"):
        prepare_gene(
            toy_gene_graph,
            catalog,
            ec_rows,
            state=state,
            dna=reordered_gate_axis,
            rna=rna,
            choice_identifiability=identifiability,
            choice_audit=_choice_audit(catalog),
            sources=PreparationSources("toy_graph_generation", "toy_split_source"),
        )


def test_prepare_gene_rejects_uncentered_state(toy_gene_graph):
    catalog, ec_rows, state, dna, rna, identifiability = _preparation_inputs(
        toy_gene_graph
    )
    shifted = replace(state, values=state.values + 1.0)
    with pytest.raises(ValueError, match="State is not centered"):
        prepare_gene(
            toy_gene_graph,
            catalog,
            ec_rows,
            state=shifted,
            dna=dna,
            rna=rna,
            choice_identifiability=identifiability,
            choice_audit=_choice_audit(catalog),
            sources=PreparationSources("toy_graph_generation", "toy_split_source"),
        )


def test_prepare_gene_rejects_same_set_reordered_gate_cell_axis(toy_gene_graph):
    catalog, ec_rows, state, dna, rna, identifiability = _preparation_inputs(
        toy_gene_graph
    )
    reordered_cell_axis = replace(dna, gate_cell_ids=tuple(reversed(dna.gate_cell_ids)))
    with pytest.raises(ValueError, match="DNA gate cell ID order"):
        prepare_gene(
            toy_gene_graph,
            catalog,
            ec_rows,
            state=state,
            dna=reordered_cell_axis,
            rna=rna,
            choice_identifiability=identifiability,
            choice_audit=_choice_audit(catalog),
            sources=PreparationSources("toy_graph_generation", "toy_split_source"),
        )


def test_prepare_gene_rejects_noncanonical_ec_path_order(toy_gene_graph):
    catalog, ec_rows, state, dna, rna, identifiability = _preparation_inputs(
        toy_gene_graph
    )
    misordered = ec_rows.copy()
    misordered.at[2, "compatible_path_ids"] = ["p1", "p0"]
    with pytest.raises(ValueError, match="canonical path order"):
        prepare_gene(
            toy_gene_graph,
            catalog,
            misordered,
            state=state,
            dna=dna,
            rna=rna,
            choice_identifiability=identifiability,
            choice_audit=_choice_audit(catalog),
            sources=PreparationSources("toy_graph_generation", "toy_split_source"),
        )


def test_explicit_zero_ec_gene_is_retained_but_has_no_eligible_choice():
    gene = make_toy_genes()[0]
    empty = replace(
        gene,
        alternative_eligible=torch.zeros_like(gene.alternative_eligible),
        state_features=gene.state_features[:0],
        dna_gate=gene.dna_gate[:0],
        rna_gate=gene.rna_gate[:0],
        compatible_path_indices=gene.compatible_path_indices[:0],
        compatible_path_mask=gene.compatible_path_mask[:0],
        row_cell_index=gene.row_cell_index[:0],
        molecule_count=gene.molecule_count[:0],
        split=(),
        identifiable_row_mask=gene.identifiable_row_mask[:0],
        cell_ids=(),
    )
    _validate_prepared_genes([empty])
    dataset = prepare_dataset_identity([empty], factor_mapping_reviewed=False)
    assert dataset.target_gene_ids == (gene.gene_id,)

    with pytest.raises(ValueError, match="eligible choices without train"):
        _validate_prepared_genes(
            [replace(empty, alternative_eligible=gene.alternative_eligible)]
        )
