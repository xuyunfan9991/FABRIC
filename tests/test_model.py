from __future__ import annotations

import torch

from fabric.model import (
    AugmentedPathReadout,
    EdgeGraphGPS,
    EventBatch,
    EventScorer,
    PathReadoutInput,
    StateBatch,
    StateScorer,
    build_frozen_alternative_state,
    freeze_cis_parent,
)
from fabric.train import make_toy_genes


def test_zero_initialized_children_recover_parent_and_keep_frozen_h_base():
    gene = make_toy_genes()[0]
    cis = EdgeGraphGPS(gene.graph.edge_features.shape[1], 8, 2)
    cis_output = cis(gene.graph)
    freeze_cis_parent(cis)
    frozen = build_frozen_alternative_state(cis_output.edge_states, gene.alternatives)
    assert not frozen.edge_states.requires_grad
    assert not frozen.h_base.requires_grad
    assert all(not parameter.requires_grad for parameter in cis.parameters())

    state = StateScorer(4, frozen.h_base.shape[1], 3)
    state_output = state(StateBatch(gene.state_features[:3]), frozen)
    assert torch.count_nonzero(state_output.correction) == 0
    event = EventScorer(6, frozen.h_base.shape[1], 3)
    event_output = event(
        EventBatch(
            gene.dna_event_features,
            gene.dna_event_relation,
            gene.dna_event_choice_index,
            gene.dna_gate[:3],
        ),
        frozen,
    )
    assert torch.count_nonzero(event_output.correction) == 0

    readout = AugmentedPathReadout(length_penalty=0.0)
    zero = torch.zeros((3, 2))
    paths = readout(
        PathReadoutInput(
            cis_output.edge_energy,
            state_output.correction,
            event_output.correction,
            zero,
            gene.path_edge_incidence,
            gene.path_choice_incidence,
        )
    )
    torch.testing.assert_close(paths.total_logits, paths.edge_logits)


def test_choice_centering_and_sparse_readout_match_manual_sums():
    gene = make_toy_genes()[0]
    cis = EdgeGraphGPS(gene.graph.edge_features.shape[1], 8, 2)
    output = cis(gene.graph)
    frozen = build_frozen_alternative_state(output.edge_states, gene.alternatives)
    scorer = EventScorer(6, frozen.h_base.shape[1], 2)
    torch.nn.init.constant_(scorer.U.weight, 0.2)
    torch.nn.init.constant_(scorer.V.weight, 0.1)
    batch = EventBatch(
        gene.dna_event_features,
        gene.dna_event_relation,
        gene.dna_event_choice_index,
        gene.dna_gate[:4],
    )
    event = scorer(batch, frozen)
    torch.testing.assert_close(
        event.centered_sensitivity.sum(dim=1),
        torch.zeros(event.centered_sensitivity.shape[0]),
        atol=1e-6,
        rtol=1e-6,
    )
    zero = torch.zeros_like(event.correction)
    paths = AugmentedPathReadout(0.3)(
        PathReadoutInput(
            output.edge_energy,
            zero,
            event.correction,
            zero,
            gene.path_edge_incidence,
            gene.path_choice_incidence,
        )
    )
    edge_manual = output.edge_energy @ gene.path_edge_incidence.to_dense().T
    dna_manual = event.correction @ gene.path_choice_incidence.to_dense().T
    length_manual = -0.3 * gene.path_edge_incidence.to_dense().sum(1)
    torch.testing.assert_close(paths.edge_logits, edge_manual.expand(4, -1))
    torch.testing.assert_close(paths.dna_logits, dna_manual)
    torch.testing.assert_close(paths.length_logits, length_manual.expand(4, -1))
    torch.testing.assert_close(
        paths.total_logits,
        paths.edge_logits
        + paths.state_logits
        + paths.dna_logits
        + paths.rna_logits
        + paths.length_logits,
    )
