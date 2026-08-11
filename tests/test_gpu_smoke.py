from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fabric.dataset import build_exact_stage_neighbors
from fabric.likelihood import compatible_path_nll
from fabric.model import (
    AugmentedPathReadout,
    EdgeGraphGPS,
    PathReadoutInput,
    StateBatch,
    StateScorer,
    build_frozen_alternative_state,
)
from fabric.train import make_toy_genes


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("FABRIC_GPU_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="set FABRIC_GPU_SMOKE=1 with a visible CUDA device",
)
def test_cuda_graph_likelihood_backward_and_exact_stage_knn():
    device = torch.device("cuda:0")
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    gene = make_toy_genes()[0]
    graph = type(gene.graph)(
        gene.graph.edge_features.to(device),
        gene.graph.local_edge_index.to(device),
        gene.graph.edge_gene_index.to(device),
    )
    alternatives = type(gene.alternatives)(
        gene.alternatives.edge_index.to(device),
        gene.alternatives.edge_mask.to(device),
        gene.alternatives.choice_index.to(device),
        gene.alternatives.scope_index.to(device),
    )
    cis = EdgeGraphGPS(graph.edge_features.shape[1], 16, 2).to(device)
    output = cis(graph)
    frozen = build_frozen_alternative_state(output.edge_states, alternatives)
    state = StateScorer(4, frozen.h_base.shape[1], 4).to(device)
    torch.nn.init.constant_(state.V.weight, 0.1)
    correction = state(
        StateBatch(gene.state_features[:8].to(device)), frozen
    ).correction
    zeros = torch.zeros_like(correction)
    logits = (
        AugmentedPathReadout(0.1)
        .to(device)(
            PathReadoutInput(
                output.edge_energy,
                correction,
                zeros,
                zeros,
                gene.path_edge_incidence.to(device),
                gene.path_choice_incidence.to(device),
            )
        )
        .total_logits
    )
    loss = compatible_path_nll(
        logits,
        gene.compatible_path_indices[:8].to(device),
        gene.compatible_path_mask[:8].to(device),
        gene.molecule_count[:8].to(device),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in (*cis.parameters(), *state.parameters())
    )

    rng = np.random.default_rng(7)
    rna = rng.normal(size=(64, 50)).astype(np.float32)
    atac = rng.normal(size=(128, 50)).astype(np.float32)
    neighbors, status = build_exact_stage_neighbors(
        rna_cell_ids=[f"r{index}" for index in range(64)],
        rna_embedding=rna,
        rna_stage=["CS12"] * 64,
        atac_cell_ids=[f"a{index}" for index in range(128)],
        atac_embedding=atac,
        atac_stage=["CS12"] * 128,
        atac_donor_ids=[f"donor{index // 16}" for index in range(128)],
        atac_donor_eligible=[index % 3 != 0 for index in range(128)],
        k=8,
        temperature=1.0,
        device="cuda:0",
        query_chunk_size=17,
    )
    assert len(neighbors) == 64 * 8
    assert status["observed_atac"].all()
    np.testing.assert_allclose(
        neighbors.groupby("cell_id")["neighbor_weight"].sum(), 1.0, atol=1e-6
    )
    assert torch.cuda.max_memory_allocated(device) < 1_000_000_000
