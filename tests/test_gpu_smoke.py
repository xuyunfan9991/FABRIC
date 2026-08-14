from __future__ import annotations

import os

import pytest
import torch

from fabric.likelihood import compatible_path_nll
from fabric.train import build_paired_models, load_config, make_toy_genes


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("FABRIC_GPU_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="set FABRIC_GPU_SMOKE=1 with a visible CUDA device",
)
def test_cuda_v2_three_block_path_context_likelihood_backward_profile():
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    model = build_paired_models(
        gene, config["model"], seed=101, device=device
    )["full"]
    from fabric.train import _routed_to_device
    from fabric.model import GeneCellModelInput

    source = gene.model_input
    cells = torch.arange(len(gene.cell_ids))
    inputs = GeneCellModelInput(
        cis_features=source.cis_features.to(device),
        local_edge_index=source.local_edge_index.to(device),
        dna=_routed_to_device(source.dna, device, cells),
        rna=_routed_to_device(source.rna, device, cells),
        path_edge_incidence=source.path_edge_incidence.to(device),
        path_first_edge_index=source.path_first_edge_index.to(device),
        path_last_edge_index=source.path_last_edge_index.to(device),
        log_edge_count=source.log_edge_count.to(device),
    )
    output = model(inputs, condition="full")
    loss = compatible_path_nll(
        output.path_logits,
        gene.compatible_path_indices.to(device),
        gene.compatible_path_mask.to(device),
        gene.molecule_count.to(device),
        row_cell_index=gene.row_cell_index.to(device),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    peak_bytes = torch.cuda.max_memory_allocated(device)
    print(f"FABRIC V2 toy CUDA peak allocated bytes: {peak_bytes}")
    assert peak_bytes < 1_000_000_000
