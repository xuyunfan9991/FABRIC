from __future__ import annotations

from dataclasses import replace
import math

import torch

from fabric.likelihood import compatible_path_nll
from fabric.model import FABRICV2Model, RoutedModalityInput
from fabric.train import build_paired_models, load_config, make_toy_genes


def _same_token_burden_input(event_count: int):
    gene = make_toy_genes()[0]
    source = gene.model_input
    dna = RoutedModalityInput(
        route_event_index=torch.arange(event_count),
        route_edge_index=torch.zeros(event_count, dtype=torch.long),
        route_weight=torch.ones(event_count),
        route_base_features=torch.ones(
            event_count, source.dna.route_base_features.shape[1]
        ),
        route_interaction_features=torch.zeros(
            event_count, source.dna.route_interaction_features.shape[1]
        ),
        interaction_active_mask=source.dna.interaction_active_mask,
        event_gate_key_index=torch.zeros(event_count, dtype=torch.long),
        gate=torch.ones(len(gene.cell_ids), 1),
    )
    return gene, replace(source, dna=dna)


def _attention_score_reference(
    model: FABRICV2Model, tokens: torch.Tensor
) -> torch.Tensor:
    """Test-local MHA score calculation; this is not a runtime branch."""

    attention = model.graphgps.global_attention
    hidden = model.hidden_dim
    head_count = attention.num_heads
    head_width = hidden // head_count
    weight = attention.in_proj_weight
    bias = attention.in_proj_bias
    query = tokens @ weight[:hidden].T + bias[:hidden]
    key = tokens @ weight[hidden : 2 * hidden].T + bias[hidden : 2 * hidden]
    query = query.view(*tokens.shape[:2], head_count, head_width).transpose(1, 2)
    key = key.view(*tokens.shape[:2], head_count, head_width).transpose(1, 2)
    return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_width)


def test_one_four_sixteen_event_burden_is_bounded_and_has_finite_gradients():
    config = load_config("configs/fabric_v2_toy.yaml")
    raw_norms: list[float] = []
    raw_attention_rms: list[float] = []
    normalized_attention_rms: list[float] = []
    for event_count in (1, 4, 16):
        gene, model_input = _same_token_burden_input(event_count)
        model = build_paired_models(gene, config["model"], seed=5, device="cpu")["full"]
        output = model(model_input, condition="full")
        raw_scores = _attention_score_reference(model, output.joint_projected)
        normalized_scores = _attention_score_reference(model, output.normalized_tokens)
        raw_norms.append(
            float(torch.linalg.vector_norm(output.joint_projected[:, 0], dim=-1).mean())
        )
        raw_attention_rms.append(float(raw_scores.square().mean().sqrt()))
        normalized_attention_rms.append(float(normalized_scores.square().mean().sqrt()))
        assert (
            float(torch.linalg.vector_norm(output.normalized_tokens, dim=-1).max())
            <= math.sqrt(model.hidden_dim) + 1.0e-5
        )
        loss = compatible_path_nll(
            output.path_logits,
            gene.compatible_path_indices,
            gene.compatible_path_mask,
            gene.molecule_count,
            row_cell_index=gene.row_cell_index,
        )
        loss.backward()
        assert torch.isfinite(output.edge_states).all()
        assert torch.isfinite(output.path_logits).all()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    # Removing the fixed pre-normalization in this calculation exposes the
    # intended burden amplification.  Production always uses the normalized
    # tokens and offers no switch for this test-local reference.
    assert raw_norms[-1] > 8.0 * raw_norms[0]
    assert raw_attention_rms[-1] > 4.0 * raw_attention_rms[0]
    assert normalized_attention_rms[-1] < 2.0 * normalized_attention_rms[0]


def test_common_upstream_event_recovers_internal_exon_pas_context_dependence():
    """A common event learns opposite effects on two downstream full paths."""

    source = make_toy_genes()[0].model_input
    gate = torch.linspace(-1.0, 1.0, 25)[:, None]
    dna = RoutedModalityInput(
        route_event_index=torch.tensor([0]),
        route_edge_index=torch.tensor([0]),
        route_weight=torch.tensor([1.0]),
        route_base_features=torch.ones(1, 2),
        route_interaction_features=torch.empty(1, 0),
        interaction_active_mask=torch.empty(0, dtype=torch.bool),
        event_gate_key_index=torch.tensor([0]),
        gate=gate,
    )
    rna = RoutedModalityInput(
        route_event_index=torch.empty(0, dtype=torch.long),
        route_edge_index=torch.empty(0, dtype=torch.long),
        route_weight=torch.empty(0),
        route_base_features=torch.empty(0, 1),
        route_interaction_features=torch.empty(0, 0),
        interaction_active_mask=torch.empty(0, dtype=torch.bool),
        event_gate_key_index=torch.empty(0, dtype=torch.long),
        gate=torch.empty(len(gate), 0),
    )
    model_input = replace(source, dna=dna, rna=rna)
    torch.manual_seed(1)
    model = FABRICV2Model(
        cis_dim=source.cis_features.shape[1],
        dna_base_dim=2,
        dna_interaction_dim=0,
        rna_base_dim=1,
        rna_interaction_dim=0,
        dynamic_dim=4,
        hidden_dim=8,
        attention_heads=2,
        path_hidden_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    planted_difference = 1.5 * gate[:, 0]
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits = model(model_input, condition="full").path_logits
        loss = torch.mean(((logits[:, 0] - logits[:, 1]) - planted_difference) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(model_input, condition="full").path_logits
        recovered = logits[:, 0] - logits[:, 1]
    torch.testing.assert_close(recovered, planted_difference, atol=0.02, rtol=0.02)
    assert recovered[0] < -1.0
    assert recovered[-1] > 1.0
