from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from fabric.likelihood import compatible_path_nll
from fabric.model import (
    PRIMARY_ABLATIONS,
    PathContextReadout,
    RoutedEventAggregator,
    RoutedModalityInput,
)
from fabric.train import build_paired_models, load_config, make_toy_genes


def _model_and_gene(seed: int = 19):
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    model = build_paired_models(
        gene, config["model"], seed=seed, device="cpu"
    )["full"]
    return model, gene


def test_three_block_forward_shapes_fixed_pre_norm_and_all_full_gradients():
    model, gene = _model_and_gene()
    output = model(gene.model_input, condition="full")
    cells = len(gene.cell_ids)
    edges = gene.model_input.cis_features.shape[0]
    assert output.path_logits.shape == (cells, len(gene.path_ids))
    assert output.dna_aggregate.shape == (cells, edges, model.dynamic_dim)
    assert output.rna_aggregate.shape == (cells, edges, model.dynamic_dim)
    assert output.joint_input.shape[-1] == model.cis_dim + 2 * model.dynamic_dim
    assert output.joint_projected.shape == output.normalized_tokens.shape
    assert output.edge_states.shape == output.normalized_tokens.shape
    bound = math.sqrt(model.hidden_dim)
    assert float(torch.linalg.vector_norm(output.normalized_tokens, dim=-1).max()) <= bound + 1e-5

    loss = compatible_path_nll(
        output.path_logits,
        gene.compatible_path_indices,
        gene.compatible_path_mask,
        gene.molecule_count,
        row_cell_index=gene.row_cell_index,
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_graphgps_global_attention_is_strictly_cell_isolated():
    model, gene = _model_and_gene()
    model.eval()
    baseline = model(gene.model_input, condition="full").path_logits
    changed_dna_gate = gene.model_input.dna.gate.clone()
    changed_rna_gate = gene.model_input.rna.gate.clone()
    changed_dna_gate[1] += 100.0
    changed_rna_gate[1] -= 100.0
    changed = replace(
        gene.model_input,
        dna=replace(gene.model_input.dna, gate=changed_dna_gate),
        rna=replace(gene.model_input.rna, gate=changed_rna_gate),
    )
    observed = model(changed, condition="full").path_logits
    torch.testing.assert_close(observed[0], baseline[0], atol=0, rtol=0)
    assert not torch.allclose(observed[1], baseline[1])
    torch.testing.assert_close(observed[2:], baseline[2:], atol=0, rtol=0)


def test_sparse_route_features_match_dense_projection_and_gradients():
    generator = torch.Generator().manual_seed(41)
    base = torch.randn(7, 5, generator=generator)
    interaction = torch.randn(7, 4, generator=generator)
    base[base.abs() < 0.7] = 0
    interaction[interaction.abs() < 0.7] = 0
    routed = RoutedModalityInput(
        route_event_index=torch.tensor([0, 0, 1, 1, 2, 2, 2]),
        route_edge_index=torch.tensor([0, 1, 0, 2, 0, 1, 2]),
        route_weight=torch.tensor([0.5, 0.5, 0.4, 0.6, 0.2, 0.3, 0.5]),
        route_base_features=base,
        route_interaction_features=interaction,
        interaction_active_mask=torch.tensor([True, False, True, True]),
        event_gate_key_index=torch.tensor([0, 1, 0]),
        gate=torch.tensor([[0.2, 0.8], [0.7, 0.3]]),
    )
    sparse_routed = replace(
        routed,
        route_base_features=base.to_sparse_coo().coalesce(),
        route_interaction_features=interaction.to_sparse_coo().coalesce(),
    )
    dense_aggregator = RoutedEventAggregator(5, 4, 3)
    sparse_aggregator = RoutedEventAggregator(5, 4, 3)
    sparse_aggregator.load_state_dict(dense_aggregator.state_dict())
    dense_output = dense_aggregator(routed, edge_count=3)
    sparse_output = sparse_aggregator(sparse_routed, edge_count=3)
    torch.testing.assert_close(sparse_output, dense_output, atol=1e-6, rtol=1e-6)
    dense_output.square().sum().backward()
    sparse_output.square().sum().backward()
    for dense_parameter, sparse_parameter in zip(
        dense_aggregator.parameters(), sparse_aggregator.parameters(), strict=True
    ):
        torch.testing.assert_close(
            sparse_parameter.grad, dense_parameter.grad, atol=1e-6, rtol=1e-6
        )


def test_single_event_and_shared_gate_injection_are_affine_before_pre_layernorm():
    model, gene = _model_and_gene()
    routed = gene.model_input.dna
    full = model.dna_aggregator(routed, gene.model_input.cis_features.shape[0])
    keep = torch.tensor([False, True])
    neutralized_routed = routed.with_event_keep_mask(keep)
    neutralized = model.dna_aggregator(
        neutralized_routed, gene.model_input.cis_features.shape[0]
    )
    route_projection = model.dna_aggregator.route_projection(routed)
    expected_direction = torch.zeros_like(full[0])
    event_routes = torch.nonzero(routed.route_event_index == 0).reshape(-1)
    expected_direction.index_add_(
        0,
        routed.route_edge_index[event_routes],
        routed.route_weight[event_routes, None] * route_projection[event_routes],
    )
    expected = routed.gate[:, 0, None, None] * expected_direction[None, :, :]
    torch.testing.assert_close(full - neutralized, expected, atol=1e-6, rtol=1e-6)

    # Both events now share one key.  A key perturbation must change both event
    # route unions together, not pretend one event retains a fixed dose.
    shared = replace(
        routed,
        event_gate_key_index=torch.tensor([0, 0]),
        gate=routed.gate[:, :1].clone(),
    )
    shifted = replace(shared, gate=shared.gate + 0.75)
    a1 = model.dna_aggregator(shared, gene.model_input.cis_features.shape[0])
    a2 = model.dna_aggregator(shifted, gene.model_input.cis_features.shape[0])
    all_direction = torch.zeros_like(a1[0])
    all_direction.index_add_(
        0,
        shared.route_edge_index,
        shared.route_weight[:, None] * model.dna_aggregator.route_projection(shared),
    )
    torch.testing.assert_close(
        a2 - a1,
        0.75 * all_direction[None, :, :].expand_as(a1),
        atol=1e-6,
        rtol=1e-6,
    )

    # W_X is the exact linear image of the aggregate change; pre-LN/output are
    # intentionally not constrained to be affine.
    first = replace(gene.model_input, dna=shared)
    second = replace(gene.model_input, dna=shifted)
    out1 = model(first, condition="full")
    out2 = model(second, condition="full")
    delta_x = out2.joint_input - out1.joint_input
    expected_y = torch.matmul(delta_x, model.joint_projection.weight.T)
    torch.testing.assert_close(
        out2.joint_projected - out1.joint_projected,
        expected_y,
        atol=1e-6,
        rtol=1e-6,
    )


def test_route_level_anchor_selector_deletes_only_selected_terms_without_renormalizing():
    aggregator = RoutedEventAggregator(base_dim=1, interaction_dim=0, output_dim=1)
    with torch.no_grad():
        aggregator.base_projection.weight.fill_(1.0)
    routed = RoutedModalityInput(
        route_event_index=torch.tensor([0, 0]),
        route_edge_index=torch.tensor([0, 1]),
        route_weight=torch.tensor([0.5, 0.5]),
        route_base_features=torch.ones(2, 1),
        route_interaction_features=torch.empty(2, 0),
        interaction_active_mask=torch.empty(0, dtype=torch.bool),
        event_gate_key_index=torch.tensor([0]),
        gate=torch.tensor([[2.0]]),
    )
    full = aggregator(routed, 2)
    partial = aggregator(
        routed.with_route_keep_mask(torch.tensor([True, False])), 2
    )
    torch.testing.assert_close(full[0, :, 0], torch.tensor([1.0, 1.0]))
    # The retained route stays at its frozen 0.5 weight rather than becoming 1.
    torch.testing.assert_close(partial[0, :, 0], torch.tensor([1.0, 0.0]))


def test_path_centered_residual_sum_has_constitutive_padding_and_jacobian_invariance():
    torch.manual_seed(3)
    readout = PathContextReadout(hidden_dim=3, path_hidden_dim=5)
    states = torch.randn(1, 4, 3, requires_grad=True)
    incidence = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]]
    ).to_sparse_coo()
    first = torch.tensor([0, 0])
    last = torch.tensor([3, 3])
    length = torch.log1p(torch.tensor([3.0, 3.0]))
    output = readout(states, incidence, first, last, length)
    difference = output.path_residual[:, 0] - output.path_residual[:, 1]
    gradient = torch.autograd.grad(difference.sum(), states, retain_graph=True)[0]

    padded_states = torch.cat(
        (states.detach(), torch.randn(1, 1, 3)), dim=1
    ).requires_grad_()
    padded_incidence = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0]]
    ).to_sparse_coo()
    padded = readout(
        padded_states,
        padded_incidence,
        first,
        last,
        torch.log1p(torch.tensor([4.0, 4.0])),
    )
    padded_difference = padded.path_residual[:, 0] - padded.path_residual[:, 1]
    padded_gradient = torch.autograd.grad(padded_difference.sum(), padded_states)[0]
    torch.testing.assert_close(padded_difference, difference.detach(), atol=0, rtol=0)
    torch.testing.assert_close(padded_gradient[:, :4], gradient, atol=0, rtol=0)
    torch.testing.assert_close(padded_gradient[:, 4], torch.zeros_like(padded_gradient[:, 4]))

    mean_difference = states[:, [0, 1, 3]].mean(1) - states[:, [0, 2, 3]].mean(1)
    padded_mean_difference = (
        padded_states[:, [0, 1, 3, 4]].mean(1)
        - padded_states[:, [0, 2, 3, 4]].mean(1)
    )
    torch.testing.assert_close(
        padded_mean_difference,
        0.75 * mean_difference.detach(),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    ("incidence", "first", "last", "log_count", "message"),
    [
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 0, 1], [0, 1, 1]]),
                torch.tensor([1.0, 0.5, 1.0]),
                (2, 2),
            ),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
            torch.log1p(torch.tensor([2.0, 1.0])),
            "binary unit",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
                torch.ones(4),
                (2, 2),
            ),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
            torch.log1p(torch.tensor([2.0, 1.0])),
            "duplicate",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]), torch.ones(1), (2, 2)
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.log1p(torch.tensor([1.0, 1.0])),
            "at least one edge",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 3)
            ),
            torch.tensor([2, 1]),
            torch.tensor([0, 1]),
            torch.log1p(torch.tensor([1.0, 1.0])),
            "first-edge index is not contained",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 3)
            ),
            torch.tensor([0, 1]),
            torch.tensor([2, 1]),
            torch.log1p(torch.tensor([1.0, 1.0])),
            "last-edge index is not contained",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 2)
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.log1p(torch.tensor([2.0, 1.0])),
            "differs from binary path incidence",
        ),
    ],
)
def test_path_structural_identity_fails_closed(
    incidence, first, last, log_count, message
):
    readout = PathContextReadout(hidden_dim=3, path_hidden_dim=5)
    states = torch.randn(1, incidence.shape[1], 3)
    with pytest.raises(ValueError, match=message):
        readout(states, incidence, first, last, log_count)


def test_one_event_has_path_context_dependent_probability_effect():
    model, gene = _model_and_gene(seed=31)
    model.eval()
    full = model(gene.model_input, condition="full").path_logits
    keep = torch.tensor([False, True])
    without_input = replace(
        gene.model_input,
        dna=gene.model_input.dna.with_event_keep_mask(keep),
    )
    without = model(without_input, condition="full").path_logits
    probability_delta = full.softmax(1) - without.softmax(1)
    assert torch.isfinite(probability_delta).all()
    # A single TSS/region event route union changes the two downstream paths in
    # opposite directions after the gene softmax, not as one uniform shift.
    assert bool((probability_delta[:, 0] * probability_delta[:, 1] <= 0).all())
    assert float(probability_delta.abs().max()) > 1e-7


def test_same_seed_reproduces_logits_and_primary_parameterization_is_identical():
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    left = build_paired_models(gene, config["model"], seed=101, device="cpu")
    right = build_paired_models(gene, config["model"], seed=101, device="cpu")
    counts = {
        name: sum(parameter.numel() for parameter in left[name].parameters())
        for name in PRIMARY_ABLATIONS
    }
    assert len(set(counts.values())) == 1
    reference = left["cis"].state_dict()
    for name in PRIMARY_ABLATIONS:
        observed = left[name].state_dict()
        assert observed.keys() == reference.keys()
        for key in reference:
            torch.testing.assert_close(observed[key], reference[key], atol=0, rtol=0)
        logits_left = left[name](gene.model_input, condition=name).path_logits
        logits_right = right[name](gene.model_input, condition=name).path_logits
        torch.testing.assert_close(logits_left, logits_right, atol=0, rtol=0)
    comparator = left["full_additive_edge"].state_dict()
    for key, value in reference.items():
        if not key.startswith("readout."):
            torch.testing.assert_close(comparator[key], value, atol=0, rtol=0)


def test_degree_partition_identity_and_pre_layernorm_scale_control():
    aggregator = RoutedEventAggregator(base_dim=1, interaction_dim=0, output_dim=1)
    with torch.no_grad():
        aggregator.base_projection.weight.fill_(1.0)
    for degree in (2, 4, 8):
        routed = RoutedModalityInput(
            route_event_index=torch.zeros(degree, dtype=torch.long),
            route_edge_index=torch.arange(degree),
            route_weight=torch.full((degree,), 1.0 / degree),
            route_base_features=torch.ones(degree, 1),
            route_interaction_features=torch.empty(degree, 0),
            interaction_active_mask=torch.empty(0, dtype=torch.bool),
            event_gate_key_index=torch.tensor([0]),
            gate=torch.ones(1, 1),
        )
        aggregate = aggregator(routed, degree)[0, :, 0]
        torch.testing.assert_close(aggregate, torch.full((degree,), 1.0 / degree))
        torch.testing.assert_close(
            torch.linalg.vector_norm(aggregate),
            torch.tensor(1.0 / math.sqrt(degree)),
        )

    # Same-token burden changes the pre-normalization vector, while fixed
    # no-affine LayerNorm bounds the token passed into attention.
    model, gene = _model_and_gene(seed=5)
    norms = []
    normalized_norms = []
    for event_count in (1, 4, 16):
        dna = RoutedModalityInput(
            route_event_index=torch.arange(event_count),
            route_edge_index=torch.zeros(event_count, dtype=torch.long),
            route_weight=torch.ones(event_count),
            route_base_features=torch.ones(
                event_count, gene.model_input.dna.route_base_features.shape[1]
            ),
            route_interaction_features=torch.zeros(
                event_count,
                gene.model_input.dna.route_interaction_features.shape[1],
            ),
            interaction_active_mask=gene.model_input.dna.interaction_active_mask,
            event_gate_key_index=torch.zeros(event_count, dtype=torch.long),
            gate=torch.ones(len(gene.cell_ids), 1),
        )
        output = model(replace(gene.model_input, dna=dna), condition="full")
        norms.append(float(torch.linalg.vector_norm(output.joint_projected[:, 0], dim=-1).mean()))
        normalized_norms.append(
            float(torch.linalg.vector_norm(output.normalized_tokens[:, 0], dim=-1).max())
        )
        assert torch.isfinite(output.edge_states).all()
        assert torch.isfinite(output.path_logits).all()
    assert norms[-1] > norms[0]
    assert max(normalized_norms) <= math.sqrt(model.hidden_dim) + 1e-5
