from __future__ import annotations

import torch

from fabric.likelihood import brute_force_compatible_path_nll, compatible_path_nll


def test_compatible_path_nll_matches_brute_force_and_backpropagates():
    logits = torch.tensor([[1.2, -0.3, 0.7], [-0.4, 0.8, 0.1]], requires_grad=True)
    compatible = torch.tensor([[0, -1], [1, 2], [0, 2]])
    mask = compatible >= 0
    cells = torch.tensor([0, 0, 1])
    molecule_count = torch.tensor([2.0, 3.0, 5.0])
    observed = compatible_path_nll(
        logits,
        compatible,
        mask,
        molecule_count,
        row_cell_index=cells,
    )
    expected = brute_force_compatible_path_nll(
        logits,
        [[0], [1, 2], [0, 2]],
        molecule_count,
        row_cell_index=cells,
    )
    torch.testing.assert_close(observed, expected, atol=1e-7, rtol=1e-7)
    observed.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_grouped_paths_do_not_compete_across_genes():
    logits = torch.tensor([[2.0, 0.0, 100.0, 99.0]])
    compatible = torch.tensor([[0], [2]])
    details = compatible_path_nll(
        logits,
        compatible,
        torch.ones_like(compatible, dtype=torch.bool),
        torch.ones(2),
        row_cell_index=torch.zeros(2, dtype=torch.long),
        path_gene_index=torch.tensor([0, 0, 1, 1]),
        return_details=True,
    )
    expected = torch.stack(
        [
            -torch.log_softmax(logits[:, :2], 1)[0, 0],
            -torch.log_softmax(logits[:, 2:], 1)[0, 0],
        ]
    )
    torch.testing.assert_close(details.per_row_nll, expected)
