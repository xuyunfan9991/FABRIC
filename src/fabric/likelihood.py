"""Grouped legal-path probabilities and compatible-path likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CompatiblePathNLL:
    loss: torch.Tensor
    weighted_sum: torch.Tensor
    molecule_mass: torch.Tensor
    per_row_nll: torch.Tensor
    posterior_mass: torch.Tensor
    informative_row_mask: torch.Tensor


def grouped_log_softmax(
    path_logits: torch.Tensor,
    path_gene_index: torch.Tensor | None = None,
    *,
    prevalidated: bool = False,
) -> torch.Tensor:
    """Normalize only among legal paths of the same gene."""

    if path_logits.ndim == 1:
        path_logits = path_logits.unsqueeze(0)
    if path_logits.ndim != 2:
        raise ValueError("path_logits must have shape [cells, paths]")
    if not torch.is_floating_point(path_logits):
        raise TypeError("path_logits must use a floating dtype")
    if not prevalidated and not bool(torch.isfinite(path_logits).all().item()):
        raise ValueError("path_logits must be finite")
    if path_gene_index is None:
        return F.log_softmax(path_logits, dim=1)
    path_gene_index = path_gene_index.to(path_logits.device, dtype=torch.long)
    if path_gene_index.ndim != 1 or path_gene_index.numel() != path_logits.shape[1]:
        raise ValueError("path_gene_index must have one entry per path")
    if not prevalidated and bool((path_gene_index < 0).any().item()):
        raise ValueError("path_gene_index must be non-negative")
    result = torch.empty_like(path_logits)
    for gene_index in torch.unique(path_gene_index, sorted=True):
        mask = path_gene_index == gene_index
        result[:, mask] = F.log_softmax(path_logits[:, mask], dim=1)
    return result


def compatible_path_nll(
    path_logits: torch.Tensor,
    compatible_path_indices: torch.Tensor,
    compatible_path_mask: torch.Tensor,
    molecule_count: torch.Tensor,
    *,
    row_cell_index: torch.Tensor | None = None,
    path_gene_index: torch.Tensor | None = None,
    reduction: str = "mean",
    return_details: bool = False,
    prevalidated: bool = False,
) -> torch.Tensor | CompatiblePathNLL:
    """Exact molecule-weighted NLL for compatible legal-path sets.

    ``path_logits`` contains one legal-path vector per unique cell.  EC rows
    point to those cells through ``row_cell_index`` and may contain different
    compatible subsets.  Padding is represented only by the explicit mask.

    ``prevalidated`` asserts only that the EC row block already satisfies the
    frozen shard contract established at ``_validate_genes`` load time (index
    ranges, left-aligned mask, strictly increasing rows, finite positive
    molecule counts).  It never disables a guard over values computed here
    from live model weights: a non-finite row posterior and an empty
    informative mass still raise on every batch.
    """

    if path_logits.ndim == 1:
        path_logits = path_logits.unsqueeze(0)
    if compatible_path_indices.ndim != 2:
        raise ValueError("compatible_path_indices must have shape [rows, width]")
    if compatible_path_mask.shape != compatible_path_indices.shape:
        raise ValueError("compatible_path_mask must match compatible_path_indices")
    if compatible_path_mask.dtype != torch.bool:
        raise TypeError("compatible_path_mask must use bool dtype")
    compatible_path_mask = compatible_path_mask.to(path_logits.device)
    row_count = compatible_path_indices.shape[0]
    molecule_count = molecule_count.to(path_logits.device, dtype=path_logits.dtype)
    if molecule_count.ndim != 1 or molecule_count.numel() != row_count:
        raise ValueError("molecule_count must have one value per EC row")
    if not prevalidated:
        if not bool(torch.isfinite(molecule_count).all().item()):
            raise ValueError("molecule_count must be finite")
        if bool((molecule_count < 0).any().item()):
            raise ValueError("molecule_count must be non-negative")
    if row_cell_index is None:
        if path_logits.shape[0] != row_count:
            raise ValueError("row_cell_index is required when cells and EC rows differ")
        row_cell_index = torch.arange(row_count, device=path_logits.device)
    else:
        row_cell_index = row_cell_index.to(path_logits.device, dtype=torch.long)
    if row_cell_index.ndim != 1 or row_cell_index.numel() != row_count:
        raise ValueError("row_cell_index must have one value per EC row")
    if not prevalidated and bool(
        ((row_cell_index < 0) | (row_cell_index >= path_logits.shape[0])).any().item()
    ):
        raise IndexError("row_cell_index is out of range")
    compatible_path_indices = compatible_path_indices.to(
        path_logits.device, dtype=torch.long
    )
    if not prevalidated:
        for row in range(row_count):
            row_indices = compatible_path_indices[row, compatible_path_mask[row]]
            if row_indices.numel() == 0:
                raise ValueError("compatible-path row cannot be empty")
            if torch.unique(row_indices).numel() != row_indices.numel():
                raise ValueError("compatible-path row contains duplicate path indices")
        valid_indices = compatible_path_indices[compatible_path_mask]
        if valid_indices.numel() == 0:
            raise ValueError("compatible-path batch contains no non-empty row")
        if bool(
            ((valid_indices < 0) | (valid_indices >= path_logits.shape[1])).any().item()
        ):
            raise IndexError("compatible path index is out of range")

    log_probs = grouped_log_softmax(
        path_logits, path_gene_index, prevalidated=prevalidated
    )
    padded = torch.where(
        compatible_path_mask,
        compatible_path_indices,
        torch.zeros_like(compatible_path_indices),
    )
    selected = log_probs[row_cell_index[:, None], padded]
    selected = selected.masked_fill(
        ~compatible_path_mask, -torch.inf
    )
    log_mass = torch.logsumexp(selected, dim=1)
    if bool((~torch.isfinite(log_mass)).any().item()):
        raise ValueError("compatible-path row has zero posterior mass")
    per_row = -log_mass
    positive = molecule_count > 0
    if path_gene_index is None:
        total_paths = torch.full(
            (row_count,),
            path_logits.shape[1],
            device=path_logits.device,
            dtype=torch.long,
        )
    else:
        path_gene_index = path_gene_index.to(path_logits.device, dtype=torch.long)
        if not prevalidated and bool((path_gene_index < 0).any().item()):
            raise ValueError("path_gene_index must be non-negative")
        gene_sizes = torch.bincount(path_gene_index)
        first_valid_column = compatible_path_mask.long().argmax(dim=1)
        first_path = compatible_path_indices.gather(
            1, first_valid_column[:, None]
        ).squeeze(1)
        row_gene = path_gene_index[first_path]
        selected_genes = path_gene_index[padded]
        selected_genes = torch.where(
            compatible_path_mask, selected_genes, row_gene[:, None]
        )
        if not prevalidated and bool(
            (selected_genes != row_gene[:, None]).any().item()
        ):
            raise ValueError("one compatible row crosses gene path axes")
        total_paths = gene_sizes[row_gene]
    compatible_sizes = compatible_path_mask.sum(dim=1)
    informative = positive & (compatible_sizes > 0) & (compatible_sizes < total_paths)
    molecule_mass = molecule_count[informative].sum()
    if not bool((molecule_mass > 0).item()):
        raise ValueError("compatible-path batch has zero informative molecule mass")
    weighted_sum = (per_row[informative] * molecule_count[informative]).sum()
    if reduction == "mean":
        loss = weighted_sum / molecule_mass
    elif reduction == "sum":
        loss = weighted_sum
    elif reduction == "none":
        loss = per_row
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    details = CompatiblePathNLL(
        loss=loss,
        weighted_sum=weighted_sum,
        molecule_mass=molecule_mass,
        per_row_nll=per_row,
        posterior_mass=log_mass.exp(),
        informative_row_mask=informative,
    )
    return details if return_details else loss


def brute_force_compatible_path_nll(
    path_logits: torch.Tensor,
    compatible_paths: list[list[int]],
    molecule_count: torch.Tensor,
    *,
    row_cell_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent small-fixture reference implementation."""

    if path_logits.ndim == 1:
        path_logits = path_logits.unsqueeze(0)
    if row_cell_index is None:
        row_cell_index = torch.arange(len(compatible_paths), device=path_logits.device)
    probabilities = torch.softmax(path_logits, dim=1)
    terms = []
    informative_weight = molecule_count.new_zeros(())
    for row, compatible in enumerate(compatible_paths):
        if not compatible:
            raise ValueError("compatible set must be non-empty")
        if len(set(compatible)) != len(compatible):
            raise ValueError("compatible set must contain unique path indices")
        if len(compatible) == path_logits.shape[1] or molecule_count[row] <= 0:
            continue
        mass = probabilities[int(row_cell_index[row]), compatible].sum()
        terms.append(-torch.log(mass) * molecule_count[row])
        informative_weight = informative_weight + molecule_count[row]
    if not terms or not bool((informative_weight > 0).item()):
        raise ValueError("compatible-path reference has zero informative mass")
    return torch.stack(terms).sum() / informative_weight
