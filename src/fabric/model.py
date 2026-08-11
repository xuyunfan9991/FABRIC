"""The frozen FABRIC V1 model blocks.

The module keeps the scientific objects visible: processing edges are the
GraphGPS states, alternatives are fixed ordered edge subsequences, event
sensitivity is centered within its own choice, and path logits are sparse
incidence sums.  No branch owns a fusion layer or an additional output head.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

from .choices import ChoiceCatalog


CHOICE_SCOPES = ("internal", "tss", "pas", "full_length")
_SCOPE_INDEX = {name: index for index, name in enumerate(CHOICE_SCOPES)}


@dataclass(frozen=True)
class GraphGPSBatch:
    """One or more gene graphs whose processing edges are graph tokens.

    ``local_edge_index`` connects adjacent processing edges.  ``edge_gene_index``
    must group tokens by gene so dense attention can never cross a gene boundary.
    """

    edge_features: torch.Tensor  # [E, F]
    local_edge_index: torch.Tensor  # [2, L]
    edge_gene_index: torch.Tensor  # [E]


@dataclass(frozen=True)
class CISOutput:
    edge_states: torch.Tensor  # q_e, [E, H]
    edge_energy: torch.Tensor  # psi_cis, [E]


@dataclass(frozen=True)
class AlternativeBatch:
    """Ordered edge membership and fixed choice metadata for alternatives."""

    edge_index: torch.Tensor  # [A, W], padded with -1
    edge_mask: torch.Tensor  # bool [A, W]
    choice_index: torch.Tensor  # [A]
    scope_index: torch.Tensor  # [A], indexes CHOICE_SCOPES


@dataclass(frozen=True)
class FrozenAlternativeState:
    """CIS-derived quantities shared exactly by every dynamic child."""

    edge_states: torch.Tensor  # detached q_e, [E, H]
    h_base: torch.Tensor  # [A, 3H + 1 + len(CHOICE_SCOPES)]
    choice_index: torch.Tensor  # [A]
    scope_index: torch.Tensor  # [A]


@dataclass(frozen=True)
class StateBatch:
    features: torch.Tensor  # centered RNA-only state, [B, Z]


@dataclass(frozen=True)
class StateOutput:
    raw_potential: torch.Tensor  # [B, A]
    correction: torch.Tensor  # choice-centered [B, A]


@dataclass(frozen=True)
class EventBatch:
    """Fixed event evidence and cell-specific centered gates for one modality.

    ``relation[j, a]`` is R for event ``j`` and alternative ``a``.  Values
    outside the event's own choice must be zero.  Missing observations are
    represented upstream by a zero gate, not by changing event identity.
    """

    features: torch.Tensor  # fixed u_j, [J, U]
    relation: torch.Tensor  # fixed R, [J, A]
    event_choice_index: torch.Tensor  # [J]
    gate: torch.Tensor  # centered cell-specific gate, [B, J]


@dataclass(frozen=True)
class EventOutput:
    raw_sensitivity: torch.Tensor  # beta before choice centering, [J, A]
    centered_sensitivity: torch.Tensor  # [J, A]
    correction: torch.Tensor  # direct gate-weighted event sum, [B, A]


@dataclass(frozen=True)
class PathReadoutInput:
    edge_energy: torch.Tensor  # [E] or [B, E]
    state_correction: torch.Tensor  # [B, A]
    dna_correction: torch.Tensor  # [B, A]
    rna_correction: torch.Tensor  # [B, A]
    path_edge_incidence: torch.Tensor  # sparse [P, E]
    path_choice_incidence: torch.Tensor  # sparse [P, A]


@dataclass(frozen=True)
class PathLogits:
    edge_logits: torch.Tensor  # [B, P]
    state_logits: torch.Tensor  # [B, P]
    dna_logits: torch.Tensor  # [B, P]
    rna_logits: torch.Tensor  # [B, P]
    length_logits: torch.Tensor  # [B, P]
    total_logits: torch.Tensor  # exact sum of the five terms, [B, P]


class EdgeGraphGPS(nn.Module):
    """One shallow edge-state GraphGPS block and one CIS energy readout."""

    def __init__(
        self,
        edge_feature_dim: int,
        hidden_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if edge_feature_dim < 1 or hidden_dim < 1 or attention_heads < 1:
            raise ValueError("GraphGPS dimensions must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.edge_projection = nn.Linear(edge_feature_dim, hidden_dim)
        local_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(local_update)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.edge_energy_readout = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, batch: GraphGPSBatch) -> CISOutput:
        _validate_graphgps_batch(batch, self.edge_feature_dim)
        x = self.edge_projection(batch.edge_features)

        # The imported local graph has adjacency but no additional relation
        # variable.  Zero GINE attributes therefore add no undocumented signal.
        local_attributes = x.new_zeros(
            (batch.local_edge_index.shape[1], self.hidden_dim)
        )
        local_state = self.local_conv(
            x, batch.local_edge_index, edge_attr=local_attributes
        )

        dense, valid = to_dense_batch(x, batch.edge_gene_index)
        global_dense, _ = self.global_attention(
            dense,
            dense,
            dense,
            key_padding_mask=~valid,
            need_weights=False,
        )
        global_state = global_dense[valid]
        edge_states = self.output_norm(x + local_state + global_state)
        edge_energy = self.edge_energy_readout(edge_states).squeeze(-1)
        return CISOutput(edge_states=edge_states, edge_energy=edge_energy)


def alternative_batch_from_catalog(
    catalog: ChoiceCatalog,
    *,
    device: torch.device | str | None = None,
) -> AlternativeBatch:
    """Convert the deterministic ChoiceCatalog order to padded model indices."""

    alternatives = [
        alternative for choice in catalog.choices for alternative in choice.alternatives
    ]
    if not alternatives:
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        return AlternativeBatch(
            edge_index=torch.empty((0, 0), dtype=torch.long, device=device),
            edge_mask=torch.empty((0, 0), dtype=torch.bool, device=device),
            choice_index=empty_long,
            scope_index=empty_long.clone(),
        )

    width = max(len(alternative.edge_indices) for alternative in alternatives)
    edge_index = torch.full(
        (len(alternatives), width), -1, dtype=torch.long, device=device
    )
    edge_mask = torch.zeros((len(alternatives), width), dtype=torch.bool, device=device)
    choice_indices: list[int] = []
    scope_indices: list[int] = []
    alternative_row = 0
    for choice_index, choice in enumerate(catalog.choices):
        try:
            scope_index = _SCOPE_INDEX[choice.scope]
        except KeyError as exc:
            raise ValueError(f"unknown choice scope: {choice.scope}") from exc
        for alternative in choice.alternatives:
            count = len(alternative.edge_indices)
            if count == 0:
                raise ValueError("an alternative cannot have zero processing edges")
            edge_index[alternative_row, :count] = torch.as_tensor(
                alternative.edge_indices, dtype=torch.long, device=device
            )
            edge_mask[alternative_row, :count] = True
            choice_indices.append(choice_index)
            scope_indices.append(scope_index)
            alternative_row += 1
    return AlternativeBatch(
        edge_index=edge_index,
        edge_mask=edge_mask,
        choice_index=torch.tensor(choice_indices, dtype=torch.long, device=device),
        scope_index=torch.tensor(scope_indices, dtype=torch.long, device=device),
    )


def build_frozen_alternative_state(
    edge_states: torch.Tensor,
    alternatives: AlternativeBatch,
) -> FrozenAlternativeState:
    """Build the exact frozen ``h_base`` specified by Architecture V1."""

    if edge_states.ndim != 2:
        raise ValueError("edge_states must have shape [edges, hidden_dim]")
    _validate_alternative_batch(alternatives, edge_states.shape[0], edge_states.device)
    frozen_edges = edge_states.detach().clone()
    if alternatives.edge_index.shape[0] == 0:
        h_base = frozen_edges.new_empty(
            (0, 3 * frozen_edges.shape[1] + 1 + len(CHOICE_SCOPES))
        )
    else:
        safe_index = alternatives.edge_index.clamp_min(0)
        selected = frozen_edges[safe_index]
        mask = alternatives.edge_mask.unsqueeze(-1)
        counts = alternatives.edge_mask.sum(dim=1)
        mean_state = (selected * mask).sum(dim=1) / counts[:, None]
        first_state = selected[:, 0]
        last_position = counts - 1
        last_state = selected[
            torch.arange(selected.shape[0], device=selected.device), last_position
        ]
        log_edge_count = counts.to(dtype=frozen_edges.dtype).log1p().unsqueeze(1)
        scope = F.one_hot(alternatives.scope_index, num_classes=len(CHOICE_SCOPES)).to(
            dtype=frozen_edges.dtype
        )
        h_base = torch.cat(
            (mean_state, first_state, last_state, log_edge_count, scope), dim=1
        )
    return FrozenAlternativeState(
        edge_states=frozen_edges,
        h_base=h_base.detach().clone(),
        choice_index=alternatives.choice_index.detach().clone(),
        scope_index=alternatives.scope_index.detach().clone(),
    )


class StateScorer(nn.Module):
    """The single bias-free low-rank State scorer."""

    def __init__(self, state_dim: int, alternative_dim: int, rank: int) -> None:
        super().__init__()
        _require_positive_projection_dims(state_dim, alternative_dim, rank)
        self.state_dim = state_dim
        self.alternative_dim = alternative_dim
        self.rank = rank
        self.U = nn.Linear(state_dim, rank, bias=False)
        self.V = nn.Linear(alternative_dim, rank, bias=False)
        nn.init.zeros_(self.V.weight)

    def forward(
        self,
        batch: StateBatch,
        alternatives: FrozenAlternativeState,
    ) -> StateOutput:
        if batch.features.ndim != 2 or batch.features.shape[1] != self.state_dim:
            raise ValueError("State features have the wrong shape")
        _validate_frozen_alternatives(alternatives, self.alternative_dim)
        if batch.features.device != alternatives.h_base.device:
            raise ValueError("State features and h_base must share a device")
        state_latent = self.U(batch.features)
        alternative_latent = self.V(alternatives.h_base)
        raw = state_latent @ alternative_latent.transpose(0, 1)
        correction = center_within_choice(raw, alternatives.choice_index)
        return StateOutput(raw_potential=raw, correction=correction)


class EventScorer(nn.Module):
    """One bias-free low-rank DNA or RNA event scorer."""

    def __init__(self, event_dim: int, alternative_dim: int, rank: int) -> None:
        super().__init__()
        _require_positive_projection_dims(event_dim, alternative_dim, rank)
        self.event_dim = event_dim
        self.alternative_dim = alternative_dim
        self.rank = rank
        self.U = nn.Linear(event_dim, rank, bias=False)
        self.V = nn.Linear(alternative_dim, rank, bias=False)
        nn.init.zeros_(self.V.weight)

    def forward(
        self,
        batch: EventBatch,
        alternatives: FrozenAlternativeState,
    ) -> EventOutput:
        _validate_frozen_alternatives(alternatives, self.alternative_dim)
        _validate_event_batch(batch, alternatives)
        if batch.features.shape[1] != self.event_dim:
            raise ValueError("event features have the wrong width")

        event_latent = self.U(batch.features)
        alternative_latent = self.V(alternatives.h_base)
        raw = (event_latent @ alternative_latent.transpose(0, 1)) * batch.relation
        centered = center_event_sensitivity(
            raw,
            event_choice_index=batch.event_choice_index,
            alternative_choice_index=alternatives.choice_index,
        )
        correction = batch.gate @ centered
        return EventOutput(
            raw_sensitivity=raw,
            centered_sensitivity=centered,
            correction=correction,
        )


class AugmentedPathReadout(nn.Module):
    """Sparse incidence sums for the five fixed V1 path-logit terms."""

    def __init__(self, length_penalty: float) -> None:
        super().__init__()
        if length_penalty < 0:
            raise ValueError("length_penalty must be non-negative")
        self.length_penalty = float(length_penalty)

    def forward(self, inputs: PathReadoutInput) -> PathLogits:
        state = _require_matrix("state_correction", inputs.state_correction)
        dna = _require_matrix("dna_correction", inputs.dna_correction)
        rna = _require_matrix("rna_correction", inputs.rna_correction)
        if dna.shape != state.shape or rna.shape != state.shape:
            raise ValueError("State, DNA, and RNA corrections must have the same shape")
        if dna.device != state.device or rna.device != state.device:
            raise ValueError("State, DNA, and RNA corrections must share a device")
        batch_size, alternative_count = state.shape

        edge = inputs.edge_energy
        if edge.ndim == 1:
            edge = edge.unsqueeze(0).expand(batch_size, -1)
        elif edge.ndim != 2 or edge.shape[0] != batch_size:
            raise ValueError("edge_energy must have shape [edges] or [cells, edges]")

        path_edge = _coalesced_sparse_matrix(
            "path_edge_incidence", inputs.path_edge_incidence
        )
        path_choice = _coalesced_sparse_matrix(
            "path_choice_incidence", inputs.path_choice_incidence
        )
        if path_edge.shape[1] != edge.shape[1]:
            raise ValueError("path-edge incidence and edge energy axes differ")
        if path_choice.shape != (path_edge.shape[0], alternative_count):
            raise ValueError("path-choice incidence and correction axes differ")

        edge_logits = _incidence_sum(path_edge, edge)
        state_logits = _incidence_sum(path_choice, state)
        dna_logits = _incidence_sum(path_choice, dna)
        rna_logits = _incidence_sum(path_choice, rna)
        path_lengths = _sparse_row_sum(path_edge, edge.dtype, edge.device)
        length_logits = (
            (-self.length_penalty * path_lengths).unsqueeze(0).expand(batch_size, -1)
        )
        total_logits = (
            edge_logits + state_logits + dna_logits + rna_logits + length_logits
        )
        return PathLogits(
            edge_logits=edge_logits,
            state_logits=state_logits,
            dna_logits=dna_logits,
            rna_logits=rna_logits,
            length_logits=length_logits,
            total_logits=total_logits,
        )


def center_within_choice(
    values: torch.Tensor, choice_index: torch.Tensor
) -> torch.Tensor:
    """Subtract the alternative mean separately for every choice."""

    if values.ndim != 2:
        raise ValueError("choice values must have shape [rows, alternatives]")
    if choice_index.ndim != 1 or choice_index.numel() != values.shape[1]:
        raise ValueError("choice_index must contain one value per alternative")
    if choice_index.numel() == 0:
        return values.clone()
    choice_index = choice_index.to(device=values.device, dtype=torch.long)
    if bool((choice_index < 0).any().item()):
        raise ValueError("choice_index must be non-negative")
    choice_count = int(choice_index.max().item()) + 1
    counts = torch.bincount(choice_index, minlength=choice_count)
    if bool((counts == 0).any().item()):
        raise ValueError("choice_index must be contiguous")
    sums = values.new_zeros((values.shape[0], choice_count))
    sums.index_add_(1, choice_index, values)
    means = sums / counts.to(dtype=values.dtype).unsqueeze(0)
    return values - means[:, choice_index]


def center_event_sensitivity(
    raw_sensitivity: torch.Tensor,
    *,
    event_choice_index: torch.Tensor,
    alternative_choice_index: torch.Tensor,
) -> torch.Tensor:
    """Center every event over all alternatives of its declared choice."""

    if raw_sensitivity.ndim != 2:
        raise ValueError("raw event sensitivity must have shape [events, alternatives]")
    event_choice_index = event_choice_index.to(
        device=raw_sensitivity.device, dtype=torch.long
    )
    alternative_choice_index = alternative_choice_index.to(
        device=raw_sensitivity.device, dtype=torch.long
    )
    membership = event_choice_index[:, None] == alternative_choice_index[None, :]
    counts = membership.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ValueError("an event references a choice with no alternatives")
    mean = raw_sensitivity.sum(dim=1) / counts.to(raw_sensitivity.dtype)
    return (raw_sensitivity - mean[:, None]) * membership


def freeze_cis_parent(cis: EdgeGraphGPS) -> EdgeGraphGPS:
    """Freeze the admitted CIS parent in place before building ``h_base``."""

    cis.requires_grad_(False)
    cis.eval()
    return cis


def clone_from_state_parent(state_parent: StateScorer) -> StateScorer:
    """Clone one selected State parent and freeze it for DNA/RNA/Full children."""

    child_parent = deepcopy(state_parent)
    child_parent.requires_grad_(False)
    child_parent.eval()
    return child_parent


def _validate_graphgps_batch(batch: GraphGPSBatch, feature_dim: int) -> None:
    if batch.edge_features.ndim != 2 or batch.edge_features.shape[1] != feature_dim:
        raise ValueError("edge_features have the wrong shape")
    if not torch.is_floating_point(batch.edge_features):
        raise TypeError("edge_features must use a floating dtype")
    edge_count = batch.edge_features.shape[0]
    if edge_count == 0:
        raise ValueError("GraphGPS requires at least one processing edge")
    if batch.local_edge_index.shape[0:1] != (2,) or batch.local_edge_index.ndim != 2:
        raise ValueError("local_edge_index must have shape [2, local_edges]")
    if batch.local_edge_index.dtype != torch.long:
        raise TypeError("local_edge_index must use torch.long")
    if batch.edge_gene_index.ndim != 1 or batch.edge_gene_index.numel() != edge_count:
        raise ValueError("edge_gene_index must contain one value per processing edge")
    if batch.edge_gene_index.dtype != torch.long:
        raise TypeError("edge_gene_index must use torch.long")
    if (
        batch.local_edge_index.device != batch.edge_features.device
        or batch.edge_gene_index.device != batch.edge_features.device
    ):
        raise ValueError("GraphGPS batch tensors must share a device")
    gene_index = batch.edge_gene_index
    if bool((gene_index < 0).any().item()) or bool(
        (gene_index[1:] < gene_index[:-1]).any().item()
    ):
        raise ValueError("edge_gene_index must be grouped and non-negative")
    observed_genes = torch.unique_consecutive(gene_index)
    expected_genes = torch.arange(
        observed_genes.numel(), device=gene_index.device, dtype=torch.long
    )
    if not torch.equal(observed_genes, expected_genes):
        raise ValueError("edge_gene_index must be contiguous from zero")
    if batch.local_edge_index.numel():
        if bool(
            ((batch.local_edge_index < 0) | (batch.local_edge_index >= edge_count))
            .any()
            .item()
        ):
            raise IndexError("local_edge_index is out of range")
        source, target = batch.local_edge_index
        if not torch.equal(gene_index[source], gene_index[target]):
            raise ValueError("local message passing crosses a gene boundary")


def _validate_alternative_batch(
    alternatives: AlternativeBatch, edge_count: int, device: torch.device
) -> None:
    if (
        alternatives.edge_index.ndim != 2
        or alternatives.edge_mask.shape != alternatives.edge_index.shape
    ):
        raise ValueError("alternative edge index and mask shapes differ")
    if (
        alternatives.edge_index.dtype != torch.long
        or alternatives.edge_mask.dtype != torch.bool
    ):
        raise TypeError("alternative edge indices must be long and masks bool")
    alternative_count = alternatives.edge_index.shape[0]
    if alternatives.choice_index.shape != (
        alternative_count,
    ) or alternatives.scope_index.shape != (alternative_count,):
        raise ValueError("alternative metadata must contain one value per alternative")
    if (
        alternatives.choice_index.dtype != torch.long
        or alternatives.scope_index.dtype != torch.long
    ):
        raise TypeError("alternative choice and scope indices must use torch.long")
    tensors = (
        alternatives.edge_index,
        alternatives.edge_mask,
        alternatives.choice_index,
        alternatives.scope_index,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("alternative tensors and edge states must share a device")
    if alternative_count == 0:
        return
    counts = alternatives.edge_mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ValueError("each alternative requires at least one edge")
    expected_mask = (
        torch.arange(alternatives.edge_index.shape[1], device=device)[None, :]
        < counts[:, None]
    )
    if not torch.equal(alternatives.edge_mask, expected_mask):
        raise ValueError(
            "alternative edge masks must be contiguous from the first edge"
        )
    valid = alternatives.edge_index[alternatives.edge_mask]
    if bool(((valid < 0) | (valid >= edge_count)).any().item()):
        raise IndexError("alternative edge index is out of range")
    if bool(
        (
            (alternatives.scope_index < 0)
            | (alternatives.scope_index >= len(CHOICE_SCOPES))
        )
        .any()
        .item()
    ):
        raise ValueError("alternative scope index is out of range")


def _validate_frozen_alternatives(
    alternatives: FrozenAlternativeState, alternative_dim: int
) -> None:
    if alternatives.h_base.ndim != 2 or alternatives.h_base.shape[1] != alternative_dim:
        raise ValueError("frozen h_base has the wrong shape")
    if alternatives.choice_index.shape != (alternatives.h_base.shape[0],):
        raise ValueError("frozen choice index and h_base axes differ")
    if alternatives.scope_index.shape != (alternatives.h_base.shape[0],):
        raise ValueError("frozen scope index and h_base axes differ")
    if (
        alternatives.choice_index.dtype != torch.long
        or alternatives.scope_index.dtype != torch.long
    ):
        raise TypeError("frozen choice and scope indices must use torch.long")
    if (
        alternatives.edge_states.device != alternatives.h_base.device
        or alternatives.choice_index.device != alternatives.h_base.device
        or alternatives.scope_index.device != alternatives.h_base.device
    ):
        raise ValueError("frozen CIS and alternative tensors must share a device")
    if alternatives.h_base.requires_grad or alternatives.edge_states.requires_grad:
        raise ValueError("q_e and h_base must be detached before dynamic scoring")


def _validate_event_batch(
    batch: EventBatch, alternatives: FrozenAlternativeState
) -> None:
    if batch.features.ndim != 2:
        raise ValueError("event features must have shape [events, features]")
    event_count = batch.features.shape[0]
    alternative_count = alternatives.h_base.shape[0]
    if batch.relation.shape != (event_count, alternative_count):
        raise ValueError("event relation must have shape [events, alternatives]")
    if batch.event_choice_index.shape != (event_count,):
        raise ValueError("event_choice_index must contain one value per event")
    if batch.event_choice_index.dtype != torch.long:
        raise TypeError("event_choice_index must use torch.long")
    if batch.gate.ndim != 2 or batch.gate.shape[1] != event_count:
        raise ValueError("event gate must have shape [cells, events]")
    tensors = (batch.features, batch.relation, batch.event_choice_index, batch.gate)
    if any(value.device != alternatives.h_base.device for value in tensors):
        raise ValueError("event and frozen alternative tensors must share a device")
    if event_count == 0:
        return
    event_choice = batch.event_choice_index.to(torch.long)
    membership = event_choice[:, None] == alternatives.choice_index[None, :]
    if bool((~membership & (batch.relation != 0)).any().item()):
        raise ValueError("event relation is non-zero outside its declared choice")


def _require_positive_projection_dims(left: int, right: int, rank: int) -> None:
    if left < 1 or right < 1 or rank < 1:
        raise ValueError("low-rank projection dimensions must be positive")


def _require_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [cells, alternatives]")
    return value


def _coalesced_sparse_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional sparse tensor")
    if value.layout == torch.sparse_coo:
        return value.coalesce()
    if value.layout in {torch.sparse_csr, torch.sparse_csc}:
        return value.to_sparse_coo().coalesce()
    raise TypeError(f"{name} must use a sparse COO/CSR/CSC layout")


def _incidence_sum(incidence: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    incidence = incidence.to(device=values.device, dtype=values.dtype)
    return torch.sparse.mm(incidence, values.transpose(0, 1)).transpose(0, 1)


def _sparse_row_sum(
    incidence: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    incidence = incidence.to(device=device, dtype=dtype).coalesce()
    result = torch.zeros(incidence.shape[0], dtype=dtype, device=device)
    result.index_add_(0, incidence.indices()[0], incidence.values())
    return result


__all__ = [
    "CHOICE_SCOPES",
    "AlternativeBatch",
    "AugmentedPathReadout",
    "CISOutput",
    "EdgeGraphGPS",
    "EventBatch",
    "EventOutput",
    "EventScorer",
    "FrozenAlternativeState",
    "GraphGPSBatch",
    "PathLogits",
    "PathReadoutInput",
    "StateBatch",
    "StateOutput",
    "StateScorer",
    "alternative_batch_from_catalog",
    "build_frozen_alternative_state",
    "center_event_sensitivity",
    "center_within_choice",
    "clone_from_state_parent",
    "freeze_cis_parent",
]
