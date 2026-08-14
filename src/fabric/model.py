"""FABRIC V2 cell-conditioned graph model.

The runtime has one scientific path: fixed post-cap event routes are aggregated
into DNA and RNA edge-token blocks, concatenated with static CIS features,
contextualized by one GraphGPS block, and scored over the frozen structural
path catalog.  Reporting indices never alter the forward multiplicity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv


PRIMARY_ABLATIONS = ("cis", "cis_dna", "cis_rna", "full")
ARCHITECTURE_COMPARATOR = "full_additive_edge"
TRAINING_CONDITIONS = (*PRIMARY_ABLATIONS, ARCHITECTURE_COMPARATOR)


@dataclass(frozen=True)
class RoutedModalityInput:
    """Post-cap production routes and cell-specific shared-key gates.

    Route feature rows are already encoded in the frozen EventFeatureManifest
    order.  ``event_gate_key_index`` deliberately separates physical-event and
    gate-key axes: several events may reference one biological proxy.
    """

    route_event_index: torch.Tensor  # long [R], indexes the active event axis
    route_edge_index: torch.Tensor  # long [R], indexes the gene edge axis
    route_weight: torch.Tensor  # float [R], sums to one within each event
    route_base_features: torch.Tensor  # float [R, U_base]
    route_interaction_features: torch.Tensor  # float [R, U_int]
    interaction_active_mask: torch.Tensor  # bool [U_int]
    event_gate_key_index: torch.Tensor  # long [J], indexes gate columns
    gate: torch.Tensor  # float [B, K]
    event_keep_mask: torch.Tensor | None = None  # bool [J], counterfactual only
    route_keep_mask: torch.Tensor | None = None  # bool [R], route-term selector

    def with_event_keep_mask(self, mask: torch.Tensor) -> "RoutedModalityInput":
        """Return an explicit evidence-neutralization input."""

        return replace(self, event_keep_mask=mask)

    def with_route_keep_mask(self, mask: torch.Tensor) -> "RoutedModalityInput":
        """Return a route-union/anchor-region neutralization input.

        The frozen production weights are not renormalized: deleting route
        terms is the counterfactual defined by V2, not a new routing catalog.
        """

        return replace(self, route_keep_mask=mask)


@dataclass(frozen=True)
class GeneCellModelInput:
    """One frozen gene graph evaluated for ``B`` independent cells."""

    cis_features: torch.Tensor  # float [E, C]
    local_edge_index: torch.Tensor  # long [2, L]
    dna: RoutedModalityInput
    rna: RoutedModalityInput
    path_edge_incidence: torch.Tensor  # sparse [P, E]
    path_first_edge_index: torch.Tensor  # long [P], transcript direction
    path_last_edge_index: torch.Tensor  # long [P], transcript direction
    log_edge_count: torch.Tensor  # float [P], exactly log1p(path edge count)


@dataclass(frozen=True)
class PathReadoutOutput:
    logits: torch.Tensor  # [B, P]
    path_residual: torch.Tensor | None = None  # zeta, [B, P, H]
    path_vector: torch.Tensor | None = None  # [B, P, 3H+1]
    first_layer_preactivation: torch.Tensor | None = None  # [B, P, H_path]


@dataclass(frozen=True)
class FABRICOutput:
    """Predictions plus the already-computed named audit intermediates."""

    path_logits: torch.Tensor  # [B, P]
    dna_aggregate: torch.Tensor  # a^DNA after the modality ablation, [B,E,D]
    rna_aggregate: torch.Tensor  # a^RNA after the modality ablation, [B,E,D]
    full_dna_aggregate: torch.Tensor  # unmasked aggregate for algebra audits
    full_rna_aggregate: torch.Tensor
    joint_input: torch.Tensor  # x=[c,d,r], [B,E,C+2D]
    joint_projected: torch.Tensor  # y, [B,E,H]
    normalized_tokens: torch.Tensor  # y_hat, [B,E,H]
    edge_states: torch.Tensor  # H, [B,E,H]
    path_residual: torch.Tensor | None
    path_vector: torch.Tensor | None
    path_first_layer_preactivation: torch.Tensor | None


class RoutedEventAggregator(nn.Module):
    """Shared bias-free base/interaction projections followed by route sums."""

    def __init__(self, base_dim: int, interaction_dim: int, output_dim: int) -> None:
        super().__init__()
        if base_dim < 1 or interaction_dim < 0 or output_dim < 1:
            raise ValueError("event projection dimensions are invalid")
        self.base_dim = int(base_dim)
        self.interaction_dim = int(interaction_dim)
        self.output_dim = int(output_dim)
        self.base_projection = nn.Linear(base_dim, output_dim, bias=False)
        self.interaction_projection = (
            nn.Linear(interaction_dim, output_dim, bias=False)
            if interaction_dim
            else None
        )

    def route_projection(self, routed: RoutedModalityInput) -> torch.Tensor:
        """Return the fixed-route projection before gate and routing weights."""

        _validate_routed_modality(routed, self.base_dim, self.interaction_dim)
        projected = _bias_free_projection(
            routed.route_base_features, self.base_projection
        )
        if self.interaction_projection is not None:
            active = routed.interaction_active_mask.to(
                device=routed.route_interaction_features.device,
                dtype=routed.route_interaction_features.dtype,
            )
            projected = projected + _bias_free_projection(
                _scale_sparse_or_dense_columns(
                    routed.route_interaction_features, active
                ),
                self.interaction_projection,
            )
        return projected

    def forward(self, routed: RoutedModalityInput, edge_count: int) -> torch.Tensor:
        if edge_count < 1:
            raise ValueError("event aggregation requires at least one edge token")
        projected = self.route_projection(routed)
        if routed.route_edge_index.numel() and bool(
            ((routed.route_edge_index < 0) | (routed.route_edge_index >= edge_count))
            .any()
            .item()
        ):
            raise IndexError("route_edge_index is out of range")
        batch_size = routed.gate.shape[0]
        result = projected.new_zeros((batch_size, edge_count, self.output_dim))
        if projected.shape[0] == 0:
            return result
        event_gate = routed.gate[:, routed.event_gate_key_index]
        if routed.event_keep_mask is not None:
            event_gate = event_gate * routed.event_keep_mask.to(
                event_gate.device, dtype=event_gate.dtype
            ).unsqueeze(0)
        route_gate = event_gate[:, routed.route_event_index]
        if routed.route_keep_mask is not None:
            route_gate = route_gate * routed.route_keep_mask.to(
                route_gate.device, dtype=route_gate.dtype
            ).unsqueeze(0)
        contribution = (
            route_gate.unsqueeze(-1)
            * routed.route_weight.to(projected.dtype)[None, :, None]
            * projected[None, :, :]
        )
        result.index_add_(1, routed.route_edge_index, contribution)
        return result


class GraphGPSBlock(nn.Module):
    """Exactly one local-message/global-attention block on edge tokens."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        if hidden_dim < 1 or attention_heads < 1:
            raise ValueError("GraphGPS dimensions must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = int(hidden_dim)
        local_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(local_update)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, normalized_tokens: torch.Tensor, local_edge_index: torch.Tensor
    ) -> torch.Tensor:
        if normalized_tokens.ndim != 3 or normalized_tokens.shape[2] != self.hidden_dim:
            raise ValueError("GraphGPS tokens must have shape [cells, edges, hidden]")
        if local_edge_index.ndim != 2 or local_edge_index.shape[0] != 2:
            raise ValueError("local_edge_index must have shape [2, local edges]")
        if local_edge_index.dtype != torch.long:
            raise TypeError("local_edge_index must use torch.long")
        batch_size, edge_count, hidden_dim = normalized_tokens.shape
        if batch_size < 1 or edge_count < 1:
            raise ValueError("GraphGPS requires non-empty cell and edge axes")
        if local_edge_index.numel() and bool(
            ((local_edge_index < 0) | (local_edge_index >= edge_count)).any().item()
        ):
            raise IndexError("local_edge_index is out of range")

        # Each cell receives a disjoint copy of the line graph.  Global
        # attention operates on [B,E,H] directly, so no key/value can cross a
        # cell boundary.
        flat = normalized_tokens.reshape(batch_size * edge_count, hidden_dim)
        if local_edge_index.shape[1]:
            offsets = (
                torch.arange(batch_size, device=flat.device, dtype=torch.long)
                * edge_count
            )
            expanded = local_edge_index[:, None, :] + offsets[None, :, None]
            expanded = expanded.permute(0, 2, 1).reshape(2, -1)
        else:
            expanded = local_edge_index.new_empty((2, 0))
        local_attributes = flat.new_zeros((expanded.shape[1], hidden_dim))
        local = self.local_conv(flat, expanded, edge_attr=local_attributes).reshape(
            batch_size, edge_count, hidden_dim
        )
        global_state, _ = self.global_attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        return self.output_norm(normalized_tokens + local + global_state)


class PathContextReadout(nn.Module):
    """Gene-centered residual path pooling followed by one shared hidden layer."""

    def __init__(self, hidden_dim: int, path_hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1 or path_hidden_dim < 1:
            raise ValueError("path readout dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.path_hidden_dim = int(path_hidden_dim)
        self.hidden = nn.Linear(3 * hidden_dim + 1, path_hidden_dim)
        self.output = nn.Linear(path_hidden_dim, 1)

    def forward(
        self,
        edge_states: torch.Tensor,
        path_edge_incidence: torch.Tensor,
        path_first_edge_index: torch.Tensor,
        path_last_edge_index: torch.Tensor,
        log_edge_count: torch.Tensor,
    ) -> PathReadoutOutput:
        _validate_path_inputs(
            edge_states,
            path_edge_incidence,
            path_first_edge_index,
            path_last_edge_index,
            log_edge_count,
        )
        incidence = _coalesced_sparse_matrix(path_edge_incidence).to(
            device=edge_states.device, dtype=edge_states.dtype
        )
        pooled = torch.stack(
            [torch.sparse.mm(incidence, states) for states in edge_states], dim=0
        )
        residual = pooled - pooled.mean(dim=1, keepdim=True)
        first = edge_states[:, path_first_edge_index, :]
        last = edge_states[:, path_last_edge_index, :]
        length = log_edge_count.to(edge_states).view(1, -1, 1).expand(
            edge_states.shape[0], -1, -1
        )
        path_vector = torch.cat((residual, first, last, length), dim=-1)
        preactivation = self.hidden(path_vector)
        logits = self.output(F.gelu(preactivation)).squeeze(-1)
        return PathReadoutOutput(
            logits=logits,
            path_residual=residual,
            path_vector=path_vector,
            first_layer_preactivation=preactivation,
        )


class AdditiveEdgeReadout(nn.Module):
    """The single frozen architecture comparator from V2 section 15.1."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("additive readout hidden_dim must be positive")
        self.edge_readout = nn.Linear(hidden_dim, 1, bias=False)
        self.beta_len = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        edge_states: torch.Tensor,
        path_edge_incidence: torch.Tensor,
        path_first_edge_index: torch.Tensor,
        path_last_edge_index: torch.Tensor,
        log_edge_count: torch.Tensor,
    ) -> PathReadoutOutput:
        _validate_path_inputs(
            edge_states,
            path_edge_incidence,
            path_first_edge_index,
            path_last_edge_index,
            log_edge_count,
        )
        incidence = _coalesced_sparse_matrix(path_edge_incidence).to(
            device=edge_states.device, dtype=edge_states.dtype
        )
        edge_scores = self.edge_readout(edge_states).squeeze(-1)
        path_scores = torch.stack(
            [torch.sparse.mm(incidence, values[:, None]).squeeze(1) for values in edge_scores],
            dim=0,
        )
        logits = path_scores + self.beta_len * log_edge_count.to(edge_states)[None, :]
        return PathReadoutOutput(logits=logits)


class FABRICV2Model(nn.Module):
    """The only FABRIC V2 production model runtime."""

    def __init__(
        self,
        *,
        cis_dim: int,
        dna_base_dim: int,
        dna_interaction_dim: int,
        rna_base_dim: int,
        rna_interaction_dim: int,
        dynamic_dim: int,
        hidden_dim: int,
        attention_heads: int,
        path_hidden_dim: int,
        readout_kind: str = "path_context",
    ) -> None:
        super().__init__()
        if cis_dim < 1 or dynamic_dim < 1 or hidden_dim < 1:
            raise ValueError("FABRIC model dimensions must be positive")
        if readout_kind not in {"path_context", "additive_edge"}:
            raise ValueError("readout_kind must be path_context or additive_edge")
        self.cis_dim = int(cis_dim)
        self.dynamic_dim = int(dynamic_dim)
        self.hidden_dim = int(hidden_dim)
        self.readout_kind = readout_kind
        self.dna_aggregator = RoutedEventAggregator(
            dna_base_dim, dna_interaction_dim, dynamic_dim
        )
        self.rna_aggregator = RoutedEventAggregator(
            rna_base_dim, rna_interaction_dim, dynamic_dim
        )
        self.joint_projection = nn.Linear(cis_dim + 2 * dynamic_dim, hidden_dim)
        self.pre_layer_norm = nn.LayerNorm(
            hidden_dim, eps=1.0e-5, elementwise_affine=False
        )
        self.graphgps = GraphGPSBlock(hidden_dim, attention_heads)
        self.readout: PathContextReadout | AdditiveEdgeReadout
        if readout_kind == "path_context":
            self.readout = PathContextReadout(hidden_dim, path_hidden_dim)
        else:
            self.readout = AdditiveEdgeReadout(hidden_dim)

    def forward(
        self, inputs: GeneCellModelInput, *, condition: str = "full"
    ) -> FABRICOutput:
        if condition not in PRIMARY_ABLATIONS:
            raise ValueError(f"unknown primary modality condition: {condition}")
        _validate_gene_cell_input(inputs, self.cis_dim)
        edge_count = inputs.cis_features.shape[0]
        full_dna = self.dna_aggregator(inputs.dna, edge_count)
        full_rna = self.rna_aggregator(inputs.rna, edge_count)
        use_dna = condition in {"cis_dna", "full"}
        use_rna = condition in {"cis_rna", "full"}
        dna = full_dna if use_dna else torch.zeros_like(full_dna)
        rna = full_rna if use_rna else torch.zeros_like(full_rna)
        batch_size = full_dna.shape[0]
        cis = inputs.cis_features.to(full_dna).unsqueeze(0).expand(batch_size, -1, -1)
        joint_input = torch.cat((cis, dna, rna), dim=-1)
        joint_projected = self.joint_projection(joint_input)
        normalized = self.pre_layer_norm(joint_projected)
        edge_states = self.graphgps(normalized, inputs.local_edge_index)
        path = self.readout(
            edge_states,
            inputs.path_edge_incidence,
            inputs.path_first_edge_index,
            inputs.path_last_edge_index,
            inputs.log_edge_count,
        )
        if not torch.isfinite(path.logits).all():
            raise FloatingPointError("FABRIC produced non-finite path logits")
        return FABRICOutput(
            path_logits=path.logits,
            dna_aggregate=dna,
            rna_aggregate=rna,
            full_dna_aggregate=full_dna,
            full_rna_aggregate=full_rna,
            joint_input=joint_input,
            joint_projected=joint_projected,
            normalized_tokens=normalized,
            edge_states=edge_states,
            path_residual=path.path_residual,
            path_vector=path.path_vector,
            path_first_layer_preactivation=path.first_layer_preactivation,
        )


def _validate_routed_modality(
    routed: RoutedModalityInput, base_dim: int, interaction_dim: int
) -> None:
    route_count = routed.route_event_index.numel()
    if routed.route_event_index.ndim != 1 or routed.route_edge_index.shape != (route_count,):
        raise ValueError("route event and edge indices must be one-dimensional and aligned")
    if routed.route_event_index.dtype != torch.long or routed.route_edge_index.dtype != torch.long:
        raise TypeError("route indices must use torch.long")
    if routed.route_weight.shape != (route_count,) or not torch.is_floating_point(
        routed.route_weight
    ):
        raise ValueError("route_weight must be floating [routes]")
    if routed.route_base_features.shape != (route_count, base_dim):
        raise ValueError("route base-feature axis differs from the frozen model axis")
    if routed.route_interaction_features.shape != (route_count, interaction_dim):
        raise ValueError("route interaction-feature axis differs from the frozen model axis")
    if routed.interaction_active_mask.shape != (interaction_dim,):
        raise ValueError("interaction active mask has the wrong shape")
    if routed.interaction_active_mask.dtype != torch.bool:
        raise TypeError("interaction_active_mask must use bool dtype")
    event_count = routed.event_gate_key_index.numel()
    if routed.event_gate_key_index.ndim != 1 or routed.event_gate_key_index.dtype != torch.long:
        raise TypeError("event_gate_key_index must be a one-dimensional long tensor")
    if routed.gate.ndim != 2 or not torch.is_floating_point(routed.gate):
        raise ValueError("gate must be floating [cells, gate keys]")
    tensors = (
        routed.route_event_index,
        routed.route_edge_index,
        routed.route_weight,
        routed.route_base_features,
        routed.route_interaction_features,
        routed.interaction_active_mask,
        routed.event_gate_key_index,
        routed.gate,
    )
    if len({value.device for value in tensors}) != 1:
        raise ValueError("all routed modality tensors must share one device")
    if routed.event_keep_mask is not None:
        if routed.event_keep_mask.shape != (event_count,) or routed.event_keep_mask.dtype != torch.bool:
            raise TypeError("event_keep_mask must be bool [events]")
        if routed.event_keep_mask.device != routed.gate.device:
            raise ValueError("event_keep_mask and routed tensors must share a device")
    if routed.route_keep_mask is not None:
        if routed.route_keep_mask.shape != (route_count,) or routed.route_keep_mask.dtype != torch.bool:
            raise TypeError("route_keep_mask must be bool [routes]")
        if routed.route_keep_mask.device != routed.gate.device:
            raise ValueError("route_keep_mask and routed tensors must share a device")
    if not all(
        _all_finite(value)
        for value in (
            routed.route_weight,
            routed.route_base_features,
            routed.route_interaction_features,
            routed.gate,
        )
    ):
        raise ValueError("routed modality contains a non-finite model value")
    if route_count:
        if event_count == 0 or bool(
            ((routed.route_event_index < 0) | (routed.route_event_index >= event_count))
            .any()
            .item()
        ):
            raise IndexError("route_event_index is out of range")
        if bool((routed.route_weight <= 0).any().item()):
            raise ValueError("production route weights must be strictly positive")
        totals = routed.route_weight.new_zeros(event_count)
        totals.index_add_(0, routed.route_event_index, routed.route_weight)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=1e-6):
            raise ValueError("production route weights must sum to one for every event")
    elif event_count:
        raise ValueError("a model-active event cannot have zero retained routes")
    if event_count and (
        routed.gate.shape[1] == 0
        or bool(
            (
                (routed.event_gate_key_index < 0)
                | (routed.event_gate_key_index >= routed.gate.shape[1])
            )
            .any()
            .item()
        )
    ):
        raise IndexError("event_gate_key_index is out of range")


def _bias_free_projection(values: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """Apply one bias-free projection to dense or sparse route features."""

    if layer.bias is not None:
        raise ValueError("sparse route projection requires the contracted bias-free layer")
    if values.layout == torch.strided:
        return layer(values)
    if values.layout not in {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}:
        raise TypeError("route features must use dense, COO, CSR, or CSC layout")
    sparse_values = (
        values.coalesce()
        if values.layout == torch.sparse_coo
        else values.to_sparse_coo().coalesce()
    )
    return torch.sparse.mm(sparse_values, layer.weight.transpose(0, 1))


def _scale_sparse_or_dense_columns(
    values: torch.Tensor, column_scale: torch.Tensor
) -> torch.Tensor:
    if values.layout == torch.strided:
        return values * column_scale.unsqueeze(0)
    sparse_values = (
        values.coalesce()
        if values.layout == torch.sparse_coo
        else values.to_sparse_coo().coalesce()
    )
    indices = sparse_values.indices()
    scaled = sparse_values.values() * column_scale[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        scaled,
        sparse_values.shape,
        device=sparse_values.device,
        dtype=sparse_values.dtype,
    ).coalesce()


def _all_finite(value: torch.Tensor) -> bool:
    if value.layout == torch.strided:
        checked = value
    elif value.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}:
        checked = (
            value.coalesce().values()
            if value.layout == torch.sparse_coo
            else value.values()
        )
    else:
        raise TypeError("model tensor uses an unsupported layout")
    return bool(torch.isfinite(checked).all().item())


def _validate_gene_cell_input(inputs: GeneCellModelInput, cis_dim: int) -> None:
    if inputs.cis_features.ndim != 2 or inputs.cis_features.shape[1] != cis_dim:
        raise ValueError("cis_features have the wrong shape")
    if not torch.is_floating_point(inputs.cis_features) or not torch.isfinite(
        inputs.cis_features
    ).all():
        raise ValueError("cis_features must be finite floating values")
    if inputs.local_edge_index.device != inputs.cis_features.device:
        raise ValueError("graph input tensors must share one device")
    if inputs.dna.gate.shape[0] != inputs.rna.gate.shape[0]:
        raise ValueError("DNA and RNA gate cell axes differ")
    if inputs.dna.gate.shape[0] < 1:
        raise ValueError("a gene-cell input requires at least one cell")
    if inputs.dna.gate.device != inputs.cis_features.device or inputs.rna.gate.device != inputs.cis_features.device:
        raise ValueError("all model input tensors must share one device")


def _validate_path_inputs(
    edge_states: torch.Tensor,
    path_edge_incidence: torch.Tensor,
    first: torch.Tensor,
    last: torch.Tensor,
    log_edge_count: torch.Tensor,
) -> None:
    if edge_states.ndim != 3:
        raise ValueError("edge_states must have shape [cells, edges, hidden]")
    if path_edge_incidence.layout == torch.sparse_coo:
        if path_edge_incidence._nnz() != path_edge_incidence.coalesce()._nnz():
            raise ValueError("path incidence contains duplicate edge entries")
    elif path_edge_incidence.layout in {torch.sparse_csr, torch.sparse_csc}:
        sparse_coo = path_edge_incidence.to_sparse_coo()
        if path_edge_incidence._nnz() != sparse_coo.coalesce()._nnz():
            raise ValueError("path incidence contains duplicate edge entries")
    incidence = _coalesced_sparse_matrix(path_edge_incidence)
    path_count, edge_count = incidence.shape
    if edge_states.shape[1] != edge_count:
        raise ValueError("path incidence and edge-state axes differ")
    if path_count < 1:
        raise ValueError("path catalog must be non-empty")
    if incidence.values().dtype == torch.bool:
        binary_values = incidence.values()
    else:
        binary_values = incidence.values() == 1
    if not bool(binary_values.all().item()):
        raise ValueError("path incidence must contain only binary unit entries")
    row_counts = torch.bincount(
        incidence.indices()[0], minlength=path_count
    )
    if bool((row_counts == 0).any().item()):
        raise ValueError("every structural path must contain at least one edge")
    for name, value in (("first", first), ("last", last)):
        if value.shape != (path_count,) or value.dtype != torch.long:
            raise TypeError(f"path {name}-edge index must be long [paths]")
        if bool(((value < 0) | (value >= edge_count)).any().item()):
            raise IndexError(f"path {name}-edge index is out of range")
    if log_edge_count.shape != (path_count,) or not torch.is_floating_point(log_edge_count):
        raise ValueError("log_edge_count must be floating [paths]")
    tensors = (incidence, first, last, log_edge_count)
    if any(value.device != edge_states.device for value in tensors):
        raise ValueError("path and edge-state tensors must share one device")
    if not torch.isfinite(log_edge_count).all():
        raise ValueError("log_edge_count contains non-finite values")
    row_index, edge_index = incidence.indices()
    for name, endpoints in (("first", first), ("last", last)):
        present = torch.zeros(path_count, dtype=torch.bool, device=incidence.device)
        present[row_index[edge_index == endpoints[row_index]]] = True
        if not bool(present.all().item()):
            raise ValueError(f"path {name}-edge index is not contained in its path")
    expected_log_count = torch.log1p(row_counts.to(log_edge_count.dtype))
    if not torch.allclose(
        log_edge_count, expected_log_count, atol=1e-7, rtol=1e-7
    ):
        raise ValueError("log_edge_count differs from binary path incidence")


def _coalesced_sparse_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("path_edge_incidence must be two-dimensional")
    if value.layout == torch.sparse_coo:
        return value.coalesce()
    if value.layout in {torch.sparse_csr, torch.sparse_csc}:
        return value.to_sparse_coo().coalesce()
    raise TypeError("path_edge_incidence must use a sparse layout")


__all__ = [
    "ARCHITECTURE_COMPARATOR",
    "PRIMARY_ABLATIONS",
    "TRAINING_CONDITIONS",
    "AdditiveEdgeReadout",
    "FABRICOutput",
    "FABRICV2Model",
    "GeneCellModelInput",
    "GraphGPSBlock",
    "PathContextReadout",
    "PathReadoutOutput",
    "RoutedEventAggregator",
    "RoutedModalityInput",
]
