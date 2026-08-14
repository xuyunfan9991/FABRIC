from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import torch

from fabric.dataset import (
    ActivityContext,
    GateRawSignals,
    GateValues,
    RouteBaseDesign,
    assess_atac_mapping,
    build_canonical_interaction_design,
    build_event_feature_manifest,
    build_production_modality_tensors,
    build_raw_gate_signals,
    encode_base_route_features,
    fit_gate_admission,
    map_atac_accessibility,
    transform_gates,
)
from fabric.evaluate import (
    EvidenceSelectorResolution,
    apply_joint_cell_permutation,
    build_pairing_permutation_assignments,
    run_evidence_counterfactual,
)
from fabric.model import PRIMARY_ABLATIONS, RoutedEventAggregator, RoutedModalityInput
from fabric.motifs import EVENT_ROUTE_COLUMNS, PHYSICAL_EVENT_COLUMNS
from fabric.train import build_paired_models, load_config, make_toy_genes


def _catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a full-rank split-neutral route catalog with provenance sentinels."""

    specifications = (
        ("A", "promoter", "UPSTREAM", -2.0, 1.0),
        ("A", "intragenic", "DOWNSTREAM", 3.0, 2.0),
        ("A", "promoter", "DOWNSTREAM", 7.0, 4.0),
        ("A", "intragenic", "UPSTREAM", -8.0, 3.0),
        ("B", "promoter", "DOWNSTREAM", 2.0, 5.0),
        ("B", "intragenic", "UPSTREAM", -3.0, 7.0),
        ("B", "promoter", "UPSTREAM", -7.0, 8.0),
        ("B", "intragenic", "DOWNSTREAM", 8.0, 6.0),
    )
    events: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    for index, (factor, region, side, distance, peak_support) in enumerate(
        specifications
    ):
        event = {column: None for column in PHYSICAL_EVENT_COLUMNS}
        event.update(
            {
                "event_id": f"event-provenance-{index}",
                "target_gene_id": f"gene-provenance-{index % 2}",
                "factor_entity_id": factor,
                "factor_identity_kind": "unique",
                "cap_evidence_class": "motif_anchored",
                "candidate_factor_ids": [factor],
                "activity_entity_id": factor,
                "activity_gene_ids": [factor],
                "activity_proxy_rule": "unique_gene_cp10k_log1p",
                "modality": "DNA",
                "event_kind": "TSS_PROXIMAL",
                "motif_id": f"motif-provenance-{index}",
                "motif_equivalence_family_id": f"family-provenance-{index}",
                "source_motif_ids": [f"motif-provenance-{index}"],
                "chromosome": "chr-provenance",
                "start": 900_000 + 10 * index,
                "end": 900_004 + 10 * index,
                "strand": "+",
                "source_hit_coordinates": [
                    {
                        "source_hit_id": f"hit-provenance-{index}",
                        "start": 900_000 + 10 * index,
                        "end": 900_004 + 10 * index,
                    }
                ],
                "motif_score": 0.5,
                "orientation": "same_transcript",
                "peak_id": f"peak-provenance-{index}",
                "peak_support": peak_support,
                "gate_key_id": f"gate-{index}",
                "is_self_factor": False,
                "source_valid": True,
                "has_retained_route": True,
                "gate_key_active": True,
                "model_active": True,
                "admission_reasons": [],
            }
        )
        events.append(event)
        route = {column: None for column in EVENT_ROUTE_COLUMNS}
        route.update(
            {
                "route_id": f"route-{index}",
                "event_id": f"event-provenance-{index}",
                "target_gene_id": f"gene-provenance-{index % 2}",
                "modality": "DNA",
                "anchor_region_id": f"anchor-{index}",
                "anchor_site_id": f"site-{index}",
                "edge_id": f"edge-{index}",
                "route_weight": 1.0,
                "region_type": region,
                "anchor_type": "TSS" if index % 3 else "PAS",
                "transcript_oriented_side": side,
                "signed_distance_bp": distance,
                "edge_relative_position": np.nan,
                "distance_to_5prime_boundary_bp": np.nan,
                "distance_to_3prime_boundary_bp": np.nan,
                "geometry_kind": "site_window",
            }
        )
        routes.append(route)

    # OPEN_ONLY participates in the bias-free base identity but is excluded
    # from the factor-specific interaction denominator.
    open_event = dict(events[0])
    open_event.update(
        event_id="event-provenance-open",
        factor_entity_id=None,
        factor_identity_kind="accessibility_only",
        cap_evidence_class="accessibility_only",
        candidate_factor_ids=[],
        activity_entity_id=None,
        activity_gene_ids=[],
        activity_proxy_rule=None,
        motif_id=None,
        motif_equivalence_family_id=None,
        source_motif_ids=[],
        orientation=None,
        peak_id="peak-provenance-open",
        gate_key_id="gate-open",
        is_self_factor=None,
    )
    events.append(open_event)
    open_route = dict(routes[0])
    open_route.update(
        route_id="route-open",
        event_id="event-provenance-open",
        edge_id="edge-open",
    )
    routes.append(open_route)
    return pd.DataFrame(events), pd.DataFrame(routes)


def _manifest_and_design(
    events: pd.DataFrame, routes: pd.DataFrame
) -> tuple[dict[str, object], RouteBaseDesign, object]:
    manifest = build_event_feature_manifest(
        events,
        routes,
        distance_bin_boundaries={"DNA": [5], "RNA": [5]},
        scientific_context_pairs={
            "DNA": {
                "region_type": [["intragenic", "promoter"]],
                "anchor_type": [["PAS", "TSS"]],
                "distance_bin": [["[0,5)", "[5,inf]"]],
            },
            "RNA": {},
        },
    )
    base = encode_base_route_features(events, routes, manifest)
    support_rows: list[dict[str, object]] = []
    for modality in ("DNA", "RNA"):
        configuration = manifest["modalities"][modality]
        for field, field_configuration in configuration["context_fields"].items():
            for factor in configuration["interaction_factor_vocabulary"]:
                for level in field_configuration["raw_levels"]:
                    support_rows.append(
                        {
                            "modality": modality,
                            "context_field": field,
                            "factor_entity_id": factor,
                            "context_level": level,
                            "raw_cell_supported": True,
                        }
                    )
    interaction = build_canonical_interaction_design(
        base, pd.DataFrame(support_rows)
    )
    return manifest, base, interaction


def _toy_model(seed: int = 43):
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    model = build_paired_models(
        gene, config["model"], seed=seed, device="cpu"
    )["full"]
    model.eval()
    return model, gene


def test_provenance_ids_and_absolute_coordinates_are_invariant_to_numeric_route_design():
    events, routes = _catalog()
    manifest, base, interaction = _manifest_and_design(events, routes)

    changed_events = events.copy(deep=True)
    changed_routes = routes.copy(deep=True)
    event_mapping = {
        event_id: f"changed-event-{index}"
        for index, event_id in enumerate(changed_events["event_id"].astype(str))
    }
    gene_mapping = {
        gene_id: f"changed-gene-{index}"
        for index, gene_id in enumerate(
            sorted(set(changed_events["target_gene_id"].astype(str)))
        )
    }
    changed_events["event_id"] = changed_events["event_id"].astype(str).map(
        event_mapping
    )
    changed_routes["event_id"] = changed_routes["event_id"].astype(str).map(
        event_mapping
    )
    changed_events["target_gene_id"] = (
        changed_events["target_gene_id"].astype(str).map(gene_mapping)
    )
    changed_routes["target_gene_id"] = (
        changed_routes["target_gene_id"].astype(str).map(gene_mapping)
    )
    changed_events["chromosome"] = "chr-changed-provenance"
    changed_events["start"] = np.arange(len(changed_events)) * 17 + 7_000_000
    changed_events["end"] = changed_events["start"] + 4
    changed_events["peak_id"] = [
        f"changed-peak-{index}" for index in range(len(changed_events))
    ]
    changed_events["transcript_id"] = [
        f"changed-transcript-{index}" for index in range(len(changed_events))
    ]
    changed_routes["transcript_id"] = [
        f"changed-transcript-{index}" for index in range(len(changed_routes))
    ]
    motif_rows = changed_events["factor_identity_kind"].ne("accessibility_only")
    changed_events.loc[motif_rows, "motif_id"] = [
        f"changed-motif-{index}" for index in np.flatnonzero(motif_rows)
    ]
    changed_events.loc[motif_rows, "motif_equivalence_family_id"] = [
        f"changed-family-{index}" for index in np.flatnonzero(motif_rows)
    ]
    for index in np.flatnonzero(motif_rows):
        changed_events.at[index, "source_motif_ids"] = [
            changed_events.at[index, "motif_id"]
        ]

    changed_manifest, changed_base, changed_interaction = _manifest_and_design(
        changed_events, changed_routes
    )
    assert changed_manifest == manifest
    assert changed_base.column_names == base.column_names
    np.testing.assert_array_equal(changed_base.values, base.values)
    for modality in ("DNA", "RNA"):
        np.testing.assert_array_equal(
            changed_interaction.values_by_modality[modality],
            interaction.values_by_modality[modality],
        )
        np.testing.assert_array_equal(
            changed_interaction.active_mask_by_modality[modality],
            interaction.active_mask_by_modality[modality],
        )

    serialized_columns = json.dumps(base.column_names)
    serialized_interaction = json.dumps(interaction.manifest, sort_keys=True)
    for prohibited_token in (
        "event-provenance",
        "peak-provenance",
        "motif-provenance",
        "family-provenance",
        "gene-provenance",
        "changed-transcript",
        "900000",
    ):
        assert prohibited_token not in serialized_columns
        assert prohibited_token not in serialized_interaction


def test_supported_factor_position_swap_collides_without_interaction_and_is_selectable_with_it():
    context = pd.DataFrame(
        {
            "route_id": ("A-up", "A-down", "B-up", "B-down"),
            "event_id": ("e-A-up", "e-A-down", "e-B-up", "e-B-down"),
            "target_gene_id": ["g"] * 4,
            "modality": ["DNA"] * 4,
            "gate_key_id": ("k-A-up", "k-A-down", "k-B-up", "k-B-down"),
            "factor_identity_kind": ["unique"] * 4,
            "interaction_factor_id": ("A", "A", "B", "B"),
            "region_type": ("upstream", "downstream", "upstream", "downstream"),
        }
    )
    manifest = {
        "event_feature_manifest_identity": "supported-swap-fixture",
        "numeric_rank_audit": {"tolerance": 1.0e-10},
        "modalities": {
            "DNA": {
                "interaction_factor_vocabulary": ["A", "B"],
                "context_fields": {
                    "region_type": {
                        "raw_levels": ["upstream", "downstream"],
                        "scientific_context_pairs": [["upstream", "downstream"]],
                        "p_max": 1,
                    }
                },
                "padded_interaction_width": 1,
            },
            "RNA": {
                "interaction_factor_vocabulary": [],
                "context_fields": {},
                "padded_interaction_width": 0,
            },
        },
    }
    base = RouteBaseDesign(
        route_ids=tuple(context["route_id"]),
        values=np.asarray(
            [[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]],
            dtype=np.float32,
        ),
        column_names=(
            "DNA:factor=A",
            "DNA:factor=B",
            "DNA:region_type=downstream",
        ),
        manifest=manifest,
        route_context=context,
    )
    support = pd.DataFrame(
        [
            {
                "modality": "DNA",
                "context_field": "region_type",
                "factor_entity_id": factor,
                "context_level": position,
                "raw_cell_supported": True,
            }
            for factor in ("A", "B")
            for position in ("upstream", "downstream")
        ]
    )
    interaction = build_canonical_interaction_design(base, support)
    assert interaction.active_mask_by_modality["DNA"].tolist() == [True]
    comparator = interaction.raw_contrasts.loc[
        interaction.raw_contrasts["row_kind"].eq("comparator")
    ].iloc[0]
    assert comparator.raw_interaction_claim_status == "factor_specific_grammar_estimable"

    # A-down + B-up and A-up + B-down have the same additive factor/position
    # marginals, but opposite canonical rectangle coordinates.
    left_indices = np.asarray([1, 2])
    right_indices = np.asarray([0, 3])
    left_base = base.values[left_indices].sum(axis=0)
    right_base = base.values[right_indices].sum(axis=0)
    np.testing.assert_array_equal(left_base, right_base)
    left_interaction = interaction.values_by_modality["DNA"][left_indices].sum(
        axis=0
    )
    right_interaction = interaction.values_by_modality["DNA"][right_indices].sum(
        axis=0
    )
    assert not np.array_equal(left_interaction, right_interaction)

    def routed(indices: np.ndarray, *, interaction_on: bool) -> RoutedModalityInput:
        return RoutedModalityInput(
            route_event_index=torch.tensor([0, 1]),
            route_edge_index=torch.tensor([0, 0]),
            route_weight=torch.ones(2),
            route_base_features=torch.from_numpy(base.values[indices]),
            route_interaction_features=torch.from_numpy(
                interaction.values_by_modality["DNA"][indices]
            ),
            interaction_active_mask=torch.tensor([interaction_on]),
            event_gate_key_index=torch.tensor([0, 1]),
            gate=torch.ones(1, 2),
        )

    projector = RoutedEventAggregator(base_dim=3, interaction_dim=1, output_dim=1)
    with torch.no_grad():
        projector.base_projection.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        projector.interaction_projection.weight.fill_(1.0)
    left_off = projector(routed(left_indices, interaction_on=False), 1)
    right_off = projector(routed(right_indices, interaction_on=False), 1)
    torch.testing.assert_close(left_off, right_off, atol=0, rtol=0)
    left_on = projector(routed(left_indices, interaction_on=True), 1)
    right_on = projector(routed(right_indices, interaction_on=True), 1)
    assert not torch.equal(left_on, right_on)
    # With W_base=0 and the fixed identity selector W_int=1, the admitted
    # rectangle direction alone still expresses the swap.
    with torch.no_grad():
        projector.base_projection.weight.zero_()
        projector.interaction_projection.weight.fill_(1.0)
    selected_left = projector(routed(left_indices, interaction_on=True), 1)
    selected_right = projector(routed(right_indices, interaction_on=True), 1)
    torch.testing.assert_close(
        selected_left.reshape(()), torch.from_numpy(left_interaction).reshape(())
    )
    torch.testing.assert_close(
        selected_right.reshape(()), torch.from_numpy(right_interaction).reshape(())
    )


def test_atac_support_metadata_cannot_change_mapping_gate_forward_or_counterfactual():
    model, gene = _toy_model(seed=47)
    cells = gene.cell_ids
    one_rows: list[dict[str, object]] = []
    copies_rows: list[dict[str, object]] = []
    one_counts: list[list[float]] = []
    copies_counts: list[list[float]] = []
    one_atac_ids: list[str] = []
    copies_atac_ids: list[str] = []
    for cell_index, cell_id in enumerate(cells):
        counts = [float(cell_index + 1), float(24 - cell_index)]
        one_id = f"one-{cell_index}"
        one_atac_ids.append(one_id)
        one_counts.append(counts)
        one_rows.append(
            {
                "cell_id": cell_id,
                "neighbor_atac_cell_id": one_id,
                "neighbor_weight": 1.0,
                "distance": 1.0,
                "rna_qc_pass": True,
                "atac_qc_pass": True,
                "pairing_valid": True,
                "neighborhood_consistency_status": "not_estimable",
            }
        )
        for copy_index, weight in enumerate((0.5, 0.3, 0.2)):
            copy_id = f"copy-{cell_index}-{copy_index}"
            copies_atac_ids.append(copy_id)
            copies_counts.append(counts)
            copies_rows.append(
                {
                    "cell_id": cell_id,
                    "neighbor_atac_cell_id": copy_id,
                    "neighbor_weight": weight,
                    "distance": 1.0,
                    "rna_qc_pass": True,
                    "atac_qc_pass": True,
                    "pairing_valid": True,
                    "neighborhood_consistency_status": "pass",
                }
            )
    one_neighbors = pd.DataFrame(one_rows)
    copies_neighbors = pd.DataFrame(copies_rows)
    one_audit = assess_atac_mapping(
        one_neighbors, target_cell_ids=cells, expected_k=3, maximum_distance=2.0
    )
    copies_audit = assess_atac_mapping(
        copies_neighbors, target_cell_ids=cells, expected_k=3, maximum_distance=2.0
    )
    assert one_audit["mapping_valid"].all()
    assert copies_audit["mapping_valid"].all()
    assert not np.allclose(one_audit["ess_atac"], copies_audit["ess_atac"])
    assert not np.allclose(one_audit["evenness_atac"], copies_audit["evenness_atac"])
    assert not np.allclose(one_audit["coverage_atac"], copies_audit["coverage_atac"])

    one = map_atac_accessibility(
        one_counts,
        atac_cell_ids=one_atac_ids,
        peak_ids=("peak", "other"),
        target_cell_ids=cells,
        neighbors=one_neighbors,
        mapping_audit=one_audit,
    )
    copies = map_atac_accessibility(
        copies_counts,
        atac_cell_ids=copies_atac_ids,
        peak_ids=("peak", "other"),
        target_cell_ids=cells,
        neighbors=copies_neighbors,
        mapping_audit=copies_audit,
    )
    np.testing.assert_array_equal(one.mapping_valid, copies.mapping_valid)
    np.testing.assert_allclose(
        one.accessibility.toarray(), copies.accessibility.toarray(), atol=1e-6
    )
    activity = ActivityContext(
        cell_ids=cells,
        activity_entity_ids=("factor",),
        values=np.linspace(0.5, 2.0, len(cells), dtype=np.float32)[:, None],
        observed=np.ones((len(cells), 1), dtype=bool),
        library_size=np.full(len(cells), 100.0),
    )
    gate_keys = pd.DataFrame(
        {
            "gate_key_id": ["dna-gate"],
            "target_gene_id": [gene.gene_id],
            "channel": ["DNA"],
            "activity_entity_id": ["factor"],
            "peak_id": ["peak"],
        }
    )
    raw_one = build_raw_gate_signals(gate_keys, activity=activity, atac=one)
    raw_copies = build_raw_gate_signals(gate_keys, activity=activity, atac=copies)
    np.testing.assert_allclose(raw_one.raw, raw_copies.raw, atol=1e-6)
    np.testing.assert_array_equal(raw_one.observed, raw_copies.observed)
    thresholds = {
        "DNA": {
            "minimum_valid_cells": 2,
            "minimum_effective_cells": 2,
            "minimum_informative_molecules": 2,
            "minimum_standard_deviation": 0,
        }
    }
    fit_one = fit_gate_admission(
        raw_one,
        gate_keys,
        train_mask=[True] * 6 + [False] * 6,
        informative_molecule_mass=np.ones_like(raw_one.raw),
        thresholds_by_channel=thresholds,
    )
    fit_copies = fit_gate_admission(
        raw_copies,
        gate_keys,
        train_mask=[True] * 6 + [False] * 6,
        informative_molecule_mass=np.ones_like(raw_copies.raw),
        thresholds_by_channel=thresholds,
    )
    pd.testing.assert_frame_equal(fit_one, fit_copies)
    gates_one = transform_gates(raw_one, fit_one)
    gates_copies = transform_gates(raw_copies, fit_copies)
    np.testing.assert_allclose(gates_one.gate, gates_copies.gate, atol=1e-6)

    dna_one = gene.model_input.dna.gate.clone()
    dna_copies = gene.model_input.dna.gate.clone()
    dna_one[:, 0] = torch.from_numpy(gates_one.gate[:, 0])
    dna_copies[:, 0] = torch.from_numpy(gates_copies.gate[:, 0])
    input_one = replace(
        gene.model_input, dna=replace(gene.model_input.dna, gate=dna_one)
    )
    input_copies = replace(
        gene.model_input, dna=replace(gene.model_input.dna, gate=dna_copies)
    )
    keep_second_event = torch.tensor([False, True])
    counter_one = replace(
        input_one, dna=input_one.dna.with_event_keep_mask(keep_second_event)
    )
    counter_copies = replace(
        input_copies, dna=input_copies.dna.with_event_keep_mask(keep_second_event)
    )
    with torch.no_grad():
        full_one = model(input_one, condition="full")
        full_copies = model(input_copies, condition="full")
        without_one = model(counter_one, condition="full")
        without_copies = model(counter_copies, condition="full")
    for field in (
        "path_logits",
        "dna_aggregate",
        "joint_projected",
        "normalized_tokens",
        "edge_states",
    ):
        torch.testing.assert_close(
            getattr(full_one, field), getattr(full_copies, field), atol=0, rtol=0
        )
    torch.testing.assert_close(
        full_one.path_logits - without_one.path_logits,
        full_copies.path_logits - without_copies.path_logits,
        atol=0,
        rtol=0,
    )


def test_inactive_catalog_event_is_retained_for_audit_but_absent_from_production_tensors():
    events, routes = _catalog()
    _, base, interaction = _manifest_and_design(events, routes)
    inactive_event_id = "event-provenance-0"
    inactive_route_id = "route-0"
    inactive = events["event_id"].eq(inactive_event_id)
    events.loc[inactive, "gate_key_active"] = False
    events.loc[inactive, "model_active"] = False
    events.at[int(np.flatnonzero(inactive)[0]), "admission_reasons"] = [
        "inactive_gate_key"
    ]
    assert inactive_event_id in set(events["event_id"])

    gate_ids = tuple(events["gate_key_id"].astype(str))
    gates = GateValues(
        cell_ids=("cell",),
        gate_key_ids=gate_ids,
        raw=np.ones((1, len(gate_ids)), dtype=np.float32),
        standardized_residual=np.ones((1, len(gate_ids)), dtype=np.float32),
        gate=np.ones((1, len(gate_ids)), dtype=np.float32),
        observed=np.ones((1, len(gate_ids)), dtype=bool),
        out_of_train_range=np.zeros((1, len(gate_ids)), dtype=bool),
        out_of_train_quantile_support=np.zeros((1, len(gate_ids)), dtype=bool),
    )
    tensors = build_production_modality_tensors(
        events,
        routes,
        base,
        interaction,
        gates,
        target_gene_id="gene-provenance-0",
        modality="DNA",
        ordered_edge_ids=("edge-0", "edge-2", "edge-4", "edge-6", "edge-open"),
    )
    assert inactive_event_id not in tensors.event_ids
    assert inactive_route_id not in tensors.route_ids
    assert inactive_route_id in base.route_ids
    assert set(tensors.event_ids) == {
        "event-provenance-2",
        "event-provenance-4",
        "event-provenance-6",
        "event-provenance-open",
    }


def test_primary_ablations_zero_only_the_declared_modality_block_before_graphgps():
    model, gene = _toy_model(seed=53)
    with torch.no_grad():
        outputs = {
            condition: model(gene.model_input, condition=condition)
            for condition in PRIMARY_ABLATIONS
        }
    cis_end = model.cis_dim
    dna_end = cis_end + model.dynamic_dim
    full = outputs["full"]
    for condition, output in outputs.items():
        torch.testing.assert_close(
            output.joint_input[..., :cis_end],
            full.joint_input[..., :cis_end],
            atol=0,
            rtol=0,
        )
        expected_dna = (
            full.full_dna_aggregate
            if condition in {"cis_dna", "full"}
            else torch.zeros_like(full.full_dna_aggregate)
        )
        expected_rna = (
            full.full_rna_aggregate
            if condition in {"cis_rna", "full"}
            else torch.zeros_like(full.full_rna_aggregate)
        )
        torch.testing.assert_close(output.dna_aggregate, expected_dna, atol=0, rtol=0)
        torch.testing.assert_close(output.rna_aggregate, expected_rna, atol=0, rtol=0)
        torch.testing.assert_close(
            output.joint_input[..., cis_end:dna_end], expected_dna, atol=0, rtol=0
        )
        torch.testing.assert_close(
            output.joint_input[..., dna_end:], expected_rna, atol=0, rtol=0
        )


def test_inference_only_permutation_preserves_checkpoint_and_frozen_gate_fit():
    model, gene = _toy_model(seed=59)
    cells = tuple(f"permutation-cell-{index:02d}" for index in range(20))
    metadata = pd.DataFrame(
        {
            "cell_id": cells,
            "stage": ["stage"] * 20,
            "developmental_system": ["system"] * 20,
            "donor": ["donor"] * 20,
        }
    )
    manifest = build_pairing_permutation_assignments(
        metadata,
        strata_fields=("stage", "developmental_system", "donor"),
        seed=20260725,
        repetitions=1,
        minimum_stratum_cells=20,
        null_kind="factor_activity",
    )
    evidence = pd.DataFrame(
        {
            "cell_id": cells,
            "gate_0_raw": np.arange(20, dtype=np.float32),
            "gate_1_raw": -np.arange(20, dtype=np.float32),
            "gate_0_observed": [index % 4 != 0 for index in range(20)],
            "gate_1_observed": [index % 5 != 0 for index in range(20)],
        }
    )
    evidence_before = evidence.copy(deep=True)
    permuted = apply_joint_cell_permutation(
        evidence,
        manifest,
        permutation_index=0,
        value_columns=(
            "gate_0_raw",
            "gate_1_raw",
            "gate_0_observed",
            "gate_1_observed",
        ),
    ).sort_values("cell_id", kind="mergesort")
    source = evidence.set_index("cell_id")
    for row in permuted.itertuples(index=False):
        expected = source.loc[row.source_cell_id]
        assert row.gate_0_raw == expected.gate_0_raw
        assert row.gate_1_raw == expected.gate_1_raw
        assert bool(row.gate_0_observed) == bool(expected.gate_0_observed)
        assert bool(row.gate_1_observed) == bool(expected.gate_1_observed)
    pd.testing.assert_frame_equal(evidence, evidence_before)

    frozen_fit = pd.DataFrame(
        {
            "gate_key_id": ["gate-0", "gate-1"],
            "train_mean": [9.5, -9.5],
            "train_standard_deviation": [5.766281297, 5.766281297],
            "train_raw_minimum": [0.0, -19.0],
            "train_raw_maximum": [19.0, 0.0],
            "train_lower_weighted_quantile": [0.0, -19.0],
            "train_upper_weighted_quantile": [19.0, 0.0],
            "gate_key_active": [True, True],
        }
    )
    frozen_fit_before = frozen_fit.copy(deep=True)
    raw = GateRawSignals(
        cell_ids=tuple(permuted["cell_id"].astype(str)),
        gate_key_ids=("gate-0", "gate-1"),
        raw=permuted[["gate_0_raw", "gate_1_raw"]].to_numpy(np.float32),
        observed=permuted[
            ["gate_0_observed", "gate_1_observed"]
        ].to_numpy(bool),
    )
    permuted_gates = transform_gates(raw, frozen_fit)
    checkpoint_before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    expanded_input = replace(
        gene.model_input,
        dna=replace(
            gene.model_input.dna,
            gate=torch.from_numpy(permuted_gates.gate),
        ),
        rna=replace(
            gene.model_input.rna,
            gate=gene.model_input.rna.gate[:1].expand(20, -1).clone(),
        ),
    )
    with torch.no_grad():
        output = model(expanded_input, condition="full")
    assert torch.isfinite(output.path_logits).all()
    pd.testing.assert_frame_equal(frozen_fit, frozen_fit_before)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint_before[name], atol=0, rtol=0)
    torch.testing.assert_close(
        expanded_input.dna.event_gate_key_index,
        gene.model_input.dna.event_gate_key_index,
        atol=0,
        rtol=0,
    )


def test_counterfactual_attribution_is_deterministic_for_fixed_model_and_inputs():
    model, gene = _toy_model(seed=61)
    selector = EvidenceSelectorResolution(
        selector_id="dna-event-0",
        selector_kind="event",
        route_ids=("dna-route-0", "dna-route-1"),
        event_ids=("dna-event-0",),
        route_count=2,
        complete_model_injection_group_ids=("injection-singleton",),
        partial_model_injection_group_ids=(),
        model_injection_scope="singleton_supported",
    )
    arguments = {
        "gene_id": gene.gene_id,
        "cell_ids": gene.cell_ids,
        "path_ids": gene.path_ids,
        "dna_route_ids": tuple(f"dna-route-{index}" for index in range(4)),
        "rna_route_ids": tuple(f"rna-route-{index}" for index in range(4)),
        "dna_event_ids": ("dna-event-0", "dna-event-1"),
        "rna_event_ids": ("rna-event-0", "rna-event-1"),
        "dna_gate_observed": torch.ones_like(gene.model_input.dna.gate, dtype=torch.bool),
        "rna_gate_observed": torch.ones_like(gene.model_input.rna.gate, dtype=torch.bool),
        "condition": "full",
    }
    first = run_evidence_counterfactual(model, gene.model_input, selector, **arguments)
    second = run_evidence_counterfactual(model, gene.model_input, selector, **arguments)
    pd.testing.assert_frame_equal(first.path_records, second.path_records, check_exact=True)
    pd.testing.assert_frame_equal(first.group_records, second.group_records, check_exact=True)
    pd.testing.assert_frame_equal(
        first.alternative_records, second.alternative_records, check_exact=True
    )
    pd.testing.assert_frame_equal(
        first.intermediate_audit, second.intermediate_audit, check_exact=True
    )
    for name in first.named_intermediate_deltas:
        torch.testing.assert_close(
            first.named_intermediate_deltas[name],
            second.named_intermediate_deltas[name],
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        first.full_output.path_logits, second.full_output.path_logits, atol=0, rtol=0
    )
    torch.testing.assert_close(
        first.full_output.path_logits.softmax(dim=-1),
        second.full_output.path_logits.softmax(dim=-1),
        atol=0,
        rtol=0,
    )
