from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from fabric.dataset import (
    GateValues,
    RouteBaseDesign,
    build_canonical_interaction_design,
    build_event_feature_manifest,
    build_model_injection_equivalence_index,
    build_production_modality_tensors,
    encode_base_route_features,
    measure_raw_interaction_support,
)
from fabric.motifs import EVENT_ROUTE_COLUMNS, PHYSICAL_EVENT_COLUMNS


def _catalog():
    events: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    specifications = [
        ("A", "promoter", "UPSTREAM", -2.0, 1.0),
        ("A", "intragenic", "DOWNSTREAM", 3.0, 2.0),
        ("A", "promoter", "DOWNSTREAM", 7.0, 4.0),
        ("A", "intragenic", "UPSTREAM", -8.0, 3.0),
        ("B", "promoter", "DOWNSTREAM", 2.0, 5.0),
        ("B", "intragenic", "UPSTREAM", -3.0, 7.0),
        ("B", "promoter", "UPSTREAM", -7.0, 8.0),
        ("B", "intragenic", "DOWNSTREAM", 8.0, 6.0),
    ]
    for index, (factor, region, side, distance, peak_support) in enumerate(specifications):
        event = {column: None for column in PHYSICAL_EVENT_COLUMNS}
        event.update(
            {
                "event_id": f"e{index}",
                "target_gene_id": f"g{index % 2}",
                "factor_entity_id": factor,
                "factor_identity_kind": "unique",
                "cap_evidence_class": "motif_anchored",
                "candidate_factor_ids": [factor],
                "activity_entity_id": factor,
                "activity_gene_ids": [factor],
                "activity_proxy_rule": "unique_gene_cp10k_log1p",
                "modality": "DNA",
                "motif_id": f"M{factor}",
                "motif_equivalence_family_id": f"family:{factor}",
                "source_motif_ids": [f"M{factor}"],
                "chromosome": "chr1",
                "start": 100 + index * 10,
                "end": 104 + index * 10,
                "strand": "+",
                "source_hit_coordinates": [{"source_hit_id": f"h{index}"}],
                "motif_score": 0.5,
                "orientation": "same_transcript",
                "peak_id": f"p{index}",
                "peak_support": peak_support,
                "gate_key_id": f"k{index}",
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
                "route_id": f"r{index}",
                "event_id": f"e{index}",
                "target_gene_id": f"g{index % 2}",
                "modality": "DNA",
                "anchor_region_id": f"anchor:{index}",
                "anchor_site_id": f"site:{index}",
                "edge_id": f"edge:{index}",
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
    # OPEN_ONLY is a complete base identity but never an interaction factor.
    open_event = {**events[0]}
    open_event.update(
        event_id="open",
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
        peak_id="p-open",
        gate_key_id="k-open",
    )
    events.append(open_event)
    open_route = {**routes[0]}
    open_route.update(route_id="r-open", event_id="open", edge_id="edge:open")
    routes.append(open_route)
    return pd.DataFrame(events), pd.DataFrame(routes)


def _design():
    events, routes = _catalog()
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
    gates = GateValues(
        cell_ids=("c0", "c1"),
        gate_key_ids=tuple(f"k{index}" for index in range(8)) + ("k-open",),
        raw=np.ones((2, 9)),
        standardized_residual=np.ones((2, 9)),
        gate=np.ones((2, 9)),
        observed=np.ones((2, 9), dtype=bool),
        out_of_train_range=np.zeros((2, 9), dtype=bool),
        out_of_train_quantile_support=np.zeros((2, 9), dtype=bool),
    )
    support = measure_raw_interaction_support(
        base,
        events,
        gates,
        train_mask=[True, True],
        informative_molecule_mass_by_gene=pd.DataFrame(
            {
                "cell_id": ["c0", "c1", "c0", "c1"],
                "target_gene_id": ["g0", "g0", "g1", "g1"],
                "informative_molecule_mass": [1, 1, 1, 1],
            }
        ),
        thresholds_by_channel={
            "DNA": {
                "minimum_distinct_events": 1,
                "minimum_distinct_genes": 1,
                "minimum_distinct_gate_keys": 1,
                "minimum_informative_molecules": 1,
            },
            "RNA": {
                "minimum_distinct_events": 0,
                "minimum_distinct_genes": 0,
                "minimum_distinct_gate_keys": 0,
                "minimum_informative_molecules": 0,
            },
        },
    )
    interaction = build_canonical_interaction_design(base, support)
    return events, routes, base, gates, interaction


def test_canonical_basis_is_exact_padded_and_open_only_excluded():
    events, routes, base, _, interaction = _design()
    dna_manifest = interaction.manifest["modalities"]["DNA"]
    region = dna_manifest["fields"]["region_type"]
    assert region["N_raw_rectangles_potential"] == 1
    assert region["N_four_corner_supported"] == 1
    assert region["N_support_span"] == 1
    assert region["N_padded"] == 1
    assert dna_manifest["combined_rank_audit"]["final_rank"] == dna_manifest[
        "combined_rank_audit"
    ]["final_column_count"]
    open_row = list(base.route_ids).index("r-open")
    modality_row = list(interaction.route_indices_by_modality["DNA"]).index(open_row)
    assert np.all(interaction.values_by_modality["DNA"][modality_row] == 0)
    assert "OPEN_ONLY" in base.manifest["modalities"]["DNA"]["factor_vocabulary"]
    assert "OPEN_ONLY" not in base.manifest["modalities"]["DNA"][
        "interaction_factor_vocabulary"
    ]
    assert set(interaction.raw_contrasts.raw_interaction_claim_status).issubset(
        {
            "factor_specific_grammar_estimable",
            "cross_field_context_not_separable",
            "raw_contrast_not_in_active_span",
            "within_factor_only",
            "unsupported_focal_arms",
        }
    )


def test_production_is_modality_specific_and_gene_local_missing_levels_are_allowed():
    events, routes, base, gates, interaction = _design()
    # g0 does not instantiate the complete cohort-wide event vocabulary; that
    # is legal because rank admission was performed globally, not per gene.
    tensors = build_production_modality_tensors(
        events,
        routes,
        base,
        interaction,
        gates,
        target_gene_id="g0",
        modality="DNA",
        ordered_edge_ids=["edge:0", "edge:2", "edge:4", "edge:6", "edge:open"],
    )
    assert tensors.target_gene_id == "g0"
    assert tensors.modality == "DNA"
    assert tensors.route_base_features.shape[1] == sum(
        name.startswith("DNA:") for name in base.column_names
    )
    assert all(not name.startswith("RNA:") for name in base.column_names[: tensors.route_base_features.shape[1]])


def test_injection_signature_keeps_exact_micro_difference_and_ignores_negative_zero():
    events, routes, base, _, interaction = _design()
    # Only two events with identical gate and edge can be compared.  Give them
    # exact-zero vs -0.0 in a masked interaction position: same group.
    selected_events = events.iloc[[0, 4]].copy()
    selected_events["target_gene_id"] = "g"
    selected_events["gate_key_id"] = "shared"
    selected_events["start"] = [100, 200]
    selected_events["end"] = [104, 204]
    selected_events["motif_equivalence_family_id"] = ["fa", "fb"]
    selected_routes = routes.iloc[[0, 4]].copy()
    selected_routes["target_gene_id"] = "g"
    selected_routes["edge_id"] = "edge"
    selected_routes["anchor_region_id"] = ["a", "b"]
    indices = [list(base.route_ids).index(value) for value in selected_routes.route_id]
    local_base = type(base)(
        route_ids=tuple(selected_routes.route_id),
        values=base.values[indices].copy(),
        column_names=base.column_names,
        manifest=base.manifest,
        route_context=base.route_context.iloc[indices].reset_index(drop=True),
    )
    local_base.values[1] = local_base.values[0]
    dna_indices = list(interaction.route_indices_by_modality["DNA"])
    source_positions = [dna_indices.index(index) for index in indices]
    local_values = interaction.values_by_modality["DNA"][source_positions].copy()
    local_values[:] = 0.0
    local_values[1, 0] = -0.0
    local_interaction = type(interaction)(
        route_ids=local_base.route_ids,
        values_by_modality={"DNA": local_values, "RNA": np.zeros((0, 0), np.float32)},
        active_mask_by_modality={"DNA": interaction.active_mask_by_modality["DNA"], "RNA": np.zeros(0, bool)},
        route_indices_by_modality={"DNA": np.array([0, 1]), "RNA": np.zeros(0, np.int64)},
        raw_support=interaction.raw_support,
        manifest=interaction.manifest,
        raw_contrasts=interaction.raw_contrasts,
    )
    grouped = build_model_injection_equivalence_index(
        selected_events,
        selected_routes,
        local_base,
        local_interaction,
        ordered_edge_ids_by_gene={"g": ["edge"]},
    )
    assert grouped.member_count.tolist() == [2]
    # Any exact nonzero difference, however small, must split the group.
    changed = local_base.values.copy()
    changed[1, 0] += np.float32(1e-6)
    changed_base = type(local_base)(
        route_ids=local_base.route_ids,
        values=changed,
        column_names=local_base.column_names,
        manifest=local_base.manifest,
        route_context=local_base.route_context,
    )
    split = build_model_injection_equivalence_index(
        selected_events,
        selected_routes,
        changed_base,
        local_interaction,
        ordered_edge_ids_by_gene={"g": ["edge"]},
    )
    assert sorted(split.member_count) == [1, 1]


def test_route_geometry_infinity_is_rejected_not_converted_to_missing():
    events, routes = _catalog()
    manifest = build_event_feature_manifest(
        events,
        routes,
        distance_bin_boundaries={"DNA": [5], "RNA": [5]},
        scientific_context_pairs={"DNA": {}, "RNA": {}},
    )
    broken = routes.copy()
    broken.loc[0, "signed_distance_bp"] = np.inf
    with pytest.raises(ValueError, match="never infinite"):
        encode_base_route_features(events, broken, manifest)


def test_canonical_raw_span_and_claims_ignore_base_reference_recoding():
    _, _, base, _, interaction = _design()
    column = list(base.column_names).index("DNA:region_type=promoter")
    recoded_values = base.values.copy()
    dna_rows = base.route_context.modality.astype(str).to_numpy() == "DNA"
    recoded_values[dna_rows, column] = 1.0 - recoded_values[dna_rows, column]
    recoded_manifest = copy.deepcopy(base.manifest)
    recoded_manifest["modalities"]["DNA"]["base_categorical_fields"]["region_type"][
        "reference_level"
    ] = "promoter"
    recoded = type(base)(
        route_ids=base.route_ids,
        values=recoded_values,
        column_names=base.column_names,
        manifest=recoded_manifest,
        route_context=base.route_context,
    )
    alternative = build_canonical_interaction_design(recoded, interaction.raw_support)
    def rank(values):
        return 0 if values.shape[1] == 0 else np.linalg.matrix_rank(values)

    for field in interaction.manifest["modalities"]["DNA"]["fields"]:
        original_h = np.asarray(
            interaction.manifest["modalities"]["DNA"]["fields"][field]["H_active"],
            dtype=int,
        )
        alternative_h = np.asarray(
            alternative.manifest["modalities"]["DNA"]["fields"][field]["H_active"],
            dtype=int,
        )
        assert rank(original_h) == rank(alternative_h)
        assert rank(np.column_stack([original_h, alternative_h])) == rank(original_h)
    comparison_columns = [
        "raw_interaction_contrast_id",
        "row_kind",
        "comparator_id",
        "raw_interaction_claim_status",
        "contrast_in_active_span",
        "cross_field_context_separable",
    ]
    pd.testing.assert_frame_equal(
        interaction.raw_contrasts[comparison_columns].reset_index(drop=True),
        alternative.raw_contrasts[comparison_columns].reset_index(drop=True),
    )


def test_cross_field_exact_alias_downgrades_both_independent_of_manifest_order():
    context = pd.DataFrame(
        {
            "route_id": ["A0", "A1", "B0", "B1"],
            "event_id": ["eA0", "eA1", "eB0", "eB1"],
            "target_gene_id": ["g"] * 4,
            "modality": ["DNA"] * 4,
            "gate_key_id": ["kA0", "kA1", "kB0", "kB1"],
            "factor_identity_kind": ["unique"] * 4,
            "interaction_factor_id": ["A", "A", "B", "B"],
            "region_type": ["L0", "L1", "L0", "L1"],
            "anchor_type": ["L0", "L1", "L0", "L1"],
            "transcript_oriented_side": ["UPSTREAM"] * 4,
            "distance_bin": ["D"] * 4,
            "orientation": ["same_transcript"] * 4,
        }
    )
    fields = {
        "region_type": {
            "raw_levels": ["L0", "L1"],
            "scientific_context_pairs": [["L0", "L1"]],
            "p_max": 1,
        },
        "anchor_type": {
            "raw_levels": ["L0", "L1"],
            "scientific_context_pairs": [["L0", "L1"]],
            "p_max": 1,
        },
    }
    manifest = {
        "event_feature_manifest_identity": "alias-fixture",
        "numeric_rank_audit": {"tolerance": 1e-10},
        "modalities": {
            "DNA": {
                "interaction_factor_vocabulary": ["A", "B"],
                "context_fields": fields,
                "padded_interaction_width": 2,
            },
            "RNA": {
                "interaction_factor_vocabulary": [],
                "context_fields": {},
                "padded_interaction_width": 0,
            },
        },
    }
    base = RouteBaseDesign(
        route_ids=("A0", "A1", "B0", "B1"),
        values=np.array(
            [[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.float32
        ),
        column_names=("DNA:factor=A", "DNA:factor=B", "DNA:shared-level=L1"),
        manifest=manifest,
        route_context=context,
    )
    support = pd.DataFrame(
        [
            {
                "modality": "DNA",
                "context_field": field,
                "factor_entity_id": factor,
                "context_level": level,
                "raw_cell_supported": True,
            }
            for field in fields
            for factor in ("A", "B")
            for level in ("L0", "L1")
        ]
    )
    first = build_canonical_interaction_design(base, support)
    reversed_manifest = copy.deepcopy(manifest)
    reversed_manifest["modalities"]["DNA"]["context_fields"] = dict(
        reversed(list(fields.items()))
    )
    second = build_canonical_interaction_design(
        type(base)(
            route_ids=base.route_ids,
            values=base.values,
            column_names=base.column_names,
            manifest=reversed_manifest,
            route_context=base.route_context,
        ),
        support,
    )
    for result in (first, second):
        summaries = result.raw_contrasts.loc[
            result.raw_contrasts.row_kind == "q_summary"
        ]
        assert set(summaries.raw_interaction_claim_status) == {
            "cross_field_context_not_separable"
        }
    first_status = first.raw_contrasts.set_index("raw_interaction_contrast_id")[
        "raw_interaction_claim_status"
    ].to_dict()
    second_status = second.raw_contrasts.set_index("raw_interaction_contrast_id")[
        "raw_interaction_claim_status"
    ].to_dict()
    assert first_status == second_status
