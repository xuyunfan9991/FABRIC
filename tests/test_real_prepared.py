from __future__ import annotations

import numpy as np
from fabric.dataset import build_event_feature_manifest
from fabric.motifs import EVENT_ROUTE_COLUMNS, PHYSICAL_EVENT_COLUMNS
from fabric.real_prepared import (
    DISTANCE_BIN_BOUNDARIES,
    _ExactRectangleBasis,
    _encode_candidate_base,
    _joined_route_context,
    _select_full_rank_base,
    _select_gf2_independent_columns,
    _supported_bipartite_cycle_space_dimension,
    _base_candidate_specs,
)


def _catalog():
    events = []
    routes = []
    for index, (factor, region, distance) in enumerate(
        (("A", "promoter", -2.0), ("A", "intragenic", 7.0),
         ("B", "promoter", -7.0), ("B", "intragenic", 2.0))
    ):
        event = {column: None for column in PHYSICAL_EVENT_COLUMNS}
        event.update(
            event_id=f"e{index}", target_gene_id="g", factor_entity_id=factor,
            factor_identity_kind="unique", cap_evidence_class="motif_anchored",
            candidate_factor_ids=[factor], activity_entity_id=factor,
            activity_gene_ids=[factor], activity_proxy_rule="unique_gene_cp10k_log1p",
            modality="DNA", motif_id=f"M{factor}",
            motif_equivalence_family_id=f"M{factor}", source_motif_ids=[f"M{factor}"],
            chromosome="chr1", start=100 + index * 10, end=104 + index * 10,
            strand="+", source_hit_coordinates=[{"source_hit_id": f"h{index}"}],
            motif_score=0.5, calibrated_motif_quality=np.nan,
            orientation="same_transcript", peak_id=f"p{index}", peak_support=1.0,
            gate_key_id=f"k{index}", is_self_factor=False, source_valid=True,
            has_retained_route=True, gate_key_active=True, model_active=True,
            admission_reasons=[], source_priority=0, source_local_rank=1.0,
        )
        route = {column: None for column in EVENT_ROUTE_COLUMNS}
        route.update(
            route_id=f"r{index}", event_id=f"e{index}", target_gene_id="g",
            modality="DNA", anchor_region_id=f"a{index}", anchor_site_id=f"s{index}",
            edge_id=f"edge{index}", route_weight=1.0, region_type=region,
            anchor_type="TSS", transcript_oriented_side=(
                "UPSTREAM" if distance < 0 else "DOWNSTREAM"
            ), signed_distance_bp=distance, edge_relative_position=np.nan,
            distance_to_5prime_boundary_bp=np.nan,
            distance_to_3prime_boundary_bp=np.nan, geometry_kind="site_window",
        )
        events.append(event)
        routes.append(route)
    import pandas as pd
    return pd.DataFrame(events), pd.DataFrame(routes)


def test_exact_rectangle_basis_rejects_dependent_cycle():
    basis = _ExactRectangleBasis()
    assert basis.add_if_independent({0: 1, 1: -1, 2: -1, 3: 1})
    assert basis.add_if_independent({0: 1, 1: -1, 4: -1, 5: 1})
    # The third rectangle is the difference of the first two.
    assert not basis.add_if_independent({2: 1, 3: -1, 4: -1, 5: 1})


def test_supported_cycle_bound_is_exact_but_does_not_invent_rectangles():
    factors = ("A", "B", "C")
    levels = ("0", "1", "2")
    full_support = {(factor, level): True for factor in factors for level in levels}
    assert _supported_bipartite_cycle_space_dimension(
        factors, levels, full_support
    ) == 4

    # A chordless six-cycle has graph cycle dimension one but no supported
    # four-corner rectangle.  The bound can terminate exact elimination once
    # reached, but it is never substituted for the measured rectangle span.
    six_cycle = {
        ("A", "0"): True,
        ("A", "1"): True,
        ("B", "1"): True,
        ("B", "2"): True,
        ("C", "2"): True,
        ("C", "0"): True,
    }
    assert _supported_bipartite_cycle_space_dimension(
        factors, levels, six_cycle
    ) == 1


def test_combined_gf2_closure_keeps_base_and_canonical_independent_columns():
    # Rows span columns 0, 1 and 3; candidate column 2 duplicates the required
    # base column 0 and must close before candidate column 3 is retained.
    retained, rank = _select_gf2_independent_columns(
        (0b0101, 0b0010, 0b1000),
        width=4,
        required_prefix_width=2,
    )
    assert retained == (0, 1, 3)
    assert rank == 3


def test_streamed_base_rank_closure_keeps_complete_active_factor_baseline():
    physical, routes = _catalog()
    manifest = build_event_feature_manifest(
        physical,
        routes,
        distance_bin_boundaries=DISTANCE_BIN_BOUNDARIES,
        scientific_context_pairs={"DNA": {}, "RNA": {}},
        motif_score_in_model=False,
        orientation_interaction_policy={"DNA": False, "RNA": False},
        numeric_rank_tolerance=1.0e-8,
    )
    context = _joined_route_context(physical, routes, manifest, active_only=False)
    specs = _base_candidate_specs(manifest, "DNA")
    matrix = _encode_candidate_base(context, manifest, "DNA", specs)
    selected, audit = _select_full_rank_base(
        (matrix.T @ matrix).toarray(), specs, tolerance=1.0e-8
    )
    retained = {specs[index].name for index in selected}
    present_factors = set(context.loc[context["modality"].eq("DNA"), "interaction_factor_id"])
    assert {f"DNA:factor={value}" for value in present_factors}.issubset(retained)
    selected_matrix = matrix[:, selected].toarray()
    assert np.linalg.matrix_rank(selected_matrix) == selected_matrix.shape[1]
    assert audit["full_column_rank"] is True
