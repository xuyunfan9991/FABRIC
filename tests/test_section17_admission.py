from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from fabric.dataset import (
    ActivityContext,
    ATACMappingContext,
    EMPTY_FATE,
    FULL_FATE,
    GateValues,
    INFORMATIVE_FATE,
    assess_atac_mapping,
    build_gate_keys,
    build_gate_collinearity_audit,
    build_long_read_compatibility_audit,
    build_raw_gate_signals,
)
from fabric.motifs import (
    build_candidate_routes,
    build_route_burden,
    build_route_degree_cap_audit,
    cap_and_finalize_routes,
    collapse_physical_events,
)
from fabric.model import PathContextReadout, RoutedEventAggregator, RoutedModalityInput
from fabric.evaluate import summarize_attribution_seeds, summarize_between_state_effects


def test_observed_zero_missing_mapping_failure_and_low_neighbor_support_remain_distinct():
    cells = ("measured-zero", "rna-missing", "mapped", "mapping-invalid")
    activity = ActivityContext(
        cell_ids=cells,
        activity_entity_ids=("F",),
        values=np.asarray([[0.0], [0.0], [1.5], [1.5]], dtype=np.float32),
        observed=np.asarray([[True], [False], [True], [True]]),
        library_size=np.asarray([10.0, 10.0, 10.0, 10.0]),
    )
    atac = ATACMappingContext(
        cell_ids=cells,
        peak_ids=("peak",),
        accessibility=sparse.csr_matrix([[2.0], [2.0], [2.0], [0.0]]),
        mapping_valid=np.asarray([True, True, True, False]),
        diagnostics=pd.DataFrame(),
    )
    gates = build_raw_gate_signals(
        pd.DataFrame(
            {
                "gate_key_id": ["rna", "open"],
                "channel": ["RNA", "Open"],
                "activity_entity_id": ["F", None],
                "peak_id": [None, "peak"],
            }
        ),
        activity=activity,
        atac=atac,
    )
    index = {key: position for position, key in enumerate(gates.gate_key_ids)}
    assert gates.raw[0, index["rna"]] == 0
    assert bool(gates.observed[0, index["rna"]])  # measured biological zero
    assert gates.raw[1, index["rna"]] == 0
    assert not bool(gates.observed[1, index["rna"]])  # missing RNA observation
    assert gates.raw[3, index["open"]] == 0
    assert not bool(gates.observed[3, index["open"]])  # invalid ATAC mapping

    one_neighbor = pd.DataFrame(
        {
            "cell_id": ["low-support"],
            "neighbor_atac_cell_id": ["a0"],
            "neighbor_weight": [1.0],
            "distance": [1.0],
            "rna_qc_pass": [True],
            "atac_qc_pass": [True],
            "pairing_valid": [True],
            "neighborhood_consistency_status": ["not_estimable"],
        }
    )
    low_support = assess_atac_mapping(
        one_neighbor,
        target_cell_ids=["low-support"],
        expected_k=3,
        maximum_distance=2.0,
    ).iloc[0]
    assert bool(low_support.mapping_valid)
    assert low_support.coverage_atac == 1 / 3
    assert low_support.mapping_failure_reasons == []

    distant = pd.DataFrame(
        {
            "cell_id": ["distant"] * 3,
            "neighbor_atac_cell_id": ["a0", "a1", "a2"],
            "neighbor_weight": [1 / 3] * 3,
            "distance": [9.0] * 3,
            "rna_qc_pass": [True] * 3,
            "atac_qc_pass": [True] * 3,
            "pairing_valid": [True] * 3,
            "neighborhood_consistency_status": ["pass"] * 3,
        }
    )
    high_ess_but_invalid = assess_atac_mapping(
        distant,
        target_cell_ids=["distant"],
        expected_k=3,
        maximum_distance=2.0,
    ).iloc[0]
    assert high_ess_but_invalid.ess_atac == 3.0
    assert not bool(high_ess_but_invalid.mapping_valid)
    assert high_ess_but_invalid.mapping_failure_reasons == [
        "distance_outside_admissible_range"
    ]


def test_gate_collinearity_distinguishes_all_frozen_pairwise_statuses():
    train_axis = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    gate = np.column_stack(
        [
            train_axis,
            -train_axis,
            train_axis + np.asarray([0.0, 0.05, 0.0, -0.05, 0.0]),
            np.asarray([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.full(5, 3.0),
            train_axis,
        ]
    )
    # A held-out outlier must not enter any train-fitted correlation.
    gate = np.row_stack([gate, np.asarray([999.0, 7.0, -500.0, 3.0, 8.0, 999.0])])
    observed = np.ones_like(gate, dtype=bool)
    observed[2:5, 5] = False
    values = GateValues(
        cell_ids=tuple(f"c{i}" for i in range(6)),
        gate_key_ids=("a", "b", "c", "d", "e", "f"),
        raw=gate.copy(),
        standardized_residual=gate.copy(),
        gate=gate,
        observed=observed,
        out_of_train_range=np.zeros_like(observed),
        out_of_train_quantile_support=np.zeros_like(observed),
    )
    keys = pd.DataFrame(
        {
            "gate_key_id": values.gate_key_ids,
            "target_gene_id": ["g"] * 6,
            "channel": ["DNA", "RNA", "DNA", "Open", "RNA", "DNA"],
            "gate_key_active": [True] * 6,
        }
    )
    masses = pd.DataFrame(
        {
            "cell_id": values.cell_ids,
            "target_gene_id": ["g"] * 6,
            "informative_molecule_mass": [1.0] * 6,
        }
    )
    audit = build_gate_collinearity_audit(
        values,
        keys,
        train_mask=[True] * 5 + [False],
        informative_molecule_mass_by_gene=masses,
        minimum_joint_effective_cells=3.0,
        absolute_correlation_threshold=0.99,
    )
    pairs = audit.pairs.set_index(["left_gate_key_id", "right_gate_key_id"])
    perfect_negative = pairs.loc[("a", "b")]
    assert perfect_negative.weighted_pearson_correlation == -1.0
    assert perfect_negative.collinearity_kind == "perfect_collinearity"
    assert perfect_negative.correlation_sign == "negative"
    assert perfect_negative.status == "correlated_evidence"

    near = pairs.loc[("a", "c")]
    assert 0.99 <= near.weighted_pearson_correlation < 1.0
    assert near.collinearity_kind == "near_collinearity"
    assert near.correlation_sign == "positive"

    below = pairs.loc[("a", "d")]
    assert below.collinearity_kind == "below_frozen_threshold"
    assert below.status == "no_high_pairwise_collinearity_detected"

    zero_variance = pairs.loc[("a", "e")]
    assert zero_variance.status == "evidence_separation_not_estimable"
    assert zero_variance.not_estimable_reason == "zero_common_variance"
    assert np.isnan(zero_variance.weighted_pearson_correlation)

    insufficient = pairs.loc[("a", "f")]
    assert insufficient.joint_valid_cell_count == 2
    assert insufficient.status == "evidence_separation_not_estimable"
    assert insufficient.not_estimable_reason == "joint_support_insufficient"
    assert np.isnan(insufficient.weighted_pearson_correlation)

    correlated_sets = audit.correlated_sets
    assert len(correlated_sets) == 1
    assert correlated_sets.iloc[0].member_gate_key_ids == ["a", "b", "c"]
    assert sorted(map(tuple, correlated_sets.iloc[0].pairwise_edges)) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def _motif_hit() -> dict[str, object]:
    return {
        "source_hit_id": "hit",
        "source_window_id": "window",
        "target_gene_id": "g",
        "factor_entity_id": "group:g:partner",
        "factor_identity_kind": "factor_equivalence_group",
        "candidate_factor_ids": ["g", "partner"],
        "activity_entity_id": "group:g:partner",
        "activity_gene_ids": ["g", "partner"],
        "activity_proxy_rule": "raw_sum_then_cp10k_log1p",
        "modality": "RNA",
        "motif_id": "M",
        "motif_equivalence_family_id": "family:M",
        "chromosome": "chr1",
        "start": 100,
        "end": 106,
        "strand": "+",
        "motif_score": 0.8,
        "calibrated_motif_quality": 0.9,
        "source_local_rank": 1,
        "source_priority": 0,
        "orientation": "transcribed",
        "peak_id": None,
        "peak_support": 0.0,
        "source_valid": True,
    }


def test_one_physical_hit_routes_to_multiple_overlapping_graph_windows_without_duplication():
    events = collapse_physical_events(pd.DataFrame([_motif_hit()]))
    assert len(events) == 1
    event = events.iloc[0]
    assert bool(event.is_self_factor)
    assert event.activity_gene_ids == ["g", "partner"]
    assert event.source_motif_ids == ["M"]
    assert (event.chromosome, event.start, event.end, event.strand) == (
        "chr1",
        100,
        106,
        "+",
    )
    anchors = pd.DataFrame(
        {
            "target_gene_id": ["g", "g"],
            "modality": ["RNA", "RNA"],
            "anchor_region_id": ["left-window", "right-window"],
            "anchor_site_id": ["left", "right"],
            "edge_id": ["edge-left", "edge-right"],
            "chromosome": ["chr1", "chr1"],
            "strand": ["+", "+"],
            "region_start": [95, 99],
            "region_end": [107, 112],
            "anchor_position": [99, 107],
            "region_type": ["splice_window", "splice_window"],
            "anchor_type": ["acceptor", "donor"],
            "geometry_kind": ["site_window", "site_window"],
        }
    )
    routes = build_candidate_routes(events, anchors)
    assert routes.event_id.nunique() == 1
    assert len(routes) == 2
    assert set(routes.anchor_region_id) == {"left-window", "right-window"}
    assert set(routes.edge_id) == {"edge-left", "edge-right"}


def test_unique_and_group_motif_events_compete_in_one_bucket_and_dropped_event_is_inactive():
    unique = {
        **_motif_hit(),
        "source_hit_id": "unique",
        "factor_entity_id": "F",
        "factor_identity_kind": "unique",
        "candidate_factor_ids": ["F"],
        "activity_entity_id": "F",
        "activity_gene_ids": ["F"],
        "activity_proxy_rule": "unique_gene_cp10k_log1p",
        "modality": "DNA",
        "motif_id": "MF",
        "motif_equivalence_family_id": "family:F",
        "orientation": "same_transcript",
        "peak_id": "peak",
        "peak_support": 1.0,
        "calibrated_motif_quality": 0.9,
    }
    group = {
        **unique,
        "source_hit_id": "group",
        "factor_entity_id": "group:G:H",
        "factor_identity_kind": "factor_equivalence_group",
        "candidate_factor_ids": ["G", "H"],
        "activity_entity_id": "group:G:H",
        "activity_gene_ids": ["G", "H"],
        "activity_proxy_rule": "raw_sum_then_cp10k_log1p",
        "motif_id": "MGH",
        "motif_equivalence_family_id": "family:GH",
        "calibrated_motif_quality": 0.8,
    }
    events = collapse_physical_events(pd.DataFrame([unique, group]))
    events, gate_keys = build_gate_keys(events)
    anchors = pd.DataFrame(
        {
            "target_gene_id": ["g"],
            "modality": ["DNA"],
            "anchor_region_id": ["promoter"],
            "anchor_site_id": ["TSS"],
            "edge_id": ["edge"],
            "chromosome": ["chr1"],
            "strand": ["+"],
            "region_start": [90],
            "region_end": [110],
            "anchor_position": [99],
            "region_type": ["promoter"],
            "anchor_type": ["TSS"],
            "geometry_kind": ["site_window"],
        }
    )
    candidates = build_candidate_routes(events, anchors)
    catalog = cap_and_finalize_routes(
        events,
        candidates,
        events_per_bucket_cap=1,
        gate_admission=gate_keys.assign(gate_key_active=True),
    )
    decisions = catalog.candidate_routes
    assert decisions.cap_bucket_id.nunique() == 1
    assert decisions.cap_selected.sum() == 1
    selected_event_id = decisions.loc[decisions.cap_selected, "event_id"].iloc[0]
    selected_event = catalog.physical_events.set_index("event_id").loc[selected_event_id]
    assert selected_event.factor_identity_kind == "unique"
    dropped_event_id = decisions.loc[~decisions.cap_selected, "event_id"].iloc[0]
    dropped_event = catalog.physical_events.set_index("event_id").loc[dropped_event_id]
    assert dropped_event.factor_identity_kind == "factor_equivalence_group"
    assert not bool(dropped_event.has_retained_route)
    assert not bool(dropped_event.model_active)
    assert dropped_event.admission_reasons == ["no_retained_route"]
    assert catalog.event_routes.groupby("event_id").route_weight.sum().eq(1.0).all()


def test_route_cap_audit_reproduces_anchor_mass_and_external_only_coupling():
    events = pd.DataFrame(
        {
            "event_id": ["event"],
            "target_gene_id": ["g"],
            "modality": ["DNA"],
            "gate_key_id": ["shared"],
        }
    )
    route_rows = []
    for anchor, count in (("focal", 2), ("four", 4), ("two", 2)):
        for index in range(count):
            route_rows.append(
                {
                    "route_id": f"{anchor}:{index}",
                    "event_id": "event",
                    "target_gene_id": "g",
                    "modality": "DNA",
                    "anchor_region_id": anchor,
                    "edge_id": f"edge:{anchor}:{index}",
                    "cap_bucket_id": f"bucket:{anchor}",
                    "cap_selected": anchor != "four",
                }
            )
    candidates = pd.DataFrame(route_rows)
    retained = candidates.loc[candidates.cap_selected].copy()
    retained["route_weight"] = 1.0 / len(retained)
    audit = build_route_degree_cap_audit(
        events,
        candidates,
        retained,
        audit_population="model_input",
    ).set_index("anchor_region_id")
    assert set(audit["D_pre"]) == {8}
    assert set(audit["D_post"]) == {4}
    assert set(audit["A_pre"]) == {3}
    assert set(audit["A_post"]) == {2}
    assert audit["m_pre"].sum() == 1.0
    assert audit["m_rawret"].sum() == 0.5
    assert audit["m_post"].sum() == 1.0
    np.testing.assert_allclose(
        audit["m_post"] - audit["m_pre"],
        audit["renorm_gain"] - audit["cap_loss"],
    )
    assert bool(audit.loc["focal", "external_only_coupling"])
    assert bool(audit.loc["two", "external_only_coupling"])
    assert not bool(audit.loc["four", "external_only_coupling"])
    assert audit.loc["four", "cap_loss"] == 0.5
    assert audit.loc["focal", "renormalization_factor"] == 2.0

    # B_gate is a gate-key burden, not an event count: two unit routes on the
    # same token give 2 for one shared key and sqrt(2) for distinct keys.
    burden_routes = pd.DataFrame(
        {
            "route_id": ["r0", "r1"],
            "event_id": ["e0", "e1"],
            "target_gene_id": ["g", "g"],
            "modality": ["DNA", "DNA"],
            "edge_id": ["token", "token"],
            "anchor_region_id": ["a", "a"],
            "route_weight": [1.0, 1.0],
        }
    )
    burden_candidates = burden_routes.assign(cap_bucket_id=["b0", "b1"])
    shared_events = pd.DataFrame(
        {"event_id": ["e0", "e1"], "gate_key_id": ["k", "k"]}
    )
    distinct_events = shared_events.assign(gate_key_id=["k0", "k1"])
    shared = build_route_burden(
        shared_events,
        burden_routes,
        burden_candidates,
        pd.DataFrame(),
        audit_population="model_input",
    ).iloc[0]
    distinct = build_route_burden(
        distinct_events,
        burden_routes,
        burden_candidates,
        pd.DataFrame(),
        audit_population="model_input",
    ).iloc[0]
    catalog = build_route_burden(
        shared_events,
        burden_routes,
        burden_candidates,
        pd.DataFrame(),
        audit_population="catalog",
    ).iloc[0]
    assert shared.B_gate == 2.0
    assert distinct.B_gate == math.sqrt(2.0)
    assert np.isnan(catalog.B_gate)
    assert shared.distinct_physical_event_count == distinct.distinct_physical_event_count == 2


def test_long_read_audit_uses_qc_pass_positive_mass_population_and_explicit_empty_fate():
    rows = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "target_gene_id": ["g"] * 4,
            "molecule_count": [5, 2, 3, 5],
            "pre_compatibility_qc_pass": [False, True, True, True],
            "final_fate": [
                "pre_compatibility_technical_qc_failure",
                EMPTY_FATE,
                INFORMATIVE_FATE,
                FULL_FATE,
            ],
            "technical_reason_code": ["low_mapq", "", "", ""],
        }
    )
    universe = pd.DataFrame(
        {
            "split": ["train", "val"],
            "target_gene_id": ["g", "g"],
        }
    )
    audit = build_long_read_compatibility_audit(
        rows,
        legal_paths_by_gene={"g": ("p0", "p1")},
        model_admitted_gene_ids=("g",),
        strata=("split", "target_gene_id"),
        stratum_universe=universe,
    )
    train = audit.query("split == 'train'").set_index("terminal_fate")
    assert set(train.index) == {EMPTY_FATE, INFORMATIVE_FATE, FULL_FATE}
    assert set(train.pre_compatibility_qc_pass_molecule_mass) == {10.0}
    assert set(train.captured_gene_assigned_molecule_mass) == {15.0}
    assert train.loc[EMPTY_FATE, "terminal_molecule_mass"] == 2.0
    assert train.loc[INFORMATIVE_FATE, "terminal_molecule_mass"] == 3.0
    assert train.loc[FULL_FATE, "terminal_molecule_mass"] == 5.0
    assert train.terminal_molecule_mass.sum() == 10.0
    assert train.terminal_fraction.sum() == 1.0
    assert "novel_isoform" not in audit.columns
    heldout_empty = audit.query("split == 'val'")
    assert heldout_empty.fraction_status.eq("not_estimable").all()
    assert heldout_empty.pre_compatibility_qc_pass_molecule_mass.eq(0).all()


def test_interaction_support_gate_suppresses_null_direction_and_recovers_supported_plant():
    gate = torch.linspace(-1.0, 1.0, 21)[:, None]

    def fit(seed: int, active: bool) -> tuple[float, float, torch.Tensor]:
        torch.manual_seed(seed)
        aggregator = RoutedEventAggregator(base_dim=1, interaction_dim=1, output_dim=1)
        with torch.no_grad():
            aggregator.base_projection.weight.zero_()
        before = float(aggregator.interaction_projection.weight.abs().item())
        routed = RoutedModalityInput(
            route_event_index=torch.tensor([0]),
            route_edge_index=torch.tensor([0]),
            route_weight=torch.tensor([1.0]),
            route_base_features=torch.zeros(1, 1),
            route_interaction_features=torch.ones(1, 1),
            interaction_active_mask=torch.tensor([active]),
            event_gate_key_index=torch.tensor([0]),
            gate=gate,
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [aggregator.interaction_projection.weight],
                    "weight_decay": 0.2,
                }
            ],
            lr=0.05,
        )
        target = gate[:, 0] if active else torch.zeros(len(gate))
        for _ in range(500):
            optimizer.zero_grad(set_to_none=True)
            prediction = aggregator(routed, edge_count=1)[:, 0, 0]
            torch.mean((prediction - target) ** 2).backward()
            optimizer.step()
        prediction = aggregator(routed, edge_count=1)[:, 0, 0].detach()
        after = float(aggregator.interaction_projection.weight.abs().item())
        return before, after, prediction

    recovered_slopes = []
    for seed in (11, 22, 33):
        _, _, supported_prediction = fit(seed, True)
        recovered_slopes.append(
            float(torch.dot(supported_prediction, gate[:, 0]) / torch.dot(gate[:, 0], gate[:, 0]))
        )
        before, after, null_prediction = fit(seed, False)
        assert after < 0.01 * before
        torch.testing.assert_close(null_prediction, torch.zeros_like(null_prediction), atol=0, rtol=0)
    np.testing.assert_allclose(recovered_slopes, 1.0, atol=0.08, rtol=0)


def test_path_scale_reference_preserves_local_swap_and_constitutive_padding():
    torch.manual_seed(41)
    readout = PathContextReadout(hidden_dim=4, path_hidden_dim=5)

    def residuals(incidence: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        path_count, edge_count = incidence.shape
        first = torch.zeros(path_count, dtype=torch.long)
        last = torch.full((path_count,), edge_count - 1, dtype=torch.long)
        edge_counts = incidence.sum(1)
        return readout(
            states,
            incidence.to_sparse_coo(),
            first,
            last,
            torch.log1p(edge_counts),
        ).path_residual

    # Two paths differing only by the focal internal edge: D_path=0.5, V=2.
    simple_incidence = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]]
    )
    simple_states = torch.randn(1, 4, 4)
    simple = residuals(simple_incidence, simple_states)
    torch.testing.assert_close(
        simple[:, 0] - simple[:, 1],
        simple_states[:, 1] - simple_states[:, 2],
        atol=2e-7,
        rtol=1e-6,
    )

    # Eight legal paths enumerate three independent swaps.  The first pair
    # keeps the latter two choices fixed and changes the same focal A/B edge.
    combinations = []
    for focal in (1, 2):
        for second in (3, 4):
            for third in (5, 6):
                row = torch.zeros(8)
                row[[0, focal, second, third, 7]] = 1.0
                combinations.append(row)
    complex_incidence = torch.stack(combinations)
    complex_states = torch.randn(1, 8, 4)
    # rows 0 and 4 differ only at the focal choice under this loop order.
    complex_residual = residuals(complex_incidence, complex_states)
    complex_difference = complex_residual[:, 0] - complex_residual[:, 4]
    torch.testing.assert_close(
        complex_difference,
        complex_states[:, 1] - complex_states[:, 2],
        atol=3e-7,
        rtol=1e-6,
    )
    path_frequency = complex_incidence.mean(0)
    d_path = float(torch.sum(path_frequency * (1.0 - path_frequency)))
    v_path = int(((path_frequency > 0) & (path_frequency < 1)).sum())
    assert d_path == 1.5
    assert v_path == 6

    # This is the contract's test-local candidate reference, not a production
    # switch: both members receive the same positive catalog scalar.
    candidate_scale = 1.0 / math.sqrt(max(1.0, d_path))
    scaled_difference = (
        candidate_scale * complex_residual[:, 0]
        - candidate_scale * complex_residual[:, 4]
    )
    torch.testing.assert_close(
        scaled_difference,
        candidate_scale * complex_difference,
        atol=1e-7,
        rtol=1e-7,
    )

    # A common constitutive token has zero centered-incidence coefficient and
    # changes neither unscaled nor reference-scaled pair difference or D_path.
    padded_incidence = torch.cat(
        [complex_incidence[:, :-1], torch.ones(8, 1), complex_incidence[:, -1:]],
        dim=1,
    )
    padded_states = torch.cat(
        [complex_states[:, :-1], torch.randn(1, 1, 4), complex_states[:, -1:]],
        dim=1,
    )
    padded = residuals(padded_incidence, padded_states)
    torch.testing.assert_close(
        padded[:, 0] - padded[:, 4], complex_difference, atol=3e-7, rtol=1e-6
    )
    padded_frequency = padded_incidence.mean(0)
    padded_d_path = float(torch.sum(padded_frequency * (1.0 - padded_frequency)))
    assert padded_d_path == d_path


def test_three_seed_attribution_stability_and_between_state_support_gates_are_distinct():
    records = pd.DataFrame(
        [
            {
                "record_id": record_id,
                "seed": seed,
                "effect": value,
                "interaction_support": support,
            }
            for record_id, support, values in (
                ("full-low", "full", (1.0, 1.1, 0.9)),
                ("full-high", "full", (3.0, 3.2, 2.8)),
                ("partial-low", "partial", (1.0, 5.0, 1.0)),
                ("partial-high", "partial", (4.0, 8.0, 4.0)),
            )
            for seed, value in zip((11, 22, 33), values, strict=True)
        ]
    )
    summary = summarize_attribution_seeds(
        records,
        seed_ids=(11, 22, 33),
        record_columns=("record_id",),
        value_column="effect",
        epsilon_num=1e-8,
        effect_floor=0.1,
        maximum_dispersion=0.5,
        interaction_support_column="interaction_support",
    )
    per_record = summary.records.set_index("record_id")
    assert per_record.loc["full-low", "direction_status"] == "stable_direction"
    assert per_record.loc["partial-low", "magnitude_status"] == "magnitude_seed_unstable"
    assert set(summary.effect_rank_correlations.interaction_support_stratum) == {
        "all",
        "full",
        "partial",
    }
    support_specific = summary.effect_rank_correlations.query(
        "interaction_support_stratum != 'all'"
    )
    np.testing.assert_allclose(support_specific.spearman_r, 1.0, atol=1e-12, rtol=0)
    assert summary.effect_rank_correlations.spearman_r.between(-1.0, 1.0).all()

    state_rows = pd.DataFrame(
        [
            {
                "selector": "S",
                "seed": seed,
                "cell_id": cell_id,
                "reporting_state": state,
                "effect": value,
                "eligible": True,
            }
            for seed in (11, 22, 33)
            for state, cells, value in (
                ("supported", ("a", "b", "c"), 2.0),
                ("underpowered", ("x", "y"), 100.0),
            )
            for cell_id in cells
        ]
    )
    state_summary = summarize_between_state_effects(
        state_rows,
        seed_ids=(11, 22, 33),
        state_pairs=(("supported", "underpowered"),),
        record_columns=("selector",),
        value_column="effect",
        minimum_state_cells=3,
        epsilon_num=1e-8,
        effect_floor=0.1,
        maximum_dispersion=1.0,
    ).iloc[0]
    assert state_summary.status == "state_contrast_not_estimable"
    assert state_summary.state_a_cell_count == 3
    assert state_summary.state_b_cell_count == 2
    assert state_summary.per_seed_contrasts is None
