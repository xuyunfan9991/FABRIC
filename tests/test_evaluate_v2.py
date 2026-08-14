from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from types import SimpleNamespace

from fabric.choices import PathIdentifiabilityIndex, build_path_identifiability_index
from fabric.dataset import (
    ATACMappingContext,
    build_raw_gate_signals,
    compute_activity_entities,
    fit_gate_admission,
)
from fabric.evaluate import (
    OntEpochMonitor,
    ValidationMonitorBundle,
    aggregate_pairing_null,
    apply_joint_cell_permutation,
    build_train_support_bin_assignments,
    build_ont_matrix_scope,
    build_pairing_permutation_assignments,
    build_event_density_token_audit,
    build_path_scale_audit,
    classify_model_injection_coverage,
    compute_compatible_score_residuals,
    compute_compatible_set_diagnostics,
    compute_ont_matrix_agreement,
    compute_validation_monitor_record,
    compare_architecture_readouts,
    evaluate_perturbation_response_curve,
    evaluate_state_residual_gate,
    explanation_manifest_coverage,
    fit_state_residual_diagnostics,
    gauge_invariant_counterfactual_outputs,
    neutralize_routed_terms,
    rebuild_member_count_perturbation,
    rebuild_observed_library_context,
    rebuild_source_proxy_perturbation,
    run_evidence_counterfactual,
    resolve_evidence_selector,
    summarize_attribution_seeds,
    summarize_between_state_effects,
    summarize_event_density_strata,
    summarize_path_scale_strata,
    summarize_predictive_seeds,
    summarize_support_stratified_sensitivity,
    validate_ont_matrix_identity,
    validate_training_run_manifest,
)
from fabric.train import build_paired_models, load_config, make_toy_genes


def _event_tables():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "factor_entity_id": ["F", "F", "G"],
            "gate_key_id": ["g1", "g2", "g3"],
            "modality": ["DNA", "DNA", "RNA"],
        }
    )
    routes = pd.DataFrame(
        {
            "route_id": ["r1", "r2", "r3", "r4"],
            "event_id": ["e1", "e1", "e2", "e3"],
            "anchor_region_id": ["a", "b", "a", "a"],
        }
    )
    injection = pd.DataFrame(
        {
            "model_injection_group_id": ["inj12", "inj3"],
            "member_event_ids": [["e1", "e2"], ["e3"]],
            "member_count": [2, 1],
            "member_route_ids": [["r1", "r2", "r3"], ["r4"]],
        }
    )
    correlated = pd.DataFrame(
        {
            "correlated_evidence_set_id": ["corr"],
            "member_gate_key_ids": [["g1", "g3"]],
        }
    )
    return events, routes, injection, correlated


def test_missing_model_injection_index_fails_closed_for_primary_attribution():
    complete, partial, scope = classify_model_injection_coverage(("r1",), None)
    assert complete == ()
    assert partial == ()
    assert scope == "not_evaluated_missing_index"


def test_primitive_and_derived_selectors_resolve_exact_route_unions():
    events, routes, injection, correlated = _event_tables()
    event = resolve_evidence_selector(
        "event", "e1", events, routes, model_injection_index=injection
    )
    assert event.route_ids == ("r1", "r2")
    assert event.model_injection_scope == "partial_model_injection_group"

    factor = resolve_evidence_selector(
        "factor", "F", events, routes, model_injection_index=injection
    )
    assert factor.route_ids == ("r1", "r2", "r3")
    assert factor.model_injection_scope == "set_supported"

    anchor = resolve_evidence_selector(
        "anchor_region", "a", events, routes, model_injection_index=injection
    )
    assert anchor.route_ids == ("r1", "r3", "r4")
    assert anchor.partial_model_injection_group_ids == ("inj12",)

    correlated_set = resolve_evidence_selector(
        "correlated_evidence_set",
        "corr",
        events,
        routes,
        model_injection_index=injection,
        correlated_gate_sets=correlated,
    )
    # It is the gate-key union, not the complete factor selector for F.
    assert correlated_set.route_ids == ("r1", "r2", "r4")
    terms = torch.arange(8.0).reshape(4, 2)
    neutralized = neutralize_routed_terms(terms, routes["route_id"], event)
    torch.testing.assert_close(neutralized[:2], torch.zeros(2, 2))
    torch.testing.assert_close(neutralized[2:], terms[2:])

    # A multi-member injection class is attributed only by forwarding its full
    # route union.  Here its two physical members have exchangeable aggregate
    # tensors, but the joint set removes two copies rather than one chosen
    # representative.
    injection_set = resolve_evidence_selector(
        "model_injection_group",
        "inj12",
        events,
        routes,
        model_injection_index=injection,
    )
    assert injection_set.route_ids == ("r1", "r2", "r3")
    assert injection_set.model_injection_scope == "set_supported"
    route_terms = torch.tensor([[0.5], [0.5], [1.0], [4.0]])
    event_one = neutralize_routed_terms(route_terms, routes["route_id"], event)
    event_two_selector = resolve_evidence_selector(
        "event", "e2", events, routes, model_injection_index=injection
    )
    event_two = neutralize_routed_terms(
        route_terms, routes["route_id"], event_two_selector
    )
    joint = neutralize_routed_terms(
        route_terms, routes["route_id"], injection_set
    )
    assert route_terms.sum() - event_one.sum() == 1.0
    assert route_terms.sum() - event_two.sum() == 1.0
    assert route_terms.sum() - joint.sum() == 2.0
    assert event.model_injection_scope == "partial_model_injection_group"
    assert event_two_selector.model_injection_scope == "partial_model_injection_group"
    broken_index = injection.copy()
    broken_index.at[0, "member_route_ids"] = ["r1", "r2"]
    with pytest.raises(ValueError, match="absent from model injection index"):
        resolve_evidence_selector(
            "factor", "F", events, routes, model_injection_index=broken_index
        )


def test_counterfactual_executor_forwards_exact_singleton_joint_and_missing_context():
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    model = build_paired_models(
        gene, config["model"], seed=17, device="cpu"
    )["full"]
    events = pd.DataFrame(
        {
            "event_id": ["d0", "d1", "r0", "r1"],
            "factor_entity_id": ["F", "F", "R", "R"],
            "gate_key_id": ["gd0", "gd1", "gr0", "gr1"],
        }
    )
    routes = pd.DataFrame(
        {
            "route_id": ["dr0", "dr1", "dr2", "dr3", "rr0", "rr1", "rr2", "rr3"],
            "event_id": ["d0", "d0", "d1", "d1", "r0", "r0", "r1", "r1"],
            "anchor_region_id": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    injection = pd.DataFrame(
        {
            "model_injection_group_id": ["dna_pair", "rna0", "rna1"],
            "member_event_ids": [["d0", "d1"], ["r0"], ["r1"]],
            "member_count": [2, 1, 1],
            "member_route_ids": [
                ["dr0", "dr1", "dr2", "dr3"],
                ["rr0", "rr1"],
                ["rr2", "rr3"],
            ],
        }
    )
    singleton0 = resolve_evidence_selector(
        "event", "d0", events, routes, model_injection_index=injection
    )
    singleton1 = resolve_evidence_selector(
        "event", "d1", events, routes, model_injection_index=injection
    )
    joint = resolve_evidence_selector(
        "model_injection_group",
        "dna_pair",
        events,
        routes,
        model_injection_index=injection,
    )
    assert singleton0.model_injection_scope == "partial_model_injection_group"
    assert joint.model_injection_scope == "set_supported"
    identifiability = PathIdentifiabilityIndex(
        genes=pd.DataFrame(
            {
                "gene_id": [gene.gene_id],
                "path_ids": [list(gene.path_ids)],
                "path_group_indices": [[0, 1]],
                "group_count": [2],
            }
        ),
        paths=pd.DataFrame(),
        groups=pd.DataFrame(
            {
                "gene_id": [gene.gene_id, gene.gene_id],
                "observational_group_index": [0, 1],
                "observational_group_id": ["E0", "E1"],
                "member_path_ids": [[gene.path_ids[0]], [gene.path_ids[1]]],
            }
        ),
        train_patterns=pd.DataFrame(),
    )
    contrasts = pd.DataFrame(
        {
            "contrast_id": ["toy:a:b|matched"],
            "gene_id": [gene.gene_id],
            "choice_id": ["toy-choice"],
            "contrast_kind": ["matched_context"],
            "context_signature": ["shared"],
            "numerator_path_ids": [[gene.path_ids[0]]],
            "denominator_path_ids": [[gene.path_ids[1]]],
            "cohort_reportable": [True],
        }
    )
    common = {
        "gene_id": gene.gene_id,
        "cell_ids": gene.cell_ids,
        "path_ids": gene.path_ids,
        "dna_route_ids": ("dr0", "dr1", "dr2", "dr3"),
        "rna_route_ids": ("rr0", "rr1", "rr2", "rr3"),
        "dna_event_ids": ("d0", "d1"),
        "rna_event_ids": ("r0", "r1"),
        "dna_gate_observed": torch.ones_like(gene.model_input.dna.gate, dtype=torch.bool),
        "rna_gate_observed": torch.ones_like(gene.model_input.rna.gate, dtype=torch.bool),
        "path_identifiability_index": identifiability,
        "alternative_contrasts": contrasts,
    }
    result0 = run_evidence_counterfactual(
        model, gene.model_input, singleton0, **common
    )
    result1 = run_evidence_counterfactual(
        model, gene.model_input, singleton1, **common
    )
    result_joint = run_evidence_counterfactual(
        model, gene.model_input, joint, **common
    )
    assert set(result_joint.named_intermediate_deltas) == {
        "a_DNA",
        "a_RNA",
        "y",
        "y_hat",
        "H",
        "path_logits",
    }
    assert result_joint.intermediate_audit["all_named_intermediates_finite"].all()
    assert result_joint.intermediate_audit["route_weights_renormalized"].eq(False).all()
    assert result_joint.path_records.groupby("cell_id").size().eq(2).all()
    assert result_joint.group_records.groupby("cell_id").size().eq(2).all()
    assert result_joint.alternative_records["delta_relative_log_mass"].notna().all()
    assert result_joint.intermediate_audit[
        "primary_mechanism_summary_eligible"
    ].all()
    assert not result0.intermediate_audit[
        "primary_mechanism_summary_eligible"
    ].any()
    # Joint neutralization is actually re-forwarded; it is not defined as an
    # additive sum of exchangeable member effects.
    assert not torch.allclose(
        result_joint.named_intermediate_deltas["path_logits"],
        result0.named_intermediate_deltas["path_logits"]
        + result1.named_intermediate_deltas["path_logits"],
        atol=1e-8,
        rtol=1e-8,
    )

    missing_common = dict(common)
    missing_observed = common["dna_gate_observed"].clone()
    missing_observed[:, 0] = False
    missing_common["dna_gate_observed"] = missing_observed
    missing = run_evidence_counterfactual(
        model, gene.model_input, singleton0, **missing_common
    )
    assert missing.intermediate_audit["attribution_status"].eq(
        "missing_context_not_estimable"
    ).all()
    assert missing.path_records["delta_log_path_probability"].isna().all()
    assert missing.alternative_records["delta_relative_log_mass"].isna().all()


def test_gauge_invariant_outputs_preserve_group_sums_and_use_equal_path_centering():
    full = torch.tensor([[10.0, 9.0, -2.0]])
    counter = torch.tensor([[3.0, 5.0, -4.0]])
    outputs = gauge_invariant_counterfactual_outputs(
        full, counter, observational_group_indices=[[0, 1], [2]]
    )
    torch.testing.assert_close(
        outputs["centered_delta_path_logit"].sum(-1), torch.zeros(1)
    )
    full_shifted = gauge_invariant_counterfactual_outputs(
        full + 1000.0, counter - 500.0, observational_group_indices=[[0, 1], [2]]
    )
    torch.testing.assert_close(
        outputs["delta_log_path_probability"],
        full_shifted["delta_log_path_probability"],
    )
    torch.testing.assert_close(
        outputs["delta_group_probability"],
        full_shifted["delta_group_probability"],
    )


def _ont_inputs():
    rows = pd.DataFrame(
        {
            "matrix_row_id": ["m0", "m1", "m2"],
            "transcript_id": ["tx0", "tx1", "tx2"],
            "gene_id": ["g", "g", "g"],
        }
    )
    cells = pd.DataFrame({"cell_id": ["c0", "c1", "c2", "c3"]})
    crosswalk = pd.DataFrame(
        {
            "matrix_row_id": ["m0", "m1", "m2"],
            "transcript_id": ["tx0", "tx1", "tx2"],
            "gene_id": ["g", "g", "g"],
            "path_id": ["p0", "p1", "p2"],
        }
    )
    paths = pd.DataFrame(
        {
            "gene_id": ["g", "g", "g"],
            "path_id": ["p0", "p1", "p2"],
            "transcript_aliases": [["tx0"], ["tx1"], ["tx2"]],
        }
    )
    candidates = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "gene_id": ["g"] * 4,
            "split": ["val"] * 4,
        }
    )
    counts = pd.DataFrame(
        {
            "cell_id": ["c0", "c0", "c2", "c3", "c3"],
            "matrix_row_id": ["m0", "m1", "m0", "m0", "m1"],
            "count": [5, 5, 7, 9, 1],
        }
    )
    return rows, cells, crosswalk, paths, candidates, counts


def _logits(third: float):
    return pd.DataFrame(
        [
            {"cell_id": cell, "gene_id": "g", "path_id": path, "logit": value}
            for cell, values in {
                "c0": (1000.0, -1000.0, third),
                "c3": (3.0, 1.0, third),
            }.items()
            for path, value in zip(("p0", "p1", "p2"), values, strict=True)
        ]
    )


def test_ont_identity_scope_ties_and_extreme_matrix_logits_are_stable():
    rows, cells, crosswalk, paths, candidates, counts = _ont_inputs()
    identity = validate_ont_matrix_identity(rows, cells, crosswalk, paths)
    scope = build_ont_matrix_scope(identity, candidates, counts)
    status = scope.candidates.set_index("cell_id")["scope_status"].to_dict()
    assert status == {
        "c0": "eligible",
        "c1": "ont_count_total_zero",
        "c2": "fewer_than_two_positive_matrix_transcripts",
        "c3": "eligible",
    }
    conservation = scope.conservation.iloc[0]
    assert bool(conservation["cell_gene_conservation_pass"])
    assert bool(conservation["raw_count_conservation_pass"])

    agreement = compute_ont_matrix_agreement(identity, scope, _logits(third=0.0))
    assert agreement.numerical_status == "valid"
    tie = agreement.records.set_index("cell_id").loc["c0"]
    assert bool(tie["observed_top_tie"])
    assert tie["ont_top1_tie_aware_hit"] == 1.0
    assert np.isnan(tie["ont_unique_top1_hit"])
    # p1 is far below the PRISM clamp, while scientific fields remain log-space.
    assert tie["ont_matrix_cross_entropy"] > 900
    assert tie["ont_cross_entropy_prism_clamped"] < 20
    assert agreement.summary.iloc[0]["prism_compatibility_status"] == (
        "PRISM_CLAMPED_COMPATIBILITY_ONLY"
    )

    reordered_identity = validate_ont_matrix_identity(
        rows.iloc[::-1], cells.iloc[::-1], crosswalk.iloc[::-1], paths.iloc[::-1]
    )
    reordered_scope = build_ont_matrix_scope(
        reordered_identity, candidates.iloc[::-1], counts.iloc[::-1]
    )
    reordered = compute_ont_matrix_agreement(
        reordered_identity, reordered_scope, _logits(third=0.0).iloc[::-1]
    )
    pd.testing.assert_frame_equal(
        agreement.summary.sort_index(axis=1), reordered.summary.sort_index(axis=1)
    )

    duplicated = pd.concat([candidates, candidates.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="coordinates are duplicated"):
        build_ont_matrix_scope(identity, duplicated, counts)
    conflicting = pd.concat(
        [
            candidates,
            candidates.iloc[[0]].assign(split="test"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="conflicting splits"):
        build_ont_matrix_scope(identity, conflicting, counts)


def test_every_matrix_isoform_is_in_probability_denominator_and_identity_fails_closed():
    rows, cells, crosswalk, paths, candidates, counts = _ont_inputs()
    identity = validate_ont_matrix_identity(rows, cells, crosswalk, paths)
    scope = build_ont_matrix_scope(identity, candidates, counts)
    baseline = compute_ont_matrix_agreement(identity, scope, _logits(third=0.0)).records.set_index("cell_id")
    shifted = compute_ont_matrix_agreement(identity, scope, _logits(third=20.0)).records.set_index("cell_id")
    assert shifted.loc["c3", "ont_matrix_cross_entropy"] > baseline.loc[
        "c3", "ont_matrix_cross_entropy"
    ]
    assert shifted.loc["c3", "ont_matrix_kl"] > baseline.loc[
        "c3", "ont_matrix_kl"
    ]

    collision = crosswalk.copy()
    collision.loc[1, "path_id"] = "p0"
    with pytest.raises(ValueError, match="two ONT matrix transcripts"):
        validate_ont_matrix_identity(rows, cells, collision, paths)
    drift = paths.copy()
    drift.at[0, "transcript_aliases"] = ["wrong_tx"]
    with pytest.raises(ValueError, match="absent from model transcript_aliases"):
        validate_ont_matrix_identity(rows, cells, crosswalk, drift)
    alias_collision = paths.copy()
    alias_collision.at[2, "transcript_aliases"] = ["tx0"]
    with pytest.raises(ValueError, match="multiple structural paths"):
        validate_ont_matrix_identity(rows, cells, crosswalk, alias_collision)

    extra_model_path = pd.concat(
        [
            paths,
            pd.DataFrame(
                {
                    "gene_id": ["g"],
                    "path_id": ["p_outside_matrix"],
                    "transcript_aliases": [["tx_outside_matrix"]],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="absent from the ONT matrix isoform axis"):
        validate_ont_matrix_identity(rows, cells, crosswalk, extra_model_path)


def test_compatible_diagnostics_reject_duplicate_paths_and_invalid_mass():
    logits = pd.DataFrame(
        {
            "cell_id": ["c", "c"],
            "gene_id": ["g", "g"],
            "path_id": ["p0", "p1"],
            "logit": [0.0, 1.0],
        }
    )
    duplicated = pd.DataFrame(
        {
            "cell_id": ["c"],
            "gene_id": ["g"],
            "compatible_path_ids": [["p0", "p0"]],
            "molecule_count": [1],
        }
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        compute_compatible_set_diagnostics(logits, duplicated)
    invalid = duplicated.copy()
    invalid.at[0, "compatible_path_ids"] = ["p0"]
    invalid.loc[0, "molecule_count"] = -1
    with pytest.raises(ValueError, match="strictly positive"):
        compute_compatible_set_diagnostics(logits, invalid)
    zero = invalid.assign(molecule_count=0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_compatible_set_diagnostics(logits, zero)
    full_set = invalid.assign(
        compatible_path_ids=[["p0", "p1"]], molecule_count=1
    )
    with pytest.raises(ValueError, match="proper subsets"):
        compute_compatible_set_diagnostics(logits, full_set)


def test_pairing_permutation_moves_joint_rows_and_aggregates_after_each_seed():
    cells = [f"c{i:02d}" for i in range(20)]
    metadata = pd.DataFrame(
        {
            "cell_id": cells,
            "stage": ["S"] * 20,
            "developmental_system": ["D"] * 20,
            "donor": ["X"] * 20,
        }
    )
    manifest = build_pairing_permutation_assignments(
        metadata,
        strata_fields=("stage", "developmental_system", "donor"),
        seed=11,
    )
    assert len(manifest.assignments) == 20 * 100
    values = pd.DataFrame(
        {
            "cell_id": cells,
            "activity_a": np.arange(20),
            "activity_b": np.arange(20) + 100,
            "mask": np.arange(20) % 2,
        }
    )
    permuted = apply_joint_cell_permutation(
        values,
        manifest,
        permutation_index=0,
        value_columns=("activity_a", "activity_b", "mask"),
    )
    assert ((permuted["activity_b"] - permuted["activity_a"]) == 100).all()
    assert (permuted["mask"] == permuted["activity_a"] % 2).all()

    stats = pd.DataFrame(
        [
            {
                "seed": seed,
                "null_kind": "coarse",
                "permutation_index": b,
                "nll_permuted": 2.0 if b < 96 else 0.5,
                "nll_paired": 1.0,
                "t_attr_permuted": 0.2,
                "t_attr_paired": 1.0,
            }
            for seed in (1, 2, 3)
            for b in range(100)
        ]
    )
    aggregated, summary = aggregate_pairing_null(stats, seed_ids=(1, 2, 3))
    assert len(aggregated) == 100
    assert summary.iloc[0]["positive_T_NLL_count"] == 96
    assert bool(summary.iloc[0]["claim_admission_pass"])


def test_score_residual_validation_only_ridge_and_state_gate(toy_gene_graph):
    train = pd.DataFrame(
        {
            "cell_id": ["t0", "t1"],
            "gene_id": [toy_gene_graph.gene_id] * 2,
            "compatible_path_ids": [["p0"], ["p1"]],
            "molecule_count": [1, 1],
            "split": ["train", "train"],
        }
    )
    index = build_path_identifiability_index(toy_gene_graph, train)
    probabilities = pd.DataFrame(
        {
            "cell_id": ["v0", "v0"],
            "gene_id": [toy_gene_graph.gene_id] * 2,
            "path_id": ["p0", "p1"],
            "probability": [0.7, 0.3],
        }
    )
    ec = pd.DataFrame(
        {
            "cell_id": ["v0", "v0"],
            "gene_id": [toy_gene_graph.gene_id] * 2,
            "compatible_path_ids": [["p0"], ["p1"]],
            "molecule_count": [3, 1],
        }
    )
    residual = compute_compatible_score_residuals(probabilities, ec, index)
    p0_group = index.paths.set_index("path_id").loc["p0", "observational_group_id"]
    observed = residual.set_index("observational_group_id").loc[p0_group, "score_residual"]
    assert observed == pytest.approx(0.05)

    n = 30
    diagnostic_rows = pd.DataFrame(
        {
            "score_residual": np.tile([-1.0, 1.0], n // 2),
            "informative_molecule_mass": np.ones(n),
            "gene_group_id": ["g|E"] * n,
            "positive_ec_row_count": np.ones(n),
            "donor": ["d"] * n,
            "stage": np.tile(["early", "late"], n // 2),
            "developmental_system": ["sys"] * n,
            "cell_type": np.tile(["a", "b"], n // 2),
            "state_pc0": np.tile([-1.0, 1.0], n // 2),
            "high_dtu": [True] * n,
        }
    )
    diagnostics = fit_state_residual_diagnostics(
        diagnostic_rows,
        state_pc_columns=("state_pc0",),
        alpha_grid=(0.01, 1.0),
        cv_folds=3,
        seed=7,
    )
    gate = evaluate_state_residual_gate(diagnostics, diagnostic_rows)
    assert (gate["delta_R2_state"] > 0.05).all()
    assert not gate["cell_state_mechanism_claim_allowed"].any()
    unseen = diagnostic_rows.copy()
    unseen.loc[0, "donor"] = "unseen_donor"
    with pytest.raises(ValueError, match="outside frozen vocabulary"):
        evaluate_state_residual_gate(diagnostics, unseen)
    constant = diagnostic_rows.copy()
    constant["score_residual"] = 0.5
    not_estimable = evaluate_state_residual_gate(diagnostics, constant)
    assert not_estimable["status"].eq("not_estimable").all()
    with pytest.raises(ValueError, match="frozen at 0.05"):
        evaluate_state_residual_gate(diagnostics, diagnostic_rows, threshold=0.1)


def test_seed_training_between_state_and_manifest_summaries_use_fixed_denominators():
    training = pd.DataFrame(
        [
            {"seed": 11, "condition": "full"},
            {"seed": 22, "condition": "atac"},
            {"seed": 33, "condition": "rbp"},
        ]
    )
    assert validate_training_run_manifest(training) == (11, 22, 33)

    attribution = pd.DataFrame(
        [
            {
                "record_id": record,
                "seed": seed,
                "delta_rho": value,
                "model_injection_scope": scope,
                "support": "full",
            }
            for record, scope, values in (
                ("stable", "set_supported", (1.0, 1.1, 0.9)),
                ("unstable", "singleton_supported", (1.0, -1.0, 0.0)),
                ("partial", "partial_model_injection_group", (2.0, 2.0, 2.0)),
            )
            for seed, value in zip((11, 22, 33), values, strict=True)
        ]
    )
    summary = summarize_attribution_seeds(
        attribution,
        seed_ids=(11, 22, 33),
        record_columns=("record_id",),
        value_column="delta_rho",
        epsilon_num=1e-8,
        effect_floor=0.1,
        maximum_dispersion=1.0,
        interaction_support_column="support",
    ).records.set_index("record_id")
    assert summary.loc["stable", "direction_status"] == "stable_direction"
    assert summary.loc["unstable", "direction_status"] == "seed_unstable"
    assert not bool(summary.loc["partial", "primary_summary_eligible"])

    per_cell = pd.DataFrame(
        [
            {
                "selector_id": "S",
                "seed": seed,
                "cell_id": f"{state}{cell}",
                "reporting_state": state,
                "effect": (2.0 if state == "A" else 0.5) + seed_offset,
                "eligible": True,
            }
            for seed, seed_offset in zip((11, 22, 33), (0.0, 0.1, -0.1), strict=True)
            for state in ("A", "B")
            for cell in range(3)
        ]
    )
    between = summarize_between_state_effects(
        per_cell,
        seed_ids=(11, 22, 33),
        state_pairs=(("A", "B"),),
        record_columns=("selector_id",),
        value_column="effect",
        minimum_state_cells=3,
        epsilon_num=1e-8,
        effect_floor=0.2,
        maximum_dispersion=1.0,
    )
    assert between.iloc[0]["status"] == "stable_state_difference"

    predictive = pd.DataFrame(
        {"seed": [11, 22, 33], "condition": ["Full"] * 3, "nll": [1.0, 1.2, 0.8]}
    )
    repeat = summarize_predictive_seeds(
        predictive, group_columns=("condition",), metric_columns=("nll",)
    )
    assert repeat.iloc[0]["sample_sd"] == pytest.approx(0.2)
    assert "not_biological" in repeat.iloc[0]["uncertainty_semantics"]

    eligible = pd.DataFrame(
        {"cell_id": ["a", "b"], "gene_id": ["g", "g"], "split": ["test", "test"]}
    )
    manifest = eligible.iloc[[0]][["cell_id", "gene_id"]]
    coverage = explanation_manifest_coverage(
        eligible,
        manifest,
        key_columns=("cell_id", "gene_id"),
        selection_rule="split-neutral case/control list",
    )
    assert coverage.query("scope == 'all'").iloc[0]["selection_coverage"] == 0.5
    with pytest.raises(ValueError, match="held-out"):
        explanation_manifest_coverage(
            eligible,
            manifest.assign(delta_rho=9.0),
            key_columns=("cell_id", "gene_id"),
            selection_rule="invalid",
        )


def test_epoch_monitor_is_once_per_completed_epoch_sealed_and_gradient_neutral():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.ones(2, 2)
    loss = model(x).sum()
    loss.backward()
    before = [parameter.grad.clone() for parameter in model.parameters()]
    monitor = OntEpochMonitor(
        lambda fixed: {
            "validation_compatible_path_nll": float(fixed.weight.square().sum()),
            "ont_matrix_kl_count_weighted": float(fixed.weight.sum().abs()),
        }
    )
    monitor.record_completed_epoch(1, model)
    for parameter, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter.grad, expected)
    with pytest.raises(ValueError, match="exactly once"):
        monitor.record_completed_epoch(1, model)
    with pytest.raises(RuntimeError, match="post-selection reporting"):
        monitor.read_for_post_selection_reporting(
            selection_and_reporting_rules_frozen=False
        )
    report = monitor.read_for_post_selection_reporting(
        selection_and_reporting_rules_frozen=True
    )
    assert report["completed_epoch"].tolist() == [1]
    optimizer.step()


def test_validation_monitor_bundle_uses_one_complete_val_snapshot_and_json_safe_fields():
    rows, cells, crosswalk, paths, candidates, counts = _ont_inputs()
    identity = validate_ont_matrix_identity(rows, cells, crosswalk, paths)
    scope = build_ont_matrix_scope(identity, candidates, counts)
    prediction = SimpleNamespace(
        gene_id="g",
        cell_ids=("c0", "c3"),
        path_ids=("p0", "p1", "p2"),
        path_logits=torch.tensor([[3.0, 2.0, 0.0], [2.0, 1.0, -1.0]]),
        compatible_path_indices=torch.tensor([[0, -1], [0, 1]]),
        compatible_path_mask=torch.tensor([[True, False], [True, True]]),
        row_cell_index=torch.tensor([0, 1]),
        molecule_count=torch.tensor([2.0, 3.0]),
    )
    snapshot = SimpleNamespace(
        split="val",
        nll=0.75,
        informative_molecule_mass=5.0,
        predictions=(prediction,),
    )
    ec = pd.DataFrame(
        {
            "cell_id": ["c0", "c3"],
            "gene_id": ["g", "g"],
            "split": ["val", "val"],
            "compatible_path_ids": [["p0"], ["p0", "p1"]],
            "molecule_count": [2.0, 3.0],
        }
    )
    bundle = ValidationMonitorBundle(
        ont_identity=identity,
        ont_scope=scope,
        compatible_validation_ec_rows=ec,
        matrix_identity="matrix-v1",
        crosswalk_identity="crosswalk-v1",
        path_identity="paths-v1",
        split_identity="split-v1",
        observation_process_status="PASS_CROSS_PIPELINE",
    )
    fields = compute_validation_monitor_record(snapshot, bundle)
    for key in (
        "metric_schema",
        "matrix_identity",
        "crosswalk_identity",
        "path_identity",
        "split_identity",
        "observation_process_status",
        "model_output_dtype",
        "aggregation_dtype",
        "numerical_tolerance",
        "sealed",
        "selection_eligible",
        "validation_compatible_path_nll",
        "ont_matrix_kl_count_weighted",
        "ont_eligible_count_denominator",
    ):
        assert key in fields
    assert fields["sealed"] is True
    assert fields["selection_eligible"] is False
    assert fields["same_validation_prediction_traversal"] is True
    assert fields["ont_matrix_kl_count_weighted"] >= 0
    bad_snapshot = SimpleNamespace(**{**snapshot.__dict__, "split": "test"})
    with pytest.raises(ValueError, match="validation snapshots only"):
        compute_validation_monitor_record(bad_snapshot, bundle)


def _perturbation_inputs():
    cells = ("c0", "c1")
    genes = ("f1", "f2", "other")
    raw_counts = sparse.csr_matrix([[10, 20, 70], [20, 10, 70]])
    entities = pd.DataFrame(
        {
            "activity_entity_id": ["F1", "FG"],
            "activity_gene_ids": [["f1"], ["f1", "f2"]],
            "source_valid": [True, True],
        }
    )
    activity = compute_activity_entities(
        raw_counts,
        cell_ids=cells,
        frozen_gene_axis=genes,
        entity_table=entities,
    )
    atac = ATACMappingContext(
        cell_ids=cells,
        peak_ids=("peak", "other_peak"),
        accessibility=sparse.csr_matrix([[2.0, 1.0], [3.0, 1.0]]),
        mapping_valid=np.asarray([True, True]),
        diagnostics=pd.DataFrame(
            {"cell_id": cells, "mapping_valid": [True, True]}
        ),
    )
    gate_keys = pd.DataFrame(
        {
            "gate_key_id": ["dna_f1", "dna_fg", "open", "rna_fg"],
            "target_gene_id": ["target"] * 4,
            "channel": ["DNA", "DNA", "Open", "RNA"],
            "activity_entity_id": ["F1", "FG", None, "FG"],
            "peak_id": ["peak", "peak", "peak", None],
        }
    )
    raw_signals = build_raw_gate_signals(gate_keys, activity=activity, atac=atac)
    gate_admission = fit_gate_admission(
        raw_signals,
        gate_keys,
        train_mask=[True, True],
        informative_molecule_mass=np.ones((2, 4)),
        thresholds_by_channel={
            channel: {
                "minimum_valid_cells": 1,
                "minimum_effective_cells": 1,
                "minimum_informative_molecules": 1,
                "minimum_standard_deviation": 0,
            }
            for channel in ("DNA", "RNA", "Open")
        },
        support_quantiles=(0.0, 1.0),
    )
    events = pd.DataFrame(
        {
            "event_id": ["e_dna_f1", "e_dna_fg", "e_open", "e_rna_fg"],
            "gate_key_id": gate_keys["gate_key_id"],
        }
    )
    return raw_counts, genes, entities, activity, atac, gate_keys, gate_admission, events


def test_source_member_and_observed_library_perturbations_rebuild_the_right_layers():
    (
        raw_counts,
        genes,
        entities,
        activity,
        atac,
        gate_keys,
        gate_admission,
        events,
    ) = _perturbation_inputs()
    # This is an in-support source proxy point; the DNA gates sharing the peak
    # or entity are rebuilt together, never a selected final gate in isolation.
    source = rebuild_source_proxy_perturbation(
        cell_id="c0",
        source_kind="mapped_accessibility",
        source_id="peak",
        source_value=3.0,
        activity=activity,
        atac=atac,
        gate_keys=gate_keys,
        gate_admission=gate_admission,
        physical_events=events,
    )
    assert source.affected_gate_key_ids == ("dna_f1", "dna_fg", "open")
    assert source.affected_event_ids == ("e_dna_f1", "e_dna_fg", "e_open")
    assert source.support_status == "supported_model_counterfactual"
    assert set(source.gate_audit["dynamic_context_status"]) == {"observed"}

    member = rebuild_member_count_perturbation(
        cell_id="c0",
        member_gene_id="f1",
        member_count=20,
        raw_rna_counts=raw_counts,
        frozen_gene_axis=genes,
        entity_table=entities,
        activity=activity,
        atac=atac,
        gate_keys=gate_keys,
        gate_admission=gate_admission,
        physical_events=events,
    )
    assert member.affected_gate_key_ids == ("dna_f1", "dna_fg", "rna_fg")
    assert set(member.gate_audit["fixed_library_denominator"]) == {100.0}
    expected_unique = np.log1p(10_000 * 20 / 100)
    expected_group = np.log1p(10_000 * (20 + 20) / 100)
    np.testing.assert_allclose(member.activity.values[0], [expected_unique, expected_group])

    # A newly measured library instead normalizes with its own complete 110 count
    # denominator, and remains explicitly distinct from the fixed-denominator
    # in-silico member intervention.
    observed_counts = sparse.csr_matrix([[20, 20, 70], [20, 10, 70]])
    observed = rebuild_observed_library_context(
        cell_id="c0",
        raw_rna_counts=observed_counts,
        frozen_gene_axis=genes,
        entity_table=entities,
        rna_observation_valid=[True, True],
        atac=atac,
        gate_keys=gate_keys,
        gate_admission=gate_admission,
        physical_events=events,
    )
    assert observed.perturbation_kind == "observed_library_context"
    assert set(observed.gate_audit["fixed_library_denominator"]) == {110.0}
    assert observed.activity.values[0, 0] != pytest.approx(member.activity.values[0, 0])

    extrapolated = rebuild_source_proxy_perturbation(
        cell_id="c0",
        source_kind="mapped_accessibility",
        source_id="peak",
        source_value=30.0,
        activity=activity,
        atac=atac,
        gate_keys=gate_keys,
        gate_admission=gate_admission,
        physical_events=events,
    )
    assert extrapolated.support_status == "model_extrapolation"
    assert not extrapolated.gate_audit["primary_supported_claim_allowed"].any()

    points = [source, extrapolated]
    expected_outputs = pd.DataFrame(
        {
            "output_kind": [
                "matched_context_relative_log_mass",
                "path_probability",
                "group_probability",
            ],
            "output_id": ["choice:a:b|context", "p0", "E0"],
        }
    )

    def predictor(point):
        gate = float(point.gate_audit["final_gate_G"].sum())
        probability = float(torch.sigmoid(torch.tensor(gate)).item())
        return expected_outputs.assign(value=[gate, probability, probability])

    curve = evaluate_perturbation_response_curve(
        points, predictor=predictor, expected_outputs=expected_outputs
    )
    assert curve.gate_records.groupby("response_point_index").size().tolist() == [3, 3]
    assert curve.response_records.groupby("response_point_index").size().tolist() == [3, 3]
    assert not curve.response_records.loc[
        curve.response_records["support_status"].eq("model_extrapolation"),
        "primary_supported_claim_allowed",
    ].any()
    assert curve.response_records["claim_semantics"].nunique() == 1


def test_architecture_path_scale_and_event_density_tables_keep_exact_units_and_axes():
    seeds = (11, 22, 33)
    conditions = ("full", "full_additive_edge")
    ec_rows = []
    for seed in seeds:
        for condition in conditions:
            for index, (cell, gene, length, internal) in enumerate(
                (("c0", "g0", "short", "internal"), ("c1", "g1", "long", "TSS"))
            ):
                full_numerator = (1.0 + 0.1 * index) * 10
                ec_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "parameter_count": (
                            100 if condition == "full" else 80
                        ),
                        "cell_id": cell,
                        "gene_id": gene,
                        "ec_id": f"ec{index}",
                        "path_axis_identity": f"paths-{gene}",
                        "compatible_path_ids": ["p0"],
                        "molecule_count": 10.0,
                        "nll_numerator": full_numerator
                        + (2.0 if condition == "full_additive_edge" else 0.0),
                        "calibration_error": 0.1
                        + (0.02 if condition == "full_additive_edge" else 0.0),
                        "path_length_stratum": length,
                        "choice_kind": internal,
                    }
                )
    architecture = compare_architecture_readouts(
        pd.DataFrame(ec_rows),
        epsilon_num=1.0e-8,
        strata_columns=("path_length_stratum", "choice_kind"),
    )
    assert architecture.claim_summary.iloc[0][
        "consistent_empirical_predictive_gain_allowed"
    ]
    np.testing.assert_allclose(
        architecture.per_seed.query("stratum_axis == 'overall'")[
            "delta_nll_arch"
        ],
        0.2,
    )

    prediction = pd.DataFrame(
        [
            {
                "seed": seed,
                "condition": condition,
                "cell_id": cell,
                "gene_id": gene,
                "compatible_nll_numerator": 5.0 + index,
                "informative_molecule_mass": 5.0,
                "compatible_calibration_error": 0.1 + 0.01 * index,
                "dynamic_block_norm": 2.0 + index,
                "pre_normalization_token_norm": 3.0 + index,
                "post_normalization_token_norm": 1.0,
            }
            for seed in seeds
            for condition in ("full",)
            for index, (cell, gene) in enumerate((("c0", "g0"), ("c1", "g1")))
        ]
    )
    path_audit = prediction[["seed", "condition", "cell_id", "gene_id"]].copy()
    path_audit["D_g_path_stratum"] = np.where(
        path_audit["gene_id"].eq("g0"), "low", "high"
    )
    path_audit["V_g_stratum"] = np.where(
        path_audit["gene_id"].eq("g0"), "few", "many"
    )
    path_audit["legal_path_count_stratum"] = np.where(
        path_audit["gene_id"].eq("g0"), "3", "6-10"
    )
    for column, value in {
        "zeta_norm_median": 1.0,
        "zeta_norm_q95": 1.5,
        "zeta_norm_max": 2.0,
        "relative_token_rms": 0.5,
        "path_mlp_preactivation_norm": 1.0,
        "path_logit_sd": 0.2,
        "path_logit_range": 0.4,
        "softmax_entropy": 0.7,
        "local_contrast_gradient_norm": 0.3,
        "prediction_seed_stability": 0.9,
        "attribution_seed_stability": 0.8,
    }.items():
        path_audit[column] = value
    path_scale = summarize_path_scale_strata(
        prediction,
        path_audit,
        strata_columns=(
            "D_g_path_stratum",
            "V_g_stratum",
            "legal_path_count_stratum",
        ),
    )
    assert set(path_scale["stratum_axis"]) == {
        "D_g_path_stratum",
        "V_g_stratum",
        "legal_path_count_stratum",
    }
    assert path_scale["numerical_status"].eq("valid").all()

    gene_strata = pd.DataFrame(
        {
            "gene_id": ["g0", "g1"],
            "catalog_token_burden_stratum": ["low", "high"],
            "model_input_token_burden_stratum": ["low", "medium"],
        }
    )
    attribution = pd.DataFrame(
        {
            "condition": ["full", "full"],
            "cell_id": ["c0", "c1"],
            "gene_id": ["g0", "g1"],
            "record_id": ["r0", "r1"],
            "across_seed_median": [0.2, 0.3],
            "across_seed_iqr": [0.01, 0.02],
            "sign_agreement": [1.0, 1.0],
            "D_Q": [0.05, 0.07],
            "direction_status": ["stable_direction"] * 2,
            "magnitude_status": ["stable_magnitude"] * 2,
            "primary_summary_eligible": [True, True],
        }
    )
    density = summarize_event_density_strata(prediction, attribution, gene_strata)
    assert density.prediction["prediction_unit"].eq(
        "unique_seed_condition_cell_gene"
    ).all()
    assert density.attribution["attribution_record_count"].sum() == 4
    duplicated = pd.concat([prediction, prediction.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate gene-cell performance"):
        summarize_event_density_strata(duplicated, attribution, gene_strata)


def test_path_scale_and_token_burden_audits_are_built_from_one_toy_forward():
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    model = build_paired_models(
        gene, config["model"], seed=23, device="cpu"
    )["full"]
    contrasts = pd.DataFrame(
        {
            "contrast_id": ["toy:path-a:path-b"],
            "gene_id": [gene.gene_id],
            "numerator_path_ids": [[gene.path_ids[0]]],
            "denominator_path_ids": [[gene.path_ids[1]]],
        }
    )
    informative = gene.informative_row_mask
    audit = build_path_scale_audit(
        model,
        gene.model_input,
        gene_id=gene.gene_id,
        cell_ids=gene.cell_ids,
        path_ids=gene.path_ids,
        compatible_path_indices=gene.compatible_path_indices[informative],
        compatible_path_mask=gene.compatible_path_mask[informative],
        row_cell_index=gene.row_cell_index[informative],
        molecule_count=gene.molecule_count[informative],
        alternative_contrasts=contrasts,
        representation_collision_relative_tolerance=1.0e-6,
    )
    assert len(audit.records) == len(gene.cell_ids)
    assert audit.records["finite_output_pass"].all()
    assert audit.records["variable_edge_count_V_g"].eq(4).all()
    assert audit.records["legal_path_count"].eq(2).all()
    assert audit.records["local_contrast_gradient_norm"].gt(0).all()
    assert audit.records["production_path_scaling"].eq(
        "unscaled_gene_centered_residual_sum"
    ).all()
    assert audit.records["compatible_set_posterior_mass_molecule_weighted"].between(
        0, 1
    ).all()
    assert audit.output.path_residual is not None
    assert len(audit.representation_collisions) == 1
    collision = audit.representation_collisions.iloc[0]
    assert bool(collision.supervision_distinguishable)
    assert collision.status == "no_representation_collision_detected"
    assert collision.near_collision_cell_count == 0

    edges = tuple(f"edge-{index}" for index in range(gene.model_input.cis_features.shape[0]))
    burden = pd.DataFrame(
        [
            {
                "audit_population": population,
                "target_gene_id": gene.gene_id,
                "modality": modality,
                "edge_token_id": edge,
                "distinct_physical_event_count": event_count,
                "distinct_active_gate_key_count": (
                    gate_count if population == "model_input" else np.nan
                ),
                "saturated_anchor_group_count": 0,
                "saturated_cap_bucket_count": 0,
                "route_l1_mass": route_mass,
                "B_gate": b_gate if population == "model_input" else np.nan,
            }
            for population, modality, edge, event_count, gate_count, route_mass, b_gate in (
                ("catalog", "DNA", "edge-2", 5, 0, 1.5, 0.0),
                ("model_input", "DNA", "edge-2", 3, 2, 1.0, 0.7),
                ("catalog", "RNA", "edge-2", 2, 0, 0.8, 0.0),
                ("model_input", "RNA", "edge-2", 1, 1, 0.5, 0.4),
            )
        ]
    )
    tokens = build_event_density_token_audit(
        audit.output,
        burden,
        gene_id=gene.gene_id,
        cell_ids=gene.cell_ids,
        edge_token_ids=edges,
    )
    assert len(tokens) == len(gene.cell_ids) * len(edges)
    assert tokens["all_named_intermediates_finite"].all()
    edge_two = tokens.query("edge_token_id == 'edge-2'")
    assert edge_two["catalog_distinct_physical_event_count"].eq(7).all()
    assert edge_two["model_input_distinct_physical_event_count"].eq(4).all()
    np.testing.assert_allclose(
        edge_two["model_input_B_gate"], np.sqrt(0.7**2 + 0.4**2)
    )
    empty_edge = tokens.query("edge_token_id == 'edge-0'")
    assert empty_edge["catalog_token_burden_status"].eq("no_routes_on_token").all()
    assert empty_edge["catalog_distinct_physical_event_count"].eq(0).all()
    # Named norms are computed from the exact output, not accepted as table inputs.
    expected_pre = torch.linalg.vector_norm(
        audit.output.joint_projected[:, 2], dim=-1
    ).detach().numpy()
    np.testing.assert_allclose(edge_two["pre_normalization_token_norm"], expected_pre)


def test_three_train_frozen_support_tables_keep_empty_subgroups_not_estimable():
    metadata = pd.DataFrame(
        {
            "gene_id": ["g2", "g3", "g4", "g6", "g11", "g21"],
            "matrix_path_count": [2, 3, 4, 6, 11, 21],
            "train_ont_raw_count": [1, 2, 4, 8, 16, 32],
            "train_positive_cell_support": [2, 4, 8, 16, 32, 64],
        }
    )
    assignments = build_train_support_bin_assignments(metadata)
    assert assignments.query("stratifier == 'matrix_path_count'")[
        "support_bin"
    ].tolist() == ["2", "3", "4-5", "6-10", "11-20", ">20"]
    rows = []
    for condition in ("full", "full_additive_edge"):
        for seed in (11, 22, 33):
            for index, gene in enumerate(metadata["gene_id"]):
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "cell_id": f"c{index}",
                        "gene_id": gene,
                        "path_axis_identity": f"paths:{gene}",
                        "high_dtu": gene in {"g3", "g6"},
                        "ont_count_total": float(index + 1),
                        "ont_top1_tie_aware_hit": float(index % 2),
                        "ont_matrix_cross_entropy": 0.5 + 0.1 * index,
                    }
                )
    tables = summarize_support_stratified_sensitivity(
        pd.DataFrame(rows),
        assignments,
        metric_columns=(
            "ont_top1_tie_aware_hit",
            "ont_matrix_cross_entropy",
        ),
        dtu_provenance_status="PASS",
    )
    assert set(tables.across_seed["stratifier"]) == {
        "matrix_path_count",
        "train_ont_raw_count",
        "train_positive_cell_support",
    }
    empty = tables.across_seed.query(
        "stratifier == 'matrix_path_count' and support_bin == '>20' and dtu_stratum == 'high_DTU'"
    )
    # Two conditions x two required metrics x macro/count-weighted views.
    assert len(empty) == 8
    assert empty["status"].eq("not_estimable").all()
    assert empty["eligible_cell_gene_count"].eq(0).all()
    assert empty["raw_count_denominator"].eq(0).all()
    assert not any("adjust" in column.lower() for column in tables.across_seed)

    drift = pd.DataFrame(rows)
    drift.loc[
        (drift["condition"].eq("full_additive_edge"))
        & (drift["seed"].eq(11))
        & (drift["gene_id"].eq("g3")),
        "ont_count_total",
    ] = 999
    with pytest.raises(ValueError, match="weights, denominators"):
        summarize_support_stratified_sensitivity(
            drift,
            assignments,
            metric_columns=("ont_top1_tie_aware_hit",),
            dtu_provenance_status="PASS",
        )
