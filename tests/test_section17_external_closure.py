from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fabric.choices import build_path_identifiability_index
from fabric.dataset import (
    FULL_FATE,
    IRPolicy,
    InteractionDesign,
    RouteBaseDesign,
    build_model_injection_equivalence_index,
    classify_retained_intron_evidence,
    rebuild_compatible_sets_after_ir_censoring,
)
from fabric.evaluate import (
    OntEpochMonitor,
    build_train_support_bin_assignments,
    summarize_support_stratified_sensitivity,
    validate_ont_matrix_identity,
)
from fabric.train import (
    TrainingRunManifest,
    build_paired_models,
    load_config,
    make_toy_genes,
    select_lambda_pair,
    training_manifest_from_config,
)


def _ir_evidence_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "molecule_id": "short",
        "intron_id": "ri",
        "target_gene_id": "g",
        "is_primary": True,
        "is_chimeric": False,
        "mapq": 60,
        "left_exon_aligned_bp": 12,
        "left_intron_aligned_bp": 12,
        "right_intron_aligned_bp": 12,
        "right_exon_aligned_bp": 0,
        "supports_excising_junction": False,
        "other_canonical_splice_junction_count": 0,
        "unspliced_intron_count": 1,
        "internal_priming_flag": False,
        "genomic_dna_contamination_flag": False,
        "protocol_mature_transcript_evidence": False,
        "library_id": "lib",
        "donor_id": "donor",
        "degradation_or_truncation_flag": True,
    }
    row.update(updates)
    return row


def test_section17_32_truncation_only_widens_compatible_set_and_never_creates_ir_positive():
    classified = classify_retained_intron_evidence(
        pd.DataFrame([_ir_evidence_row()]),
        policy=IRPolicy(20, 10, 10),
    ).iloc[0]
    assert classified.IR_evidence_class == "single_boundary_only"
    assert bool(classified.IR_evidence_censored)
    assert not bool(classified.IR_alignment_supported)
    assert classified.IR_interpretation_scope == "not_IR_positive"

    rebuilt = rebuild_compatible_sets_after_ir_censoring(
        pd.DataFrame(
            {
                "molecule_id": ["short"],
                "target_gene_id": ["g"],
                # The remaining junction/end evidence is compatible with two
                # spliced paths because the read is truncated.
                "accepted_non_ir_compatible_path_ids": [["spliced_a", "spliced_b"]],
                "ir_supported_path_ids": [["retained"]],
                "IR_alignment_supported": [False],
                "IR_evidence_censored": [True],
            }
        ),
        legal_paths_by_gene={"g": ("spliced_a", "spliced_b", "retained")},
    ).iloc[0]
    assert rebuilt.compatible_path_ids_after_ir_policy == [
        "spliced_a",
        "spliced_b",
    ]
    assert "retained" not in rebuilt.compatible_path_ids_after_ir_policy
    assert len(rebuilt.compatible_path_ids_after_ir_policy) > 1
    assert rebuilt.final_fate_after_ir_policy != FULL_FATE


def test_section17_32_ir_biogenesis_context_is_orthogonal_to_train_path_identifiability(
    toy_gene_graph,
):
    base = pd.DataFrame(
        {
            "cell_id": ["c0", "c1"],
            "gene_id": [toy_gene_graph.gene_id] * 2,
            "split": ["train", "train"],
            "compatible_path_ids": [["p0"], ["p1"]],
            "molecule_count": [3, 4],
            "IR_biogenesis_context": [
                "processed_context_supported",
                "mature_vs_nascent_unresolved",
            ],
        }
    )
    swapped = base.copy()
    swapped["IR_biogenesis_context"] = (
        swapped["IR_biogenesis_context"].iloc[::-1].to_numpy()
    )
    first = build_path_identifiability_index(toy_gene_graph, base)
    second = build_path_identifiability_index(toy_gene_graph, swapped)
    for name in ("genes", "paths", "groups", "train_patterns"):
        pd.testing.assert_frame_equal(
            getattr(first, name), getattr(second, name), check_exact=True
        )


def test_section17_40_residual_hits_that_should_have_physically_collapsed_fail_closed():
    physical_events = pd.DataFrame(
        {
            "event_id": ["e0", "e1"],
            "target_gene_id": ["g", "g"],
            "modality": ["DNA", "DNA"],
            "gate_key_id": ["shared", "shared"],
            "model_active": [True, True],
            "factor_entity_id": ["TF", "TF"],
            "factor_identity_kind": ["unique", "unique"],
            "motif_equivalence_family_id": ["family", "family"],
            "chromosome": ["chr1", "chr1"],
            "start": [100, 101],
            "end": [104, 105],
            "strand": ["+", "+"],
            "peak_id": ["peak", "peak"],
        }
    )
    routes = pd.DataFrame(
        {
            "route_id": ["r0", "r1"],
            "event_id": ["e0", "e1"],
            "target_gene_id": ["g", "g"],
            "modality": ["DNA", "DNA"],
            "edge_id": ["edge", "edge"],
            "anchor_region_id": ["anchor", "anchor"],
            "route_weight": [1.0, 1.0],
        }
    )
    base = RouteBaseDesign(
        route_ids=("r0", "r1"),
        values=np.ones((2, 1), dtype=np.float32),
        column_names=("DNA:factor_identity=TF",),
        manifest={},
        route_context=pd.DataFrame({"route_id": ["r0", "r1"]}),
    )
    interaction = InteractionDesign(
        route_ids=("r0", "r1"),
        values_by_modality={
            "DNA": np.zeros((2, 1), dtype=np.float32),
            "RNA": np.zeros((0, 0), dtype=np.float32),
        },
        active_mask_by_modality={
            "DNA": np.zeros(1, dtype=bool),
            "RNA": np.zeros(0, dtype=bool),
        },
        route_indices_by_modality={
            "DNA": np.asarray([0, 1], dtype=np.int64),
            "RNA": np.zeros(0, dtype=np.int64),
        },
        raw_support=pd.DataFrame(),
        manifest={},
        raw_contrasts=pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="should have collapsed"):
        build_model_injection_equivalence_index(
            physical_events,
            routes,
            base,
            interaction,
            ordered_edge_ids_by_gene={"g": ("edge",)},
        )


def _ont_identity_tables():
    rows = pd.DataFrame(
        {
            "matrix_row_id": ["m0", "m1"],
            "transcript_id": ["tx0", "tx1"],
            "gene_id": ["g", "g"],
        }
    )
    cells = pd.DataFrame({"cell_id": ["c0"]})
    crosswalk = pd.DataFrame(
        {
            "matrix_row_id": ["m0", "m1"],
            "transcript_id": ["tx0", "tx1"],
            "gene_id": ["g", "g"],
            "path_id": ["p0", "p1"],
        }
    )
    paths = pd.DataFrame(
        {
            "gene_id": ["g", "g"],
            "path_id": ["p0", "p1"],
            "transcript_aliases": [["tx0"], ["tx1"]],
        }
    )
    return rows, cells, crosswalk, paths


def test_section17_42_every_static_ont_resolution_failure_is_an_identity_error():
    rows, cells, crosswalk, paths = _ont_identity_tables()
    assert validate_ont_matrix_identity(rows, cells, crosswalk, paths).status == "PASS"

    with pytest.raises(ValueError, match="every ONT matrix row must resolve"):
        validate_ont_matrix_identity(rows, cells, crosswalk.iloc[[0]], paths)

    duplicated_resolution = pd.concat(
        [crosswalk, crosswalk.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="crosswalk matrix row.*unique"):
        validate_ont_matrix_identity(rows, cells, duplicated_resolution, paths)

    transcript_to_multiple_paths = crosswalk.copy()
    transcript_to_multiple_paths.loc[1, "transcript_id"] = "tx0"
    with pytest.raises(ValueError, match="crosswalk transcript.*unique"):
        validate_ont_matrix_identity(rows, cells, transcript_to_multiple_paths, paths)

    missing_model_path = crosswalk.copy()
    missing_model_path.loc[1, "path_id"] = "absent"
    with pytest.raises(ValueError, match="absent from live model"):
        validate_ont_matrix_identity(rows, cells, missing_model_path, paths)

    many_transcripts_one_path = crosswalk.copy()
    many_transcripts_one_path.loc[1, "path_id"] = "p0"
    with pytest.raises(ValueError, match="two ONT matrix transcripts"):
        validate_ont_matrix_identity(rows, cells, many_transcripts_one_path, paths)

    alias_collision = paths.copy()
    alias_collision.at[1, "transcript_aliases"] = ["tx0"]
    with pytest.raises(ValueError, match="multiple structural paths"):
        validate_ont_matrix_identity(rows, cells, crosswalk, alias_collision)

    extra_model_path = pd.concat(
        [
            paths,
            pd.DataFrame(
                {
                    "gene_id": ["g"],
                    "path_id": ["outside_matrix"],
                    "transcript_aliases": [["tx_outside_matrix"]],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="absent from the ONT matrix isoform axis"):
        validate_ont_matrix_identity(rows, cells, crosswalk, extra_model_path)


def _support_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = build_train_support_bin_assignments(
        pd.DataFrame(
            {
                "gene_id": ["g"],
                "matrix_path_count": [3],
                "train_ont_raw_count": [12],
                "train_positive_cell_support": [6],
            }
        )
    )
    records = pd.DataFrame(
        [
            {
                "condition": condition,
                "seed": seed,
                "cell_id": "c",
                "gene_id": "g",
                "path_axis_identity": "paths:g:v1",
                "high_dtu": False,
                "ont_count_total": 10,
                "metric": 0.25,
            }
            for condition in ("full", "full_additive_edge")
            for seed in (11, 22, 33)
        ]
    )
    return records, assignments


def test_section17_43_selection_rejects_monitor_fields_and_final_axes_are_frozen():
    with pytest.raises(ValueError, match=r"exactly CIS\+DNA, CIS\+RNA, and Full"):
        select_lambda_pair(
            {
                (0.0, 0.01): {
                    "cis_dna": 1.0,
                    "cis_rna": 1.0,
                    "full": 1.0,
                    "ont_matrix_kl_count_weighted": 1.0,
                }
            },
            [(0.0, 0.01)],
        )

    model = torch.nn.Linear(1, 1)
    sealed = OntEpochMonitor(
        lambda _: {
            "validation_compatible_path_nll": 1.0,
            "ont_matrix_kl_count_weighted": 1.0,
        }
    )
    sealed.record_completed_epoch(1, model)
    with pytest.raises(RuntimeError, match="post-selection reporting"):
        sealed.read_for_post_selection_reporting(
            selection_and_reporting_rules_frozen=False
        )

    records, assignments = _support_records()
    result = summarize_support_stratified_sensitivity(
        records,
        assignments,
        metric_columns=("metric",),
        dtu_provenance_status="PASS",
    )
    assert set(result.across_seed["uncertainty_semantics"]) == {
        "optimization_repeat_not_biological_confidence_interval"
    }

    drift = records.copy()
    drift.loc[
        drift["condition"].eq("full_additive_edge") & drift["seed"].eq(22),
        "path_axis_identity",
    ] = "paths:g:changed"
    with pytest.raises(ValueError, match="path axes, weights, denominators"):
        summarize_support_stratified_sensitivity(
            drift,
            assignments,
            metric_columns=("metric",),
            dtu_provenance_status="PASS",
        )

    missing_identity = records.copy()
    missing_identity.loc[0, "path_axis_identity"] = None
    with pytest.raises(ValueError, match="explicit non-empty string"):
        summarize_support_stratified_sensitivity(
            missing_identity,
            assignments,
            metric_columns=("metric",),
            dtu_provenance_status="PASS",
        )


def test_section17_33_command_seed_and_condition_are_exact_and_reproducible():
    config = load_config("configs/fabric_v2_toy.yaml")
    frozen = training_manifest_from_config(config, seed=1103, condition="full")
    expected = TrainingRunManifest(
        seed=1103,
        condition="full",
        learning_rate=0.01,
        lr_scheduler_name="constant",
        lr_scheduler_factor=None,
        lr_scheduler_patience=None,
        lr_scheduler_min_lr=None,
        gradient_clip_norm=0.0,
        lambda_base=0.001,
        lambda_int=0.01,
        max_epochs=3,
        early_stopping_patience=3,
        inputs_frozen=True,
        resources_frozen=True,
        primary_epoch_unit="sampled_informative_gene_cell_horvitz_thompson",
        max_train_gene_cells_per_gene_per_epoch=512,
        resample_train_gene_cells_each_epoch=True,
        selected_gene_cell_ec_rows="all_informative_rows",
        sampling_estimator="horvitz_thompson_full_train_molecule_total",
        optimizer_step_unit="train_positive_gene",
        gene_microbatch_gradient_accumulation=True,
        batch_policy="gene_shape_adaptive_v1",
        target_gpu_allocated_bytes=67_108_864,
        cuda_allocator_limit_bytes=None,
        max_cells_per_gpu_batch=32_768,
        prefetch_backed_gene_shards=False,
        compute_precision="float32_highest",
    )
    assert frozen == expected

    gene = make_toy_genes()[0]
    first = build_paired_models(gene, config["model"], seed=1103, device="cpu")
    rerun = build_paired_models(gene, config["model"], seed=1103, device="cpu")
    for name, value in first["full"].state_dict().items():
        torch.testing.assert_close(
            rerun["full"].state_dict()[name], value, atol=0, rtol=0
        )
