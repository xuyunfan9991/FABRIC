from __future__ import annotations

import pandas as pd
import pytest

from fabric.dataset import (
    FULL_FATE,
    IRPolicy,
    build_ir_library_audit,
    build_ont_observation_process_audit,
    build_ont_observation_admission_record,
    build_rna_window_coverage_audit,
    classify_retained_intron_evidence,
    rebuild_compatible_sets_after_ir_censoring,
)


def _ir_rows():
    base = {
        "target_gene_id": "g",
        "is_primary": True,
        "is_chimeric": False,
        "mapq": 60,
        "left_exon_aligned_bp": 12,
        "left_intron_aligned_bp": 12,
        "right_intron_aligned_bp": 12,
        "right_exon_aligned_bp": 12,
        "supports_excising_junction": False,
        "other_canonical_splice_junction_count": 1,
        "unspliced_intron_count": 1,
        "internal_priming_flag": False,
        "genomic_dna_contamination_flag": False,
        "protocol_mature_transcript_evidence": False,
        "library_id": "lib",
        "donor_id": "donor",
    }
    rows = []
    changes = [
        {},
        {"right_exon_aligned_bp": 0},
        {"left_exon_aligned_bp": 0, "right_exon_aligned_bp": 0},
        {"supports_excising_junction": True},
        {"unspliced_intron_count": 3, "other_canonical_splice_junction_count": 0},
    ]
    for index, change in enumerate(changes):
        rows.append({**base, **change, "molecule_id": f"m{index}", "intron_id": f"i{index}"})
    return pd.DataFrame(rows)


def test_ir_policy_distinguishes_boundary_and_excising_evidence_and_rebuilds_axis_order():
    classified = classify_retained_intron_evidence(
        _ir_rows(),
        policy=IRPolicy(20, 10, 10),
    )
    assert classified.IR_evidence_class.tolist() == [
        "bilateral_boundary_supported",
        "single_boundary_only",
        "intron_only",
        "excising_junction",
        "bilateral_boundary_supported",
    ]
    assert bool(classified.iloc[4].multi_intron_unspliced_pattern)
    assert classified.iloc[4].IR_biogenesis_context == "mature_vs_nascent_unresolved"
    audit = build_ir_library_audit(classified)
    assert audit.iloc[0].IR_alignment_supported_count == 2

    rebuilt = rebuild_compatible_sets_after_ir_censoring(
        pd.DataFrame(
            {
                "molecule_id": ["supported", "censored"],
                "target_gene_id": ["g", "g"],
                "accepted_non_ir_compatible_path_ids": [["z"], []],
                "ir_supported_path_ids": [["a"], ["a"]],
                "IR_alignment_supported": [True, False],
                "IR_evidence_censored": [False, True],
            }
        ),
        legal_paths_by_gene={"g": ("z", "a")},
    )
    assert rebuilt.iloc[0].compatible_path_ids_after_ir_policy == ["z", "a"]
    assert rebuilt.iloc[1].compatible_path_ids_after_ir_policy == ["z", "a"]
    assert rebuilt.iloc[1].final_fate_after_ir_policy == FULL_FATE
    duplicate = pd.DataFrame(
        {
            "molecule_id": ["x"],
            "target_gene_id": ["g"],
            "accepted_non_ir_compatible_path_ids": [["z", "z"]],
            "ir_supported_path_ids": [[]],
            "IR_alignment_supported": [False],
            "IR_evidence_censored": [True],
        }
    )
    with pytest.raises(ValueError, match="outside the frozen legal catalog"):
        rebuild_compatible_sets_after_ir_censoring(
            duplicate, legal_paths_by_gene={"g": ("z", "a")}
        )


def test_rna_window_waterfall_uses_region_priority_and_separate_active_suffix():
    frame = pd.DataFrame(
        {
            "reference_site_id": ["exon", "splice", "pas", "deep"],
            "factor_entity_id": ["RBP"] * 4,
            "reference_uniquely_mappable": [True] * 4,
            "eligible_gene": [True] * 4,
            "factor_entity_mappable": [True] * 4,
            "exonic_in_any_legal_transcript": [True, False, False, False],
            "inside_allowed_splice_window": [True, True, False, False],
            "inside_allowed_tss_pas_window": [True, False, True, False],
            "inside_allowed_rna_window": [True, True, True, False],
            "has_legal_route": [True, True, True, False],
            "retained_after_cap": [True, True, False, False],
            "model_active": [True, False, False, False],
            "denominator_kind": ["reference_experimentally_supported_site_coverage"] * 4,
            "assay": ["eCLIP"] * 4,
            "biosample_context": ["frozen-context"] * 4,
            "reference_build_identity": ["GRCh38-v1"] * 4,
        }
    )
    audit = build_rna_window_coverage_audit(frame)
    assert set(audit.region_class) == {
        "exonic",
        "splice_proximal_intronic",
        "other_allowed_site_proximal",
        "deep_intronic",
    }
    assert set(audit.model_active_scope) == {"train_derived_gate_admission_suffix"}


def test_ont_observation_process_stays_cross_pipeline_and_pending_until_rebuild():
    matrix = {
        "software_identity": "matrix-tool",
        "config_identity": "c1",
        "reference_identity": "r",
        "gtf_identity": "g",
        "feature_identity": "f",
        "barcode_identity": "b",
        "qc_policy_identity": "q",
        "assignment_policy_identity": "matrix-assignment",
    }
    compatible = dict(matrix, software_identity="compatible-tool", assignment_policy_identity="Ck")
    conservation = pd.DataFrame(
        {
            "split": ["val"],
            "cell_id": ["c"],
            "target_gene_id": ["g"],
            "matrix_count": [10],
            "pre_compatibility_mass": [10],
            "empty_compatible_mass": [1],
            "proper_subset_compatible_mass": [6],
            "full_set_compatible_mass": [2],
            "other_explicit_fate_mass": [1],
            "matrix_compatible_overlap_mass": [8],
            "matrix_only_mass": [2],
            "compatibility_only_mass": [2],
        }
    )
    audit = build_ont_observation_process_audit(
        matrix_manifest=matrix,
        compatibility_manifest=compatible,
        conservation_rows=conservation,
        compatible_rebuild_performed=False,
    )
    assert audit.status == "PENDING_OBSERVATION_PROCESS_AUDIT"
    assert audit.comparison_name == "same_library_cross_pipeline_ont_matrix_agreement"
    assert "compatible_read_rebuild_not_performed" in audit.reasons
    completed = build_ont_observation_process_audit(
        matrix_manifest=matrix,
        compatibility_manifest=compatible,
        conservation_rows=conservation,
        compatible_rebuild_performed=True,
    )
    assert completed.status == "ADMITTED"
    assert completed.comparison_name == "same_library_cross_pipeline_ont_matrix_agreement"
    admission = build_ont_observation_admission_record(
        completed,
        matrix_identity="matrix-v2",
        crosswalk_identity="crosswalk-v2",
        path_identity="paths-v2",
        split_identity="splits-v2",
        metric_schema_version="fabric_v2_validation_monitor_v2",
    )
    assert admission["status"] == "ADMITTED"
    assert admission["comparison_name"] == completed.comparison_name
    with pytest.raises(RuntimeError, match="pending"):
        build_ont_observation_admission_record(
            audit,
            matrix_identity="matrix-v2",
            crosswalk_identity="crosswalk-v2",
            path_identity="paths-v2",
            split_identity="splits-v2",
            metric_schema_version="fabric_v2_validation_monitor_v2",
        )
