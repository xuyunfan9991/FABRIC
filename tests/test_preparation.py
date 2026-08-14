from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fabric.dataset import (
    FULL_FATE,
    INFORMATIVE_FATE,
    ProductionModalityTensors,
    assemble_gene_cell_model_input,
    build_compatibility_admission_record,
    validate_compatibility_artifact,
)
from fabric.likelihood import brute_force_compatible_path_nll, compatible_path_nll
from fabric.model import FABRICV2Model


def _compatibility_fixture():
    legal = {"g": ("path:z", "path:a"), "g_zero": ("q",)}
    rows = pd.DataFrame(
        {
            "compatibility_class_id": ["ec0", "ec1"],
            "cell_id": ["c0", "c1"],
            "target_gene_id": ["g", "g"],
            "split": ["train", "val"],
            "molecule_count": [2, 3],
            "pre_compatibility_qc_pass": [True, True],
            "compatible_path_ids": [["path:z"], ["path:z", "path:a"]],
            "final_fate": [INFORMATIVE_FATE, FULL_FATE],
            "technical_reason_code": ["", ""],
        }
    )
    manifest = {
        "producer": "frozen-toy-producer",
        "command": "producer --frozen",
        "code_version": "abc123",
        "alignment_identity": "align-v1",
        "reference_identity": "GRCh38-v1",
        "matrix_observation_input_identity": "ont-matrix-v1",
        "legal_path_catalog_identity": "paths-v1",
        "model_isoform_universe": "resolved_ont_matrix_structural_paths_only",
        "matrix_structural_path_count": 3,
        "cell_split_identity": "split-v1",
        "qc_policy": {"mapq": 20},
        "compatibility_policy": {"operator": "exact-frozen"},
        "candidate_gene_ids": ["g", "g_zero"],
        "candidate_support_status": [
            {"target_gene_id": "g", "support_status": "train_informative"},
            {"target_gene_id": "g_zero", "support_status": "zero_support"},
        ],
        "train_policy_identity": "policy-v1",
        "validation_policy_identity": "policy-v1",
        "test_exposure": "not_materialized_before_checkpoint",
        "artifact_complete": True,
        "G_fit_freeze_status": "FROZEN_FROM_TRAIN_ONLY",
        "test_rows_written": False,
        "training_authorized_or_started": False,
        "split_conservation": [
            {
                "split": "train",
                "captured_molecule_mass": 2,
                "pre_qc_pass_molecule_mass": 2,
                "technical_qc_failure_molecule_mass": 0,
                "empty_compatible_molecule_mass": 0,
                "proper_subset_compatible_molecule_mass": 2,
                "full_set_compatible_molecule_mass": 0,
                "other_explicit_fate_molecule_mass": 0,
            },
            {
                "split": "val",
                "captured_molecule_mass": 3,
                "pre_qc_pass_molecule_mass": 3,
                "technical_qc_failure_molecule_mass": 0,
                "empty_compatible_molecule_mass": 0,
                "proper_subset_compatible_molecule_mass": 0,
                "full_set_compatible_molecule_mass": 3,
                "other_explicit_fate_molecule_mass": 0,
            },
        ],
    }
    return manifest, rows, legal


def test_compatibility_manifest_accepts_frozen_nonlexicographic_axis_and_fails_drift():
    manifest, rows, legal = _compatibility_fixture()
    validation = validate_compatibility_artifact(
        manifest,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert validation.status == "ADMITTED"
    assert validation.informative_gene_ids == ("g",)
    admission = build_compatibility_admission_record(
        validation,
        input_manifest_id="input-v2",
        compatibility_artifact_id="compatible-v2",
    )
    assert admission["validation_status"] == "ADMITTED"
    assert admission["informative_gene_ids"] == ["g"]
    assert admission["structural_candidate_count"] == 2
    assert admission["matrix_structural_path_count"] == 3
    assert admission["model_isoform_universe"] == (
        "resolved_ont_matrix_structural_paths_only"
    )
    reversed_row = rows.copy()
    reversed_row.at[1, "compatible_path_ids"] = ["path:a", "path:z"]
    rejected = validate_compatibility_artifact(
        manifest,
        reversed_row,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert rejected.status == "REJECTED"
    assert any("frozen_axis_order" in reason for reason in rejected.reasons)
    empty_identity = dict(manifest, alignment_identity=None)
    rejected = validate_compatibility_artifact(
        empty_identity,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert any("alignment_identity" in reason for reason in rejected.reasons)

    noninteger = rows.copy()
    noninteger["molecule_count"] = noninteger["molecule_count"].astype(object)
    noninteger.loc[0, "molecule_count"] = 2.5
    rejected = validate_compatibility_artifact(
        manifest,
        noninteger,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert any("non_positive_or_non_integer_ec_mass" in reason for reason in rejected.reasons)

    conservation_drift = dict(manifest)
    conservation_drift["split_conservation"] = [
        dict(row) for row in manifest["split_conservation"]
    ]
    conservation_drift["split_conservation"][0]["captured_molecule_mass"] = 999
    rejected = validate_compatibility_artifact(
        conservation_drift,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "split_conservation_totals_mismatch" in rejected.reasons

    policy_drift = dict(manifest, validation_policy_identity="different-policy")
    rejected = validate_compatibility_artifact(
        policy_drift,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "train_validation_policy_drift" in rejected.reasons

    test_exposed_under_hidden_marker = pd.concat(
        [rows, rows.iloc[[0]].assign(cell_id="test-cell", split="test")],
        ignore_index=True,
    )
    rejected = validate_compatibility_artifact(
        manifest,
        test_exposed_under_hidden_marker,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "test_rows_materialized_despite_unexposed_marker" in rejected.reasons

    empty_support = dict(manifest)
    empty_support["candidate_support_status"] = [
        dict(row) for row in manifest["candidate_support_status"]
    ]
    empty_support["candidate_support_status"][1]["support_status"] = ""
    rejected = validate_compatibility_artifact(
        empty_support,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "empty_candidate_support_status" in rejected.reasons

    invalid_split = rows.copy()
    invalid_split.loc[1, "split"] = "validation"
    rejected = validate_compatibility_artifact(
        manifest,
        invalid_split,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert any(reason.startswith("invalid_ec_split_labels") for reason in rejected.reasons)

    partial = dict(
        manifest,
        artifact_complete=False,
        G_fit_freeze_status="NOT_FROZEN_PARTIAL",
    )
    rejected = validate_compatibility_artifact(
        partial,
        rows,
        legal_paths_by_gene=legal,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "partial_compatibility_artifact_not_admissible" in rejected.reasons
    assert "G_fit_not_frozen_from_complete_train_only_mass" in rejected.reasons

    duplicate_paths = {**legal, "g": ("path:z", "path:z")}
    rejected = validate_compatibility_artifact(
        manifest,
        rows,
        legal_paths_by_gene=duplicate_paths,
        expected_candidate_gene_ids=["g", "g_zero"],
        expected_candidate_gene_count=2,
    )
    assert "duplicate_legal_path_identity:g" in rejected.reasons


def _empty_modality(
    cell_ids: tuple[str, ...], gene_id: str, modality: str, edge_ids: tuple[str, ...]
) -> ProductionModalityTensors:
    return ProductionModalityTensors(
        cell_ids=cell_ids,
        target_gene_id=gene_id,
        modality=modality,
        ordered_edge_ids=edge_ids,
        event_ids=(),
        gate_key_ids=(),
        route_ids=(),
        route_event_index=np.zeros(0, dtype=np.int64),
        route_edge_index=np.zeros(0, dtype=np.int64),
        route_weight=np.zeros(0, dtype=np.float32),
        route_base_features=np.zeros((0, 1), dtype=np.float32),
        route_interaction_features=np.zeros((0, 0), dtype=np.float32),
        interaction_active_mask=np.zeros(0, dtype=bool),
        event_gate_key_index=np.zeros(0, dtype=np.int64),
        gate=np.zeros((len(cell_ids), 0), dtype=np.float32),
    )


def test_catalog_to_model_to_compatible_likelihood_closes(toy_gene_graph):
    cells = ("c0", "c1")
    normalized_cis = pd.DataFrame(
        {
            "edge_id": list(toy_gene_graph.edge_ids),
            "cis": np.linspace(-1, 1, len(toy_gene_graph.edge_ids)),
        }
    )
    ec = pd.DataFrame(
        {
            "compatibility_class_id": ["ec0", "ec1", "ec2"],
            "cell_id": ["c0", "c1", "c1"],
            "target_gene_id": [toy_gene_graph.gene_id] * 3,
            "split": ["train", "val", "val"],
            "molecule_count": [3, 4, 100],
            "compatible_path_ids": [
                [toy_gene_graph.path_ids[0]],
                [toy_gene_graph.path_ids[1]],
                list(toy_gene_graph.path_ids),
            ],
            "pre_compatibility_qc_pass": [True, True, True],
            "final_fate": [INFORMATIVE_FATE, INFORMATIVE_FATE, FULL_FATE],
        }
    )
    assembly = assemble_gene_cell_model_input(
        toy_gene_graph,
        cell_split=pd.DataFrame({"cell_id": cells, "split": ["train", "val"]}),
        normalized_cis_edges=normalized_cis,
        cis_feature_names=["cis"],
        dna=_empty_modality(cells, toy_gene_graph.gene_id, "DNA", toy_gene_graph.edge_ids),
        rna=_empty_modality(cells, toy_gene_graph.gene_id, "RNA", toy_gene_graph.edge_ids),
        compatibility_rows=ec,
    )
    model = FABRICV2Model(
        cis_dim=1,
        dna_base_dim=1,
        dna_interaction_dim=0,
        rna_base_dim=1,
        rna_interaction_dim=0,
        dynamic_dim=2,
        hidden_dim=4,
        attention_heads=1,
        path_hidden_dim=3,
    )
    output = model(assembly.model_input)
    observed = compatible_path_nll(
        output.path_logits,
        assembly.compatible_path_indices,
        assembly.compatible_path_mask,
        assembly.molecule_count,
        row_cell_index=assembly.row_cell_index,
        return_details=True,
    )
    lists = [[0], [1], [0, 1]]
    reference = brute_force_compatible_path_nll(
        output.path_logits,
        lists,
        assembly.molecule_count,
        row_cell_index=assembly.row_cell_index,
    )
    torch.testing.assert_close(observed.loss, reference)
    assert assembly.informative_row_mask.tolist() == [True, True, False]
    assert torch.isfinite(output.path_logits).all()
