from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from scipy import sparse
import yaml

from fabric.compatibility_artifact import (
    EMPTY_FATE,
    FULL_FATE,
    INFORMATIVE_FATE,
    TECHNICAL_FAILURE_FATE,
    AlignmentEvidence,
    CompatibilityPolicy,
    GeneAccumulator,
    GenePathCatalog,
    PathEvidence,
    RetainedIntronOpportunity,
    _candidate_support_rows,
    _augment_long_read_audit_with_ir,
    _long_read_audit_frame,
    _merge_candidate_support,
    _merge_long_read_audits,
    _missing_quantifier_fields,
    _observation_process_config,
    _register_unique_cell_gene_umi,
    _combine_reconciliation_summaries,
    _reconciliation_summary_partials,
    _sparse_pair_values,
    _validate_reconciliation_axes,
    _update_audit_groups,
    build_cell_lookup,
    build_path_catalog,
    evaluate_alignment_compatibility,
    parse_alignment_evidence,
    refresh_observation_process_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/processed/fabric_ont_gene_selection_v3"
CONFIG = ROOT / "configs/fabric_v2_compatible_ec.yaml"


def _policy() -> CompatibilityPolicy:
    return CompatibilityPolicy(
        minimum_mapq=20,
        minimum_junction_anchor_bp=8,
        maximum_deletion_bp=20,
        junction_tolerance_bp=2,
        terminal_tolerance_bp=2,
        ir_minimum_exon_aligned_bp_each_side=8,
        ir_minimum_intron_aligned_bp_each_side=8,
    )


def _catalog() -> GenePathCatalog:
    spliced = PathEvidence(
        path_id="tx_spliced",
        matrix_row_0based=0,
        exons=((100, 150), (200, 250)),
        junctions=((150, 200),),
    )
    retained = PathEvidence(
        path_id="tx_retained",
        matrix_row_0based=1,
        exons=((100, 250),),
        junctions=(),
    )
    return GenePathCatalog(
        gene_id="gene_fixture",
        chrom="chr1",
        strand="+",
        start_0based=100,
        end_0based_exclusive=250,
        ordered_path_ids=("tx_spliced", "tx_retained"),
        paths={"tx_spliced": spliced, "tx_retained": retained},
        retained_introns=(
            RetainedIntronOpportunity(
                intron=(150, 200),
                spliced_path_ids=("tx_spliced",),
                retained_path_ids=("tx_retained",),
            ),
        ),
    )


def _evaluate(evidence: AlignmentEvidence, **overrides):
    arguments = {
        "mapq": 60,
        "alignment_strand": "+",
        "is_primary": True,
        "sa_tag_present": False,
        "read_name_parse_status": "parsed",
    }
    arguments.update(overrides)
    return evaluate_alignment_compatibility(
        evidence,
        _catalog(),
        _policy(),
        **arguments,
    )


def test_duplicate_cell_gene_umi_fails_before_molecule_counting() -> None:
    accumulator = GeneAccumulator()
    _register_unique_cell_gene_umi(
        accumulator, cell_id="cell", gene_id="gene", umi="umi"
    )
    with pytest.raises(ValueError, match="cell-gene-UMI cell/gene/umi"):
        _register_unique_cell_gene_umi(
            accumulator, cell_id="cell", gene_id="gene", umi="umi"
        )
    assert accumulator.ec_counts == Counter()
    assert accumulator.cell_counts == {}


def test_compatible_sets_are_ordered_and_do_not_use_hard_tx_labels() -> None:
    spliced = _evaluate(
        AlignmentEvidence(
            covered_blocks=((100, 150), (200, 250)),
            observed_junctions=((150, 200),),
            junction_anchors=((50, 50),),
            deletion_intervals=(),
            aligned_reference_bp=100,
            soft_clip_bp=0,
            hard_clip_bp=0,
        )
    )
    retained = _evaluate(
        AlignmentEvidence(
            covered_blocks=((100, 250),),
            observed_junctions=(),
            junction_anchors=(),
            deletion_intervals=(),
            aligned_reference_bp=150,
            soft_clip_bp=0,
            hard_clip_bp=0,
        )
    )
    assert spliced.compatible_path_ids == ("tx_spliced",)
    assert spliced.final_fate == INFORMATIVE_FATE
    assert retained.compatible_path_ids == ("tx_retained",)
    assert retained.final_fate == INFORMATIVE_FATE
    assert retained.ir_alignment_supported_count == 1


def test_insufficient_retained_intron_evidence_is_censored_before_rebuild() -> None:
    result = _evaluate(
        AlignmentEvidence(
            covered_blocks=((140, 165),),
            observed_junctions=(),
            junction_anchors=(),
            deletion_intervals=(),
            aligned_reference_bp=25,
            soft_clip_bp=0,
            hard_clip_bp=0,
        )
    )
    assert result.ir_evidence_censored_count == 1
    assert result.compatible_path_ids == ("tx_spliced", "tx_retained")
    assert result.final_fate == FULL_FATE


def test_empty_fate_and_technical_failure_are_separate() -> None:
    empty = _evaluate(
        AlignmentEvidence(
            covered_blocks=((100, 160), (210, 250)),
            observed_junctions=((160, 210),),
            junction_anchors=((60, 40),),
            deletion_intervals=(),
            aligned_reference_bp=100,
            soft_clip_bp=0,
            hard_clip_bp=0,
        )
    )
    technical = _evaluate(
        AlignmentEvidence(
            covered_blocks=((100, 150),),
            observed_junctions=(),
            junction_anchors=(),
            deletion_intervals=(),
            aligned_reference_bp=50,
            soft_clip_bp=0,
            hard_clip_bp=0,
        ),
        mapq=5,
    )
    assert empty.pre_compatibility_qc_pass
    assert empty.final_fate == EMPTY_FATE
    assert empty.compatible_path_ids == ()
    assert not technical.pre_compatibility_qc_pass
    assert technical.final_fate == TECHNICAL_FAILURE_FATE
    assert technical.technical_reason_code == "mapq_below_minimum"

    outside_frozen_span = _evaluate(
        AlignmentEvidence(
            covered_blocks=((260, 300),),
            observed_junctions=(),
            junction_anchors=(),
            deletion_intervals=(),
            aligned_reference_bp=40,
            soft_clip_bp=0,
            hard_clip_bp=0,
        )
    )
    assert outside_frozen_span.pre_compatibility_qc_pass
    assert outside_frozen_span.final_fate == EMPTY_FATE


def test_cigar_parser_uses_half_open_genomic_coordinates() -> None:
    read = pysam.AlignedSegment()
    read.reference_start = 100
    read.cigartuples = ((0, 50), (3, 50), (0, 50))
    evidence = parse_alignment_evidence(read)
    assert evidence.covered_blocks == ((100, 150), (200, 250))
    assert evidence.observed_junctions == ((150, 200),)
    assert evidence.junction_anchors == ((50, 50),)


def test_integer_mass_conservation_and_train_only_g_fit_rule() -> None:
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for fate, mass in ((EMPTY_FATE, 2), (INFORMATIVE_FATE, 3), (FULL_FATE, 5)):
        _update_audit_groups(
            groups,
            split="train",
            library_id="lib",
            donor_id="emb",
            gene_id="gene_fit",
            cell_state="state",
            qc_pass=True,
            final_fate=fate,
            molecule_count=mass,
        )
    audit = _long_read_audit_frame(groups, ("gene_fit", "gene_zero"))
    global_rows = audit.loc[audit["stratum_type"].eq("global")]
    assert global_rows["pre_compatibility_qc_pass_molecule_mass"].unique().tolist() == [10]
    assert global_rows["terminal_molecule_mass"].sum() == 10
    assert global_rows["mass_conservation_pass"].all()

    selected = pd.DataFrame(
        {
            "gene_id": ["gene_fit", "gene_zero"],
            "DTU_score": [0.9, 0.1],
            "top_DTU_gene": [True, False],
        }
    ).set_index("gene_id")
    support = {
        "gene_fit": Counter({f"train:{INFORMATIVE_FATE}": 3}),
        "gene_zero": Counter({f"val:{INFORMATIVE_FATE}": 9}),
    }
    rows = _candidate_support_rows(
        ("gene_fit", "gene_zero"), selected, support
    )
    by_gene = {row["target_gene_id"]: row for row in rows}
    assert by_gene["gene_fit"]["support_status"].startswith("likelihood_fit")
    assert by_gene["gene_zero"]["support_status"].startswith("graph_only")
    assert by_gene["gene_fit"]["top_DTU_gene"] is True
    assert by_gene["gene_zero"]["validation_positive_informative_ec_mass"] == 9


def test_chromosome_shard_merge_preserves_integer_mass_and_metadata() -> None:
    selected = pd.DataFrame(
        {
            "gene_id": ["gene_fit", "gene_zero"],
            "DTU_score": [0.9, 0.1],
            "top_DTU_gene": [True, False],
        }
    ).set_index("gene_id")
    first_support = pd.DataFrame(
        _candidate_support_rows(
            ("gene_fit", "gene_zero"),
            selected,
            {
                "gene_fit": Counter({f"train:{INFORMATIVE_FATE}": 2}),
                "gene_zero": Counter(),
            },
        )
    )
    second_support = pd.DataFrame(
        _candidate_support_rows(
            ("gene_fit", "gene_zero"),
            selected,
            {
                "gene_fit": Counter(),
                "gene_zero": Counter({f"val:{INFORMATIVE_FATE}": 4}),
            },
        )
    )
    merged_support = _merge_candidate_support([first_support, second_support])
    merged_by_gene = merged_support.set_index("target_gene_id")
    assert merged_by_gene.loc["gene_fit", "train_positive_informative_ec_mass"] == 2
    assert merged_by_gene.loc["gene_zero", "validation_positive_informative_ec_mass"] == 4
    assert bool(merged_by_gene.loc["gene_fit", "top_DTU_gene"])

    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    _update_audit_groups(
        groups,
        split="train",
        library_id="lib",
        donor_id="emb",
        gene_id="gene_fit",
        cell_state="state",
        qc_pass=True,
        final_fate=INFORMATIVE_FATE,
        molecule_count=2,
    )
    audit = _long_read_audit_frame(groups, ("gene_fit", "gene_zero"))
    merged_audit = _merge_long_read_audits([audit, audit.copy()])
    global_rows = merged_audit.loc[merged_audit["stratum_type"].eq("global")]
    assert global_rows["pre_compatibility_qc_pass_molecule_mass"].unique().tolist() == [4]
    assert global_rows["terminal_molecule_mass"].sum() == 4
    assert global_rows["mass_conservation_pass"].all()


def test_ir_protocol_audit_reports_missing_flags_as_not_estimable(tmp_path: Path) -> None:
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for fate in (INFORMATIVE_FATE, FULL_FATE):
        _update_audit_groups(
            groups,
            split="train",
            library_id="lib",
            donor_id="emb",
            gene_id="gene_fit",
            cell_state="state",
            qc_pass=True,
            final_fate=fate,
            molecule_count=1,
        )
    audit = _long_read_audit_frame(groups, ("gene_fit",))
    molecules = pd.DataFrame(
        {
            "split": ["train", "train"],
            "library_id": ["lib", "lib"],
            "donor_id": ["emb", "emb"],
            "target_gene_id": ["gene_fit", "gene_fit"],
            "reporting_cell_state": ["state", "state"],
            "pre_compatibility_qc_pass": [True, True],
            "molecule_count": [1, 1],
            "ir_alignment_supported_count": [1, 0],
            "ir_evidence_censored_count": [0, 2],
            "multi_intron_unspliced_pattern": [False, False],
            "ir_biogenesis_context": [
                "mature_vs_nascent_unresolved",
                "not_applicable_no_ir_alignment_support",
            ],
            "internal_priming_status": [
                "not_available_from_frozen_bam",
                "not_available_from_frozen_bam",
            ],
            "genomic_dna_contamination_status": [
                "not_available_from_frozen_bam",
                "not_available_from_frozen_bam",
            ],
            "protocol_mature_transcript_evidence_status": [
                "not_available_from_frozen_bam",
                "not_available_from_frozen_bam",
            ],
        }
    )
    molecule_path = tmp_path / "molecule_fates.parquet"
    molecules.to_parquet(molecule_path, index=False)
    result = _augment_long_read_audit_with_ir(audit, [molecule_path])
    global_rows = result.loc[result["stratum_type"].eq("global")]
    assert global_rows["ir_alignment_supported_molecule_mass"].unique().tolist() == [1]
    assert global_rows["ir_evidence_censored_opportunity_count"].unique().tolist() == [2]
    assert global_rows["mature_vs_nascent_unresolved_fraction"].unique().tolist() == [0.5]
    assert set(global_rows["internal_priming_fraction_status"]) == {
        "not_estimable_missing_upstream_flag"
    }


def test_protocol_qc_not_performed_is_not_reported_as_zero_prevalence(
    tmp_path: Path,
) -> None:
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    _update_audit_groups(
        groups,
        split="train",
        library_id="lib",
        donor_id="emb",
        gene_id="gene_fit",
        cell_state="state",
        qc_pass=True,
        final_fate=INFORMATIVE_FATE,
        molecule_count=1,
    )
    audit = _long_read_audit_frame(groups, ("gene_fit",))
    molecules = pd.DataFrame(
        {
            "split": ["train"],
            "library_id": ["lib"],
            "donor_id": ["emb"],
            "target_gene_id": ["gene_fit"],
            "reporting_cell_state": ["state"],
            "pre_compatibility_qc_pass": [True],
            "molecule_count": [1],
            "ir_alignment_supported_count": [0],
            "ir_evidence_censored_count": [0],
            "multi_intron_unspliced_pattern": [False],
            "ir_biogenesis_context": ["not_applicable_no_ir_alignment_support"],
            "internal_priming_status": ["not_available_from_frozen_bam"],
            "genomic_dna_contamination_status": ["not_available_from_frozen_bam"],
            "protocol_mature_transcript_evidence_status": [
                "not_available_from_frozen_bam"
            ],
        }
    )
    molecule_path = tmp_path / "molecule_fates.parquet"
    molecules.to_parquet(molecule_path, index=False)
    provenance = {
        "internal_priming_qc_provenance": "NOT_PERFORMED_USER_CONFIRMED_DEFAULT",
        "genomic_dna_contamination_qc_provenance": (
            "NOT_PERFORMED_USER_CONFIRMED_DEFAULT"
        ),
        "protocol_mature_transcript_qc_provenance": (
            "NOT_PERFORMED_USER_CONFIRMED_DEFAULT"
        ),
    }
    result = _augment_long_read_audit_with_ir(
        audit, [molecule_path], provenance
    )
    assert set(result["internal_priming_assessment_status"]) == {
        "not_performed_upstream"
    }
    assert set(result["internal_priming_fraction_status"]) == {
        "not_applicable_qc_not_performed"
    }
    assert result["internal_priming_positive_fraction"].isna().all()


def test_documented_quantifier_provenance_records_cross_pipeline_result() -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    assert _missing_quantifier_fields(raw["matrix_quantifier_provenance"]) == []
    process = _observation_process_config(raw)
    assert process["matrix_cell_gene_reconciliation"] == "CROSS_PIPELINE_RECONCILED"
    assert process["matrix_count_semantics_verified_same_population"] is False
    assert process["comparison_name"] == (
        "same_library_cross_pipeline_ont_matrix_agreement"
    )
    assert process["compatible_test_row_exposure"] == (
        "not_materialized_before_checkpoint"
    )
    assert process["matrix_test_count_exposure"] == (
        "previously_materialized_held_out_test"
    )
    assert process["test_predictions_or_metrics_computed"] is False


def test_refresh_observation_audit_updates_only_audit_layer(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "artifact_complete": True,
        "training_authorized_or_started": False,
        "artifact_components": {},
    }
    observation = {
        "status": "PENDING_OBSERVATION_PROCESS_AUDIT",
        "reasons": ["missing_matrix_quantifier_provenance:software_identity"],
        "training_authorized_or_started": False,
    }
    (artifact / "CompatibilityArtifactManifest.json").write_text(
        json.dumps(manifest)
    )
    (artifact / "OntObservationProcessAudit.json").write_text(
        json.dumps(observation)
    )
    pd.DataFrame({"stratum_type": ["global"]}).to_parquet(
        artifact / "LongReadCompatibilityAudit.parquet", index=False
    )

    raw = yaml.safe_load(CONFIG.read_text())
    raw["observation_process"][
        "matrix_cell_gene_reconciliation"
    ] = "PENDING_NUMERICAL_MATRIX_JOIN"
    pending_config = tmp_path / "pending.yaml"
    pending_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    result = refresh_observation_process_audit(pending_config, artifact)
    updated = json.loads((artifact / "OntObservationProcessAudit.json").read_text())
    updated_manifest = json.loads(
        (artifact / "CompatibilityArtifactManifest.json").read_text()
    )
    long_read = pd.read_parquet(artifact / "LongReadCompatibilityAudit.parquet")
    assert result["missing_provenance_fields"] == []
    assert result["admission_pass"] is False
    assert updated["reasons"] == [
        "numerical_matrix_cell_gene_reconciliation_pending"
    ]
    assert updated["matrix_quantifier_provenance"]["software_identity"].startswith(
        "BLAZE_v2.5.1"
    )
    assert updated_manifest["admission_pass"] is False
    assert updated["test_predictions_or_metrics_computed"] is False
    assert updated_manifest["test_predictions_or_metrics_computed"] is False
    assert set(long_read["internal_priming_assessment_status"]) == {
        "not_performed_upstream"
    }
    assert (artifact / "observation_audit_source_snapshot.py").is_file()
    assert (artifact / "observation_audit_config_snapshot.yaml").is_file()


def test_matrix_reconciliation_scope_and_mass_conservation() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "val"],
            "rna_embryo_id": ["Emb01", "Emb01", "Emb02"],
            "matrix_library_prefix": ["lib1", "lib1", "lib2"],
            "target_gene_id": ["g1", "g1", "g2"],
            "matrix_mapped_count": [0, 2, 7],
            "captured_gene_assigned_mass": [3, 5, 8],
            "pre_compatibility_mass": [2, 4, 7],
            "technical_qc_failure_mass": [1, 1, 1],
            "empty_compatible_mass": [0, 1, 0],
            "proper_subset_compatible_mass": [2, 2, 6],
            "full_set_compatible_mass": [0, 1, 1],
            "other_explicit_fate_mass": [0, 0, 0],
            "matrix_scope_fate": [
                "ont_count_total_zero",
                "fewer_than_two_positive_matrix_transcripts",
                "eligible",
            ],
            "matrix_pre_compatibility_relation": [
                "matrix_zero",
                "matrix_less_than_compatible",
                "matrix_equals_compatible",
            ],
        }
    )
    summary = _combine_reconciliation_summaries(
        _reconciliation_summary_partials(frame)
    )
    global_row = summary.loc[summary["stratum_type"].eq("global")].iloc[0]
    assert global_row["candidate_cell_gene_count"] == 3
    assert global_row["matrix_mapped_count_mass"] == 9
    assert global_row["pre_compatibility_mass"] == 13
    assert global_row["ont_count_total_zero_cell_gene_count"] == 1
    assert global_row[
        "fewer_than_two_positive_matrix_transcripts_cell_gene_count"
    ] == 1
    assert global_row["eligible_cell_gene_count"] == 1
    assert bool(global_row["mass_conservation_pass"])


def test_sparse_pair_lookup_preserves_requested_order_and_implicit_zeros() -> None:
    matrix = sparse.csr_matrix(
        np.asarray([[0, 4, 0, 2], [3, 0, 5, 0], [0, 0, 0, 7]], dtype=np.int64)
    )
    rows = np.asarray([2, 0, 1, 0, 1, 2], dtype=np.int64)
    columns = np.asarray([3, 0, 2, 1, 0, 1], dtype=np.int64)
    assert _sparse_pair_values(matrix, rows, columns).tolist() == [7, 0, 5, 4, 3, 0]


def test_matrix_reconciliation_axis_validation_is_fail_closed() -> None:
    crosswalk = pd.DataFrame(
        {
            "matrix_row_0based": range(101_067),
            "matrix_transcript_name": [f"name{i}" for i in range(101_067)],
            "resolved_transcript_id": [f"tx{i}" for i in range(101_067)],
            "gene_id": ["g_other"] * 101_067,
        }
    )
    cells = pd.DataFrame(
        {
            "matrix_column_0based": range(217_933),
            "matrix_barcode": [f"bc{i}" for i in range(217_933)],
            "cell_id": [f"cell{i}" for i in range(217_933)],
        }
    )
    candidates = [f"g{i}" for i in range(17_706)]
    paths = pd.DataFrame(
        {
            "matrix_row_0based": range(90_672),
            "path_id": [f"tx{i}" for i in range(90_672)],
            "gene_id": [candidates[i % len(candidates)] for i in range(90_672)],
        }
    )
    selected = pd.DataFrame({"gene_id": candidates})
    manifest = {"candidate_gene_ids": candidates}
    with pytest.raises(ValueError, match="transcript-to-gene mapping differs"):
        _validate_reconciliation_axes(
            crosswalk=crosswalk,
            cells=cells,
            paths=paths,
            selected=selected,
            transcript_axis=crosswalk["matrix_transcript_name"],
            barcode_axis=cells["matrix_barcode"],
            manifest=manifest,
        )


@pytest.mark.external
def test_real_frozen_path_and_cell_identities_are_exact() -> None:
    paths, catalogs, path_identity = build_path_catalog(SELECTION)
    raw = yaml.safe_load(CONFIG.read_text())
    cells, split_row_identity = build_cell_lookup(
        SELECTION, raw["library_to_matrix_prefix"]
    )
    assert len(paths) == 90_672
    assert len(catalogs) == 17_706
    assert path_identity.endswith("selected_gene_catalog_ordered_matrix_paths::90672_rows")
    assert len(cells) == 217_933
    assert split_row_identity.endswith("XE_TS_barcode_to_cell_split::217933_rows")
    assert cells[("emb6", "h22", "AAACCCACACTCTGCT")] == (
        "RNA__Emb06_head_AAACCCACACTCTGCT",
        "train",
    )
