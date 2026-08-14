from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fabric.cis import (
    CONTINUOUS_FEATURES,
    GEOMETRY_TRANSFORMS,
    RAW_FEATURE_ORDER,
    SEQUENCE_FEATURES,
    CISFeatureManifest,
    CISNormalizationPolicy,
    CISSequenceFeatureSpec,
    apply_cis_normalization,
    build_explicit_cis_table,
    fit_cis_normalization,
)


def _manifest() -> CISFeatureManifest:
    return CISFeatureManifest(
        reference_build="GRCh38.p14",
        strand_convention="transcript_oriented_5prime_to_3prime",
        sequence_features=tuple(
            CISSequenceFeatureSpec(
                feature_name=feature,
                scanner_name=f"frozen_{feature}_scanner",
                scanner_version="1.0",
                sequence_window=f"frozen_transcript_oriented_{feature}_window_v1",
                fixed_transform="precomputed_final_score_identity",
            )
            for feature in SEQUENCE_FEATURES
        ),
        normalization=CISNormalizationPolicy(numerical_tolerance=1.0e-8),
    )


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": ["e0", "e1", "e2", "e3", "v0", "v1"],
            "target_gene_id": ["g_train"] * 4 + ["g_val"] * 2,
            "edge_type": [
                "EXON_CONTINUATION",
                "SPLICE",
                "EXON_CONTINUATION",
                "EXON_CONTINUATION",
                "EXON_CONTINUATION",
                "EXON_CONTINUATION",
            ],
            "src_node_type": [
                "TSS",
                "donor",
                "acceptor",
                "acceptor",
                "TSS",
                "acceptor",
            ],
            "dst_node_type": [
                "donor",
                "acceptor",
                "donor",
                "PAS",
                "donor",
                "PAS",
            ],
            "span_bp": [100, 200, 300, 400, 1_000, 2_000],
            "length_bp": [90, 0, 280, 390, 900, 1_900],
            "relative_edge_pos": [0.0, 0.3, 0.6, 1.0, 0.0, 1.0],
            "annotation_confidence": [0.8] * 6,
            "edge_prior_score": [0.1, 0.2, 0.3, 0.4, 0.9, 1.1],
        }
    )


def _scores(edges: pd.DataFrame | None = None) -> pd.DataFrame:
    edges = _edges() if edges is None else edges
    numeric = {
        "edge_gc_fraction": [0.2, 0.4, 0.6, 0.8, 0.95, 0.1],
        # e1 is a true observed zero, distinct from unavailable zero on e3/v1.
        "donor_strength": [1.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        "acceptor_strength": [0.0, 2.0, 4.0, 6.0, 0.0, 8.0],
        "branchpoint_score": [0.0, 1.0, 3.0, 5.0, 0.0, 7.0],
        "polypyrimidine_tract_score": [0.0, 1.5, 2.5, 3.5, 0.0, 4.5],
        "tss_core_promoter_score": [2.0, 0.0, 0.0, 0.0, 9.0, 0.0],
        "polya_hexamer_score": [0.0, 0.0, 0.0, 4.0, 0.0, 8.0],
        "pas_downstream_u_gu_fraction": [0.0, 0.0, 0.0, 0.25, 0.0, 0.75],
    }
    if len(edges) != 6:
        raise ValueError("the score fixture expects the six canonical edge rows")
    frame = pd.DataFrame({"edge_id": edges.edge_id.astype(str)})
    src = edges.src_node_type.astype(str)
    dst = edges.dst_node_type.astype(str)
    applicability = {
        "edge_gc_fraction": np.ones(len(edges), dtype=bool),
        "donor_strength": ((src == "donor") | (dst == "donor")).to_numpy(),
        "acceptor_strength": (
            (src == "acceptor") | (dst == "acceptor")
        ).to_numpy(),
        "branchpoint_score": (
            (src == "acceptor") | (dst == "acceptor")
        ).to_numpy(),
        "polypyrimidine_tract_score": (
            (src == "acceptor") | (dst == "acceptor")
        ).to_numpy(),
        "tss_core_promoter_score": ((src == "TSS") | (dst == "TSS")).to_numpy(),
        "polya_hexamer_score": ((src == "PAS") | (dst == "PAS")).to_numpy(),
        "pas_downstream_u_gu_fraction": (
            (src == "PAS") | (dst == "PAS")
        ).to_numpy(),
    }
    for feature in SEQUENCE_FEATURES:
        frame[feature] = numeric[feature]
        frame[f"{feature}_available"] = applicability[feature]
    return frame


def test_explicit_cis_is_key_aligned_complete_and_ordered():
    edges = _edges()
    scores = _scores(edges).iloc[::-1].reset_index(drop=True)
    manifest = _manifest()
    raw = build_explicit_cis_table(edges, scores, manifest=manifest)

    assert raw.edge_ids == tuple(edges.edge_id)
    assert raw.column_names == RAW_FEATURE_ORDER
    assert raw.cis_feature_manifest_identity == manifest.identity
    frame = raw.to_frame()
    np.testing.assert_allclose(frame.edge_gc_fraction, [0.2, 0.4, 0.6, 0.8, 0.95, 0.1])
    assert frame.loc[1, "donor_strength"] == 0.0
    assert frame.loc[1, "donor_strength_available"] == 1.0
    assert frame.loc[3, "donor_strength"] == 0.0
    assert frame.loc[3, "donor_strength_available"] == 0.0
    np.testing.assert_allclose(
        frame[[name for name in RAW_FEATURE_ORDER if name.startswith("edge_type__")]].sum(axis=1),
        1.0,
    )
    np.testing.assert_allclose(frame.log1p_span_bp, np.log1p(edges.span_bp))


def test_train_normalization_counts_unique_edges_only_and_drops_constant_columns():
    edges = _edges().assign(
        cell_representation_count=[1, 10, 100, 1_000, 1, 1],
        molecule_mass=[1, 2, 4, 8, 16, 32],
        transcript_path_count=[7, 7, 70, 700, 3, 3],
    )
    manifest = _manifest()
    raw = build_explicit_cis_table(edges, _scores(_edges()), manifest=manifest)
    fit = fit_cis_normalization(
        raw,
        train_admitted_gene_ids=["g_train", "g_train", "g_train"],
        manifest=manifest,
    )
    transformed = apply_cis_normalization(raw, normalization=fit, manifest=manifest)

    assert fit.train_edge_ids == ("e0", "e1", "e2", "e3")
    assert fit.train_admitted_gene_ids == ("g_train",)
    audit = {row.feature_name: row for row in fit.statistics}
    assert audit["edge_gc_fraction"].available_unique_edge_count == 4
    assert audit["edge_gc_fraction"].mean == pytest.approx(0.5)
    assert audit["annotation_confidence"].status == "constant_cis_feature"
    assert audit["tss_core_promoter_score"].status == "constant_cis_feature"
    assert "annotation_confidence" not in transformed.column_names
    assert "tss_core_promoter_score" not in transformed.column_names
    assert "edge_type__SPLICE" in transformed.column_names
    assert "donor_strength_available" in transformed.column_names

    raw_without_weights = build_explicit_cis_table(
        _edges(), _scores(_edges()), manifest=manifest
    )
    fit_without_weights = fit_cis_normalization(
        raw_without_weights,
        train_admitted_gene_ids=["g_train"],
        manifest=manifest,
    )
    assert fit.identity == fit_without_weights.identity


def test_frozen_normalization_applies_to_held_out_edges_without_clipping():
    manifest = _manifest()
    raw = build_explicit_cis_table(_edges(), _scores(), manifest=manifest)
    fit = fit_cis_normalization(
        raw, train_admitted_gene_ids=["g_train"], manifest=manifest
    )
    normalized = apply_cis_normalization(
        raw,
        normalization=fit,
        manifest=manifest,
        expected_edge_ids=["e0", "e1", "e2", "e3", "v0", "v1"],
    ).to_frame()

    gc_scale = np.std([0.2, 0.4, 0.6, 0.8], ddof=0)
    assert normalized.loc[4, "edge_gc_fraction"] == pytest.approx(
        (0.95 - 0.5) / gc_scale
    )
    assert normalized.loc[3, "donor_strength"] == 0.0
    assert normalized.loc[3, "donor_strength_available"] == 0.0
    np.testing.assert_array_equal(
        normalized["edge_type__SPLICE"], [0, 1, 0, 0, 0, 0]
    )
    with pytest.raises(ValueError, match="graph edge axis"):
        apply_cis_normalization(
            raw,
            normalization=fit,
            manifest=manifest,
            expected_edge_ids=["e1", "e0", "e2", "e3", "v0", "v1"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["branchpoint_score"]), "missing required"),
        (lambda frame: frame.iloc[:-1].copy(), "does not exactly match"),
        (
            lambda frame: pd.concat(
                [frame, frame.iloc[[0]].assign(edge_id="extra")], ignore_index=True
            ),
            "does not exactly match",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate edge_id",
        ),
    ],
)
def test_precomputed_sequence_boundary_rejects_incomplete_or_ambiguous_ids(
    mutation, message
):
    with pytest.raises(ValueError, match=message):
        build_explicit_cis_table(
            _edges(), mutation(_scores()), manifest=_manifest()
        )


def test_structural_edge_boundary_rejects_duplicate_ids():
    edges = _edges().copy()
    edges.loc[1, "edge_id"] = "e0"
    with pytest.raises(ValueError, match="each edge_id once"):
        build_explicit_cis_table(edges, _scores(), manifest=_manifest())


@pytest.mark.parametrize(
    ("feature", "mask", "row", "replacement", "message"),
    [
        ("donor_strength", "donor_strength_available", 3, 1.0, "must be numeric zero"),
        ("donor_strength", "donor_strength_available", 1, np.nan, "non-finite"),
        ("edge_gc_fraction", "edge_gc_fraction_available", 0, 1.1, "within"),
    ],
)
def test_sequence_scores_fail_closed_on_value_semantic_violations(
    feature, mask, row, replacement, message
):
    scores = _scores()
    scores.loc[row, feature] = replacement
    with pytest.raises(ValueError, match=message):
        build_explicit_cis_table(_edges(), scores, manifest=_manifest())


def test_sequence_masks_must_be_binary_and_match_endpoint_applicability():
    scores = _scores()
    scores.loc[0, "donor_strength_available"] = False
    with pytest.raises(ValueError, match="endpoint applicability"):
        build_explicit_cis_table(_edges(), scores, manifest=_manifest())

    scores = _scores()
    scores["donor_strength_available"] = scores[
        "donor_strength_available"
    ].astype(float)
    scores.loc[0, "donor_strength_available"] = 0.5
    with pytest.raises(ValueError, match="only 0/1"):
        build_explicit_cis_table(_edges(), scores, manifest=_manifest())


def test_manifest_freezes_scanners_transforms_order_and_normalization():
    manifest = _manifest()
    record = manifest.to_record()
    assert record["reference_build"] == "GRCh38.p14"
    assert record["geometry_transforms"] == GEOMETRY_TRANSFORMS
    assert record["output_order"] == RAW_FEATURE_ORDER
    assert record["normalization"]["population"] == (
        "train_admitted_unique_structural_edges"
    )
    assert record["normalization"]["weighting"] == (
        "each_edge_once_no_cell_molecule_path_transcript_weights"
    )
    assert record["alphagenome_status"] == "DEFERRED_CIS_EXTENSION_DISABLED"

    with pytest.raises(ValueError, match="output order"):
        replace(manifest, output_order=tuple(reversed(RAW_FEATURE_ORDER)))
    with pytest.raises(ValueError, match="sequence feature specifications"):
        replace(manifest, sequence_features=manifest.sequence_features[:-1])
    with pytest.raises(ValueError, match="AlphaGenome"):
        replace(manifest, alphagenome_status="ENABLED")


def test_all_contract_continuous_fields_have_frozen_normalization_statistics():
    manifest = _manifest()
    raw = build_explicit_cis_table(_edges(), _scores(), manifest=manifest)
    fit = fit_cis_normalization(
        raw, train_admitted_gene_ids=["g_train"], manifest=manifest
    )
    assert tuple(row.feature_name for row in fit.statistics) == CONTINUOUS_FEATURES
    assert fit.model_output_order == tuple(
        name
        for name in RAW_FEATURE_ORDER
        if name not in {
            row.feature_name
            for row in fit.statistics
            if row.status.startswith("constant_cis_feature")
        }
    )


def test_forged_raw_table_cannot_bypass_fit_or_apply_semantic_validation():
    manifest = _manifest()
    raw = build_explicit_cis_table(_edges(), _scores(), manifest=manifest)
    valid_fit = fit_cis_normalization(
        raw, train_admitted_gene_ids=["g_train"], manifest=manifest
    )
    column_index = {name: index for index, name in enumerate(raw.column_names)}

    inapplicable_nonzero = raw.values.copy()
    inapplicable_nonzero[3, column_index["donor_strength"]] = 9.0
    nonbinary_mask = raw.values.copy()
    nonbinary_mask[0, column_index["donor_strength_available"]] = 0.5
    for forged_values, message in (
        (inapplicable_nonzero, "must be numeric zero"),
        (nonbinary_mask, "must contain only 0/1"),
    ):
        forged = replace(raw, values=forged_values)
        with pytest.raises(ValueError, match=message):
            fit_cis_normalization(
                forged,
                train_admitted_gene_ids=["g_train"],
                manifest=manifest,
            )
        with pytest.raises(ValueError, match=message):
            apply_cis_normalization(
                forged, normalization=valid_fit, manifest=manifest
            )
