from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fabric.dataset import (
    ATACMappingContext,
    GateValues,
    build_gate_collinearity_audit,
    build_raw_gate_signals,
    compute_activity_entities,
    fit_gate_admission,
    map_atac_accessibility,
    assess_atac_mapping,
    transform_gates,
)


def test_group_activity_sums_raw_counts_before_one_cp10k_log1p():
    activity = compute_activity_entities(
        sparse.csr_matrix([[1, 3, 6], [0, 0, 0], [2, 0, 8]]),
        cell_ids=["c0", "c1", "c2"],
        frozen_gene_axis=["A", "B", "other"],
        entity_table=pd.DataFrame(
            {
                "activity_entity_id": ["group:A:B"],
                "activity_gene_ids": [["A", "B"]],
                "source_valid": [True],
            }
        ),
    )
    np.testing.assert_allclose(
        activity.values[:, 0], [np.log1p(4000), 0, np.log1p(2000)]
    )
    assert activity.observed[:, 0].tolist() == [True, False, True]


def _neighbors(distances=(1.0, 1.0), weights=(0.5, 0.5)):
    return pd.DataFrame(
        {
            "cell_id": ["c", "c"],
            "neighbor_atac_cell_id": ["a0", "a1"],
            "neighbor_weight": list(weights),
            "distance": list(distances),
            "rna_qc_pass": [True, True],
            "atac_qc_pass": [True, True],
            "pairing_valid": [True, True],
            "neighborhood_consistency_status": ["pass", "pass"],
        }
    )


def test_atac_mapping_uses_absolute_qc_not_support_metadata():
    valid = assess_atac_mapping(
        _neighbors(), target_cell_ids=["c"], expected_k=2, maximum_distance=2
    )
    distant = assess_atac_mapping(
        _neighbors(distances=(5, 5)),
        target_cell_ids=["c"],
        expected_k=2,
        maximum_distance=2,
    )
    assert bool(valid.iloc[0].mapping_valid)
    assert not bool(distant.iloc[0].mapping_valid)
    assert distant.iloc[0].ess_atac == pytest.approx(2.0)


def test_atac_normalizes_each_neighbor_then_maps_and_dna_is_product_before_z():
    neighbors = _neighbors()
    audit = assess_atac_mapping(
        neighbors, target_cell_ids=["c"], expected_k=2, maximum_distance=2
    )
    atac = map_atac_accessibility(
        [[1, 9], [9, 1]],
        atac_cell_ids=["a0", "a1"],
        peak_ids=["p", "q"],
        target_cell_ids=["c"],
        neighbors=neighbors,
        mapping_audit=audit,
    )
    expected_peak = 0.5 * np.log1p(1000) + 0.5 * np.log1p(9000)
    assert float(atac.accessibility[0, 0]) == pytest.approx(expected_peak)
    activity = compute_activity_entities(
        [[2, 8]],
        cell_ids=["c"],
        frozen_gene_axis=["F", "other"],
        entity_table=pd.DataFrame(
            {
                "activity_entity_id": ["F"],
                "activity_gene_ids": [["F"]],
                "source_valid": [True],
            }
        ),
    )
    raw = build_raw_gate_signals(
        pd.DataFrame(
            {
                "gate_key_id": ["gate"],
                "target_gene_id": ["g"],
                "channel": ["DNA"],
                "activity_entity_id": ["F"],
                "peak_id": ["p"],
            }
        ),
        activity=activity,
        atac=atac,
    )
    assert raw.raw[0, 0] == pytest.approx(activity.values[0, 0] * expected_peak)


def test_gate_train_only_admission_and_support_flags():
    activity = compute_activity_entities(
        [[0, 10], [2, 8], [10, 0]],
        cell_ids=["train0", "train1", "heldout"],
        frozen_gene_axis=["F", "other"],
        entity_table=pd.DataFrame(
            {
                "activity_entity_id": ["F"],
                "activity_gene_ids": [["F"]],
                "source_valid": [True],
            }
        ),
    )
    atac = ATACMappingContext(
        cell_ids=activity.cell_ids,
        peak_ids=("p",),
        accessibility=sparse.csr_matrix(np.ones((3, 1))),
        mapping_valid=np.ones(3, dtype=bool),
        diagnostics=pd.DataFrame(),
    )
    keys = pd.DataFrame(
        {
            "gate_key_id": ["gate"],
            "target_gene_id": ["g"],
            "channel": ["RNA"],
            "activity_entity_id": ["F"],
            "peak_id": [None],
        }
    )
    raw = build_raw_gate_signals(keys, activity=activity, atac=atac)
    manifest = fit_gate_admission(
        raw,
        keys,
        train_mask=[True, True, False],
        informative_molecule_mass=np.ones((3, 1)),
        thresholds_by_channel={
            "RNA": {
                "minimum_valid_cells": 2,
                "minimum_effective_cells": 2,
                "minimum_informative_molecules": 2,
                "minimum_standard_deviation": 1e-8,
            }
        },
        support_quantiles=(0, 1),
    )
    gates = transform_gates(raw, manifest)
    assert bool(manifest.iloc[0].gate_key_active)
    assert gates.gate[0, 0] < 0 < gates.gate[1, 0]
    assert bool(gates.out_of_train_range[2, 0])


def test_collinearity_uses_only_active_keys_and_keeps_exact_negative_one():
    gates = GateValues(
        cell_ids=("c0", "c1", "c2"),
        gate_key_ids=("a", "b", "inactive"),
        raw=np.zeros((3, 3)),
        standardized_residual=np.zeros((3, 3)),
        gate=np.array([[-1, 1, 0], [0, 0, 0], [1, -1, 0]], dtype=np.float32),
        observed=np.ones((3, 3), dtype=bool),
        out_of_train_range=np.zeros((3, 3), dtype=bool),
        out_of_train_quantile_support=np.zeros((3, 3), dtype=bool),
    )
    audit = build_gate_collinearity_audit(
        gates,
        pd.DataFrame(
            {
                "gate_key_id": ["a", "b", "inactive"],
                "target_gene_id": ["g", "g", "g"],
                "channel": ["RNA", "DNA", "Open"],
                "gate_key_active": [True, True, False],
            }
        ),
        train_mask=[True, True, True],
        informative_molecule_mass_by_gene=pd.DataFrame(
            {
                "cell_id": ["c0", "c1", "c2"],
                "target_gene_id": ["g"] * 3,
                "informative_molecule_mass": [1, 1, 1],
            }
        ),
        minimum_joint_effective_cells=3,
        absolute_correlation_threshold=0.99,
    )
    assert len(audit.pairs) == 1
    assert audit.pairs.iloc[0].weighted_pearson_correlation == pytest.approx(-1)
    assert audit.correlated_sets.iloc[0].member_gate_key_ids == ["a", "b"]


def test_identical_distinct_atac_copies_preserve_mapped_value_and_duplicate_id_fails():
    one = pd.DataFrame(
        {
            "cell_id": ["c"],
            "neighbor_atac_cell_id": ["a0"],
            "neighbor_weight": [1.0],
            "distance": [1.0],
            "rna_qc_pass": [True],
            "atac_qc_pass": [True],
            "pairing_valid": [True],
            "neighborhood_consistency_status": ["not_estimable"],
        }
    )
    copies = pd.DataFrame(
        {
            "cell_id": ["c", "c", "c"],
            "neighbor_atac_cell_id": ["a0", "a1", "a2"],
            "neighbor_weight": [1 / 3] * 3,
            "distance": [1.0] * 3,
            "rna_qc_pass": [True] * 3,
            "atac_qc_pass": [True] * 3,
            "pairing_valid": [True] * 3,
            "neighborhood_consistency_status": ["pass"] * 3,
        }
    )
    one_audit = assess_atac_mapping(
        one, target_cell_ids=["c"], expected_k=1, maximum_distance=2
    )
    copies_audit = assess_atac_mapping(
        copies, target_cell_ids=["c"], expected_k=3, maximum_distance=2
    )
    mapped_one = map_atac_accessibility(
        [[2, 8]],
        atac_cell_ids=["a0"],
        peak_ids=["p", "q"],
        target_cell_ids=["c"],
        neighbors=one,
        mapping_audit=one_audit,
    )
    mapped_copies = map_atac_accessibility(
        [[2, 8], [2, 8], [2, 8]],
        atac_cell_ids=["a0", "a1", "a2"],
        peak_ids=["p", "q"],
        target_cell_ids=["c"],
        neighbors=copies,
        mapping_audit=copies_audit,
    )
    np.testing.assert_allclose(
        mapped_one.accessibility.toarray(), mapped_copies.accessibility.toarray()
    )
    duplicate_id = copies.assign(neighbor_atac_cell_id=["a0", "a0", "a2"])
    with pytest.raises(ValueError, match="unique per target"):
        map_atac_accessibility(
            [[2, 8], [2, 8], [2, 8]],
            atac_cell_ids=["a0", "a1", "a2"],
            peak_ids=["p", "q"],
            target_cell_ids=["c"],
            neighbors=duplicate_id,
            mapping_audit=copies_audit,
        )


def test_gate_thresholds_fail_on_nan_negative_or_fractional_count():
    activity = compute_activity_entities(
        [[1, 9], [2, 8]],
        cell_ids=["c0", "c1"],
        frozen_gene_axis=["F", "other"],
        entity_table=pd.DataFrame(
            {
                "activity_entity_id": ["F"],
                "activity_gene_ids": [["F"]],
                "source_valid": [True],
            }
        ),
    )
    atac = ATACMappingContext(
        cell_ids=activity.cell_ids,
        peak_ids=("p",),
        accessibility=sparse.csr_matrix(np.ones((2, 1))),
        mapping_valid=np.ones(2, dtype=bool),
        diagnostics=pd.DataFrame(),
    )
    keys = pd.DataFrame(
        {
            "gate_key_id": ["gate"],
            "target_gene_id": ["g"],
            "channel": ["RNA"],
            "activity_entity_id": ["F"],
            "peak_id": [None],
        }
    )
    raw = build_raw_gate_signals(keys, activity=activity, atac=atac)
    common = {
        "minimum_effective_cells": 1,
        "minimum_informative_molecules": 1,
        "minimum_standard_deviation": 0,
    }
    for invalid in (np.nan, -1, 1.5):
        with pytest.raises(ValueError, match="finite and non-negative|integer count"):
            fit_gate_admission(
                raw,
                keys,
                train_mask=[True, True],
                informative_molecule_mass=np.ones((2, 1)),
                thresholds_by_channel={
                    "RNA": {**common, "minimum_valid_cells": invalid}
                },
            )
