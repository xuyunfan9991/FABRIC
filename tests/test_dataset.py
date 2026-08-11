from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy import sparse

from fabric.dataset import (
    apply_centering,
    apply_dna_centering,
    cell_gene_molecule_mass,
    fit_centering_baseline,
    fit_dna_centering_baseline,
    fit_rna_state_pca,
    transform_rna_state,
    build_exact_stage_neighbors,
)


def test_observed_zero_and_missing_are_distinct_and_centered():
    values = np.array([[0.0], [2.0], [999.0]], dtype=np.float32)
    observed = np.array([[True], [True], [False]])
    weights = np.array([1.0, 3.0, 8.0])
    fit = fit_centering_baseline(
        values,
        observed,
        weights,
        minimum_valid_mass=1,
        minimum_variance=1e-8,
    )
    centered = apply_centering(values, observed, fit)
    assert fit.mean[0] == 1.5
    assert centered[0, 0] == -1.5
    assert centered[2, 0] == 0.0
    assert abs(float((centered[:, 0] * weights).sum())) < 1e-7


def test_dna_gate_multiplies_nonnegative_inputs_before_centering():
    factor = np.array([[1.0], [3.0], [5.0]])
    accessibility = np.array([[4.0], [2.0], [1.0]])
    observed = np.array([[True], [True], [False]])
    reliability = np.array([[1.0], [0.5], [1.0]])
    weights = np.array([1.0, 2.0, 10.0])
    fit = fit_dna_centering_baseline(
        factor,
        observed,
        accessibility,
        observed,
        reliability,
        weights,
        minimum_valid_mass=1,
        minimum_variance=1e-8,
    )
    # Raw products are 4 and 6; reliability-weighted train mean is 5.
    assert fit.mean[0] == 5.0
    centered = apply_dna_centering(
        factor, observed, accessibility, observed, reliability, fit
    )
    assert centered[:, 0].tolist() == [-1.0, 0.5, 0.0]
    assert abs(float((centered[:, 0] * weights).sum())) < 1e-7


def test_cell_gene_mass_is_invariant_to_ec_row_splitting():
    original = pd.DataFrame(
        {
            "cell_id": ["c0", "c1"],
            "gene_id": ["g", "g"],
            "molecule_count": [5, 4],
            "split": ["train", "train"],
        }
    )
    split = pd.DataFrame(
        {
            "cell_id": ["c0", "c0", "c1"],
            "gene_id": ["g", "g", "g"],
            "molecule_count": [2, 3, 4],
            "split": ["train", "train", "train"],
        }
    )
    mass_original = cell_gene_molecule_mass(original, informative_row_mask=[True, True])
    mass_split = cell_gene_molecule_mass(split, informative_row_mask=[True, True, True])
    pd.testing.assert_frame_equal(mass_original, mass_split)


def test_rna_state_pca_fits_train_rows_only(tmp_path):
    counts = np.array(
        [
            [8, 1, 0, 0],
            [4, 2, 1, 0],
            [2, 5, 0, 1],
            [1, 1, 6, 2],
            [3, 0, 2, 8],
            [1, 9, 1, 1],
        ],
        dtype=np.int32,
    )
    changed = counts.copy()
    changed[4:] = np.array([[1, 1, 1, 2000], [1, 2000, 1, 1]])
    first = tmp_path / "first.h5ad"
    second = tmp_path / "second.h5ad"
    ad.AnnData(sparse.csr_matrix(counts)).write_h5ad(first)
    ad.AnnData(sparse.csr_matrix(changed)).write_h5ad(second)
    fit_first = fit_rna_state_pca(
        first, [0, 1, 2, 3], n_components=2, target_sum=1000, batch_size=4
    )
    fit_second = fit_rna_state_pca(
        second, [0, 1, 2, 3], n_components=2, target_sum=1000, batch_size=4
    )
    np.testing.assert_allclose(fit_first.components, fit_second.components)
    np.testing.assert_allclose(fit_first.mean, fit_second.mean)
    assert fit_first.log_library_mean == fit_second.log_library_mean
    transformed_first = transform_rna_state(first, [4, 5], fit_first, batch_size=2)
    transformed_second = transform_rna_state(second, [4, 5], fit_second, batch_size=2)
    assert not np.allclose(transformed_first, transformed_second)


def test_exact_stage_knn_requires_and_applies_donor_eligibility():
    neighbors, status = build_exact_stage_neighbors(
        rna_cell_ids=["r0", "r1"],
        rna_embedding=np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32),
        rna_stage=["CS11", "CS12"],
        atac_cell_ids=["a0", "a1", "a2"],
        atac_embedding=np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.2]], dtype=np.float32),
        atac_stage=["CS11", "CS11", "CS12"],
        atac_donor_ids=["d0", "d1", "d2"],
        atac_donor_eligible=[False, True, True],
        k=2,
        temperature=1.0,
        device="cpu",
    )
    assert neighbors.groupby("cell_id")["neighbor_atac_cell_id"].apply(
        list
    ).to_dict() == {
        "r0": ["a1"],
        "r1": ["a2"],
    }
    assert neighbors.groupby("cell_id")["neighbor_atac_donor_id"].apply(
        list
    ).to_dict() == {
        "r0": ["d1"],
        "r1": ["d2"],
    }
    assert status["observed_atac"].all()
    with pytest.raises(ValueError, match="stage is missing"):
        build_exact_stage_neighbors(
            rna_cell_ids=["r0"],
            rna_embedding=np.zeros((1, 2), dtype=np.float32),
            rna_stage=["Unknown"],
            atac_cell_ids=["a0"],
            atac_embedding=np.zeros((1, 2), dtype=np.float32),
            atac_stage=["CS11"],
            atac_donor_ids=["d0"],
            atac_donor_eligible=[True],
            k=1,
            temperature=1.0,
            device="cpu",
        )
