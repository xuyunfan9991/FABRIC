from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import torch
import pytest

from fabric.evaluate import (
    FactorActivityContext,
    GeneNullContext,
    _train_likelihood_informative_cell_mass,
    correction_scale_diagnostic,
    permute_factor_activity_within_strata,
    rebuild_stage_system_factor_null,
    trained_scale_diagnostics,
)
from fabric.model import (
    AlternativeBatch,
    AugmentedPathReadout,
    EventOutput,
    EventScorer,
    StateScorer,
)
from fabric.train import (
    HierarchyResult,
    NORMALIZED_SOURCE_ROLES,
    PreparedDataset,
    VARIANTS,
    VariantModules,
    _freeze_cis_outputs,
    _loss_for_rows,
    assert_full7198_ready,
    fit_b0_path_logits,
    frozen_toy_likelihood_parity_error,
    load_config,
    make_toy_genes,
    prepare_dataset_identity,
    preparation_values_from_config,
    train_paired_seeds,
    train_hierarchy,
)
from fabric.model import EdgeGraphGPS


def _config(epochs: int = 8):
    return {
        "model": {
            "cis_hidden_dim": 12,
            "cis_heads": 3,
            "state_rank": 3,
            "dna_rank": 3,
            "rna_rank": 3,
            "path_length_prior_weight": 0.0,
        },
        "training": {
            "seed": 17,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "max_epochs": epochs,
            "ec_batch_rows": 64,
            "variants": list(VARIANTS),
            "formal_full7198_authorized": False,
        },
        "admission": {
            "minimum_b0_validation_improvement": 0.0,
            "real_fixture_directory": "tests/fixtures/real",
        },
    }


def test_toy_hierarchy_runs_all_variants_and_children_share_exact_parents():
    result = train_hierarchy(make_toy_genes(), _config(), device="cpu")
    assert result.admission["passed"]
    assert set(result.metrics["variant"]) == set(VARIANTS)
    assert np.isfinite(result.metrics["compatible_path_nll"]).all()
    assert result.metrics["screening_evidence_only"].all()
    for _, history in result.history.groupby("variant", sort=False):
        assert history["train_nll"].min() < history.iloc[0]["train_nll"]

    cis = result.modules["cis"].cis
    assert all(result.modules[variant].cis is cis for variant in VARIANTS)
    state_parent = result.modules["state"].state.state_dict()
    for variant in ("state_dna", "state_rna", "state_dna_rna"):
        child_state = result.modules[variant].state.state_dict()
        for key in state_parent:
            torch.testing.assert_close(
                child_state[key], state_parent[key], atol=0, rtol=0
            )
    # Full owns fresh jointly trained scorers rather than either trained child.
    assert result.modules["state_dna_rna"].dna is not result.modules["state_dna"].dna
    assert result.modules["state_dna_rna"].rna is not result.modules["state_rna"].rna


def test_frozen_toy_admission_covers_multi_path_sets_and_centered_inputs():
    assert frozen_toy_likelihood_parity_error() <= 1.0e-15
    gene = make_toy_genes()[0]
    for values in (gene.state_features, gene.dna_gate, gene.rna_gate):
        torch.testing.assert_close(
            values[:20].sum(dim=0),
            torch.zeros(values.shape[1]),
            atol=2.0e-6,
            rtol=0.0,
        )


def test_toy_uses_real_16d_processing_features_and_legal_path_adjacency():
    gene = make_toy_genes()[0]
    assert gene.graph.edge_features.shape == (7, 16)
    observed = set(map(tuple, gene.graph.local_edge_index.T.tolist()))
    expected = {
        pair
        for path in ((0, 1, 2, 3, 6), (0, 1, 4, 5, 6))
        for left, right in zip(path[:-1], path[1:])
        for pair in ((left, right), (right, left))
    }
    assert observed == expected


def test_b0_uses_only_train_likelihood_informative_ec_with_laplace_one():
    gene = make_toy_genes()[0]
    logits = fit_b0_path_logits(gene)
    counts = torch.ones(2, dtype=torch.float64)
    for row in range(20):
        compatible = gene.compatible_path_indices[row][gene.compatible_path_mask[row]]
        counts[compatible] += gene.molecule_count[row].double() / len(compatible)
    expected = (counts / counts.sum()).log().float()
    torch.testing.assert_close(logits, expected)


def test_all_dynamic_branches_are_zero_on_ineligible_choice():
    gene = replace(
        make_toy_genes()[0], alternative_eligible=torch.zeros(2, dtype=torch.bool)
    )
    cis = EdgeGraphGPS(gene.graph.edge_features.shape[1], 8, 2)
    frozen = _freeze_cis_outputs(cis, [gene])
    alternative_dim = frozen[gene.gene_id][1].h_base.shape[1]
    state = StateScorer(4, alternative_dim, 2)
    dna = EventScorer(6, alternative_dim, 2)
    rna = EventScorer(5, alternative_dim, 2)
    for module in (state, dna, rna):
        torch.nn.init.constant_(module.U.weight, 0.2)
        torch.nn.init.constant_(module.V.weight, 0.3)
    rows = torch.arange(6)
    readout = AugmentedPathReadout(0.0)
    parent = _loss_for_rows(
        gene,
        rows,
        VariantModules(cis, None, None, None),
        frozen[gene.gene_id],
        readout,
    )
    dynamic = _loss_for_rows(
        gene,
        rows,
        VariantModules(cis, state, dna, rna),
        frozen[gene.gene_id],
        readout,
    )
    torch.testing.assert_close(dynamic.per_row_nll, parent.per_row_nll, atol=0, rtol=0)


def test_fixed_stage_system_permutation_preserves_each_stratum_joint_rows():
    activity = np.arange(30).reshape(10, 3)
    stage = ["CS11"] * 5 + ["CS12"] * 5
    system = ["A", "A", "B", "B", "B"] * 2
    permuted, source = permute_factor_activity_within_strata(
        activity, stage, system, seed=41
    )
    for key in sorted(set(zip(stage, system))):
        rows = [index for index, value in enumerate(zip(stage, system)) if value == key]
        assert sorted(map(tuple, permuted[rows])) == sorted(map(tuple, activity[rows]))
        assert {(stage[index], system[index]) for index in source[rows]} == {key}


def test_formal_null_rebuilds_train_centered_rna_and_dna_gates():
    original = make_toy_genes()[0]
    gene = replace(
        original,
        identifiable_row_mask=torch.zeros_like(original.identifiable_row_mask),
    )
    cell_count = len(gene.cell_ids)
    activity = np.stack(
        [np.linspace(0.2, 2.0, cell_count), np.linspace(2.5, 0.4, cell_count)],
        axis=1,
    )
    factor_context = FactorActivityContext(
        cell_ids=gene.cell_ids,
        factor_ids=("f0", "f1"),
        activity=activity,
        observed=np.ones_like(activity, dtype=bool),
        stage=tuple(["CS11"] * 15 + ["CS12"] * 15),
        developmental_system=tuple(
            ["neural"] * 10 + ["mesoderm"] * 10 + ["neural"] * 10
        ),
    )
    accessibility = np.stack(
        [np.linspace(0.5, 1.5, cell_count), np.linspace(1.7, 0.7, cell_count)],
        axis=1,
    )
    context = GeneNullContext(
        gene_id=gene.gene_id,
        cell_ids=gene.cell_ids,
        dna_event_factor_index=np.array([0, 1]),
        rna_event_factor_index=np.array([0, 1]),
        dna_accessibility=accessibility,
        dna_accessibility_observed=np.ones_like(accessibility, dtype=bool),
        dna_reliability=np.ones_like(accessibility),
    )
    rebuilt, source = rebuild_stage_system_factor_null(
        [gene],
        factor_context,
        [context],
        seed=11,
        minimum_valid_molecule_mass=1,
        minimum_weighted_variance=1e-10,
    )
    train = np.arange(20)
    assert torch.count_nonzero(rebuilt[0].rna_gate[train]) > 0
    assert torch.count_nonzero(rebuilt[0].dna_gate[train]) > 0
    np.testing.assert_allclose(rebuilt[0].rna_gate[train].sum(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(rebuilt[0].dna_gate[train].sum(0), 0.0, atol=3e-6)
    for index, source_row in enumerate(source):
        assert factor_context.stage[index] == factor_context.stage[source_row]
        assert (
            factor_context.developmental_system[index]
            == factor_context.developmental_system[source_row]
        )


def test_null_centering_mass_uses_only_train_likelihood_informative_ec():
    gene = make_toy_genes()[0]
    row_count = len(gene.split)
    compatible = torch.full((row_count, 2), -1, dtype=torch.long)
    compatible[:, 0] = gene.compatible_path_indices[:, 0]
    mask = torch.zeros_like(compatible, dtype=torch.bool)
    mask[:, 0] = True
    # A train all-path row carries no path-likelihood information and must not
    # enter a gate baseline even though it has positive molecule mass.
    compatible[0] = torch.tensor([0, 1])
    mask[0] = True
    molecule_count = torch.arange(1, row_count + 1, dtype=torch.float32)
    gene = replace(
        gene,
        compatible_path_indices=compatible,
        compatible_path_mask=mask,
        molecule_count=molecule_count,
        identifiable_row_mask=torch.zeros(row_count, dtype=torch.bool),
    )

    observed = _train_likelihood_informative_cell_mass(gene)
    expected = np.zeros(row_count, dtype=np.float64)
    expected[1:20] = np.arange(2, 21, dtype=np.float64)
    np.testing.assert_array_equal(observed, expected)


def test_real_fixture_config_loads_and_full_config_refuses_launch():
    real = load_config("configs/fabric_v1_real_fixture.yaml")
    assert tuple(real["training"]["variants"]) == VARIANTS
    full = load_config("configs/fabric_v1_full7198.yaml")
    with pytest.raises(RuntimeError, match="not authorized"):
        assert_full7198_ready(full)


def test_formal_gate_rejects_bool_only_prepared_identity():
    gene = make_toy_genes()[0]
    fake = PreparedDataset(
        genes=(gene,),
        target_gene_ids=(gene.gene_id,),
        graph_generation=gene.graph_generation,
        split_source=gene.split_source,
        factor_mapping_reviewed=True,
    )
    config = _config()
    config["target_gene_count"] = 7_198
    config["training"]["formal_full7198_authorized"] = True
    with pytest.raises(RuntimeError, match="normalized source path identity"):
        assert_full7198_ready(config, fake)


def test_formal_gate_compares_prepared_normalized_source_paths(monkeypatch, tmp_path):
    expected = {role: tmp_path / f"expected_{role}" for role in NORMALIZED_SOURCE_ROLES}
    prepared_paths = dict(expected)
    prepared_paths["rna_counts"] = tmp_path / "wrong_rna_counts"
    reviewed_mapping = tmp_path / "reviewed_factor_mapping.tsv"
    reviewed_mapping.touch()
    donor_eligibility = tmp_path / "atac_donor_eligibility.tsv"
    donor_eligibility.touch()
    peak_support = tmp_path / "peak_support.tsv"
    peak_support.touch()
    config = load_config("configs/fabric_v1_toy.yaml")
    config.update(
        target_gene_count=7_198,
        external_inputs="unused-by-test",
        factor_identity={"reviewed_mapping": str(reviewed_mapping)},
    )
    config["data"]["atac_neighbors"]["donor_eligibility_path"] = str(donor_eligibility)
    config["motifs"]["peak_support_path"] = str(peak_support)
    config["training"]["formal_full7198_authorized"] = True
    prepared = prepare_dataset_identity(
        make_toy_genes(),
        factor_mapping_reviewed=True,
        normalized_source_paths=prepared_paths,
        reviewed_factor_mapping=reviewed_mapping,
        atac_donor_eligibility_source=donor_eligibility,
        peak_support_source=peak_support,
        preparation_config_source=tmp_path / "preparation.yaml",
        preparation_values=preparation_values_from_config(config),
    )

    class ExternalInputs:
        def path(self, role):
            return expected[role]

    monkeypatch.setattr(
        "fabric.annotation.load_external_inputs", lambda _: ExternalInputs()
    )
    with pytest.raises(RuntimeError, match="rna_counts"):
        assert_full7198_ready(config, prepared)


def test_formal_gate_rejects_changed_preparation_value(monkeypatch, tmp_path):
    expected = {role: tmp_path / f"expected_{role}" for role in NORMALIZED_SOURCE_ROLES}
    reviewed_mapping = tmp_path / "reviewed_factor_mapping.tsv"
    donor_eligibility = tmp_path / "atac_donor_eligibility.tsv"
    peak_support = tmp_path / "peak_support.tsv"
    for path in (reviewed_mapping, donor_eligibility, peak_support):
        path.touch()
    config = load_config("configs/fabric_v1_toy.yaml")
    config.update(
        target_gene_count=7_198,
        external_inputs="unused-by-test",
        factor_identity={"reviewed_mapping": str(reviewed_mapping)},
    )
    config["data"]["atac_neighbors"]["donor_eligibility_path"] = str(donor_eligibility)
    config["motifs"]["peak_support_path"] = str(peak_support)
    config["training"]["formal_full7198_authorized"] = True
    prepared = prepare_dataset_identity(
        make_toy_genes(),
        factor_mapping_reviewed=True,
        normalized_source_paths=expected,
        reviewed_factor_mapping=reviewed_mapping,
        atac_donor_eligibility_source=donor_eligibility,
        peak_support_source=peak_support,
        preparation_config_source=tmp_path / "preparation.yaml",
        preparation_values=preparation_values_from_config(config),
    )
    config["data"]["target_sum_rna"] = 20_000.0

    class ExternalInputs:
        def path(self, role):
            return expected[role]

    monkeypatch.setattr(
        "fabric.annotation.load_external_inputs", lambda _: ExternalInputs()
    )
    with pytest.raises(RuntimeError, match="preparation values differ"):
        assert_full7198_ready(config, prepared)


def test_paired_seed_driver_runs_independent_fixed_hierarchies(tmp_path):
    config = _config(epochs=2)
    config["training"].pop("seed")
    config["training"]["seeds"] = [101, 202]
    config["admission"]["minimum_b0_validation_improvement"] = -1.0e9
    results = train_paired_seeds(
        make_toy_genes(), config, device="cpu", run_dir=tmp_path
    )
    assert set(results) == {101, 202}
    assert (tmp_path / "paired_seed_metrics.tsv").exists()
    assert (tmp_path / "seed_101" / "input_identity.json").exists()
    assert results[101].modules["cis"].cis is not results[202].modules["cis"].cis
    assert all(
        result.metrics["screening_evidence_only"].all() for result in results.values()
    )
    primary = pd.read_csv(
        tmp_path / "seed_101" / "evaluation" / "primary_metrics.tsv", sep="\t"
    )
    assert primary["screening_evidence_only"].all()


def test_authorized_full7198_cannot_label_toy_results_as_formal(tmp_path):
    config = _config(epochs=1)
    config["target_gene_count"] = 7_198
    config["training"].pop("seed")
    config["training"]["seeds"] = [303, 404]
    config["training"]["formal_full7198_authorized"] = True
    config["admission"]["minimum_b0_validation_improvement"] = -1.0e9
    with pytest.raises(RuntimeError, match="validated PreparedDataset"):
        train_paired_seeds(
            make_toy_genes(),
            config,
            device="cpu",
            run_dir=tmp_path,
        )
    assert not any(tmp_path.iterdir())


def test_scale_diagnostic_preserves_original_sparse_choice_indices():
    correction = np.asarray(
        [
            [1.0, -1.0, 3.0, -3.0],
            [1.0, -1.0, 3.0, -3.0],
        ]
    )
    values, correlations = correction_scale_diagnostic(
        correction,
        [0, 0, 2, 2],
        event_count=[1.0, 99.0, 2.0],
        alternative_span=[10.0, 1_000.0, 30.0],
        cap_saturated=[0.0, 1.0, 1.0],
        audited_choice_indices=[0, 2],
    )

    assert values["choice_index"].tolist() == [0, 2]
    np.testing.assert_allclose(values["rms_delta"], [1.0, 3.0])
    np.testing.assert_allclose(values["alternative_span"], [10.0, 30.0])
    span_correlation = correlations.set_index("covariate").loc[
        "alternative_span", "spearman_r"
    ]
    assert span_correlation == pytest.approx(1.0)


class _FixedEventCorrection(torch.nn.Module):
    def forward(self, batch, alternatives):
        correction = batch.gate.new_tensor([1.0, -1.0, 0.0, 0.0, 3.0, -3.0])
        correction = correction.expand(batch.gate.shape[0], -1)
        sensitivity = batch.features.new_zeros(
            (batch.features.shape[0], alternatives.h_base.shape[0])
        )
        return EventOutput(sensitivity, sensitivity, correction)


def _three_choice_scale_gene():
    gene = make_toy_genes()[0]
    alternatives = AlternativeBatch(
        edge_index=gene.alternatives.edge_index.repeat(3, 1),
        edge_mask=gene.alternatives.edge_mask.repeat(3, 1),
        choice_index=torch.tensor([0, 0, 1, 1, 2, 2]),
        scope_index=gene.alternatives.scope_index.repeat(3),
    )
    event_choices = torch.tensor([0, 1, 1, 2, 2, 2])
    relation = torch.zeros((len(event_choices), 6))
    for event_index, choice in enumerate(event_choices):
        relation[event_index, 2 * choice : 2 * choice + 2] = torch.tensor([1.0, -1.0])
    return replace(
        gene,
        alternatives=alternatives,
        alternative_eligible=torch.tensor([True, True, False, False, True, True]),
        dna_event_features=gene.dna_event_features[:1].repeat(6, 1),
        dna_event_relation=relation,
        dna_event_choice_index=event_choices,
        dna_gate=gene.dna_gate[:, :1].repeat(1, 6),
        dna_event_ids=tuple(f"dna_{index}" for index in range(6)),
        dna_event_factor_ids=tuple(f"factor_{index}" for index in range(6)),
        dna_event_peak_ids=tuple(f"peak_{index}" for index in range(6)),
        rna_event_features=gene.rna_event_features[:1].repeat(6, 1),
        rna_event_relation=relation,
        rna_event_choice_index=event_choices,
        rna_gate=gene.rna_gate[:, :1].repeat(1, 6),
        rna_event_ids=tuple(f"rna_{index}" for index in range(6)),
        rna_event_factor_ids=tuple(f"factor_{index}" for index in range(6)),
        alternative_span=torch.tensor([10.0, 1_000.0, 30.0]),
        dna_candidate_event_count=torch.tensor([1.0, 2.0, 3.0]),
        dna_selected_event_count=torch.tensor([1.0, 2.0, 3.0]),
        dna_cap_saturated=torch.tensor([0.0, 0.0, 1.0]),
        dna_boundary_rank_motif_score=torch.tensor([0.9, 0.8, 0.7]),
        rna_candidate_event_count=torch.tensor([1.0, 2.0, 3.0]),
        rna_selected_event_count=torch.tensor([1.0, 2.0, 3.0]),
        rna_cap_saturated=torch.tensor([1.0, 0.0, 0.0]),
        rna_boundary_rank_motif_score=torch.tensor([0.9, 0.8, 0.7]),
    )


def _fixed_scale_result(gene):
    cis = EdgeGraphGPS(gene.graph.edge_features.shape[1], 12, 3)
    event = _FixedEventCorrection()
    return HierarchyResult(
        modules={
            "cis": VariantModules(cis, None, None, None),
            "state_dna": VariantModules(cis, None, event, None),
            "state_rna": VariantModules(cis, None, None, event),
            "state_dna_rna": VariantModules(cis, None, event, event),
        },
        metrics=pd.DataFrame(),
        admission={},
        history=pd.DataFrame(),
    )


def test_trained_scale_diagnostics_excludes_ineligible_choices_per_modality():
    gene = _three_choice_scale_gene()
    diagnostic, correlations = trained_scale_diagnostics(
        _fixed_scale_result(gene), [gene], device="cpu", cell_batch_size=7
    )

    assert set(diagnostic["choice_index"]) == {0, 2}
    assert len(diagnostic) == 8  # two eligible choices in four variant/modalities
    assert (diagnostic.groupby(["variant", "modality"]).size() == 2).all()
    np.testing.assert_allclose(
        diagnostic.sort_values(["variant", "modality", "choice_index"])[
            "rms_delta"
        ].to_numpy(),
        np.tile([1.0, 3.0], 4),
    )
    span = correlations.loc[correlations["covariate"] == "alternative_span"]
    np.testing.assert_allclose(span["spearman_r"], 1.0)
    dna = diagnostic.loc[diagnostic["modality"] == "DNA"].sort_values(
        ["variant", "choice_index"]
    )
    rna = diagnostic.loc[diagnostic["modality"] == "RNA"].sort_values(
        ["variant", "choice_index"]
    )
    assert dna["cap_saturated"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert rna["cap_saturated"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_trained_scale_diagnostics_rejects_mixed_eligibility_within_choice():
    gene = _three_choice_scale_gene()
    gene = replace(
        gene,
        alternative_eligible=torch.tensor([True, False, False, False, True, True]),
    )
    with pytest.raises(ValueError, match="eligibility differs within choice 0"):
        trained_scale_diagnostics(_fixed_scale_result(gene), [gene], device="cpu")


def test_trained_scale_diagnostics_checks_selected_event_count_identity():
    gene = _three_choice_scale_gene()
    gene = replace(gene, dna_selected_event_count=torch.tensor([1.0, 9.0, 3.0]))
    with pytest.raises(ValueError, match="DNA selected event count differs"):
        trained_scale_diagnostics(_fixed_scale_result(gene), [gene], device="cpu")


def test_trained_scale_diagnostics_skips_graph_only_gene_without_cells():
    gene = _three_choice_scale_gene()
    graph_only = replace(
        gene,
        state_features=gene.state_features[:0],
        dna_gate=gene.dna_gate[:0],
        rna_gate=gene.rna_gate[:0],
        compatible_path_indices=gene.compatible_path_indices[:0],
        compatible_path_mask=gene.compatible_path_mask[:0],
        row_cell_index=gene.row_cell_index[:0],
        molecule_count=gene.molecule_count[:0],
        split=(),
        identifiable_row_mask=gene.identifiable_row_mask[:0],
        cell_ids=(),
    )
    diagnostic, correlations = trained_scale_diagnostics(
        _fixed_scale_result(graph_only), [graph_only], device="cpu"
    )
    assert diagnostic.empty
    assert correlations.empty
