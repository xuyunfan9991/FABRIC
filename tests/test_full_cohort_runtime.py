from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from fabric.evaluate import OntMatrixKlTarget
from fabric.real_dataset import compile_gene_graph_tables
from fabric.profile_real import (
    _fit_conservative_nonnegative_cost_model,
    _project_epoch_seconds,
    _record_batch_cell_limit,
    _select_profile_records,
)
from fabric.train import (
    BackedPreparedDataset,
    FULL_COHORT_SCOPE,
    RUN_CONDITIONS,
    assert_execution_admitted,
    evaluate_final_test,
    load_config,
    make_toy_genes,
    rows_for_split,
    resolve_run_config,
    sample_train_gene_cells_for_epoch,
    split_informative_molecule_mass,
    train_run,
    training_manifest_from_config,
)
from fabric.train import _iter_gene_order


def test_full_cohort_config_is_single_run_and_test_blind():
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    assert config["execution"] == {
        "scope": FULL_COHORT_SCOPE,
        "training_authorized": False,
        "final_test_authorized": False,
    }
    manifest = training_manifest_from_config(config, seed=1103, condition="full")
    assert manifest.seed == 1103
    assert manifest.condition == "full"
    assert manifest.learning_rate == 5.0e-5
    assert manifest.lr_scheduler_name == "reduce_on_plateau"
    assert manifest.lr_scheduler_factor == 0.5
    assert manifest.lr_scheduler_patience == 1
    assert manifest.lr_scheduler_min_lr == 1.0e-5
    assert manifest.gradient_clip_norm == 1.0
    assert (manifest.lambda_base, manifest.lambda_int) == (0.001, 0.01)
    assert manifest.max_epochs == 30
    assert manifest.early_stopping_patience == 5
    assert manifest.max_train_gene_cells_per_gene_per_epoch == 512
    assert manifest.resample_train_gene_cells_each_epoch is True
    assert manifest.selected_gene_cell_ec_rows == "all_informative_rows"
    assert manifest.sampling_estimator == ("horvitz_thompson_full_train_molecule_total")
    assert manifest.optimizer_step_unit == "train_positive_gene"
    assert manifest.gene_microbatch_gradient_accumulation is True
    assert manifest.batch_policy == "gene_shape_adaptive_v1"
    assert manifest.target_gpu_allocated_bytes == 20 << 30
    assert manifest.max_cells_per_gpu_batch == 32_768
    assert manifest.prefetch_backed_gene_shards is True
    assert manifest.compute_precision == "float32_highest"
    assert "max_attention_elements_per_batch" not in config["resources"]
    with pytest.raises(RuntimeError, match="training is not authorized"):
        assert_execution_admitted(config)

    genes = make_toy_genes()
    fixture = load_config("configs/fabric_v2_toy.yaml")
    fixture["training"]["max_epochs"] = 1
    fixture["training"]["early_stopping_patience"] = 1
    result = train_run(genes, fixture, seed=101, condition="full", device="cpu")
    with pytest.raises(RuntimeError, match="not authorized"):
        evaluate_final_test(
            genes,
            result,
            config,
            checkpoints_frozen=True,
            report_rules_frozen=True,
        )


def test_real_validation_ont_kl_target_is_g_fit_complete_and_test_absent():
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    root = config["monitor"]["target_root"]
    target = OntMatrixKlTarget.load(root)
    manifest = json.loads((Path(root) / "OntMatrixKlTargetManifest.json").read_text())
    assert target.counts.shape == (90_361, 21_788)
    assert target.counts.nnz == 11_785_211
    assert int(target.counts.sum()) == 23_200_849
    assert len(set(target.path_gene_ids)) == 17_600
    assert target.scope_policy == (
        "likelihood_informative_validation_cell_gene_with_at_least_two_"
        "positive_ont_paths"
    )
    assert manifest["test_cells_or_counts_included"] is False
    assert manifest["optimizer_step_called"] is False
    assert manifest["training_started"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", [1103, 2207], "selected per command"),
        ("conditions", ["full", "atac"], "selected per command"),
        ("sampling_seed", 1103, "command seed"),
    ],
)
def test_config_rejects_embedded_run_identity(tmp_path, field, value, message):
    import yaml

    config = load_config("configs/fabric_v2_full_cohort.yaml")
    changed = deepcopy(config)
    changed["training"][field] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises((ValueError, RuntimeError), match=message):
        load_config(path)


def test_all_configs_share_one_configurable_gene_cell_sampling_contract(tmp_path):
    import yaml

    for path in (
        "configs/fabric_v2_full_cohort.yaml",
        "configs/fabric_v2_fixture.yaml",
        "configs/fabric_v2_toy.yaml",
    ):
        config = load_config(path)
        manifest = training_manifest_from_config(config, seed=1103, condition="full")
        assert manifest.primary_epoch_unit == (
            "sampled_informative_gene_cell_horvitz_thompson"
        )
        assert manifest.max_train_gene_cells_per_gene_per_epoch == 512
        assert manifest.optimizer_step_unit == "train_positive_gene"
        assert manifest.gene_microbatch_gradient_accumulation is True

    full_cohort = load_config("configs/fabric_v2_full_cohort.yaml")
    resolved = resolve_run_config(
        full_cohort,
        learning_rate=1.0e-4,
        lr_scheduler="constant",
        gradient_clip_norm=0.0,
        lambda_base=0.002,
        lambda_int=0.02,
        max_train_gene_cells_per_gene=1024,
        max_epochs=12,
        early_stopping_patience=3,
    )
    manifest = training_manifest_from_config(resolved, seed=2207, condition="rbp")
    assert manifest.learning_rate == 1.0e-4
    assert manifest.lr_scheduler_name == "constant"
    assert manifest.lr_scheduler_factor is None
    assert manifest.gradient_clip_norm == 0.0
    assert (manifest.lambda_base, manifest.lambda_int) == (0.002, 0.02)
    assert manifest.max_train_gene_cells_per_gene_per_epoch == 1024
    assert manifest.max_epochs == 12
    assert manifest.early_stopping_patience == 3

    changed = deepcopy(full_cohort)
    changed["training"]["max_train_gene_cells_per_gene_per_epoch"] = 2048
    path = tmp_path / "cap2048.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    assert (
        training_manifest_from_config(
            load_config(path), seed=7, condition="atac"
        ).max_train_gene_cells_per_gene_per_epoch
        == 2048
    )

    changed["training"]["max_train_gene_cells_per_gene_per_epoch"] = 0
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="positive integer"):
        load_config(path)

    changed = deepcopy(full_cohort)
    changed["training"]["optimizer_step_unit"] = "epoch"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="one train-positive gene"):
        load_config(path)

    changed = deepcopy(full_cohort)
    changed["training"]["gene_microbatch_gradient_accumulation"] = False
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="accumulate all attention microbatches"):
        load_config(path)

    with pytest.raises(ValueError, match="condition"):
        training_manifest_from_config(full_cohort, seed=7, condition="cis_dna")


def test_gene_cell_sampler_is_deterministic_group_closed_and_cap_aware():
    gene = make_toy_genes()[0]
    train_rows = rows_for_split(gene, "train")

    first = sample_train_gene_cells_for_epoch(
        gene,
        train_rows,
        max_gene_cells=2,
        seed=1103,
        epoch=1,
        gene_order_0based=0,
        gene_count=1,
    )
    repeated = sample_train_gene_cells_for_epoch(
        gene,
        train_rows,
        max_gene_cells=2,
        seed=1103,
        epoch=1,
        gene_order_0based=0,
        gene_count=1,
    )
    assert torch.equal(first.selected_cells, repeated.selected_cells)
    assert torch.equal(first.selected_rows, repeated.selected_rows)
    assert (first.available_cell_count, first.selected_cell_count) == (6, 2)
    assert first.inclusion_multiplier == 3.0
    for cell in first.selected_cells:
        expected_rows = train_rows[gene.row_cell_index[train_rows].eq(cell)]
        actual_rows = first.selected_rows[
            gene.row_cell_index[first.selected_rows].eq(cell)
        ]
        assert torch.equal(actual_rows, expected_rows)

    selections = {
        tuple(
            sample_train_gene_cells_for_epoch(
                gene,
                train_rows,
                max_gene_cells=2,
                seed=1103,
                epoch=epoch,
                gene_order_0based=0,
                gene_count=1,
            ).selected_cells.tolist()
        )
        for epoch in range(1, 9)
    }
    assert len(selections) > 1

    uncapped = sample_train_gene_cells_for_epoch(
        gene,
        train_rows,
        max_gene_cells=2048,
        seed=1103,
        epoch=1,
        gene_order_0based=0,
        gene_count=1,
    )
    assert uncapped.selected_cell_count == uncapped.available_cell_count == 6
    assert torch.equal(uncapped.selected_rows, train_rows)
    assert uncapped.inclusion_multiplier == 1.0


def test_each_runtime_condition_executes_one_capped_gene_cell_epoch():
    config = load_config("configs/fabric_v2_toy.yaml")
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["training"]["max_train_gene_cells_per_gene_per_epoch"] = 2
    for condition in RUN_CONDITIONS:
        run = train_run(
            make_toy_genes(),
            config,
            seed=101,
            condition=condition,
            device="cpu",
        )
        row = run.result.history.iloc[0]
        assert row["train_sampling_unit"] == (
            "sampled_informative_gene_cell_horvitz_thompson"
        )
        assert row["available_train_instances"] == 6
        assert row["sampled_train_instances"] == 2
        assert row["visited_train_instances"] == 2
        assert row["maximum_sampling_multiplier"] == 3.0
        assert row["optimizer_step_unit"] == "train_positive_gene"
        assert row["optimizer_steps"] == 1
        assert row["gene_microbatch_gradient_accumulation"]


def test_epoch_projection_caps_only_training_and_keeps_evaluation_complete():
    records = [
        {
            "gene_id": "projection_gene",
            "edge_count": 10,
            "path_count": 2,
            "dna_route_count": 0,
            "rna_route_count": 0,
            "ec_row_count": 1_100,
            "cell_count": 1_100,
            "train_cell_count": 1_000,
            "validation_cell_count": 100,
        }
    ]
    coefficients = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64).numpy()
    config = load_config("configs/fabric_v2_toy.yaml")
    projection = _project_epoch_seconds(
        records,
        resources=config["resources"],
        model_config=config["model"],
        cis_dim=4,
        dynamic_dim=32,
        max_train_gene_cells_per_gene_per_epoch=512,
        train_coefficients=coefficients,
        train_safety=1.0,
        evaluation_coefficients=coefficients,
        evaluation_safety=1.0,
    )
    assert projection["projected_available_train_gene_cell_count"] == 1_000
    assert projection["projected_sampled_train_gene_cell_count"] == 512
    assert projection["projected_train_batch_count_per_epoch"] == 1
    assert projection["projected_optimizer_step_count_per_epoch"] == 1
    assert projection["projected_validation_batch_count_per_epoch"] == 1
    assert "projected_train_evaluation_seconds" not in projection


def test_real_graph_compiler_preserves_frozen_path_order_and_retained_intron():
    rows = pd.DataFrame(
        [
            {
                "gene_id": "ENSG_REAL_FIXTURE",
                "path_id": "path_spliced",
                "resolved_transcript_id": "tx_spliced",
                "path_order_0based": 0,
                "chrom": "chr1",
                "strand": "+",
                "exon_starts_0based": [100, 200],
                "exon_ends_0based_exclusive": [150, 250],
                "transcript_aliases": ["tx_spliced"],
            },
            {
                "gene_id": "ENSG_REAL_FIXTURE",
                "path_id": "path_retained",
                "resolved_transcript_id": "tx_retained",
                "path_order_0based": 1,
                "chrom": "chr1",
                "strand": "+",
                "exon_starts_0based": [90],
                "exon_ends_0based_exclusive": [260],
                "transcript_aliases": ["tx_retained"],
            },
        ]
    )
    tables = compile_gene_graph_tables(rows)
    assert tables.paths["path_id"].tolist() == ["path_spliced", "path_retained"]
    retained = tables.path_edges.loc[
        tables.path_edges["path_id"].eq("path_retained"), "edge_type"
    ]
    assert retained.tolist() == [
        "EXON_CONTINUATION",
        "RETAINED_INTRON",
        "EXON_CONTINUATION",
    ]
    assert tables.edges["edge_id"].is_unique


def test_backed_prepared_dataset_loads_only_requested_gene_shard(tmp_path):
    first = make_toy_genes()[0]
    genes = (first, replace(first, gene_id="TOY_GENE_PREFETCH_SECOND"))
    train_mass = int(split_informative_molecule_mass(genes, "train"))
    validation_mass = int(split_informative_molecule_mass(genes, "val"))
    records = []
    for index, gene in enumerate(genes):
        relative = f"genes/gene_{index}.pt"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gene, destination)
        records.append({"gene_id": gene.gene_id, "relative_path": relative})
    (tmp_path / "PreparedDatasetManifest.json").write_text(
        json.dumps(
            {
                "schema_version": "fabric.backed_prepared_dataset.v1",
                "input_manifest_id": "real-fixture",
                "compatibility_artifact_id": "compatible-fixture",
                "informative_gene_ids": [gene.gene_id for gene in genes],
                "gene_shards": records,
                "expected_train_informative_molecule_mass": train_mass,
                "expected_validation_informative_molecule_mass": validation_mass,
            }
        )
    )
    backed = BackedPreparedDataset.load(tmp_path)
    backed.genes._load.cache_clear()
    assert len(backed.genes) == len(genes)
    assert backed.genes._load.cache_info().currsize == 0
    assert split_informative_molecule_mass(backed.genes, "train") == train_mass
    assert split_informative_molecule_mass(backed.genes, "val") == validation_mass
    assert backed.genes._load.cache_info().currsize == 0
    assert backed.genes[0].gene_id == genes[0].gene_id
    assert backed.genes._load.cache_info().currsize == 1
    assert backed.genes[0].model_input.cis_features.shape == (
        genes[0].model_input.cis_features.shape
    )
    assert backed.genes._load.cache_info().currsize == 1
    observed = [
        (index, gene.gene_id)
        for index, gene in _iter_gene_order(backed.genes, [1, 0], prefetch=True)
    ]
    assert observed == [(1, genes[1].gene_id), (0, genes[0].gene_id)]


def test_full_shape_profiler_selects_worst_dynamic_route_batch():
    records = [
        {
            "gene_id": "maximum_edge_gene",
            "edge_count": 100,
            "path_count": 2,
            "dna_route_count": 1,
            "rna_route_count": 1,
            "train_cell_count": 100,
            "validation_cell_count": 10,
            "ec_row_count": 100,
            "cell_count": 100,
        },
        {
            "gene_id": "maximum_route_projection_gene",
            "edge_count": 5,
            "path_count": 2,
            "dna_route_count": 120,
            "rna_route_count": 80,
            "train_cell_count": 1_000,
            "validation_cell_count": 100,
            "ec_row_count": 1_000,
            "cell_count": 1_000,
        },
        {
            "gene_id": "maximum_cell_gene",
            "edge_count": 1,
            "path_count": 2,
            "dna_route_count": 1,
            "rna_route_count": 1,
            "train_cell_count": 10_000,
            "validation_cell_count": 1_000,
            "ec_row_count": 10_000,
            "cell_count": 10_000,
        },
    ]
    config = load_config("configs/fabric_v2_toy.yaml")
    selected = _select_profile_records(
        records,
        resources=config["resources"],
        model_config=config["model"],
        cis_dim=4,
    )
    by_selector = {
        selector: record["gene_id"]
        for record in selected
        for selector in record["profile_selectors"]
    }
    assert (
        by_selector["maximum_route_projection_elements"]
        == "maximum_route_projection_gene"
    )


def test_adaptive_batch_limit_uses_full_gene_shape_and_cuda_kernel_cap():
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    base = {
        "gene_id": "light",
        "edge_count": 20,
        "path_count": 2,
        "dna_route_count": 10,
        "rna_route_count": 10,
        "ec_row_count": 100_000,
        "cell_count": 100_000,
    }
    heavy = {**base, "gene_id": "heavy", "rna_route_count": 10_000}
    light_limit = _record_batch_cell_limit(
        base,
        available_cells=100_000,
        resources=config["resources"],
        model_config=config["model"],
        cis_dim=30,
        phase="train",
    )
    heavy_limit = _record_batch_cell_limit(
        heavy,
        available_cells=100_000,
        resources=config["resources"],
        model_config=config["model"],
        cis_dim=30,
        phase="train",
    )
    assert light_limit == config["resources"]["max_cells_per_gpu_batch"]
    assert 0 < heavy_limit < light_limit


def test_real_cohort_current_train_cap_and_complete_validation_fit_one_batch_per_gene():
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    prepared_root = Path("data/processed/fabric_v2_real_dataset_v1/prepared_dataset")
    prepared = BackedPreparedDataset.load(prepared_root)
    cis_dim = int(prepared.genes[0].model_input.cis_features.shape[1])
    manifest = json.loads((prepared_root / "PreparedDatasetManifest.json").read_text())
    records = manifest["gene_record_audit"]
    cap = config["training"]["max_train_gene_cells_per_gene_per_epoch"]
    for record in records:
        train_cells = min(int(record["train_cell_count"]), cap)
        assert (
            _record_batch_cell_limit(
                record,
                available_cells=train_cells,
                resources=config["resources"],
                model_config=config["model"],
                cis_dim=cis_dim,
                phase="train",
            )
            == train_cells
        )
        validation_cells = int(record["validation_cell_count"])
        if validation_cells:
            assert (
                _record_batch_cell_limit(
                    record,
                    available_cells=validation_cells,
                    resources=config["resources"],
                    model_config=config["model"],
                    cis_dim=cis_dim,
                    phase="evaluation",
                )
                == validation_cells
            )


def test_profile_cost_model_upper_bounds_every_measured_batch():
    matrix = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float64
    ).numpy()
    measured = torch.tensor([0.1, 0.3, 0.55], dtype=torch.float64).numpy()
    coefficients, safety, audit = _fit_conservative_nonnegative_cost_model(
        matrix, measured
    )
    assert (matrix @ coefficients * safety >= measured).all()
    assert audit["all_profiled_batches_upper_bounded_after_safety"] is True
    assert safety >= 1.15
