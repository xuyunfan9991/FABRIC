from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy import sparse

from fabric.evaluate import OntMatrixKlTarget

from fabric.train import (
    FULL_COHORT_SCOPE,
    PreparedDataset,
    RouteDegreeCapSyntheticConfig,
    ValidationSnapshot,
    bind_route_degree_cap_structural_audit,
    build_optimizer,
    build_paired_models,
    evaluate_final_test,
    load_config,
    main,
    make_toy_genes,
    optimizer_parameter_groups,
    run_route_degree_cap_synthetic,
    select_lambda_pair,
    split_informative_molecule_mass,
    train_run,
    training_manifest_from_config,
    tune_optimizer_grid,
    validation_ont_matrix_kl_monitor,
)
from fabric.train import _route_synthetic_inputs
from fabric.train import _evaluate_split as evaluate_split


def _one_epoch_config():
    config = load_config("configs/fabric_v2_toy.yaml")
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    return config


def _train_full(genes, config, *, seed, device, monitor_callback=None):
    return train_run(
        genes,
        config,
        seed=seed,
        condition="full",
        device=device,
        monitor_callback=monitor_callback,
    )


def _toy_ont_target(gene):
    val_cells = tuple(
        cell_id
        for cell_id, split in zip(gene.cell_ids, gene.cell_split, strict=True)
        if split == "val"
    )
    counts = np.ones((len(gene.path_ids), len(val_cells)), dtype=np.int64)
    counts[0] = np.arange(2, 2 + len(val_cells), dtype=np.int64)
    return OntMatrixKlTarget(
        counts=sparse.csr_matrix(counts),
        path_ids=gene.path_ids,
        path_gene_ids=(gene.gene_id,) * len(gene.path_ids),
        cell_ids=val_cells,
        matrix_identity="toy-ont-matrix",
        path_identity="toy-path-axis",
        split_identity="toy-validation-split",
        scope_policy=(
            "likelihood_informative_validation_cell_gene_with_at_least_two_"
            "positive_ont_paths"
        ),
    )


def _enable_toy_ont_monitor(config):
    config["monitor"]["enabled"] = True
    config["monitor"]["target_root"] = "provided-by-test-callback"


def _route_manifest(**overrides):
    return run_route_degree_cap_synthetic(
        replace(RouteDegreeCapSyntheticConfig(), **overrides)
    )


def test_multiple_ec_rows_reuse_one_gene_cell_forward_and_fixed_split_denominator():
    gene = make_toy_genes()[0]
    config = _one_epoch_config()
    model = build_paired_models(gene, config["model"], seed=7, device="cpu")["full"]
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    snapshot = evaluate_split(
        [gene],
        model,
        condition="full",
        split="val",
        model_config=config["model"],
        resources=config["resources"],
    )
    handle.remove()
    assert len(calls) == 1
    assert snapshot.predictions[0].path_logits.shape[0] == 3
    assert snapshot.predictions[0].compatible_path_indices.shape[0] == 6
    expected = float(
        gene.molecule_count[
            gene.informative_row_mask
            & torch.tensor(
                [gene.cell_split[int(cell)] == "val" for cell in gene.row_cell_index]
            )
        ].sum()
    )
    assert snapshot.informative_molecule_mass == expected
    # The 50-molecule full-path audit rows are present but cannot dilute NLL.
    assert snapshot.informative_molecule_mass < float(
        gene.molecule_count[
            torch.tensor(
                [gene.cell_split[int(cell)] == "val" for cell in gene.row_cell_index]
            )
        ].sum()
    )


def test_prepared_gene_identity_and_integer_mass_contract_fails_closed():
    gene = make_toy_genes()[0]
    noninteger = gene.molecule_count.clone()
    noninteger[0] = 1.5
    with pytest.raises(ValueError, match="integer molecule mass"):
        _train_full(
            [replace(gene, molecule_count=noninteger)],
            _one_epoch_config(),
            seed=101,
            device="cpu",
        )

    with pytest.raises(ValueError, match="path identity axis"):
        _train_full(
            [replace(gene, path_ids=("path_a", "path_a"))],
            _one_epoch_config(),
            seed=101,
            device="cpu",
        )

    duplicate = gene.compatible_path_indices.clone()
    duplicate_mask = gene.compatible_path_mask.clone()
    duplicate[0] = torch.tensor([0, 0])
    duplicate_mask[0] = True
    with pytest.raises(ValueError, match="duplicate path identities"):
        _train_full(
            [
                replace(
                    gene,
                    compatible_path_indices=duplicate,
                    compatible_path_mask=duplicate_mask,
                )
            ],
            _one_epoch_config(),
            seed=101,
            device="cpu",
        )

    reversed_indices = gene.compatible_path_indices.clone()
    reversed_mask = gene.compatible_path_mask.clone()
    reversed_indices[0] = torch.tensor([1, 0])
    reversed_mask[0] = True
    with pytest.raises(ValueError, match="not in frozen path order"):
        _train_full(
            [
                replace(
                    gene,
                    compatible_path_indices=reversed_indices,
                    compatible_path_mask=reversed_mask,
                )
            ],
            _one_epoch_config(),
            seed=101,
            device="cpu",
        )

    gapped_mask = gene.compatible_path_mask.clone()
    gapped_mask[0] = torch.tensor([False, True])
    with pytest.raises(ValueError, match="left-aligned prefix"):
        _train_full(
            [replace(gene, compatible_path_mask=gapped_mask)],
            _one_epoch_config(),
            seed=101,
            device="cpu",
        )


def test_epoch_uses_one_condition_and_the_global_train_denominator():
    genes = make_toy_genes()
    result = _train_full(genes, _one_epoch_config(), seed=101, device="cpu")
    expected_mass = split_informative_molecule_mass(genes, "train")
    history = result.result.history
    assert history.loc[0, "epoch_train_denominator"] == expected_mass
    assert history.loc[0, "visited_train_instances"] == 6
    assert history.loc[0, "optimizer_step_unit"] == "train_positive_gene"
    assert history.loc[0, "optimizer_steps"] == len(genes)
    assert history.loc[0, "uniform_gene_step_objective_multiplier"] == len(genes)
    assert set(result.metrics["condition"]) == {"full"}
    assert set(result.metrics["split"]) == {"val"}
    assert np.isfinite(result.metrics["validation_compatible_path_nll"]).all()


def test_epoch_and_finalization_never_run_a_complete_train_evaluation(monkeypatch):
    import fabric.train as train_module

    observed_splits = []
    original = train_module._evaluate_split

    def counted_evaluation(*args, **kwargs):
        observed_splits.append(kwargs["split"])
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "_evaluate_split", counted_evaluation)
    result = _train_full(make_toy_genes(), _one_epoch_config(), seed=101, device="cpu")
    assert observed_splits == ["val"]
    assert "train_nll" not in result.result.history
    assert set(result.metrics["split"]) == {"val"}


def test_gene_shape_budget_batches_cells_without_changing_epoch_estimand_or_update():
    genes = make_toy_genes()
    small = _one_epoch_config()
    large = copy.deepcopy(small)
    small["resources"]["target_gpu_allocated_bytes"] = 1_075_000
    large["resources"]["target_gpu_allocated_bytes"] = 2_000_000
    batched = _train_full(genes, small, seed=101, device="cpu")
    unbatched = _train_full(genes, large, seed=101, device="cpu")
    batched_history = batched.result.history.iloc[0]
    unbatched_history = unbatched.result.history.iloc[0]
    assert batched_history["visited_train_instances"] == 6
    assert unbatched_history["visited_train_instances"] == 6
    assert batched_history["train_cell_batch_count"] == 6
    assert unbatched_history["train_cell_batch_count"] == 1
    assert batched_history["maximum_train_batch_cells"] == 1
    assert unbatched_history["maximum_train_batch_cells"] == 6
    assert batched_history["optimizer_steps"] == 1
    assert unbatched_history["optimizer_steps"] == 1
    assert (
        batched_history["epoch_train_denominator"]
        == unbatched_history["epoch_train_denominator"]
    )
    assert batched_history["validation_compatible_path_nll"] == pytest.approx(
        unbatched_history["validation_compatible_path_nll"], abs=1e-7
    )
    tolerance = small["resources"]["batching_probability_tolerance"]
    for split in ("train", "val"):
        left = evaluate_split(
            genes,
            batched.result.model,
            condition="full",
            split=split,
            model_config=small["model"],
            resources=large["resources"],
        )
        right = evaluate_split(
            genes,
            unbatched.result.model,
            condition="full",
            split=split,
            model_config=large["model"],
            resources=large["resources"],
        )
        assert left.nll == pytest.approx(right.nll, abs=tolerance)
        for left_prediction, right_prediction in zip(
            left.predictions, right.predictions, strict=True
        ):
            torch.testing.assert_close(
                left_prediction.path_logits.log_softmax(dim=-1),
                right_prediction.path_logits.log_softmax(dim=-1),
                atol=tolerance,
                rtol=0,
            )


def test_optimizer_steps_once_per_gene_after_all_gene_cell_microbatches(monkeypatch):
    gene = make_toy_genes()[0]
    genes = (gene, replace(gene, gene_id="TOY_GENE_SECOND"))
    config = _one_epoch_config()
    config["resources"]["target_gpu_allocated_bytes"] = 1_075_000
    calls = 0
    original_step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    result = _train_full(genes, config, seed=101, device="cpu")
    row = result.result.history.iloc[0]
    assert row["train_cell_batch_count"] == 12
    assert row["optimizer_steps"] == calls == 2
    assert row["train_positive_gene_count"] == 2
    assert row["uniform_gene_step_objective_multiplier"] == 2
    assert row["gene_microbatch_gradient_accumulation"]


def test_gradient_clipping_precedes_each_gene_step_and_plateau_uses_validation_nll(
    monkeypatch,
):
    gene = make_toy_genes()[0]
    config = load_config("configs/fabric_v2_toy.yaml")
    config["training"]["max_epochs"] = 3
    config["training"]["early_stopping_patience"] = 3
    config["optimizer"]["learning_rate"] = 0.01
    config["optimizer"]["lr_scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 0,
        "min_lr": 0.001,
    }
    config["optimizer"]["gradient_clip_norm"] = 1.0
    events = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.AdamW.step

    def recorded_clip(parameters, *args, **kwargs):
        events.append("clip")
        return original_clip(parameters, *args, **kwargs)

    def recorded_step(optimizer, *args, **kwargs):
        events.append("step")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recorded_clip)
    monkeypatch.setattr(torch.optim.AdamW, "step", recorded_step)
    monkeypatch.setattr(
        "fabric.train._evaluate_split",
        lambda *args, **kwargs: ValidationSnapshot(
            split="val",
            weighted_nll_sum=1.0,
            informative_molecule_mass=1.0,
            predictions=(),
        ),
    )
    result = _train_full((gene,), config, seed=101, device="cpu")
    history = result.result.history
    assert events == ["clip", "step"] * 3
    assert history["epoch_learning_rate"].tolist() == pytest.approx([0.01, 0.01, 0.005])
    assert history["next_epoch_learning_rate"].tolist() == pytest.approx(
        [0.01, 0.005, 0.0025]
    )
    assert set(history["lr_scheduler"]) == {"reduce_on_plateau"}
    assert set(history["gradient_clip_norm"]) == {1.0}
    assert result.result.lr_scheduler_state_dict is not None


def test_optimizer_groups_are_complete_disjoint_and_decay_only_declared_parameters():
    gene = make_toy_genes()[0]
    config = _one_epoch_config()
    model = build_paired_models(gene, config["model"], seed=17, device="cpu")["full"]
    groups = optimizer_parameter_groups(model, lambda_base=0.1, lambda_int=0.3)
    assert [group["group_name"] for group in groups] == [
        "no_decay",
        "base",
        "interaction",
    ]
    names = [name for group in groups for name in group["parameter_names"]]
    assert len(names) == len(set(names)) == len(dict(model.named_parameters()))
    assert set(groups[2]["parameter_names"]) == {
        "dna_aggregator.interaction_projection.weight",
        "rna_aggregator.interaction_projection.weight",
    }
    assert all(
        name.endswith("bias") or "output_norm" in name
        for name in groups[0]["parameter_names"]
    )

    optimizer = build_optimizer(
        model, learning_rate=0.1, lambda_base=0.1, lambda_int=0.3
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    after = dict(model.named_parameters())
    no_decay = set(groups[0]["parameter_names"])
    interaction = set(groups[2]["parameter_names"])
    for name, original in before.items():
        if name in no_decay:
            torch.testing.assert_close(after[name], original, atol=0, rtol=0)
        elif name in interaction:
            torch.testing.assert_close(
                after[name], original * 0.97, atol=1e-7, rtol=1e-7
            )
        else:
            torch.testing.assert_close(
                after[name], original * 0.99, atol=1e-7, rtol=1e-7
            )


def test_lambda_grid_selection_uses_only_three_dynamic_models_and_frozen_ties():
    scores = {
        (0.0, 0.01): {"cis_dna": 1.0, "cis_rna": 2.0, "full": 3.0},
        (0.001, 0.02): {"cis_dna": 2.0, "cis_rna": 2.0, "full": 2.0},
        (0.002, 0.02): {"cis_dna": 1.5, "cis_rna": 2.0, "full": 2.5},
    }
    # Same aggregate: larger lambda_int, then larger lambda_base wins.
    assert select_lambda_pair(scores, list(scores)) == (0.002, 0.02)


def test_optimizer_grid_orchestration_retrains_every_pair_seed_condition():
    config = _one_epoch_config()
    config["optimizer"]["tuning_seeds"] = [19]
    selection = tune_optimizer_grid(
        make_toy_genes(), config, device="cpu", monitor_callback=None
    )
    assert len(selection.runs) == 2 * 1 * 3
    assert {
        (run.lambda_base, run.lambda_int, run.tuning_seed, run.condition)
        for run in selection.runs
    } == {
        (pair[0], pair[1], 19, condition)
        for pair in selection.grid_order
        for condition in ("cis_dna", "cis_rna", "full")
    }
    expected_aggregate = tuple(
        float(np.mean(condition_means))
        for condition_means in selection.condition_mean_validation_nll
    )
    assert selection.aggregate_validation_nll == pytest.approx(expected_aggregate)
    assert selection.selected_pair == select_lambda_pair(
        {
            pair: dict(
                zip(
                    selection.selection_conditions,
                    selection.condition_mean_validation_nll[index],
                )
            )
            for index, pair in enumerate(selection.grid_order)
        },
        selection.grid_order,
    )
    assert selection.frozen_config_fields()["selection_status"] == "FROZEN"


def test_training_manifest_has_exactly_one_command_seed_and_condition():
    config = load_config("configs/fabric_v2_toy.yaml")
    manifest = training_manifest_from_config(config, seed=2207, condition="rbp")
    manifest.validate()
    assert manifest.seed == 2207
    assert manifest.condition == "rbp"
    with pytest.raises(ValueError, match="condition"):
        replace(manifest, condition="cis_rna").validate()


def test_config_freezes_optimizer_selection_and_claim_semantics(tmp_path):
    config = _one_epoch_config()
    changed = copy.deepcopy(config)
    changed["optimizer"]["selected_pair"] = [0.0, 0.001]
    path = tmp_path / "bad_selected_pair.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="differs from the runtime penalties"):
        load_config(path)

    changed = copy.deepcopy(config)
    changed["optimizer"]["lr_scheduler"] = {"name": "cosine"}
    path = tmp_path / "bad_schedule.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="constant or reduce_on_plateau"):
        load_config(path)

    changed = copy.deepcopy(config)
    changed["optimizer"]["gradient_clip_norm"] = -1.0
    path = tmp_path / "bad_clipping.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="finite and non-negative"):
        load_config(path)

    result = _train_full(make_toy_genes(), config, seed=101, device="cpu")
    assert set(result.metrics["execution_scope"]) == {"toy"}


def test_route_audit_manifest_freezes_generator_seeds_and_all_tolerances():
    synthetic = _route_synthetic_inputs(torch.tensor([-1.0, 0.0, 1.0]))
    assert synthetic["degree_2"].dna.route_edge_index.tolist() == [2, 3]
    assert synthetic["degree_4"].dna.route_edge_index.tolist() == [2, 3, 4, 5]
    assert synthetic["degree_8"].dna.route_edge_index.tolist() == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert [
        synthetic[name].dna.route_weight.numel()
        for name in ("cap_4_plus_2", "cap_2", "cap_0")
    ] == [8, 4, 2]
    for inputs in synthetic.values():
        dense = inputs.path_edge_incidence.to_dense()
        assert bool((dense.sum(dim=0) > 0).all())
        assert torch.unique(inputs.dna.route_base_features, dim=0).shape[0] == 1
    manifest = _route_manifest()
    manifest.validate()
    assert manifest.implementation_valid
    assert manifest.baseline_capability_pass
    assert manifest.route_degree_pass
    assert manifest.cap_coupling_pass
    assert manifest.failure_reasons == ()
    assert len(manifest.measurements) == 18
    with pytest.raises(ValueError, match="exactly three"):
        _route_manifest(audit_seeds=(7, 7, 11))
    with pytest.raises(ValueError, match="positive and finite"):
        _route_manifest(cap_gradient_drift_tolerance=0.0)
    with pytest.raises(ValueError, match="generator identity"):
        _route_manifest(generator_identity="")
    tampered_measurement = replace(
        manifest.measurements[0],
        matched_delta_rho=manifest.measurements[0].matched_delta_rho + 1.0,
    )
    with pytest.raises(ValueError, match="recovery_error differs"):
        replace(
            manifest,
            measurements=(tampered_measurement, *manifest.measurements[1:]),
        )
    structural = pd.DataFrame(
        {
            "audit_population": ["model_input", "model_input", "catalog"],
            "event_id": ["high-degree", "external", "catalog-only"],
            "D_post": [4, 2, 8],
            "external_only_coupling": [False, True, True],
        }
    )
    bound = bind_route_degree_cap_structural_audit(
        RouteDegreeCapSyntheticConfig(),
        structural,
        structural_route_audit_identity="real-route-audit-v2",
    )
    assert bound.model_input_degree_gt2_event_count == 1
    assert bound.model_input_external_only_coupling_event_count == 1
    no_applicability = bind_route_degree_cap_structural_audit(
        RouteDegreeCapSyntheticConfig(),
        structural.loc[structural.audit_population.eq("catalog")],
        structural_route_audit_identity="empty-model-input-v2",
    )
    no_applicability_manifest = replace(
        manifest,
        structural_route_audit_identity=(
            no_applicability.structural_route_audit_identity
        ),
        model_input_degree_gt2_event_count=(
            no_applicability.model_input_degree_gt2_event_count
        ),
        model_input_external_only_coupling_event_count=(
            no_applicability.model_input_external_only_coupling_event_count
        ),
    )
    assert not no_applicability_manifest.route_degree_catalog_applicable
    assert not no_applicability_manifest.cap_coupling_catalog_applicable


def test_full_cohort_guard_fails_before_data_gpu_or_run_directory(tmp_path):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    assert config["execution"]["scope"] == FULL_COHORT_SCOPE
    run_dir = tmp_path / "must_not_exist"
    with pytest.raises(RuntimeError, match="training is not authorized"):
        main(
            [
                "--config",
                "configs/fabric_v2_full_cohort.yaml",
                "--run-dir",
                str(run_dir),
                "--device",
                "cuda:999",
                "--seed",
                "1103",
                "--condition",
                "full",
                "--fixture",
                str(tmp_path / "missing.pt"),
            ]
        )
    assert not run_dir.exists()


def test_final_test_is_separate_from_training_and_gate_fails_closed():
    genes = make_toy_genes()
    toy = _one_epoch_config()
    result = _train_full(genes, toy, seed=101, device="cpu")
    # Training metrics contain no held-out predictions, even for toy execution.
    assert set(result.metrics["split"]) == {"val"}
    assert not toy["execution"]["final_test_authorized"]
    with pytest.raises(RuntimeError, match="test inference is not authorized"):
        evaluate_final_test(
            genes,
            result,
            toy,
            checkpoints_frozen=True,
            report_rules_frozen=True,
        )
    authorized = copy.deepcopy(toy)
    authorized["execution"]["final_test_authorized"] = True
    toy_test = evaluate_final_test(
        genes,
        result,
        authorized,
        checkpoints_frozen=True,
        report_rules_frozen=True,
    )
    assert set(toy_test["split"]) == {"test"}
    with pytest.raises(RuntimeError, match="frozen checkpoints"):
        evaluate_final_test(
            genes,
            result,
            authorized,
            checkpoints_frozen=False,
            report_rules_frozen=True,
        )


def test_monitor_runs_once_after_each_epoch_and_cannot_change_checkpoint():
    genes = make_toy_genes()
    config = _one_epoch_config()
    without = _train_full(genes, config, seed=303, device="cpu")
    _enable_toy_ont_monitor(config)
    target = _toy_ont_target(genes[0])
    calls = []

    def monitor(condition, epoch, snapshot):
        calls.append((condition, epoch, len(snapshot.predictions)))
        # Deliberately consume all RNG sources; trainer restores them.
        _ = torch.rand(7)
        _ = np.random.random(7)
        return validation_ont_matrix_kl_monitor(
            condition, epoch, snapshot, target=target
        )

    with_monitor = _train_full(
        genes, config, seed=303, device="cpu", monitor_callback=monitor
    )
    assert calls == [("full", 1, 1)]
    left = without.result
    right = with_monitor.result
    assert left.best_epoch == right.best_epoch
    assert left.best_validation_nll == right.best_validation_nll
    assert len(right.monitor_records) == 1
    assert right.monitor_records[0].sealed
    assert not right.monitor_records[0].selection_eligible
    assert set(right.history.columns) >= {
        "validation_compatible_path_nll",
        "ont_matrix_kl_count_weighted",
    }
    assert np.isfinite(right.history["ont_matrix_kl_count_weighted"]).all()
    assert "train_nll" not in right.history
    for key, value in left.model.state_dict().items():
        torch.testing.assert_close(right.model.state_dict()[key], value, atol=0, rtol=0)


def test_one_run_is_end_to_end_and_writes_v2_artifacts(tmp_path):
    config = _one_epoch_config()
    _enable_toy_ont_monitor(config)
    genes = make_toy_genes()
    target = _toy_ont_target(genes[0])
    run_dir = tmp_path / "toy_run"
    result = train_run(
        genes,
        config,
        seed=101,
        condition="atac",
        device="cpu",
        run_dir=run_dir,
        monitor_callback=lambda condition, epoch, snapshot: (
            validation_ont_matrix_kl_monitor(condition, epoch, snapshot, target=target)
        ),
    )
    assert result.manifest.seed == 101
    assert result.manifest.condition == "atac"
    assert (run_dir / "training_run_manifest.json").exists()
    assert (run_dir / "input_manifest.json").exists()
    assert (run_dir / "metrics.tsv").exists()
    assert (run_dir / "sealed_validation_monitor.jsonl").exists()
    assert (run_dir / "optimizer_manifest.json").exists()
    assert (run_dir / "checkpoint_manifest.json").exists()
    assert (run_dir / "monitor_manifest.json").exists()
    input_manifest = json.loads((run_dir / "input_manifest.json").read_text())
    assert input_manifest["test_model_predictions_status"] == (
        "NOT_COMPUTED_DURING_TRAINING"
    )
    assert input_manifest["gene_shape_adaptive_batching"]["batch_policy"] == (
        "gene_shape_adaptive_v1"
    )
    assert (
        input_manifest["gene_shape_adaptive_batching"][
            "probability_numerical_tolerance"
        ]
        == 1.0e-6
    )
    optimizer_manifest = json.loads((run_dir / "optimizer_manifest.json").read_text())
    assert optimizer_manifest["lr_scheduler"] == {"name": "constant"}
    assert optimizer_manifest["gradient_clip_norm"] == 0.0
    assert [
        group["group_name"] for group in optimizer_manifest["parameter_groups"]
    ] == [
        "no_decay",
        "base",
        "interaction",
    ]
    checkpoint_manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    assert not checkpoint_manifest["selection_rules_used_held_out_test"]
    assert checkpoint_manifest["record"]["seed"] == 101
    assert checkpoint_manifest["record"]["condition"] == "atac"
    assert not checkpoint_manifest["record"]["held_out_test_evaluated"]
    monitor_manifest = json.loads((run_dir / "monitor_manifest.json").read_text())
    assert monitor_manifest["record_count"] == 1
    assert monitor_manifest["all_records_sealed"]
    assert not monitor_manifest["any_record_selection_eligible"]
    assert not monitor_manifest["held_out_test_model_predictions_computed"]
    assert len(result.metrics) == 1
    assert set(result.metrics["split"]) == {"val"}
    assert (run_dir / "checkpoint.pt").exists()
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["schema_version"] == "fabric.training_checkpoint.v2"
    assert checkpoint["model_state_dict"]
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["lr_scheduler_state_dict"] is None
    assert (run_dir / "history.tsv").exists()


def test_python_module_cli_loads_exact_prepared_dataset_identity(tmp_path):
    config = _one_epoch_config()
    config["execution"]["scope"] = "fixture"
    config["inputs"]["compatible_ec_scope"] = "serialized_v2_fixture"
    config["inputs"]["test_exposure"] = "fixture_only"
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    bundle_path = tmp_path / "prepared.pt"
    torch.save(
        PreparedDataset(
            genes=tuple(make_toy_genes(seed=17)),
            input_manifest_id="serialized-fixture-v2",
            compatibility_artifact_id="toy-compatible-ec-v2",
        ),
        bundle_path,
    )
    run_dir = tmp_path / "module_run"
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fabric.train",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--seed",
            "17",
            "--condition",
            "rbp",
            "--learning-rate",
            "0.002",
            "--lr-scheduler",
            "reduce_on_plateau",
            "--lr-factor",
            "0.5",
            "--lr-patience",
            "0",
            "--min-lr",
            "0.0005",
            "--gradient-clip-norm",
            "0.7",
            "--max-train-gene-cells-per-gene",
            "2",
            "--max-epochs",
            "1",
            "--early-stopping-patience",
            "1",
            "--fixture",
            str(bundle_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    metrics = pd.read_csv(run_dir / "metrics.tsv", sep="\t")
    assert len(metrics) == 1
    assert set(metrics["split"]) == {"val"}
    assert set(metrics["condition"]) == {"rbp"}
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert resolved["optimizer"]["learning_rate"] == 0.002
    assert resolved["optimizer"]["lr_scheduler"] == {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 0,
        "min_lr": 0.0005,
    }
    assert resolved["optimizer"]["gradient_clip_norm"] == 0.7
    assert resolved["training"]["max_train_gene_cells_per_gene_per_epoch"] == 2
    manifest = json.loads((run_dir / "training_run_manifest.json").read_text())
    assert manifest["learning_rate"] == 0.002
    assert manifest["lr_scheduler_name"] == "reduce_on_plateau"
    assert manifest["gradient_clip_norm"] == 0.7
