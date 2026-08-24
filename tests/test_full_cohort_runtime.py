from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import torch

from fabric.evaluate import OntMatrixKlTarget
from fabric.real_dataset import compile_gene_graph_tables
from fabric.source_identity import committed_source_identity
from fabric.profile_real import (
    _fit_conservative_nonnegative_cost_model,
    _project_epoch_seconds,
    _record_batch_cell_limit,
    _select_profile_records,
    main as profile_real_main,
)
from fabric.readiness_real import _validate_launch_condition
from fabric.train import (
    BackedPreparedDataset,
    FULL_COHORT_SCOPE,
    RUN_CONDITIONS,
    _MODEL_CONDITION,
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
from fabric.train import _configure_cuda_allocator, _iter_gene_order


def _as_condition_profile_v1(profile, config, *, condition):
    result = deepcopy(profile)
    allocated = int(result["maximum_profile_peak_gpu_allocated_bytes"])
    reserved = int(result.get("maximum_profile_peak_gpu_reserved_bytes", allocated))
    gene_id = "profile_fixture_gene"
    common = {
        "gene_id": gene_id,
        "peak_gpu_allocated_bytes": allocated,
        "peak_gpu_reserved_bytes": reserved,
        "adaptive_estimated_gpu_bytes": allocated,
    }
    result.update(
        {
            "schema_version": "fabric.real_condition_shape_profile.v1",
            "profiled_condition": condition,
            "profiled_model_condition": _MODEL_CONDITION[condition],
            "profiled_model_config": deepcopy(config["model"]),
            "input_manifest_id": config["inputs"]["input_manifest_id"],
            "compatibility_artifact_id": config["inputs"][
                "compatibility_artifact_id"
            ],
            "profile_source_git_commit": committed_source_identity(
                require_clean=False
            ),
            "epoch_evaluation_policy": (
                "projected_complete_train_and_validation_from_profiled_batches"
            ),
            "profile_execution_policy": (
                "selected_train_backward_and_validation_no_grad_batches"
            ),
            "profile_batches": [
                {
                    **common,
                    "phase": "train",
                    "split": "train",
                    "model_mode": "train",
                    "profile_shape_role": (
                        "highest_ec_row_count_capped_train_sample"
                    ),
                    "gradient_audit": {
                        "mode": "backward",
                        "backward_called": True,
                        "passed": True,
                    },
                },
                {
                    **common,
                    "phase": "evaluation",
                    "split": "validation",
                    "model_mode": "eval",
                    "profile_shape_role": (
                        "complete_validation_largest_estimated_batch"
                    ),
                    "gradient_audit": {
                        "mode": "evaluation_no_grad",
                        "backward_called": False,
                        "autograd_enabled_during_forward": False,
                        "passed": True,
                    },
                },
            ],
            "profile_batch_count": 2,
            "profile_train_batch_count": 1,
            "profile_evaluation_batch_count": 1,
            "maximum_profile_peak_gpu_reserved_bytes": reserved,
            "maximum_train_profile_peak_gpu_allocated_bytes": allocated,
            "maximum_evaluation_profile_peak_gpu_allocated_bytes": allocated,
            "maximum_train_profile_peak_gpu_reserved_bytes": reserved,
            "maximum_evaluation_profile_peak_gpu_reserved_bytes": reserved,
        }
    )
    return result


def test_source_identity_tracks_the_committed_source_tree():
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "src/fabric"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert committed_source_identity(require_clean=False) == expected


def test_full_cohort_config_is_single_run_and_test_blind():
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    assert config["execution"] == {
        "scope": FULL_COHORT_SCOPE,
        "training_authorized": True,
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
    assert manifest.backed_gene_cache_capacity == 2
    assert manifest.compute_precision == "float32_highest"
    assert "max_attention_elements_per_batch" not in config["resources"]
    assert_execution_admitted(config, condition="full")
    unauthorized = deepcopy(config)
    unauthorized["execution"]["training_authorized"] = False
    with pytest.raises(RuntimeError, match="training is not authorized"):
        assert_execution_admitted(unauthorized, condition="full")

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


@pytest.mark.parametrize("condition", ("atac", "rbp"))
def test_legacy_full_profile_cannot_admit_an_ablation_condition(
    condition, tmp_path
):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    with pytest.raises(RuntimeError, match="exact condition|launch boundary"):
        assert_execution_admitted(config, condition=condition)

    relabeled = json.loads(Path(config["resources"]["profile_artifact"]).read_text())
    relabeled["profiled_condition"] = condition
    relabeled_path = tmp_path / f"RelabeledLegacyProfile.{condition}.json"
    relabeled_path.write_text(json.dumps(relabeled))
    config["resources"]["profile_artifact"] = str(relabeled_path)
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(config, condition=condition)


def test_condition_profile_admission_selects_and_verifies_the_exact_condition(
    tmp_path,
):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    legacy_profile = json.loads(
        Path(config["resources"]["profile_artifact"]).read_text()
    )
    profile_paths = {}
    for condition in RUN_CONDITIONS:
        profile = _as_condition_profile_v1(
            legacy_profile, config, condition=condition
        )
        path = tmp_path / f"ResourceProfile.{condition}.json"
        path.write_text(json.dumps(profile))
        profile_paths[condition] = path

    for condition in RUN_CONDITIONS:
        exact = deepcopy(config)
        exact["resources"]["profile_artifact"] = str(profile_paths[condition])
        assert_execution_admitted(exact, condition=condition)

        mismatch = deepcopy(exact)
        other_condition = next(item for item in RUN_CONDITIONS if item != condition)
        mismatch["resources"]["profile_artifact"] = str(
            profile_paths[other_condition]
        )
        with pytest.raises(RuntimeError, match="launch boundary"):
            assert_execution_admitted(mismatch, condition=condition)

        wrong_internal = deepcopy(exact)
        wrong_profile = json.loads(profile_paths[condition].read_text())
        wrong_profile["profiled_model_condition"] = _MODEL_CONDITION[other_condition]
        wrong_path = tmp_path / f"ResourceProfile.{condition}.wrong-internal.json"
        wrong_path.write_text(json.dumps(wrong_profile))
        wrong_internal["resources"]["profile_artifact"] = str(wrong_path)
        with pytest.raises(RuntimeError, match="launch boundary"):
            assert_execution_admitted(wrong_internal, condition=condition)


@pytest.mark.parametrize(
    "tamper",
    ("model", "input", "compatibility", "source", "evaluation_phase"),
)
def test_condition_profile_binds_model_data_source_and_both_runtime_phases(
    tmp_path, tamper
):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    profile = _as_condition_profile_v1(
        json.loads(Path(config["resources"]["profile_artifact"]).read_text()),
        config,
        condition="full",
    )
    if tamper == "model":
        profile["profiled_model_config"]["hidden_dim"] += 1
    elif tamper == "input":
        profile["input_manifest_id"] = "different_input"
    elif tamper == "compatibility":
        profile["compatibility_artifact_id"] = "different_compatibility"
    elif tamper == "source":
        profile["profile_source_git_commit"] = "different_source"
    else:
        profile["profile_batches"][1]["phase"] = "train"
    path = tmp_path / f"ResourceProfile.full.{tamper}.json"
    path.write_text(json.dumps(profile))
    config["resources"]["profile_artifact"] = str(path)
    with pytest.raises(RuntimeError, match="launch boundary|batch structure"):
        assert_execution_admitted(config, condition="full")


def test_condition_profile_binds_backed_gene_cache_capacity(tmp_path):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    profile = _as_condition_profile_v1(
        json.loads(Path(config["resources"]["profile_artifact"]).read_text()),
        config,
        condition="full",
    )
    config["resources"]["backed_gene_cache_capacity"] = 17_600
    profile_path = tmp_path / "ResourceProfile.full.cache.json"
    profile_path.write_text(json.dumps(profile))
    config["resources"]["profile_artifact"] = str(profile_path)
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(config, condition="full")

    profile["backed_gene_cache_capacity"] = 17_600
    profile_path.write_text(json.dumps(profile))
    assert_execution_admitted(config, condition="full")


def test_cuda_allocator_limit_is_validated_and_frozen_in_the_run_manifest(tmp_path):
    import yaml

    config = load_config("configs/fabric_v2_toy.yaml")
    config["resources"].update(
        {
            "target_gpu_allocated_bytes": 6 << 30,
            "cuda_allocator_limit_bytes": 8 << 30,
            "gpu_total_memory_bytes": 24 << 30,
        }
    )
    path = tmp_path / "allocator_limit.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = load_config(path)
    manifest = training_manifest_from_config(loaded, seed=1103, condition="atac")
    assert manifest.target_gpu_allocated_bytes == 6 << 30
    assert manifest.cuda_allocator_limit_bytes == 8 << 30

    for invalid_limit in (True, (6 << 30) - 1, (24 << 30) + 1):
        invalid = deepcopy(config)
        invalid["resources"]["cuda_allocator_limit_bytes"] = invalid_limit
        invalid_path = tmp_path / f"invalid_{invalid_limit}.yaml"
        invalid_path.write_text(yaml.safe_dump(invalid, sort_keys=False))
        with pytest.raises(ValueError, match="CUDA allocator limit|allocator_limit"):
            load_config(invalid_path)


def test_cuda_allocator_limit_is_applied_as_a_per_process_fraction(monkeypatch):
    from types import SimpleNamespace

    device = torch.device("cuda:1")
    observed = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda requested: SimpleNamespace(total_memory=16 << 30),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, *, device: observed.append((fraction, device)),
    )

    limit = _configure_cuda_allocator(
        device, {"cuda_allocator_limit_bytes": 8 << 30}
    )
    assert limit == 8 << 30
    assert observed == [(0.5, device)]

    observed.clear()
    assert _configure_cuda_allocator(device, {}) is None
    assert observed == []
    with pytest.raises(ValueError, match="device-sized integer"):
        _configure_cuda_allocator(
            device, {"cuda_allocator_limit_bytes": (16 << 30) + 1}
        )


def test_admission_rejects_a_profile_whose_reserved_peak_exceeds_the_allocator_limit(
    tmp_path,
):
    config = load_config("configs/fabric_v2_full_cohort.yaml")
    profile = _as_condition_profile_v1(
        json.loads(Path(config["resources"]["profile_artifact"]).read_text()),
        config,
        condition="full",
    )
    allocator_limit = 22 << 30
    config["resources"]["cuda_allocator_limit_bytes"] = allocator_limit
    profile.update(
        {
            "cuda_allocator_limit_bytes": allocator_limit,
            "maximum_profile_peak_gpu_reserved_bytes": allocator_limit,
            "maximum_train_profile_peak_gpu_reserved_bytes": allocator_limit,
            "maximum_evaluation_profile_peak_gpu_reserved_bytes": allocator_limit,
        }
    )
    for value in profile["profile_batches"]:
        value["peak_gpu_reserved_bytes"] = allocator_limit
    profile_path = tmp_path / "ResourceProfile.full.json"
    profile_path.write_text(json.dumps(profile))
    config["resources"]["profile_artifact"] = str(profile_path)
    assert_execution_admitted(config, condition="full")

    profile["maximum_profile_peak_gpu_reserved_bytes"] = allocator_limit + 1
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(config, condition="full")

    profile["maximum_profile_peak_gpu_reserved_bytes"] = allocator_limit
    allocated = config["resources"]["target_gpu_allocated_bytes"]
    profile["maximum_profile_peak_gpu_allocated_bytes"] = allocated
    profile["maximum_train_profile_peak_gpu_allocated_bytes"] = allocated
    profile["maximum_evaluation_profile_peak_gpu_allocated_bytes"] = allocated
    for value in profile["profile_batches"]:
        value["peak_gpu_allocated_bytes"] = allocated
        value["adaptive_estimated_gpu_bytes"] = allocated
    profile["frozen_host_ram_requirement_bytes"] = (
        config["resources"]["frozen_host_ram_requirement_bytes"] + 1
    )
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(config, condition="full")

    profile["frozen_host_ram_requirement_bytes"] = config["resources"][
        "frozen_host_ram_requirement_bytes"
    ]
    allocated += 1
    profile["maximum_profile_peak_gpu_allocated_bytes"] = allocated
    profile["maximum_train_profile_peak_gpu_allocated_bytes"] = allocated
    profile["maximum_evaluation_profile_peak_gpu_allocated_bytes"] = allocated
    for value in profile["profile_batches"]:
        value["peak_gpu_allocated_bytes"] = allocated
        value["adaptive_estimated_gpu_bytes"] = allocated
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(config, condition="full")


@pytest.mark.parametrize(
    ("condition", "config_path"),
    [
        (
            "full",
            "configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g_full.yaml",
        ),
        (
            "atac",
            "configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g_atac.yaml",
        ),
        (
            "rbp",
            "configs/fabric_v2_full_cohort_reliability_dtu_macro_shared6g_rbp.yaml",
        ),
    ],
)
def test_shared_gpu_three_arm_configs_freeze_one_scheduler_and_resource_policy(
    condition, config_path
):
    config = load_config(config_path)
    manifest = training_manifest_from_config(config, seed=1103, condition=condition)
    assert manifest.gene_objective == "reliability_dtu_macro"
    assert manifest.lr_scheduler_name == "reduce_on_plateau"
    assert manifest.lr_scheduler_patience == 2
    assert manifest.lr_scheduler_fixed_initial_epochs == 0
    assert manifest.target_gpu_allocated_bytes == 6 << 30
    assert manifest.cuda_allocator_limit_bytes == 8 << 30
    assert manifest.max_cells_per_gpu_batch == 4096
    assert manifest.backed_gene_cache_capacity == 2
    assert config["resources"]["unmodeled_gpu_reserve_bytes"] == 2 << 30
    assert "profile_artifacts" not in config["resources"]
    assert Path(config["resources"]["profile_artifact"]).name == (
        f"ResourceProfile.{condition}.json"
    )
    assert config["execution"]["final_test_authorized"] is False
    assert manifest.learning_rate == pytest.approx(5e-5)
    assert manifest.lr_scheduler_factor == pytest.approx(0.5)
    assert manifest.lr_scheduler_min_lr == pytest.approx(1e-5)


def test_shared_gpu_three_arm_configs_differ_only_by_exact_profile_path():
    configs = []
    for condition in RUN_CONDITIONS:
        config = load_config(
            "configs/"
            "fabric_v2_full_cohort_reliability_dtu_macro_shared6g_"
            f"{condition}.yaml"
        )
        assert Path(config["resources"]["profile_artifact"]).name == (
            f"ResourceProfile.{condition}.json"
        )
        config["resources"]["profile_artifact"] = "<condition-specific-profile>"
        configs.append(config)
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("condition", RUN_CONDITIONS)
def test_resident_cache_configs_are_explicit_and_launch_authorized(condition):
    """These configs record one authorized resident-cache run per arm.

    Training authorization is deliberate here; held-out test exposure is not,
    and must stay closed however the training gate moves.
    """

    config = load_config(
        "configs/"
        "fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_resident_"
        f"{condition}.yaml"
    )
    manifest = training_manifest_from_config(config, seed=1103, condition=condition)
    assert manifest.backed_gene_cache_capacity == 17_600
    assert config["resources"]["frozen_host_ram_requirement_bytes"] == 160 << 30
    assert config["resources"]["frozen"] is True
    assert config["resources"]["profile_status"] == "FROZEN_REAL_FULL_SHAPE_PROFILE"
    assert config["execution"]["training_authorized"] is True
    assert config["execution"]["final_test_authorized"] is False
    assert Path(config["resources"]["profile_artifact"]).name == (
        f"ResourceProfile.{condition}.json"
    )


@pytest.mark.parametrize("condition", RUN_CONDITIONS)
def test_resident_cache_configs_fail_closed_without_their_frozen_profile(condition):
    """The authorized config is admitted only by its own regenerated profile."""

    config = load_config(
        "configs/"
        "fabric_v2_full_cohort_reliability_dtu_macro_shared6g2x_resident_"
        f"{condition}.yaml"
    )
    revoked = deepcopy(config)
    revoked["execution"]["training_authorized"] = False
    with pytest.raises(RuntimeError, match="not authorized"):
        assert_execution_admitted(revoked, condition=condition)

    streaming = deepcopy(config)
    streaming["resources"]["backed_gene_cache_capacity"] = 2
    with pytest.raises(RuntimeError, match="launch boundary"):
        assert_execution_admitted(streaming, condition=condition)


def test_condition_profile_cli_refuses_to_overwrite_an_immutable_artifact(tmp_path):
    destination = tmp_path / "ResourceProfile.full.json"
    destination.write_text("already frozen\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        profile_real_main(
            [
                "--prepared",
                "not-read-because-output-exists",
                "--config",
                "not-read-because-output-exists",
                "--condition",
                "full",
                "--output",
                str(destination),
            ]
        )
    assert destination.read_text() == "already frozen\n"


def test_readiness_launch_command_selects_one_exact_condition():
    _validate_launch_condition(
        "python -m fabric.train --condition rbp --seed 1103", "rbp"
    )
    with pytest.raises(ValueError, match="condition exactly"):
        _validate_launch_condition(
            "python -m fabric.train --condition atac --seed 1103", "rbp"
        )
    with pytest.raises(ValueError, match="condition exactly"):
        _validate_launch_condition(
            "python -m fabric.train --condition rbp --condition full", "rbp"
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
    assert len(target.expected_cell_gene_keys) == manifest["expected_cell_gene_count"]
    assert manifest["expected_cell_gene_axis"] == "expected_cell_gene_axis.parquet"
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


@pytest.mark.parametrize("strand", ["+", "-"])
def test_real_graph_compiler_preserves_frozen_path_order_and_retained_intron(strand):
    rows = pd.DataFrame(
        [
            {
                "gene_id": "ENSG_REAL_FIXTURE",
                "path_id": "path_spliced",
                "resolved_transcript_id": "tx_spliced",
                "path_order_0based": 0,
                "chrom": "chr1",
                "strand": strand,
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
                "strand": strand,
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
    splice = tables.edges.loc[tables.edges["edge_type"].eq("SPLICE")].iloc[0]
    assert splice.span_bp == 50
    assert splice.length_bp == 0
    retained_edge = tables.edges.loc[
        tables.edges["edge_type"].eq("RETAINED_INTRON")
    ].iloc[0]
    assert retained_edge.length_bp == retained_edge.span_bp == 50
    assert tables.paths.set_index("path_id")["path_length_bp"].to_dict() == {
        "path_spliced": 100,
        "path_retained": 170,
    }
    assert tables.edges["edge_id"].is_unique


@pytest.mark.parametrize("strand", ["+", "-"])
def test_real_graph_compiler_atomizes_overlapping_retained_introns(strand):
    rows = pd.DataFrame(
        [
            {
                "gene_id": "ENSG_OVERLAP",
                "path_id": "splice_a",
                "resolved_transcript_id": "splice_a",
                "path_order_0based": 0,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [100, 220],
                "exon_ends_0based_exclusive": [150, 300],
                "transcript_aliases": ["splice_a"],
            },
            {
                "gene_id": "ENSG_OVERLAP",
                "path_id": "splice_b",
                "resolved_transcript_id": "splice_b",
                "path_order_0based": 1,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [100, 250],
                "exon_ends_0based_exclusive": [180, 300],
                "transcript_aliases": ["splice_b"],
            },
            {
                "gene_id": "ENSG_OVERLAP",
                "path_id": "retained",
                "resolved_transcript_id": "retained",
                "path_order_0based": 2,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [90],
                "exon_ends_0based_exclusive": [310],
                "transcript_aliases": ["retained"],
            },
        ]
    )
    tables = compile_gene_graph_tables(rows)
    retained = tables.path_edges.loc[
        tables.path_edges["path_id"].eq("retained")
    ].merge(tables.edges, on="edge_id", suffixes=("_path", ""))
    retained_atoms = retained.loc[retained["edge_type"].eq("RETAINED_INTRON")]
    assert sorted(
        retained_atoms[["start_0based", "end_0based_exclusive"]]
        .itertuples(index=False, name=None)
    ) == [(150, 180), (180, 220), (220, 250)]
    assert not (
        (tables.edges["start_0based"] == 150)
        & (tables.edges["end_0based_exclusive"] == 250)
    ).any()
    assert tables.paths.set_index("path_id")["path_length_bp"].to_dict() == {
        "splice_a": 130,
        "splice_b": 130,
        "retained": 220,
    }


@pytest.mark.parametrize("strand", ["+", "-"])
def test_real_graph_compiler_preserves_coincident_acceptor_and_donor(strand):
    rows = pd.DataFrame(
        [
            {
                "gene_id": "ENSG_TOUCH",
                "path_id": "splice_left",
                "resolved_transcript_id": "splice_left",
                "path_order_0based": 0,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [100, 200],
                "exon_ends_0based_exclusive": [150, 300],
                "transcript_aliases": ["splice_left"],
            },
            {
                "gene_id": "ENSG_TOUCH",
                "path_id": "splice_right",
                "resolved_transcript_id": "splice_right",
                "path_order_0based": 1,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [100, 250],
                "exon_ends_0based_exclusive": [200, 300],
                "transcript_aliases": ["splice_right"],
            },
            {
                "gene_id": "ENSG_TOUCH",
                "path_id": "retained",
                "resolved_transcript_id": "retained",
                "path_order_0based": 2,
                "chrom": "chr1",
                "strand": strand,
                "exon_starts_0based": [90],
                "exon_ends_0based_exclusive": [310],
                "transcript_aliases": ["retained"],
            },
        ]
    )
    tables = compile_gene_graph_tables(rows)
    shared = tables.nodes.loc[tables.nodes["pos_0based"].eq(200), "node_type"]
    assert set(shared) == {"acceptor", "donor"}
    retained = tables.path_edges.loc[
        tables.path_edges["path_id"].eq("retained")
    ].merge(tables.edges, on="edge_id", suffixes=("_path", ""))
    bridge = retained.loc[
        retained["start_0based"].eq(200)
        & retained["end_0based_exclusive"].eq(200)
    ].iloc[0]
    assert bridge.edge_type == "EXON_CONTINUATION"
    assert (bridge.src_node_type, bridge.dst_node_type) == ("acceptor", "donor")
    assert bridge.length_bp == 0
    retained_atoms = retained.loc[retained["edge_type"].eq("RETAINED_INTRON")]
    assert sorted(
        retained_atoms[["start_0based", "end_0based_exclusive"]]
        .itertuples(index=False, name=None)
    ) == [(150, 200), (200, 250)]
    assert tables.paths.set_index("path_id").loc["retained", "path_length_bp"] == 220


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


def test_configured_backed_gene_cache_keeps_each_shard_resident(
    tmp_path, monkeypatch
):
    first = make_toy_genes()[0]
    genes = tuple(
        replace(first, gene_id=f"TOY_RESIDENT_{index}") for index in range(3)
    )
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
                "input_manifest_id": "resident-fixture",
                "compatibility_artifact_id": "resident-compatible-fixture",
                "informative_gene_ids": [gene.gene_id for gene in genes],
                "gene_shards": records,
            }
        )
    )

    original_load = torch.load
    shard_loads = []

    def counted_load(path, *args, **kwargs):
        if Path(path).parent.name == "genes":
            shard_loads.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    backed = BackedPreparedDataset.load(tmp_path, gene_cache_capacity=len(genes))
    first_pass = {
        index: gene
        for index, gene in _iter_gene_order(
            backed.genes, [2, 0, 1], prefetch=True
        )
    }
    second_pass = {
        index: gene
        for index, gene in _iter_gene_order(
            backed.genes, [0, 1, 2], prefetch=True
        )
    }

    assert sorted(shard_loads) == [f"gene_{index}.pt" for index in range(3)]
    assert all(second_pass[index] is first_pass[index] for index in range(3))
    assert backed.genes.cache_capacity == len(genes)
    assert backed.genes._load.cache_info().currsize == len(genes)


def test_backed_gene_cache_capacity_is_a_positive_resource_contract():
    config = load_config("configs/fabric_v2_toy.yaml")
    config["resources"]["backed_gene_cache_capacity"] = 17_600
    manifest = training_manifest_from_config(config, seed=101, condition="full")
    assert manifest.backed_gene_cache_capacity == 17_600

    config["resources"]["backed_gene_cache_capacity"] = 0
    with pytest.raises(ValueError, match="cache_capacity must be positive"):
        training_manifest_from_config(config, seed=101, condition="full")


def test_train_run_binds_configured_cache_before_first_shard_access(
    tmp_path, monkeypatch
):
    first = make_toy_genes()[0]
    genes = tuple(replace(first, gene_id=f"TOY_TRAIN_{index}") for index in range(3))
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
                "input_manifest_id": "train-cache-fixture",
                "compatibility_artifact_id": "train-cache-compatible-fixture",
                "informative_gene_ids": [gene.gene_id for gene in genes],
                "gene_shards": records,
            }
        )
    )
    prepared = BackedPreparedDataset.load(tmp_path)
    original_load = torch.load
    shard_loads = []

    def counted_load(path, *args, **kwargs):
        if Path(path).parent.name == "genes":
            shard_loads.append(Path(path).name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    config = load_config("configs/fabric_v2_toy.yaml")
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["resources"]["backed_gene_cache_capacity"] = len(genes)
    resident_run = train_run(
        prepared, config, seed=101, condition="full", device="cpu"
    )

    assert sorted(shard_loads) == [f"gene_{index}.pt" for index in range(3)]
    assert prepared.genes.cache_capacity == len(genes)
    assert prepared.genes._load.cache_info().currsize == len(genes)

    streaming = BackedPreparedDataset.load(tmp_path, gene_cache_capacity=2)
    streaming_config = deepcopy(config)
    streaming_config["resources"]["backed_gene_cache_capacity"] = 2
    streaming_run = train_run(
        streaming, streaming_config, seed=101, condition="full", device="cpu"
    )
    pd.testing.assert_frame_equal(
        resident_run.result.history,
        streaming_run.result.history,
        check_exact=True,
    )
    for name, value in resident_run.result.model.state_dict().items():
        torch.testing.assert_close(
            value,
            streaming_run.result.model.state_dict()[name],
            atol=0,
            rtol=0,
        )


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
