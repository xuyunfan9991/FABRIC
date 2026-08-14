"""Freeze the single full-cohort runtime readiness record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from .train import FULL_COHORT_SCOPE, load_config


def _tree_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def build_readiness_report(
    *,
    real_root: str | Path,
    compatible_root: str | Path,
    config_path: str | Path,
    validation_path: str | Path,
    profile_path: str | Path,
    test_report_path: str | Path,
    launch_command: str,
) -> dict[str, object]:
    root = Path(real_root).resolve()
    compatible = Path(compatible_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validation = json.loads(Path(validation_path).read_text())
    profile = json.loads(Path(profile_path).read_text())
    tests = json.loads(Path(test_report_path).read_text())
    prepared = json.loads(
        (root / "prepared_dataset" / "PreparedDatasetManifest.json").read_text()
    )
    source = json.loads((root / "SourceValidation.json").read_text())

    if config["execution"] != {
        "scope": FULL_COHORT_SCOPE,
        "training_authorized": False,
        "final_test_authorized": False,
    }:
        raise ValueError("full-cohort readiness config execution identity differs")
    if config["resources"].get("profile_status") != "FROZEN_REAL_FULL_SHAPE_PROFILE":
        raise ValueError("real full-shape profile is not frozen in the config")
    if validation.get("status") != "ADMITTED_FOR_PRELAUNCH":
        raise ValueError("strict real-dataset validation is not admitted")
    if (
        profile.get("scope") != FULL_COHORT_SCOPE
        or profile.get("profiled_condition") != "full"
    ):
        raise ValueError("resource profile identity differs")
    training = config["training"]
    if (
        profile.get("schema_version") != "fabric.real_full_shape_profile.v3"
        or profile.get("status") != "FROZEN_REAL_FULL_SHAPE_PROFILE"
        or profile.get("train_sampling_unit") != training["primary_epoch_unit"]
        or profile.get("max_train_gene_cells_per_gene_per_epoch")
        != training["max_train_gene_cells_per_gene_per_epoch"]
        or profile.get("resample_train_gene_cells_each_epoch")
        is not training["resample_train_gene_cells_each_epoch"]
        or profile.get("selected_gene_cell_ec_rows")
        != training["selected_gene_cell_ec_rows"]
        or profile.get("sampling_estimator") != training["sampling_estimator"]
        or profile.get("optimizer_step_unit") != training["optimizer_step_unit"]
        or profile.get("gene_microbatch_gradient_accumulation")
        is not training["gene_microbatch_gradient_accumulation"]
        or profile.get("batch_policy") != config["resources"]["batch_policy"]
        or profile.get("compute_precision") != config["resources"]["compute_precision"]
        or profile.get("prefetch_backed_gene_shards")
        is not config["resources"]["prefetch_backed_gene_shards"]
        or profile.get("target_gpu_allocated_bytes")
        != config["resources"]["target_gpu_allocated_bytes"]
        or profile.get("unmodeled_gpu_reserve_bytes")
        != config["resources"]["unmodeled_gpu_reserve_bytes"]
        or profile.get("max_cells_per_gpu_batch")
        != config["resources"]["max_cells_per_gpu_batch"]
        or profile.get("train_bytes_per_shape_element")
        != config["resources"]["train_bytes_per_shape_element"]
        or profile.get("evaluation_bytes_per_shape_element")
        != config["resources"]["evaluation_bytes_per_shape_element"]
        or profile.get("projected_optimizer_step_count_per_epoch") != 17_600
        or profile.get("epoch_evaluation_policy")
        != "one_complete_validation_no_complete_train"
        or profile.get("epoch_core_metrics")
        != ["validation_compatible_path_nll", "ont_matrix_kl_count_weighted"]
        or profile.get("checkpoint_selection_metric")
        != "validation_compatible_path_nll"
        or profile.get("reporting_only_metric") != "ont_matrix_kl_count_weighted"
        or profile.get("ont_validation_target_root") != config["monitor"]["target_root"]
        or profile.get("adaptive_memory_estimate_upper_bounds_all_profiled_batches")
        is not True
    ):
        raise ValueError("resource profile train sampling identity differs")
    if any(
        profile.get(name) is not False
        for name in (
            "optimizer_constructed",
            "optimizer_step_called",
            "checkpoint_written",
            "test_rows_or_test_statistics_read",
            "test_predictions_or_metrics_computed",
        )
    ):
        raise ValueError("prelaunch profile crossed a forbidden execution boundary")
    if (
        tests.get("status") != "PASS"
        or tests.get("real_full_shape_profile_pass") is not True
    ):
        raise ValueError("prelaunch test report is not complete")
    if (
        prepared.get("g_fit_gene_count") != 17_600
        or prepared.get("test_compatible_rows") != 0
        or source.get("historical_7198_graph_or_ec_used") is not False
        or source.get("historical_167235_split_used") is not False
    ):
        raise ValueError("prepared/source scope differs from the frozen V2 cohort")

    stat = os.statvfs(root)
    epoch_hours = float(profile["projected_hours_per_epoch"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    rerun_reserve_fraction = float(
        config["resources"].get("rerun_reserve_fraction", 0.0)
    )
    if not 0 <= rerun_reserve_fraction <= 1:
        raise ValueError("rerun reserve fraction must lie in [0, 1]")
    report = {
        "schema_version": "fabric.full_cohort_readiness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_AWAITING_TRAINING_AUTHORIZATION",
        "contract": "FABRIC_ARCHITECTURE_V2",
        "scope": FULL_COHORT_SCOPE,
        "allowed_conditions": ["full", "atac", "rbp"],
        "seed": "USER_SELECTED_PER_COMMAND",
        "lambda_base": float(config["optimizer"]["lambda_base"]),
        "lambda_int": float(config["optimizer"]["lambda_int"]),
        "optimizer_policy": {
            "family": config["optimizer"]["family"],
            "learning_rate": float(config["optimizer"]["learning_rate"]),
            "lr_scheduler": dict(config["optimizer"]["lr_scheduler"]),
            "gradient_clip_norm": float(config["optimizer"]["gradient_clip_norm"]),
            "scheduler_selection_metric": "validation_compatible_path_nll",
            "ont_matrix_kl_controls_scheduler": False,
            "test_controls_scheduler": False,
        },
        "g_fit_gene_count": 17_600,
        "structural_candidate_count": 17_706,
        "structural_path_count": 90_672,
        "cell_count": 217_933,
        "graph_only_gene_count": 106,
        "validation_report": str(Path(validation_path).resolve()),
        "resource_profile": str(Path(profile_path).resolve()),
        "test_report": str(Path(test_report_path).resolve()),
        "config": str(config_path),
        "prepared_dataset": str((root / "prepared_dataset").resolve()),
        "resource_freeze": {
            "gpu_count_for_run": 1,
            "gpu_name": profile["gpu_name"],
            "gpu_total_memory_bytes": int(profile["gpu_total_memory_bytes"]),
            "profile_peak_gpu_allocated_bytes": max(
                int(value["peak_gpu_allocated_bytes"])
                for value in profile["profile_batches"]
            ),
            "profile_process_peak_rss_bytes": int(profile["process_peak_rss_bytes"]),
            "validation_snapshot_tensor_upper_bound_bytes": int(
                profile["validation_snapshot_tensor_upper_bound_bytes"]
            ),
            "frozen_host_ram_requirement_bytes": int(
                profile["frozen_host_ram_requirement_bytes"]
            ),
            "batch_policy": profile["batch_policy"],
            "compute_precision": profile["compute_precision"],
            "prefetch_backed_gene_shards": profile["prefetch_backed_gene_shards"],
            "target_gpu_allocated_bytes": int(profile["target_gpu_allocated_bytes"]),
            "target_gpu_memory_fraction": float(profile["target_gpu_memory_fraction"]),
            "max_cells_per_gpu_batch": int(profile["max_cells_per_gpu_batch"]),
            "train_bytes_per_shape_element": float(
                profile["train_bytes_per_shape_element"]
            ),
            "evaluation_bytes_per_shape_element": float(
                profile["evaluation_bytes_per_shape_element"]
            ),
            "train_sampling_unit": profile["train_sampling_unit"],
            "max_train_gene_cells_per_gene_per_epoch": int(
                profile["max_train_gene_cells_per_gene_per_epoch"]
            ),
            "resample_train_gene_cells_each_epoch": profile[
                "resample_train_gene_cells_each_epoch"
            ],
            "selected_gene_cell_ec_rows": profile["selected_gene_cell_ec_rows"],
            "sampling_estimator": profile["sampling_estimator"],
            "optimizer_step_unit": profile["optimizer_step_unit"],
            "gene_microbatch_gradient_accumulation": profile[
                "gene_microbatch_gradient_accumulation"
            ],
            "projected_optimizer_step_count_per_epoch": int(
                profile["projected_optimizer_step_count_per_epoch"]
            ),
            "epoch_evaluation_policy": profile["epoch_evaluation_policy"],
            "epoch_core_metrics": profile["epoch_core_metrics"],
            "projected_available_train_gene_cell_count": int(
                profile["projected_available_train_gene_cell_count"]
            ),
            "projected_sampled_train_gene_cell_count": int(
                profile["projected_sampled_train_gene_cell_count"]
            ),
            "projected_train_gene_cell_sampling_fraction": float(
                profile["projected_train_gene_cell_sampling_fraction"]
            ),
            "maximum_profiled_route_projection_elements": max(
                int(value["route_projection_elements"])
                for value in profile["profile_batches"]
            ),
            "minimum_adaptive_train_cell_batch_limit": int(
                profile["minimum_adaptive_train_cell_batch_limit"]
            ),
            "maximum_adaptive_train_cell_batch_limit": int(
                profile["maximum_adaptive_train_cell_batch_limit"]
            ),
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "projected_hours_per_epoch": epoch_hours,
            "projected_epoch_excludes_optimizer_step_and_gradient_clipping": True,
            "projected_train_forward_backward_hours_per_epoch": float(
                profile["projected_train_forward_backward_seconds"]
            )
            / 3_600.0,
            "projected_validation_evaluation_hours_per_epoch": float(
                profile["projected_validation_evaluation_seconds"]
            )
            / 3_600.0,
            "projected_host_load_hours_per_epoch": float(
                profile["projected_host_load_seconds_per_epoch"]
            )
            / 3_600.0,
            "projected_hidden_host_load_hours_per_epoch": float(
                profile["projected_hidden_host_load_seconds_per_epoch"]
            )
            / 3_600.0,
            "projected_hours_through_first_patience_stop": epoch_hours * (patience + 1),
            "projected_max_hours": epoch_hours * max_epochs,
            "rerun_reserve_fraction": rerun_reserve_fraction,
            "projected_max_gpu_hours_with_rerun_reserve": (
                epoch_hours * max_epochs * (1.0 + rerun_reserve_fraction)
            ),
            "real_dataset_storage_bytes": _tree_bytes(root),
            "prepared_runtime_storage_bytes": _tree_bytes(root / "prepared_dataset"),
            "compatible_ec_storage_bytes": _tree_bytes(compatible),
            "run_output_storage_reserve_bytes": 2 << 30,
            "available_storage_bytes_at_freeze": stat.f_bavail * stat.f_frsize,
        },
        "launch_command_template": launch_command,
        "launch_executed": False,
        "real_training_or_full_epoch_executed": False,
        "lambda_tuning_executed": False,
        "ablation_or_architecture_comparator_executed": False,
        "test_compatible_rows": "ABSENT_NOT_MATERIALIZED",
        "test_predictions_or_metrics_computed": False,
        "training_authorized": False,
        "final_test_authorized": False,
        "remaining_blockers": [
            "execution.training_authorized remains false; no training command may run"
        ],
    }
    return report


def _markdown(report: dict[str, object]) -> str:
    resource = report["resource_freeze"]
    optimizer = report["optimizer_policy"]
    scheduler = optimizer["lr_scheduler"]
    scheduler_text = str(scheduler["name"])
    if scheduler["name"] == "reduce_on_plateau":
        scheduler_text += (
            f"(factor={scheduler['factor']}, patience={scheduler['patience']}, "
            f"min_lr={scheduler['min_lr']})"
        )
    return "\n".join(
        [
            "# FABRIC V2 full-cohort runtime readiness",
            "",
            f"Status: `{report['status']}`",
            "",
            "- Scope: `full_cohort`; each command independently selects `full`, `atac`, or `rbp` and one integer seed.",
            "- Universe: 17,600 G_fit genes; 17,706 candidates; 90,672 paths; 217,933 cells; 106 graph-only audit genes.",
            f"- Penalties: lambda_base={report['lambda_base']}, lambda_int={report['lambda_int']}; read directly from the selected config.",
            f"- Optimizer: {optimizer['family']}; learning rate {optimizer['learning_rate']}; scheduler {scheduler_text}; global gradient clip norm {optimizer['gradient_clip_norm']}.",
            f"- GPU batch policy: {resource['batch_policy']}; target allocation {resource['target_gpu_allocated_bytes'] / 2**30:.2f} GiB ({resource['target_gpu_memory_fraction']:.1%} of the card), with gene-shape-adaptive cell packing.",
            f"- Compute precision: {resource['compute_precision']}; TF32 matmul is not enabled because the real-shape numerical audit exceeded the frozen probability/gradient tolerance.",
            f"- Backed-shard prefetch: {resource['prefetch_backed_gene_shards']}; projected hidden host-load time {resource['projected_hidden_host_load_hours_per_epoch']:.2f} h of {resource['projected_host_load_hours_per_epoch']:.2f} h raw I/O per epoch.",
            f"- Train sampling: at most {resource['max_train_gene_cells_per_gene_per_epoch']:,} complete gene-cell groups per gene per epoch; deterministic epoch-wise resampling uses the command seed; all informative EC rows retained and Horvitz–Thompson scaled.",
            f"- Scheduled train gene-cells: {resource['projected_sampled_train_gene_cell_count']:,} of {resource['projected_available_train_gene_cell_count']:,} per epoch ({resource['projected_train_gene_cell_sampling_fraction']:.2%}); validation remains complete.",
            f"- Worst profiled dynamic route projection: {resource['maximum_profiled_route_projection_elements']:,} float elements.",
            f"- Training cap: {resource['max_epochs']} epochs; early-stopping patience {resource['early_stopping_patience']}.",
            f"- Projected wall time: {resource['projected_hours_per_epoch']:.2f} h/epoch; {resource['projected_hours_through_first_patience_stop']:.2f} h through the earliest patience stop; {resource['projected_max_hours']:.2f} h maximum.",
            "- Timing boundary: the prelaunch projection excludes AdamW optimizer-step and gradient-clipping wall time; actual end-to-end epoch time must come from the authorized run log.",
            f"- GPU: one {resource['gpu_name']}; profiled peak allocation {resource['profile_peak_gpu_allocated_bytes'] / 2**30:.2f} GiB of {resource['gpu_total_memory_bytes'] / 2**30:.2f} GiB.",
            f"- Host RAM: profiled process peak {resource['profile_process_peak_rss_bytes'] / 2**30:.2f} GiB; validation snapshot tensor upper bound {resource['validation_snapshot_tensor_upper_bound_bytes'] / 2**30:.2f} GiB; frozen requirement {resource['frozen_host_ram_requirement_bytes'] / 2**30:.0f} GiB.",
            f"- Storage: real package {resource['real_dataset_storage_bytes'] / 2**30:.2f} GiB; runtime PreparedDataset {resource['prepared_runtime_storage_bytes'] / 2**30:.2f} GiB; compatible EC {resource['compatible_ec_storage_bytes'] / 2**30:.2f} GiB; run-output reserve 2.00 GiB.",
            f"- Rerun reserve: {resource['rerun_reserve_fraction']:.0%}; maximum GPU-hours including reserve {resource['projected_max_gpu_hours_with_rerun_reserve']:.2f}.",
            "- Test boundary: compatible test rows absent; no test predictions or metrics computed.",
            "- Authorization: training false; final test false until separately changed.",
            "- Remaining blocker: execution.training_authorized remains false; this report does not authorize launch.",
            "",
            "Launch command template (not executed):",
            "",
            "```bash",
            str(report["launch_command_template"]),
            "```",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--compatible-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--launch-command", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args(argv)
    report = build_readiness_report(
        real_root=args.real_root,
        compatible_root=args.compatible_root,
        config_path=args.config,
        validation_path=args.validation,
        profile_path=args.profile,
        test_report_path=args.tests,
        launch_command=args.launch_command,
    )
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.output_markdown).write_text(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
