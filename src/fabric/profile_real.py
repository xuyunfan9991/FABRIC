"""Profile real full-shape FABRIC V2 batches without an optimizer or checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import nnls

from .likelihood import compatible_path_nll
from .model import FABRICV2Model
from .train import (
    BackedPreparedDataset,
    FULL_COHORT_SCOPE,
    _gene_shape_components,
    _model_spec,
    _plan_gene_cell_batches,
    _subset_gene_cells,
    load_config,
    rows_for_split,
)


def _record_batch_cell_limit(
    value: dict[str, object],
    *,
    available_cells: int,
    resources: dict[str, object],
    model_config: dict[str, object],
    cis_dim: int,
    phase: str,
) -> int:
    if available_cells < 1:
        return 0
    static, per_cell, per_row = _gene_shape_components(
        edge_count=int(value["edge_count"]),
        path_count=int(value["path_count"]),
        route_count=int(value["dna_route_count"]) + int(value["rna_route_count"]),
        compatible_width=int(value["path_count"]),
        cis_dim=cis_dim,
        dynamic_dim=int(model_config["dynamic_projection_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        path_hidden_dim=int(model_config["path_hidden_dim"]),
    )
    multiplier = float(resources[f"{phase}_bytes_per_shape_element"])
    capacity = (
        int(resources["target_gpu_allocated_bytes"])
        - int(resources["unmodeled_gpu_reserve_bytes"])
    ) / multiplier - static
    rows_per_cell = float(value["ec_row_count"]) / max(int(value["cell_count"]), 1)
    limit = int(capacity // (per_cell + rows_per_cell * per_row))
    if limit < 1:
        raise RuntimeError(
            f"gene {value['gene_id']} one-cell {phase} shape exceeds the GPU target"
        )
    return min(available_cells, limit, int(resources["max_cells_per_gpu_batch"]))


def _select_profile_records(
    records: Sequence[dict[str, object]],
    *,
    resources: dict[str, object],
    model_config: dict[str, object],
    cis_dim: int,
    max_train_gene_cells_per_gene_per_epoch: int | None = None,
) -> list[dict[str, object]]:
    def scheduled_train_cells(value: dict[str, object]) -> int:
        available = int(value["train_cell_count"])
        if max_train_gene_cells_per_gene_per_epoch is None:
            return available
        return min(available, max_train_gene_cells_per_gene_per_epoch)

    def route_projection_elements(value: dict[str, object]) -> int:
        cells = _record_batch_cell_limit(
            value,
            available_cells=scheduled_train_cells(value),
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="train",
        )
        routes = int(value["dna_route_count"]) + int(value["rna_route_count"])
        return cells * routes * int(model_config["dynamic_projection_dim"])

    def train_batch_cells(value: dict[str, object]) -> int:
        return _record_batch_cell_limit(
            value,
            available_cells=scheduled_train_cells(value),
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="train",
        )

    def evaluation_batch_cells(value: dict[str, object]) -> int:
        return _record_batch_cell_limit(
            value,
            available_cells=int(value["validation_cell_count"]),
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="evaluation",
        )

    selectors = {
        "maximum_edges": max(records, key=lambda value: int(value["edge_count"])),
        "maximum_active_routes": max(
            records,
            key=lambda value: int(value["dna_route_count"]) + int(value["rna_route_count"]),
        ),
        "maximum_route_projection_elements": max(
            records, key=route_projection_elements
        ),
        "maximum_cells": max(records, key=lambda value: int(value["cell_count"])),
        "maximum_train_adaptive_batch": max(records, key=train_batch_cells),
        "maximum_validation_adaptive_batch": max(
            records, key=evaluation_batch_cells
        ),
    }
    ordered = sorted(records, key=lambda value: int(value["edge_count"]))
    selectors["median_edges"] = ordered[len(ordered) // 2]
    metrics = {
        "attention_compute_elements": lambda value: train_batch_cells(value)
        * int(value["edge_count"]) ** 2,
        "route_projection_elements": route_projection_elements,
        "cell_edge_elements": lambda value: train_batch_cells(value)
        * int(value["edge_count"]),
        "estimated_ec_rows": lambda value: train_batch_cells(value)
        * float(value.get("ec_row_count", value["cell_count"]))
        / max(int(value["cell_count"]), 1),
    }
    for metric_name, metric in metrics.items():
        ranked = sorted(records, key=metric)
        for quantile in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            position = round(quantile * (len(ranked) - 1))
            selectors[
                f"{metric_name}_q{int(quantile * 100):02d}"
            ] = ranked[position]
    result = []
    position_by_gene: dict[str, int] = {}
    for selector, record in selectors.items():
        gene_id = str(record["gene_id"])
        if gene_id in position_by_gene:
            result[position_by_gene[gene_id]]["profile_selectors"].append(selector)
            continue
        position_by_gene[gene_id] = len(result)
        result.append(
            {
                **record,
                "profile_selector": selector,
                "profile_selectors": [selector],
            }
        )
    return result


def _workload_vector(
    *,
    cells: int,
    edges: int,
    routes: int,
    ec_rows: float,
    dynamic_dim: int,
) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            cells / 10_000.0,
            edges / 1_000.0,
            cells * routes * dynamic_dim / 100_000_000.0,
            ec_rows / 10_000.0,
        ],
        dtype=np.float64,
    )


def _fit_conservative_nonnegative_cost_model(
    matrix: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, float, dict[str, object]]:
    coefficients, residual_norm = nnls(matrix, values)
    fitted = matrix @ coefficients
    if not np.isfinite(fitted).all() or bool((fitted <= 0).any()):
        raise RuntimeError("resource cost model produced a non-positive fitted time")
    observed_to_fitted = values / fitted
    # Preserve the slowest measured sample, then retain explicit headroom for
    # unprofiled kernel/allocator variation without multiplying one worst batch
    # across the entire heterogeneous cohort.
    safety_factor = max(1.15, float(observed_to_fitted.max()) * 1.15)
    return coefficients, safety_factor, {
        "algorithm": "scipy.optimize.nnls",
        "feature_order": [
            "intercept",
            "cells_per_10000",
            "edge_count_per_1000",
            "route_projection_per_100_million_elements",
            "compatible_rows_per_10000",
        ],
        "coefficients_seconds": coefficients.tolist(),
        "residual_l2_seconds": float(residual_norm),
        "maximum_observed_to_fitted_ratio": float(observed_to_fitted.max()),
        "safety_factor": safety_factor,
        "all_profiled_batches_upper_bounded_after_safety": bool(
            np.all(fitted * safety_factor >= values)
        ),
    }


def _project_epoch_seconds(
    records: Sequence[dict[str, object]],
    *,
    resources: dict[str, object],
    model_config: dict[str, object],
    cis_dim: int,
    dynamic_dim: int,
    max_train_gene_cells_per_gene_per_epoch: int,
    train_coefficients: np.ndarray,
    train_safety: float,
    evaluation_coefficients: np.ndarray,
    evaluation_safety: float,
) -> dict[str, object]:
    train_seconds = 0.0
    validation_evaluation_seconds = 0.0
    train_batches = 0
    validation_batches = 0
    available_train_gene_cells = 0
    sampled_train_gene_cells = 0
    for value in records:
        edges = int(value["edge_count"])
        routes = int(value["dna_route_count"]) + int(value["rna_route_count"])
        ec_per_cell = float(value["ec_row_count"]) / max(
            int(value["cell_count"]), 1
        )
        full_train_cells = int(value["train_cell_count"])
        scheduled_train_cells = min(
            full_train_cells, max_train_gene_cells_per_gene_per_epoch
        )
        available_train_gene_cells += full_train_cells
        sampled_train_gene_cells += scheduled_train_cells
        train_cell_limit = _record_batch_cell_limit(
            value,
            available_cells=scheduled_train_cells,
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="train",
        )
        for start in range(0, scheduled_train_cells, train_cell_limit):
            cells = min(train_cell_limit, scheduled_train_cells - start)
            features = _workload_vector(
                cells=cells,
                edges=edges,
                routes=routes,
                ec_rows=cells * ec_per_cell,
                dynamic_dim=dynamic_dim,
            )
            train_batches += 1
            train_seconds += float(features @ train_coefficients) * train_safety
        validation_cells = int(value["validation_cell_count"])
        if validation_cells == 0:
            continue
        evaluation_cell_limit = _record_batch_cell_limit(
            value,
            available_cells=validation_cells,
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="evaluation",
        )
        for start in range(0, validation_cells, evaluation_cell_limit):
            cells = min(evaluation_cell_limit, validation_cells - start)
            features = _workload_vector(
                cells=cells,
                edges=edges,
                routes=routes,
                ec_rows=cells * ec_per_cell,
                dynamic_dim=dynamic_dim,
            )
            validation_batches += 1
            validation_evaluation_seconds += (
                float(features @ evaluation_coefficients) * evaluation_safety
            )
    return {
        "projected_available_train_gene_cell_count": available_train_gene_cells,
        "projected_sampled_train_gene_cell_count": sampled_train_gene_cells,
        "projected_train_gene_cell_sampling_fraction": (
            sampled_train_gene_cells / available_train_gene_cells
        ),
        "projected_train_batch_count_per_epoch": train_batches,
        "projected_optimizer_step_count_per_epoch": len(records),
        "projected_validation_batch_count_per_epoch": validation_batches,
        "projected_train_forward_backward_seconds": train_seconds,
        "projected_validation_evaluation_seconds": validation_evaluation_seconds,
        "projected_compute_seconds_per_epoch": (
            train_seconds + validation_evaluation_seconds
        ),
    }


def profile_real_batches(
    prepared_root: str | Path,
    config_path: str | Path,
    *,
    device: str,
) -> dict[str, object]:
    prepared_root = Path(prepared_root)
    config = load_config(config_path)
    if config["execution"]["scope"] != FULL_COHORT_SCOPE:
        raise ValueError("real prelaunch profiler accepts only full_cohort")
    prepared = BackedPreparedDataset.load(prepared_root)
    manifest = json.loads((prepared_root / "PreparedDatasetManifest.json").read_text())
    records = manifest["gene_record_audit"]
    resources = dict(config["resources"])
    model_config = dict(config["model"])
    dynamic_dim = int(config["model"]["dynamic_projection_dim"])
    training = config["training"]
    gene_cell_cap = int(training["max_train_gene_cells_per_gene_per_epoch"])
    index = {
        gene_id: position for position, gene_id in enumerate(prepared.informative_gene_ids)
    }
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real full-shape profiling requires one available CUDA device")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    first = prepared.genes[0]
    cis_dim = int(first.model_input.cis_features.shape[1])
    selected = _select_profile_records(
        records,
        resources=resources,
        model_config=model_config,
        cis_dim=cis_dim,
    )
    torch.manual_seed(1103)
    model = FABRICV2Model(
        **_model_spec(first, config["model"]), readout_kind="path_context"
    ).to(torch_device)
    model.train()
    parameter_count = sum(value.numel() for value in model.parameters())
    gpu_total_bytes = int(torch.cuda.get_device_properties(torch_device).total_memory)
    if gpu_total_bytes < int(resources["target_gpu_allocated_bytes"]):
        raise RuntimeError("profile GPU is smaller than the frozen allocation target")
    gpu_free_before_profile_bytes = int(torch.cuda.mem_get_info(torch_device)[0])
    if gpu_free_before_profile_bytes < int(resources["target_gpu_allocated_bytes"]):
        raise RuntimeError("profile GPU does not have the frozen target memory available")
    maximum_edge_count = max(int(value["edge_count"]) for value in records)
    profile_rows = []
    total_wall = 0.0
    for record in selected:
        gene_id = str(record["gene_id"])
        shard_path = prepared_root / str(record["relative_path"])
        shard_bytes = shard_path.stat().st_size
        load_start = time.perf_counter()
        gene = prepared.genes[index[gene_id]]
        load_seconds = time.perf_counter() - load_start
        rows = rows_for_split(gene, "train")
        cells = torch.unique(gene.row_cell_index[rows], sorted=True)
        batch_plan = _plan_gene_cell_batches(
            gene,
            cells,
            rows,
            model_config=model_config,
            resources=resources,
            phase="train",
        )
        profile_batch_index = max(
            range(len(batch_plan.batches)),
            key=lambda position: batch_plan.estimated_bytes[position],
        )
        selected_cells = batch_plan.batches[profile_batch_index]
        batch_cells = len(selected_cells)
        selected_rows = rows[torch.isin(gene.row_cell_index[rows], selected_cells)]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch_device)
        torch.cuda.synchronize(torch_device)
        start = time.perf_counter()
        batch_input, row_cell_index = _subset_gene_cells(
            gene, selected_cells, selected_rows, model
        )
        load_to_gpu_seconds = time.perf_counter() - start
        torch.cuda.synchronize(torch_device)
        forward_start = time.perf_counter()
        output = model(batch_input, condition="full")
        details = compatible_path_nll(
            output.path_logits,
            gene.compatible_path_indices[selected_rows].to(torch_device),
            gene.compatible_path_mask[selected_rows].to(torch_device),
            gene.molecule_count[selected_rows].to(torch_device),
            row_cell_index=row_cell_index,
            return_details=True,
        )
        torch.cuda.synchronize(torch_device)
        forward_seconds = time.perf_counter() - forward_start
        backward_start = time.perf_counter()
        details.loss.backward()
        torch.cuda.synchronize(torch_device)
        backward_seconds = time.perf_counter() - backward_start
        if not torch.isfinite(output.path_logits).all() or not torch.isfinite(details.loss):
            raise FloatingPointError(f"non-finite full-shape profile for {gene_id}")
        missing_gradients = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                missing_gradients.append(name)
            elif not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite Full gradient: {gene_id}/{name}")
        if "maximum_active_routes" in record["profile_selectors"] and missing_gradients:
            raise FloatingPointError(
                f"maximum-route Full batch misses gradients: {missing_gradients}"
            )
        peak_bytes = int(torch.cuda.max_memory_allocated(torch_device))
        wall = load_to_gpu_seconds + forward_seconds + backward_seconds
        total_wall += wall
        profile_rows.append(
            {
                "profile_selector": record["profile_selector"],
                "profile_selectors": list(record["profile_selectors"]),
                "gene_id": gene_id,
                "edge_count": int(record["edge_count"]),
                "path_count": int(record["path_count"]),
                "available_train_cells": len(cells),
                "scheduled_train_cells": min(len(cells), gene_cell_cap),
                "profile_shape_role": (
                    "complete_evaluation_shape_with_backward_upper_bound"
                ),
                "profile_batch_cells": batch_cells,
                "profile_ec_rows": len(selected_rows),
                "dna_route_count": int(record["dna_route_count"]),
                "rna_route_count": int(record["rna_route_count"]),
                "attention_elements": batch_cells * int(record["edge_count"]) ** 2,
                "route_projection_elements": batch_cells
                * (
                    int(record["dna_route_count"])
                    + int(record["rna_route_count"])
                )
                * dynamic_dim,
                "cell_edge_elements": batch_cells * int(record["edge_count"]),
                "adaptive_estimated_gpu_bytes": batch_plan.estimated_bytes[
                    profile_batch_index
                ],
                "adaptive_plan_batch_count": len(batch_plan.batches),
                "adaptive_per_cell_shape_elements": (
                    batch_plan.per_cell_shape_elements
                ),
                "adaptive_per_compatible_row_shape_elements": (
                    batch_plan.per_compatible_row_shape_elements
                ),
                "host_shard_bytes": shard_bytes,
                "host_shard_load_seconds": load_seconds,
                "host_to_gpu_seconds": load_to_gpu_seconds,
                "full_forward_plus_nll_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "batch_wall_seconds": wall,
                "peak_gpu_allocated_bytes": peak_bytes,
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
                "compatible_nll": float(details.loss.detach().cpu()),
                "informative_molecule_mass": float(details.molecule_mass.detach().cpu()),
                "finite_logits_loss_gradients": True,
                "parameters_without_gene_local_gradient": missing_gradients,
                "optimizer_constructed": False,
                "optimizer_step_called": False,
                "checkpoint_written": False,
            }
        )
    workload_matrix = np.stack(
        [
            _workload_vector(
                cells=int(value["profile_batch_cells"]),
                edges=int(value["edge_count"]),
                routes=int(value["dna_route_count"])
                + int(value["rna_route_count"]),
                ec_rows=float(value["profile_ec_rows"]),
                dynamic_dim=dynamic_dim,
            )
            for value in profile_rows
        ]
    )
    train_values = np.asarray(
        [value["batch_wall_seconds"] for value in profile_rows], dtype=np.float64
    )
    evaluation_values = np.asarray(
        [
            value["host_to_gpu_seconds"]
            + value["full_forward_plus_nll_seconds"]
            for value in profile_rows
        ],
        dtype=np.float64,
    )
    train_coefficients, train_safety, train_model = (
        _fit_conservative_nonnegative_cost_model(workload_matrix, train_values)
    )
    evaluation_coefficients, evaluation_safety, evaluation_model = (
        _fit_conservative_nonnegative_cost_model(workload_matrix, evaluation_values)
    )
    projection = _project_epoch_seconds(
        records,
        resources=resources,
        model_config=model_config,
        cis_dim=cis_dim,
        dynamic_dim=dynamic_dim,
        max_train_gene_cells_per_gene_per_epoch=gene_cell_cap,
        train_coefficients=train_coefficients,
        train_safety=train_safety,
        evaluation_coefficients=evaluation_coefficients,
        evaluation_safety=evaluation_safety,
    )
    load_matrix = np.asarray(
        [
            [1.0, float(value["host_shard_bytes"]) / (1 << 30)]
            for value in profile_rows
        ],
        dtype=np.float64,
    )
    load_values = np.asarray(
        [value["host_shard_load_seconds"] for value in profile_rows],
        dtype=np.float64,
    )
    load_coefficients, load_safety, load_model = (
        _fit_conservative_nonnegative_cost_model(load_matrix, load_values)
    )
    all_shard_bytes = [
        (prepared_root / str(value["relative_path"])).stat().st_size
        for value in records
    ]
    host_load_seconds_per_traversal = float(
        np.asarray(
            [len(records), sum(all_shard_bytes) / (1 << 30)],
            dtype=np.float64,
        )
        @ load_coefficients
    ) * load_safety
    projected_host_load_seconds_per_epoch = 2.0 * host_load_seconds_per_traversal
    maximum_shard_load_seconds = float(
        np.asarray(
            [1.0, max(all_shard_bytes) / (1 << 30)], dtype=np.float64
        )
        @ load_coefficients
    ) * load_safety
    train_compute_seconds = float(projection["projected_train_forward_backward_seconds"])
    validation_compute_seconds = float(
        projection["projected_validation_evaluation_seconds"]
    )
    if resources["prefetch_backed_gene_shards"] is True:
        remaining_host_load_seconds = max(
            0.0, host_load_seconds_per_traversal - maximum_shard_load_seconds
        )
        projected_train_traversal_seconds = maximum_shard_load_seconds + max(
            train_compute_seconds, remaining_host_load_seconds
        )
        projected_validation_traversal_seconds = maximum_shard_load_seconds + max(
            validation_compute_seconds, remaining_host_load_seconds
        )
    else:
        projected_train_traversal_seconds = (
            train_compute_seconds + host_load_seconds_per_traversal
        )
        projected_validation_traversal_seconds = (
            validation_compute_seconds + host_load_seconds_per_traversal
        )
    epoch_seconds = (
        projected_train_traversal_seconds
        + projected_validation_traversal_seconds
    )
    projected_hidden_host_load_seconds_per_epoch = (
        float(projection["projected_compute_seconds_per_epoch"])
        + projected_host_load_seconds_per_epoch
        - epoch_seconds
    )
    # The validation snapshot is the sole complete evaluation object at epoch
    # end.  This strict tensor upper bound pessimistically pads every compatible
    # row to the complete per-gene path count; actual ordered compatible widths
    # are no larger.  Python container overhead is covered by the 2x RAM reserve.
    validation_snapshot_tensor_upper_bound_bytes = sum(
        int(value["cell_count"]) * int(value["path_count"]) * 4
        + int(value["ec_row_count"])
        * (int(value["path_count"]) * (8 + 1) + 8 + 4)
        for value in records
    )
    maximum_prepared_shard_bytes = max(all_shard_bytes)
    profiled_process_peak_rss_bytes = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    host_ram_working_upper_bound_bytes = (
        profiled_process_peak_rss_bytes
        + validation_snapshot_tensor_upper_bound_bytes
        + maximum_prepared_shard_bytes
    )
    frozen_host_ram_requirement_bytes = max(
        32 << 30, 2 * host_ram_working_upper_bound_bytes
    )
    batch_cell_limits = [
        _record_batch_cell_limit(
            value,
            available_cells=min(int(value["train_cell_count"]), gene_cell_cap),
            resources=resources,
            model_config=model_config,
            cis_dim=cis_dim,
            phase="train",
        )
        for value in records
    ]
    maximum_profile_peak_bytes = max(
        int(value["peak_gpu_allocated_bytes"]) for value in profile_rows
    )
    memory_estimate_violations = [
        str(value["gene_id"])
        for value in profile_rows
        if int(value["peak_gpu_allocated_bytes"])
        > int(value["adaptive_estimated_gpu_bytes"])
    ]
    if memory_estimate_violations:
        raise RuntimeError(
            "adaptive GPU estimator under-bounds profiled genes: "
            + ",".join(memory_estimate_violations)
        )
    if maximum_profile_peak_bytes > int(resources["target_gpu_allocated_bytes"]):
        raise RuntimeError("profiled batch exceeded the frozen GPU allocation target")
    return {
        "schema_version": "fabric.real_full_shape_profile.v3",
        "status": "FROZEN_REAL_FULL_SHAPE_PROFILE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": FULL_COHORT_SCOPE,
        "profiled_condition": "full",
        "profile_initialization_seed": 1103,
        "epoch_evaluation_policy": "one_complete_validation_no_complete_train",
        "epoch_core_metrics": [
            "validation_compatible_path_nll",
            "ont_matrix_kl_count_weighted",
        ],
        "checkpoint_selection_metric": "validation_compatible_path_nll",
        "reporting_only_metric": "ont_matrix_kl_count_weighted",
        "ont_validation_target_root": config["monitor"]["target_root"],
        "train_sampling_unit": training["primary_epoch_unit"],
        "max_train_gene_cells_per_gene_per_epoch": gene_cell_cap,
        "resample_train_gene_cells_each_epoch": training[
            "resample_train_gene_cells_each_epoch"
        ],
        "selected_gene_cell_ec_rows": training["selected_gene_cell_ec_rows"],
        "sampling_estimator": training["sampling_estimator"],
        "optimizer_step_unit": training["optimizer_step_unit"],
        "gene_microbatch_gradient_accumulation": training[
            "gene_microbatch_gradient_accumulation"
        ],
        "device": str(torch_device),
        "gpu_name": torch.cuda.get_device_name(torch_device),
        "gpu_total_memory_bytes": gpu_total_bytes,
        "gpu_free_before_profile_bytes": gpu_free_before_profile_bytes,
        "model_parameter_count": parameter_count,
        "batch_policy": resources["batch_policy"],
        "compute_precision": resources["compute_precision"],
        "prefetch_backed_gene_shards": resources["prefetch_backed_gene_shards"],
        "target_gpu_allocated_bytes": resources["target_gpu_allocated_bytes"],
        "unmodeled_gpu_reserve_bytes": resources["unmodeled_gpu_reserve_bytes"],
        "max_cells_per_gpu_batch": resources["max_cells_per_gpu_batch"],
        "train_bytes_per_shape_element": resources[
            "train_bytes_per_shape_element"
        ],
        "evaluation_bytes_per_shape_element": resources[
            "evaluation_bytes_per_shape_element"
        ],
        "maximum_edge_count": maximum_edge_count,
        "target_gpu_memory_fraction": (
            int(resources["target_gpu_allocated_bytes"]) / gpu_total_bytes
        ),
        "maximum_profile_peak_gpu_allocated_bytes": maximum_profile_peak_bytes,
        "maximum_profile_peak_fraction_of_gpu": (
            maximum_profile_peak_bytes / gpu_total_bytes
        ),
        "adaptive_memory_estimate_upper_bounds_all_profiled_batches": True,
        "minimum_adaptive_train_cell_batch_limit": min(batch_cell_limits),
        "maximum_adaptive_train_cell_batch_limit": max(batch_cell_limits),
        "profile_batches": profile_rows,
        "profile_batch_count": len(profile_rows),
        "profile_wall_seconds": total_wall,
        **projection,
        "train_batch_cost_model": train_model,
        "evaluation_batch_cost_model": evaluation_model,
        "host_shard_load_cost_model": {
            **load_model,
            "feature_order": ["intercept", "shard_GiB"],
            "coefficients_seconds": load_coefficients.tolist(),
        },
        "prepared_shard_bytes": sum(all_shard_bytes),
        "maximum_prepared_shard_bytes": maximum_prepared_shard_bytes,
        "validation_snapshot_tensor_upper_bound_bytes": (
            validation_snapshot_tensor_upper_bound_bytes
        ),
        "host_ram_working_upper_bound_bytes": host_ram_working_upper_bound_bytes,
        "frozen_host_ram_requirement_bytes": frozen_host_ram_requirement_bytes,
        "projected_host_load_seconds_per_traversal": (
            host_load_seconds_per_traversal
        ),
        "projected_initial_shard_load_seconds_per_traversal": (
            maximum_shard_load_seconds
        ),
        "projected_host_load_seconds_per_epoch": (
            projected_host_load_seconds_per_epoch
        ),
        "projected_hidden_host_load_seconds_per_epoch": (
            projected_hidden_host_load_seconds_per_epoch
        ),
        "projected_train_traversal_seconds_with_prefetch": (
            projected_train_traversal_seconds
        ),
        "projected_validation_traversal_seconds_with_prefetch": (
            projected_validation_traversal_seconds
        ),
        "projected_seconds_per_epoch": epoch_seconds,
        "projected_hours_per_epoch": epoch_seconds / 3_600.0,
        "process_peak_rss_bytes": profiled_process_peak_rss_bytes,
        "projection_semantics": (
            "nonnegative cost models over real workload quantiles and extrema; "
            "each model is inflated to upper-bound every profiled batch plus 15% "
            "headroom, then summed over the per-gene capped training sample, "
            "one complete validation, and two backed-shard traversals; the next "
            "immutable CPU shard is prefetched on one worker and traversal wall "
            "is the maximum of modeled compute and modeled shard load; ONT KL "
            "reuses validation logits and is covered by timing "
            "headroom; optimizer-update wall time is not measured because the "
            "prelaunch profile is forbidden to call optimizer.step"
        ),
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "checkpoint_written": False,
        "test_rows_or_test_statistics_read": False,
        "test_predictions_or_metrics_computed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    record = profile_real_batches(
        args.prepared, args.config, device=args.device
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
