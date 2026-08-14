"""Freeze the real V2 route design and assemble the sole backed dataset.

This module does not discover alternative inputs or intersect axes to make
them fit.  The static event vocabulary is split-neutral, dynamic
admission/support is fitted from train rows only, and production tensors
contain train/validation cells only because compatible test rows do not exist.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from scipy import linalg, sparse

from .dataset import (
    GateValues,
    InteractionDesign,
    ProductionModalityTensors,
    RouteBaseDesign,
    _route_context_levels,
    assemble_gene_cell_model_input,
    build_event_feature_manifest,
    build_model_injection_equivalence_index,
)
from .graph import build_gene_graph
from .source_identity import committed_source_identity
from .train import prepared_gene_from_assembly


CHROMOSOMES = tuple([f"chr{value}" for value in range(1, 23)] + ["chrX", "chrY"])
DISTANCE_BIN_BOUNDARIES = {
    "DNA": (50.0, 250.0, 500.0, 2_000.0),
    "RNA": (25.0, 100.0, 250.0, 500.0),
}
INTERACTION_SUPPORT_THRESHOLDS = {
    modality: {
        "minimum_distinct_events": 4,
        "minimum_distinct_genes": 3,
        "minimum_distinct_gate_keys": 3,
        "minimum_informative_molecules": 100.0,
    }
    for modality in ("DNA", "RNA")
}


def _clean_source_commit() -> str:
    return committed_source_identity(require_clean=True)


def _validated_real_dataset_source(root: Path) -> tuple[str, str]:
    identity_path = root / "SourceValidation.json"
    if not identity_path.is_file():
        raise FileNotFoundError("real dataset SourceValidation.json is absent")
    identity = json.loads(identity_path.read_text())
    source_commit = _clean_source_commit()
    if identity.get("source_git_commit") != source_commit:
        raise RuntimeError("real dataset source commit differs from the current source")
    return source_commit, str(identity["created_at_utc"])


def _prepared_artifact_identities(
    real_root: Path, compatible_root: Path
) -> tuple[str, str]:
    """Use frozen artifact directory names, independently of build time."""

    return real_root.name, compatible_root.name


@dataclass(frozen=True)
class BaseColumnSpec:
    name: str
    kind: str
    value: str | None = None


def _load_external_paths(path: str | Path) -> dict[str, Path]:
    source = Path(path)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise TypeError("external input manifest root must be a mapping")
    result: dict[str, Path] = {}
    for section in ("sources", "derived"):
        values = config.get(section)
        if not isinstance(values, dict):
            raise TypeError(f"external input section {section} must be a mapping")
        for name, record in values.items():
            raw = record["path"] if isinstance(record, dict) else record
            result[str(name)] = Path(str(raw))
    return result


def _read_event_shards(root: Path, table: str, *, gated: bool) -> pd.DataFrame:
    directory = root / "events" / ("gated" if gated else "") / table
    paths = [directory / f"part-{chromosome}.parquet" for chromosome in CHROMOSOMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {table} chromosome shards: {missing[:5]}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _joined_route_context(
    physical_events: pd.DataFrame,
    event_routes: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    active_only: bool,
) -> pd.DataFrame:
    events = physical_events
    if active_only:
        events = events.loc[events["model_active"].astype(bool)].copy()
    routes = event_routes.loc[
        event_routes["event_id"].astype(str).isin(events["event_id"].astype(str))
    ].copy()
    joined = routes.merge(
        events,
        on=["event_id", "target_gene_id", "modality"],
        how="left",
        validate="many_to_one",
        suffixes=("_route", "_event"),
        indicator=True,
    )
    if bool((joined["_merge"] != "both").any()):
        raise ValueError("route context references an absent physical event")
    joined = joined.drop(columns="_merge")
    joined = _route_context_levels(joined, manifest["distance_bin_boundaries"])
    joined["interaction_factor_id"] = np.where(
        joined["factor_identity_kind"].astype(str).eq("accessibility_only"),
        "OPEN_ONLY",
        joined["factor_entity_id"].astype(str),
    )
    return joined.sort_values("route_id", kind="mergesort").reset_index(drop=True)


def _base_candidate_specs(
    manifest: Mapping[str, object], modality: str
) -> tuple[BaseColumnSpec, ...]:
    config = manifest["modalities"][modality]
    specs = [
        BaseColumnSpec(f"{modality}:factor={factor}", "factor", str(factor))
        for factor in config["factor_vocabulary"]
    ]
    for field, coding in config["base_categorical_fields"].items():
        for level in coding["raw_levels"]:
            if level != coding["reference_level"]:
                specs.append(
                    BaseColumnSpec(
                        f"{modality}:{field}={level}", f"categorical:{field}", str(level)
                    )
                )
    if bool(manifest["motif_score_in_model"]):
        specs.append(
            BaseColumnSpec(f"{modality}:calibrated_motif_score", "motif_score")
        )
    if modality == "DNA":
        specs.append(BaseColumnSpec("DNA:log1p_peak_support", "peak_support"))
    for name in (
        "signed_distance_bp_scaled",
        "edge_relative_position",
        "log1p_distance_to_5prime_boundary_bp",
        "log1p_distance_to_3prime_boundary_bp",
    ):
        specs.append(BaseColumnSpec(f"{modality}:{name}", f"continuous:{name}"))
        specs.append(BaseColumnSpec(f"{modality}:{name}:available", f"available:{name}"))
    return tuple(specs)


def _encode_candidate_base(
    context: pd.DataFrame,
    manifest: Mapping[str, object],
    modality: str,
    specs: Sequence[BaseColumnSpec],
) -> sparse.csr_matrix:
    frame = context.loc[context["modality"].astype(str).eq(modality)].reset_index(drop=True)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    n = len(frame)

    def add(column: int, vector: np.ndarray) -> None:
        selected = np.flatnonzero(vector != 0)
        if len(selected):
            rows.append(selected.astype(np.int64, copy=False))
            columns.append(np.full(len(selected), column, dtype=np.int64))
            values.append(vector[selected].astype(np.float64, copy=False))

    factors = frame["interaction_factor_id"].astype(str).to_numpy()
    signed = frame["signed_distance_bp"].to_numpy(np.float64)
    edge_relative = frame["edge_relative_position"].to_numpy(np.float64)
    d5 = frame["distance_to_5prime_boundary_bp"].to_numpy(np.float64)
    d3 = frame["distance_to_3prime_boundary_bp"].to_numpy(np.float64)
    boundaries = tuple(float(value) for value in manifest["distance_bin_boundaries"][modality])
    scale = max([abs(value) for value in boundaries] + [1.0])
    continuous = {
        "signed_distance_bp_scaled": np.nan_to_num(signed / scale),
        "edge_relative_position": np.nan_to_num(edge_relative),
        "log1p_distance_to_5prime_boundary_bp": np.log1p(np.nan_to_num(d5)),
        "log1p_distance_to_3prime_boundary_bp": np.log1p(np.nan_to_num(d3)),
    }
    available = {
        "signed_distance_bp_scaled": np.isfinite(signed),
        "edge_relative_position": np.isfinite(edge_relative),
        "log1p_distance_to_5prime_boundary_bp": np.isfinite(d5),
        "log1p_distance_to_3prime_boundary_bp": np.isfinite(d3),
    }
    if any(np.isinf(value).any() for value in (signed, edge_relative, d5, d3)):
        raise ValueError("route geometry contains infinity")
    for column, spec in enumerate(specs):
        if spec.kind == "factor":
            vector = (factors == spec.value).astype(np.float64)
        elif spec.kind.startswith("categorical:"):
            field = spec.kind.split(":", 1)[1]
            vector = (frame[field].astype(str).to_numpy() == spec.value).astype(np.float64)
        elif spec.kind == "motif_score":
            vector = np.nan_to_num(
                frame["calibrated_motif_quality"].to_numpy(np.float64)
            )
        elif spec.kind == "peak_support":
            peak = frame["peak_support"].to_numpy(np.float64)
            if not np.isfinite(peak).all() or bool((peak < 0).any()):
                raise ValueError("DNA peak support is non-finite or negative")
            vector = np.log1p(peak)
        elif spec.kind.startswith("continuous:"):
            vector = continuous[spec.kind.split(":", 1)[1]]
        elif spec.kind.startswith("available:"):
            vector = available[spec.kind.split(":", 1)[1]].astype(np.float64)
        else:
            raise ValueError(f"unknown base column kind: {spec.kind}")
        if not np.isfinite(vector).all() or vector.shape != (n,):
            raise ValueError(f"invalid candidate base column {spec.name}")
        add(column, vector)
    if not rows:
        return sparse.csr_matrix((n, len(specs)), dtype=np.float64)
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n, len(specs)),
        dtype=np.float64,
    )


def _select_full_rank_base(
    gram: np.ndarray,
    specs: Sequence[BaseColumnSpec],
    *,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    factor_count = sum(spec.kind == "factor" for spec in specs)
    if factor_count < 1 or any(spec.kind != "factor" for spec in specs[:factor_count]):
        raise ValueError("complete factor baseline must lead the base design")
    factor_diag = np.diag(gram)[:factor_count]
    active_factor_indices = np.flatnonzero(factor_diag > 0).astype(np.int64)
    if not len(active_factor_indices):
        raise ValueError("production route design has no active factor baseline")
    selected = active_factor_indices.tolist()
    reasons = {
        spec.name: (
            "retained_complete_model_active_factor_baseline"
            if factor_diag[index] > 0
            else "dropped_no_model_active_event_after_train_gate_admission"
        )
        for index, spec in enumerate(specs[:factor_count])
    }
    if factor_count < len(specs):
        active_diag = factor_diag[active_factor_indices]
        g_ff_inv = np.diag(1.0 / active_diag)
        g_fo = gram[np.ix_(active_factor_indices, np.arange(factor_count, len(specs)))]
        residual = (
            gram[factor_count:, factor_count:]
            - g_fo.T @ g_ff_inv @ g_fo
        )
        residual = (residual + residual.T) / 2.0
        scale = max(float(np.diag(residual).max(initial=0.0)), 1.0)
        _, r, pivots = linalg.qr(residual, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        rank = int((diagonal > tolerance * scale).sum())
        retained_other = sorted(int(value) for value in pivots[:rank])
        selected.extend(factor_count + value for value in retained_other)
        retained_set = set(retained_other)
        for local, spec in enumerate(specs[factor_count:]):
            if local in retained_set:
                reasons[spec.name] = "retained_rank_independent_base_column"
            elif gram[factor_count + local, factor_count + local] == 0:
                reasons[spec.name] = "dropped_globally_zero_base_column"
            else:
                reasons[spec.name] = "dropped_rank_redundant_base_column"
    selected_array = np.asarray(selected, dtype=np.int64)
    final_gram = gram[np.ix_(selected_array, selected_array)]
    eigenvalues = np.linalg.eigvalsh((final_gram + final_gram.T) / 2.0)
    if bool((eigenvalues <= tolerance * max(float(eigenvalues.max()), 1.0)).any()):
        raise ValueError("selected base design is not full column rank")
    # SVD of a Cholesky witness has exactly the singular spectrum implied by
    # the full streamed X'X design without materializing the route-by-column matrix.
    witness = np.linalg.cholesky(final_gram).T
    singular = np.linalg.svd(witness, compute_uv=False)
    audit = {
        "candidate_column_count": len(specs),
        "retained_column_count": len(selected),
        "retained_candidate_indices": selected,
        "column_closure_reasons": reasons,
        "algorithm": "float64_streamed_XtX_then_numpy_SVD_of_Cholesky_witness",
        "tolerance": tolerance,
        "singular_values": singular.tolist(),
        "full_column_rank": True,
    }
    return selected_array, audit


def build_split_neutral_feature_design(real_root: str | Path) -> None:
    """Freeze one split-neutral vocabulary and a full-rank base column axis."""

    root = Path(real_root)
    vocabulary_events = []
    vocabulary_routes = []
    for chromosome in CHROMOSOMES:
        physical = pd.read_parquet(
            root / "events" / "physical_events" / f"part-{chromosome}.parquet"
        )
        routes = pd.read_parquet(
            root / "events" / "event_routes" / f"part-{chromosome}.parquet"
        )
        joined = routes.merge(
            physical[
                [
                    "event_id", "target_gene_id", "modality", "factor_identity_kind",
                    "factor_entity_id", "orientation",
                ]
            ],
            on=["event_id", "target_gene_id", "modality"],
            how="left",
            validate="many_to_one",
        )
        joined = _route_context_levels(joined, DISTANCE_BIN_BOUNDARIES)
        joined["interaction_factor_id"] = np.where(
            joined["factor_identity_kind"].astype(str).eq("accessibility_only"),
            "OPEN_ONLY",
            joined["factor_entity_id"].astype(str),
        )
        vocabulary_key = [
            "modality", "interaction_factor_id", "orientation", "geometry_kind",
            "region_type", "anchor_type", "transcript_oriented_side", "distance_bin",
        ]
        representatives = joined.drop_duplicates(vocabulary_key, keep="first")
        selected_route_ids = set(representatives["route_id"].astype(str))
        selected_event_ids = set(representatives["event_id"].astype(str))
        vocabulary_routes.append(
            routes.loc[routes["route_id"].astype(str).isin(selected_route_ids)].copy()
        )
        vocabulary_events.append(
            physical.loc[physical["event_id"].astype(str).isin(selected_event_ids)].copy()
        )
    physical = pd.concat(vocabulary_events, ignore_index=True).drop_duplicates(
        "event_id", keep="first"
    )
    routes = pd.concat(vocabulary_routes, ignore_index=True).drop_duplicates(
        "route_id", keep="first"
    )
    feature_manifest = build_event_feature_manifest(
        physical,
        routes,
        distance_bin_boundaries=DISTANCE_BIN_BOUNDARIES,
        # The current full-cohort runtime does not predeclare mechanism claims.
        # The interaction block is still constructed; only claim contrasts are empty.
        scientific_context_pairs={"DNA": {}, "RNA": {}},
        motif_score_in_model=False,
        orientation_interaction_policy={"DNA": False, "RNA": False},
        numeric_rank_tolerance=1.0e-8,
    )
    design_root = root / "design"
    design_root.mkdir(parents=True, exist_ok=True)
    (design_root / "EventFeatureManifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n"
    )


def build_production_base_design(real_root: str | Path) -> None:
    """Freeze full-rank base axes on train-admitted model-active routes."""

    root = Path(real_root)
    design_root = root / "design"
    feature_manifest = json.loads(
        (design_root / "EventFeatureManifest.json").read_text()
    )
    base_manifest: dict[str, object] = {
        "schema_version": "fabric.real_base_design.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_dataset_root": str(root.resolve()),
        "event_feature_manifest_identity": feature_manifest[
            "event_feature_manifest_identity"
        ],
        "catalog_population": (
            "model-active post-cap routes after train-only gate admission; "
            "raw vocabulary remains split-neutral in EventFeatureManifest"
        ),
        "motif_score_in_model": False,
        "test_rows_or_test_outcomes_read": False,
        "modalities": {},
    }
    modalities = ("DNA", "RNA")
    specs_by_modality = {
        modality: _base_candidate_specs(feature_manifest, modality)
        for modality in modalities
    }
    grams = {
        modality: np.zeros(
            (len(specs_by_modality[modality]), len(specs_by_modality[modality])),
            dtype=np.float64,
        )
        for modality in modalities
    }
    route_counts = {modality: 0 for modality in modalities}
    for chromosome in CHROMOSOMES:
        event_path = (
            root / "events" / "gated" / "physical_events"
            / f"part-{chromosome}.parquet"
        )
        route_path = (
            root / "events" / "gated" / "event_routes"
            / f"part-{chromosome}.parquet"
        )
        chromosome_events = pd.read_parquet(event_path)
        chromosome_routes = pd.read_parquet(route_path)
        context = _joined_route_context(
            chromosome_events,
            chromosome_routes,
            feature_manifest,
            active_only=True,
        )
        for modality in modalities:
            specs = specs_by_modality[modality]
            matrix = _encode_candidate_base(
                context, feature_manifest, modality, specs
            )
            grams[modality] += (matrix.T @ matrix).toarray()
            route_counts[modality] += matrix.shape[0]
    for modality in modalities:
        specs = specs_by_modality[modality]
        selected, rank_audit = _select_full_rank_base(
            grams[modality],
            specs,
            tolerance=float(feature_manifest["numeric_rank_audit"]["tolerance"]),
        )
        retained_specs = [specs[index] for index in selected]
        base_manifest["modalities"][modality] = {
            "candidate_specs": [asdict(value) for value in specs],
            "retained_specs": [asdict(value) for value in retained_specs],
            "base_column_names": [value.name for value in retained_specs],
            "route_count": route_counts[modality],
            "rank_audit": rank_audit,
        }
        feature_manifest["modalities"][modality]["base_column_names"] = [
            value.name for value in retained_specs
        ]
        feature_manifest["modalities"][modality]["base_column_closure"] = rank_audit
    (design_root / "BaseDesignManifest.json").write_text(
        json.dumps(base_manifest, indent=2) + "\n"
    )
    # Re-write the vocabulary with the explicitly closed production base axes.
    (design_root / "EventFeatureManifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n"
    )


def _gate_record(root: Path, chromosome: str, gene_id: str) -> dict[str, object]:
    path = root / "gates" / chromosome / f"{gene_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing fitted gate tensor for {gene_id}: {path}")
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or record.get("gene_id") != gene_id:
        raise TypeError(f"gate shard identity differs for {gene_id}")
    values = record.get("gate_values")
    if not isinstance(values, GateValues):
        raise TypeError(f"gate shard does not contain GateValues for {gene_id}")
    if tuple(record["cell_ids"]) != values.cell_ids:
        raise ValueError(f"gate cell identity differs for {gene_id}")
    return record


def _add_integer_counts(
    target: dict[tuple[str, str, str, str], dict[str, float]],
    key: tuple[str, str, str, str],
    *,
    events: int = 0,
    genes: int = 0,
    gates: int = 0,
    mass: float = 0.0,
    cell_gene_pairs: int = 0,
) -> None:
    row = target.setdefault(
        key,
        {
            "distinct_physical_event_count": 0.0,
            "distinct_target_gene_count": 0.0,
            "distinct_active_gate_key_count": 0.0,
            "informative_molecule_mass": 0.0,
            "supported_cell_gene_pair_count": 0.0,
        },
    )
    row["distinct_physical_event_count"] += events
    row["distinct_target_gene_count"] += genes
    row["distinct_active_gate_key_count"] += gates
    row["informative_molecule_mass"] += mass
    row["supported_cell_gene_pair_count"] += cell_gene_pairs


def measure_real_raw_interaction_support(
    real_root: str | Path, compatible_root: str | Path
) -> pd.DataFrame:
    """Measure train-only support from EC mass and explicit channel masks.

    The gate observation contract is channel-wide within a gene-cell: RNA is
    observed exactly when the full RNA library is positive, Open exactly when
    ATAC mapping is valid, and DNA when both are true.  Reading these frozen
    masks plus compatible-EC mass is therefore exactly equivalent to loading
    the much larger per-gate raw/standardized/value tensors merely to inspect
    their repeated ``observed`` columns.
    """

    root = Path(real_root)
    compatible = Path(compatible_root)
    design_root = root / "design"
    feature_manifest = json.loads((design_root / "EventFeatureManifest.json").read_text())
    activity_cells = pd.read_parquet(
        root / "cell_context" / "rna_activity_cell_axis.parquet",
        columns=["cell_id"],
    )
    library_size = np.load(
        root / "cell_context" / "rna_library_size.npy", mmap_mode="r"
    )
    if library_size.shape != (len(activity_cells),):
        raise ValueError("RNA library-size observation axis differs from its cell axis")
    rna_observed = pd.Series(
        np.asarray(library_size > 0, dtype=bool),
        index=activity_cells["cell_id"].astype(str),
    )
    if rna_observed.index.duplicated().any():
        raise ValueError("RNA activity cell axis is duplicated")
    mapping = pd.read_parquet(
        root / "cell_context" / "ATACMappingAudit.parquet",
        columns=["cell_id", "mapping_valid"],
    )
    if mapping["cell_id"].astype(str).duplicated().any():
        raise ValueError("ATAC mapping audit cell axis is duplicated")
    atac_observed = pd.Series(
        mapping["mapping_valid"].to_numpy(bool),
        index=mapping["cell_id"].astype(str),
    )
    totals: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for chromosome in CHROMOSOMES:
        physical = pd.read_parquet(
            root / "events" / "gated" / "physical_events" / f"part-{chromosome}.parquet"
        )
        routes = pd.read_parquet(
            root / "events" / "gated" / "event_routes" / f"part-{chromosome}.parquet"
        )
        context = _joined_route_context(
            physical, routes, feature_manifest, active_only=True
        )
        admission = pd.read_parquet(
            root / "events" / "gated" / "gate_admission"
            / f"part-{chromosome}.parquet",
            columns=[
                "target_gene_id", "gate_key_id", "channel", "gate_key_active"
            ],
        )
        active_channels = admission.loc[
            admission["gate_key_active"].astype(bool),
            ["target_gene_id", "gate_key_id", "channel"],
        ].copy()
        if active_channels.duplicated(["target_gene_id", "gate_key_id"]).any():
            raise ValueError(f"active gate admission keys are duplicated on {chromosome}")
        channel_check = context[
            ["target_gene_id", "gate_key_id", "modality", "interaction_factor_id"]
        ].merge(
            active_channels,
            on=["target_gene_id", "gate_key_id"],
            how="left",
            validate="many_to_one",
        )
        if channel_check["channel"].isna().any():
            raise ValueError(
                f"model-active route references a non-admitted gate on {chromosome}"
            )
        expected_channels = np.where(
            channel_check["modality"].astype(str).eq("RNA"),
            "RNA",
            np.where(
                channel_check["interaction_factor_id"].astype(str).eq("OPEN_ONLY"),
                "Open",
                "DNA",
            ),
        )
        if not np.array_equal(
            channel_check["channel"].astype(str).to_numpy(), expected_channels
        ):
            raise ValueError(f"event modality and gate channel differ on {chromosome}")
        ec = pd.read_parquet(
            compatible / "compatible_ec" / f"part-{chromosome}.parquet",
            filters=[("final_fate", "==", "likelihood_informative")],
            columns=["target_gene_id", "cell_id", "split", "molecule_count"],
        )
        if ec.empty or set(ec["split"].astype(str)) - {"train", "val"}:
            raise ValueError(
                f"raw interaction support EC is empty or contains test: {chromosome}"
            )
        gene_mass = (
            ec.groupby(["target_gene_id", "cell_id", "split"], sort=False)[
                "molecule_count"
            ]
            .sum()
            .reset_index()
        )
        ec_groups = {
            str(key): value
            for key, value in gene_mass.groupby("target_gene_id", sort=False)
        }
        for modality in ("DNA", "RNA"):
            modality_rows = context.loc[
                context["modality"].astype(str).eq(modality)
                & ~context["interaction_factor_id"].astype(str).eq("OPEN_ONLY")
            ].copy()
            for field in feature_manifest["modalities"][modality]["context_fields"]:
                summary = modality_rows.groupby(
                    ["interaction_factor_id", field], sort=False
                ).agg(
                    distinct_physical_event_count=("event_id", "nunique"),
                    distinct_target_gene_count=("target_gene_id", "nunique"),
                    distinct_active_gate_key_count=("gate_key_id", "nunique"),
                )
                for (factor, level), row in summary.iterrows():
                    _add_integer_counts(
                        totals,
                        (modality, field, str(factor), str(level)),
                        events=int(row["distinct_physical_event_count"]),
                        genes=int(row["distinct_target_gene_count"]),
                        gates=int(row["distinct_active_gate_key_count"]),
                    )

        for gene_id, gene_context in context.groupby("target_gene_id", sort=False):
            gene_id = str(gene_id)
            if gene_id not in ec_groups:
                raise ValueError(f"model-active gene lacks informative EC mass: {gene_id}")
            gene_ec = ec_groups[gene_id]
            mass = gene_ec["molecule_count"].to_numpy(np.float64)
            splits = gene_ec["split"].astype(str).to_numpy()
            cell_ids = gene_ec["cell_id"].astype(str)
            train = splits == "train"
            if set(splits) - {"train", "val"}:
                raise ValueError(f"support mass contains a forbidden split for {gene_id}")
            if not np.isfinite(mass).all() or bool((mass <= 0).any()):
                raise ValueError(f"informative gene-cell mass is invalid for {gene_id}")
            gene_rna_observed = cell_ids.map(rna_observed)
            gene_atac_observed = cell_ids.map(atac_observed)
            if gene_rna_observed.isna().any() or gene_atac_observed.isna().any():
                raise ValueError(f"support cells are absent from context axes: {gene_id}")
            observed_by_modality = {
                "RNA": gene_rna_observed.to_numpy(bool),
                "DNA": (
                    gene_rna_observed.to_numpy(bool)
                    & gene_atac_observed.to_numpy(bool)
                ),
            }
            for modality in ("DNA", "RNA"):
                modality_rows = gene_context.loc[
                    gene_context["modality"].astype(str).eq(modality)
                    & ~gene_context["interaction_factor_id"].astype(str).eq("OPEN_ONLY")
                ]
                valid = train & observed_by_modality[modality]
                supported_mass = float(mass[valid].sum())
                supported_pairs = int(valid.sum())
                for field in feature_manifest["modalities"][modality]["context_fields"]:
                    raw_cell_ids = tuple(
                        (str(factor), str(level))
                        for factor, level in modality_rows[
                            ["interaction_factor_id", field]
                        ].drop_duplicates().itertuples(index=False, name=None)
                    )
                    for factor, level in raw_cell_ids:
                        _add_integer_counts(
                            totals,
                            (modality, field, str(factor), str(level)),
                            mass=supported_mass,
                            cell_gene_pairs=supported_pairs,
                        )

    rows: list[dict[str, object]] = []
    for modality in ("DNA", "RNA"):
        config = feature_manifest["modalities"][modality]
        factors = tuple(config["interaction_factor_vocabulary"])
        threshold = INTERACTION_SUPPORT_THRESHOLDS[modality]
        for field, field_config in config["context_fields"].items():
            for factor, level in itertools.product(factors, field_config["raw_levels"]):
                key = (modality, field, str(factor), str(level))
                counts = totals.get(
                    key,
                    {
                        "distinct_physical_event_count": 0.0,
                        "distinct_target_gene_count": 0.0,
                        "distinct_active_gate_key_count": 0.0,
                        "informative_molecule_mass": 0.0,
                        "supported_cell_gene_pair_count": 0.0,
                    },
                )
                failures = []
                comparisons = (
                    ("distinct_physical_event_count", "minimum_distinct_events"),
                    ("distinct_target_gene_count", "minimum_distinct_genes"),
                    ("distinct_active_gate_key_count", "minimum_distinct_gate_keys"),
                    ("informative_molecule_mass", "minimum_informative_molecules"),
                )
                for count_name, threshold_name in comparisons:
                    if counts[count_name] < threshold[threshold_name]:
                        failures.append(f"insufficient_{count_name}")
                rows.append(
                    {
                        "modality": modality,
                        "context_field": field,
                        "factor_entity_id": str(factor),
                        "context_level": str(level),
                        **{
                            name: int(value) if name != "informative_molecule_mass" else float(value)
                            for name, value in counts.items()
                        },
                        "raw_cell_supported": not failures,
                        "support_failure_reasons": failures,
                        "support_population": "train_only_after_cap_gate_event_admission",
                    }
                )
    result = pd.DataFrame(rows)
    result.to_parquet(design_root / "RawInteractionSupport.parquet", index=False)
    manifest = {
        "schema_version": "fabric.real_raw_interaction_support.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_by_channel": INTERACTION_SUPPORT_THRESHOLDS,
        "support_population": "train_only_after_cap_gate_event_admission",
        "support_observation_source": (
            "frozen likelihood-informative EC cell-gene mass plus full-library "
            "RNA-observed and ATAC-mapping-valid channel masks"
        ),
        "per_gate_value_tensors_read": False,
        "validation_statistics_used": False,
        "test_rows_or_test_statistics_used": False,
        "raw_cell_count": len(result),
        "supported_raw_cell_count": int(result["raw_cell_supported"].sum()),
    }
    (design_root / "RawInteractionSupportManifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return result


class _ExactRectangleBasis:
    """Fixed-order exact elimination for sparse integer rectangle columns."""

    def __init__(self) -> None:
        self._pivots: dict[int, dict[int, int]] = {}

    def add_if_independent(self, entries: Mapping[int, int]) -> bool:
        vector = {int(key): int(value) for key, value in entries.items() if value}
        while vector:
            pivot = min(vector)
            if pivot not in self._pivots:
                divisor = math.gcd(*(abs(value) for value in vector.values()))
                if divisor > 1:
                    vector = {
                        key: value // divisor for key, value in vector.items()
                    }
                if vector[pivot] < 0:
                    vector = {key: -value for key, value in vector.items()}
                self._pivots[pivot] = vector
                return True
            basis = self._pivots[pivot]
            basis_scale = basis[pivot]
            vector_scale = vector[pivot]
            keys = vector.keys() | basis.keys()
            vector = {
                key: basis_scale * vector.get(key, 0)
                - vector_scale * basis.get(key, 0)
                for key in keys
            }
            vector = {key: value for key, value in vector.items() if value}
            if vector:
                divisor = math.gcd(*(abs(value) for value in vector.values()))
                if divisor > 1:
                    vector = {
                        key: value // divisor for key, value in vector.items()
                    }
        return False

    @property
    def rank(self) -> int:
        return len(self._pivots)


def _supported_bipartite_cycle_space_dimension(
    factors: Sequence[str],
    levels: Sequence[str],
    support: Mapping[tuple[str, str], bool],
) -> int:
    """Return the exact graph-cycle upper bound for supported rectangles.

    A supported raw cell is one edge of the factor-by-level bipartite graph,
    and every four-corner rectangle is an oriented cycle in that graph.  Its
    cycle-space dimension ``E - V + C`` is therefore a strict exact upper
    bound on the rank of every supported rectangle column.  Once exact greedy
    elimination reaches this value, all later rectangles are mathematically
    forced to be dependent and do not need another integer elimination.
    Isolated vocabulary vertices remain in ``V`` and ``C`` so they cancel.
    """

    factor_nodes = tuple(("factor", str(value)) for value in factors)
    level_nodes = tuple(("level", str(value)) for value in levels)
    nodes = factor_nodes + level_nodes
    parent = {value: value for value in nodes}

    def find(value: tuple[str, str]) -> tuple[str, str]:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_count = 0
    for factor in factors:
        for level in levels:
            if not support.get((str(factor), str(level)), False):
                continue
            edge_count += 1
            union(("factor", str(factor)), ("level", str(level)))
    component_count = len({find(value) for value in nodes})
    dimension = edge_count - len(nodes) + component_count
    if dimension < 0:
        raise RuntimeError("supported bipartite cycle-space dimension is negative")
    return dimension


def build_real_interaction_basis(real_root: str | Path) -> None:
    """Build the canonical support-closed rectangle basis with padded axes."""

    root = Path(real_root)
    design_root = root / "design"
    feature_manifest = json.loads((design_root / "EventFeatureManifest.json").read_text())
    support = pd.read_parquet(design_root / "RawInteractionSupport.parquet")
    rectangle_rows: list[dict[str, object]] = []
    rectangle_path = design_root / "SupportedInteractionRectangleAudit.parquet"
    rectangle_writer: pq.ParquetWriter | None = None
    rectangle_schema = pa.schema(
        [
            pa.field("modality", pa.string(), nullable=False),
            pa.field("context_field", pa.string(), nullable=False),
            pa.field("factor_left", pa.string(), nullable=False),
            pa.field("factor_right", pa.string(), nullable=False),
            pa.field("level_left", pa.string(), nullable=False),
            pa.field("level_right", pa.string(), nullable=False),
            pa.field("four_corner_supported", pa.bool_(), nullable=False),
            pa.field("selected_as_canonical_pivot", pa.bool_(), nullable=False),
            pa.field("padded_column_index", pa.int64(), nullable=True),
        ]
    )

    def flush_rectangles() -> None:
        nonlocal rectangle_writer
        if not rectangle_rows:
            return
        table = pa.Table.from_pylist(rectangle_rows, schema=rectangle_schema)
        if rectangle_writer is None:
            rectangle_writer = pq.ParquetWriter(
                rectangle_path,
                rectangle_schema,
                compression="zstd",
                use_dictionary=True,
            )
        rectangle_writer.write_table(table)
        rectangle_rows.clear()
    interaction_manifest: dict[str, object] = {
        "schema_version": "FABRIC_V2_INTERACTION_SUPPORT_MANIFEST_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "support_population": "train_only_after_cap_gate_event_admission",
        "thresholds_by_channel": INTERACTION_SUPPORT_THRESHOLDS,
        "validation_test_may_activate_columns": False,
        "test_rows_or_test_statistics_used": False,
        "modalities": {},
    }
    for modality in ("DNA", "RNA"):
        config = feature_manifest["modalities"][modality]
        factors = tuple(str(value) for value in config["interaction_factor_vocabulary"])
        modality_support = support.loc[support["modality"].astype(str).eq(modality)]
        offset = 0
        active_indices: list[int] = []
        fields: dict[str, object] = {}
        for field, field_config in config["context_fields"].items():
            levels = tuple(str(value) for value in field_config["raw_levels"])
            lookup = {
                (str(row.factor_entity_id), str(row.context_level)): bool(row.raw_cell_supported)
                for row in modality_support.loc[
                    modality_support["context_field"].astype(str).eq(field)
                ].itertuples(index=False)
            }
            cycle_space_upper_bound = _supported_bipartite_cycle_space_dimension(
                factors, levels, lookup
            )
            raw_index = {
                (factor, level): factor_index * len(levels) + level_index
                for factor_index, factor in enumerate(factors)
                for level_index, level in enumerate(levels)
            }
            basis = _ExactRectangleBasis()
            pivots: list[dict[str, object]] = []
            supported_rectangle_count = 0
            for factor_left_index, factor_right_index in itertools.combinations(
                range(len(factors)), 2
            ):
                factor_left = factors[factor_left_index]
                factor_right = factors[factor_right_index]
                common_levels = {
                    level
                    for level in levels
                    if lookup.get((factor_left, level), False)
                    and lookup.get((factor_right, level), False)
                }
                for level_left, level_right in itertools.combinations(levels, 2):
                    if level_left not in common_levels or level_right not in common_levels:
                        continue
                    supported_rectangle_count += 1
                    entries = {
                        raw_index[(factor_left, level_left)]: 1,
                        raw_index[(factor_left, level_right)]: -1,
                        raw_index[(factor_right, level_left)]: -1,
                        raw_index[(factor_right, level_right)]: 1,
                    }
                    exact_rank_limit = min(
                        int(field_config["p_max"]), cycle_space_upper_bound
                    )
                    retained = (
                        basis.add_if_independent(entries)
                        if basis.rank < exact_rank_limit
                        else False
                    )
                    padded_column = offset + len(pivots) if retained else None
                    if retained:
                        pivot = {
                            "factor_left": factor_left,
                            "factor_right": factor_right,
                            "level_left": level_left,
                            "level_right": level_right,
                            "support_column_index": len(pivots),
                            "padded_column_index": padded_column,
                        }
                        pivots.append(pivot)
                        active_indices.append(int(padded_column))
                    rectangle_rows.append(
                        {
                            "modality": modality,
                            "context_field": field,
                            "factor_left": factor_left,
                            "factor_right": factor_right,
                            "level_left": level_left,
                            "level_right": level_right,
                            "four_corner_supported": True,
                            "selected_as_canonical_pivot": retained,
                            "padded_column_index": padded_column,
                        }
                    )
                    if len(rectangle_rows) >= 100_000:
                        flush_rectangles()
            p_max = int(field_config["p_max"])
            if len(pivots) > p_max:
                raise RuntimeError("exact supported rectangle rank exceeds p_max")
            if len(pivots) > cycle_space_upper_bound:
                raise RuntimeError(
                    "supported rectangle rank exceeds its graph cycle-space bound"
                )
            potential = (
                len(factors) * (len(factors) - 1) // 2
                * (len(levels) * (len(levels) - 1) // 2)
            )
            fields[field] = {
                "raw_factor_order": list(factors),
                "raw_level_order": list(levels),
                "N_raw_rectangles_potential": potential,
                "N_four_corner_supported": supported_rectangle_count,
                "N_support_span": len(pivots),
                "N_rank_retained": len(pivots),
                "N_padded": p_max,
                "supported_bipartite_cycle_space_dimension": (
                    cycle_space_upper_bound
                ),
                "exact_rank_closure": (
                    "reached_exact_graph_cycle_space_upper_bound"
                    if len(pivots) == cycle_space_upper_bound
                    else "all_supported_rectangles_exhaustively_eliminated"
                ),
                "padded_offset": offset,
                "canonical_pivot_rectangles": pivots,
                "unsupported_rectangle_status": (
                    "exactly_derivable_as_not_four_corner_supported_from_RawInteractionSupport"
                ),
                "basis_coverage": (
                    "not_applicable_no_supported_rectangle" if not pivots else "full"
                ),
            }
            offset += p_max
        expected_width = int(config["padded_interaction_width"])
        if offset != expected_width:
            raise RuntimeError("interaction padded field offsets do not close")
        mask = np.zeros(expected_width, dtype=bool)
        mask[np.asarray(active_indices, dtype=np.int64)] = True
        np.save(design_root / f"{modality.lower()}_interaction_active_mask.npy", mask)
        interaction_manifest["modalities"][modality] = {
            "padded_width": expected_width,
            "active_width": int(mask.sum()),
            "support_padded_column_indices": active_indices,
            "active_padded_column_indices": active_indices,
            "fields": fields,
            "combined_rank_audit": "PENDING_PRODUCTION_ROUTE_WITNESS",
        }
    flush_rectangles()
    if rectangle_writer is not None:
        rectangle_writer.close()
    elif not rectangle_path.exists():
        pd.DataFrame(
            columns=[
                "modality", "context_field", "factor_left", "factor_right",
                "level_left", "level_right", "four_corner_supported",
                "selected_as_canonical_pivot", "padded_column_index",
            ]
        ).to_parquet(rectangle_path, index=False)
    (design_root / "InteractionSupportManifest.json").write_text(
        json.dumps(interaction_manifest, indent=2) + "\n"
    )


def _select_exact_integer_columns(
    integer_rows: Sequence[Sequence[tuple[int, int]]],
    *,
    width: int,
    required_prefix_width: int,
) -> tuple[tuple[int, ...], int]:
    """Keep a fixed prefix, then close signed integer rank over the rationals."""

    if not integer_rows or not 0 <= required_prefix_width <= width:
        raise ValueError("invalid exact-integer column-selection dimensions")
    columns: list[dict[int, int]] = [dict() for _ in range(width)]
    for row_index, entries in enumerate(integer_rows):
        for column, value in entries:
            if not 0 <= int(column) < width or not int(value):
                raise ValueError("exact integer row contains an invalid sparse entry")
            columns[int(column)][row_index] = int(value)
    basis = _ExactRectangleBasis()
    retained: list[int] = []
    for column in range(width):
        independent = basis.add_if_independent(columns[column])
        if column < required_prefix_width:
            if not independent:
                raise ValueError(
                    "required integer base block is not full rank over the rationals"
                )
            retained.append(column)
        elif independent:
            retained.append(column)
    return tuple(retained), basis.rank


def _support_interaction_columns(
    modality_manifest: Mapping[str, object],
) -> tuple[int, ...]:
    explicit = modality_manifest.get("support_padded_column_indices")
    if explicit is not None:
        return tuple(int(value) for value in explicit)
    # Admission artifacts built before combined closure stored the same axis
    # only inside the canonical field pivot records.
    return tuple(
        int(pivot["padded_column_index"])
        for details in modality_manifest["fields"].values()
        for pivot in details["canonical_pivot_rectangles"]
    )


def audit_production_route_design(real_root: str | Path) -> None:
    """Certify combined base/interaction rank on all model-active routes.

    Integer categorical and signed canonical rectangle columns receive an exact
    fraction-free rational-rank certificate.  The small continuous block is
    then certified by an SVD of within-identical-integer-design contrasts, so it
    cannot borrow rank from categorical terms.
    """

    root = Path(real_root)
    design_root = root / "design"
    feature_manifest = json.loads((design_root / "EventFeatureManifest.json").read_text())
    base_manifest = json.loads((design_root / "BaseDesignManifest.json").read_text())
    interaction_manifest = json.loads(
        (design_root / "InteractionSupportManifest.json").read_text()
    )
    for modality in ("DNA", "RNA"):
        retained_specs = _specs_from_records(
            base_manifest["modalities"][modality]["retained_specs"]
        )
        integer_base = np.asarray(
            [
                index
                for index, spec in enumerate(retained_specs)
                if spec.kind == "factor"
                or spec.kind.startswith("categorical:")
                or spec.kind.startswith("available:")
            ],
            dtype=np.int64,
        )
        continuous_base = np.asarray(
            [index for index in range(len(retained_specs)) if index not in set(integer_base)],
            dtype=np.int64,
        )
        modality_manifest = interaction_manifest["modalities"][modality]
        support_interaction = np.asarray(
            _support_interaction_columns(modality_manifest),
            dtype=np.int64,
        )
        if len(set(support_interaction.tolist())) != len(support_interaction):
            raise ValueError(f"{modality} support interaction columns are duplicated")
        candidate_integer_width = len(integer_base) + len(support_interaction)
        integer_rows: list[tuple[tuple[int, int], ...]] = []
        integer_row_set: set[tuple[tuple[int, int], ...]] = set()
        integer_row_basis = _ExactRectangleBasis()
        unique_integer_rows_examined = 0
        continuous_differences: list[np.ndarray] = []
        continuous_rank = 0
        production_route_rows_examined = 0
        for chromosome in CHROMOSOMES:
            physical = pd.read_parquet(
                root / "events" / "gated" / "physical_events" / f"part-{chromosome}.parquet"
            )
            routes = pd.read_parquet(
                root / "events" / "gated" / "event_routes" / f"part-{chromosome}.parquet"
            )
            context = _joined_route_context(
                physical, routes, feature_manifest, active_only=True
            )
            context = context.loc[context["modality"].astype(str).eq(modality)].reset_index(drop=True)
            production_route_rows_examined += len(context)
            signature = context[
                [
                    "interaction_factor_id",
                    *feature_manifest["modalities"][modality][
                        "base_categorical_fields"
                    ].keys(),
                ]
            ].astype(str).copy()
            for column in (
                "signed_distance_bp",
                "edge_relative_position",
                "distance_to_5prime_boundary_bp",
                "distance_to_3prime_boundary_bp",
            ):
                values = context[column].to_numpy(np.float64)
                if np.isinf(values).any():
                    raise ValueError("route geometry contains infinity")
                signature[f"available:{column}"] = np.isfinite(values)
            unique_positions = signature.drop_duplicates(
                keep="first"
            ).index.to_numpy(np.int64)
            integer_context = context.iloc[unique_positions].reset_index(drop=True)
            base = _encode_production_base(
                integer_context, feature_manifest, base_manifest, modality
            ).tocsr()
            interaction = _encode_production_interaction(
                integer_context, interaction_manifest, modality
            ).tocsr()
            integer_matrix = sparse.hstack(
                [base[:, integer_base], interaction[:, support_interaction]],
                format="csr",
                dtype=np.float32,
            )
            for row in range(integer_matrix.shape[0]):
                start, end = integer_matrix.indptr[row : row + 2]
                indices = integer_matrix.indices[start:end]
                values = integer_matrix.data[start:end]
                entries: list[tuple[int, int]] = []
                for column, value in zip(indices, values, strict=True):
                    rounded = int(round(float(value)))
                    if not np.isclose(value, rounded, atol=1.0e-7, rtol=0):
                        raise ValueError("integer production design contains a non-integer value")
                    if rounded:
                        entries.append((int(column), rounded))
                identity = tuple(sorted(entries))
                if identity not in integer_row_set:
                    integer_row_set.add(identity)
                    unique_integer_rows_examined += 1
                    if integer_row_basis.add_if_independent(dict(identity)):
                        integer_rows.append(identity)
            if len(continuous_base) and continuous_rank < len(continuous_base):
                # Continuous columns may only claim rank that cannot be
                # borrowed from the integer design.  Differences between real
                # production routes with exactly the same complete integer
                # signature remove that integer row identically.
                continuous = _encode_production_base(
                    context, feature_manifest, base_manifest, modality
                )[:, continuous_base].toarray().astype(np.float64)
                for positions in signature.groupby(
                    list(signature.columns), sort=False, observed=True
                ).indices.values():
                    positions = np.asarray(positions, dtype=np.int64)
                    if len(positions) < 2:
                        continue
                    reference = continuous[positions[0]]
                    for position in positions[1:]:
                        difference = continuous[position] - reference
                        if not bool(np.any(difference != 0)):
                            continue
                        continuous_differences.append(difference)
                        values_svd = np.linalg.svd(
                            np.asarray(continuous_differences), compute_uv=False
                        )
                        continuous_rank = int(
                            (
                                values_svd
                                > 1.0e-8 * max(float(values_svd.max()), 1.0)
                            ).sum()
                        )
                        if continuous_rank == len(continuous_base):
                            break
                    if continuous_rank == len(continuous_base):
                        break
        if continuous_rank != len(continuous_base):
            raise ValueError(
                f"{modality} continuous base block adds only {continuous_rank}/{len(continuous_base)} ranks"
            )
        retained_columns, candidate_rank = _select_exact_integer_columns(
            integer_rows,
            width=candidate_integer_width,
            required_prefix_width=len(integer_base),
        )
        if candidate_rank != integer_row_basis.rank:
            raise RuntimeError("exact row-rank and column-rank certificates disagree")
        retained_interaction_positions = tuple(
            value - len(integer_base)
            for value in retained_columns
            if value >= len(integer_base)
        )
        active_interaction = tuple(
            int(support_interaction[position])
            for position in retained_interaction_positions
        )
        final_integer_width = len(integer_base) + len(active_interaction)
        if final_integer_width != candidate_rank:
            raise RuntimeError("combined integer design did not close to exact rank")
        active_set = set(active_interaction)
        active_mask = np.zeros(int(modality_manifest["padded_width"]), dtype=bool)
        active_mask[np.asarray(active_interaction, dtype=np.int64)] = True
        np.save(
            design_root / f"{modality.lower()}_interaction_active_mask.npy",
            active_mask,
        )
        modality_manifest["support_padded_column_indices"] = (
            support_interaction.tolist()
        )
        modality_manifest["active_padded_column_indices"] = list(active_interaction)
        modality_manifest["active_width"] = len(active_interaction)
        for details in modality_manifest["fields"].values():
            support_columns = [
                int(value["padded_column_index"])
                for value in details["canonical_pivot_rectangles"]
            ]
            details["active_padded_column_indices"] = [
                value for value in support_columns if value in active_set
            ]
            details["N_combined_design_active"] = len(
                details["active_padded_column_indices"]
            )
            details["combined_design_closure"] = {
                "retained_rank_increasing_support_columns": int(
                    details["N_combined_design_active"]
                ),
                "dropped_rank_redundant_support_columns": len(support_columns)
                - int(details["N_combined_design_active"]),
            }
        difference_singular = (
            np.linalg.svd(np.asarray(continuous_differences), compute_uv=False).tolist()
            if continuous_differences
            else []
        )
        modality_manifest["combined_rank_audit"] = {
            "status": "PASS",
            "production_route_rows_examined_for_certificate": (
                production_route_rows_examined
            ),
            "unique_integer_route_rows_examined_for_certificate": (
                unique_integer_rows_examined
            ),
            "independent_integer_route_rows_in_certificate": len(integer_rows),
            "integer_row_deduplication_identity": (
                "interaction_factor_all_base_categorical_fields_and_all_"
                "continuous_availability_masks"
            ),
            "candidate_integer_column_count_before_combined_closure": (
                candidate_integer_width
            ),
            "candidate_exact_rational_rank": candidate_rank,
            "integer_base_column_count": len(integer_base),
            "support_interaction_column_count": len(support_interaction),
            "rank_redundant_support_interaction_column_count": (
                len(support_interaction) - len(active_interaction)
            ),
            "active_interaction_column_count": len(active_interaction),
            "final_integer_column_count": final_integer_width,
            "final_exact_rational_rank": final_integer_width,
            "signed_interaction_values_preserved": True,
            "continuous_column_count": len(continuous_base),
            "within_identical_integer_design_continuous_rank": continuous_rank,
            "continuous_difference_singular_values": difference_singular,
            "numeric_tolerance": 1.0e-8,
            "zero_column_count": 0,
            "exact_duplicate_column_count": 0,
            "unrecorded_rank_deficiency_count": 0,
        }
    (design_root / "InteractionSupportManifest.json").write_text(
        json.dumps(interaction_manifest, indent=2) + "\n"
    )


def _specs_from_records(records: Sequence[Mapping[str, object]]) -> tuple[BaseColumnSpec, ...]:
    return tuple(
        BaseColumnSpec(
            name=str(record["name"]),
            kind=str(record["kind"]),
            value=None if record.get("value") is None else str(record["value"]),
        )
        for record in records
    )


def _encode_production_base(
    context: pd.DataFrame,
    feature_manifest: Mapping[str, object],
    base_manifest: Mapping[str, object],
    modality: str,
) -> sparse.csr_matrix:
    records = base_manifest["modalities"][modality]
    candidates = _specs_from_records(records["candidate_specs"])
    retained = _specs_from_records(records["retained_specs"])
    candidate_index = {value.name: index for index, value in enumerate(candidates)}
    if set(value.name for value in retained) - set(candidate_index):
        raise ValueError("retained base spec is absent from candidate axis")
    matrix = _encode_candidate_base(context, feature_manifest, modality, candidates)
    return matrix[
        :, [candidate_index[value.name] for value in retained]
    ].astype(np.float32)


def _interaction_raw_cell_map(
    interaction_manifest: Mapping[str, object], modality: str
) -> dict[str, dict[tuple[str, str], tuple[tuple[int, int], ...]]]:
    result: dict[str, dict[tuple[str, str], list[tuple[int, int]]]] = {}
    for field, details in interaction_manifest["modalities"][modality]["fields"].items():
        mapping: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for pivot in details["canonical_pivot_rectangles"]:
            column = int(pivot["padded_column_index"])
            corners = (
                ((str(pivot["factor_left"]), str(pivot["level_left"])), 1),
                ((str(pivot["factor_left"]), str(pivot["level_right"])), -1),
                ((str(pivot["factor_right"]), str(pivot["level_left"])), -1),
                ((str(pivot["factor_right"]), str(pivot["level_right"])), 1),
            )
            for raw_cell, sign in corners:
                mapping.setdefault(raw_cell, []).append((column, sign))
        result[str(field)] = {
            key: tuple(value) for key, value in mapping.items()
        }
    return result


def _encode_production_interaction(
    context: pd.DataFrame,
    interaction_manifest: Mapping[str, object],
    modality: str,
) -> sparse.csr_matrix:
    frame = context.loc[context["modality"].astype(str).eq(modality)].reset_index(drop=True)
    width = int(interaction_manifest["modalities"][modality]["padded_width"])
    raw_maps = _interaction_raw_cell_map(interaction_manifest, modality)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    factors = frame["interaction_factor_id"].astype(str).to_numpy()
    for field, mapping in raw_maps.items():
        levels = frame[field].astype(str).to_numpy()
        for row, raw_cell in enumerate(zip(factors, levels, strict=True)):
            entries = mapping.get(raw_cell, ())
            if entries:
                row_parts.append(np.full(len(entries), row, dtype=np.int64))
                column_parts.append(
                    np.fromiter((value[0] for value in entries), dtype=np.int64)
                )
                value_parts.append(
                    np.fromiter((value[1] for value in entries), dtype=np.float32)
                )
    if not row_parts:
        return sparse.csr_matrix((len(frame), width), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(len(frame), width),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _production_modality_tensors(
    physical: pd.DataFrame,
    routes: pd.DataFrame,
    gate_values: GateValues,
    *,
    target_gene_id: str,
    modality: str,
    ordered_edge_ids: Sequence[str],
    feature_manifest: Mapping[str, object],
    base_manifest: Mapping[str, object],
    interaction_manifest: Mapping[str, object],
) -> ProductionModalityTensors:
    context = _joined_route_context(
        physical, routes, feature_manifest, active_only=True
    )
    frame = context.loc[
        context["target_gene_id"].astype(str).eq(target_gene_id)
        & context["modality"].astype(str).eq(modality)
    ].sort_values("route_id", kind="mergesort").reset_index(drop=True)
    events = physical.loc[
        physical["target_gene_id"].astype(str).eq(target_gene_id)
        & physical["modality"].astype(str).eq(modality)
        & physical["model_active"].astype(bool)
    ].sort_values("event_id", kind="mergesort")
    event_ids = tuple(events["event_id"].astype(str))
    route_ids = tuple(frame["route_id"].astype(str))
    if set(frame["event_id"].astype(str)) != set(event_ids):
        if event_ids or len(frame):
            raise ValueError(f"active event/route identity differs for {target_gene_id}/{modality}")
    edges = tuple(str(value) for value in ordered_edge_ids)
    edge_index = {value: index for index, value in enumerate(edges)}
    missing_edges = sorted(set(frame["edge_id"].astype(str)) - set(edge_index))
    if missing_edges:
        raise ValueError(f"production routes reference absent edges: {missing_edges[:5]}")
    base = _encode_production_base(
        frame, feature_manifest, base_manifest, modality
    )
    interaction = _encode_production_interaction(
        frame, interaction_manifest, modality
    )
    active_mask = np.load(
        Path(base_manifest["real_dataset_root"])
        / "design"
        / f"{modality.lower()}_interaction_active_mask.npy"
    )
    if active_mask.shape != (interaction.shape[1],):
        raise ValueError("interaction active mask differs from padded feature axis")
    event_index = {value: index for index, value in enumerate(event_ids)}
    gate_key_ids = tuple(sorted(set(events["gate_key_id"].astype(str))))
    source_gate_index = {value: index for index, value in enumerate(gate_values.gate_key_ids)}
    missing_gates = sorted(set(gate_key_ids) - set(source_gate_index))
    if missing_gates:
        raise ValueError(f"production event references absent gates: {missing_gates[:5]}")
    local_gate_index = {value: index for index, value in enumerate(gate_key_ids)}
    weights = frame["route_weight"].to_numpy(np.float64)
    if len(frame):
        if not np.isfinite(weights).all() or bool((weights <= 0).any()):
            raise ValueError("production route weights are invalid")
        sums = frame.groupby("event_id", sort=False)["route_weight"].sum().to_numpy(float)
        if not np.allclose(sums, 1.0, atol=1.0e-12, rtol=0):
            raise ValueError("production route weights do not sum to one")
    return ProductionModalityTensors(
        cell_ids=gate_values.cell_ids,
        target_gene_id=target_gene_id,
        modality=modality,
        ordered_edge_ids=edges,
        event_ids=event_ids,
        gate_key_ids=gate_key_ids,
        route_ids=route_ids,
        route_event_index=np.asarray(
            [event_index[str(value)] for value in frame["event_id"]], dtype=np.int64
        ),
        route_edge_index=np.asarray(
            [edge_index[str(value)] for value in frame["edge_id"]], dtype=np.int64
        ),
        route_weight=weights.astype(np.float32),
        route_base_features=base,
        route_interaction_features=interaction,
        interaction_active_mask=active_mask.astype(bool),
        event_gate_key_index=np.asarray(
            [local_gate_index[str(value)] for value in events["gate_key_id"]],
            dtype=np.int64,
        ),
        gate=gate_values.gate[
            :, [source_gate_index[value] for value in gate_key_ids]
        ].astype(np.float32, copy=True),
    )


def _gene_model_injection_index(
    physical: pd.DataFrame,
    routes: pd.DataFrame,
    *,
    ordered_edge_ids: Sequence[str],
    dna: ProductionModalityTensors,
    rna: ProductionModalityTensors,
    base_manifest: Mapping[str, object],
    interaction_manifest: Mapping[str, object],
) -> pd.DataFrame:
    route_ids = dna.route_ids + rna.route_ids
    dna_names = tuple(
        str(value["name"])
        for value in base_manifest["modalities"]["DNA"]["retained_specs"]
    )
    rna_names = tuple(
        str(value["name"])
        for value in base_manifest["modalities"]["RNA"]["retained_specs"]
    )
    base_values = np.zeros((len(route_ids), len(dna_names) + len(rna_names)), dtype=np.float32)
    dna_base = (
        dna.route_base_features.toarray()
        if sparse.issparse(dna.route_base_features)
        else np.asarray(dna.route_base_features)
    )
    rna_base = (
        rna.route_base_features.toarray()
        if sparse.issparse(rna.route_base_features)
        else np.asarray(rna.route_base_features)
    )
    base_values[: len(dna.route_ids), : len(dna_names)] = dna_base
    base_values[len(dna.route_ids) :, len(dna_names) :] = rna_base
    route_base = RouteBaseDesign(
        route_ids=route_ids,
        values=base_values,
        column_names=dna_names + rna_names,
        manifest=base_manifest,
        route_context=pd.DataFrame(),
    )
    dna_interaction = (
        dna.route_interaction_features.tocsr()
        if sparse.issparse(dna.route_interaction_features)
        else np.asarray(dna.route_interaction_features)
    )
    rna_interaction = (
        rna.route_interaction_features.tocsr()
        if sparse.issparse(rna.route_interaction_features)
        else np.asarray(rna.route_interaction_features)
    )
    interaction = InteractionDesign(
        route_ids=route_ids,
        values_by_modality={"DNA": dna_interaction, "RNA": rna_interaction},
        active_mask_by_modality={
            "DNA": dna.interaction_active_mask,
            "RNA": rna.interaction_active_mask,
        },
        route_indices_by_modality={
            "DNA": np.arange(len(dna.route_ids), dtype=np.int64),
            "RNA": np.arange(
                len(dna.route_ids), len(route_ids), dtype=np.int64
            ),
        },
        raw_support=pd.DataFrame(),
        manifest=interaction_manifest,
        raw_contrasts=pd.DataFrame(),
    )
    gene_ids = set(physical["target_gene_id"].astype(str))
    if len(gene_ids) > 1:
        raise ValueError("model injection index input mixes genes")
    if not gene_ids:
        return pd.DataFrame()
    gene_id = next(iter(gene_ids))
    return build_model_injection_equivalence_index(
        physical,
        routes,
        route_base,
        interaction,
        ordered_edge_ids_by_gene={gene_id: tuple(map(str, ordered_edge_ids))},
    )


def build_prepared_chromosome(
    real_root: str | Path,
    compatible_root: str | Path,
    *,
    chromosome: str,
) -> None:
    """Assemble and serialize every G_fit gene on one chromosome."""

    if chromosome not in CHROMOSOMES:
        raise ValueError(f"unsupported chromosome: {chromosome}")
    root = Path(real_root)
    compatible = Path(compatible_root)
    design_root = root / "design"
    feature_manifest = json.loads((design_root / "EventFeatureManifest.json").read_text())
    base_manifest = json.loads((design_root / "BaseDesignManifest.json").read_text())
    interaction_manifest = json.loads(
        (design_root / "InteractionSupportManifest.json").read_text()
    )
    cis_manifest = json.loads((root / "cis" / "CISManifest.json").read_text())
    cis_feature_names = tuple(str(value) for value in cis_manifest["model_feature_order"])
    graph_root = root / "graph"
    nodes = pd.read_parquet(
        graph_root / "node_table.parquet", filters=[("chrom", "==", chromosome)]
    )
    edges = pd.read_parquet(
        graph_root / "edge_table.parquet", filters=[("chrom", "==", chromosome)]
    )
    paths = pd.read_parquet(
        graph_root / "path_table.parquet", filters=[("chrom", "==", chromosome)]
    )
    path_edges = pd.read_parquet(
        graph_root / "path_edge_table.parquet", filters=[("chrom", "==", chromosome)]
    )
    if any(frame.empty for frame in (nodes, edges, paths, path_edges)):
        raise ValueError(f"graph tables are empty for {chromosome}")
    graph_gene_ids = tuple(paths["gene_id"].astype(str).drop_duplicates())
    g_fit = pd.read_csv(compatible / "G_fit.tsv", sep="\t")
    g_fit_axis = tuple(g_fit["target_gene_id"].astype(str))
    g_fit_order = {value: index for index, value in enumerate(g_fit_axis)}
    selected_genes = tuple(value for value in g_fit_axis if value in set(graph_gene_ids))
    if not selected_genes:
        raise ValueError(f"G_fit has no genes on {chromosome}")

    normalized_cis = pd.read_parquet(root / "cis" / "normalized_cis_features.parquet")
    normalized_cis = normalized_cis.loc[
        normalized_cis["target_gene_id"].astype(str).isin(selected_genes)
    ].copy()
    physical = pd.read_parquet(
        root / "events" / "gated" / "physical_events" / f"part-{chromosome}.parquet"
    )
    routes = pd.read_parquet(
        root / "events" / "gated" / "event_routes" / f"part-{chromosome}.parquet"
    )
    ec = pd.read_parquet(
        compatible / "compatible_ec" / f"part-{chromosome}.parquet",
        filters=[("final_fate", "==", "likelihood_informative")],
    )
    if set(ec["split"].astype(str)) - {"train", "val"}:
        raise ValueError("prepared compatibility rows contain test")
    authoritative_cells = pd.read_parquet(root / "cell_context" / "cell_metadata.parquet")
    split_lookup = authoritative_cells.set_index(
        authoritative_cells["cell_id"].astype(str)
    )["split"].astype(str)

    groups = {
        "nodes": {str(key): value for key, value in nodes.groupby("gene_id", sort=False)},
        "edges": {str(key): value for key, value in edges.groupby("gene_id", sort=False)},
        "paths": {str(key): value for key, value in paths.groupby("gene_id", sort=False)},
        "path_edges": {
            str(key): value for key, value in path_edges.groupby("gene_id", sort=False)
        },
        "cis": {
            str(key): value for key, value in normalized_cis.groupby("target_gene_id", sort=False)
        },
        "physical": {
            str(key): value for key, value in physical.groupby("target_gene_id", sort=False)
        },
        "routes": {
            str(key): value for key, value in routes.groupby("target_gene_id", sort=False)
        },
        "ec": {str(key): value for key, value in ec.groupby("target_gene_id", sort=False)},
    }
    shard_root = root / "prepared_dataset" / "genes" / chromosome
    shard_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    injection_rows: list[pd.DataFrame] = []
    for local_index, gene_id in enumerate(selected_genes):
        missing = [name for name in ("nodes", "edges", "paths", "path_edges", "cis", "ec") if gene_id not in groups[name]]
        if missing:
            raise ValueError(f"prepared gene {gene_id} misses required tables: {missing}")
        gene_graph = build_gene_graph(
            gene_id,
            nodes=groups["nodes"][gene_id],
            edges=groups["edges"][gene_id],
            paths=groups["paths"][gene_id],
            path_edges=groups["path_edges"][gene_id],
        )
        gate_record = _gate_record(root, chromosome, gene_id)
        gate_values: GateValues = gate_record["gate_values"]
        cell_ids = tuple(str(value) for value in gate_record["cell_ids"])
        splits = tuple(str(value) for value in gate_record["cell_split"])
        expected_splits = tuple(split_lookup.loc[list(cell_ids)])
        if splits != expected_splits or set(splits) - {"train", "val"}:
            raise ValueError(f"prepared gate cell/split identity differs for {gene_id}")
        gene_ec = groups["ec"][gene_id].copy()
        ec_mass_by_cell = (
            gene_ec.groupby("cell_id", sort=False)["molecule_count"].sum().astype(np.int64)
        )
        expected_mass = ec_mass_by_cell.reindex(list(cell_ids))
        if expected_mass.isna().any() or not np.array_equal(
            expected_mass.to_numpy(np.int64),
            np.asarray(gate_record["informative_molecule_mass"], dtype=np.int64),
        ):
            raise ValueError(f"gate/prepared informative molecule mass differs for {gene_id}")
        gene_physical = groups["physical"].get(gene_id, physical.iloc[:0].copy())
        gene_routes = groups["routes"].get(gene_id, routes.iloc[:0].copy())
        dna = _production_modality_tensors(
            gene_physical,
            gene_routes,
            gate_values,
            target_gene_id=gene_id,
            modality="DNA",
            ordered_edge_ids=gene_graph.edge_ids,
            feature_manifest=feature_manifest,
            base_manifest=base_manifest,
            interaction_manifest=interaction_manifest,
        )
        rna = _production_modality_tensors(
            gene_physical,
            gene_routes,
            gate_values,
            target_gene_id=gene_id,
            modality="RNA",
            ordered_edge_ids=gene_graph.edge_ids,
            feature_manifest=feature_manifest,
            base_manifest=base_manifest,
            interaction_manifest=interaction_manifest,
        )
        gene_injection_index = _gene_model_injection_index(
            gene_physical,
            gene_routes,
            ordered_edge_ids=gene_graph.edge_ids,
            dna=dna,
            rna=rna,
            base_manifest=base_manifest,
            interaction_manifest=interaction_manifest,
        )
        if not gene_injection_index.empty:
            injection_rows.append(gene_injection_index)
        assembly = assemble_gene_cell_model_input(
            gene_graph,
            cell_split=pd.DataFrame({"cell_id": cell_ids, "split": splits}),
            normalized_cis_edges=groups["cis"][gene_id],
            cis_feature_names=cis_feature_names,
            dna=dna,
            rna=rna,
            compatibility_rows=gene_ec,
        )
        prepared = prepared_gene_from_assembly(assembly)
        destination = shard_root / f"{gene_id}.pt"
        torch.save(prepared, destination)
        records.append(
            {
                "gene_id": gene_id,
                "g_fit_order_0based": g_fit_order[gene_id],
                "relative_path": str(destination.relative_to(root / "prepared_dataset")),
                "edge_count": len(gene_graph.edge_ids),
                "path_count": len(gene_graph.path_ids),
                "cell_count": len(cell_ids),
                "train_cell_count": int(sum(value == "train" for value in splits)),
                "validation_cell_count": int(sum(value == "val" for value in splits)),
                "ec_row_count": len(gene_ec),
                "informative_molecule_mass": int(gene_ec["molecule_count"].sum()),
                "dna_active_event_count": len(dna.event_ids),
                "dna_route_count": len(dna.route_ids),
                "rna_active_event_count": len(rna.event_ids),
                "rna_route_count": len(rna.route_ids),
                "dna_base_nnz": int(dna.route_base_features.nnz),
                "dna_interaction_nnz": int(dna.route_interaction_features.nnz),
                "rna_base_nnz": int(rna.route_base_features.nnz),
                "rna_interaction_nnz": int(rna.route_interaction_features.nnz),
            }
        )
        if (local_index + 1) % 100 == 0:
            print(f"prepared {chromosome}: {local_index + 1}/{len(selected_genes)} genes", flush=True)
    manifest_root = root / "prepared_dataset" / "shard_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    reporting_root = root / "reporting" / "model_injection_equivalence"
    reporting_root.mkdir(parents=True, exist_ok=True)
    injection_index = (
        pd.concat(injection_rows, ignore_index=True)
        if injection_rows
        else pd.DataFrame()
    )
    if injection_index.empty:
        raise ValueError(f"model injection equivalence index is empty for {chromosome}")
    expected_injection_events = sum(
        int(value["dna_active_event_count"]) + int(value["rna_active_event_count"])
        for value in records
    )
    if int(injection_index["member_count"].sum()) != expected_injection_events:
        raise ValueError(
            f"model injection equivalence index does not cover every active event on {chromosome}"
        )
    injection_index.to_parquet(
        reporting_root / f"part-{chromosome}.parquet", index=False
    )
    record = {
        "schema_version": "fabric.prepared_chromosome_shard.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chromosome": chromosome,
        "gene_count": len(records),
        "gene_records": records,
        "model_injection_group_count": len(injection_index),
        "model_injection_event_count": int(injection_index["member_count"].sum()),
        "model_injection_equivalence_index": str(
            (reporting_root / f"part-{chromosome}.parquet").relative_to(root)
        ),
        "test_compatible_rows": 0,
        "test_predictions_or_metrics_computed": False,
    }
    (manifest_root / f"{chromosome}.json").write_text(json.dumps(record, indent=2) + "\n")


def finalize_backed_prepared_dataset(
    real_root: str | Path, compatible_root: str | Path
) -> None:
    root = Path(real_root)
    compatible = Path(compatible_root)
    g_fit = tuple(
        pd.read_csv(compatible / "G_fit.tsv", sep="\t")["target_gene_id"].astype(str)
    )
    records: list[dict[str, object]] = []
    for chromosome in CHROMOSOMES:
        path = root / "prepared_dataset" / "shard_manifests" / f"{chromosome}.json"
        if not path.is_file():
            raise FileNotFoundError(f"prepared chromosome manifest is absent: {path}")
        records.extend(json.loads(path.read_text())["gene_records"])
    records.sort(key=lambda value: int(value["g_fit_order_0based"]))
    if tuple(str(value["gene_id"]) for value in records) != g_fit:
        raise ValueError("prepared shard order/identity differs from frozen G_fit")
    if len(records) != 17_600:
        raise ValueError("prepared dataset does not contain all 17,600 G_fit genes")
    upstream = json.loads((compatible / "CompatibilityArtifactManifest.json").read_text())
    candidate_expected_mass = {
        str(row["split"]): int(row["proper_subset_compatible_molecule_mass"])
        for row in upstream["split_conservation"]
    }
    g_fit_set = set(g_fit)
    expected_mass = {"train": 0, "val": 0}
    graph_only_mass = {"train": 0, "val": 0}
    graph_only_mass_by_gene: dict[str, dict[str, int]] = {}
    for chromosome in CHROMOSOMES:
        ec = pd.read_parquet(
            compatible / "compatible_ec" / f"part-{chromosome}.parquet",
            filters=[("final_fate", "==", "likelihood_informative")],
            columns=["target_gene_id", "split", "molecule_count"],
        )
        if set(ec["split"].astype(str)) - {"train", "val"}:
            raise ValueError("finalize compatibility rows contain test")
        for split in ("train", "val"):
            split_rows = ec.loc[ec["split"].astype(str).eq(split)]
            split_genes = split_rows["target_gene_id"].astype(str)
            in_g_fit = split_genes.isin(g_fit_set)
            expected_mass[split] += int(
                split_rows.loc[in_g_fit, "molecule_count"].sum()
            )
            outside = split_rows.loc[~in_g_fit].groupby(
                split_rows.loc[~in_g_fit, "target_gene_id"].astype(str),
                sort=False,
            )["molecule_count"].sum()
            for gene_id, mass in outside.items():
                value = int(mass)
                graph_only_mass[split] += value
                graph_only_mass_by_gene.setdefault(str(gene_id), {})[split] = value
    for split in ("train", "val"):
        if (
            expected_mass[split] + graph_only_mass[split]
            != candidate_expected_mass[split]
        ):
            raise ValueError(
                f"G_fit plus graph-only {split} mass differs from upstream conservation"
            )
    if graph_only_mass["train"] != 0:
        raise ValueError("graph-only genes unexpectedly contain train informative mass")
    actual_total = sum(int(value["informative_molecule_mass"]) for value in records)
    if actual_total != expected_mass["train"] + expected_mass["val"]:
        raise ValueError("prepared molecule mass differs from G_fit K^inf mass")
    created_at = datetime.now(timezone.utc).isoformat()
    source_commit, _ = _validated_real_dataset_source(root)
    input_manifest_id, compatibility_artifact_id = _prepared_artifact_identities(
        root, compatible
    )
    manifest = {
        "schema_version": "fabric.backed_prepared_dataset.v1",
        "created_at_utc": created_at,
        "source_git_commit": source_commit,
        "input_manifest_id": input_manifest_id,
        "compatibility_artifact_id": compatibility_artifact_id,
        "informative_gene_ids": list(g_fit),
        "gene_shards": [
            {"gene_id": value["gene_id"], "relative_path": value["relative_path"]}
            for value in records
        ],
        "gene_record_audit": records,
        "g_fit_gene_count": len(records),
        "train_validation_informative_molecule_mass": actual_total,
        "expected_train_informative_molecule_mass": expected_mass["train"],
        "expected_validation_informative_molecule_mass": expected_mass["val"],
        "candidate_axis_train_informative_molecule_mass": (
            candidate_expected_mass["train"]
        ),
        "candidate_axis_validation_informative_molecule_mass": (
            candidate_expected_mass["val"]
        ),
        "graph_only_train_informative_molecule_mass": graph_only_mass["train"],
        "graph_only_validation_informative_molecule_mass": graph_only_mass["val"],
        "graph_only_informative_mass_by_gene_and_split": graph_only_mass_by_gene,
        "test_compatible_rows": 0,
        "test_predictions_or_metrics_computed": False,
        "training_started": False,
        "final_test_authorized": False,
    }
    destination = root / "prepared_dataset" / "PreparedDatasetManifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-inputs", default="data/external_inputs.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "feature-design", "raw-support", "interaction-basis", "base-design",
            "route-rank", "prepared", "finalize",
        ),
    )
    parser.add_argument("--chromosome")
    args = parser.parse_args(argv)
    paths = _load_external_paths(args.external_inputs)
    real_root = paths["real_dataset"]
    compatible_root = paths["compatible_ec"]
    _validated_real_dataset_source(real_root)
    if args.stage == "feature-design":
        build_split_neutral_feature_design(real_root)
    elif args.stage == "raw-support":
        measure_real_raw_interaction_support(real_root, compatible_root)
    elif args.stage == "interaction-basis":
        build_real_interaction_basis(real_root)
    elif args.stage == "base-design":
        build_production_base_design(real_root)
    elif args.stage == "route-rank":
        audit_production_route_design(real_root)
    elif args.stage == "prepared":
        if not args.chromosome:
            raise ValueError("prepared stage requires --chromosome")
        build_prepared_chromosome(
            real_root, compatible_root, chromosome=args.chromosome
        )
    elif args.stage == "finalize":
        finalize_backed_prepared_dataset(real_root, compatible_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
