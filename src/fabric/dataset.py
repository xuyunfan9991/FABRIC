"""Sparse cell context, observed masks, and train-only gate baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import anndata as ad
import hdf5plugin  # noqa: F401 - registers the Blosc filters used by the ATAC H5AD
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import IncrementalPCA

from .annotation import (
    ExternalInputs,
    ReferenceSequence,
    canonical_atac_cell_id,
    canonical_rna_cell_id,
    iter_peak_bed,
    load_external_inputs,
    load_gene_symbol_map,
    load_split_rows,
    resolve_and_validate_graph_generation,
    validate_peak_axis,
)
from .choices import (
    ChoiceCatalog,
    choice_identifiability,
    extract_elementary_choices,
)
from .graph import (
    GeneGraph,
    bind_authoritative_split,
    load_compatibility_rows,
    load_graph_tables,
    normalize_compatibility_path_order,
    split_gene_graphs,
    validate_compatibility_rows,
)
from .motifs import (
    FactorCatalogResult,
    PWM,
    build_dna_peak_regions,
    build_factor_catalog,
    build_rna_choice_regions,
    cap_motif_events,
    event_relation_matrix,
    fixed_event_feature_matrix,
    parse_cisbp_motifs,
    parse_meme_motifs,
    scan_motif_regions,
)

if TYPE_CHECKING:
    from .train import PreparedDataset, PreparedGene


STATE_PCA_FIT_BATCH_SIZE = 2_048


@dataclass(frozen=True)
class RNAStatePCA:
    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    log_library_mean: float
    log_library_scale: float
    target_sum: float
    fit_batch_size: int


@dataclass(frozen=True)
class CenteringBaseline:
    mean: np.ndarray
    valid_molecule_mass: np.ndarray
    weighted_variance: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class DNACenteringBaseline:
    mean: np.ndarray
    valid_molecule_mass: np.ndarray
    weighted_variance: np.ndarray
    dna_reliability_mass: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class ATACContext:
    accessibility: sparse.csr_matrix
    observed: np.ndarray
    reliability: np.ndarray
    cell_ids: tuple[str, ...]
    peak_ids: tuple[str, ...]


@dataclass(frozen=True)
class FullRNAGLUEContext:
    """The two validated modality axes from the one allowed co-embedding."""

    rna_cell_ids: tuple[str, ...]
    rna_embedding: np.ndarray
    rna_stage: tuple[str, ...]
    rna_developmental_system: tuple[str, ...]
    atac_cell_ids: tuple[str, ...]
    atac_embedding: np.ndarray
    atac_stage: tuple[str, ...]
    atac_donor_ids: tuple[str, ...]
    atac_developmental_system: tuple[str, ...]


@dataclass(frozen=True)
class FactorActivityContext:
    """Frozen global factor/activity axis for the one stage-by-system null."""

    cell_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    activity: np.ndarray
    observed: np.ndarray
    stage: tuple[str, ...]
    developmental_system: tuple[str, ...]


@dataclass(frozen=True)
class OrderedCellState:
    """RNA-only State values with their authoritative row identities."""

    cell_ids: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class OrderedEventData:
    """One modality's event tensors with every matrix axis named explicitly."""

    events: pd.DataFrame
    feature_event_ids: tuple[str, ...]
    features: np.ndarray
    relation_event_ids: tuple[str, ...]
    relation_alternative_ids: tuple[str, ...]
    relation: np.ndarray
    gate_cell_ids: tuple[str, ...]
    gate_event_ids: tuple[str, ...]
    gate: np.ndarray


@dataclass(frozen=True)
class PreparationSources:
    graph_generation: str
    split_source: str


@dataclass(frozen=True)
class OrderedChoiceAudit:
    """Static choice diagnostics; span is ``abs(exit_pos - entry_pos)`` in bp."""

    choice_ids: tuple[str, ...]
    alternative_span: np.ndarray
    dna_candidate_event_count: np.ndarray
    dna_selected_event_count: np.ndarray
    dna_cap_saturated: np.ndarray
    dna_boundary_rank_motif_score: np.ndarray
    rna_candidate_event_count: np.ndarray
    rna_selected_event_count: np.ndarray
    rna_cap_saturated: np.ndarray
    rna_boundary_rank_motif_score: np.ndarray


def prepare_dataset_from_external_inputs(
    manifest_path: str | Path,
    config_path: str | Path,
    *,
    target_gene_ids: Sequence[str],
    reviewed_factor_mapping_path: str | Path,
    atac_donor_eligibility_path: str | Path,
    peak_support_path: str | Path,
    neighbor_device: str = "cpu",
    io_batch_size: int = 2_048,
    output_path: str | Path | None = None,
) -> PreparedDataset:
    """Build the V1 training tensors directly from the frozen normalized inputs.

    The selected gene list is explicit so a real fixture or diagnostic panel can
    run without changing the full7198 identity.  EC rows are read per gene, and
    RNA/ATAC matrices are accessed in backed row chunks and selected event columns.
    No GLUE coordinate enters the State feature matrix.
    """

    from .train import (
        NORMALIZED_SOURCE_ROLES,
        load_config,
        preparation_values_from_config,
        prepare_dataset_identity,
    )

    if io_batch_size <= 0:
        raise ValueError("io_batch_size must be positive")
    targets = _ordered_unique_ids("target gene", target_gene_ids, allow_empty=False)
    inputs = load_external_inputs(manifest_path)
    config = load_config(config_path)
    preparation_values = preparation_values_from_config(config)
    _require_manifest_identity(config, config_path, manifest_path)
    _require_auxiliary_source_identity(
        config,
        reviewed_factor_mapping_path=reviewed_factor_mapping_path,
        atac_donor_eligibility_path=atac_donor_eligibility_path,
        peak_support_path=peak_support_path,
    )
    parameters = _preparation_parameters(config)
    graph_generation = resolve_and_validate_graph_generation(inputs)

    graph_tables = load_graph_tables(graph_generation, gene_ids=targets)
    graph_by_gene = {graph.gene_id: graph for graph in split_gene_graphs(graph_tables)}
    missing_graphs = [gene_id for gene_id in targets if gene_id not in graph_by_gene]
    if missing_graphs:
        raise ValueError(
            f"target genes are absent from the graph: {missing_graphs[:10]}"
        )
    graphs = tuple(graph_by_gene[gene_id] for gene_id in targets)

    split_rows = load_split_rows(inputs.path("cell_split"))
    split_cell_ids = tuple(split_rows["cell_id"].astype(str))
    catalogs: dict[str, ChoiceCatalog] = {}
    identifiability: dict[str, pd.DataFrame] = {}
    for graph in graphs:
        catalog = extract_elementary_choices(graph)
        rows = _load_gene_ec_rows(
            inputs.path("compatibility_ec"), graph, split_rows, split_cell_ids
        )
        catalogs[graph.gene_id] = catalog
        if rows.empty:
            identifiability[graph.gene_id] = _no_supervision_identifiability(catalog)
            continue
        identifiability[graph.gene_id] = choice_identifiability(
            catalog,
            rows,
            rank_tolerance=float(parameters["rank_tolerance"]),
            minimum_informative_molecule_mass=float(
                parameters["minimum_informative_molecule_mass"]
            ),
            minimum_alternative_support=float(
                parameters["minimum_alternative_support"]
            ),
        )
    rna_cell_ids, rna_rows = _target_rna_rows(
        inputs.path("rna_counts"),
        set(split_cell_ids),
        expected_shape=_expected_shape(inputs.expected, "rna_count_shape"),
    )
    split_by_cell = split_rows.set_index("cell_id")["split"].astype(str).to_dict()
    train_rows = np.asarray(
        [
            row
            for cell_id, row in zip(rna_cell_ids, rna_rows, strict=True)
            if split_by_cell[cell_id] == "train"
        ],
        dtype=np.int64,
    )
    state_components = int(parameters["state_pca_dim"])
    state_fit = fit_rna_state_pca(
        inputs.path("rna_counts"),
        train_rows,
        n_components=state_components,
        target_sum=float(parameters["target_sum_rna"]),
        batch_size=STATE_PCA_FIT_BATCH_SIZE,
    )
    state_values = transform_rna_state(
        inputs.path("rna_counts"),
        rna_rows,
        state_fit,
        batch_size=io_batch_size,
    )

    glue_shape = _expected_shape(inputs.expected, "glue_embedding_shape")
    glue = load_full_rna_glue_context(
        inputs.path("full_rna_glue_embedding"),
        inputs.path("full_rna_atac_peak_counts"),
        target_rna_cell_ids=rna_cell_ids,
        expected_rna_count=_expected_int(inputs.expected, "glue_rna_count"),
        expected_atac_count=_expected_int(inputs.expected, "glue_atac_count"),
        expected_embedding_dim=glue_shape[1],
    )
    donor_eligible = _load_donor_eligibility(
        atac_donor_eligibility_path, glue.atac_donor_ids
    )
    neighbors, _ = build_exact_stage_neighbors(
        rna_cell_ids=glue.rna_cell_ids,
        rna_embedding=glue.rna_embedding,
        rna_stage=glue.rna_stage,
        atac_cell_ids=glue.atac_cell_ids,
        atac_embedding=glue.atac_embedding,
        atac_stage=glue.atac_stage,
        atac_donor_ids=glue.atac_donor_ids,
        atac_donor_eligible=donor_eligible,
        k=int(parameters["atac_neighbor_k"]),
        temperature=float(parameters["atac_neighbor_temperature"]),
        device=neighbor_device,
        query_chunk_size=io_batch_size,
    )

    factor_catalog = _load_reviewed_factor_catalog(
        inputs,
        reviewed_factor_mapping_path,
    )
    factor_order = tuple(factor_catalog.factors["factor_id"].astype(str))
    activity_gene_by_factor = (
        factor_catalog.factors.set_index("factor_id")["activity_gene_id"]
        .astype(str)
        .to_dict()
    )
    activity_gene_ids = tuple(
        dict.fromkeys(factor_catalog.factors["activity_gene_id"].astype(str).tolist())
    )
    activity_by_gene, observed_by_gene = read_factor_activity(
        inputs.path("rna_counts"),
        rna_rows,
        activity_gene_ids,
        target_sum=float(parameters["target_sum_rna"]),
        batch_size=io_batch_size,
    )
    activity_gene_index = {
        gene_id: index for index, gene_id in enumerate(activity_gene_ids)
    }
    activity_columns = [
        activity_gene_index[activity_gene_by_factor[factor_id]]
        for factor_id in factor_order
    ]
    factor_activity = np.asarray(
        activity_by_gene[:, activity_columns], dtype=np.float32
    )
    factor_observed = np.asarray(observed_by_gene[:, activity_columns], dtype=bool)
    del activity_by_gene, observed_by_gene
    activity_column_by_factor = {
        factor_id: index for index, factor_id in enumerate(factor_order)
    }
    dna_motifs = parse_meme_motifs(inputs.path("dna_motif_library"))
    rna_motif_ids = tuple(
        factor_catalog.motif_mapping.loc[
            factor_catalog.motif_mapping["modality"].astype(str) == "RNA",
            "motif_id",
        ].astype(str)
    )
    rna_motifs = parse_cisbp_motifs(
        inputs.path("rna_motif_directory"), motif_ids=rna_motif_ids
    )

    expected_peak_count = _expected_int(inputs.expected, "consensus_peak_count")
    validate_peak_axis(
        inputs.path("full_rna_consensus_peak_bed"),
        inputs.path("full_rna_atac_peak_counts"),
        expected_count=expected_peak_count,
    )
    peaks = _load_consensus_peaks(
        inputs.path("full_rna_consensus_peak_bed"), peak_support_path
    )
    event_catalogs: dict[
        str, tuple[pd.DataFrame, pd.DataFrame, OrderedChoiceAudit]
    ] = {}
    with ReferenceSequence(inputs.path("reference_fasta")) as reference:
        for graph in graphs:
            event_catalogs[graph.gene_id] = _build_gene_event_catalogs(
                graph,
                catalogs[graph.gene_id],
                peaks,
                reference,
                dna_motifs=dna_motifs,
                rna_motifs=rna_motifs,
                factor_mapping=factor_catalog.motif_mapping,
                dna_window_bp=int(parameters["dna_window_bp"]),
                rna_window_bp=int(parameters["rna_window_bp"]),
                dna_minimum_relative_score=float(
                    parameters["dna_minimum_relative_score"]
                ),
                rna_minimum_relative_score=float(
                    parameters["rna_minimum_relative_score"]
                ),
                dna_events_per_choice_cap=int(parameters["dna_events_per_choice_cap"]),
                rna_events_per_choice_cap=int(parameters["rna_events_per_choice_cap"]),
            )

    global_dna_peak_ids = tuple(
        dict.fromkeys(
            peak_id
            for gene_id in targets
            for peak_id in event_catalogs[gene_id][0]["peak_id"].astype(str)
        )
    )
    if global_dna_peak_ids:
        atac_context = aggregate_atac_accessibility(
            inputs.path("full_rna_atac_peak_counts"),
            neighbors,
            target_cell_ids=rna_cell_ids,
            peak_ids=global_dna_peak_ids,
            target_sum=float(parameters["target_sum_atac"]),
            expected_k=int(parameters["atac_neighbor_k"]),
            batch_size=io_batch_size,
        )
    else:
        atac_context = ATACContext(
            accessibility=sparse.csr_matrix((len(rna_cell_ids), 0), dtype=np.float32),
            observed=np.zeros(len(rna_cell_ids), dtype=bool),
            reliability=np.zeros(len(rna_cell_ids), dtype=np.float32),
            cell_ids=rna_cell_ids,
            peak_ids=(),
        )
    global_peak_index = {
        peak_id: index for index, peak_id in enumerate(global_dna_peak_ids)
    }

    global_cell_index = {cell_id: index for index, cell_id in enumerate(rna_cell_ids)}
    prepared_genes: list[PreparedGene] = []
    sources = PreparationSources(
        graph_generation=str(graph_generation.resolve()),
        split_source=str(inputs.path("cell_split").resolve()),
    )
    for graph in graphs:
        ec_rows = _load_gene_ec_rows(
            inputs.path("compatibility_ec"), graph, split_rows, split_cell_ids
        )
        dna_events, rna_events, choice_audit = event_catalogs[graph.gene_id]
        prepared_genes.append(
            _prepare_external_gene(
                graph,
                catalogs[graph.gene_id],
                ec_rows,
                identifiability[graph.gene_id],
                dna_events=dna_events,
                rna_events=rna_events,
                choice_audit=choice_audit,
                target_cell_ids=rna_cell_ids,
                global_cell_index=global_cell_index,
                state_values=state_values,
                factor_activity=factor_activity,
                factor_observed=factor_observed,
                activity_column_by_factor=activity_column_by_factor,
                atac_context=atac_context,
                global_peak_index=global_peak_index,
                factor_order=factor_order,
                minimum_valid_mass=float(parameters["minimum_valid_mass"]),
                minimum_variance=float(parameters["minimum_weighted_variance"]),
                dna_distance_scale=float(parameters["dna_window_bp"]),
                rna_distance_scale=float(parameters["rna_window_bp"]),
                sources=sources,
            )
        )

    normalized_sources = {role: inputs.path(role) for role in NORMALIZED_SOURCE_ROLES}
    dataset = prepare_dataset_identity(
        prepared_genes,
        factor_mapping_reviewed=True,
        normalized_source_paths=normalized_sources,
        reviewed_factor_mapping=reviewed_factor_mapping_path,
        atac_donor_eligibility_source=atac_donor_eligibility_path,
        peak_support_source=peak_support_path,
        preparation_config_source=config_path,
        preparation_values=preparation_values,
        state_pca=state_fit,
        factor_context=FactorActivityContext(
            cell_ids=rna_cell_ids,
            factor_ids=factor_order,
            activity=factor_activity,
            observed=factor_observed,
            stage=glue.rna_stage,
            developmental_system=glue.rna_developmental_system,
        ),
        atac_context=atac_context,
    )
    validate_prepared_external_context(dataset)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, destination)
    return dataset


def validate_prepared_external_context(prepared: PreparedDataset) -> None:
    """Validate the minimal frozen context needed for F2 audit and the F3 null."""

    state_pca = prepared.state_pca
    factor = prepared.factor_context
    atac = prepared.atac_context
    if state_pca is None or factor is None or atac is None:
        raise ValueError("prepared dataset lacks State/factor/ATAC frozen context")
    if state_pca.fit_batch_size != STATE_PCA_FIT_BATCH_SIZE:
        raise ValueError("State PCA was not fitted with the frozen V1 batch size")
    factor_values = np.asarray(factor.activity)
    factor_observed = np.asarray(factor.observed, dtype=bool)
    cell_ids = tuple(str(value) for value in factor.cell_ids)
    if (
        factor_values.shape != factor_observed.shape
        or factor_values.shape != (len(cell_ids), len(factor.factor_ids))
        or len(set(cell_ids)) != len(cell_ids)
        or len(set(factor.factor_ids)) != len(factor.factor_ids)
        or len(factor.stage) != len(cell_ids)
        or len(factor.developmental_system) != len(cell_ids)
    ):
        raise ValueError("factor null-context axes differ")
    if not np.isfinite(factor_values[factor_observed]).all() or bool(
        (factor_values[factor_observed] < 0).any()
    ):
        raise ValueError("observed factor activity must be finite and non-negative")
    if atac.cell_ids != cell_ids or atac.accessibility.shape != (
        len(cell_ids),
        len(atac.peak_ids),
    ):
        raise ValueError("ATAC and factor null-context axes differ")
    if not sparse.isspmatrix_csr(atac.accessibility):
        raise TypeError("prepared ATAC context must remain CSR sparse")
    if (
        np.asarray(atac.observed).shape != (len(cell_ids),)
        or np.asarray(atac.reliability).shape != (len(cell_ids),)
        or bool(
            (
                (np.asarray(atac.reliability) < 0) | (np.asarray(atac.reliability) > 1)
            ).any()
        )
    ):
        raise ValueError("prepared ATAC observed/reliability axes differ")
    factor_ids = set(factor.factor_ids)
    peak_ids = set(atac.peak_ids)
    for gene in prepared.genes:
        if (
            len(gene.dna_event_factor_ids) != len(gene.dna_event_ids)
            or len(gene.rna_event_factor_ids) != len(gene.rna_event_ids)
            or len(gene.dna_event_peak_ids) != len(gene.dna_event_ids)
        ):
            raise ValueError(f"gene {gene.gene_id} lacks event null-context identity")
        if not set(gene.dna_event_factor_ids).issubset(factor_ids) or not set(
            gene.rna_event_factor_ids
        ).issubset(factor_ids):
            raise ValueError(f"gene {gene.gene_id} event factor is outside context")
        if not set(gene.dna_event_peak_ids).issubset(peak_ids):
            raise ValueError(f"gene {gene.gene_id} event peak is outside ATAC context")
        if any(
            baseline is None
            for baseline in (gene.state_baseline, gene.dna_baseline, gene.rna_baseline)
        ):
            raise ValueError(f"gene {gene.gene_id} lacks frozen gate baseline audit")


def _require_manifest_identity(
    config: Mapping[str, object],
    config_path: str | Path,
    manifest_path: str | Path,
) -> None:
    declared = config.get("external_inputs")
    if not isinstance(declared, str) or not declared:
        raise ValueError("preparation config requires one external_inputs path")
    if Path(declared).resolve() != Path(manifest_path).resolve():
        raise ValueError(
            f"config {config_path} and requested external-input manifest differ"
        )


def _require_auxiliary_source_identity(
    config: Mapping[str, object],
    *,
    reviewed_factor_mapping_path: str | Path,
    atac_donor_eligibility_path: str | Path,
    peak_support_path: str | Path,
) -> None:
    data = config.get("data")
    motifs = config.get("motifs")
    factor_identity = config.get("factor_identity")
    if not all(isinstance(value, Mapping) for value in (data, motifs, factor_identity)):
        raise ValueError(
            "preparation config requires data, motifs, and factor_identity mappings"
        )
    assert isinstance(data, Mapping)
    assert isinstance(motifs, Mapping)
    assert isinstance(factor_identity, Mapping)
    neighbors = data.get("atac_neighbors")
    if not isinstance(neighbors, Mapping):
        raise ValueError("preparation config misses data.atac_neighbors")
    declared = {
        "factor_identity.reviewed_mapping": (
            factor_identity.get("reviewed_mapping"),
            reviewed_factor_mapping_path,
        ),
        "data.atac_neighbors.donor_eligibility_path": (
            neighbors.get("donor_eligibility_path"),
            atac_donor_eligibility_path,
        ),
        "motifs.peak_support_path": (
            motifs.get("peak_support_path"),
            peak_support_path,
        ),
    }
    unresolved = [label for label, (value, _) in declared.items() if value is None]
    if unresolved:
        raise ValueError(f"preparation source paths are unresolved: {unresolved}")
    for label, (value, observed) in declared.items():
        if (
            not isinstance(value, str)
            or Path(value).resolve() != Path(observed).resolve()
        ):
            raise ValueError(f"prepared source differs from config identity: {label}")


def _preparation_parameters(config: Mapping[str, object]) -> dict[str, float | int]:
    data = config.get("data")
    choices = config.get("choices")
    motifs = config.get("motifs")
    gates = config.get("gates")
    if not all(isinstance(value, Mapping) for value in (data, choices, motifs, gates)):
        raise ValueError("preparation config misses data/choices/motifs/gates mappings")
    assert isinstance(data, Mapping)
    assert isinstance(choices, Mapping)
    assert isinstance(motifs, Mapping)
    assert isinstance(gates, Mapping)
    neighbor = data.get("atac_neighbors")
    if not isinstance(neighbor, Mapping):
        raise ValueError("preparation config misses data.atac_neighbors")
    if data.get("reuse_documented_cell_split") is not True:
        raise ValueError("FABRIC V1 preparation must reuse the documented cell split")
    fixed_neighbor_contract = {
        "exact_stage": True,
        "stage_field": "stage_scanvi",
        "donor_id_field": "sample_id",
        "donor_eligibility_rule": "explicit_boolean_mask",
        "weighting": "softmax_negative_euclidean_distance",
    }
    for field, expected in fixed_neighbor_contract.items():
        if neighbor.get(field) != expected:
            raise ValueError(
                f"data.atac_neighbors.{field} must be fixed to {expected!r}"
            )

    fields = {
        "target_sum_rna": data.get("target_sum_rna"),
        "target_sum_atac": data.get("target_sum_atac"),
        "state_pca_dim": data.get("state_pca_dim"),
        "atac_neighbor_k": neighbor.get("k"),
        "atac_neighbor_temperature": neighbor.get("temperature"),
        "rank_tolerance": choices.get("rank_tolerance"),
        "minimum_informative_molecule_mass": choices.get(
            "minimum_informative_molecule_mass"
        ),
        "minimum_alternative_support": choices.get("minimum_alternative_support"),
        "dna_window_bp": motifs.get("dna_window_bp"),
        "rna_window_bp": motifs.get("rna_window_bp"),
        "dna_minimum_relative_score": motifs.get("dna_minimum_relative_score"),
        "rna_minimum_relative_score": motifs.get("rna_minimum_relative_score"),
        "dna_events_per_choice_cap": motifs.get("dna_events_per_choice_cap"),
        "rna_events_per_choice_cap": motifs.get("rna_events_per_choice_cap"),
        "minimum_valid_mass": gates.get("minimum_valid_molecule_mass"),
        "minimum_weighted_variance": gates.get("minimum_weighted_variance"),
    }
    unresolved = [name for name, value in fields.items() if value is None]
    if unresolved:
        raise ValueError(f"preparation numerical fields are unresolved: {unresolved}")
    integer_fields = {
        "state_pca_dim",
        "atac_neighbor_k",
        "dna_window_bp",
        "rna_window_bp",
        "dna_events_per_choice_cap",
        "rna_events_per_choice_cap",
    }
    non_negative_fields = {
        "minimum_informative_molecule_mass",
        "minimum_alternative_support",
        "minimum_valid_mass",
        "minimum_weighted_variance",
        "dna_minimum_relative_score",
        "rna_minimum_relative_score",
    }
    result: dict[str, float | int] = {}
    for name, value in fields.items():
        numeric = float(value)  # type: ignore[arg-type]
        if not np.isfinite(numeric) or (
            numeric < 0 if name in non_negative_fields else numeric <= 0
        ):
            qualifier = "non-negative" if name in non_negative_fields else "positive"
            raise ValueError(f"preparation numerical field {name} must be {qualifier}")
        if name in integer_fields:
            if not numeric.is_integer():
                raise ValueError(f"preparation numerical field {name} must be integral")
            result[name] = int(numeric)
        else:
            result[name] = numeric
    for name in ("dna_minimum_relative_score", "rna_minimum_relative_score"):
        if float(result[name]) > 1:
            raise ValueError(f"preparation numerical field {name} must not exceed 1")
    return result


def _expected_shape(expected: Mapping[str, object], field: str) -> tuple[int, int]:
    value = expected.get(field)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"external-input expected.{field} must be a two-item shape")
    shape = tuple(int(item) for item in value)
    if any(item < 0 for item in shape):
        raise ValueError(f"external-input expected.{field} contains a negative size")
    return shape  # type: ignore[return-value]


def _expected_int(expected: Mapping[str, object], field: str) -> int:
    value = expected.get(field)
    if value is None or isinstance(value, bool):
        raise ValueError(f"external-input expected.{field} is missing")
    result = int(value)
    if result <= 0:
        raise ValueError(f"external-input expected.{field} must be positive")
    return result


def _load_gene_ec_rows(
    path: str | Path,
    graph: GeneGraph,
    split_rows: pd.DataFrame,
    split_cell_ids: Sequence[str],
) -> pd.DataFrame:
    parts = list(
        load_compatibility_rows(
            path,
            gene_ids=[graph.gene_id],
            cell_ids=split_cell_ids,
        )
    )
    if not parts:
        return pd.DataFrame(
            {
                "cell_id": pd.Series(dtype=str),
                "gene_id": pd.Series(dtype=str),
                "compatible_path_ids": pd.Series(dtype=object),
                "compatible_path_indices": pd.Series(dtype=object),
                "compatible_path_count": pd.Series(dtype=np.int64),
                "molecule_count": pd.Series(dtype=np.int64),
                "split": pd.Series(dtype=str),
            }
        )
    rows = pd.concat(parts, ignore_index=True)
    rows = bind_authoritative_split(rows, split_rows)
    return normalize_compatibility_path_order(rows, graph)


def _no_supervision_identifiability(catalog: ChoiceCatalog) -> pd.DataFrame:
    rows = []
    for choice in catalog.choices:
        alternative_count = len(choice.alternatives)
        rows.append(
            {
                "choice_id": choice.choice_id,
                "gene_id": catalog.gene_id,
                "alternative_count": alternative_count,
                "structural_rank": alternative_count - 1,
                "supervision_rank": 0,
                "informative_ec_count": 0,
                "informative_molecule_mass": 0.0,
                "alternative_support": [0.0] * alternative_count,
                "eligible": False,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "choice_id",
            "gene_id",
            "alternative_count",
            "structural_rank",
            "supervision_rank",
            "informative_ec_count",
            "informative_molecule_mass",
            "alternative_support",
            "eligible",
        ],
    )


def _target_rna_rows(
    path: str | Path,
    selected_cell_ids: set[str],
    *,
    expected_shape: tuple[int, int],
) -> tuple[tuple[str, ...], np.ndarray]:
    matrix = ad.read_h5ad(path, backed="r")
    try:
        if matrix.shape != expected_shape:
            raise ValueError(
                f"RNA count shape differs from frozen manifest: {matrix.shape}"
            )
        axis = tuple(canonical_rna_cell_id(value) for value in matrix.obs_names)
    finally:
        matrix.file.close()
    if len(set(axis)) != len(axis):
        raise ValueError("RNA count cell axis is not unique after canonicalization")
    missing = sorted(selected_cell_ids - set(axis))
    if missing:
        raise ValueError(
            f"supervision cells are absent from RNA counts: {missing[:10]}"
        )
    rows = np.asarray(
        [index for index, cell_id in enumerate(axis) if cell_id in selected_cell_ids],
        dtype=np.int64,
    )
    return tuple(axis[index] for index in rows), rows


def _load_donor_eligibility(
    path: str | Path, atac_donor_ids: Sequence[str]
) -> np.ndarray:
    rows = pd.read_csv(path, sep="\t")
    required = {"donor_id", "eligible"}
    if required - set(rows):
        raise ValueError("ATAC donor eligibility TSV requires donor_id and eligible")
    donor = rows["donor_id"].astype(str)
    if donor.str.len().eq(0).any() or donor.duplicated().any():
        raise ValueError("ATAC donor eligibility donor_id values must be unique")
    raw = rows["eligible"]
    if pd.api.types.is_bool_dtype(raw):
        eligible = raw.astype(bool)
    else:
        text = raw.astype(str).str.lower()
        if not text.isin({"true", "false"}).all():
            raise ValueError("ATAC donor eligibility must contain explicit true/false")
        eligible = text == "true"
    mapping = dict(zip(donor, eligible, strict=True))
    expected = set(str(value) for value in atac_donor_ids)
    missing = sorted(expected - set(mapping))
    extra = sorted(set(mapping) - expected)
    if missing or extra:
        raise ValueError(
            "ATAC donor eligibility axis differs from peak metadata: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return np.asarray([mapping[str(value)] for value in atac_donor_ids], dtype=bool)


def _load_reviewed_factor_catalog(
    inputs: ExternalInputs, reviewed_mapping_path: str | Path
) -> FactorCatalogResult:
    path = Path(reviewed_mapping_path)
    reviewed = pd.read_csv(path, sep="\t", dtype=str)
    required = {
        "modality",
        "motif_id",
        "factor_id",
        "factor_name",
        "activity_gene_id",
        "factor_group_id",
    }
    if required - set(reviewed):
        raise ValueError("reviewed factor mapping misses required identity columns")
    reviewed = reviewed[list(required)].copy()
    reviewed["modality"] = reviewed["modality"].str.upper()
    if not set(reviewed["modality"]).issubset({"DNA", "RNA"}):
        raise ValueError("reviewed factor mapping modality must be DNA or RNA")
    if reviewed.empty or reviewed.duplicated(["modality", "motif_id"]).any():
        raise ValueError("reviewed factor mapping keys must be non-empty and unique")

    source_catalog = build_factor_catalog(
        inputs.path("dna_motif_index"),
        inputs.path("rna_motif_gene_map"),
        gene_symbol_to_id=load_gene_symbol_map(inputs.path("rna_gene_gtf")),
        explicit_mapping=reviewed,
    )
    source_mapping = source_catalog.motif_mapping.copy()
    source_mapping["modality"] = source_mapping["modality"].astype(str).str.upper()
    keyed = source_mapping.set_index(["modality", "motif_id"], drop=False)
    selected_rows = []
    identity_columns = (
        "factor_id",
        "factor_name",
        "activity_gene_id",
        "factor_group_id",
    )
    for row in reviewed.itertuples(index=False):
        key = (str(row.modality), str(row.motif_id))
        if key not in keyed.index:
            raise ValueError(f"reviewed mapping motif is absent from its source: {key}")
        source = keyed.loc[key]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"source factor mapping is not unique for {key}")
        for column in identity_columns:
            expected = str(getattr(row, column)).split(".", 1)[0]
            observed = str(source[column]).split(".", 1)[0]
            if observed != expected:
                raise ValueError(f"reviewed factor identity differs for {key} {column}")
        selected_rows.append(source.to_dict())
    mapping = (
        pd.DataFrame(selected_rows)
        .sort_values(["factor_id", "modality", "motif_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    factors = []
    for factor_id, group in mapping.groupby("factor_id", sort=True):
        activity = set(group["activity_gene_id"].astype(str))
        factor_groups = set(group["factor_group_id"].astype(str))
        names = sorted(set(group["factor_name"].astype(str)))
        if len(activity) != 1 or len(factor_groups) != 1:
            raise ValueError(f"reviewed factor {factor_id} has inconsistent identity")
        dna = sorted(group.loc[group["modality"] == "DNA", "motif_id"].astype(str))
        rna = sorted(group.loc[group["modality"] == "RNA", "motif_id"].astype(str))
        factors.append(
            {
                "factor_id": str(factor_id),
                "factor_name": names[0],
                "activity_gene_id": next(iter(activity)),
                "factor_group_id": next(iter(factor_groups)),
                "has_dna_motif": bool(dna),
                "has_rna_motif": bool(rna),
                "canonical_label": names[0],
                "dna_motif_ids": dna,
                "rna_motif_ids": rna,
            }
        )
    return FactorCatalogResult(
        factors=pd.DataFrame(factors),
        motif_mapping=mapping,
        excluded_motifs=source_catalog.excluded_motifs,
    )


def _load_consensus_peaks(
    bed_path: str | Path, support_path: str | Path
) -> pd.DataFrame:
    support_source = Path(support_path)
    support = (
        pd.read_parquet(support_source, columns=["peak_id", "peak_support"])
        if support_source.suffix == ".parquet"
        else pd.read_csv(support_source, sep="\t")
    )
    required = {"peak_id", "peak_support"}
    if required - set(support):
        raise ValueError("reviewed peak support requires peak_id and peak_support")
    support = support[["peak_id", "peak_support"]].copy()
    support["peak_id"] = support["peak_id"].astype(str)
    if (
        support["peak_id"].str.len().eq(0).any()
        or support["peak_id"].duplicated().any()
    ):
        raise ValueError("reviewed peak-support IDs must be unique and non-empty")
    values = support["peak_support"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("reviewed peak support must be finite and non-negative")
    bed_rows = list(iter_peak_bed(bed_path))
    bed_ids = tuple(row[3] for row in bed_rows)
    if tuple(support["peak_id"]) != bed_ids:
        raise ValueError("reviewed peak-support axis differs from the consensus BED")
    return pd.DataFrame(
        {
            "chrom": [row[0] for row in bed_rows],
            "start_0based": [row[1] for row in bed_rows],
            "end_0based": [row[2] for row in bed_rows],
            "peak_id": bed_ids,
            "peak_support": values,
        }
    )


def _empty_event_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "modality",
            "gene_id",
            "choice_id",
            "relation_alternative_ids",
            "factor_id",
            "factor_group_id",
            "motif_id",
            "chrom",
            "start_0based",
            "end_0based",
            "orientation",
            "anchor_0based",
            "signed_distance_bp",
            "region_type",
            "peak_id",
            "peak_support",
            "motif_score",
        ]
    )


def _build_gene_event_catalogs(
    graph: GeneGraph,
    catalog: ChoiceCatalog,
    peaks: pd.DataFrame,
    reference: ReferenceSequence,
    *,
    dna_motifs: Mapping[str, PWM],
    rna_motifs: Mapping[str, PWM],
    factor_mapping: pd.DataFrame,
    dna_window_bp: int,
    rna_window_bp: int,
    dna_minimum_relative_score: float,
    rna_minimum_relative_score: float,
    dna_events_per_choice_cap: int,
    rna_events_per_choice_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, OrderedChoiceAudit]:
    if catalog.choices:
        dna_regions = build_dna_peak_regions(
            graph, catalog, peaks, reference, window_bp=dna_window_bp
        )
        rna_regions = build_rna_choice_regions(
            graph, catalog, reference, window_bp=rna_window_bp
        )
        dna_candidates = (
            scan_motif_regions(
                dna_regions,
                dna_motifs,
                factor_mapping,
                modality="DNA",
                minimum_relative_score=dna_minimum_relative_score,
            )
            if len(dna_regions)
            else _empty_event_table()
        )
        rna_candidates = (
            scan_motif_regions(
                rna_regions,
                rna_motifs,
                factor_mapping,
                modality="RNA",
                minimum_relative_score=rna_minimum_relative_score,
            )
            if len(rna_regions)
            else _empty_event_table()
        )
    else:
        dna_candidates = _empty_event_table()
        rna_candidates = _empty_event_table()
    dna_events, dna_audit = cap_motif_events(
        dna_candidates, events_per_choice_cap=dna_events_per_choice_cap
    )
    rna_events, rna_audit = cap_motif_events(
        rna_candidates, events_per_choice_cap=rna_events_per_choice_cap
    )
    choice_ids = tuple(choice.choice_id for choice in catalog.choices)

    def audit_axis(
        audit: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        by_choice = (
            {}
            if audit.empty
            else audit.assign(choice_id=audit["choice_id"].astype(str))
            .set_index("choice_id")
            .to_dict(orient="index")
        )
        candidate = np.asarray(
            [
                int(by_choice.get(choice_id, {}).get("candidate_event_count", 0))
                for choice_id in choice_ids
            ],
            dtype=np.int64,
        )
        selected = np.asarray(
            [
                int(by_choice.get(choice_id, {}).get("selected_event_count", 0))
                for choice_id in choice_ids
            ],
            dtype=np.int64,
        )
        saturated = np.asarray(
            [
                bool(by_choice.get(choice_id, {}).get("cap_saturated", False))
                for choice_id in choice_ids
            ],
            dtype=bool,
        )
        boundary = np.asarray(
            [
                float(
                    by_choice.get(choice_id, {}).get(
                        "boundary_rank_motif_score", np.nan
                    )
                )
                for choice_id in choice_ids
            ],
            dtype=np.float32,
        )
        return candidate, selected, saturated, boundary

    dna_candidate, dna_selected, dna_saturated, dna_boundary = audit_axis(dna_audit)
    rna_candidate, rna_selected, rna_saturated, rna_boundary = audit_axis(rna_audit)
    node_positions = graph.nodes.set_index(graph.nodes["node_id"].astype(str))[
        "pos_0based"
    ].to_dict()
    return (
        dna_events.reset_index(drop=True),
        rna_events.reset_index(drop=True),
        OrderedChoiceAudit(
            choice_ids=choice_ids,
            alternative_span=np.asarray(
                [
                    abs(
                        int(node_positions[choice.exit_node_id])
                        - int(node_positions[choice.entry_node_id])
                    )
                    for choice in catalog.choices
                ],
                dtype=np.float32,
            ),
            dna_candidate_event_count=dna_candidate,
            dna_selected_event_count=dna_selected,
            dna_cap_saturated=dna_saturated,
            dna_boundary_rank_motif_score=dna_boundary,
            rna_candidate_event_count=rna_candidate,
            rna_selected_event_count=rna_selected,
            rna_cap_saturated=rna_saturated,
            rna_boundary_rank_motif_score=rna_boundary,
        ),
    )


def _prepare_external_gene(
    graph: GeneGraph,
    catalog: ChoiceCatalog,
    ec_rows: pd.DataFrame,
    identifiability: pd.DataFrame,
    *,
    dna_events: pd.DataFrame,
    rna_events: pd.DataFrame,
    choice_audit: OrderedChoiceAudit,
    target_cell_ids: Sequence[str],
    global_cell_index: Mapping[str, int],
    state_values: np.ndarray,
    factor_activity: np.ndarray,
    factor_observed: np.ndarray,
    activity_column_by_factor: Mapping[str, int],
    atac_context: ATACContext,
    global_peak_index: Mapping[str, int],
    factor_order: Sequence[str],
    minimum_valid_mass: float,
    minimum_variance: float,
    dna_distance_scale: float,
    rna_distance_scale: float,
    sources: PreparationSources,
) -> PreparedGene:
    ec_cell_set = set(ec_rows["cell_id"].astype(str)) if len(ec_rows) else set()
    unknown_cells = sorted(ec_cell_set - set(global_cell_index))
    if unknown_cells:
        raise ValueError(
            f"gene {graph.gene_id} EC cells are absent from RNA context: "
            f"{unknown_cells[:10]}"
        )
    cell_ids = tuple(cell_id for cell_id in target_cell_ids if cell_id in ec_cell_set)
    state_rows = np.asarray(
        [global_cell_index[cell_id] for cell_id in cell_ids], dtype=np.int64
    )
    raw_state = np.asarray(state_values[state_rows], dtype=np.float32)
    molecule_mass = _gene_molecule_weights(graph, ec_rows, cell_ids)
    state_observed = np.ones_like(raw_state, dtype=bool)
    state_baseline = fit_centering_baseline(
        raw_state,
        state_observed,
        molecule_mass,
        minimum_valid_mass=minimum_valid_mass,
        minimum_variance=minimum_variance,
    )
    centered_state = apply_centering(raw_state, state_observed, state_baseline)

    all_event_factors = tuple(
        dict.fromkeys(
            [
                *dna_events["factor_id"].astype(str).tolist(),
                *rna_events["factor_id"].astype(str).tolist(),
            ]
        )
    )
    unknown_factors = sorted(set(all_event_factors) - set(activity_column_by_factor))
    if unknown_factors:
        raise ValueError(
            f"events reference factors absent from reviewed mapping: {unknown_factors}"
        )
    gene_factor_activity = np.asarray(factor_activity[state_rows], dtype=np.float32)
    gene_factor_observed = np.asarray(factor_observed[state_rows], dtype=bool)

    def event_activity(
        events: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = [
            activity_column_by_factor[str(factor_id)]
            for factor_id in events["factor_id"].astype(str)
        ]
        return gene_factor_activity[:, columns], gene_factor_observed[:, columns]

    rna_activity, rna_observed = event_activity(rna_events)
    rna_baseline = fit_centering_baseline(
        rna_activity,
        rna_observed,
        molecule_mass,
        minimum_valid_mass=minimum_valid_mass,
        minimum_variance=minimum_variance,
    )
    rna_gate = apply_centering(rna_activity, rna_observed, rna_baseline)

    if len(dna_events):
        event_peak_columns = [
            global_peak_index[str(value)] for value in dna_events["peak_id"]
        ]
        dna_accessibility = atac_context.accessibility[state_rows][
            :, event_peak_columns
        ].toarray()
        dna_accessibility_observed = np.repeat(
            atac_context.observed[state_rows, None], len(dna_events), axis=1
        )
        dna_reliability = np.repeat(
            atac_context.reliability[state_rows, None], len(dna_events), axis=1
        )
    else:
        dna_accessibility = np.zeros((len(cell_ids), 0), dtype=np.float32)
        dna_accessibility_observed = np.zeros_like(dna_accessibility, dtype=bool)
        dna_reliability = np.zeros_like(dna_accessibility, dtype=np.float32)
    dna_activity, dna_factor_observed = event_activity(dna_events)
    dna_baseline = fit_dna_centering_baseline(
        dna_activity,
        dna_factor_observed,
        dna_accessibility,
        dna_accessibility_observed,
        dna_reliability,
        molecule_mass,
        minimum_valid_mass=minimum_valid_mass,
        minimum_variance=minimum_variance,
    )
    dna_gate = apply_dna_centering(
        dna_activity,
        dna_factor_observed,
        dna_accessibility,
        dna_accessibility_observed,
        dna_reliability,
        dna_baseline,
    )

    alternative_ids = tuple(
        alternative.alternative_id
        for choice in catalog.choices
        for alternative in choice.alternatives
    )
    dna_event_ids = tuple(dna_events["event_id"].astype(str))
    rna_event_ids = tuple(rna_events["event_id"].astype(str))
    dna = OrderedEventData(
        events=dna_events,
        feature_event_ids=dna_event_ids,
        features=fixed_event_feature_matrix(
            dna_events,
            modality="DNA",
            factor_order=factor_order,
            region_order=("peak",),
            distance_scale_bp=dna_distance_scale,
        ),
        relation_event_ids=dna_event_ids,
        relation_alternative_ids=alternative_ids,
        relation=event_relation_matrix(dna_events, alternative_ids),
        gate_cell_ids=cell_ids,
        gate_event_ids=dna_event_ids,
        gate=dna_gate,
    )
    rna = OrderedEventData(
        events=rna_events,
        feature_event_ids=rna_event_ids,
        features=fixed_event_feature_matrix(
            rna_events,
            modality="RNA",
            factor_order=factor_order,
            region_order=("exon", "intron"),
            distance_scale_bp=rna_distance_scale,
        ),
        relation_event_ids=rna_event_ids,
        relation_alternative_ids=alternative_ids,
        relation=event_relation_matrix(rna_events, alternative_ids),
        gate_cell_ids=cell_ids,
        gate_event_ids=rna_event_ids,
        gate=rna_gate,
    )
    prepared = prepare_gene(
        graph,
        catalog,
        ec_rows,
        state=OrderedCellState(cell_ids=cell_ids, values=centered_state),
        dna=dna,
        rna=rna,
        choice_identifiability=identifiability,
        choice_audit=choice_audit,
        sources=sources,
    )
    from .train import PreparedGateBaseline

    def prepared_baseline(
        baseline: CenteringBaseline | DNACenteringBaseline,
    ) -> PreparedGateBaseline:
        return PreparedGateBaseline(
            mean=torch.from_numpy(np.asarray(baseline.mean, dtype=np.float32)),
            valid_molecule_mass=torch.from_numpy(
                np.asarray(baseline.valid_molecule_mass, dtype=np.float64)
            ),
            weighted_variance=torch.from_numpy(
                np.asarray(baseline.weighted_variance, dtype=np.float64)
            ),
            eligible=torch.from_numpy(np.asarray(baseline.eligible, dtype=bool)),
            dna_reliability_mass=(
                None
                if not isinstance(baseline, DNACenteringBaseline)
                else torch.from_numpy(
                    np.asarray(baseline.dna_reliability_mass, dtype=np.float64)
                )
            ),
        )

    return replace(
        prepared,
        dna_event_factor_ids=tuple(dna_events["factor_id"].astype(str)),
        rna_event_factor_ids=tuple(rna_events["factor_id"].astype(str)),
        dna_event_peak_ids=tuple(dna_events["peak_id"].astype(str)),
        state_baseline=prepared_baseline(state_baseline),
        dna_baseline=prepared_baseline(dna_baseline),
        rna_baseline=prepared_baseline(rna_baseline),
    )


def _gene_molecule_weights(
    graph: GeneGraph,
    ec_rows: pd.DataFrame,
    cell_ids: Sequence[str],
) -> np.ndarray:
    if ec_rows.empty:
        return np.zeros(len(cell_ids), dtype=np.float64)
    informative = ec_rows["compatible_path_count"].to_numpy(dtype=np.int64) < len(
        graph.path_ids
    )
    mass = cell_gene_molecule_mass(ec_rows, informative_row_mask=informative)
    by_cell = mass.set_index("cell_id")["molecule_mass"].to_dict()
    return np.asarray([float(by_cell.get(cell_id, 0.0)) for cell_id in cell_ids])


def load_full_rna_glue_context(
    glue_h5ad: str | Path,
    peak_h5ad: str | Path,
    *,
    target_rna_cell_ids: Sequence[str],
    expected_rna_count: int = 205_864,
    expected_atac_count: int = 232_474,
    expected_embedding_dim: int = 50,
) -> FullRNAGLUEContext:
    """Load only X_glue and validated metadata from the frozen full-RNA axes.

    ``stage_scanvi`` is the single stage label present for both modalities.
    ATAC donor/system identity is joined from the current peak H5AD because the
    combined GLUE container deliberately stores ``Unknown`` for ATAC system.
    """

    target_ids = tuple(canonical_rna_cell_id(value) for value in target_rna_cell_ids)
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("target RNA GLUE cell IDs must be non-empty and unique")
    glue = ad.read_h5ad(glue_h5ad, backed="r")
    try:
        if glue.shape != (expected_rna_count + expected_atac_count, 0):
            raise ValueError(f"full-RNA GLUE shape is unexpected: {glue.shape}")
        if "X_glue" not in glue.obsm:
            raise ValueError("full-RNA GLUE container has no X_glue")
        modality = glue.obs["modality"].astype(str).to_numpy()
        rna_rows = np.flatnonzero(modality == "RNA")
        atac_rows = np.flatnonzero(modality == "ATAC")
        if len(rna_rows) != expected_rna_count or len(atac_rows) != expected_atac_count:
            raise ValueError(
                "full-RNA GLUE modality counts differ from the frozen generation"
            )
        rna_axis = [canonical_rna_cell_id(glue.obs_names[index]) for index in rna_rows]
        if len(set(rna_axis)) != len(rna_axis):
            raise ValueError("full-RNA GLUE RNA axis is not unique")
        rna_index = {value: index for index, value in enumerate(rna_axis)}
        missing = [value for value in target_ids if value not in rna_index]
        if missing:
            raise ValueError(
                f"target RNA cells are absent from full-RNA GLUE: {missing[:10]}"
            )
        local_rna_rows = np.asarray(
            [rna_rows[rna_index[value]] for value in target_ids]
        )
        embedding = np.asarray(glue.obsm["X_glue"], dtype=np.float32)
        if (
            embedding.shape
            != (
                expected_rna_count + expected_atac_count,
                expected_embedding_dim,
            )
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("full-RNA X_glue shape or finite-value contract failed")
        stage = glue.obs["stage_scanvi"].astype(str).to_numpy()
        developmental_system = glue.obs["developmental_system"].astype(str).to_numpy()
        _require_known_stage(stage[local_rna_rows], "target RNA")
        _require_known_stage(stage[atac_rows], "ATAC")
        if any(
            value in {"", "nan", "Unknown"}
            for value in developmental_system[local_rna_rows]
        ):
            raise ValueError(
                "RNA developmental-system identity is required for the fixed null"
            )
        rna_embedding = embedding[local_rna_rows].copy()
        atac_embedding = embedding[atac_rows].copy()
        rna_stage = tuple(stage[local_rna_rows])
        rna_developmental_system = tuple(developmental_system[local_rna_rows])
        atac_stage = tuple(stage[atac_rows])
        atac_ids = tuple(
            canonical_atac_cell_id(glue.obs_names[index]) for index in atac_rows
        )
    finally:
        glue.file.close()

    peaks = ad.read_h5ad(peak_h5ad, backed="r")
    try:
        if peaks.n_obs != expected_atac_count:
            raise ValueError(
                "current peak matrix ATAC count differs from full-RNA GLUE"
            )
        peak_atac_ids = tuple(
            canonical_atac_cell_id(value) for value in peaks.obs_names
        )
        if peak_atac_ids != atac_ids:
            raise ValueError("full-RNA GLUE and current peak-matrix ATAC axes differ")
        donor_ids = tuple(peaks.obs["sample_id"].astype(str))
        systems = tuple(peaks.obs["developmental_system"].astype(str))
        if any(value in {"", "nan", "Unknown"} for value in donor_ids):
            raise ValueError(
                "ATAC donor identity is missing in the current peak matrix"
            )
        if any(value in {"", "nan", "Unknown"} for value in systems):
            raise ValueError(
                "ATAC developmental system is missing in the current peak matrix"
            )
    finally:
        peaks.file.close()
    return FullRNAGLUEContext(
        rna_cell_ids=target_ids,
        rna_embedding=rna_embedding,
        rna_stage=rna_stage,
        rna_developmental_system=rna_developmental_system,
        atac_cell_ids=atac_ids,
        atac_embedding=atac_embedding,
        atac_stage=atac_stage,
        atac_donor_ids=donor_ids,
        atac_developmental_system=systems,
    )


def normalize_log1p_counts(
    counts: sparse.spmatrix, *, target_sum: float
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Library-normalize non-negative counts and apply log1p without densifying."""

    if target_sum <= 0:
        raise ValueError("normalization target_sum must be positive")
    matrix = sparse.csr_matrix(counts, dtype=np.float32)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or bool((matrix.data < 0).any())
    ):
        raise ValueError("count matrix must contain finite non-negative values")
    library_size = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float64)
    if bool((library_size <= 0).any()):
        raise ValueError("observed cells require a positive library size")
    scales = target_sum / library_size
    normalized = sparse.diags(scales.astype(np.float32)) @ matrix
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized, library_size


def iter_normalized_h5ad_rows(
    path: str | Path,
    row_indices: Sequence[int],
    *,
    target_sum: float,
    batch_size: int,
    column_indices: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, sparse.csr_matrix, np.ndarray]]:
    """Yield sparse normalized chunks while always using the full library denominator."""

    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("row_indices must be a non-empty unique vector")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    columns = (
        None if column_indices is None else np.asarray(column_indices, dtype=np.int64)
    )
    matrix = ad.read_h5ad(path, backed="r")
    try:
        if bool(((rows < 0) | (rows >= matrix.n_obs)).any()):
            raise IndexError("H5AD row index is out of range")
        if columns is not None and bool(
            ((columns < 0) | (columns >= matrix.n_vars)).any()
        ):
            raise IndexError("H5AD column index is out of range")
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            raw = sparse.csr_matrix(matrix.X[batch_rows, :])
            normalized, library = normalize_log1p_counts(raw, target_sum=target_sum)
            if columns is not None:
                normalized = normalized[:, columns].tocsr()
            yield batch_rows, normalized, library
    finally:
        matrix.file.close()


def fit_rna_state_pca(
    rna_h5ad: str | Path,
    train_row_indices: Sequence[int],
    *,
    n_components: int,
    target_sum: float,
    batch_size: int,
) -> RNAStatePCA:
    """Fit the RNA-only PCA and library-size standardization on train cells only."""

    train_rows = np.asarray(train_row_indices, dtype=np.int64)
    if n_components <= 0 or len(train_rows) <= n_components:
        raise ValueError("PCA requires more train cells than components")
    batches = _pca_batches(train_rows, batch_size=batch_size, minimum=n_components)
    pca = IncrementalPCA(n_components=n_components)
    log_libraries: list[np.ndarray] = []
    for batch_rows in batches:
        ((_, normalized, library),) = iter_normalized_h5ad_rows(
            rna_h5ad,
            batch_rows,
            target_sum=target_sum,
            batch_size=len(batch_rows),
        )
        pca.partial_fit(normalized.toarray())
        log_libraries.append(np.log1p(library))
    log_library = np.concatenate(log_libraries)
    scale = float(log_library.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("train RNA library-size covariate has no variation")
    return RNAStatePCA(
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        log_library_mean=float(log_library.mean()),
        log_library_scale=scale,
        target_sum=float(target_sum),
        fit_batch_size=int(batch_size),
    )


def transform_rna_state(
    rna_h5ad: str | Path,
    row_indices: Sequence[int],
    fit: RNAStatePCA,
    *,
    batch_size: int,
) -> np.ndarray:
    """Apply frozen train PCA and append standardized log1p library size."""

    parts: list[np.ndarray] = []
    for _, normalized, library in iter_normalized_h5ad_rows(
        rna_h5ad,
        row_indices,
        target_sum=fit.target_sum,
        batch_size=batch_size,
    ):
        centered = normalized.toarray() - fit.mean[None, :]
        scores = centered @ fit.components.T
        log_library = (
            (np.log1p(library) - fit.log_library_mean) / fit.log_library_scale
        )[:, None]
        parts.append(np.concatenate([scores, log_library], axis=1).astype(np.float32))
    return np.concatenate(parts, axis=0)


def read_factor_activity(
    rna_h5ad: str | Path,
    row_indices: Sequence[int],
    activity_gene_ids: Sequence[str],
    *,
    target_sum: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read factor genes in the same non-negative normalized-log1p space."""

    matrix = ad.read_h5ad(rna_h5ad, backed="r")
    try:
        canonical_genes = [str(value).split(".", 1)[0] for value in matrix.var_names]
        if len(set(canonical_genes)) != len(canonical_genes):
            raise ValueError(
                "RNA gene axis is not unique after removing Ensembl versions"
            )
        gene_index = {value: index for index, value in enumerate(canonical_genes)}
    finally:
        matrix.file.close()
    requested = [str(value).split(".", 1)[0] for value in activity_gene_ids]
    present_positions = [
        index for index, gene in enumerate(requested) if gene in gene_index
    ]
    present_columns = [gene_index[requested[index]] for index in present_positions]
    values = np.zeros((len(row_indices), len(requested)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    if not present_columns:
        return values, observed
    offset = 0
    for _, normalized, _ in iter_normalized_h5ad_rows(
        rna_h5ad,
        row_indices,
        target_sum=target_sum,
        batch_size=batch_size,
        column_indices=present_columns,
    ):
        dense = normalized.toarray()
        stop = offset + dense.shape[0]
        values[offset:stop, present_positions] = dense
        observed[offset:stop, present_positions] = True
        offset = stop
    return values, observed


def build_exact_stage_neighbors(
    *,
    rna_cell_ids: Sequence[str],
    rna_embedding: np.ndarray,
    rna_stage: Sequence[str],
    atac_cell_ids: Sequence[str],
    atac_embedding: np.ndarray,
    atac_stage: Sequence[str],
    atac_donor_ids: Sequence[str],
    atac_donor_eligible: Sequence[bool],
    k: int,
    temperature: float,
    device: str,
    query_chunk_size: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direct exact-stage RNA→ATAC KNN in the frozen full-RNA GLUE space."""

    if k <= 0 or temperature <= 0 or query_chunk_size <= 0:
        raise ValueError("KNN k, temperature, and chunk size must be positive")
    rna_ids = np.asarray([canonical_rna_cell_id(value) for value in rna_cell_ids])
    atac_ids = np.asarray([canonical_atac_cell_id(value) for value in atac_cell_ids])
    if len(set(rna_ids)) != len(rna_ids) or len(set(atac_ids)) != len(atac_ids):
        raise ValueError("RNA and ATAC neighbor axes must each be unique")
    rna_values = np.asarray(rna_embedding, dtype=np.float32)
    atac_values = np.asarray(atac_embedding, dtype=np.float32)
    if rna_values.ndim != 2 or atac_values.ndim != 2:
        raise ValueError("RNA and ATAC GLUE embeddings must be rank two")
    if rna_values.shape != (len(rna_ids), atac_values.shape[1]):
        raise ValueError("RNA GLUE shape differs from cell axis or ATAC dimension")
    if atac_values.shape[0] != len(atac_ids):
        raise ValueError("ATAC GLUE shape differs from cell axis")
    if not np.isfinite(rna_values).all() or not np.isfinite(atac_values).all():
        raise ValueError("GLUE embeddings must be finite")
    rna_stage_values = np.asarray(rna_stage, dtype=str)
    atac_stage_values = np.asarray(atac_stage, dtype=str)
    donor_ids = np.asarray(atac_donor_ids, dtype=str)
    donor_eligible = np.asarray(atac_donor_eligible, dtype=bool)
    if len(rna_stage_values) != len(rna_ids) or len(atac_stage_values) != len(atac_ids):
        raise ValueError("stage axes must match their modality cell axes")
    if len(donor_ids) != len(atac_ids) or len(donor_eligible) != len(atac_ids):
        raise ValueError("ATAC donor identity/eligibility axes must match ATAC cells")
    if any(value in {"", "nan", "Unknown"} for value in donor_ids):
        raise ValueError("ATAC donor identity must be explicit before KNN")
    _require_known_stage(rna_stage_values, "RNA")
    _require_known_stage(atac_stage_values, "ATAC")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA is unavailable for neighbor device {device}")

    neighbor_parts: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for stage in sorted(set(rna_stage_values)):
        query_indices = np.flatnonzero(rna_stage_values == stage)
        reference_indices = np.flatnonzero(
            (atac_stage_values == stage) & donor_eligible
        )
        if len(reference_indices) == 0:
            status_rows.extend(
                {
                    "cell_id": rna_ids[index],
                    "stage": stage,
                    "observed_atac": False,
                    "neighbor_count": 0,
                }
                for index in query_indices
            )
            continue
        selected_k = min(k, len(reference_indices))
        reference = torch.from_numpy(atac_values[reference_indices]).to(torch_device)
        reference_ids = atac_ids[reference_indices]
        for start in range(0, len(query_indices), query_chunk_size):
            local_query = query_indices[start : start + query_chunk_size]
            query = torch.from_numpy(rna_values[local_query]).to(torch_device)
            with torch.inference_mode():
                distances = torch.cdist(query, reference, p=2.0)
                distance_np = distances.detach().cpu().numpy()
            for local_row, rna_index in enumerate(local_query):
                order = np.lexsort((reference_ids, distance_np[local_row]))[:selected_k]
                chosen_distances = distance_np[local_row, order].astype(np.float64)
                scaled = -chosen_distances / temperature
                weights = np.exp(scaled - scaled.max())
                weights /= weights.sum()
                neighbor_parts.append(
                    pd.DataFrame(
                        {
                            "cell_id": rna_ids[rna_index],
                            "rna_stage": stage,
                            "neighbor_atac_cell_id": reference_ids[order],
                            "neighbor_atac_donor_id": donor_ids[reference_indices][
                                order
                            ],
                            "atac_stage": stage,
                            "neighbor_rank": np.arange(
                                1, selected_k + 1, dtype=np.int16
                            ),
                            "glue_distance": chosen_distances.astype(np.float32),
                            "neighbor_weight": weights.astype(np.float32),
                        }
                    )
                )
                status_rows.append(
                    {
                        "cell_id": rna_ids[rna_index],
                        "stage": stage,
                        "observed_atac": True,
                        "neighbor_count": selected_k,
                    }
                )
            del distances, query
        del reference
    neighbors = (
        pd.concat(neighbor_parts, ignore_index=True)
        if neighbor_parts
        else pd.DataFrame(
            columns=[
                "cell_id",
                "rna_stage",
                "neighbor_atac_cell_id",
                "neighbor_atac_donor_id",
                "atac_stage",
                "neighbor_rank",
                "glue_distance",
                "neighbor_weight",
            ]
        )
    )
    status = (
        pd.DataFrame(status_rows)
        .sort_values("cell_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(neighbors):
        sums = neighbors.groupby("cell_id", sort=False)["neighbor_weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-6):
            raise RuntimeError("neighbor weights do not sum to one")
    return neighbors, status


def aggregate_atac_accessibility(
    peak_h5ad: str | Path,
    neighbors: pd.DataFrame,
    *,
    target_cell_ids: Sequence[str],
    peak_ids: Sequence[str],
    target_sum: float,
    expected_k: int,
    batch_size: int,
) -> ATACContext:
    """Aggregate only selected event peaks after per-ATAC-cell normalization."""

    target_ids = tuple(canonical_rna_cell_id(value) for value in target_cell_ids)
    selected_peaks = tuple(str(value) for value in peak_ids)
    if len(set(target_ids)) != len(target_ids) or len(set(selected_peaks)) != len(
        selected_peaks
    ):
        raise ValueError("target cell and event peak axes must be unique")
    if expected_k <= 0 or batch_size <= 0:
        raise ValueError("expected_k and batch_size must be positive")
    required_neighbor_columns = {
        "cell_id",
        "neighbor_atac_cell_id",
        "neighbor_rank",
        "neighbor_weight",
    }
    if required_neighbor_columns - set(neighbors):
        raise ValueError("ATAC neighbor table misses required columns")
    if neighbors.empty:
        return ATACContext(
            accessibility=sparse.csr_matrix(
                (len(target_ids), len(selected_peaks)), dtype=np.float32
            ),
            observed=np.zeros(len(target_ids), dtype=bool),
            reliability=np.zeros(len(target_ids), dtype=np.float32),
            cell_ids=target_ids,
            peak_ids=selected_peaks,
        )
    neighbor_rows = neighbors.copy()
    neighbor_rows["cell_id"] = (
        neighbor_rows["cell_id"].astype(str).map(canonical_rna_cell_id)
    )
    neighbor_rows["neighbor_atac_cell_id"] = (
        neighbor_rows["neighbor_atac_cell_id"].astype(str).map(canonical_atac_cell_id)
    )
    extra_targets = sorted(set(neighbor_rows["cell_id"]) - set(target_ids))
    if extra_targets:
        raise ValueError(
            f"neighbor rows contain non-target RNA cells: {extra_targets[:10]}"
        )
    if (
        neighbor_rows.duplicated(["cell_id", "neighbor_rank"]).any()
        or neighbor_rows.duplicated(["cell_id", "neighbor_atac_cell_id"]).any()
    ):
        raise ValueError("ATAC neighbors must be unique within each target cell")
    weights = neighbor_rows["neighbor_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("ATAC neighbor weights must be finite and positive")
    group_sizes = neighbor_rows.groupby("cell_id", sort=False).size()
    if bool((group_sizes > expected_k).any()):
        raise ValueError("ATAC neighbor count exceeds expected_k")
    weight_sums = neighbor_rows.groupby("cell_id", sort=False)["neighbor_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-6):
        raise ValueError("ATAC neighbor weights do not sum to one per target cell")

    matrix = ad.read_h5ad(peak_h5ad, backed="r")
    try:
        atac_axis = [canonical_atac_cell_id(value) for value in matrix.obs_names]
        peak_axis = [str(value) for value in matrix.var_names]
    finally:
        matrix.file.close()
    atac_index = {value: index for index, value in enumerate(atac_axis)}
    peak_index = {value: index for index, value in enumerate(peak_axis)}
    unknown_neighbors = sorted(
        set(neighbor_rows["neighbor_atac_cell_id"]) - set(atac_index)
    )
    unknown_peaks = sorted(set(selected_peaks) - set(peak_index))
    if unknown_neighbors:
        raise ValueError(
            f"ATAC neighbors are absent from peak matrix: {unknown_neighbors[:10]}"
        )
    if unknown_peaks:
        raise ValueError(
            f"event peaks are absent from peak matrix: {unknown_peaks[:10]}"
        )
    donor_ids = sorted(
        set(neighbor_rows["neighbor_atac_cell_id"]),
        key=atac_index.__getitem__,
    )
    donor_rows = [atac_index[value] for value in donor_ids]
    selected_columns = [peak_index[value] for value in selected_peaks]
    donor_parts = [
        normalized
        for _, normalized, _ in iter_normalized_h5ad_rows(
            peak_h5ad,
            donor_rows,
            target_sum=target_sum,
            batch_size=batch_size,
            column_indices=selected_columns,
        )
    ]
    donor_values = sparse.vstack(donor_parts, format="csr")
    target_index = {value: index for index, value in enumerate(target_ids)}
    donor_index = {value: index for index, value in enumerate(donor_ids)}
    weight_rows: list[int] = []
    weight_cols: list[int] = []
    weight_values: list[float] = []
    for row in neighbor_rows.itertuples(index=False):
        cell_id = str(row.cell_id)
        weight_rows.append(target_index[cell_id])
        weight_cols.append(donor_index[str(row.neighbor_atac_cell_id)])
        weight_values.append(float(row.neighbor_weight))
    weight_matrix = sparse.csr_matrix(
        (weight_values, (weight_rows, weight_cols)),
        shape=(len(target_ids), len(donor_ids)),
    )
    accessibility = (weight_matrix @ donor_values).tocsr()
    observed = np.asarray(weight_matrix.getnnz(axis=1) > 0, dtype=bool)
    reliability = np.zeros(len(target_ids), dtype=np.float32)
    for target, target_row in target_index.items():
        weights = weight_matrix.getrow(target_row).data.astype(np.float64)
        if len(weights):
            effective_fraction = 1.0 / (len(weights) * float(np.square(weights).sum()))
            coverage_fraction = min(len(weights) / expected_k, 1.0)
            reliability[target_row] = float(effective_fraction * coverage_fraction)
    return ATACContext(
        accessibility=accessibility,
        observed=observed,
        reliability=reliability,
        cell_ids=target_ids,
        peak_ids=selected_peaks,
    )


def cell_gene_molecule_mass(
    ec_rows: pd.DataFrame, *, informative_row_mask: Sequence[bool]
) -> pd.DataFrame:
    """Compute each train cell-gene mass once, independent of EC row splitting."""

    informative = np.asarray(informative_row_mask, dtype=bool)
    if len(informative) != len(ec_rows):
        raise ValueError("informative_row_mask must match EC rows")
    selected = ec_rows.loc[
        informative & (ec_rows["split"].astype(str).to_numpy() == "train"),
        ["cell_id", "gene_id", "molecule_count"],
    ].copy()
    selected["cell_id"] = selected["cell_id"].astype(str).map(canonical_rna_cell_id)
    result = (
        selected.groupby(["cell_id", "gene_id"], sort=True, as_index=False)[
            "molecule_count"
        ]
        .sum()
        .rename(columns={"molecule_count": "molecule_mass"})
    )
    return result


def fit_centering_baseline(
    values: np.ndarray,
    observed: np.ndarray,
    molecule_mass: np.ndarray,
    *,
    minimum_valid_mass: float,
    minimum_variance: float,
) -> CenteringBaseline:
    """Fit a train-only State or RNA baseline along the cell axis."""

    value_array = np.asarray(values, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=bool)
    if value_array.ndim == 1:
        value_array = value_array[:, None]
    if observed_array.ndim == 1:
        observed_array = observed_array[:, None]
    if value_array.shape != observed_array.shape:
        raise ValueError("values and observed mask must have the same shape")
    weights = np.asarray(molecule_mass, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != value_array.shape[0]:
        raise ValueError("molecule_mass must have one value per cell")
    if bool((weights < 0).any()) or not np.isfinite(weights).all():
        raise ValueError("molecule_mass must be finite and non-negative")
    if not np.isfinite(value_array[observed_array]).all():
        raise ValueError("observed values must be finite")
    effective = observed_array * weights[:, None]
    valid_mass = effective.sum(axis=0)
    covered = valid_mass > 0
    mean = np.full(value_array.shape[1], np.nan, dtype=np.float64)
    variance = np.full(value_array.shape[1], np.nan, dtype=np.float64)
    mean[covered] = (effective[:, covered] * value_array[:, covered]).sum(
        axis=0
    ) / valid_mass[covered]
    variance[covered] = (
        effective[:, covered] * np.square(value_array[:, covered] - mean[None, covered])
    ).sum(axis=0) / valid_mass[covered]
    eligible = (
        covered & (valid_mass >= minimum_valid_mass) & (variance >= minimum_variance)
    )
    return CenteringBaseline(
        mean=mean.astype(np.float32),
        valid_molecule_mass=valid_mass.astype(np.float64),
        weighted_variance=variance.astype(np.float64),
        eligible=eligible,
    )


def apply_centering(
    values: np.ndarray, observed: np.ndarray, baseline: CenteringBaseline
) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float32)
    observed_array = np.asarray(observed, dtype=bool)
    if value_array.ndim == 1:
        value_array = value_array[:, None]
    if observed_array.ndim == 1:
        observed_array = observed_array[:, None]
    if value_array.shape != observed_array.shape or value_array.shape[1] != len(
        baseline.mean
    ):
        raise ValueError("centering inputs differ from fitted key axis")
    result = np.zeros_like(value_array, dtype=np.float32)
    active = np.asarray(baseline.eligible, dtype=bool)
    result[:, active] = observed_array[:, active] * (
        value_array[:, active] - baseline.mean[None, active]
    )
    return result


def fit_dna_centering_baseline(
    factor_activity: np.ndarray,
    factor_observed: np.ndarray,
    accessibility: np.ndarray,
    accessibility_observed: np.ndarray,
    reliability: np.ndarray,
    molecule_mass: np.ndarray,
    *,
    minimum_valid_mass: float,
    minimum_variance: float,
) -> DNACenteringBaseline:
    """Fit the raw non-negative product baseline before any subtraction."""

    factor = np.asarray(factor_activity, dtype=np.float64)
    access = np.asarray(accessibility, dtype=np.float64)
    factor_mask = np.asarray(factor_observed, dtype=bool)
    access_mask = np.asarray(accessibility_observed, dtype=bool)
    reliability_values = np.asarray(reliability, dtype=np.float64)
    weights = np.asarray(molecule_mass, dtype=np.float64)
    if not (
        factor.shape
        == access.shape
        == factor_mask.shape
        == access_mask.shape
        == reliability_values.shape
    ):
        raise ValueError("DNA gate arrays must have the same [cell, event-key] shape")
    if weights.ndim != 1 or len(weights) != factor.shape[0]:
        raise ValueError("molecule_mass must have one value per cell")
    observed = factor_mask & access_mask
    if bool((factor[observed] < 0).any()) or bool((access[observed] < 0).any()):
        raise ValueError("DNA factor and accessibility gates must be non-negative")
    if bool(((reliability_values < 0) | (reliability_values > 1)).any()):
        raise ValueError("DNA reliability must lie in [0, 1]")
    raw_product = factor * access
    valid_weight = observed * weights[:, None]
    reliability_weight = valid_weight * reliability_values
    valid_mass = valid_weight.sum(axis=0)
    reliability_mass = reliability_weight.sum(axis=0)
    covered = reliability_mass > 0
    mean = np.full(factor.shape[1], np.nan, dtype=np.float64)
    variance = np.full(factor.shape[1], np.nan, dtype=np.float64)
    mean[covered] = (reliability_weight[:, covered] * raw_product[:, covered]).sum(
        axis=0
    ) / reliability_mass[covered]
    variance[covered] = (
        reliability_weight[:, covered]
        * np.square(raw_product[:, covered] - mean[None, covered])
    ).sum(axis=0) / reliability_mass[covered]
    eligible = (
        covered
        & (valid_mass >= minimum_valid_mass)
        & (variance >= minimum_variance)
        & (reliability_mass > 0)
    )
    return DNACenteringBaseline(
        mean=mean.astype(np.float32),
        valid_molecule_mass=valid_mass,
        weighted_variance=variance,
        dna_reliability_mass=reliability_mass,
        eligible=eligible,
    )


def apply_dna_centering(
    factor_activity: np.ndarray,
    factor_observed: np.ndarray,
    accessibility: np.ndarray,
    accessibility_observed: np.ndarray,
    reliability: np.ndarray,
    baseline: DNACenteringBaseline,
) -> np.ndarray:
    factor = np.asarray(factor_activity, dtype=np.float32)
    access = np.asarray(accessibility, dtype=np.float32)
    factor_mask = np.asarray(factor_observed, dtype=bool)
    access_mask = np.asarray(accessibility_observed, dtype=bool)
    reliability_values = np.asarray(reliability, dtype=np.float32)
    if factor.shape[1] != len(baseline.mean):
        raise ValueError("DNA gate input differs from fitted key axis")
    observed = factor_mask & access_mask
    result = np.zeros_like(factor, dtype=np.float32)
    active = np.asarray(baseline.eligible, dtype=bool)
    result[:, active] = (
        observed[:, active]
        * reliability_values[:, active]
        * (factor[:, active] * access[:, active] - baseline.mean[None, active])
    )
    return result


def as_torch_sparse(
    matrix: sparse.spmatrix, *, device: torch.device | str
) -> torch.Tensor:
    coo = sparse.coo_matrix(matrix)
    indices = torch.tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, device=device
    ).coalesce()


def prepare_gene(
    graph: GeneGraph,
    catalog: ChoiceCatalog,
    ec_rows: pd.DataFrame,
    *,
    state: OrderedCellState,
    dna: OrderedEventData,
    rna: OrderedEventData,
    choice_identifiability: pd.DataFrame,
    choice_audit: OrderedChoiceAudit,
    sources: PreparationSources,
) -> PreparedGene:
    """Convert one validated F1/F2 gene into the exact training tensor contract.

    Every numeric axis is accompanied by IDs and must already follow its
    declared authority.  This boundary does not intersect, reorder, or fill
    scientific objects to make mismatched inputs fit.
    """

    from .model import GraphGPSBatch, alternative_batch_from_catalog
    from .train import PreparedGene

    _validate_graph_choice_identity(graph, catalog)
    if not sources.graph_generation or not sources.split_source:
        raise ValueError("graph_generation and split_source must be explicit")

    cell_ids = _ordered_unique_ids(
        "State cell", state.cell_ids, allow_empty=ec_rows.empty
    )
    if tuple(canonical_rna_cell_id(value) for value in cell_ids) != cell_ids:
        raise ValueError("State cell IDs must already use canonical RNA identities")
    state_values = _finite_matrix("State values", state.values)
    if state_values.shape[0] != len(cell_ids):
        raise ValueError("State value rows and State cell IDs differ")

    compatible_indices, compatible_padded, compatible_mask, row_cell_index = (
        _prepare_compatibility_rows(ec_rows, graph, cell_ids)
    )
    eligibility = _prepare_choice_eligibility(catalog, choice_identifiability)
    choice_ids = tuple(choice.choice_id for choice in catalog.choices)
    _require_exact_axis("choice audit", choice_audit.choice_ids, choice_ids)
    alternative_span = np.asarray(choice_audit.alternative_span, dtype=np.float32)
    audit_arrays = {
        "DNA candidate event count": np.asarray(
            choice_audit.dna_candidate_event_count, dtype=np.int64
        ),
        "DNA selected event count": np.asarray(
            choice_audit.dna_selected_event_count, dtype=np.int64
        ),
        "DNA cap saturation": np.asarray(choice_audit.dna_cap_saturated, dtype=bool),
        "DNA boundary motif score": np.asarray(
            choice_audit.dna_boundary_rank_motif_score, dtype=np.float32
        ),
        "RNA candidate event count": np.asarray(
            choice_audit.rna_candidate_event_count, dtype=np.int64
        ),
        "RNA selected event count": np.asarray(
            choice_audit.rna_selected_event_count, dtype=np.int64
        ),
        "RNA cap saturation": np.asarray(choice_audit.rna_cap_saturated, dtype=bool),
        "RNA boundary motif score": np.asarray(
            choice_audit.rna_boundary_rank_motif_score, dtype=np.float32
        ),
    }
    if alternative_span.shape != (len(choice_ids),) or any(
        values.shape != (len(choice_ids),) for values in audit_arrays.values()
    ):
        raise ValueError("choice audit covariates must have one value per choice")
    if not np.isfinite(alternative_span).all() or bool((alternative_span < 0).any()):
        raise ValueError("choice alternative span must be finite and non-negative")
    for modality in ("DNA", "RNA"):
        candidate = audit_arrays[f"{modality} candidate event count"]
        selected = audit_arrays[f"{modality} selected event count"]
        saturated = audit_arrays[f"{modality} cap saturation"]
        boundary = audit_arrays[f"{modality} boundary motif score"]
        if (
            bool((candidate < 0).any())
            or bool((selected < 0).any())
            or bool((selected > candidate).any())
        ):
            raise ValueError(f"{modality} candidate/selected event counts are invalid")
        if not np.array_equal(saturated, candidate > selected):
            raise ValueError(f"{modality} cap saturation differs from event counts")
        if bool((selected > 0).any()) and not np.isfinite(boundary[selected > 0]).all():
            raise ValueError(
                f"{modality} selected choices require a boundary motif score"
            )
        if bool(np.isfinite(boundary[selected == 0]).any()):
            raise ValueError(
                f"{modality} empty choices cannot have a boundary motif score"
            )
    identifiable_rows = _identifiable_ec_rows(catalog, compatible_indices, eligibility)

    alternative_ids = tuple(
        alternative.alternative_id
        for choice in catalog.choices
        for alternative in choice.alternatives
    )
    dna_tensors = _prepare_event_tensors(
        "DNA", dna, graph.gene_id, catalog, alternative_ids, cell_ids
    )
    rna_tensors = _prepare_event_tensors(
        "RNA", rna, graph.gene_id, catalog, alternative_ids, cell_ids
    )
    if set(dna_tensors[4]) & set(rna_tensors[4]):
        raise ValueError("DNA and RNA event IDs must be modality-specific")
    for modality, event_choice in (("DNA", dna_tensors[2]), ("RNA", rna_tensors[2])):
        observed_count = np.bincount(
            event_choice.numpy(), minlength=len(choice_ids)
        ).astype(np.int64)
        if not np.array_equal(
            observed_count, audit_arrays[f"{modality} selected event count"]
        ):
            raise ValueError(
                f"{modality} selected-event audit differs from event-choice rows"
            )
    molecule_mass = _gene_molecule_weights(graph, ec_rows, cell_ids)
    _require_train_centered("State", state_values, molecule_mass)
    _require_train_centered("DNA gate", dna_tensors[3].numpy(), molecule_mass)
    _require_train_centered("RNA gate", rna_tensors[3].numpy(), molecule_mass)

    alternatives = alternative_batch_from_catalog(catalog, device="cpu")
    alternative_eligible = torch.tensor(
        [
            bool(eligibility[choice_index])
            for choice_index, choice in enumerate(catalog.choices)
            for _ in choice.alternatives
        ],
        dtype=torch.bool,
    )
    splits = tuple(ec_rows["split"].astype(str))
    molecule_count = torch.tensor(
        ec_rows["molecule_count"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    return PreparedGene(
        gene_id=graph.gene_id,
        graph=GraphGPSBatch(
            edge_features=torch.tensor(graph.edge_features, dtype=torch.float32),
            local_edge_index=torch.tensor(graph.local_edge_index, dtype=torch.long),
            edge_gene_index=torch.zeros(len(graph.edge_ids), dtype=torch.long),
        ),
        alternatives=alternatives,
        path_edge_incidence=as_torch_sparse(graph.path_edge_incidence, device="cpu"),
        path_choice_incidence=as_torch_sparse(
            catalog.path_choice_incidence, device="cpu"
        ),
        alternative_eligible=alternative_eligible,
        state_features=torch.tensor(state_values, dtype=torch.float32),
        dna_event_features=dna_tensors[0],
        dna_event_relation=dna_tensors[1],
        dna_event_choice_index=dna_tensors[2],
        dna_gate=dna_tensors[3],
        rna_event_features=rna_tensors[0],
        rna_event_relation=rna_tensors[1],
        rna_event_choice_index=rna_tensors[2],
        rna_gate=rna_tensors[3],
        compatible_path_indices=compatible_padded,
        compatible_path_mask=compatible_mask,
        row_cell_index=row_cell_index,
        molecule_count=molecule_count,
        split=splits,
        identifiable_row_mask=torch.tensor(identifiable_rows, dtype=torch.bool),
        cell_ids=cell_ids,
        path_ids=graph.path_ids,
        dna_event_ids=dna_tensors[4],
        rna_event_ids=rna_tensors[4],
        graph_generation=sources.graph_generation,
        split_source=sources.split_source,
        alternative_span=torch.tensor(alternative_span, dtype=torch.float32),
        dna_candidate_event_count=torch.tensor(
            audit_arrays["DNA candidate event count"], dtype=torch.float32
        ),
        dna_selected_event_count=torch.tensor(
            audit_arrays["DNA selected event count"], dtype=torch.float32
        ),
        dna_cap_saturated=torch.tensor(
            audit_arrays["DNA cap saturation"], dtype=torch.float32
        ),
        dna_boundary_rank_motif_score=torch.tensor(
            audit_arrays["DNA boundary motif score"], dtype=torch.float32
        ),
        rna_candidate_event_count=torch.tensor(
            audit_arrays["RNA candidate event count"], dtype=torch.float32
        ),
        rna_selected_event_count=torch.tensor(
            audit_arrays["RNA selected event count"], dtype=torch.float32
        ),
        rna_cap_saturated=torch.tensor(
            audit_arrays["RNA cap saturation"], dtype=torch.float32
        ),
        rna_boundary_rank_motif_score=torch.tensor(
            audit_arrays["RNA boundary motif score"], dtype=torch.float32
        ),
    )


def _require_train_centered(
    label: str, values: np.ndarray, molecule_mass: np.ndarray
) -> None:
    """Enforce the frozen molecule-weighted centering estimand at F2 output."""

    matrix = np.asarray(values, dtype=np.float64)
    weights = np.asarray(molecule_mass, dtype=np.float64)
    if matrix.ndim != 2 or weights.shape != (matrix.shape[0],):
        raise ValueError(f"{label} centering axes differ")
    total_mass = float(weights.sum())
    if total_mass == 0:
        if matrix.size and not np.allclose(matrix, 0.0, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{label} has values without train informative mass")
        return
    if matrix.shape[1] == 0:
        return
    weighted_mean = (weights[:, None] * matrix).sum(axis=0) / total_mass
    scale = np.maximum(1.0, np.max(np.abs(matrix), axis=0))
    if bool((np.abs(weighted_mean) > 5.0e-6 * scale).any()):
        raise ValueError(
            f"{label} is not centered by train likelihood-informative molecule mass"
        )


def _validate_graph_choice_identity(graph: GeneGraph, catalog: ChoiceCatalog) -> None:
    if graph.gene_id != catalog.gene_id:
        raise ValueError("graph and ChoiceCatalog gene IDs differ")
    if graph.path_ids != catalog.path_ids:
        raise ValueError("graph and ChoiceCatalog path ID order differs")
    alternative_count = sum(len(choice.alternatives) for choice in catalog.choices)
    if graph.path_edge_incidence.shape != (len(graph.path_ids), len(graph.edge_ids)):
        raise ValueError("graph path-edge incidence axes differ from graph IDs")
    if catalog.path_choice_incidence.shape != (
        len(graph.path_ids),
        alternative_count,
    ):
        raise ValueError("path-choice incidence axes differ from catalog IDs")


def _prepare_compatibility_rows(
    rows: pd.DataFrame,
    graph: GeneGraph,
    cell_ids: tuple[str, ...],
) -> tuple[list[tuple[int, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {
        "cell_id",
        "gene_id",
        "compatible_path_ids",
        "compatible_path_indices",
        "compatible_path_count",
        "molecule_count",
        "split",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"normalized EC rows miss columns: {missing}")
    if rows.empty:
        if cell_ids:
            raise ValueError("an unsupervised gene must have an empty State cell axis")
        empty_long = torch.empty((0,), dtype=torch.long)
        return (
            [],
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            empty_long,
        )
    validate_compatibility_rows(rows, graph)
    split_values = tuple(rows["split"].astype(str))
    if not set(split_values).issubset({"train", "val", "test"}):
        raise ValueError("EC rows contain an unknown authoritative split")

    ec_cell_ids = tuple(rows["cell_id"].astype(str))
    if tuple(canonical_rna_cell_id(value) for value in ec_cell_ids) != ec_cell_ids:
        raise ValueError("EC cell IDs must already use canonical RNA identities")
    ec_cell_set = set(ec_cell_ids)
    state_cell_set = set(cell_ids)
    if ec_cell_set != state_cell_set:
        missing_state = sorted(ec_cell_set - state_cell_set)
        extra_state = sorted(state_cell_set - ec_cell_set)
        raise ValueError(
            "EC and State cell identities differ: "
            f"missing_state={missing_state[:5]}, extra_state={extra_state[:5]}"
        )
    cell_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    row_cell_index = torch.tensor(
        [cell_index[cell_id] for cell_id in ec_cell_ids], dtype=torch.long
    )

    path_index = {path_id: index for index, path_id in enumerate(graph.path_ids)}
    compatible: list[tuple[int, ...]] = []
    for row_number, row in enumerate(rows.itertuples(index=False)):
        path_ids = tuple(str(value) for value in row.compatible_path_ids)
        expected = tuple(path_index[path_id] for path_id in path_ids)
        observed = tuple(int(value) for value in row.compatible_path_indices)
        if observed != expected:
            raise ValueError(
                f"EC row {row_number} path IDs and normalized path indices differ"
            )
        compatible.append(observed)
    width = max(len(values) for values in compatible)
    padded = torch.full((len(compatible), width), -1, dtype=torch.long)
    mask = torch.zeros((len(compatible), width), dtype=torch.bool)
    for row_index, values in enumerate(compatible):
        padded[row_index, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row_index, : len(values)] = True
    return compatible, padded, mask, row_cell_index


def _prepare_choice_eligibility(
    catalog: ChoiceCatalog, identifiability: pd.DataFrame
) -> np.ndarray:
    choice_ids = tuple(choice.choice_id for choice in catalog.choices)
    if not choice_ids:
        if len(identifiability):
            raise ValueError("choice identifiability has rows for a choice-free gene")
        return np.empty(0, dtype=bool)
    required = {"gene_id", "choice_id", "eligible"}
    missing = sorted(required - set(identifiability.columns))
    if missing:
        raise ValueError(f"choice identifiability misses columns: {missing}")
    observed_ids = tuple(identifiability["choice_id"].astype(str))
    _require_exact_axis("choice identifiability", observed_ids, choice_ids)
    if set(identifiability["gene_id"].astype(str)) != {catalog.gene_id}:
        raise ValueError("choice identifiability and catalog gene IDs differ")
    if not pd.api.types.is_bool_dtype(identifiability["eligible"]):
        raise TypeError("choice identifiability eligible must use bool dtype")
    return identifiability["eligible"].to_numpy(dtype=bool)


def _prepare_event_tensors(
    modality: str,
    data: OrderedEventData,
    gene_id: str,
    catalog: ChoiceCatalog,
    alternative_ids: tuple[str, ...],
    cell_ids: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    from .motifs import event_relation_matrix

    required = {
        "event_id",
        "modality",
        "gene_id",
        "choice_id",
        "relation_alternative_ids",
    }
    missing = sorted(required - set(data.events.columns))
    if missing:
        raise ValueError(f"{modality} event table misses columns: {missing}")
    event_ids = _ordered_unique_ids(
        f"{modality} event",
        tuple(data.events["event_id"].astype(str)),
        allow_empty=True,
    )
    _require_exact_axis(f"{modality} feature event", data.feature_event_ids, event_ids)
    _require_exact_axis(
        f"{modality} relation event", data.relation_event_ids, event_ids
    )
    _require_exact_axis(f"{modality} gate event", data.gate_event_ids, event_ids)
    _require_exact_axis(
        f"{modality} relation alternative",
        data.relation_alternative_ids,
        alternative_ids,
    )
    _require_exact_axis(f"{modality} gate cell", data.gate_cell_ids, cell_ids)

    if len(data.events):
        if set(data.events["gene_id"].astype(str)) != {gene_id}:
            raise ValueError(f"{modality} events and graph gene IDs differ")
        if set(data.events["modality"].astype(str).str.upper()) != {modality}:
            raise ValueError(f"{modality} event table contains another modality")
    choice_ids = tuple(choice.choice_id for choice in catalog.choices)
    choice_index = {choice_id: index for index, choice_id in enumerate(choice_ids)}
    event_choice_ids = tuple(data.events["choice_id"].astype(str))
    unknown_choices = sorted(set(event_choice_ids) - set(choice_ids))
    if unknown_choices:
        raise ValueError(
            f"{modality} events reference unknown choices: {unknown_choices[:5]}"
        )
    alternative_choice = {
        alternative.alternative_id: choice.choice_id
        for choice in catalog.choices
        for alternative in choice.alternatives
    }
    for event_id, event_choice_id, related in zip(
        event_ids,
        event_choice_ids,
        data.events["relation_alternative_ids"],
        strict=True,
    ):
        related_ids = tuple(str(value) for value in related)
        if not related_ids:
            raise ValueError(f"event {event_id} has no related alternative")
        if any(
            alternative_choice.get(value) != event_choice_id for value in related_ids
        ):
            raise ValueError(
                f"event {event_id} relates to an alternative outside its choice"
            )

    features = _finite_matrix(f"{modality} event features", data.features)
    relation = _finite_matrix(f"{modality} event relation", data.relation)
    gate = _finite_matrix(f"{modality} event gate", data.gate)
    if features.shape[0] != len(event_ids):
        raise ValueError(f"{modality} feature rows and event IDs differ")
    if relation.shape != (len(event_ids), len(alternative_ids)):
        raise ValueError(f"{modality} relation axes differ from event/alternative IDs")
    if gate.shape != (len(cell_ids), len(event_ids)):
        raise ValueError(f"{modality} gate axes differ from cell/event IDs")
    expected_relation = event_relation_matrix(data.events, alternative_ids)
    if not np.array_equal(relation, expected_relation):
        raise ValueError(f"{modality} relation values differ from event identities")

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(relation, dtype=torch.float32),
        torch.tensor(
            [choice_index[value] for value in event_choice_ids], dtype=torch.long
        ),
        torch.tensor(gate, dtype=torch.float32),
        event_ids,
    )


def _identifiable_ec_rows(
    catalog: ChoiceCatalog,
    compatible_indices: Sequence[tuple[int, ...]],
    choice_eligible: np.ndarray,
) -> np.ndarray:
    """Mark ECs informative for at least one admitted identifiable choice."""

    result = np.zeros(len(compatible_indices), dtype=bool)
    path_count = len(catalog.path_ids)
    for choice_number, eligible in enumerate(choice_eligible):
        if not eligible:
            continue
        start = catalog.alternative_offsets[choice_number]
        stop = catalog.alternative_offsets[choice_number + 1]
        incidence = catalog.path_choice_incidence[:, start:stop].astype(np.int64)
        total = np.asarray(incidence.sum(axis=0)).reshape(-1)
        total_contrast = total[:-1] - total[-1]
        for row_number, path_indices in enumerate(compatible_indices):
            selected = np.asarray(incidence[list(path_indices)].sum(axis=0)).reshape(-1)
            selected_contrast = selected[:-1] - selected[-1]
            if np.any(
                selected_contrast * path_count != total_contrast * len(path_indices)
            ):
                result[row_number] = True
    return result


def _ordered_unique_ids(
    label: str, values: Sequence[str], *, allow_empty: bool
) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{label} ID axis must be non-empty")
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} IDs must be unique and non-empty")
    return result


def _require_exact_axis(
    label: str, observed: Sequence[str], expected: Sequence[str]
) -> None:
    observed_ids = tuple(str(value) for value in observed)
    expected_ids = tuple(str(value) for value in expected)
    if observed_ids != expected_ids:
        raise ValueError(f"{label} ID order differs from its authoritative axis")


def _finite_matrix(label: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{label} must contain numeric values")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.asarray(matrix, dtype=np.float32)


def _pca_batches(
    rows: np.ndarray, *, batch_size: int, minimum: int
) -> list[np.ndarray]:
    if batch_size < minimum:
        raise ValueError("PCA batch_size must be at least n_components")
    batches = [
        rows[start : start + batch_size] for start in range(0, len(rows), batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) < minimum:
        batches[-2] = np.concatenate([batches[-2], batches[-1]])
        batches.pop()
    return batches


def _require_known_stage(values: np.ndarray, label: str) -> None:
    if any(str(value) in {"", "nan", "Unknown"} for value in values):
        raise ValueError(f"{label} stage is missing; stage-restricted KNN is undefined")
