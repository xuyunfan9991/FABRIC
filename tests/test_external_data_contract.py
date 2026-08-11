from __future__ import annotations

import json
from pathlib import Path

import hdf5plugin  # noqa: F401 - register the Blosc filter before reading peak X
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fabric.annotation import (
    load_external_inputs,
    load_split_rows,
    resolve_and_validate_graph_generation,
)
from fabric.dataset import load_full_rna_glue_context
from fabric.graph import (
    bind_authoritative_split,
    load_compatibility_rows,
    load_graph_tables,
    normalize_compatibility_path_order,
    split_gene_graphs,
    validate_compatibility_rows,
)
from fabric.motifs import parse_cisbp_motifs, parse_meme_motifs


ROOT = Path(__file__).parents[1]
EXTERNAL_INPUTS = ROOT / "data" / "external_inputs.yaml"

RNA_OBS_COLUMNS = (
    "emb_id",
    "stage",
    "dissection_part",
    "system_knn_score",
    "type_knn_score",
    "developmental_system",
    "cell_type",
    "cell_type_merged",
    "high_quality",
    "in_system",
    "in_cell_type",
)
GLUE_OBS_COLUMNS = (
    "modality",
    "anatomy",
    "stage_scanvi",
    "stage_window",
    "developmental_system",
    "cell_type",
    "cell_type_merged",
    "emb_id",
    "sample_name",
    "tissue",
    "broad_tissue",
    "carnegie_stage",
)
PEAK_OBS_COLUMNS = (
    "sample",
    "barcode",
    "sample_name",
    "sample_id",
    "cell_id",
    "glue_guided_cluster",
    "pseudoreplicate",
    "developmental_system",
)

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def inputs():
    return load_external_inputs(EXTERNAL_INPUTS, require_exists=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _assert_finite_in_row_chunks(
    dataset: h5py.Dataset, chunk_rows: int = 65_536
) -> None:
    for start in range(0, dataset.shape[0], chunk_rows):
        values = dataset[start : start + chunk_rows]
        assert np.isfinite(values).all()


def test_full_rna_glue_and_supervised_split_have_exact_cell_identity(inputs):
    rna = ad.read_h5ad(inputs.path("rna_counts"), backed="r")
    glue = ad.read_h5ad(inputs.path("full_rna_glue_embedding"), backed="r")
    try:
        assert rna.shape == (217_933, 32_351)
        assert tuple(rna.obs.columns) == RNA_OBS_COLUMNS
        assert tuple(rna.var.columns) == ("feature_type",)
        assert rna.obs.index.name == "cell_id"
        assert rna.var.index.name == "gene_id"
        assert rna.obs_names.is_unique and rna.var_names.is_unique
        assert rna.X.format == "csr"
        assert rna.X.dtype == np.dtype("int32")

        broad_mask = rna.obs["in_system"].to_numpy() == 1
        broad_rna_ids = rna.obs_names[broad_mask].to_numpy(dtype=str)
        assert len(broad_rna_ids) == 205_864
        assert int((rna.obs["high_quality"].to_numpy() == 1).sum()) == 217_933

        assert glue.shape == (438_338, 0)
        assert tuple(glue.obs.columns) == GLUE_OBS_COLUMNS
        assert tuple(glue.var.columns) == ()
        assert glue.obs_names.is_unique
        modality = glue.obs["modality"].astype(str).to_numpy()
        assert np.all(modality[:205_864] == "RNA")
        assert np.all(modality[205_864:] == "ATAC")
        glue_rna_ids = glue.obs_names[:205_864].to_numpy(dtype=str)
        np.testing.assert_array_equal(
            glue_rna_ids,
            np.asarray([f"RNA__{cell_id}" for cell_id in broad_rna_ids]),
        )

        split = pd.read_parquet(
            inputs.path("cell_split"),
            columns=["cell_id", "rna_embryo_id", "split"],
        )
        assert len(split) == 167_235
        assert split["cell_id"].is_unique
        assert split["cell_id"].str.startswith("RNA__").all()
        assert split["split"].value_counts().to_dict() == {
            "train": 133_799,
            "test": 16_718,
            "val": 16_718,
        }
        glue_rna_index = pd.Index(glue_rna_ids)
        assert np.all(glue_rna_index.get_indexer(split["cell_id"].astype(str)) >= 0)

        canonical_split_ids = split["cell_id"].str.removeprefix("RNA__")
        rna_index = rna.obs_names.get_indexer(canonical_split_ids)
        assert np.all(rna_index >= 0)
        selected_obs = rna.obs.iloc[rna_index]
        assert np.all(selected_obs["in_system"].to_numpy() == 1)
        np.testing.assert_array_equal(
            split["rna_embryo_id"].astype(str),
            selected_obs["emb_id"].astype(str),
        )
    finally:
        rna.file.close()
        glue.file.close()

    with h5py.File(inputs.path("full_rna_glue_embedding"), "r") as handle:
        assert handle["X"].shape == (438_338, 0)
        assert handle["obsm/X_glue"].shape == (438_338, 50)
        assert handle["obsm/X_umap"].shape == (438_338, 2)
        assert handle["obsm/X_glue"].dtype == np.dtype("float32")
        assert handle["obsm/X_umap"].dtype == np.dtype("float32")
        _assert_finite_in_row_chunks(handle["obsm/X_glue"])
        _assert_finite_in_row_chunks(handle["obsm/X_umap"])


def test_current_753753_peak_axis_and_atac_glue_block_are_exact(inputs):
    peak_path = inputs.path("full_rna_atac_peak_counts")
    glue_path = inputs.path("full_rna_glue_embedding")
    bed_path = inputs.path("full_rna_consensus_peak_bed")
    peak = ad.read_h5ad(peak_path, backed="r")
    glue = ad.read_h5ad(glue_path, backed="r")
    try:
        assert peak.shape == (232_474, 753_753)
        assert peak.X.format == "csr"
        assert peak.X.dtype == np.dtype("uint32")
        assert tuple(peak.obs.columns) == PEAK_OBS_COLUMNS
        assert tuple(peak.var.columns) == ()
        assert peak.obs_names.is_unique and peak.var_names.is_unique
        np.testing.assert_array_equal(
            peak.obs["cell_id"].astype(str), peak.obs_names.to_numpy(dtype=str)
        )

        glue_atac_ids = glue.obs_names[205_864:].to_numpy(dtype=str)
        assert all(cell_id.startswith("ATAC__") for cell_id in glue_atac_ids)
        np.testing.assert_array_equal(
            np.asarray([cell_id.removeprefix("ATAC__") for cell_id in glue_atac_ids]),
            peak.obs_names.to_numpy(dtype=str),
        )

        previous_chromosome: str | None = None
        previous_end = -1
        chromosome_blocks: list[str] = []
        bed_rows = 0
        with bed_path.open() as handle:
            for bed_rows, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                assert len(fields) >= 3
                chrom, start_text, end_text = fields[:3]
                start, end = int(start_text), int(end_text)
                assert start >= 0 and end - start == 501
                assert peak.var_names[bed_rows - 1] == f"{chrom}:{start}-{end}"
                if chrom != previous_chromosome:
                    assert chrom not in chromosome_blocks
                    chromosome_blocks.append(chrom)
                    previous_chromosome = chrom
                    previous_end = -1
                assert start >= previous_end
                previous_end = end
        assert bed_rows == 753_753
        assert chromosome_blocks == [
            "chr1",
            "chr10",
            "chr11",
            "chr12",
            "chr13",
            "chr14",
            "chr15",
            "chr16",
            "chr17",
            "chr18",
            "chr19",
            "chr2",
            "chr20",
            "chr21",
            "chr22",
            "chr3",
            "chr4",
            "chr5",
            "chr6",
            "chr7",
            "chr8",
            "chr9",
            "chrX",
            "chrY",
        ]
    finally:
        peak.file.close()
        glue.file.close()

    with h5py.File(glue_path, "r") as glue_h5, h5py.File(peak_path, "r") as peak_h5:
        for key in ("X_glue", "X_umap"):
            assert peak_h5[f"obsm/{key}"].shape[0] == 232_474
            for start in range(0, 232_474, 32_768):
                stop = min(start + 32_768, 232_474)
                np.testing.assert_array_equal(
                    glue_h5[f"obsm/{key}"][205_864 + start : 205_864 + stop],
                    peak_h5[f"obsm/{key}"][start:stop],
                )


def test_backed_count_fixtures_remain_sparse_and_blosc_is_registered(inputs):
    rna = ad.read_h5ad(inputs.path("rna_counts"), backed="r")
    try:
        rna_chunk, start, stop = next(rna.chunked_X(8))
        assert (start, stop) == (0, 8)
        assert sparse.isspmatrix_csr(rna_chunk)
        assert rna_chunk.shape == (8, 32_351)
        assert rna_chunk.dtype == np.dtype("int32")
        assert rna_chunk.nnz == 17_503
        assert np.all(rna_chunk.data > 0)
        first_rna_row = rna_chunk.getrow(0)
        np.testing.assert_array_equal(first_rna_row.indices[:5], [19, 24, 51, 53, 64])
        np.testing.assert_array_equal(first_rna_row.data[:5], [1, 2, 1, 1, 1])
    finally:
        rna.file.close()

    peak_path = inputs.path("full_rna_atac_peak_counts")
    with h5py.File(peak_path, "r") as handle:
        creation = handle["X/data"].id.get_create_plist()
        filter_ids = [
            creation.get_filter(index)[0] for index in range(creation.get_nfilters())
        ]
        assert 32_001 in filter_ids

    peak = ad.read_h5ad(peak_path, backed="r")
    try:
        peak_chunk, start, stop = next(peak.chunked_X(8))
        assert (start, stop) == (0, 8)
        assert sparse.isspmatrix_csr(peak_chunk)
        assert peak_chunk.shape == (8, 753_753)
        assert peak_chunk.dtype == np.dtype("uint32")
        assert peak_chunk.nnz == 83_024
        assert np.all(peak_chunk.data > 0)
        first_peak_row = peak_chunk.getrow(0)
        np.testing.assert_array_equal(
            first_peak_row.indices[:5], [54, 72, 113, 117, 173]
        )
        np.testing.assert_array_equal(first_peak_row.data[:5], [1, 1, 2, 1, 2])
        selected = first_peak_row[:, [54, 72, 113, 117, 173]]
        assert sparse.isspmatrix_csr(selected)
        np.testing.assert_array_equal(selected.data, [1, 1, 2, 1, 2])
    finally:
        peak.file.close()


def test_provenance_chain_selects_the_full_rna_peak_generation(inputs):
    glue_path = inputs.path("full_rna_glue_embedding")
    bed_path = inputs.path("full_rna_consensus_peak_bed")
    peak_path = inputs.path("full_rna_atac_peak_counts")
    peak_dir = bed_path.parent
    prepare = _read_json(
        glue_path.parent.parent / "inputs" / "prepare_glue_5kb_inputs_provenance.json"
    )
    glue_run = _read_json(inputs.path("full_rna_glue_provenance"))
    peak_calling = _read_json(inputs.path("full_rna_peak_provenance"))
    peak_validation = _read_json(inputs.path("full_rna_peak_validation"))
    matrix_provenance = _read_json(peak_dir / "peak_matrix.provenance.json")
    matrix_validation = _read_json(peak_dir / "peak_matrix.validation.json")
    annotated_audit = _read_json(
        peak_dir / "peak_matrix_with_developmental_system.audit.json"
    )

    assert prepare["parameters"]["rna_filter_mode"] == "broad_system"
    assert prepare["parameters"]["active_rna_filter_criteria"] == (
        "high_quality & in_system & valid(developmental_system)"
    )
    assert prepare["rna_shape"] == [205_864, 22_882]
    assert prepare["atac_shape"] == [232_474, 80_377]
    assert glue_run["rna_shape"] == [205_864, 22_882]
    assert glue_run["atac_shape"] == [232_474, 80_377]
    assert glue_run["combined_shape"] == [438_338, 0]

    glue_sha256 = "2757f5e077dbcb6b0cb8ddebe84294f8dc6be1a8a53d646b93bd47d3480a2804"
    bed_sha256 = "101cc19965e83b23f22f017f5ab715d432e47ee33afd9037740e0ca12e91cc42"
    assert peak_calling["status"] == "PASS"
    assert peak_calling["inputs"]["combined_embedding"]["path"] == str(glue_path)
    assert peak_calling["inputs"]["combined_embedding"]["sha256"] == glue_sha256
    assert peak_calling["counts"]["n_atac_cells"] == 232_474
    assert peak_calling["counts"]["n_consensus_peaks"] == 753_753
    assert peak_calling["outputs"]["consensus_peaks"]["path"] == str(bed_path)
    assert peak_calling["outputs"]["consensus_peaks"]["sha256"] == bed_sha256

    assert peak_validation["status"] == "PASS"
    assert peak_validation["coembedding"]["n_rna_cells"] == 205_864
    assert peak_validation["coembedding"]["n_atac_cells"] == 232_474
    assert peak_validation["consensus"]["n_peaks"] == 753_753
    assert peak_validation["consensus"]["sha256_matches_provenance"] is True

    assert matrix_provenance["status"] == "PASS"
    assert matrix_provenance["matrix_semantics"] == "raw fragment-overlap counts"
    assert (
        matrix_provenance["inputs"]["combined_glue_coembedding"]["sha256"]
        == glue_sha256
    )
    assert matrix_provenance["inputs"]["consensus_peaks"]["sha256"] == bed_sha256
    assert matrix_provenance["counts"] == {
        "n_cells": 232_474,
        "n_peaks": 753_753,
        "n_peak_bed_rows": 753_753,
        "nnz": 2_041_277_276,
    }
    assert matrix_validation["status"] == "PASS"
    assert matrix_validation["shape"] == [232_474, 753_753]
    assert matrix_validation["matrix_format"] == "csr"
    assert matrix_validation["X_dtype"] == "uint32"

    assert annotated_audit["status"] == "PASS"
    assert annotated_audit["rna_reference_cells"] == 205_864
    assert annotated_audit["source_shape"] == [232_474, 753_753]
    assert annotated_audit["output"]["path"] == str(peak_path)
    assert annotated_audit["checks"]["only_developmental_system_added"] is True
    assert annotated_audit["checks"]["obs_names_identical"] is True
    assert annotated_audit["checks"]["var_names_identical"] is True
    assert annotated_audit["checks"]["X_glue_exact"] is True
    assert annotated_audit["checks"]["X_umap_exact"] is True
    assert annotated_audit["X_payload_validation"]["data"]["exact"] is True
    assert annotated_audit["X_payload_validation"]["indices"]["exact"] is True
    assert annotated_audit["X_payload_validation"]["indptr"]["exact"] is True


def test_frozen_v1_motif_parsers_select_only_the_documented_libraries(inputs):
    dna = parse_meme_motifs(inputs.path("dna_motif_library"))
    dna_index = pd.read_csv(inputs.path("dna_motif_index"), sep="\t", dtype=str)
    assert len(dna) == len(dna_index) == 1_019
    assert tuple(dna) == tuple(dna_index["motif_id"])
    assert [motif.width for motif in dna.values()] == dna_index["width"].astype(
        int
    ).tolist()

    rna_map = pd.read_csv(inputs.path("rna_motif_gene_map"), sep="\t", dtype=str)
    mapped_ids = tuple(sorted(rna_map["motif_id"].unique()))
    rna = parse_cisbp_motifs(inputs.path("rna_motif_directory"), motif_ids=mapped_ids)
    assert len(rna) == len(mapped_ids) == 492
    assert set(rna) == set(mapped_ids)
    assert len(list(inputs.path("rna_motif_directory").glob("*.txt"))) == 586


def test_full_rna_glue_loader_uses_stage_scanvi_and_peak_donor_axis(inputs):
    split = pd.read_parquet(inputs.path("cell_split"), columns=["cell_id"]).head(3)
    target_ids = tuple(split["cell_id"].str.removeprefix("RNA__"))
    context = load_full_rna_glue_context(
        inputs.path("full_rna_glue_embedding"),
        inputs.path("full_rna_atac_peak_counts"),
        target_rna_cell_ids=target_ids,
    )
    assert context.rna_cell_ids == target_ids
    assert context.rna_embedding.shape == (3, 50)
    assert context.atac_embedding.shape == (232_474, 50)
    assert set(context.rna_stage) <= {"CS10", "CS11", "CS12", "CS13", "CS14", "CS15"}
    assert len(context.atac_cell_ids) == len(context.atac_donor_ids) == 232_474
    assert all(value != "Unknown" for value in context.atac_developmental_system)


def test_current_graph_ec_and_split_convert_without_old_embedded_split(inputs):
    generation = resolve_and_validate_graph_generation(inputs)
    assert generation == inputs.path("graph_generation").resolve()
    gene_id = "ENSG00000275074"
    graph = list(split_gene_graphs(load_graph_tables(generation, gene_ids=[gene_id])))
    assert len(graph) == 1
    graph = graph[0]
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 7
    assert graph.path_edge_incidence.shape == (2, 7)
    assert graph.path_edge_incidence.nnz == 10
    assert set(graph.edges["edge_type"]) == {
        "EXON_CONTINUATION",
        "SPLICE",
        "RETAINED_INTRON",
    }

    split = load_split_rows(inputs.path("cell_split"))
    authority_ids = set(split["cell_id"].astype(str))
    source_ec = pd.concat(
        load_compatibility_rows(inputs.path("compatibility_ec"), gene_ids=[gene_id]),
        ignore_index=True,
    )
    assert len(source_ec) == 97
    current_ec = source_ec.loc[source_ec["cell_id"].astype(str).isin(authority_ids)]
    assert len(current_ec) == 66
    bound = bind_authoritative_split(current_ec, split)
    assert (
        int(
            (
                bound["split"] != source_ec.loc[current_ec.index, "split"].to_numpy()
            ).sum()
        )
        == 10
    )
    normalized = normalize_compatibility_path_order(bound, graph)
    validate_compatibility_rows(normalized, graph)
    assert normalized["split"].value_counts().to_dict() == {
        "train": 56,
        "test": 5,
        "val": 5,
    }
