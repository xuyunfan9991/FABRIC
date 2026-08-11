from __future__ import annotations

import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import sparse
import torch
import yaml

from fabric.choices import choice_identifiability, extract_elementary_choices
from fabric.dataset import (
    STATE_PCA_FIT_BATCH_SIZE,
    OrderedCellState,
    OrderedChoiceAudit,
    OrderedEventData,
    PreparationSources,
    prepare_dataset_from_external_inputs,
    prepare_gene,
    validate_prepared_external_context,
)
from fabric.evaluate import (
    null_contexts_from_prepared,
    rebuild_stage_system_factor_null,
    retrain_stage_system_factor_null,
)
from fabric.train import (
    NORMALIZED_SOURCE_ROLES,
    preparation_values_from_config,
    prepare_dataset_identity,
)


REAL_FIXTURE = Path(__file__).parent / "fixtures" / "real"
GENE_ID = "ENSG00000275074"


def test_real_external_inputs_build_one_identity_bound_prepared_dataset(tmp_path):
    manifest, config, reviewed, donor_eligibility, peak_support = (
        _write_normalized_inputs(tmp_path)
    )
    bundle_path = tmp_path / "prepared" / "real_fixture.pt"
    dataset = prepare_dataset_from_external_inputs(
        manifest,
        config,
        target_gene_ids=[GENE_ID],
        reviewed_factor_mapping_path=reviewed,
        atac_donor_eligibility_path=donor_eligibility,
        peak_support_path=peak_support,
        neighbor_device="cpu",
        io_batch_size=16,
        output_path=bundle_path,
    )

    assert dataset.target_gene_ids == (GENE_ID,)
    assert dataset.factor_mapping_reviewed
    assert {role for role, _ in dataset.normalized_source_paths} == set(
        NORMALIZED_SOURCE_ROLES
    )
    assert dataset.reviewed_factor_mapping == str(reviewed.resolve())
    assert dataset.atac_donor_eligibility_source == str(donor_eligibility.resolve())
    assert dataset.peak_support_source == str(peak_support.resolve())
    assert dataset.preparation_config_source == str(config.resolve())
    expected_preparation = tuple(
        preparation_values_from_config(yaml.safe_load(config.read_text())).items()
    )
    assert dataset.preparation_values == expected_preparation
    loaded = torch.load(bundle_path, map_location="cpu", weights_only=False)
    assert loaded.target_gene_ids == dataset.target_gene_ids
    assert loaded.normalized_source_paths == dataset.normalized_source_paths
    assert loaded.preparation_config_source == dataset.preparation_config_source
    assert loaded.preparation_values == dataset.preparation_values
    validate_prepared_external_context(loaded)
    assert dataset.state_pca is not None
    assert dataset.state_pca.fit_batch_size == STATE_PCA_FIT_BATCH_SIZE
    assert dataset.factor_context is not None
    assert dataset.atac_context is not None
    assert dataset.factor_context.cell_ids == dataset.atac_context.cell_ids
    assert sparse.isspmatrix_csr(dataset.atac_context.accessibility)
    gene = dataset.genes[0]
    assert gene.preparation_config_source == dataset.preparation_config_source
    assert gene.preparation_values == dataset.preparation_values
    assert gene.state_features.shape == (66, 4)  # 3 train PCA axes + library size
    assert gene.dna_event_features.shape[0] > 0
    assert gene.rna_event_features.shape[0] > 0
    assert gene.dna_gate.shape == (66, len(gene.dna_event_ids))
    assert gene.rna_gate.shape == (66, len(gene.rna_event_ids))
    assert gene.path_edge_incidence.is_sparse
    assert gene.path_choice_incidence.is_sparse
    assert gene.alternative_eligible.tolist() == [True, True]
    assert torch_finite(gene.state_features)
    assert torch_finite(gene.dna_gate)
    assert torch_finite(gene.rna_gate)
    assert gene.state_baseline is not None
    assert gene.dna_baseline is not None
    assert gene.rna_baseline is not None
    assert len(gene.dna_event_factor_ids) == len(gene.dna_event_ids)
    assert len(gene.rna_event_factor_ids) == len(gene.rna_event_ids)
    assert len(gene.dna_event_peak_ids) == len(gene.dna_event_ids)

    train_informative = np.asarray(gene.split) == "train"
    train_informative &= gene.compatible_path_mask.sum(dim=1).numpy() < len(
        gene.path_ids
    )
    weights = np.bincount(
        gene.row_cell_index.numpy()[train_informative],
        weights=gene.molecule_count.numpy()[train_informative],
        minlength=len(gene.cell_ids),
    )
    for values in (gene.state_features, gene.dna_gate, gene.rna_gate):
        weighted = (values.numpy() * weights[:, None]).sum(axis=0)
        np.testing.assert_allclose(weighted, 0.0, atol=2e-4)
    assert np.count_nonzero(gene.rna_gate.numpy()) > 0
    assert np.count_nonzero(gene.dna_gate.numpy()) > 0

    factor_context, null_contexts = null_contexts_from_prepared(dataset)
    assert factor_context.factor_ids == ("F_DNA", "F_RNA")
    assert set(factor_context.developmental_system) == {"fixture_system"}
    assert len(null_contexts) == 1
    assert null_contexts[0].dna_accessibility.shape == gene.dna_gate.shape
    rebuilt, _ = rebuild_stage_system_factor_null(
        dataset.genes,
        factor_context,
        null_contexts,
        seed=71,
        minimum_valid_molecule_mass=1.0,
        minimum_weighted_variance=1.0e-8,
    )
    for values in (rebuilt[0].dna_gate, rebuilt[0].rna_gate):
        np.testing.assert_allclose(
            (values.numpy() * weights[:, None]).sum(axis=0), 0.0, atol=2e-4
        )
    assert rebuilt[0].dna_baseline is not gene.dna_baseline
    assert rebuilt[0].rna_baseline is not gene.rna_baseline

    second = prepare_dataset_from_external_inputs(
        manifest,
        config,
        target_gene_ids=[GENE_ID],
        reviewed_factor_mapping_path=reviewed,
        atac_donor_eligibility_path=donor_eligibility,
        peak_support_path=peak_support,
        neighbor_device="cpu",
        io_batch_size=7,
    )
    np.testing.assert_array_equal(
        second.state_pca.components, dataset.state_pca.components
    )
    torch.testing.assert_close(
        second.genes[0].state_features,
        gene.state_features,
        atol=0.0,
        rtol=0.0,
    )

    null_config = yaml.safe_load(config.read_text())
    null_config["diagnostic_panel"] = {"frozen_gene_ids": [GENE_ID]}
    null_config["training"]["max_epochs"] = 1
    null_config["admission"]["minimum_b0_validation_improvement"] = -1.0e9
    null_result = retrain_stage_system_factor_null(
        dataset,
        null_config,
        seed=73,
        device="cpu",
        run_dir=tmp_path / "null_run",
    )
    assert null_result.metrics["screening_evidence_only"].all()
    assert (tmp_path / "null_run" / "factor_permutation.tsv").exists()


def test_no_ec_gene_is_explicit_graph_only_prepared_gene(toy_gene_graph):
    catalog = extract_elementary_choices(toy_gene_graph)
    ec_rows = pd.DataFrame(
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
    identifiability = choice_identifiability(
        catalog,
        ec_rows,
        rank_tolerance=1e-8,
        minimum_informative_molecule_mass=1,
        minimum_alternative_support=1,
    )
    alternative_ids = tuple(
        alternative.alternative_id
        for choice in catalog.choices
        for alternative in choice.alternatives
    )

    def empty_events(modality: str, feature_width: int) -> OrderedEventData:
        events = pd.DataFrame(
            columns=[
                "event_id",
                "modality",
                "gene_id",
                "choice_id",
                "relation_alternative_ids",
            ]
        )
        return OrderedEventData(
            events=events,
            feature_event_ids=(),
            features=np.empty((0, feature_width), dtype=np.float32),
            relation_event_ids=(),
            relation_alternative_ids=alternative_ids,
            relation=np.empty((0, len(alternative_ids)), dtype=np.float32),
            gate_cell_ids=(),
            gate_event_ids=(),
            gate=np.empty((0, 0), dtype=np.float32),
        )

    prepared = prepare_gene(
        toy_gene_graph,
        catalog,
        ec_rows,
        state=OrderedCellState(cell_ids=(), values=np.empty((0, 4), dtype=np.float32)),
        dna=empty_events("DNA", 8),
        rna=empty_events("RNA", 7),
        choice_identifiability=identifiability,
        choice_audit=OrderedChoiceAudit(
            choice_ids=tuple(choice.choice_id for choice in catalog.choices),
            alternative_span=np.asarray([30.0], dtype=np.float32),
            dna_candidate_event_count=np.asarray([0]),
            dna_selected_event_count=np.asarray([0]),
            dna_cap_saturated=np.asarray([False]),
            dna_boundary_rank_motif_score=np.asarray([np.nan], dtype=np.float32),
            rna_candidate_event_count=np.asarray([0]),
            rna_selected_event_count=np.asarray([0]),
            rna_cap_saturated=np.asarray([False]),
            rna_boundary_rank_motif_score=np.asarray([np.nan], dtype=np.float32),
        ),
        sources=PreparationSources("graph", "split"),
    )
    assert prepared.split == ()
    assert prepared.cell_ids == ()
    assert prepared.compatible_path_indices.shape == (0, 0)
    assert not prepared.alternative_eligible.any()
    dataset = prepare_dataset_identity([prepared], factor_mapping_reviewed=False)
    assert dataset.target_gene_ids == (toy_gene_graph.gene_id,)


def torch_finite(values) -> bool:
    return bool(np.isfinite(values.numpy()).all())


def _write_normalized_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    generation = root / "graph" / "generations" / "fixture"
    shutil.copytree(REAL_FIXTURE / "graph_generation", generation)
    contract = generation / "outputs" / "graph" / "graph_artifact_contract.json"
    contract.write_text(
        json.dumps(
            {
                "coordinate_system": "0-based-half-open",
                "transcript_direction": "5prime_to_3prime",
            }
        )
    )
    pointer = root / "graph" / "CURRENT.json"
    pointer.write_text(json.dumps({"generation": str(generation.resolve())}))

    ec = REAL_FIXTURE / "compatibility_equivalence_classes.parquet"
    split = REAL_FIXTURE / "split_rows.parquet"
    cells = pd.read_parquet(ec, columns=["cell_id"])["cell_id"].astype(str).tolist()
    n_cells = len(cells)

    rna_counts = root / "rna_counts.h5ad"
    count_rows = np.asarray(
        [
            [
                1 + index % 7,
                2 + (index * 3) % 11,
                1 + index % 3,
                3 + index % 13,
            ]
            for index in range(n_cells)
        ],
        dtype=np.int32,
    )
    rna = ad.AnnData(
        X=sparse.csr_matrix(count_rows),
        obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")),
        var=pd.DataFrame(
            index=pd.Index(["ENSGTF", "ENSGRBP", "ENSGA", "ENSGB"], name="gene_id")
        ),
    )
    rna.write_h5ad(rna_counts)

    atac_ids = [f"ATAC_{index}" for index in range(6)]
    glue_obs = pd.DataFrame(
        {
            "modality": ["RNA"] * n_cells + ["ATAC"] * len(atac_ids),
            "stage_scanvi": ["CS11"] * (n_cells + len(atac_ids)),
            "developmental_system": ["fixture_system"] * n_cells
            + ["Unknown"] * len(atac_ids),
        },
        index=pd.Index(
            [f"RNA__{cell_id}" for cell_id in cells]
            + [f"ATAC__{cell_id}" for cell_id in atac_ids],
            name="cell_id",
        ),
    )
    glue = ad.AnnData(X=np.empty((len(glue_obs), 0), dtype=np.float32), obs=glue_obs)
    glue.obsm["X_glue"] = np.vstack(
        [
            np.column_stack(
                (
                    np.linspace(0.0, 1.0, n_cells, dtype=np.float32),
                    np.zeros(n_cells, dtype=np.float32),
                )
            ),
            np.column_stack(
                (
                    np.linspace(0.0, 1.0, len(atac_ids), dtype=np.float32),
                    np.zeros(len(atac_ids), dtype=np.float32),
                )
            ),
        ]
    )
    glue_path = root / "glue.h5ad"
    glue.write_h5ad(glue_path)

    peak_rows = [("chr8", 22107880, 22107920), ("chr8", 22108120, 22108145)]
    peak_ids = [f"{chrom}:{start}-{end}" for chrom, start, end in peak_rows]
    peak_counts = root / "peak_counts.h5ad"
    atac = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [[1 + index, 2 + (index * 2) % 5] for index in range(len(atac_ids))],
                dtype=np.int32,
            )
        ),
        obs=pd.DataFrame(
            {
                "sample_id": ["donor_bad"] * 3 + ["donor_good"] * 3,
                "developmental_system": ["system"] * len(atac_ids),
            },
            index=pd.Index(atac_ids, name="cell_id"),
        ),
        var=pd.DataFrame(index=pd.Index(peak_ids, name="peak_id")),
    )
    atac.write_h5ad(peak_counts)
    peak_bed = root / "consensus_peaks.bed"
    peak_bed.write_text(
        "".join(f"{chrom}\t{start}\t{end}\n" for chrom, start, end in peak_rows)
    )
    peak_support = root / "reviewed_peak_support.tsv"
    pd.DataFrame({"peak_id": peak_ids, "peak_support": [11.0, 7.0]}).to_csv(
        peak_support, sep="\t", index=False
    )

    fasta = root / "genome.fa"
    _write_chr8_reference(fasta, length=22_110_000)
    indexed = Fasta(str(fasta))
    indexed.close()

    rna_gtf = root / "rna_genes.gtf"
    rna_gtf.write_text(
        'chr8\ttest\tgene\t1\t10\t.\t+\t.\tgene_id "ENSGTF"; gene_name "TF1";\n'
        'chr8\ttest\tgene\t20\t30\t.\t+\t.\tgene_id "ENSGRBP"; gene_name "RBP1";\n'
    )
    transcript_gtf = root / "transcripts.gtf"
    transcript_gtf.write_text("# normalized fixture placeholder\n")
    dna_index = root / "jaspar_index.tsv"
    dna_index.write_text("motif_id\ttf_name\nM_DNA\tTF1\n")
    dna_library = root / "dna.meme"
    dna_library.write_text(
        "MEME version 4\n\n"
        "MOTIF M_DNA TF1\n"
        "letter-probability matrix: alength= 4 w= 1 nsites= 20 E= 0\n"
        "0.97 0.01 0.01 0.01\n"
    )
    rna_map = root / "cisbp_map.tsv"
    rna_map.write_text("motif_id\trbp_gene\tgene_id\nM_RNA\tRBP1\tENSGRBP\n")
    rna_directory = root / "rna_motifs"
    rna_directory.mkdir()
    (rna_directory / "M_RNA.txt").write_text(
        "Pos\tA\tC\tG\tU\n1\t0.97\t0.01\t0.01\t0.01\n"
    )
    reviewed = root / "reviewed_factor_mapping.tsv"
    pd.DataFrame(
        [
            {
                "modality": "DNA",
                "motif_id": "M_DNA",
                "factor_id": "F_DNA",
                "factor_name": "TF1",
                "activity_gene_id": "ENSGTF",
                "factor_group_id": "F_DNA",
            },
            {
                "modality": "RNA",
                "motif_id": "M_RNA",
                "factor_id": "F_RNA",
                "factor_name": "RBP1",
                "activity_gene_id": "ENSGRBP",
                "factor_group_id": "F_RNA",
            },
        ]
    ).to_csv(reviewed, sep="\t", index=False)
    donor_eligibility = root / "donor_eligibility.tsv"
    pd.DataFrame(
        {
            "donor_id": ["donor_bad", "donor_good"],
            "eligible": [False, True],
        }
    ).to_csv(donor_eligibility, sep="\t", index=False)

    manifest = root / "external_inputs.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "reference_build": "GRCh38",
                "coordinate_system_internal": "0-based-half-open",
                "sources": {
                    "rna_counts": str(rna_counts),
                    "full_rna_glue_embedding": str(glue_path),
                    "full_rna_consensus_peak_bed": str(peak_bed),
                    "full_rna_atac_peak_counts": str(peak_counts),
                    "graph_discovery_pointer": str(pointer),
                    "graph_generation": str(generation),
                    "compatibility_ec": str(ec),
                    "cell_split": str(split),
                    "reference_fasta": str(fasta),
                    "reference_fasta_index": str(fasta) + ".fai",
                    "transcript_model_gtf": str(transcript_gtf),
                    "rna_gene_gtf": str(rna_gtf),
                    "dna_motif_library": str(dna_library),
                    "dna_motif_index": str(dna_index),
                    "rna_motif_directory": str(rna_directory),
                    "rna_motif_gene_map": str(rna_map),
                },
                "derived": {
                    "fabric_context_neighbors": str(root / "neighbors.parquet")
                },
                "expected": {
                    "rna_count_shape": [n_cells, 4],
                    "glue_embedding_shape": [n_cells + len(atac_ids), 2],
                    "glue_rna_count": n_cells,
                    "glue_atac_count": len(atac_ids),
                    "atac_peak_count_shape": [len(atac_ids), len(peak_ids)],
                    "consensus_peak_count": len(peak_ids),
                },
            },
            sort_keys=False,
        )
    )
    config_values = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs" / "fabric_v1_real_fixture.yaml"
        ).read_text()
    )
    config_values["external_inputs"] = str(manifest.resolve())
    config_values["data"]["atac_neighbors"]["donor_eligibility_path"] = str(
        donor_eligibility.resolve()
    )
    config_values["motifs"]["peak_support_path"] = str(peak_support.resolve())
    config_values["factor_identity"]["reviewed_mapping"] = str(reviewed.resolve())
    config_values["motifs"]["dna_events_per_choice_cap"] = 4
    config_values["motifs"]["rna_events_per_choice_cap"] = 4
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(config_values, sort_keys=False))
    return manifest, config, reviewed, donor_eligibility, peak_support


def _write_chr8_reference(path: Path, *, length: int) -> None:
    line = "ACGT" * 20 + "\n"
    line_count, remainder = divmod(length, 80)
    block = line * 10_000
    with path.open("w") as handle:
        handle.write(">chr8\n")
        while line_count >= 10_000:
            handle.write(block)
            line_count -= 10_000
        handle.write(line * line_count)
        if remainder:
            handle.write(("ACGT" * 20)[:remainder] + "\n")
