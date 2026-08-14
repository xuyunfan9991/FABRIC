"""Select the FABRIC training-gene catalog from train-only ONT observations.

The selection boundary is deliberately narrow and visible:

* split cells independently within each embryo using the frozen 80/10/10 rule;
* resolve every ONT matrix row to the matrix-matched GTF;
* admit a primary gene only when at least two distinct structural paths are
  observed in train cells on one canonical nuclear chromosome and strand;
* define the complete model isoform/path universe from the resolved ONT matrix
  rows; transcripts outside that axis do not enter FABRIC calculations;
* join DTU metadata only after the selection flags have been frozen.

This module builds a structural candidate catalog.  A later compatible-read
rebuild must still require positive train-only informative EC mass before a
gene can contribute likelihood loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


SPLIT_SEED = 20260725
SPLIT_POLICY = "embryo_stratified_cell_80_10_10_v1"
CANONICAL_CHROMS = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
SPLIT_ROW_COLUMNS = (
    "cell_id",
    "rna_embryo_id",
    "split",
    "split_policy",
    "split_seed",
    "stratum",
    "group_key",
    "stable_key_sha256",
)
_ATTR_RE = re.compile(r"([^\s;]+)\s+\"([^\"]*)\"")
_NOVEL_NUMBER_RE = re.compile(r"(?:^nrg_nr_|[-_]nr-)(\d+)$")
_EMBEDDED_ENST_RE = re.compile(r"_nr-(ENST\d+)$")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_cell_hash(*, split_seed: int, embryo_id: str, cell_id: str) -> str:
    payload = {
        "split_seed": split_seed,
        "rna_embryo_id": embryo_id,
        "cell_id": cell_id,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_embryo_stratified_split(
    matrix_barcodes: Iterable[str], *, split_seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the exact ADR-0018 cell-level split on the full ONT cell axis."""

    barcodes = pd.Series(list(matrix_barcodes), dtype="string", name="matrix_barcode")
    if barcodes.empty or bool(barcodes.isna().any()) or bool(barcodes.str.strip().eq("").any()):
        raise ValueError("ONT barcodes must be nonempty strings")
    if bool(barcodes.duplicated().any()):
        raise ValueError("ONT barcode axis contains duplicates")

    embryo = barcodes.str.extract(r"^(Emb\d{2})_", expand=False)
    if bool(embryo.isna().any()):
        examples = barcodes[embryo.isna()].head().tolist()
        raise ValueError(f"cannot parse embryo ID from ONT barcodes: {examples}")

    matrix_index = pd.DataFrame(
        {
            "matrix_column_0based": np.arange(len(barcodes), dtype=np.int64),
            "matrix_barcode": barcodes,
            "cell_id": "RNA__" + barcodes,
            "rna_embryo_id": embryo,
        }
    )
    matrix_index["stable_key_sha256"] = [
        stable_cell_hash(
            split_seed=split_seed,
            embryo_id=str(embryo_id),
            cell_id=str(cell_id),
        )
        for cell_id, embryo_id in zip(
            matrix_index["cell_id"], matrix_index["rna_embryo_id"]
        )
    ]

    assigned: list[pd.DataFrame] = []
    for _, group in matrix_index.groupby("rna_embryo_id", sort=True):
        ordered = group.sort_values(
            ["stable_key_sha256", "cell_id"], kind="mergesort"
        ).copy()
        n_val = math.floor(0.1 * len(ordered))
        n_test = math.floor(0.1 * len(ordered))
        n_train = len(ordered) - n_val - n_test
        ordered["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        assigned.append(ordered)
    matrix_index = pd.concat(assigned, ignore_index=True).sort_values(
        "matrix_column_0based", kind="mergesort"
    )

    split_rows = matrix_index[
        ["cell_id", "rna_embryo_id", "split", "stable_key_sha256"]
    ].copy()
    split_rows["split_policy"] = SPLIT_POLICY
    split_rows["split_seed"] = int(split_seed)
    split_rows["stratum"] = "rna_embryo_id"
    split_rows["group_key"] = "cell_id"
    split_rows = split_rows.loc[:, SPLIT_ROW_COLUMNS].sort_values(
        "cell_id", kind="mergesort"
    ).reset_index(drop=True)

    policy = {
        "schema_version": "prism.embryo_stratified_cell_split_policy.v1",
        "protocol": SPLIT_POLICY,
        "seed": int(split_seed),
        "stratum": "rna_embryo_id",
        "group_key": "cell_id",
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "stable_key_fields": ["split_seed", "rna_embryo_id", "cell_id"],
        "stable_key_encoder": "prism.artifact_contracts.canonical_json_bytes",
        "allocation": {
            "n_val": "floor(0.1*n)",
            "n_test": "floor(0.1*n)",
            "n_train": "n-n_val-n_test",
            "assignment_order": ["train", "val", "test"],
        },
    }
    split_counts = {
        name: int((split_rows["split"] == name).sum())
        for name in ("train", "val", "test")
    }
    stratum_counts = []
    for embryo_id, group in split_rows.groupby("rna_embryo_id", sort=True):
        stratum_counts.append(
            {
                "rna_embryo_id": str(embryo_id),
                "total": int(len(group)),
                **{
                    name: int((group["split"] == name).sum())
                    for name in ("train", "val", "test")
                },
            }
        )
    identity = {
        "schema_version": "prism.split_manifest.v2",
        "rows_uri": "split_rows.parquet",
        "policy": policy,
        "row_count": len(split_rows),
        "cell_id_hash": content_sha256(split_rows["cell_id"].tolist()),
        "stratum_assignment_hash": content_sha256(
            split_rows[
                ["cell_id", "rna_embryo_id", "split", "stable_key_sha256"]
            ].to_dict("records")
        ),
        "split_counts": split_counts,
        "stratum_counts": stratum_counts,
        "rows_sha256": content_sha256(split_rows.to_dict("records")),
    }
    identity["manifest_sha256"] = content_sha256(identity)

    matrix_index = matrix_index[
        [
            "matrix_column_0based",
            "matrix_barcode",
            "cell_id",
            "rna_embryo_id",
            "split",
            "stable_key_sha256",
        ]
    ].reset_index(drop=True)
    return split_rows, matrix_index, identity


def parse_attributes(raw: str) -> dict[str, str]:
    return dict(_ATTR_RE.findall(raw))


def parse_gtf_transcripts(path: Path) -> pd.DataFrame:
    """Return transcript identities and exact 1-based closed exon chains."""

    metadata: dict[str, tuple[str, str, str]] = {}
    exons: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"invalid GTF row at {path}:{line_number}")
            feature = fields[2]
            if feature not in {"transcript", "exon"}:
                continue
            attributes = parse_attributes(fields[8])
            transcript_id = attributes.get("transcript_id")
            gene_id = attributes.get("gene_id")
            if not transcript_id or not gene_id:
                raise ValueError(
                    f"missing transcript_id/gene_id at {path}:{line_number}"
                )
            identity = (gene_id, fields[0], fields[6])
            previous = metadata.setdefault(transcript_id, identity)
            if previous != identity:
                raise ValueError(
                    f"transcript identity changes inside GTF: {transcript_id}"
                )
            if feature == "exon":
                exons[transcript_id].append((int(fields[3]), int(fields[4])))

    missing_exons = sorted(set(metadata) - set(exons))
    if missing_exons:
        raise ValueError(f"GTF transcripts without exons: {missing_exons[:5]}")
    records = []
    for transcript_id, (gene_id, chrom, strand) in metadata.items():
        exon_chain = tuple(sorted(exons[transcript_id]))
        path_signature = ";".join(f"{start}-{end}" for start, end in exon_chain)
        records.append(
            {
                "transcript_id": transcript_id,
                "gene_id": gene_id,
                "chrom": chrom,
                "strand": strand,
                "path_signature": path_signature,
                "exon_count": len(exon_chain),
            }
        )
    result = pd.DataFrame.from_records(records)
    if bool(result["transcript_id"].duplicated().any()):
        raise ValueError(f"GTF transcript IDs are not unique: {path}")
    return result.sort_values("transcript_id", kind="mergesort").reset_index(drop=True)


def normalize_transcript_name(value: str) -> str:
    """Normalize punctuation only; biological letters and digits are retained."""

    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def parse_unique_gencode_transcript_names(path: Path) -> dict[str, set[str]]:
    names: defaultdict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"invalid GENCODE GTF row at {path}:{line_number}")
            if fields[2] != "transcript":
                continue
            attributes = parse_attributes(fields[8])
            transcript_id = attributes.get("transcript_id")
            transcript_name = attributes.get("transcript_name")
            if not transcript_id or not transcript_name:
                raise ValueError(
                    f"missing transcript identity at {path}:{line_number}"
                )
            names[normalize_transcript_name(transcript_name)].add(transcript_id)
    return dict(names)


def resolve_transcript_crosswalk(
    matrix_names: Iterable[str],
    legacy_index: pd.DataFrame,
    filtered_transcript_ids: set[str],
    gencode_names: Mapping[str, set[str]],
) -> pd.DataFrame:
    """Complete the legacy partial ONT-name mapping without dropping rows."""

    names = pd.Series(list(matrix_names), dtype="string")
    required = {
        "matrix_transcript_name",
        "ont_transcript_row_0based",
        "stable_transcript_id",
    }
    missing = sorted(required - set(legacy_index.columns))
    if missing:
        raise ValueError(f"legacy transcript index misses columns: {missing}")
    legacy = legacy_index.sort_values(
        "ont_transcript_row_0based", kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = np.arange(len(names), dtype=np.int64)
    if not np.array_equal(legacy["ont_transcript_row_0based"], expected_rows):
        raise ValueError("legacy transcript index row numbers do not match matrix rows")
    if not legacy["matrix_transcript_name"].astype("string").equals(names):
        raise ValueError("legacy transcript index names do not match transcripts.tsv")

    records: list[dict[str, object]] = []
    for row_index, (name, legacy_id) in enumerate(
        zip(names.astype(str), legacy["stable_transcript_id"])
    ):
        if pd.notna(legacy_id):
            resolved = str(legacy_id)
            rule = "legacy_gencode_v32_transcript_name"
        elif match := _NOVEL_NUMBER_RE.search(name):
            resolved = f"novel_transcript_{int(match.group(1)):06d}"
            rule = "novel_numeric_suffix"
        elif match := _EMBEDDED_ENST_RE.search(name):
            resolved = match.group(1)
            rule = "embedded_enst"
        else:
            key = normalize_transcript_name(name)
            candidates = set(gencode_names.get(key, set())) & filtered_transcript_ids
            if len(candidates) != 1:
                raise ValueError(
                    "custom transcript name is not uniquely resolved by GENCODE; "
                    f"row={row_index} name={name!r} candidates={sorted(candidates)}"
                )
            resolved = next(iter(candidates))
            rule = "unique_punctuation_normalized_gencode_name"
        if resolved not in filtered_transcript_ids:
            raise ValueError(
                f"resolved transcript is absent from matrix-matched GTF: {name} -> {resolved}"
            )
        records.append(
            {
                "matrix_row_0based": row_index,
                "matrix_transcript_name": name,
                "resolved_transcript_id": resolved,
                "crosswalk_rule": rule,
            }
        )

    result = pd.DataFrame.from_records(records)
    if bool(result["resolved_transcript_id"].duplicated().any()):
        duplicates = result.loc[
            result["resolved_transcript_id"].duplicated(keep=False),
            "resolved_transcript_id",
        ].head().tolist()
        raise ValueError(f"multiple ONT rows resolve to one transcript: {duplicates}")
    resolved_set = set(result["resolved_transcript_id"])
    if resolved_set != filtered_transcript_ids:
        raise ValueError(
            "resolved ONT transcript set differs from matrix-matched GTF; "
            f"missing={len(filtered_transcript_ids - resolved_set)} "
            f"extra={len(resolved_set - filtered_transcript_ids)}"
        )
    return result


def transcript_train_support(
    matrix: sparse.spmatrix, train_columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Calculate raw molecule counts and positive train cells per matrix row."""

    csr = matrix.tocsr(copy=True)
    input_nnz = int(matrix.nnz)
    csr.sum_duplicates()
    if csr.nnz != input_nnz:
        raise ValueError("ONT MatrixMarket contains duplicate transcript-cell coordinates")
    if not np.issubdtype(csr.dtype, np.integer):
        raise ValueError(f"ONT matrix must contain integer raw counts, observed={csr.dtype}")
    if csr.nnz == 0 or bool(np.any(csr.data <= 0)):
        raise ValueError("ONT matrix stored entries must be positive raw counts")
    train = csr[:, np.asarray(train_columns, dtype=np.int64)].tocsr()
    raw_count = np.asarray(train.sum(axis=1)).ravel().astype(np.int64)
    positive_cells = np.diff(train.indptr).astype(np.int64)
    return raw_count, positive_cells, csr


def assign_selection_tiers(gene_audit: pd.DataFrame) -> pd.DataFrame:
    """Freeze ONT-first tiers before any DTU metadata are joined."""

    result = gene_audit.copy()
    observed = result["train_observed_matrix_path_count"]
    canonical = result["location_class"].eq("canonical_nuclear")
    result["selected_ont_training_catalog"] = canonical & observed.ge(2)
    result["alt_contig_conditional_candidate"] = (
        result["location_class"].eq("nuclear_alt_contig") & observed.gt(2)
    )
    result["mitochondrial_audit_candidate"] = (
        result["location_class"].eq("mitochondrial") & observed.gt(2)
    )

    reason = np.full(len(result), "excluded_fewer_than_two_train_observed_ont_paths", dtype=object)
    reason[result["mitochondrial_audit_candidate"]] = "audit_only_mitochondrial_ge3_ont_paths"
    reason[result["alt_contig_conditional_candidate"]] = "conditional_alt_contig_ge3_ont_paths"
    reason[result["selected_ont_training_catalog"]] = (
        "selected_canonical_nuclear_ge2_train_observed_ont_paths"
    )
    result["selection_reason"] = reason
    result["likelihood_admission_status"] = np.where(
        result["selected_ont_training_catalog"],
        "pending_positive_train_informative_ec_mass",
        "not_in_primary_ont_structural_catalog",
    )
    return result


def _location_table(transcripts: pd.DataFrame) -> pd.DataFrame:
    records = []
    for gene_id, group in transcripts.groupby("gene_id", sort=True):
        chroms = sorted(set(group["chrom"]))
        strands = sorted(set(group["strand"]))
        if len(chroms) == 1 and len(strands) == 1 and chroms[0] in CANONICAL_CHROMS:
            location = "canonical_nuclear"
        elif len(chroms) == 1 and len(strands) == 1 and chroms[0] == "chrM":
            location = "mitochondrial"
        else:
            location = "nuclear_alt_contig"
        records.append(
            {
                "gene_id": gene_id,
                "chrom": "|".join(chroms),
                "strand": "|".join(strands),
                "location_class": location,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_gene_audit(
    transcript_audit: pd.DataFrame,
    embryo_path_presence: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    grouped = transcript_audit.groupby("gene_id", sort=True)
    gene = grouped.agg(
        matrix_path_count=("path_signature", "nunique"),
        train_observed_matrix_path_count=("train_observed", "sum"),
        train_total_raw_count=("train_raw_count", "sum"),
        matrix_transcript_count=("resolved_transcript_id", "size"),
    ).reset_index()
    if not bool((gene["matrix_path_count"] == gene["matrix_transcript_count"]).all()):
        raise ValueError("matrix-matched GTF contains duplicate structural path aliases")
    gene = gene.drop(columns="matrix_transcript_count")

    gene = gene.merge(
        _location_table(transcript_audit), on="gene_id", how="left", validate="one_to_one"
    )

    ranked = transcript_audit.sort_values(
        ["gene_id", "train_raw_count", "resolved_transcript_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["support_rank"] = ranked.groupby("gene_id", sort=False).cumcount() + 1
    for rank, prefix in ((1, "dominant"), (2, "second"), (3, "third")):
        selected = ranked.loc[
            ranked["support_rank"].eq(rank),
            ["gene_id", "resolved_transcript_id", "train_raw_count", "train_positive_cells"],
        ].rename(
            columns={
                "resolved_transcript_id": f"{prefix}_path_id",
                "train_raw_count": f"{prefix}_path_raw_count",
                "train_positive_cells": f"{prefix}_path_positive_cells",
            }
        )
        gene = gene.merge(selected, on="gene_id", how="left", validate="one_to_one")
        gene[f"{prefix}_path_raw_count"] = gene[
            f"{prefix}_path_raw_count"
        ].astype("Int64")
        gene[f"{prefix}_path_positive_cells"] = gene[
            f"{prefix}_path_positive_cells"
        ].astype("Int64")
    gene["non_dominant_path_raw_count"] = (
        gene["train_total_raw_count"] - gene["dominant_path_raw_count"]
    )

    for threshold in (2, 5, 10):
        field = f"n_paths_with_count_and_cells_ge{threshold}"
        support = (
            transcript_audit["train_raw_count"].ge(threshold)
            & transcript_audit["train_positive_cells"].ge(threshold)
        )
        values = (
            transcript_audit.assign(_support=support)
            .groupby("gene_id", sort=True)["_support"]
            .sum()
            .rename(field)
            .reset_index()
        )
        gene = gene.merge(values, on="gene_id", how="left", validate="one_to_one")

    gene_ids = gene["gene_id"].tolist()
    gene_position = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    transcript_gene_position = transcript_audit["gene_id"].map(gene_position).to_numpy()
    breadth_ge2 = np.zeros(len(gene), dtype=np.int64)
    breadth_ge3 = np.zeros(len(gene), dtype=np.int64)
    for embryo_id in sorted(embryo_path_presence):
        observed_paths = np.bincount(
            transcript_gene_position,
            weights=embryo_path_presence[embryo_id].astype(np.int64),
            minlength=len(gene),
        )
        breadth_ge2 += observed_paths >= 2
        breadth_ge3 += observed_paths >= 3
    gene["train_embryos_with_ge2_observed_paths"] = breadth_ge2
    gene["train_embryos_with_ge3_observed_paths"] = breadth_ge3
    return assign_selection_tiers(gene)


def _read_axis(path: Path) -> list[str]:
    values = [line.rstrip("\n") for line in path.open()]
    if not values or any(not value for value in values):
        raise ValueError(f"axis sidecar is empty or contains blank IDs: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"axis sidecar contains duplicate IDs: {path}")
    return values


def select_ont_training_genes(args: argparse.Namespace) -> dict[str, object]:
    barcodes = _read_axis(args.barcodes)
    matrix_names = _read_axis(args.transcripts)
    split_rows, matrix_cell_index, split_identity = build_embryo_stratified_split(
        barcodes, split_seed=args.split_seed
    )

    filtered_gtf = parse_gtf_transcripts(args.matrix_gtf)
    filtered_ids = set(filtered_gtf["transcript_id"])
    legacy_index = pd.read_parquet(args.legacy_transcript_index)
    gencode_names = parse_unique_gencode_transcript_names(args.gencode_gtf)
    crosswalk = resolve_transcript_crosswalk(
        matrix_names, legacy_index, filtered_ids, gencode_names
    )

    matrix = mmread(args.matrix)
    if not sparse.issparse(matrix):
        raise ValueError("ONT MatrixMarket input must be sparse")
    if matrix.shape != (len(matrix_names), len(barcodes)):
        raise ValueError(
            "ONT matrix shape does not match sidecars; "
            f"matrix={matrix.shape} rows={len(matrix_names)} cols={len(barcodes)}"
        )
    input_coordinate_nnz = int(matrix.nnz)
    train_columns = matrix_cell_index.loc[
        matrix_cell_index["split"].eq("train"), "matrix_column_0based"
    ].to_numpy(dtype=np.int64)
    train_raw_count, train_positive_cells, matrix_csr = transcript_train_support(
        matrix, train_columns
    )
    del matrix

    transcript_audit = crosswalk.merge(
        filtered_gtf.rename(columns={"transcript_id": "resolved_transcript_id"}),
        on="resolved_transcript_id",
        how="left",
        validate="one_to_one",
    )
    transcript_audit["train_raw_count"] = train_raw_count
    transcript_audit["train_positive_cells"] = train_positive_cells
    transcript_audit["train_observed"] = (
        transcript_audit["train_raw_count"].gt(0)
        & transcript_audit["train_positive_cells"].gt(0)
    )
    embryo_path_presence: dict[str, np.ndarray] = {}
    for embryo_id, cells in matrix_cell_index.loc[
        matrix_cell_index["split"].eq("train")
    ].groupby("rna_embryo_id", sort=True):
        columns = cells["matrix_column_0based"].to_numpy(dtype=np.int64)
        embryo_matrix = matrix_csr[:, columns].tocsr()
        embryo_path_presence[str(embryo_id)] = np.diff(embryo_matrix.indptr) > 0

    gene_audit = _build_gene_audit(transcript_audit, embryo_path_presence)

    # DTU is attached only after all selection flags have been determined.
    dtu = pd.read_excel(args.dtu_workbook)
    required_dtu = {
        "gene_id",
        "gene_name",
        "number_of_transcripts",
        "number_of_expressed_cell_types",
        "number_of_dominant_transcripts",
        "DTU_score",
        "top_DTU_gene",
    }
    if set(dtu.columns) != required_dtu:
        raise ValueError(f"unexpected DTU workbook columns: {dtu.columns.tolist()}")
    if bool(dtu["gene_id"].duplicated().any()):
        raise ValueError("DTU workbook gene IDs are not unique")
    if set(dtu["gene_id"]) != set(gene_audit["gene_id"]):
        raise ValueError("DTU workbook gene universe differs from matrix-matched GTF")
    gene_audit = gene_audit.merge(dtu, on="gene_id", how="left", validate="one_to_one")
    if not bool(
        gene_audit["number_of_transcripts"].eq(gene_audit["matrix_path_count"]).all()
    ):
        raise ValueError("DTU transcript counts differ from matrix-matched GTF paths")
    gene_audit["top_DTU_gene"] = gene_audit["top_DTU_gene"].str.lower().eq("yes")

    selected = gene_audit.loc[gene_audit["selected_ont_training_catalog"]].copy()
    output_columns = [
        "gene_id",
        "gene_name",
        "chrom",
        "strand",
        "matrix_path_count",
        "train_observed_matrix_path_count",
        "train_total_raw_count",
        "dominant_path_id",
        "dominant_path_raw_count",
        "second_path_id",
        "second_path_raw_count",
        "third_path_id",
        "third_path_raw_count",
        "non_dominant_path_raw_count",
        "train_embryos_with_ge2_observed_paths",
        "train_embryos_with_ge3_observed_paths",
        "n_paths_with_count_and_cells_ge2",
        "n_paths_with_count_and_cells_ge5",
        "n_paths_with_count_and_cells_ge10",
        "DTU_score",
        "top_DTU_gene",
        "selection_reason",
        "likelihood_admission_status",
    ]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows.to_parquet(output_dir / "split_rows.parquet", index=False)
    matrix_cell_index.to_parquet(output_dir / "matrix_cell_index.parquet", index=False)
    transcript_audit.to_parquet(output_dir / "transcript_crosswalk_audit.parquet", index=False)
    gene_audit.to_parquet(output_dir / "gene_selection_audit.parquet", index=False)
    selected[output_columns].sort_values("gene_id", kind="mergesort").to_csv(
        output_dir / "selected_ont_training_gene_catalog.tsv", sep="\t", index=False
    )
    split_summary = pd.DataFrame(split_identity["stratum_counts"])
    split_summary.to_csv(output_dir / "split_summary.tsv", sep="\t", index=False)

    rule_counts = {
        str(key): int(value)
        for key, value in crosswalk["crosswalk_rule"].value_counts().sort_index().items()
    }
    high_dtu = gene_audit["top_DTU_gene"]
    summary: dict[str, object] = {
        "schema_version": "fabric.ont_gene_selection.v3",
        "inputs": {
            "matrix": str(args.matrix.resolve()),
            "transcripts": str(args.transcripts.resolve()),
            "barcodes": str(args.barcodes.resolve()),
            "legacy_transcript_index": str(args.legacy_transcript_index.resolve()),
            "matrix_matched_gtf": str(args.matrix_gtf.resolve()),
            "gencode_name_gtf": str(args.gencode_gtf.resolve()),
            "dtu_workbook": str(args.dtu_workbook.resolve()),
        },
        "split": split_identity,
        "matrix": {
            "shape": [int(matrix_csr.shape[0]), int(matrix_csr.shape[1])],
            "raw_integer_counts": True,
            "coordinate_nnz": input_coordinate_nnz,
            "deduplicated_nnz": int(matrix_csr.nnz),
            "total_raw_count": int(matrix_csr.sum()),
            "train_cells": int(len(train_columns)),
            "train_nnz": int(np.sum(train_positive_cells)),
            "train_total_raw_count": int(np.sum(train_raw_count)),
            "train_observed_transcripts": int(np.sum(train_raw_count > 0)),
            "min_transcript_train_raw_count": int(train_raw_count.min()),
            "min_transcript_train_positive_cells": int(train_positive_cells.min()),
        },
        "crosswalk": {
            "matrix_rows": len(crosswalk),
            "resolved_rows": int(crosswalk["resolved_transcript_id"].notna().sum()),
            "unique_resolved_transcripts": int(crosswalk["resolved_transcript_id"].nunique()),
            "rule_counts": rule_counts,
            "model_isoform_universe": "resolved_ont_matrix_structural_paths_only",
        },
        "gene_universe": {
            "matrix_genes": int(len(gene_audit)),
            "canonical_nuclear_genes": int(
                gene_audit["location_class"].eq("canonical_nuclear").sum()
            ),
            "nuclear_alt_contig_genes": int(
                gene_audit["location_class"].eq("nuclear_alt_contig").sum()
            ),
            "mitochondrial_genes": int(
                gene_audit["location_class"].eq("mitochondrial").sum()
            ),
        },
        "selection": {
            "primary_rule": (
                "canonical nuclear gene with at least 2 distinct matrix-matched "
                "structural paths observed in train cells"
            ),
            "selected_ont_training_gene_catalog": int(len(selected)),
            "selected_matrix_structural_paths": int(selected["matrix_path_count"].sum()),
            "exactly_two_matrix_isoform_genes_in_primary": int(
                (
                    gene_audit["location_class"].eq("canonical_nuclear")
                    & gene_audit["train_observed_matrix_path_count"].eq(2)
                ).sum()
            ),
            "alt_contig_conditional_ge3_genes": int(
                gene_audit["alt_contig_conditional_candidate"].sum()
            ),
            "mitochondrial_audit_ge3_genes": int(
                gene_audit["mitochondrial_audit_candidate"].sum()
            ),
            "high_dtu_total": int(high_dtu.sum()),
            "high_dtu_in_primary": int((high_dtu & gene_audit["selected_ont_training_catalog"]).sum()),
            "dtu_used_for_admission": False,
            "paths_outside_ont_matrix_enter_calculations": False,
            "likelihood_gene_count_status": "pending_train_compatible_ec_rebuild",
        },
        "authorization": {
            "dataset_construction_beyond_selection": "not_performed",
            "compatible_ec_rebuild": "not_performed",
            "formal_training": "not_authorized_or_started",
        },
        "artifacts": {
            "selected_gene_catalog": "selected_ont_training_gene_catalog.tsv",
            "gene_audit": "gene_selection_audit.parquet",
            "transcript_crosswalk_audit": "transcript_crosswalk_audit.parquet",
            "split_rows": "split_rows.parquet",
            "matrix_cell_index": "matrix_cell_index.parquet",
            "split_summary": "split_summary.tsv",
        },
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--legacy-transcript-index", type=Path, required=True)
    parser.add_argument("--matrix-gtf", type=Path, required=True)
    parser.add_argument("--gencode-gtf", type=Path, required=True)
    parser.add_argument("--dtu-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser


def main() -> None:
    summary = select_ont_training_genes(_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
