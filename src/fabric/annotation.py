"""External reference, coordinate, and cell-identity import boundaries.

FABRIC converts coordinates exactly once at import and then uses GRCh38
0-based half-open intervals throughout.  Functions here deliberately validate
only identities whose drift would change the scientific interpretation.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import anndata as ad
import pandas as pd
import yaml
from pyfaidx import Fasta


REFERENCE_BUILD = "GRCh38"
INTERNAL_COORDINATES = "0-based-half-open"
TRANSCRIPT_DIRECTION = "5prime_to_3prime"


@dataclass(frozen=True)
class ExternalInputs:
    """Frozen, read-only sources selected for one FABRIC data generation."""

    reference_build: str
    coordinate_system_internal: str
    sources: Mapping[str, Path]
    derived: Mapping[str, Path]
    expected: Mapping[str, object]

    def path(self, role: str) -> Path:
        try:
            return self.sources[role]
        except KeyError as exc:
            raise KeyError(f"external input role is not declared: {role}") from exc

    def derived_path(self, role: str) -> Path:
        try:
            return self.derived[role]
        except KeyError as exc:
            raise KeyError(f"derived FABRIC role is not declared: {role}") from exc


def load_external_inputs(
    path: str | Path, *, require_exists: bool = True
) -> ExternalInputs:
    """Read the single V1 external-input manifest.

    Mutable discovery pointers may be recorded for audit, but the graph input
    used by downstream code must be the resolved generation directory.
    """

    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text())
    if raw.get("reference_build") != REFERENCE_BUILD:
        raise ValueError("FABRIC V1 external inputs must use GRCh38")
    if raw.get("coordinate_system_internal") != INTERNAL_COORDINATES:
        raise ValueError("FABRIC internal coordinates must be 0-based-half-open")
    source_raw = raw.get("sources")
    if not isinstance(source_raw, dict):
        raise ValueError("external input manifest requires a sources mapping")
    sources = {str(role): Path(value) for role, value in source_raw.items()}
    required = {
        "rna_counts",
        "full_rna_glue_embedding",
        "full_rna_consensus_peak_bed",
        "full_rna_atac_peak_counts",
        "graph_discovery_pointer",
        "graph_generation",
        "compatibility_ec",
        "cell_split",
        "reference_fasta",
        "reference_fasta_index",
        "transcript_model_gtf",
        "rna_gene_gtf",
        "dna_motif_library",
        "dna_motif_index",
        "rna_motif_directory",
        "rna_motif_gene_map",
    }
    missing_roles = sorted(required - set(sources))
    if missing_roles:
        raise ValueError(f"external input manifest misses roles: {missing_roles}")
    if require_exists:
        absent = sorted(role for role, value in sources.items() if not value.exists())
        if absent:
            raise FileNotFoundError(f"external inputs do not exist for roles: {absent}")
    forbidden = ("756188", "150000", "61002", "167235-rna", "167235_rna")
    legacy = sorted(
        role
        for role, value in sources.items()
        if any(token in str(value).lower() for token in forbidden)
    )
    if legacy:
        raise ValueError(
            f"manifest selects a forbidden historical ATAC route: {legacy}"
        )
    expected = raw.get("expected", {})
    if not isinstance(expected, dict):
        raise ValueError("external input expected field must be a mapping")
    derived_raw = raw.get("derived")
    if (
        not isinstance(derived_raw, dict)
        or "fabric_context_neighbors" not in derived_raw
    ):
        raise ValueError(
            "external input manifest requires fabric_context_neighbors output"
        )
    derived = {str(role): Path(value) for role, value in derived_raw.items()}
    return ExternalInputs(
        reference_build=REFERENCE_BUILD,
        coordinate_system_internal=INTERNAL_COORDINATES,
        sources=sources,
        derived=derived,
        expected=expected,
    )


def resolve_and_validate_graph_generation(inputs: ExternalInputs) -> Path:
    """Confirm that the recorded immutable graph generation matches CURRENT."""

    pointer_path = inputs.path("graph_discovery_pointer")
    pointer = json.loads(pointer_path.read_text())
    discovered = (pointer_path.parent / str(pointer["generation"])).resolve()
    frozen = inputs.path("graph_generation").resolve()
    if discovered != frozen:
        raise ValueError(
            "graph CURRENT pointer and frozen graph_generation differ; refresh the "
            "manifest deliberately instead of following CURRENT implicitly"
        )
    contract_path = frozen / "outputs" / "graph" / "graph_artifact_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("coordinate_system") != INTERNAL_COORDINATES:
        raise ValueError("imported graph coordinate system differs from FABRIC")
    if contract.get("transcript_direction") != TRANSCRIPT_DIRECTION:
        raise ValueError("imported graph transcript direction differs from FABRIC")
    return frozen


def gtf_interval_to_internal(start_1based: int, end_1based: int) -> tuple[int, int]:
    """Convert one GTF 1-based closed interval to 0-based half-open."""

    if start_1based < 1 or end_1based < start_1based:
        raise ValueError("invalid 1-based closed GTF interval")
    return start_1based - 1, end_1based


def canonical_rna_cell_id(value: str) -> str:
    """Remove exactly the documented GLUE RNA namespace, if present."""

    value = str(value)
    if value.startswith("RNA__"):
        value = value[5:]
    if not value:
        raise ValueError("RNA cell_id must be non-empty")
    return value


def canonical_atac_cell_id(value: str) -> str:
    value = str(value)
    if value.startswith("ATAC__"):
        value = value[6:]
    if not value:
        raise ValueError("ATAC cell_id must be non-empty")
    return value


def build_rna_glue_id_map(
    rna_cell_ids: Sequence[str], glue_cell_ids: Sequence[str]
) -> pd.DataFrame:
    """Build the only permitted exact RNA↔GLUE ID mapping."""

    rna = [canonical_rna_cell_id(value) for value in rna_cell_ids]
    if len(set(rna)) != len(rna):
        raise ValueError("RNA source cell_id values are not globally unique")
    glue_rna = [str(value) for value in glue_cell_ids if str(value).startswith("RNA__")]
    canonical = [canonical_rna_cell_id(value) for value in glue_rna]
    if len(set(canonical)) != len(canonical):
        raise ValueError("GLUE RNA cell_id values are not globally unique")
    by_canonical = dict(zip(canonical, glue_rna, strict=True))
    missing = [value for value in rna if value not in by_canonical]
    if missing:
        raise ValueError(
            "target RNA cells are absent from the frozen full-RNA GLUE axis: "
            f"{missing[:10]}"
        )
    return pd.DataFrame(
        {
            "cell_id": rna,
            "glue_cell_id": [by_canonical[value] for value in rna],
        }
    )


def load_split_rows(path: str | Path) -> pd.DataFrame:
    """Load the authoritative split and canonicalize only its RNA namespace."""

    rows = pd.read_parquet(path, columns=["cell_id", "rna_embryo_id", "split"])
    rows = rows.copy()
    rows["source_split_cell_id"] = rows["cell_id"].astype(str)
    rows["cell_id"] = rows["source_split_cell_id"].map(canonical_rna_cell_id)
    if rows["cell_id"].duplicated().any():
        raise ValueError("cell split contains duplicate canonical cell_id values")
    allowed = {"train", "val", "test"}
    observed = set(rows["split"].astype(str))
    if observed != allowed:
        raise ValueError(
            f"cell split labels differ from {sorted(allowed)}: {sorted(observed)}"
        )
    return rows


def validate_cell_split_alignment(
    ec_cell_ids: Iterable[str], split_rows: pd.DataFrame
) -> dict[str, int]:
    """Require every imported supervision cell to have one authoritative split."""

    ec_ids = {canonical_rna_cell_id(value) for value in ec_cell_ids}
    split_ids = set(split_rows["cell_id"].astype(str))
    missing = sorted(ec_ids - split_ids)
    if missing:
        raise ValueError(
            f"supervision cells absent from authoritative split: {missing[:10]}"
        )
    return {
        "supervision_cell_count": len(ec_ids),
        "split_cell_count": len(split_ids),
        "matched_supervision_cell_count": len(ec_ids),
    }


def iter_peak_bed(path: str | Path) -> Iterator[tuple[str, int, int, str]]:
    """Yield the ordered full-RNA peak axis without loading it into memory."""

    with Path(path).open() as handle:
        for row_index, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"peak BED row {row_index} has fewer than three columns"
                )
            chrom, start_text, end_text = fields[:3]
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start:
                raise ValueError(f"peak BED row {row_index} has an invalid interval")
            yield chrom, start, end, f"{chrom}:{start}-{end}"


def validate_peak_axis(
    bed_path: str | Path,
    peak_h5ad_path: str | Path,
    *,
    expected_count: int | None = None,
) -> dict[str, int]:
    """Check BED and accessibility axes item by item without touching ``X``."""

    matrix = ad.read_h5ad(peak_h5ad_path, backed="r")
    try:
        var_names = matrix.var_names.astype(str)
        n_obs, n_vars = matrix.shape
        if expected_count is not None and n_vars != expected_count:
            raise ValueError(
                f"ATAC peak count differs from expected {expected_count}: {n_vars}"
            )
        bed_count = 0
        for bed_count, (_, _, _, peak_id) in enumerate(
            iter_peak_bed(bed_path), start=1
        ):
            observed = var_names[bed_count - 1]
            if observed != peak_id:
                raise ValueError(
                    f"BED/accessibility peak axis mismatch at row {bed_count - 1}: "
                    f"{peak_id!r} != {observed!r}"
                )
        if bed_count != n_vars:
            raise ValueError(f"BED has {bed_count} peaks but H5AD has {n_vars}")
    finally:
        matrix.file.close()
    return {"atac_cell_count": int(n_obs), "peak_count": int(n_vars)}


def load_gene_symbol_map(gtf_path: str | Path) -> dict[str, str]:
    """Return only unambiguous gene-symbol→Ensembl mappings from the RNA GTF."""

    candidates: dict[str, set[str]] = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = _parse_gtf_attributes(fields[8])
            gene_id = attributes.get("gene_id", "").split(".", 1)[0]
            gene_name = attributes.get("gene_name", "")
            if gene_id and gene_name:
                candidates.setdefault(gene_name.upper(), set()).add(gene_id)
    return {
        name: next(iter(gene_ids))
        for name, gene_ids in candidates.items()
        if len(gene_ids) == 1
    }


def _parse_gtf_attributes(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item or " " not in item:
            continue
        key, raw = item.split(" ", 1)
        result[key] = raw.strip().strip('"')
    return result


class ReferenceSequence:
    """Indexed GRCh38 sequence access using 0-based half-open coordinates."""

    def __init__(self, fasta_path: str | Path) -> None:
        self._fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    def fetch(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if start < 0 or end <= start:
            raise ValueError("reference interval must be non-empty and non-negative")
        try:
            sequence = str(self._fasta[chrom][start:end])
        except KeyError as exc:
            raise ValueError(f"reference contig is absent: {chrom}") from exc
        if len(sequence) != end - start:
            raise ValueError(
                f"reference interval extends beyond contig: {chrom}:{start}-{end}"
            )
        return reverse_complement(sequence) if strand == "-" else sequence

    def close(self) -> None:
        self._fasta.close()

    def __enter__(self) -> "ReferenceSequence":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTUNacgtun", "TGCAANtgcaan"))[::-1]
