from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from fabric.ont_gene_selection import (
    assign_selection_tiers,
    build_embryo_stratified_split,
    normalize_transcript_name,
    resolve_transcript_crosswalk,
    transcript_train_support,
)


def test_embryo_split_is_order_invariant_and_exact_per_stratum() -> None:
    barcodes = [
        f"{embryo}_head_CELL{index:02d}"
        for embryo in ("Emb01", "Emb02")
        for index in range(10)
    ]
    rows, matrix_index, identity = build_embryo_stratified_split(barcodes)
    reordered, _, reordered_identity = build_embryo_stratified_split(reversed(barcodes))

    counts = (
        rows.groupby(["rna_embryo_id", "split"]).size().unstack(fill_value=0)
    )
    assert counts[["train", "val", "test"]].to_dict("index") == {
        "Emb01": {"train": 8, "val": 1, "test": 1},
        "Emb02": {"train": 8, "val": 1, "test": 1},
    }
    assert rows[["cell_id", "split"]].equals(reordered[["cell_id", "split"]])
    assert identity["manifest_sha256"] == reordered_identity["manifest_sha256"]
    assert matrix_index["matrix_column_0based"].tolist() == list(range(20))


def test_transcript_support_and_gene_selection_ignore_held_out_counts() -> None:
    # Rows 0-2 belong to gene A; row 2 is observed only in held-out cells.
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 9, 0],
                [0, 2, 0, 8],
                [0, 0, 7, 6],
                [3, 0, 0, 0],
                [0, 4, 0, 0],
                [5, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    counts, cells, _ = transcript_train_support(matrix, np.array([0, 1]))
    assert counts.tolist() == [1, 2, 0, 3, 4, 5]
    assert cells.tolist() == [1, 1, 0, 1, 1, 1]

    audit = pd.DataFrame(
        {
            "gene_id": ["A", "B", "C", "D"],
            "location_class": [
                "canonical_nuclear",
                "canonical_nuclear",
                "nuclear_alt_contig",
                "canonical_nuclear",
            ],
            "train_observed_matrix_path_count": [2, 3, 4, 1],
        }
    )
    selected = assign_selection_tiers(audit).set_index("gene_id")
    assert bool(selected.loc["B", "selected_ont_training_catalog"])
    assert bool(selected.loc["A", "selected_ont_training_catalog"])
    assert bool(selected.loc["C", "alt_contig_conditional_candidate"])
    assert not bool(selected.loc["C", "selected_ont_training_catalog"])
    assert not bool(selected.loc["D", "selected_ont_training_catalog"])


def test_complete_crosswalk_uses_explicit_name_rules() -> None:
    names = [
        "KNOWN_201",
        "nrg_nr_000177",
        "SAMD11_nr-000248",
        "MRPL20_AS1-204",
        "CUSTOM_nr-ENST00000999999",
    ]
    legacy = pd.DataFrame(
        {
            "matrix_transcript_name": names,
            "ont_transcript_row_0based": range(len(names)),
            "stable_transcript_id": ["ENST00000111111", None, None, None, None],
        }
    )
    expected = {
        "ENST00000111111",
        "novel_transcript_000177",
        "novel_transcript_000248",
        "ENST00000222222",
        "ENST00000999999",
    }
    crosswalk = resolve_transcript_crosswalk(
        names,
        legacy,
        expected,
        {normalize_transcript_name("MRPL20-AS1-204"): {"ENST00000222222"}},
    )
    assert set(crosswalk["resolved_transcript_id"]) == expected
    assert crosswalk["crosswalk_rule"].value_counts().to_dict() == {
        "novel_numeric_suffix": 2,
        "legacy_gencode_v32_transcript_name": 1,
        "unique_punctuation_normalized_gencode_name": 1,
        "embedded_enst": 1,
    }
