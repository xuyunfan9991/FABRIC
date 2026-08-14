from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fabric.real_dataset import (
    _admissible_atac_neighbors,
    _cis_sequence_scores,
    _is_jaspar_heterodimer,
    _ordered_peak_interval_arrays,
    _revcomp,
    _write_source_validation,
)


def test_cis_site_windows_are_identical_for_mirrored_transcripts(tmp_path):
    oriented = ("ACGTTGCAAGTCCTAGGATC" * 25)[:500]
    assert oriented != _revcomp(oriented)
    reference = list("N" * 1_500)
    reference[100:600] = oriented
    reference[800:1_300] = _revcomp(oriented)
    fasta = tmp_path / "mirror.fa"
    fasta.write_text(">chr1\n" + "".join(reference) + "\n")

    offsets = {"TSS": 80, "donor": 200, "acceptor": 300, "PAS": 420}
    nodes = []
    edges = []
    for strand, prefix in (("+", "plus"), ("-", "minus")):
        positions = {
            node_type: (100 + offset if strand == "+" else 1_300 - offset)
            for node_type, offset in offsets.items()
        }
        for node_type, position in positions.items():
            nodes.append(
                {
                    "node_id": f"{prefix}:{node_type}",
                    "node_type": node_type,
                    "pos_0based": position,
                }
            )
        for index, (src_type, dst_type) in enumerate(
            (("TSS", "donor"), ("donor", "acceptor"), ("acceptor", "PAS"))
        ):
            start, end = sorted((positions[src_type], positions[dst_type]))
            edges.append(
                {
                    "edge_id": f"{prefix}:{index}",
                    "src_node_id": f"{prefix}:{src_type}",
                    "dst_node_id": f"{prefix}:{dst_type}",
                    "chrom": "chr1",
                    "strand": strand,
                    "start_0based": start,
                    "end_0based_exclusive": end,
                }
            )

    scores = _cis_sequence_scores(pd.DataFrame(edges), pd.DataFrame(nodes), fasta)
    columns = [column for column in scores if column != "edge_id"]
    np.testing.assert_allclose(
        scores.iloc[:3][columns].to_numpy(dtype=np.float64),
        scores.iloc[3:][columns].to_numpy(dtype=np.float64),
        atol=0,
        rtol=0,
    )


def test_peak_query_contract_rejects_nonmonotone_interval_ends():
    valid = pd.DataFrame(
        {
            "start": [0, 10],
            "end": [10, 20],
            "peak_row_0based": [0, 1],
        }
    )
    starts, ends, rows = _ordered_peak_interval_arrays(valid)
    np.testing.assert_array_equal(starts, [0, 10])
    np.testing.assert_array_equal(ends, [10, 20])
    np.testing.assert_array_equal(rows, [0, 1])

    nested = valid.assign(end=[100, 20])
    with pytest.raises(ValueError, match="monotone starts and ends"):
        _ordered_peak_interval_arrays(nested)


def test_cis_clipped_site_window_is_unavailable_not_observed_zero(tmp_path):
    fasta = tmp_path / "boundary.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 30 + "\n")
    nodes = pd.DataFrame(
        {
            "node_id": ["donor", "acceptor"],
            "node_type": ["donor", "acceptor"],
            "pos_0based": [2, 50],
        }
    )
    edges = pd.DataFrame(
        {
            "edge_id": ["edge"],
            "src_node_id": ["donor"],
            "dst_node_id": ["acceptor"],
            "chrom": ["chr1"],
            "strand": ["+"],
            "start_0based": [2],
            "end_0based_exclusive": [50],
        }
    )
    score = _cis_sequence_scores(edges, nodes, fasta).iloc[0]
    assert not bool(score.donor_strength_available)
    assert score.donor_strength == 0.0
    assert bool(score.acceptor_strength_available)

    clipped_edge = edges.assign(start_0based=-1)
    with pytest.raises(ValueError, match="edge interval crosses"):
        _cis_sequence_scores(clipped_edge, nodes, fasta)


def test_atac_neighbors_are_filtered_before_weight_normalization():
    distances, indices, weights = _admissible_atac_neighbors(
        np.asarray([0.4, 0.8, 1.2, 2.0]),
        np.asarray([10, 11, 12, 13]),
        maximum_distance=1.0,
        temperature=0.5,
    )
    np.testing.assert_allclose(distances, [0.4, 0.8])
    np.testing.assert_array_equal(indices, [10, 11])
    assert weights.sum() == pytest.approx(1.0)
    assert len(weights) == 2


def test_jaspar_double_colon_names_are_unmodeled_heterodimers():
    assert _is_jaspar_heterodimer("Pou5f1::Sox2")
    assert not _is_jaspar_heterodimer("CEBPG(var.2)")


def test_real_build_identity_is_immutable_within_one_output_root(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "fabric.real_dataset.committed_source_identity",
        lambda *, require_clean: "source-commit",
    )
    output = tmp_path / "fresh"
    paths = {"real_dataset": output, "reference": tmp_path / "reference.fa"}
    _write_source_validation(paths, output)
    identity_path = output / "SourceValidation.json"
    first = identity_path.read_text()
    _write_source_validation(paths, output)
    assert identity_path.read_text() == first

    changed = {**paths, "reference": tmp_path / "other.fa"}
    with pytest.raises(RuntimeError, match="fresh output root"):
        _write_source_validation(changed, output)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "old.parquet").touch()
    with pytest.raises(RuntimeError, match="lacks a current build identity"):
        _write_source_validation({**paths, "real_dataset": legacy}, legacy)
