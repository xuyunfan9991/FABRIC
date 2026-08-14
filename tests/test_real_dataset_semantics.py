from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fabric.real_dataset import (
    _cis_sequence_scores,
    _ordered_peak_interval_arrays,
    _revcomp,
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
