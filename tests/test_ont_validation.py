from __future__ import annotations

import pandas as pd

from fabric.ont_validation import _expected_validation_cell_gene_axis


def test_expected_ont_scope_is_exact_unique_validation_informative_g_fit_pairs(
    tmp_path,
):
    root = tmp_path / "compatible"
    (root / "compatible_ec").mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c1", "c2"],
            "target_gene_id": ["g", "g", "g", "g", "graph_only"],
            "split": ["val", "val", "val", "train", "val"],
            "final_fate": [
                "likelihood_informative",
                "likelihood_informative",
                "matrix_catalog_compatible_uninformative",
                "likelihood_informative",
                "likelihood_informative",
            ],
        }
    ).to_parquet(root / "compatible_ec" / "part.parquet", index=False)

    axis = _expected_validation_cell_gene_axis(
        root,
        gene_order={"g": 0},
        cell_order={"c1": 0, "c2": 1},
    )
    assert axis.to_dict("records") == [
        {
            "expected_instance_order_0based": 0,
            "cell_id": "c1",
            "target_gene_id": "g",
            "gene_order_0based": 0,
            "cell_order_0based": 0,
        }
    ]
