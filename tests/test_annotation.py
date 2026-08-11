from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fabric.annotation import build_rna_glue_id_map, load_external_inputs


ROOT = Path(__file__).parents[1]


def test_rna_glue_map_preserves_target_order_and_never_silently_intersects():
    mapped = build_rna_glue_id_map(
        ["cell_b", "cell_a"],
        ["RNA__cell_a", "ATAC__donor:x", "RNA__cell_b"],
    )
    assert mapped.to_dict("list") == {
        "cell_id": ["cell_b", "cell_a"],
        "glue_cell_id": ["RNA__cell_b", "RNA__cell_a"],
    }
    with pytest.raises(ValueError, match="absent"):
        build_rna_glue_id_map(["cell_a", "missing"], ["RNA__cell_a"])


def test_external_manifest_rejects_a_forbidden_historical_peak_route(tmp_path):
    raw = yaml.safe_load((ROOT / "data" / "external_inputs.yaml").read_text())
    raw["sources"]["full_rna_atac_peak_counts"] = "/tmp/legacy_756188_peak_matrix.h5ad"
    manifest = tmp_path / "external_inputs.yaml"
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="forbidden historical ATAC route"):
        load_external_inputs(manifest, require_exists=False)
