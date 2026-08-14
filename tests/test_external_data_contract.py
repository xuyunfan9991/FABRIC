from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabric.dataset import CompatibilityArtifactValidation, require_real_v2_compatibility_admission
from fabric.graph import canonical_rna_cell_id, load_split_rows


def test_rna_namespace_is_exact_and_split_collisions_fail_closed(tmp_path: Path):
    assert canonical_rna_cell_id("RNA__cell") == "cell"
    assert canonical_rna_cell_id("XRNA__cell") == "XRNA__cell"
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(
        {
            "cell_id": ["RNA__c", "c", "RNA__v", "RNA__t"],
            "rna_embryo_id": ["e0", "e0", "e1", "e2"],
            "split": ["train", "train", "val", "test"],
        }
    ).to_parquet(split_path)
    with pytest.raises(ValueError, match="duplicate canonical"):
        load_split_rows(split_path)


def test_real_v2_guard_rejects_historical_or_pending_artifact():
    rejected = CompatibilityArtifactValidation(
        status="REJECTED",
        reasons=("historical_7198_artifact_forbidden",),
        informative_gene_ids=(),
        audit=pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="17,706"):
        require_real_v2_compatibility_admission(
            rejected, manifest_candidate_gene_count=7_198
        )
    with pytest.raises(RuntimeError, match="not admitted"):
        require_real_v2_compatibility_admission(
            rejected, manifest_candidate_gene_count=17_706
        )
