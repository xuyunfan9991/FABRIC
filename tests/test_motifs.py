from __future__ import annotations

import numpy as np
import pandas as pd

from fabric.choices import extract_elementary_choices
from fabric.motifs import (
    PWM,
    build_dna_peak_regions,
    build_factor_catalog,
    cap_motif_events,
    fixed_event_feature_matrix,
    scan_motif_regions,
)


def test_factor_identity_unifies_modalities_and_excludes_ambiguous(tmp_path):
    jaspar = tmp_path / "jaspar.tsv"
    jaspar.write_text("motif_id\ttf_name\nMA1\tFOO\nMA2\tFOO::BAR\n")
    cisbp = tmp_path / "cisbp.tsv"
    cisbp.write_text("motif_id\trbp_gene\tgene_id\nR1\tFOO\tENSG_FOO\n")
    result = build_factor_catalog(
        jaspar,
        cisbp,
        gene_symbol_to_id={"FOO": "ENSG_FOO", "BAR": "ENSG_BAR"},
    )
    assert len(result.factors) == 1
    factor = result.factors.iloc[0]
    assert factor.activity_gene_id == "ENSG_FOO"
    assert factor.dna_motif_ids == ["MA1"]
    assert factor.rna_motif_ids == ["R1"]
    assert result.excluded_motifs.iloc[0].motif_id == "MA2"


def test_motif_candidates_are_static_and_cap_uses_only_static_fields():
    pwm = PWM(
        motif_id="M1",
        name="factor",
        probabilities=np.array(
            [[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]], dtype=float
        ),
    )
    regions = pd.DataFrame(
        {
            "gene_id": ["g", "g"],
            "choice_id": ["c", "c"],
            "alternative_id": ["a0", "a1"],
            "chrom": ["chr1", "chr1"],
            "start_0based": [100, 100],
            "end_0based": [106, 106],
            "strand": ["+", "+"],
            "anchor_0based": [103, 103],
            "region_type": ["exon", "exon"],
            "sequence": ["ACACAC", "ACACAC"],
        }
    )
    mapping = pd.DataFrame(
        {
            "modality": ["RNA"],
            "motif_id": ["M1"],
            "factor_id": ["f"],
            "factor_group_id": ["f"],
        }
    )
    events = scan_motif_regions(
        regions,
        {"M1": pwm},
        mapping,
        modality="RNA",
        minimum_relative_score=0.95,
    )
    assert all(value == ["a0", "a1"] for value in events.relation_alternative_ids)
    selected, audit = cap_motif_events(events, events_per_choice_cap=2)
    assert len(selected) == 2
    assert bool(audit.loc[0, "cap_saturated"])


def test_shared_choice_boundary_peak_relates_to_every_alternative(toy_gene_graph):
    class AReference:
        def fetch(self, chrom, start, end, strand="+"):
            return "A" * (end - start)

    catalog = extract_elementary_choices(toy_gene_graph)
    choice = catalog.choices[0]
    positions = toy_gene_graph.nodes.set_index("node_id")["pos_0based"]
    entry = int(positions[choice.entry_node_id])
    peaks = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start_0based": [entry - 2],
            "end_0based": [entry + 3],
            "peak_id": [f"chr1:{entry - 2}-{entry + 3}"],
            "peak_support": [1.0],
        }
    )
    regions = build_dna_peak_regions(
        toy_gene_graph, catalog, peaks, AReference(), window_bp=5
    )
    assert set(regions["alternative_id"]) == {
        alternative.alternative_id for alternative in choice.alternatives
    }
    pwm = PWM("M", "factor", np.ones((2, 4), dtype=float) / 4)
    # Give the otherwise uniform motif one discriminating A-rich position.
    pwm = PWM("M", "factor", np.array([[0.97, 0.01, 0.01, 0.01]] * 2))
    mapping = pd.DataFrame(
        {
            "modality": ["DNA"],
            "motif_id": ["M"],
            "factor_id": ["f"],
            "factor_group_id": ["f"],
        }
    )
    events = scan_motif_regions(
        regions, {"M": pwm}, mapping, modality="DNA", minimum_relative_score=0.95
    )
    assert len(events)
    expected = sorted(alternative.alternative_id for alternative in choice.alternatives)
    assert all(value == expected for value in events["relation_alternative_ids"])


def test_negative_strand_dna_and_rna_hits_use_their_distinct_sequence_axes():
    pwm = PWM(
        "M",
        "factor",
        np.array(
            [[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]],
            dtype=float,
        ),
    )
    region = pd.DataFrame(
        {
            "gene_id": ["g"],
            "choice_id": ["c"],
            "alternative_id": ["a"],
            "chrom": ["chr1"],
            "start_0based": [100],
            "end_0based": [106],
            "strand": ["-"],
            "anchor_0based": [105],
            "region_type": ["peak"],
            # DNA interprets this as forward genomic sequence, while the RNA
            # call interprets the same test string as transcript-oriented.
            "sequence": ["AACAAA"],
        }
    )
    mapping = pd.DataFrame(
        {
            "modality": ["DNA", "RNA"],
            "motif_id": ["M", "M"],
            "factor_id": ["f", "f"],
            "factor_group_id": ["f", "f"],
        }
    )

    dna = scan_motif_regions(
        region,
        {"M": pwm},
        mapping,
        modality="DNA",
        minimum_relative_score=0.95,
    )
    rna = scan_motif_regions(
        region.assign(region_type="exon"),
        {"M": pwm},
        mapping,
        modality="RNA",
        minimum_relative_score=0.95,
    )

    assert len(dna) == len(rna) == 1
    assert (int(dna.iloc[0].start_0based), int(dna.iloc[0].end_0based)) == (
        101,
        103,
    )
    assert dna.iloc[0].orientation == "opposite_transcript"
    assert float(dna.iloc[0].signed_distance_bp) == 3.0
    assert (int(rna.iloc[0].start_0based), int(rna.iloc[0].end_0based)) == (
        103,
        105,
    )
    assert rna.iloc[0].orientation == "transcribed"
    assert float(rna.iloc[0].signed_distance_bp) == 1.0


def test_event_identity_is_explicit_and_unique_across_anchors_and_regions():
    pwm = PWM(
        "M",
        "factor",
        np.array(
            [[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]],
            dtype=float,
        ),
    )
    regions = pd.DataFrame(
        {
            "gene_id": ["g", "g", "g"],
            "choice_id": ["c", "c", "c"],
            "alternative_id": ["a", "a", "a"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start_0based": [100, 100, 100],
            "end_0based": [102, 102, 102],
            "strand": ["+", "+", "+"],
            "anchor_0based": [100, 101, 100],
            "region_type": ["exon", "exon", "intron"],
            "sequence": ["AC", "AC", "AC"],
        }
    )
    mapping = pd.DataFrame(
        {
            "modality": ["RNA"],
            "motif_id": ["M"],
            "factor_id": ["factor:FOO"],
            "factor_group_id": ["group:FOO"],
        }
    )

    events = scan_motif_regions(
        regions,
        {"M": pwm},
        mapping,
        modality="RNA",
        minimum_relative_score=0.95,
    )

    assert len(events) == events["event_id"].nunique() == 3
    assert events["event_id"].str.contains("factor_id=factor:FOO", regex=False).all()
    assert events["event_id"].str.contains("anchor_0based=", regex=False).all()
    assert events["event_id"].str.contains("region_type=", regex=False).all()


def test_empty_event_features_keep_the_fixed_nonempty_schema_width():
    cases = (
        ("DNA", "same_transcript", "peak", ("peak",)),
        ("RNA", "transcribed", "exon", ("exon", "intron")),
    )
    for modality, orientation, region_type, region_order in cases:
        events = pd.DataFrame(
            {
                "factor_id": ["f0"],
                "motif_score": [0.8],
                "orientation": [orientation],
                "signed_distance_bp": [25.0],
                "region_type": [region_type],
                "peak_support": [3.0],
            }
        )
        nonempty = fixed_event_feature_matrix(
            events,
            modality=modality,
            factor_order=("f0", "f1"),
            region_order=region_order,
            distance_scale_bp=100.0,
        )
        empty = fixed_event_feature_matrix(
            events.iloc[:0],
            modality=modality,
            factor_order=("f0", "f1"),
            region_order=region_order,
            distance_scale_bp=100.0,
        )
        assert nonempty.ndim == empty.ndim == 2
        assert empty.shape == (0, nonempty.shape[1])
        assert empty.dtype == nonempty.dtype == np.float32
