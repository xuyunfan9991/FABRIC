from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fabric.motifs import (
    EVENT_ROUTE_COLUMNS,
    PHYSICAL_EVENT_COLUMNS,
    _cap_route_decisions,
    build_candidate_routes,
    build_factor_catalog,
    build_graph_anchor_regions,
    build_route_burden,
    collapse_physical_events,
)


def _endpoint_graph(strand: str):
    positions = {"TSS": 1_000, "PAS": 2_000}
    if strand == "-":
        positions = {"TSS": 2_000, "PAS": 1_000}
    nodes = pd.DataFrame(
        [
            {
                "node_id": f"{strand}:{node_type}",
                "node_type": node_type,
                "pos_0based": position,
                "chrom": "chr1",
                "strand": strand,
            }
            for node_type, position in positions.items()
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "edge_id": f"{strand}:edge",
                "src_node_id": f"{strand}:TSS",
                "dst_node_id": f"{strand}:PAS",
                "edge_type": "EXON_CONTINUATION",
                "start_0based": 1_000,
                "end_0based_exclusive": 2_000,
                "chrom": "chr1",
                "strand": strand,
            }
        ]
    )
    return SimpleNamespace(gene_id=f"gene:{strand}", nodes=nodes, edges=edges)


def test_site_windows_follow_transcript_direction_and_ignore_gene_bounds():
    expected = {
        "+": {"TSS": (800, 1_050), "PAS": (1_950, 2_200)},
        "-": {"TSS": (1_950, 2_200), "PAS": (800, 1_050)},
    }
    for strand in ("+", "-"):
        anchors = build_graph_anchor_regions(
            _endpoint_graph(strand),
            modality="DNA",
            site_flanks={"TSS": (200, 50), "PAS": (50, 200)},
            maximum_short_exon_bp=500,
            contig_lengths={"chr1": 3_000},
            gene_bounds=(1_000, 2_001),
        )
        sites = anchors.loc[anchors["geometry_kind"].eq("site_window")]
        observed = {
            row.anchor_type: (row.region_start, row.region_end)
            for row in sites.itertuples(index=False)
        }
        assert observed == expected[strand]
        assert not sites["region_clipped"].any()

    boundary_graph = _endpoint_graph("+")
    boundary_graph.nodes["pos_0based"] -= 980
    boundary_graph.edges["start_0based"] -= 980
    boundary_graph.edges["end_0based_exclusive"] -= 980
    clipped = build_graph_anchor_regions(
        boundary_graph,
        modality="DNA",
        site_flanks={"TSS": (100, 50)},
        maximum_short_exon_bp=500,
        contig_lengths={"chr1": 1_500},
        gene_bounds=(20, 1_021),
    )
    tss = clipped.loc[clipped["anchor_type"].eq("TSS")].iloc[0]
    assert (tss.raw_region_start, tss.raw_region_end) == (-80, 70)
    assert (tss.region_start, tss.region_end) == (0, 70)
    assert tss.region_clipped


def test_factor_catalog_rejects_empty_or_duplicate_identity_members():
    base = {
        "modality": ["DNA"],
        "motif_id": ["M"],
        "factor_identity_kind": ["factor_equivalence_group"],
        "factor_entity_id": ["group:A:B"],
        "candidate_factor_ids": [["A", "B"]],
        "activity_entity_id": ["group:A:B"],
        "activity_gene_ids": [["A", "B"]],
        "activity_proxy_rule": ["raw_sum_then_cp10k_log1p"],
    }
    result = build_factor_catalog(pd.DataFrame(base), frozen_rna_gene_axis=["A", "B"])
    assert result.factors.iloc[0].activity_gene_ids == ["A", "B"]
    for field, invalid in (
        ("candidate_factor_ids", [[]]),
        ("candidate_factor_ids", [["A", "A"]]),
        ("activity_gene_ids", [[]]),
        ("activity_gene_ids", [["A", "A"]]),
    ):
        broken = pd.DataFrame(base)
        broken[field] = invalid
        with pytest.raises(ValueError, match="(non-empty and unique|unique and non-empty)"):
            build_factor_catalog(broken, frozen_rna_gene_axis=["A", "B"])


def _source_hit(**updates):
    row = {
        "source_hit_id": "h0",
        "source_window_id": "w0",
        "target_gene_id": "g",
        "factor_entity_id": "F",
        "factor_identity_kind": "unique",
        "candidate_factor_ids": ["F"],
        "activity_entity_id": "F",
        "activity_gene_ids": ["F"],
        "activity_proxy_rule": "unique_gene_cp10k_log1p",
        "modality": "RNA",
        "motif_id": "M1",
        "motif_equivalence_family_id": "fam",
        "chromosome": "chr1",
        "start": 100,
        "end": 106,
        "strand": "+",
        "motif_score": 0.8,
        "calibrated_motif_quality": 0.7,
        "source_local_rank": 1.0,
        "source_priority": 0,
        "orientation": "transcribed",
        "peak_id": None,
        "peak_support": 0.0,
        "source_valid": True,
    }
    row.update(updates)
    return row


def test_connected_component_physical_collapse_preserves_source_provenance():
    hits = pd.DataFrame(
        [
            _source_hit(source_hit_id="h0", start=100, end=108, motif_id="M1"),
            _source_hit(source_hit_id="h1", start=106, end=114, motif_id="M2"),
            _source_hit(source_hit_id="h2", start=112, end=120, motif_id="M3"),
            _source_hit(
                source_hit_id="distinct",
                start=104,
                end=112,
                motif_id="M4",
                motif_equivalence_family_id="other",
            ),
        ]
    )
    events = collapse_physical_events(
        hits, minimum_overlap_bp=2, minimum_reciprocal_overlap=0.2
    )
    collapsed = events.loc[events["motif_equivalence_family_id"] == "fam"].iloc[0]
    assert len(events) == 2
    assert collapsed.source_motif_ids == ["M1", "M2", "M3"]
    assert len(collapsed.source_hit_coordinates) == 3


def _event(event_id: str, *, modality="RNA", quality=1.0, peak=0.0, cap_class="motif_anchored"):
    row = {column: None for column in PHYSICAL_EVENT_COLUMNS}
    row.update(
        {
            "event_id": event_id,
            "target_gene_id": "g",
            "factor_entity_id": "F",
            "factor_identity_kind": "unique",
            "cap_evidence_class": cap_class,
            "candidate_factor_ids": ["F"],
            "activity_entity_id": "F",
            "activity_gene_ids": ["F"],
            "activity_proxy_rule": "unique_gene_cp10k_log1p",
            "modality": modality,
            "motif_id": "M",
            "motif_equivalence_family_id": "fam",
            "source_motif_ids": ["M"],
            "chromosome": "chr1",
            "start": 100,
            "end": 105,
            "strand": "+",
            "source_hit_coordinates": [{"source_hit_id": event_id}],
            "motif_score": 0.5,
            "orientation": "transcribed" if modality == "RNA" else "same_transcript",
            "peak_id": "p" if modality == "DNA" else None,
            "peak_support": peak,
            "gate_key_id": f"gate:{event_id}",
            "source_valid": True,
            "has_retained_route": True,
            "gate_key_active": True,
            "model_active": True,
            "admission_reasons": [],
            "calibrated_motif_quality": quality,
            "source_local_rank": 1,
            "source_priority": 0,
        }
    )
    return row


def _route(route_id: str, event_id: str, *, anchor="a", edge="e", modality="RNA"):
    row = {column: None for column in EVENT_ROUTE_COLUMNS}
    row.update(
        {
            "route_id": route_id,
            "event_id": event_id,
            "target_gene_id": "g",
            "modality": modality,
            "anchor_region_id": anchor,
            "anchor_site_id": "site",
            "edge_id": edge,
            "route_weight": 1.0,
            "region_type": "exon",
            "anchor_type": "donor",
            "transcript_oriented_side": "UPSTREAM",
            "signed_distance_bp": 2.0,
            "edge_relative_position": np.nan,
            "distance_to_5prime_boundary_bp": np.nan,
            "distance_to_3prime_boundary_bp": np.nan,
            "geometry_kind": "site_window",
        }
    )
    return row


def test_calibrated_rna_cap_does_not_read_peak_support():
    events = pd.DataFrame(
        [_event("a", quality=1.0, peak=0), _event("b", quality=1.0, peak=999)]
    ).set_index("event_id", drop=False)
    routes = pd.DataFrame([_route("ra", "a"), _route("rb", "b")])
    selected, _ = _cap_route_decisions(routes, events, 1)
    assert selected.loc[selected["cap_selected"], "event_id"].tolist() == ["a"]


def test_burden_counts_saturated_anchor_once_for_two_evidence_classes():
    events = pd.DataFrame(
        [
            _event("motif", modality="DNA", cap_class="motif_anchored"),
            _event("open", modality="DNA", cap_class="accessibility_only"),
        ]
    )
    routes = pd.DataFrame(
        [
            _route("r1", "motif", modality="DNA"),
            _route("r2", "open", modality="DNA"),
        ]
    )
    candidates = routes.assign(cap_bucket_id=["bucket:motif", "bucket:open"])
    cap_audit = pd.DataFrame(
        {
            "cap_bucket_id": ["bucket:motif", "bucket:open"],
            "cap_saturated": [True, True],
        }
    )
    burden = build_route_burden(
        events, routes, candidates, cap_audit, audit_population="model_input"
    )
    assert burden.iloc[0].saturated_anchor_group_count == 1
    assert burden.iloc[0].saturated_cap_bucket_count == 2


def test_site_route_geometry_has_strict_overlap_na_touch_and_negative_half_center():
    events = pd.DataFrame(
        [
            _event("overlap"),
            _event("starts_at_anchor"),
            _event("touch"),
            _event("negative"),
        ]
    )
    events.loc[0, ["start", "end"]] = [99, 102]
    events.loc[1, ["start", "end"]] = [100, 106]
    events.loc[2, ["start", "end"]] = [98, 100]
    events.loc[3, ["start", "end", "strand"]] = [103, 106, "-"]
    anchors = pd.DataFrame(
        {
            "target_gene_id": ["g", "g"],
            "modality": ["RNA", "RNA"],
            "anchor_region_id": ["plus", "minus"],
            "anchor_site_id": ["s+", "s-"],
            "edge_id": ["e+", "e-"],
            "chromosome": ["chr1", "chr1"],
            "strand": ["+", "-"],
            "region_start": [90, 90],
            "region_end": [110, 110],
            "anchor_position": [100, 100],
            "region_type": ["exon", "exon"],
            "anchor_type": ["donor", "donor"],
            "geometry_kind": ["site_window", "site_window"],
        }
    )
    routes = build_candidate_routes(events, anchors)
    overlap = routes.loc[
        (routes.event_id == "overlap") & (routes.anchor_region_id == "plus")
    ].iloc[0]
    touch = routes.loc[
        (routes.event_id == "touch") & (routes.anchor_region_id == "plus")
    ].iloc[0]
    starts_at_anchor = routes.loc[
        (routes.event_id == "starts_at_anchor")
        & (routes.anchor_region_id == "plus")
    ].iloc[0]
    negative = routes.loc[
        (routes.event_id == "negative") & (routes.anchor_region_id == "minus")
    ].iloc[0]
    assert overlap.transcript_oriented_side == "OVERLAP_ANCHOR"
    assert np.isnan(overlap.signed_distance_bp)
    assert starts_at_anchor.transcript_oriented_side == "OVERLAP_ANCHOR"
    assert np.isnan(starts_at_anchor.signed_distance_bp)
    assert touch.transcript_oriented_side == "UPSTREAM"
    assert touch.signed_distance_bp == -1.0
    assert negative.transcript_oriented_side == "UPSTREAM"
    assert negative.signed_distance_bp == -4.5


def test_overlap_geometry_uses_zero_cap_proximity_without_nan_ordering():
    events = pd.DataFrame([_event("overlap"), _event("near")]).set_index(
        "event_id", drop=False
    )
    routes = pd.DataFrame(
        [
            {**_route("ro", "overlap"), "transcript_oriented_side": "OVERLAP_ANCHOR", "signed_distance_bp": np.nan},
            {**_route("rn", "near"), "signed_distance_bp": 0.5},
        ]
    )
    selected, _ = _cap_route_decisions(routes, events, 1)
    assert selected.loc[selected.cap_selected, "event_id"].tolist() == ["overlap"]


def test_cap_keeps_sixteen_per_dna_evidence_class_for_thirty_two_total():
    event_rows = []
    route_rows = []
    for cap_class in ("motif_anchored", "accessibility_only"):
        for index in range(20):
            event_id = f"{cap_class}:{index:02d}"
            event_rows.append(
                _event(
                    event_id,
                    modality="DNA",
                    quality=float(20 - index),
                    peak=float(20 - index),
                    cap_class=cap_class,
                )
            )
            route_rows.append(
                _route(f"r:{event_id}", event_id, modality="DNA")
            )
    events = pd.DataFrame(event_rows).set_index("event_id", drop=False)
    decisions, audit = _cap_route_decisions(pd.DataFrame(route_rows), events, 16)
    assert int(decisions.cap_selected.sum()) == 32
    assert audit.groupby("cap_evidence_class").selected_event_count.first().to_dict() == {
        "accessibility_only": 16,
        "motif_anchored": 16,
    }
