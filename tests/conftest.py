from __future__ import annotations

import pandas as pd
import pytest

from fabric.graph import build_gene_graph


@pytest.fixture
def toy_gene_graph():
    gene = "ENSG_TOY"
    node_specs = [
        ("source", "SOURCE", 0),
        ("tss", "TSS", 10),
        ("entry", "donor", 20),
        ("alt_a", "acceptor", 30),
        ("alt_b", "acceptor", 40),
        ("exit", "donor", 50),
        ("acceptor_end", "acceptor", 60),
        ("pas", "PAS", 70),
        ("sink", "SINK", 80),
    ]
    nodes = pd.DataFrame(
        [
            {
                "gene_id": gene,
                "node_id": f"node:{gene}:{node_type}:{name}",
                "node_type": node_type,
                "chrom": "chr1",
                "strand": "+",
                "pos_0based": position,
                "site_start_0based": position,
                "site_end_0based": position + 1,
                "relative_gene_pos": position / 80,
                "annotation_confidence": 1.0,
                "site_prior_score": 0.0,
            }
            for name, node_type, position in node_specs
        ]
    )
    node_id = {
        name: f"node:{gene}:{node_type}:{name}" for name, node_type, _ in node_specs
    }
    edge_specs = [
        ("start", "START", "source", "tss"),
        ("pre", "EXON_CONTINUATION", "tss", "entry"),
        ("splice_a", "SPLICE", "entry", "alt_a"),
        ("exon_a", "EXON_CONTINUATION", "alt_a", "exit"),
        ("splice_b", "SPLICE", "entry", "alt_b"),
        ("exon_b", "EXON_CONTINUATION", "alt_b", "exit"),
        ("splice_end", "SPLICE", "exit", "acceptor_end"),
        ("last_exon", "EXON_CONTINUATION", "acceptor_end", "pas"),
        ("end", "END", "pas", "sink"),
    ]
    node_type = nodes.set_index("node_id")["node_type"].to_dict()
    node_pos = nodes.set_index("node_id")["pos_0based"].to_dict()
    edges = []
    for name, edge_type, src_name, dst_name in edge_specs:
        src, dst = node_id[src_name], node_id[dst_name]
        start, end = sorted((int(node_pos[src]), int(node_pos[dst])))
        edges.append(
            {
                "gene_id": gene,
                "edge_id": f"edge:{gene}:{name}",
                "edge_type": edge_type,
                "src_node_id": src,
                "dst_node_id": dst,
                "src_node_type": node_type[src],
                "dst_node_type": node_type[dst],
                "chrom": "chr1",
                "strand": "+",
                "start_0based": start,
                "end_0based_exclusive": end,
                "span_bp": end - start,
                "length_bp": 0 if edge_type in {"START", "END"} else end - start,
                "relative_edge_pos": start / 80,
                "annotation_confidence": 1.0,
                "edge_prior_score": 0.0,
            }
        )
    edges = pd.DataFrame(edges)
    edge_id = {name: f"edge:{gene}:{name}" for name, *_ in edge_specs}
    path_specs = [
        (
            "p0",
            ["start", "pre", "splice_a", "exon_a", "splice_end", "last_exon", "end"],
        ),
        (
            "p1",
            ["start", "pre", "splice_b", "exon_b", "splice_end", "last_exon", "end"],
        ),
    ]
    paths = pd.DataFrame(
        [
            {
                "gene_id": gene,
                "path_id": path_name,
                "transcript_id": f"tx_{path_name}",
                "chrom": "chr1",
                "strand": "+",
                "tss_node_id": node_id["tss"],
                "pas_node_id": node_id["pas"],
                "n_edges": len(sequence),
                "path_length_bp": 70,
            }
            for path_name, sequence in path_specs
        ]
    )
    edge_by_name = {
        name: row
        for (name, *_), (_, row) in zip(edge_specs, edges.iterrows(), strict=True)
    }
    path_edges = []
    for path_name, sequence in path_specs:
        for order, name in enumerate(sequence):
            row = edge_by_name[name]
            path_edges.append(
                {
                    "gene_id": gene,
                    "path_id": path_name,
                    "transcript_id": f"tx_{path_name}",
                    "edge_order": order,
                    "edge_id": edge_id[name],
                    "edge_type": row.edge_type,
                    "src_node_id": row.src_node_id,
                    "dst_node_id": row.dst_node_id,
                    "chrom": "chr1",
                    "strand": "+",
                }
            )
    return build_gene_graph(
        gene,
        nodes=nodes,
        edges=edges,
        paths=paths,
        path_edges=pd.DataFrame(path_edges),
    )
