"""Elementary branch choices and data-only supervision identifiability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .graph import GeneGraph


@dataclass(frozen=True)
class Alternative:
    alternative_id: str
    edge_indices: tuple[int, ...]
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class Choice:
    choice_id: str
    gene_id: str
    entry_node_id: str
    exit_node_id: str
    scope: str
    alternatives: tuple[Alternative, ...]
    path_to_alternative: tuple[int, ...]


@dataclass(frozen=True)
class ChoiceCatalog:
    gene_id: str
    path_ids: tuple[str, ...]
    choices: tuple[Choice, ...]
    alternative_offsets: tuple[int, ...]
    path_choice_incidence: sparse.csr_matrix
    unsupported_complex: tuple[tuple[str, str, str], ...]

    @property
    def alternative_count(self) -> int:
        return self.path_choice_incidence.shape[1]


def extract_elementary_choices(graph: GeneGraph) -> ChoiceCatalog:
    """Extract deterministic single-entry/single-exit elementary bubbles.

    Every divergent entry is paired with the earliest node reached by every
    legal path that traverses that entry.  A candidate is admitted only when
    its alternatives are exact, internally unbranched and internally
    node-disjoint.  Complex or overlapping regions remain CIS-only.
    """

    edge_src = graph.edges["src_node_id"].astype(str).tolist()
    edge_dst = graph.edges["dst_node_id"].astype(str).tolist()
    outgoing: dict[str, set[int]] = {}
    incoming: dict[str, set[int]] = {}
    for edge_index, (src, dst) in enumerate(zip(edge_src, edge_dst, strict=True)):
        outgoing.setdefault(src, set()).add(edge_index)
        incoming.setdefault(dst, set()).add(edge_index)

    occurrences: dict[str, list[tuple[int, int]]] = {}
    for path_index, nodes in enumerate(graph.path_node_rows):
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"path {graph.path_ids[path_index]} repeats a node")
        for node_index, node_id in enumerate(nodes[:-1]):
            occurrences.setdefault(node_id, []).append((path_index, node_index))

    candidates: list[Choice] = []
    unsupported: list[tuple[str, str, str]] = []
    for entry in sorted(occurrences):
        path_occurrences = occurrences[entry]
        next_edges = {
            graph.path_edge_rows[path_index][node_index]
            for path_index, node_index in path_occurrences
        }
        if len(next_edges) < 2:
            continue
        downstream_sets = [
            set(graph.path_node_rows[path_index][node_index + 1 :])
            for path_index, node_index in path_occurrences
        ]
        common_exits = set.intersection(*downstream_sets)
        if not common_exits:
            unsupported.append((entry, "", "no_common_exit"))
            continue
        ranked_exits: list[tuple[int, int, str]] = []
        for exit_node in common_exits:
            distances = []
            for path_index, entry_position in path_occurrences:
                nodes = graph.path_node_rows[path_index]
                exit_position = nodes.index(exit_node, entry_position + 1)
                distances.append(exit_position - entry_position)
            ranked_exits.append((max(distances), sum(distances), exit_node))
        _, _, exit_node = min(ranked_exits)
        path_sequences: dict[int, tuple[int, ...]] = {}
        for path_index, entry_position in path_occurrences:
            nodes = graph.path_node_rows[path_index]
            exit_position = nodes.index(exit_node, entry_position + 1)
            path_sequences[path_index] = tuple(
                graph.path_edge_rows[path_index][entry_position:exit_position]
            )
        unique_sequences = sorted(
            set(path_sequences.values()),
            key=lambda values: tuple(graph.edge_ids[index] for index in values),
        )
        if len(unique_sequences) < 2:
            continue
        reason = _candidate_rejection_reason(
            unique_sequences,
            edge_src=edge_src,
            edge_dst=edge_dst,
            outgoing=outgoing,
            incoming=incoming,
        )
        if reason is not None:
            unsupported.append((entry, exit_node, reason))
            continue
        choice_id = f"choice:{graph.gene_id}:{entry}->{exit_node}"
        alternatives: list[Alternative] = []
        sequence_to_alt: dict[tuple[int, ...], int] = {}
        for alternative_index, sequence in enumerate(unique_sequences):
            node_ids = tuple(
                [edge_src[sequence[0]], *[edge_dst[index] for index in sequence]]
            )
            alternative_id = f"{choice_id}:alt:{alternative_index}"
            alternatives.append(
                Alternative(
                    alternative_id=alternative_id,
                    edge_indices=sequence,
                    edge_ids=tuple(graph.edge_ids[index] for index in sequence),
                    node_ids=node_ids,
                )
            )
            sequence_to_alt[sequence] = alternative_index
        mapping = [-1] * len(graph.path_ids)
        for path_index, sequence in path_sequences.items():
            mapping[path_index] = sequence_to_alt[sequence]
        candidates.append(
            Choice(
                choice_id=choice_id,
                gene_id=graph.gene_id,
                entry_node_id=entry,
                exit_node_id=exit_node,
                scope=_choice_scope(graph, entry, exit_node),
                alternatives=tuple(alternatives),
                path_to_alternative=tuple(mapping),
            )
        )

    admitted: list[Choice] = []
    for candidate in sorted(candidates, key=lambda value: value.choice_id):
        conflicting = [
            other
            for other in candidates
            if other is not candidate and _overlap(candidate, other)
        ]
        if conflicting:
            unsupported.append(
                (
                    candidate.entry_node_id,
                    candidate.exit_node_id,
                    "overlapping_or_nested_choice",
                )
            )
        else:
            admitted.append(candidate)

    offsets = [0]
    row_indices: list[int] = []
    column_indices: list[int] = []
    for choice in admitted:
        offset = offsets[-1]
        for path_index, alternative_index in enumerate(choice.path_to_alternative):
            if alternative_index >= 0:
                row_indices.append(path_index)
                column_indices.append(offset + alternative_index)
        offsets.append(offset + len(choice.alternatives))
    incidence = sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (row_indices, column_indices),
        ),
        shape=(len(graph.path_ids), offsets[-1]),
    )
    return ChoiceCatalog(
        gene_id=graph.gene_id,
        path_ids=graph.path_ids,
        choices=tuple(admitted),
        alternative_offsets=tuple(offsets),
        path_choice_incidence=incidence,
        unsupported_complex=tuple(sorted(set(unsupported))),
    )


def _candidate_rejection_reason(
    sequences: Sequence[tuple[int, ...]],
    *,
    edge_src: Sequence[str],
    edge_dst: Sequence[str],
    outgoing: dict[str, set[int]],
    incoming: dict[str, set[int]],
) -> str | None:
    internal_sets: list[set[str]] = []
    for sequence in sequences:
        if not sequence:
            return "empty_alternative"
        nodes = [edge_src[sequence[0]], *[edge_dst[index] for index in sequence]]
        if any(
            edge_dst[left] != edge_src[right]
            for left, right in zip(sequence[:-1], sequence[1:])
        ):
            return "discontinuous_alternative"
        internal = set(nodes[1:-1])
        for node in internal:
            if (
                len(outgoing.get(node, set())) != 1
                or len(incoming.get(node, set())) != 1
            ):
                return "internal_branch_or_merge"
        internal_sets.append(internal)
    for left in range(len(internal_sets)):
        for right in range(left + 1, len(internal_sets)):
            if internal_sets[left] & internal_sets[right]:
                return "shared_internal_node"
    return None


def _overlap(left: Choice, right: Choice) -> bool:
    left_edges = {edge for alt in left.alternatives for edge in alt.edge_indices}
    right_edges = {edge for alt in right.alternatives for edge in alt.edge_indices}
    if left_edges & right_edges:
        return True
    left_internal = {node for alt in left.alternatives for node in alt.node_ids[1:-1]}
    right_internal = {node for alt in right.alternatives for node in alt.node_ids[1:-1]}
    return bool(
        left_internal & right_internal
        or left.entry_node_id in right_internal
        or left.exit_node_id in right_internal
        or right.entry_node_id in left_internal
        or right.exit_node_id in left_internal
    )


def _choice_scope(graph: GeneGraph, entry: str, exit_node: str) -> str:
    node_types = graph.nodes.set_index(graph.nodes["node_id"].astype(str))["node_type"]
    at_start = str(node_types.loc[entry]) == "TSS"
    at_end = str(node_types.loc[exit_node]) == "PAS"
    if at_start and at_end:
        return "full_length"
    if at_start:
        return "tss"
    if at_end:
        return "pas"
    return "internal"


def zero_sum_contrast_basis(alternative_count: int) -> np.ndarray:
    if alternative_count < 2:
        raise ValueError("a choice requires at least two alternatives")
    basis = np.zeros((alternative_count, alternative_count - 1), dtype=np.float64)
    basis[:-1, :] = np.eye(alternative_count - 1)
    basis[-1, :] = -1.0
    return basis


def choice_identifiability(
    catalog: ChoiceCatalog,
    ec_rows: pd.DataFrame,
    *,
    rank_tolerance: float,
    minimum_informative_molecule_mass: float,
    minimum_alternative_support: float,
) -> pd.DataFrame:
    """Compute structure and train-supervision rank without a checkpoint."""

    if rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be positive")
    if "split" not in ec_rows or "molecule_count" not in ec_rows:
        raise ValueError("choice identifiability requires split and molecule_count")
    train = ec_rows.loc[ec_rows["split"].astype(str) == "train"].copy()
    path_index = {value: index for index, value in enumerate(catalog.path_ids)}
    compatible: list[np.ndarray] = []
    weights = train["molecule_count"].to_numpy(dtype=np.float64)
    if bool((weights <= 0).any()) or not np.isfinite(weights).all():
        raise ValueError("train EC molecule_count must be positive and finite")
    if len(train):
        if "compatible_path_ids" not in train:
            raise ValueError("train EC rows require compatible_path_ids")
        for values in train["compatible_path_ids"]:
            ids = [str(value) for value in values]
            unknown = sorted(set(ids) - set(path_index))
            if unknown:
                raise ValueError(
                    f"EC paths are absent from choice catalog: {unknown[:5]}"
                )
            compatible.append(
                np.asarray([path_index[value] for value in ids], dtype=np.int64)
            )

    rows: list[dict[str, object]] = []
    for choice_index, choice in enumerate(catalog.choices):
        start = catalog.alternative_offsets[choice_index]
        stop = catalog.alternative_offsets[choice_index + 1]
        incidence = (
            catalog.path_choice_incidence[:, start:stop].toarray().astype(np.float64)
        )
        centered = incidence - incidence.mean(axis=1, keepdims=True)
        structural_rank = int(np.linalg.matrix_rank(centered, tol=rank_tolerance))
        gene_mean = incidence.mean(axis=0)
        d = (
            np.stack(
                [incidence[index].mean(axis=0) - gene_mean for index in compatible]
            )
            if compatible
            else np.empty((0, stop - start), dtype=np.float64)
        )
        design = d @ zero_sum_contrast_basis(stop - start)
        informative = np.linalg.norm(design, axis=1) > rank_tolerance
        supervision_rank = (
            int(np.linalg.matrix_rank(design[informative], tol=rank_tolerance))
            if informative.any()
            else 0
        )
        informative_mass = float(weights[informative].sum())
        alternative_support = [
            float(weights[informative & (np.abs(d[:, alt]) > rank_tolerance)].sum())
            for alt in range(stop - start)
        ]
        eligible = (
            structural_rank == stop - start - 1
            and supervision_rank == stop - start - 1
            and informative_mass >= minimum_informative_molecule_mass
            and min(alternative_support) >= minimum_alternative_support
        )
        rows.append(
            {
                "choice_id": choice.choice_id,
                "gene_id": choice.gene_id,
                "alternative_count": stop - start,
                "structural_rank": structural_rank,
                "supervision_rank": supervision_rank,
                "informative_ec_count": int(informative.sum()),
                "informative_molecule_mass": informative_mass,
                "alternative_support": alternative_support,
                "eligible": bool(eligible),
            }
        )
    return pd.DataFrame(rows)


def choice_coverage_report(
    graphs: Sequence[GeneGraph],
    catalogs: Sequence[ChoiceCatalog],
    ec_rows: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> dict[str, float | int]:
    """Report the five F1 coverage quantities from explicit denominators."""

    graph_by_gene = {graph.gene_id: graph for graph in graphs}
    catalog_by_gene = {catalog.gene_id: catalog for catalog in catalogs}
    total_genes = len(graph_by_gene)
    choice_genes = sum(bool(catalog_by_gene[gene].choices) for gene in graph_by_gene)
    total_paths = sum(len(graph.path_ids) for graph in graphs)
    covered_paths = sum(
        int((catalog.path_choice_incidence.getnnz(axis=1) > 0).sum())
        for catalog in catalogs
    )
    total_mass = float(ec_rows["molecule_count"].sum())
    informative_rows = ec_rows["compatible_path_count"].astype(int) < ec_rows[
        "gene_id"
    ].map({gene: len(graph.path_ids) for gene, graph in graph_by_gene.items()})
    informative_mass = float(ec_rows.loc[informative_rows, "molecule_count"].sum())
    eligible_choices = (
        int(identifiability["eligible"].sum()) if len(identifiability) else 0
    )
    return {
        "gene_count": total_genes,
        "choice_gene_count": choice_genes,
        "gene_coverage": choice_genes / total_genes if total_genes else 0.0,
        "legal_path_count": total_paths,
        "choice_covered_path_count": covered_paths,
        "legal_path_coverage": covered_paths / total_paths if total_paths else 0.0,
        "informative_ec_count": int(informative_rows.sum()),
        "ec_count": int(len(ec_rows)),
        "informative_ec_coverage": float(informative_rows.mean())
        if len(ec_rows)
        else 0.0,
        "molecule_weighted_supervision_coverage": informative_mass / total_mass
        if total_mass
        else 0.0,
        "eligible_choice_count": eligible_choices,
        "choice_count": int(len(identifiability)),
        "supervision_identifiable_choice_coverage": (
            eligible_choices / len(identifiability) if len(identifiability) else 0.0
        ),
    }
