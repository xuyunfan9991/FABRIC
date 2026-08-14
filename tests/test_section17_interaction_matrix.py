from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
import pandas as pd

from fabric.dataset import (
    GateValues,
    RouteBaseDesign,
    build_canonical_interaction_design,
    measure_raw_interaction_support,
)


FACTORS = ("A", "B", "C")
LEVELS = ("0", "1", "2")
SUPPORT_GRID = {
    "A": {"0", "1"},
    "B": {"0", "1", "2"},
    "C": {"1", "2"},
}


def _exact_rank(matrix: np.ndarray) -> int:
    """Small exact-rational rank helper for the section 17 witnesses."""

    values = [
        [Fraction(int(value)) for value in row]
        for row in np.asarray(matrix, dtype=np.int64).tolist()
    ]
    if not values:
        return 0
    n_rows = len(values)
    n_columns = len(values[0])
    pivot_row = 0
    for column in range(n_columns):
        pivot = next(
            (row for row in range(pivot_row, n_rows) if values[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        values[pivot_row], values[pivot] = values[pivot], values[pivot_row]
        scale = values[pivot_row][column]
        values[pivot_row] = [value / scale for value in values[pivot_row]]
        for row in range(n_rows):
            if row == pivot_row or values[row][column] == 0:
                continue
            multiple = values[row][column]
            values[row] = [
                left - multiple * right
                for left, right in zip(values[row], values[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def _raw_cells(
    factors: tuple[str, ...] = FACTORS,
    levels: tuple[str, ...] = LEVELS,
) -> list[tuple[str, str]]:
    return list(itertools.product(factors, levels))


def _rectangle(
    factor_left: str,
    factor_right: str,
    level_left: str,
    level_right: str,
    *,
    factors: tuple[str, ...] = FACTORS,
    levels: tuple[str, ...] = LEVELS,
) -> np.ndarray:
    cells = _raw_cells(factors, levels)
    index = {cell: position for position, cell in enumerate(cells)}
    result = np.zeros(len(cells), dtype=np.int64)
    result[index[(factor_left, level_left)]] = 1
    result[index[(factor_left, level_right)]] = -1
    result[index[(factor_right, level_left)]] = -1
    result[index[(factor_right, level_right)]] = 1
    return result


def _manifest(
    *,
    factors: tuple[str, ...] = FACTORS,
    fields: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]]
    | None = None,
) -> dict[str, object]:
    if fields is None:
        fields = {
            "region_type": (
                LEVELS,
                tuple(itertools.combinations(LEVELS, 2)),
            )
        }
    context_fields = {
        field: {
            "raw_levels": list(levels),
            "scientific_context_pairs": [list(pair) for pair in pairs],
            "p_max": max(0, (len(factors) - 1) * (len(levels) - 1)),
        }
        for field, (levels, pairs) in fields.items()
    }
    return {
        "event_feature_manifest_identity": "section17-interaction-matrix",
        "numeric_rank_audit": {"tolerance": 1.0e-10},
        "modalities": {
            "DNA": {
                "interaction_factor_vocabulary": list(factors),
                "context_fields": context_fields,
                "padded_interaction_width": sum(
                    details["p_max"] for details in context_fields.values()
                ),
            },
            "RNA": {
                "interaction_factor_vocabulary": [],
                "context_fields": {},
                "padded_interaction_width": 0,
            },
        },
    }


def _single_field_design(
    *,
    factor_reference: str = "A",
    level_reference: str = "0",
    extra_raw_columns: tuple[np.ndarray, ...] = (),
    include_open_only: bool = False,
) -> RouteBaseDesign:
    """Construct equivalent additive base spans under test-local recodings.

    The intercept-plus-treatment bases below are only regression witnesses for
    the retired reference-dependent interaction mask.  Production V2 keeps its
    complete bias-free factor one-hot; both witnesses span that same factor
    main-effect space and therefore must yield the same canonical raw basis.
    """

    if factor_reference not in FACTORS or level_reference not in LEVELS:
        raise ValueError("fixture references must be members of the raw vocabularies")
    rows: list[dict[str, object]] = []
    factor_rows: list[str] = []
    level_rows: list[str] = []
    for factor, level in _raw_cells():
        rows.append(
            {
                "route_id": f"route:{factor}:{level}",
                "event_id": f"event:{factor}:{level}",
                "target_gene_id": f"gene:{factor}:{level}",
                "modality": "DNA",
                "gate_key_id": f"gate:{factor}:{level}",
                "factor_identity_kind": "unique",
                "interaction_factor_id": factor,
                "region_type": level,
                "anchor_type": "TSS",
                "transcript_oriented_side": "UPSTREAM",
                "distance_bin": "near",
                "orientation": "same_transcript",
            }
        )
        factor_rows.append(factor)
        level_rows.append(level)
    if include_open_only:
        rows.append(
            {
                "route_id": "route:OPEN_ONLY",
                "event_id": "event:OPEN_ONLY",
                "target_gene_id": "gene:OPEN_ONLY",
                "modality": "DNA",
                "gate_key_id": "gate:OPEN_ONLY",
                "factor_identity_kind": "accessibility_only",
                "interaction_factor_id": "OPEN_ONLY",
                "region_type": "0",
                "anchor_type": "TSS",
                "transcript_oriented_side": "UPSTREAM",
                "distance_bin": "near",
                "orientation": "NA",
            }
        )
        factor_rows.append("OPEN_ONLY")
        level_rows.append("0")

    factor_levels = (*FACTORS, "OPEN_ONLY") if include_open_only else FACTORS
    factor_columns = [np.ones(len(rows), dtype=np.int64)]
    factor_names = [f"DNA:factor-reference={factor_reference}"]
    for factor in factor_levels:
        if factor == factor_reference:
            continue
        factor_columns.append(
            np.asarray([value == factor for value in factor_rows], dtype=np.int64)
        )
        factor_names.append(f"DNA:factor={factor}|reference={factor_reference}")
    level_columns = [
        np.asarray([value == level for value in level_rows], dtype=np.int64)
        for level in LEVELS
        if level != level_reference
    ]
    level_names = [
        f"DNA:region={level}|reference={level_reference}"
        for level in LEVELS
        if level != level_reference
    ]
    padded_extras = []
    for values in extra_raw_columns:
        values = np.asarray(values, dtype=np.int64)
        if values.shape != (len(FACTORS) * len(LEVELS),):
            raise ValueError("extra raw fixture column has the wrong cell axis")
        if include_open_only:
            values = np.append(values, 0)
        padded_extras.append(values)
    values = np.column_stack([*factor_columns, *level_columns, *padded_extras])
    names = [
        *factor_names,
        *level_names,
        *(f"DNA:rank-audit-extra={index}" for index in range(len(padded_extras))),
    ]
    assert np.linalg.matrix_rank(values) == values.shape[1]
    context = pd.DataFrame(rows)
    return RouteBaseDesign(
        route_ids=tuple(context["route_id"]),
        values=values.astype(np.float32),
        column_names=tuple(names),
        manifest=_manifest(),
        route_context=context,
    )


def _support(
    grid: dict[str, set[str]],
    *,
    factors: tuple[str, ...] = FACTORS,
    levels: tuple[str, ...] = LEVELS,
    field: str = "region_type",
    include_open_only: bool = False,
) -> pd.DataFrame:
    rows = [
        {
            "modality": "DNA",
            "context_field": field,
            "factor_entity_id": factor,
            "context_level": level,
            "raw_cell_supported": level in grid.get(factor, set()),
        }
        for factor, level in _raw_cells(factors, levels)
    ]
    if include_open_only:
        rows.append(
            {
                "modality": "DNA",
                "context_field": field,
                "factor_entity_id": "OPEN_ONLY",
                "context_level": levels[0],
                "raw_cell_supported": True,
            }
        )
    return pd.DataFrame(rows)


def _old_treatment_mask_span(
    grid: dict[str, set[str]],
    *,
    factor_reference: str,
    level_reference: str,
) -> np.ndarray:
    """Reproduce the retired four-corner treatment-column admission rule."""

    admitted: list[np.ndarray] = []
    for factor in FACTORS:
        if factor == factor_reference:
            continue
        for level in LEVELS:
            if level == level_reference:
                continue
            four_cells = (
                (factor_reference, level_reference),
                (factor_reference, level),
                (factor, level_reference),
                (factor, level),
            )
            if all(context in grid[current_factor] for current_factor, context in four_cells):
                admitted.append(
                    _rectangle(factor_reference, factor, level_reference, level)
                )
    return (
        np.column_stack(admitted)
        if admitted
        else np.zeros((len(FACTORS) * len(LEVELS), 0), dtype=np.int64)
    )


def _field_manifest(result, field: str = "region_type") -> dict[str, object]:
    return result.manifest["modalities"]["DNA"]["fields"][field]


def test_support_grid_is_reference_dependent_only_under_retired_treatment_mask():
    old_a0 = _old_treatment_mask_span(
        SUPPORT_GRID, factor_reference="A", level_reference="0"
    )
    old_c2 = _old_treatment_mask_span(
        SUPPORT_GRID, factor_reference="C", level_reference="2"
    )
    assert _exact_rank(old_a0) == _exact_rank(old_c2) == 1
    assert _exact_rank(np.column_stack([old_a0, old_c2])) == 2

    support = _support(SUPPORT_GRID, include_open_only=True)
    a0 = build_canonical_interaction_design(
        _single_field_design(
            factor_reference="A", level_reference="0", include_open_only=True
        ),
        support,
    )
    c2 = build_canonical_interaction_design(
        _single_field_design(
            factor_reference="C", level_reference="2", include_open_only=True
        ),
        support,
    )
    h_a0 = np.asarray(_field_manifest(a0)["H_support"], dtype=np.int64)
    h_c2 = np.asarray(_field_manifest(c2)["H_support"], dtype=np.int64)
    assert _exact_rank(h_a0) == _exact_rank(h_c2) == 2
    assert _exact_rank(np.column_stack([h_a0, h_c2])) == 2
    assert _exact_rank(np.column_stack([h_a0, old_a0, old_c2])) == 2

    claim_columns = [
        "raw_interaction_contrast_id",
        "row_kind",
        "factor_entity_id",
        "context_level_a",
        "context_level_b",
        "comparator_id",
        "raw_support_status",
        "contrast_in_active_span",
        "cross_field_context_separable",
        "raw_interaction_claim_status",
    ]
    pd.testing.assert_frame_equal(
        a0.raw_contrasts[claim_columns].reset_index(drop=True),
        c2.raw_contrasts[claim_columns].reset_index(drop=True),
    )
    field = _field_manifest(a0)
    assert (
        field["N_raw_rectangles_potential"],
        field["N_four_corner_supported"],
        field["N_support_span"],
        field["N_rank_retained"],
        field["N_padded"],
    ) == (9, 2, 2, 2, 4)
    assert "OPEN_ONLY" not in set(a0.raw_contrasts["factor_entity_id"])
    assert "OPEN_ONLY" not in set(a0.raw_contrasts["comparator_id"].dropna())


def test_support_span_and_combined_rank_produce_not_applicable_zero_partial_full():
    full_support = {factor: set(LEVELS) for factor in FACTORS}
    full = build_canonical_interaction_design(
        _single_field_design(), _support(full_support)
    )
    full_field = _field_manifest(full)
    assert full_field["N_support_span"] == (len(FACTORS) - 1) * (len(LEVELS) - 1)
    assert (
        full_field["N_raw_rectangles_potential"],
        full_field["N_four_corner_supported"],
        full_field["N_rank_retained"],
        full_field["N_padded"],
        full_field["basis_coverage"],
    ) == (9, 9, 4, 4, "full")

    no_rectangle_grid = {"A": {"0", "1", "2"}, "B": {"0"}, "C": {"2"}}
    no_rectangle = build_canonical_interaction_design(
        _single_field_design(), _support(no_rectangle_grid)
    )
    no_field = _field_manifest(no_rectangle)
    assert (
        no_field["N_four_corner_supported"],
        no_field["N_support_span"],
        no_field["N_rank_retained"],
        no_field["basis_coverage"],
    ) == (0, 0, 0, "not_applicable_no_supported_rectangle")

    ab01 = _rectangle("A", "B", "0", "1")
    bc12 = _rectangle("B", "C", "1", "2")
    zero = build_canonical_interaction_design(
        _single_field_design(extra_raw_columns=(ab01, bc12)),
        _support(SUPPORT_GRID),
    )
    zero_field = _field_manifest(zero)
    assert (
        zero_field["N_four_corner_supported"],
        zero_field["N_support_span"],
        zero_field["N_rank_retained"],
        zero_field["basis_coverage"],
    ) == (2, 2, 0, "zero")
    assert zero_field["column_closure_reasons"] == [
        "combined_design_rank_redundant",
        "combined_design_rank_redundant",
    ]

    partial = build_canonical_interaction_design(
        _single_field_design(extra_raw_columns=(bc12 - ab01,)),
        _support(SUPPORT_GRID),
    )
    partial_field = _field_manifest(partial)
    assert (
        partial_field["N_four_corner_supported"],
        partial_field["N_support_span"],
        partial_field["N_rank_retained"],
        partial_field["basis_coverage"],
    ) == (2, 2, 1, "partial")


def test_raw_claim_matrix_distinguishes_support_from_active_span():
    ab01 = _rectangle("A", "B", "0", "1")
    bc12 = _rectangle("B", "C", "1", "2")
    result = build_canonical_interaction_design(
        _single_field_design(extra_raw_columns=(bc12 - ab01,)),
        _support(SUPPORT_GRID),
    )
    summaries = result.raw_contrasts.loc[
        result.raw_contrasts["row_kind"].eq("q_summary")
    ]

    within = summaries.loc[
        summaries["factor_entity_id"].eq("B")
        & summaries["context_level_a"].eq("0")
        & summaries["context_level_b"].eq("2")
    ].squeeze()
    assert within["raw_support_status"] == "within_factor_only"
    assert within["raw_interaction_claim_status"] == "within_factor_only"
    assert pd.isna(within["comparator_id"])

    unsupported = summaries.loc[
        summaries["factor_entity_id"].eq("A")
        & summaries["context_level_a"].eq("1")
        & summaries["context_level_b"].eq("2")
    ].squeeze()
    assert unsupported["raw_interaction_claim_status"] == "unsupported_focal_arms"
    assert pd.isna(unsupported["comparator_id"])

    inactive_summary = summaries.loc[
        summaries["factor_entity_id"].eq("C")
        & summaries["context_level_a"].eq("1")
        & summaries["context_level_b"].eq("2")
    ].squeeze()
    assert inactive_summary["raw_support_status"] == "four_corner_covered"
    assert (
        inactive_summary["raw_interaction_claim_status"]
        == "raw_contrast_not_in_active_span"
    )
    assert inactive_summary["comparator_ids_passing_claim_gate"] == []
    assert pd.isna(inactive_summary["comparator_id"])
    comparator = result.raw_contrasts.loc[
        result.raw_contrasts["raw_interaction_contrast_id"].eq(
            inactive_summary["raw_interaction_contrast_id"]
        )
        & result.raw_contrasts["row_kind"].eq("comparator")
    ].squeeze()
    assert comparator["comparator_id"] == "B"
    assert comparator["contrast_in_active_span"] is False
    assert comparator["cross_field_context_separable"] is True


def test_one_event_keeps_each_field_and_declared_level_pair_as_a_distinct_q_record():
    factors = ("A", "B")
    regions = ("0", "1", "2")
    anchors = ("x", "y")
    rows = []
    base_rows = []
    for factor, region, anchor in itertools.product(factors, regions, anchors):
        rows.append(
            {
                "route_id": f"route:{factor}:{region}:{anchor}",
                # One physical event has routes in multiple contexts.  Its
                # applicable q records must remain keyed by field and pair.
                "event_id": f"shared-event:{factor}",
                "target_gene_id": "gene",
                "modality": "DNA",
                "gate_key_id": f"gate:{factor}",
                "factor_identity_kind": "unique",
                "interaction_factor_id": factor,
                "region_type": region,
                "anchor_type": anchor,
                "transcript_oriented_side": "UPSTREAM",
                "distance_bin": "near",
                "orientation": "same_transcript",
            }
        )
        base_rows.append(
            [
                float(factor == "A"),
                float(factor == "B"),
                float(region == "1"),
                float(region == "2"),
                float(anchor == "y"),
            ]
        )
    context = pd.DataFrame(rows)
    manifest = _manifest(
        factors=factors,
        fields={
            "region_type": (regions, tuple(itertools.combinations(regions, 2))),
            "anchor_type": (anchors, (("x", "y"),)),
        },
    )
    base = RouteBaseDesign(
        route_ids=tuple(context["route_id"]),
        values=np.asarray(base_rows, dtype=np.float32),
        column_names=(
            "DNA:factor=A",
            "DNA:factor=B",
            "DNA:region=1",
            "DNA:region=2",
            "DNA:anchor=y",
        ),
        manifest=manifest,
        route_context=context,
    )
    support = pd.concat(
        [
            _support(
                {factor: set(regions) for factor in factors},
                factors=factors,
                levels=regions,
                field="region_type",
            ),
            _support(
                {factor: set(anchors) for factor in factors},
                factors=factors,
                levels=anchors,
                field="anchor_type",
            ),
        ],
        ignore_index=True,
    )
    result = build_canonical_interaction_design(base, support)
    summaries = result.raw_contrasts.loc[
        result.raw_contrasts["row_kind"].eq("q_summary")
        & result.raw_contrasts["factor_entity_id"].eq("A")
    ]
    observed_keys = set(
        zip(
            summaries["context_field"],
            summaries["context_level_a"],
            summaries["context_level_b"],
        )
    )
    assert observed_keys == {
        ("region_type", "0", "1"),
        ("region_type", "0", "2"),
        ("region_type", "1", "2"),
        ("anchor_type", "x", "y"),
    }
    assert summaries["raw_interaction_contrast_id"].is_unique
    assert summaries["comparator_id"].isna().all()
    assert set(summaries["raw_interaction_claim_status"]) == {
        "factor_specific_grammar_estimable"
    }


def test_held_out_mass_cannot_activate_a_train_frozen_basis():
    factors = ("A", "B")
    levels = ("0", "1")
    base = _single_field_design()
    keep = base.route_context["interaction_factor_id"].isin(factors) & base.route_context[
        "region_type"
    ].isin(levels)
    context = base.route_context.loc[keep].reset_index(drop=True)
    columns = np.asarray([0, 1, 3], dtype=np.int64)
    small_manifest = _manifest(
        factors=factors,
        fields={"region_type": (levels, (("0", "1"),))},
    )
    small = RouteBaseDesign(
        route_ids=tuple(context["route_id"]),
        values=base.values[np.ix_(keep.to_numpy(), columns)],
        column_names=tuple(base.column_names[index] for index in columns),
        manifest=small_manifest,
        route_context=context,
    )
    assert np.linalg.matrix_rank(small.values) == small.values.shape[1]

    events = pd.DataFrame(
        {
            "event_id": context["event_id"],
            "model_active": True,
        }
    )
    gate_ids = tuple(context["gate_key_id"])
    gates = GateValues(
        cell_ids=("train", "heldout"),
        gate_key_ids=gate_ids,
        raw=np.ones((2, len(gate_ids))),
        standardized_residual=np.ones((2, len(gate_ids))),
        gate=np.ones((2, len(gate_ids))),
        observed=np.ones((2, len(gate_ids)), dtype=bool),
        out_of_train_range=np.zeros((2, len(gate_ids)), dtype=bool),
        out_of_train_quantile_support=np.zeros((2, len(gate_ids)), dtype=bool),
    )
    masses = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "target_gene_id": gene_id,
                "informative_molecule_mass": mass,
            }
            for cell_id, mass in (("train", 1.0), ("heldout", 100.0))
            for gene_id in context["target_gene_id"]
        ]
    )
    thresholds = {
        "DNA": {
            "minimum_distinct_events": 1,
            "minimum_distinct_genes": 1,
            "minimum_distinct_gate_keys": 1,
            "minimum_informative_molecules": 10,
        },
        "RNA": {
            "minimum_distinct_events": 0,
            "minimum_distinct_genes": 0,
            "minimum_distinct_gate_keys": 0,
            "minimum_informative_molecules": 0,
        },
    }
    train_support = measure_raw_interaction_support(
        small,
        events,
        gates,
        train_mask=[True, False],
        informative_molecule_mass_by_gene=masses,
        thresholds_by_channel=thresholds,
    )
    frozen = build_canonical_interaction_design(small, train_support)
    assert not train_support["raw_cell_supported"].any()
    assert _field_manifest(frozen)["basis_coverage"] == (
        "not_applicable_no_supported_rectangle"
    )
    assert frozen.manifest["validation_test_may_activate_columns"] is False

    # The held-out rows contain enough mass to pass if they were illegally
    # added to the fitting population; this witness makes the split boundary
    # substantive rather than relying only on a manifest label.
    leaked_support = measure_raw_interaction_support(
        small,
        events,
        gates,
        train_mask=[True, True],
        informative_molecule_mass_by_gene=masses,
        thresholds_by_channel=thresholds,
    )
    leaked = build_canonical_interaction_design(small, leaked_support)
    assert leaked_support["raw_cell_supported"].all()
    assert _field_manifest(leaked)["N_rank_retained"] == 1
    assert _field_manifest(frozen)["N_rank_retained"] == 0
