"""Configurable, drawing-neutral assumptions for preliminary rebar estimates.

This module never changes strict drawing facts.  It creates a parallel
``estimated`` takeoff whose assumptions and sensitivity values are explicit.
Printed schedules are not inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE_PATH = ROOT / "src/drawing_engine/resources/estimation_profiles/universal_estimate_v1.json"
_SYMBOLIC_HOOK_TOPOLOGY = "open_rectangular_loop_with_two_diagonal_hooks"


def load_estimation_profile(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate one versioned assumption profile."""

    profile_path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required_positive = (
        "steel_density_kg_m3",
        "stirrup_tie_inside_bend_diameter_factor",
        "stirrup_tie_centerline_radius_factor",
        "sensitivity_centerline_radius_factor",
        "hook_135_min_bar_diameters",
        "hook_135_min_mm",
        "calculation_rounding_mm",
        "fabrication_rounding_mm",
        "compound_callout_length_code_multiplier_mm",
    )
    if not profile.get("id") or any(float(profile.get(key, 0)) <= 0 for key in required_positive):
        raise ValueError(f"invalid reinforcement estimation profile: {profile_path}")
    if float(profile.get("waste_allowance_percent", -1)) < 0:
        raise ValueError(f"invalid waste allowance in reinforcement estimation profile: {profile_path}")
    expected_radius = (float(profile["stirrup_tie_inside_bend_diameter_factor"]) + 1.0) / 2.0
    if not math.isclose(expected_radius, float(profile["stirrup_tie_centerline_radius_factor"]), abs_tol=1e-9):
        raise ValueError("inside bend diameter and centerline radius factors disagree")
    return {**profile, "source_path": str(profile_path.resolve())}


def _round_increment(value: float, increment: float) -> float:
    return math.floor(value / increment + 0.5) * increment


def _unit_mass_kg_m(diameter_mm: float, density_kg_m3: float) -> float:
    return density_kg_m3 * math.pi * diameter_mm**2 / 4_000_000.0


def _mark_sort_key(mark: str) -> tuple[int, int | str]:
    return (0, int(mark)) if mark.isdigit() else (1, mark)


def _family_detail(
    family: dict[str, Any],
    details_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    details = [details_by_id[item] for item in family.get("detail_ids", []) if item in details_by_id]
    return details[0] if len(details) == 1 else None


def _symbolic_length(
    solution: dict[str, Any],
    diameter_mm: float,
    radius_factor: float,
    rounding_mm: float,
) -> tuple[float, float, float]:
    radius = radius_factor * diameter_mm
    raw = float(solution["sharp_vertex_length_mm"]) - float(solution["centerline_radius_coefficient"]) * radius
    return radius, raw, _round_increment(raw, rounding_mm)


def estimate_rebar_quantities(
    groups: list[dict[str, Any]],
    native_details: dict[str, Any],
    profile: dict[str, Any] | None = None,
    object_instance_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate only families with a drawing-derived count and fabrication shape.

    Explicit drawing dimensions win.  The profile supplies bend radius, steel
    density, rounding, and—only when unique—a diameter inherited from another
    drawing-derived transverse family in the same object scope.
    """

    profile = dict(profile or load_estimation_profile())
    details_by_id = {item["id"]: item for item in native_details.get("details", [])}
    families = native_details.get("physical_families", [])
    families_by_group = {
        family.get("source_group_id"): family
        for family in families
        if family.get("source_group_id")
    }
    group_by_id = {group["id"]: group for group in groups}
    transverse_diameters: dict[float, list[str]] = {}
    for group in groups:
        topology = group.get("topology", {}).get("family", {}).get("value")
        diameter = group.get("bar_spec", {}).get("diameter_mm", {})
        if topology == "closed_polyline" and diameter.get("value") is not None:
            transverse_diameters.setdefault(float(diameter["value"]), []).append(group["id"])
    inherited_diameter = next(iter(transverse_diameters)) if len(transverse_diameters) == 1 else None
    inherited_refs = transverse_diameters.get(inherited_diameter, []) if inherited_diameter is not None else []
    multi_object_scope = len((object_instance_graph or {}).get("instances", [])) > 1

    rows: list[dict[str, Any]] = []
    counted_detail_ids: set[str] = set()
    eligible_marks: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    density = float(profile["steel_density_kg_m3"])
    rounding = float(profile["fabrication_rounding_mm"])
    primary_radius_factor = float(profile["stirrup_tie_centerline_radius_factor"])
    sensitivity_radius_factor = float(profile["sensitivity_centerline_radius_factor"])

    for group in groups:
        family = families_by_group.get(group.get("id"))
        if not family:
            continue
        mark = str(family.get("mark"))
        eligible_marks.add(mark)
        count = group.get("quantity", {}).get("value")
        diameter_observation = group.get("bar_spec", {}).get("diameter_mm", {})
        diameter = diameter_observation.get("value")
        fabrication = group.get("fabrication", {})
        detail = _family_detail(family, details_by_id)
        solution = (detail or {}).get("fabrication_geometry_solution", {})
        length_each = fabrication.get("cutting_length_each_mm")
        length_raw = length_each
        radius = sensitivity_radius = sensitivity_length = None
        assumptions = ["steel_density_kg_m3"]
        evidence_refs = [family["id"], group["id"]]
        if detail:
            counted_detail_ids.add(detail["id"])
            evidence_refs.append(detail["id"])
        if diameter is None or count is None:
            unresolved.append({"mark": mark, "reason": "diameter or drawing-derived multiplicity is unavailable"})
            continue
        diameter = float(diameter)
        if diameter_observation.get("state") == "convention_dependent":
            if not profile.get("allow_convention_dependent_diameter"):
                unresolved.append({"mark": mark, "reason": "profile forbids convention-dependent diameter"})
                continue
            assumptions.append("convention_dependent_drawing_diameter")
        if length_each is None and solution.get("topology") == _SYMBOLIC_HOOK_TOPOLOGY:
            radius, length_raw, length_each = _symbolic_length(
                solution, diameter, primary_radius_factor, rounding
            )
            sensitivity_radius, _, sensitivity_length = _symbolic_length(
                solution, diameter, sensitivity_radius_factor, rounding
            )
            assumptions.extend(("stirrup_tie_centerline_radius_factor", "fabrication_rounding_mm"))
        if length_each is None:
            unresolved.append({"mark": mark, "reason": "drawing detail has no closed fabrication expression"})
            continue
        count = int(count)
        unit_mass = _unit_mass_kg_m(diameter, density)
        total_length_m = float(length_each) * count / 1000.0
        rows.append(
            {
                "physical_family_id": family["id"],
                "group_id": group["id"],
                "detail_id": detail.get("id") if detail else None,
                "mark": mark,
                "count": count,
                "count_state": group.get("quantity", {}).get("state"),
                "diameter_mm": diameter,
                "diameter_state": diameter_observation.get("state", "unknown"),
                "diameter_basis": "drawing group",
                "fabrication_length_each_raw_mm": round(float(length_raw), 3),
                "fabrication_length_each_mm": round(float(length_each), 3),
                "fabrication_length_total_m": round(total_length_m, 6),
                "centerline_bend_radius_mm": radius,
                "length_expression": solution.get("length_expression") or fabrication.get("equation"),
                "unit_mass_kg_m": round(unit_mass, 6),
                "mass_kg": round(total_length_m * unit_mass, 6),
                "sensitivity": (
                    {
                        "centerline_radius_factor": sensitivity_radius_factor,
                        "centerline_bend_radius_mm": sensitivity_radius,
                        "fabrication_length_each_mm": sensitivity_length,
                    }
                    if sensitivity_length is not None
                    else None
                ),
                "identity_state": family.get("constraint_status"),
                "assumptions_used": assumptions,
                "evidence_refs": evidence_refs,
            }
        )

    for family in families:
        if family.get("source_group_id"):
            continue
        multiplicity = family.get("multiplicity", {})
        count = multiplicity.get("value")
        mark = str(family.get("mark"))
        eligible_marks.add(mark)
        if count is None:
            unresolved.append({"mark": mark, "reason": "drawing-derived multiplicity is unavailable"})
            continue
        detail = _family_detail(family, details_by_id)
        callout = family.get("compound_callout_observation") or {}
        if (
            detail is None
            and callout.get("diameter_mm") is not None
            and callout.get("length_code") is not None
            and profile.get("allow_compound_callout_nominal_length")
        ):
            family_has_object_scope = bool(
                family.get("object_instance_id")
                or family.get("object_instance_ids")
                or family.get("object_scope_ids")
            )
            if multi_object_scope and not family_has_object_scope:
                unresolved.append({"mark": mark, "reason": "multiple object scopes exist and this family has no unique ownership"})
                continue
            diameter = float(callout["diameter_mm"])
            length_each = _round_increment(
                float(callout["length_code"])
                * float(profile["compound_callout_length_code_multiplier_mm"]),
                rounding,
            )
            count = int(count)
            unit_mass = _unit_mass_kg_m(diameter, density)
            total_length_m = length_each * count / 1000.0
            rows.append(
                {
                    "physical_family_id": family["id"],
                    "group_id": None,
                    "detail_id": None,
                    "mark": mark,
                    "count": count,
                    "count_state": multiplicity.get("state"),
                    "diameter_mm": diameter,
                    "diameter_state": "convention_dependent_callout_semantics",
                    "diameter_basis": callout.get("basis"),
                    "fabrication_length_each_raw_mm": length_each,
                    "fabrication_length_each_mm": length_each,
                    "fabrication_length_total_m": round(total_length_m, 6),
                    "centerline_bend_radius_mm": None,
                    "length_expression": "length_code × profile.compound_callout_length_code_multiplier_mm",
                    "unit_mass_kg_m": round(unit_mass, 6),
                    "mass_kg": round(total_length_m * unit_mass, 6),
                    "sensitivity": None,
                    "identity_state": family.get("constraint_status"),
                    "assumptions_used": [
                        "compound_callout_length_code_multiplier_mm",
                        "fabrication_rounding_mm",
                        "steel_density_kg_m3",
                    ],
                    "evidence_refs": [family["id"], *callout.get("evidence_refs", [])],
                }
            )
            continue
        if not detail:
            unresolved.append({"mark": mark, "reason": "no unique drawing detail defines fabrication geometry"})
            continue
        if detail["id"] in counted_detail_ids:
            continue
        solution = detail.get("fabrication_geometry_solution", {})
        if solution.get("topology") != _SYMBOLIC_HOOK_TOPOLOGY:
            unresolved.append({"mark": mark, "reason": "drawing detail has no closed fabrication expression"})
            continue
        family_has_object_scope = bool(
            family.get("object_instance_id")
            or family.get("object_instance_ids")
            or family.get("object_scope_ids")
        )
        if multi_object_scope and not family_has_object_scope:
            unresolved.append({"mark": mark, "reason": "multiple object scopes exist and this family has no unique ownership"})
            continue
        if inherited_diameter is None or not profile.get("allow_unique_role_diameter_inheritance"):
            unresolved.append({"mark": mark, "reason": "no unique transverse-family diameter can be inherited"})
            continue
        diameter = inherited_diameter
        radius, length_raw, length_each = _symbolic_length(solution, diameter, primary_radius_factor, rounding)
        sensitivity_radius, _, sensitivity_length = _symbolic_length(
            solution, diameter, sensitivity_radius_factor, rounding
        )
        count = int(count)
        unit_mass = _unit_mass_kg_m(diameter, density)
        total_length_m = length_each * count / 1000.0
        rows.append(
            {
                "physical_family_id": family["id"],
                "group_id": None,
                "detail_id": detail["id"],
                "mark": mark,
                "count": count,
                "count_state": multiplicity.get("state"),
                "diameter_mm": diameter,
                "diameter_state": "assumed_unique_role_inheritance",
                "diameter_basis": "unique drawing-derived closed transverse family",
                "diameter_source_refs": inherited_refs,
                "fabrication_length_each_raw_mm": round(length_raw, 3),
                "fabrication_length_each_mm": round(length_each, 3),
                "fabrication_length_total_m": round(total_length_m, 6),
                "centerline_bend_radius_mm": radius,
                "length_expression": solution["length_expression"],
                "unit_mass_kg_m": round(unit_mass, 6),
                "mass_kg": round(total_length_m * unit_mass, 6),
                "sensitivity": {
                    "centerline_radius_factor": sensitivity_radius_factor,
                    "centerline_bend_radius_mm": sensitivity_radius,
                    "fabrication_length_each_mm": sensitivity_length,
                },
                "identity_state": family.get("constraint_status"),
                "assumptions_used": [
                    "unique_role_diameter_inheritance",
                    "stirrup_tie_centerline_radius_factor",
                    "fabrication_rounding_mm",
                    "steel_density_kg_m3",
                ],
                "evidence_refs": [family["id"], detail["id"], *inherited_refs],
            }
        )

    rows.sort(key=lambda item: _mark_sort_key(item["mark"]))
    by_diameter = []
    for diameter in sorted({item["diameter_mm"] for item in rows}):
        matching = [item for item in rows if item["diameter_mm"] == diameter]
        by_diameter.append(
            {
                "diameter_mm": diameter,
                "marks": [item["mark"] for item in matching],
                "physical_bar_count": sum(item["count"] for item in matching),
                "total_length_m": round(sum(item["fabrication_length_total_m"] for item in matching), 6),
                "mass_kg": round(sum(item["mass_kg"] for item in matching), 6),
            }
        )
    unresolved_marks = sorted(
        {item["mark"] for item in unresolved}, key=_mark_sort_key
    )
    return {
        "status": "estimated_partial" if unresolved_marks else ("estimated" if rows else "unavailable"),
        "takeoff_kind": "profile_assumption_estimate",
        "state": "estimated",
        "profile": {
            key: value
            for key, value in profile.items()
            if key not in {"title"}
        },
        "families": rows,
        "by_diameter": by_diameter,
        "totals": {
            "physical_bar_count": sum(item["count"] for item in rows),
            "fabrication_length_m": round(sum(item["fabrication_length_total_m"] for item in rows), 6),
            "mass_kg": round(sum(item["mass_kg"] for item in rows), 6),
        },
        "coverage": {
            "eligible_family_marks": sorted(eligible_marks, key=_mark_sort_key),
            "estimated_family_marks": [item["mark"] for item in rows],
            "unresolved_family_marks": unresolved_marks,
            "estimated_family_count": len(rows),
            "eligible_family_count": len(eligible_marks),
        },
        "unresolved": unresolved,
        "validation": {
            "strict_drawing_takeoff_unchanged": True,
            "schedule_values_used": False,
            "all_assumptions_explicit": True,
            "waste_allowance_percent": float(profile["waste_allowance_percent"]),
            "multi_object_inheritance_blocked_without_scope": multi_object_scope,
        },
        "schedule_values_used": False,
    }
