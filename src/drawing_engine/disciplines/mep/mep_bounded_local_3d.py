"""M5A bounded, quantity-ineligible local MEP 3D certificates.

M5A resolves one canonical projected segment into a local XYZ centreline and
measured envelope.  It consumes accepted M1 plan scale/registration, M3.5
outlined-route composites, M4 attributes, and M5 overlap canonicalisation.
The result deliberately stops at unresolved analysis caps: it establishes no
cross-sheet continuation, complete physical run, installed length, clash, or
quantity authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import validate_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import validate_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_bounded_local_3d_segments"
METHOD_VERSION = "1.0.0"
LABEL = "bounded local geometry - continuation unresolved"
_ATTRIBUTE_TYPES = ("route_system", "route_size", "route_elevation")


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_positive(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _metres(value: object, unit: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    factors = {"m": 1.0, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
    factor = factors.get(str(unit or "").lower())
    return None if factor is None else number * factor


def _apply(matrix: Iterable[object], point: Iterable[object]) -> list[float]:
    a, b, c, d, e, f = (float(value) for value in matrix)
    x, y = (float(value) for value in point)
    return [a * x + c * y + e, b * x + d * y + f]


def _inverse(matrix: Iterable[object]) -> list[float] | None:
    a, b, c, d, e, f = (float(value) for value in matrix)
    determinant = a * d - b * c
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    return [
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
        (c * f - d * e) / determinant,
        (b * e - a * f) / determinant,
    ]


def _semantic(value: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            key: item
            for key, item in value.items()
            if key not in {"raw_text", "terminology_entry_ref"}
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _walk_items(
    value: object, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk_items(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, (*path, f"[{index}]"))


def _upstream_errors(
    sheet_registry: Mapping[str, Any],
    outlined_route_composites: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    cross_sheet_runs: Mapping[str, Any],
) -> list[str]:
    checks = (
        ("M1", validate_mep_sheet_registry(sheet_registry)),
        ("M3.5", validate_mep_outlined_route_composites(outlined_route_composites)),
        ("M4", validate_mep_attribute_bindings(attribute_bindings)),
        ("M5", validate_mep_cross_sheet_runs(cross_sheet_runs)),
    )
    return [f"{layer}: {error}" for layer, rows in checks for error in rows]


def _page_scale(page: Mapping[str, Any]) -> tuple[float | None, list[str]]:
    field = page.get("fields", {}).get("scale", {})
    scale = _finite_positive(field.get("drawing_inches_per_paper_inch"))
    if scale is None or field.get("state") not in {"observed", "derived"}:
        return None, []
    return scale * 0.0254 / 72.0, sorted(str(ref) for ref in field.get("evidence_refs", []))


def _direct_transforms(
    sheet_registry: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sheet_registry.get("adjoining_sheet_transforms", []):
        if row.get("state") != "accepted":
            continue
        matrix = row.get("matrix_source_display_to_target_display")
        if not isinstance(matrix, list) or len(matrix) != 6:
            continue
        inverse = _inverse(matrix)
        if inverse is None:
            continue
        source = str(row.get("source_page_ref"))
        target = str(row.get("target_page_ref"))
        common = {
            "registration_ref": str(row.get("id")),
            "tolerance_display_points": _finite_positive(
                row.get("maximum_residual_tolerance_display")
            ),
        }
        output[(source, target)] = {**common, "matrix": [float(v) for v in matrix]}
        output[(target, source)] = {**common, "matrix": inverse}
    return output


def _accepted_attributes(
    attribute_bindings: Mapping[str, Any],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    output: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for relation in attribute_bindings.get("relations", []):
        if relation.get("state") != "accepted":
            continue
        kind = str(relation.get("relation_type"))
        if kind not in _ATTRIBUTE_TYPES:
            continue
        for target_ref in relation.get("target_refs", []):
            output.setdefault(str(target_ref), {}).setdefault(kind, []).append(relation)
    return output


def _unique_attribute(
    target_refs: list[str],
    relation_type: str,
    attributes: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    values: list[dict[str, Any]] = []
    relation_refs: list[str] = []
    for target_ref in target_refs:
        rows = list(attributes.get(target_ref, {}).get(relation_type, []))
        distinct = {_semantic(dict(row.get("candidate", {}))) for row in rows}
        relation_refs.extend(str(row.get("id")) for row in rows)
        if len(rows) == 0:
            return None, sorted(set(relation_refs)), f"missing_{relation_type}"
        if len(distinct) != 1:
            return None, sorted(set(relation_refs)), f"conflicting_{relation_type}"
        values.append(deepcopy(dict(rows[0].get("candidate", {}))))
    if len({_semantic(value) for value in values}) != 1:
        return None, sorted(set(relation_refs)), f"duplicate_occurrence_{relation_type}_disagreement"
    return values[0], sorted(set(relation_refs)), None


def _aligned_path_residual(
    left: list[list[float]], right: list[list[float]]
) -> tuple[float, list[list[float]]] | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    direct = max(math.dist(a, b) for a, b in zip(left, right))
    reversed_right = list(reversed(right))
    reverse = max(math.dist(a, b) for a, b in zip(left, reversed_right))
    return (direct, right) if direct <= reverse else (reverse, reversed_right)


def _candidate(
    canonical: Mapping[str, Any],
    *,
    pages: Mapping[str, Mapping[str, Any]],
    composites: Mapping[str, Mapping[str, Any]],
    occurrences: Mapping[str, Mapping[str, Any]],
    attributes: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    transforms: Mapping[tuple[str, str], Mapping[str, Any]],
    route_scopes: Mapping[str, Mapping[str, Any]],
    riser_relations: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    canonical_ref = str(canonical.get("id"))
    candidate_id = _stable_id("bounded_local_3d_candidate", canonical_ref)
    target_refs = sorted(str(ref) for ref in canonical.get("source_route_target_refs", []))
    source_occurrence_refs = sorted(
        str(ref) for ref in canonical.get("source_page_occurrence_refs", [])
    )
    reasons: list[str] = []

    canonical_closed = (
        canonical.get("state") == "derived"
        and canonical.get("physical_segment_identity_established") is True
        and len(target_refs) >= 2
        and len(source_occurrence_refs) == len(target_refs)
    )
    if not canonical_closed:
        reasons.append("canonical_projected_segment_not_closed")

    composite_rows = [composites.get(ref) for ref in target_refs]
    semantic_composites = bool(composite_rows) and all(
        row is not None
        and row.get("state") == "accepted"
        and row.get("native_strokes_preserved") is True
        for row in composite_rows
    )
    if not semantic_composites:
        reasons.append("accepted_semantic_composite_missing")

    closure_closed = semantic_composites and all(
        row.get("certificates", {}).get("explicit_envelope_closing_feature") is True
        and (
            row.get("supporting_closure_fragment_refs")
            or row.get("shared_terminal_geometry_refs")
            or row.get("other_envelope_closing_evidence_refs")
        )
        for row in composite_rows
        if row is not None
    )
    if not closure_closed:
        reasons.append("explicit_envelope_closure_missing")

    occurrence_rows = [occurrences.get(ref) for ref in source_occurrence_refs]
    occurrence_closed = bool(occurrence_rows) and all(
        row is not None
        and row.get("physical_segment_ref") == canonical_ref
        and row.get("route_target_ref") in target_refs
        for row in occurrence_rows
    )
    if not occurrence_closed:
        reasons.append("deduplicated_occurrence_membership_mismatch")

    page_refs = sorted(
        {str(row.get("page_ref")) for row in composite_rows if row is not None}
    )
    reference_page_ref = page_refs[0] if page_refs else ""
    scales: dict[str, float] = {}
    scale_evidence_refs: list[str] = []
    for page_ref in page_refs:
        factor, evidence_refs = _page_scale(pages.get(page_ref, {}))
        if factor is not None:
            scales[page_ref] = factor
        scale_evidence_refs.extend(evidence_refs)
    scale_closed = len(scales) == len(page_refs) and bool(scales)
    if not scale_closed:
        reasons.append("accepted_plan_scale_missing")

    frame_transforms: dict[str, list[float]] = {}
    registration_refs: list[str] = []
    frame_tolerance_points: list[float] = []
    for page_ref in page_refs:
        if page_ref == reference_page_ref:
            frame_transforms[page_ref] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        edge = transforms.get((page_ref, reference_page_ref))
        if edge is None:
            continue
        frame_transforms[page_ref] = list(edge["matrix"])
        registration_refs.append(str(edge["registration_ref"]))
        if edge.get("tolerance_display_points") is not None:
            frame_tolerance_points.append(float(edge["tolerance_display_points"]))
    frame_closed = (
        len(page_refs) >= 2
        and len(frame_transforms) == len(page_refs)
        and len(set(registration_refs)) == len(page_refs) - 1
    )
    if not frame_closed:
        reasons.append("accepted_registered_plan_frame_missing")
    frame_scale_agreement = frame_closed and scale_closed
    if frame_scale_agreement:
        reference_factor = scales[reference_page_ref]
        for page_ref in page_refs:
            if page_ref == reference_page_ref:
                continue
            matrix = frame_transforms[page_ref]
            similarity = math.hypot(float(matrix[0]), float(matrix[1]))
            expected_source_factor = similarity * reference_factor
            tolerance = max(abs(scales[page_ref]), abs(expected_source_factor), 1.0) * 1e-8
            if abs(scales[page_ref] - expected_source_factor) > tolerance:
                frame_scale_agreement = False
                break
    if not frame_scale_agreement:
        reasons.append("page_scales_conflict_with_registration_similarity")

    system, system_refs, system_reason = _unique_attribute(
        target_refs, "route_system", attributes
    )
    size, size_refs, size_reason = _unique_attribute(target_refs, "route_size", attributes)
    elevation, elevation_refs, elevation_reason = _unique_attribute(
        target_refs, "route_elevation", attributes
    )
    attribute_reasons = [reason for reason in (system_reason, size_reason, elevation_reason) if reason]
    reasons.extend(attribute_reasons)
    pipe_semantics = (
        size is not None
        and size.get("kind") == "nominal_size"
        and size.get("designation") in {"inch_size", "nominal_pipe_size", "diameter"}
    )
    if not pipe_semantics:
        reasons.append("unsupported_or_unknown_route_envelope_shape")

    widths: list[dict[str, Any]] = []
    for row in composite_rows:
        if row is None:
            continue
        page_ref = str(row.get("page_ref"))
        width_points = _finite_positive(
            row.get("derived_geometry", {}).get("corridor_width_display_points")
        )
        member_width_points = _finite_positive(
            row.get("geometry_metrics", {}).get("member_width_display_points")
        )
        if width_points is None or member_width_points is None or page_ref not in scales:
            continue
        widths.append(
            {
                "route_composite_ref": str(row.get("id")),
                "page_ref": page_ref,
                "outer_width_display_points": round(width_points, 8),
                "outer_width_m": round(width_points * scales[page_ref], 8),
                "measurement_resolution_display_points": round(member_width_points, 8),
                "measurement_resolution_m": round(member_width_points * scales[page_ref], 8),
                "source_primitive_refs": sorted(
                    str(ref) for ref in row.get("member_source_primitive_refs", [])
                ),
            }
        )
    drawing_width_closed = len(widths) == len(target_refs) and len(widths) >= 2
    if not drawing_width_closed:
        reasons.append("drawing_derived_outer_width_missing")
    width_values = [float(row["outer_width_m"]) for row in widths]
    width_mean = sum(width_values) / len(width_values) if width_values else None
    width_residual = (
        max(width_values) - min(width_values) if width_values else None
    )
    width_tolerance = max(
        (float(row["measurement_resolution_m"]) for row in widths), default=0.0
    )
    width_agreement = (
        drawing_width_closed
        and width_residual is not None
        and width_tolerance > 0
        and width_residual <= width_tolerance
    )
    if not width_agreement:
        reasons.append("duplicate_occurrence_outer_width_disagreement")

    elevation_basis = None if elevation is None else str(elevation.get("basis") or "")
    elevation_reference_m = (
        None if elevation is None else _metres(elevation.get("value"), elevation.get("unit"))
    )
    elevation_closed = (
        elevation_basis in {"bottom", "centreline"}
        and elevation_reference_m is not None
        and (elevation_basis == "centreline" or width_agreement)
    )
    if not elevation_closed:
        reasons.append("elevation_basis_or_offset_unresolved")

    transformed_paths: list[dict[str, Any]] = []
    if frame_closed and scale_closed and semantic_composites:
        reference_factor = scales[reference_page_ref]
        for row in composite_rows:
            assert row is not None
            page_ref = str(row.get("page_ref"))
            points = row.get("derived_geometry", {}).get("centreline_points_display", [])
            if not isinstance(points, list) or len(points) < 2:
                continue
            frame_points_display = [
                _apply(frame_transforms[page_ref], point) for point in points
            ]
            transformed_paths.append(
                {
                    "route_composite_ref": str(row.get("id")),
                    "page_ref": page_ref,
                    "points_frame_display": frame_points_display,
                    "points_frame_xy_m": [
                        [round(point[0] * reference_factor, 8), round(point[1] * reference_factor, 8)]
                        for point in frame_points_display
                    ],
                }
            )
    path_residual = None
    aligned_paths: list[list[list[float]]] = []
    if len(transformed_paths) == len(target_refs) and transformed_paths:
        anchor = transformed_paths[0]["points_frame_display"]
        aligned_paths.append(anchor)
        residuals = []
        for row in transformed_paths[1:]:
            aligned = _aligned_path_residual(anchor, row["points_frame_display"])
            if aligned is None:
                residuals = []
                break
            residuals.append(aligned[0])
            aligned_paths.append(aligned[1])
        if len(aligned_paths) == len(transformed_paths):
            path_residual = max(residuals, default=0.0)
    path_tolerance_points = max(
        [*frame_tolerance_points, *[float(row["measurement_resolution_display_points"]) for row in widths]],
        default=0.0,
    )
    path_agreement = (
        path_residual is not None
        and path_tolerance_points > 0
        and path_residual <= path_tolerance_points
    )
    if not path_agreement:
        reasons.append("duplicate_occurrence_path_disagreement")

    certificates = {
        "canonical_projected_segment_accepted": canonical_closed,
        "accepted_registered_metric_plan_frame": frame_scale_agreement,
        "accepted_semantic_composite_per_occurrence": semantic_composites and occurrence_closed,
        "drawing_derived_outer_width": drawing_width_closed,
        "compatible_system_size_elevation": not attribute_reasons,
        "pipe_envelope_semantics": pipe_semantics,
        "duplicate_occurrence_agreement": width_agreement and path_agreement,
        "resolved_bottom_or_centreline_elevation": elevation_closed,
        "explicit_envelope_closure": closure_closed,
        "authority_boundary_closed": True,
    }
    reasons = sorted(set(reasons))
    candidate_record = {
        "record_type": "bounded_local_3d_segment_candidate",
        "record_version": SCHEMA_VERSION,
        "id": candidate_id,
        "canonical_projected_segment_ref": canonical_ref,
        "state": "accepted" if not reasons and all(certificates.values()) else "abstained",
        "reasons": reasons,
        "certificates": certificates,
        "quantity_eligible": False,
    }
    if candidate_record["state"] != "accepted":
        return candidate_record, None

    assert width_mean is not None
    assert elevation_reference_m is not None
    reference_factor = scales[reference_page_ref]
    averaged_display = [
        [
            sum(path[index][axis] for path in aligned_paths) / len(aligned_paths)
            for axis in (0, 1)
        ]
        for index in range(len(aligned_paths[0]))
    ]
    centreline_elevation_m = (
        elevation_reference_m
        if elevation_basis == "centreline"
        else elevation_reference_m + width_mean / 2.0
    )
    centreline_xyz = [
        [
            round(point[0] * reference_factor, 8),
            round(point[1] * reference_factor, 8),
            round(centreline_elevation_m, 8),
        ]
        for point in averaged_display
    ]
    source_primitive_refs = sorted(
        {
            str(ref)
            for row in composite_rows
            if row is not None
            for ref in row.get("member_source_primitive_refs", [])
        }
    )
    canonical_fragment_refs = {
        str(ref) for ref in canonical.get("source_fragment_refs", [])
    }
    unresolved_vertical_spans = []
    for relation in riser_relations:
        related_fragments = {
            str(ref) for ref in relation.get("target_fragment_refs", [])
        }
        related_scope_refs = sorted(str(ref) for ref in relation.get("route_scope_refs", []))
        for scope_ref in related_scope_refs:
            related_fragments.update(
                str(ref) for ref in route_scopes.get(scope_ref, {}).get("fragment_refs", [])
            )
        if not canonical_fragment_refs.intersection(related_fragments):
            continue
        candidate = relation.get("candidate", {})
        destination = candidate.get("destination_elevation") or candidate.get("target_elevation")
        destination_m = (
            _metres(destination.get("value"), destination.get("unit"))
            if isinstance(destination, Mapping)
            else None
        )
        if destination_m is not None:
            continue
        unresolved_vertical_spans.append(
            {
                "record_type": "mep_unresolved_vertical_span",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_unresolved_vertical_span", relation.get("id"), canonical_ref),
                "canonical_projected_segment_ref": canonical_ref,
                "riser_drop_relation_ref": str(relation.get("id")),
                "endpoint_or_vertex_refs": sorted(str(ref) for ref in relation.get("target_refs", [])),
                "reason": "second_endpoint_elevation_is_unresolved",
                "state": "unknown",
                "quantity_eligible": False,
            }
        )
    segment_id = _stable_id("bounded_local_3d_segment", canonical_ref)
    segment = {
        "record_type": "bounded_local_3d_segment",
        "record_version": SCHEMA_VERSION,
        "id": segment_id,
        "label": LABEL,
        "canonical_projected_segment_ref": canonical_ref,
        "source_page_occurrence_refs": source_occurrence_refs,
        "source_route_composite_refs": target_refs,
        "source_fragment_refs": sorted(str(ref) for ref in canonical.get("source_fragment_refs", [])),
        "source_primitive_refs": source_primitive_refs,
        "m1_metric_frame": {
            "id": _stable_id("mep_registered_metric_plan_frame", reference_page_ref, page_refs, registration_refs),
            "state": "accepted",
            "reference_page_ref": reference_page_ref,
            "member_page_refs": page_refs,
            "display_to_reference_display_matrices": {
                key: [round(float(value), 10) for value in frame_transforms[key]]
                for key in sorted(frame_transforms)
            },
            "reference_scale_m_per_display_point": round(reference_factor, 10),
            "registration_refs": sorted(set(registration_refs)),
            "scale_evidence_refs": sorted(set(scale_evidence_refs)),
            "axes": {"x": "display_right", "y": "display_down", "z": "elevation_up"},
            "absolute_origin_established": False,
        },
        "system": system,
        "nominal_size": size,
        "physical_envelope_dimension": {
            "kind": "drawing_derived_outer_width",
            "shape": "circular_pipe",
            "occurrence_observations": widths,
            "representative_outer_width_m": round(width_mean, 8),
            "maximum_width_residual_m": round(width_residual or 0.0, 8),
            "agreement_tolerance_m": round(width_tolerance, 8),
            "agreement_tolerance_derivation": (
                "maximum source member lineweight display points multiplied by "
                "the accepted M1 scale in metres per display point"
            ),
            "nominal_size_used_as_physical_dimension": False,
        },
        "elevation": {
            "reference_basis": elevation_basis,
            "reference_elevation_m": round(elevation_reference_m, 8),
            "centreline_elevation_m": round(centreline_elevation_m, 8),
            "derivation": (
                "accepted_centreline_elevation"
                if elevation_basis == "centreline"
                else "accepted_bottom_elevation_plus_half_drawing_derived_outer_width"
            ),
        },
        "centreline_points_xyz_m": centreline_xyz,
        "local_envelope": {
            "kind": "circular_swept_corridor",
            "outer_width_m": round(width_mean, 8),
            "radius_m": round(width_mean / 2.0, 8),
            "centreline_points_xyz_m": deepcopy(centreline_xyz),
            "terminal_boundaries_are_analysis_caps": True,
        },
        "duplicate_occurrence_agreement": {
            "transformed_occurrences": transformed_paths,
            "maximum_path_residual_display_points": round(path_residual or 0.0, 8),
            "path_agreement_tolerance_display_points": round(path_tolerance_points, 8),
            "maximum_width_residual_m": round(width_residual or 0.0, 8),
            "width_agreement_tolerance_m": round(width_tolerance, 8),
            "width_agreement_tolerance_derivation": (
                "maximum source member lineweight display points multiplied by "
                "the accepted M1 scale in metres per display point"
            ),
            "elevation_semantic_agreement": True,
        },
        "analysis_caps": [
            {
                "endpoint_index": index,
                "point_xyz_m": deepcopy(point),
                "state": "unresolved",
                "physical_boundary_established": False,
                "continuation_established": False,
                "clash_authority": False,
            }
            for index, point in enumerate((centreline_xyz[0], centreline_xyz[-1]))
        ],
        "unresolved_vertical_spans": sorted(
            unresolved_vertical_spans, key=lambda row: row["id"]
        ),
        "relation_refs": sorted(set([*system_refs, *size_refs, *elevation_refs])),
        "certificates": certificates,
        "state": "accepted",
        "epistemic_state": "derived",
        "authority": {
            "bounded_local_segment_identity_established": True,
            "physical_run_identity_established": False,
            "physical_continuation_established": False,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "calculated_severity_emitted": False,
            "m7_takeoff_emitted": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    return candidate_record, segment


def _duct_surface(points, width, height):
    """Four rectangular side faces; the two analysis ends have no faces."""
    a, b = points
    length = math.dist(a[:2], b[:2])
    nx, ny = -(b[1]-a[1])/length, (b[0]-a[0])/length
    rings = [[[p[0]+s*nx*width/2, p[1]+s*ny*width/2, p[2]+z*height/2]
              for s, z in ((-1,-1),(1,-1),(1,1),(-1,1))] for p in points]
    vertices = [[round(x, 8) for x in p] for ring in rings for p in ring]
    return {'vertices_xyz_m': vertices,
            'side_faces': [[i, (i+1)%4, (i+1)%4+4, i+4] for i in range(4)],
            'end_faces': [], 'watertight_solid_established': False}


def _vector_length(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _unit_vector(vector):
    length = _vector_length(vector)
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError('duct portal axis is degenerate')
    return [float(value) / length for value in vector]


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _portal_ring(portal):
    centre = [float(value) for value in portal['centre_xyz_m']]
    width_axis = _unit_vector(portal['width_axis_xyz'])
    height_axis = _unit_vector(portal['height_axis_xyz'])
    if abs(_dot(width_axis, height_axis)) > 1e-8:
        raise ValueError('duct portal width and height axes are not orthogonal')
    width = _finite_positive(portal.get('width_m'))
    height = _finite_positive(portal.get('height_m'))
    if width is None or height is None or len(centre) != 3 or any(
            not math.isfinite(value) for value in centre):
        raise ValueError('duct portal dimensions or centre are invalid')
    return [[round(centre[i] + sw * width_axis[i] * width / 2
                         + sh * height_axis[i] * height / 2, 8)
             for i in range(3)]
            for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _mesh_is_watertight(faces):
    from collections import Counter
    edges = Counter()
    for face in faces:
        if len(face) < 3:
            return False
        for left, right in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted((left, right)))] += 1
    return bool(edges) and all(count == 2 for count in edges.values())


def _point_key(point):
    try:
        values = tuple(round(float(value), 6) for value in point)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 2 and all(math.isfinite(value) for value in values) else None


def _polyline_is_straight(points):
    if not isinstance(points, list) or len(points) < 2:
        return False
    try:
        start = [float(value) for value in points[0]]
        end = [float(value) for value in points[-1]]
    except (TypeError, ValueError):
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return False
    try:
        return all(abs(dx * (float(point[1]) - start[1])
                           - dy * (float(point[0]) - start[0])) <= length * 1e-6
                   for point in points[1:-1])
    except (IndexError, TypeError, ValueError):
        return False


def _typed_fitting(candidate):
    value = str(candidate.get('kind') or candidate.get('category') or '').lower()
    if value in {'reducer', 'transition', 'taper', 'tapered_transition'}:
        return 'taper'
    if value in {'bend', 'elbow', 'duct_bend'}:
        return 'bend'
    return None


def assemble_typed_duct_route(*, route_observations, outlined_route_composites,
                               attribute_bindings, route_composite_refs,
                               projected_interfaces=()):
    """Assemble one bounded, typed projected duct route from M3/M3.5/M4 evidence.

    The adapter types accepted outlined composites and their certified endpoint
    interfaces.  It never fills a missing contact, turns a scope end into a
    physical cap, or propagates attributes through a bend or taper.  The result
    is an input gate for the rectangular M5A kernel, not a physical run.
    """
    if (outlined_route_composites.get('m3_contract_ref', {}).get('payload_sha256')
            != _canonical_sha256(route_observations)):
        raise ValueError('typed duct route requires the exact frozen M3 graph')
    for key, payload, label in (
        ('m3_contract_ref', route_observations, 'M3'),
        ('m3_5_contract_ref', outlined_route_composites, 'M3.5'),
    ):
        if (attribute_bindings.get(key, {}).get('payload_sha256')
                != _canonical_sha256(payload)):
            raise ValueError(f'typed duct route requires the exact frozen {label} input')

    selected_refs = sorted(set(str(ref) for ref in route_composite_refs))
    if not selected_refs:
        raise ValueError('typed duct route requires an explicit bounded composite scope')
    composites = {str(row.get('id')): row
                  for row in outlined_route_composites.get('accepted_composites', [])}
    if any(ref not in composites for ref in selected_refs):
        raise ValueError('typed duct route scope contains a non-accepted composite')

    point_endpoints = {}
    for page in route_observations.get('pages', []):
        for endpoint in page.get('endpoints', []):
            key = _point_key(endpoint.get('point_display'))
            if key is not None:
                point_endpoints.setdefault(key, []).append(endpoint)

    endpoint_rows = {}
    panels = []
    reasons = []
    for composite_ref in selected_refs:
        composite = composites[composite_ref]
        source_points = composite.get('derived_geometry', {}).get(
            'centreline_points_display', [])
        points = deepcopy(source_points) if isinstance(source_points, list) else []
        valid_points = (len(points) >= 2
                        and all(_point_key(point) is not None for point in points))
        panel_kind = 'straight' if valid_points and _polyline_is_straight(points) else 'bend'
        if (composite.get('state') != 'accepted'
                or composite.get('native_strokes_preserved') is not True
                or not valid_points):
            reasons.append('selected_panel_lacks_accepted_m3_5_geometry')
        panel_id = _stable_id('mep_typed_duct_panel', composite_ref)
        panels.append({
            'record_type': 'mep_typed_duct_route_element',
            'record_version': SCHEMA_VERSION,
            'id': panel_id,
            'kind': panel_kind,
            'element_role': 'panel',
            'route_composite_ref': composite_ref,
            'member_fragment_refs': sorted(str(ref) for ref in
                                           composite.get('member_fragment_refs', [])),
            'source_primitive_refs': sorted(str(ref) for ref in
                                            composite.get('member_source_primitive_refs', [])),
            'centreline_points_display': points,
            'state': 'derived',
            'physical_continuation_established': False,
            'quantity_eligible': False,
        })
        width = _finite_positive(composite.get('derived_geometry', {}).get(
            'corridor_width_display_points'))
        for endpoint_index, point in (
                ((0, points[0]), (1, points[-1])) if valid_points else ()):
            evidence_refs = {_stable_id('mep_composite_port', composite_ref, endpoint_index)}
            if width is not None and len(points) >= 2:
                other = points[1] if endpoint_index == 0 else points[-2]
                dx, dy = float(other[0]) - float(point[0]), float(other[1]) - float(point[1])
                length = math.hypot(dx, dy)
                if length > 1e-9:
                    normal = [-dy / length, dx / length]
                    for sign in (-1, 1):
                        side = _point_key([float(point[0]) + sign * normal[0] * width / 2,
                                           float(point[1]) + sign * normal[1] * width / 2])
                        for endpoint in point_endpoints.get(side, []):
                            evidence_refs.add(str(endpoint.get('id')))
                            evidence_refs.add(str(endpoint.get('vertex_ref')))
                            evidence_refs.add(str(endpoint.get('fragment_ref')))
            endpoint_rows[composite_ref, endpoint_index] = {
                'point_display': deepcopy(point),
                'evidence_refs': evidence_refs,
            }

    fitting_relations = []
    equipment_relations = []
    for relation in attribute_bindings.get('relations', []):
        if relation.get('state') != 'accepted':
            continue
        if relation.get('relation_type') == 'fitting':
            fitting_relations.append(relation)
        elif relation.get('relation_type') == 'equipment_endpoint':
            equipment_relations.append(relation)

    candidates_by_endpoint = {key: [] for key in endpoint_rows}
    accepted_interfaces = []
    for interface in projected_interfaces:
        ports = list(interface.get('ports', []))
        endpoints = []
        evidence_refs = {str(interface.get('id')),
                         *map(str, interface.get('port_refs', [])),
                         *map(str, interface.get('source_primitive_refs', []))}
        for port in ports:
            key = (str(port.get('composite_ref')), port.get('end'))
            if key in endpoint_rows:
                endpoints.append(key)
            evidence_refs.update(str(ref) for ref in port.get('route_endpoint_refs', []))
            for point in port.get('sides', []):
                for endpoint in point_endpoints.get(_point_key(point), []):
                    evidence_refs.update((str(endpoint.get('id')),
                                          str(endpoint.get('vertex_ref')),
                                          str(endpoint.get('fragment_ref'))))
        endpoints = sorted(set(endpoints))
        if not endpoints:
            continue
        relevant_fittings = [relation for relation in fitting_relations
            if set(map(str, relation.get('target_refs', []))).intersection(evidence_refs)]
        relevant_fittings = list({str(row.get('id')): row
                                  for row in relevant_fittings}.values())
        fitting_kinds = {_typed_fitting(relation.get('candidate', {}))
                         for relation in relevant_fittings}
        fitting_kinds.discard(None)
        fitting_closed = (not relevant_fittings
                          or len(relevant_fittings) == 1 and len(fitting_kinds) == 1)
        geometric_kind = {
            'projected_native_bend': 'bend',
            'projected_collinear_boundary_join': 'straight',
            'projected_native_taper': 'taper',
            'projected_native_transition': 'taper',
        }.get(interface.get('relation_type'))
        kinds = set(fitting_kinds)
        # An explicit reducer/transition may refine a geometrically collinear
        # portal pair.  A contradictory authored bend/taper classification
        # remains ambiguous instead of being overwritten by M4 text.
        if geometric_kind and (not fitting_kinds or geometric_kind != 'straight'):
            kinds.add(geometric_kind)
        complete = (interface.get('state') == 'accepted'
                    and interface.get('search', {}).get('complete') is True)
        kind = next(iter(kinds)) if len(kinds) == 1 and fitting_closed else None
        row = {
            'record_type': 'mep_typed_duct_route_element',
            'record_version': SCHEMA_VERSION,
            'id': _stable_id('mep_typed_duct_interface', interface.get('id'), endpoints),
            'kind': kind or 'unresolved_boundary',
            'element_role': 'internal_interface' if len(endpoints) == 2 else 'scope_boundary',
            'route_composite_refs': sorted({key[0] for key in endpoints}),
            'composite_endpoint_indices': [
                {'route_composite_ref': key[0], 'endpoint_index': key[1]}
                for key in endpoints],
            'source_interface_ref': str(interface.get('id')),
            'm4_relation_refs': sorted(str(relation.get('id'))
                                       for relation in relevant_fittings),
            'source_primitive_refs': sorted(str(ref) for ref in
                                            interface.get('source_primitive_refs', [])),
            'state': ('derived' if complete and kind and len(endpoints) == 2
                      and len({key[0] for key in endpoints}) == 2 else 'unknown'),
            'reason': (None if complete and kind and len(endpoints) == 2
                       and len({key[0] for key in endpoints}) == 2
                       else 'interface_not_mutually_unique_and_typed'),
            'attribute_propagation_across_interface': False,
            'physical_continuation_established': False,
            'quantity_eligible': False,
        }
        for key in endpoints:
            candidates_by_endpoint[key].append(row)
        if row['state'] == 'derived':
            accepted_interfaces.append(row)

    accepted_by_endpoint = {}
    ambiguous_endpoints = set()
    for key, rows in candidates_by_endpoint.items():
        accepted = {row['id']: row for row in rows if row['state'] == 'derived'}
        if len(accepted) == 1:
            accepted_by_endpoint[key] = next(iter(accepted.values()))
        elif len(rows) > 1 or len(accepted) > 1:
            ambiguous_endpoints.add(key)

    terminal_elements = []
    for key in sorted(endpoint_rows):
        if key in accepted_by_endpoint and key not in ambiguous_endpoints:
            continue
        evidence_refs = endpoint_rows[key]['evidence_refs']
        equipment = [relation for relation in equipment_relations
            if set(map(str, relation.get('target_refs', []))).intersection(evidence_refs)]
        unique_equipment = {(_semantic(relation.get('candidate', {})), str(relation.get('id')))
                            for relation in equipment}
        equipment_closed = len(unique_equipment) == 1 and key not in ambiguous_endpoints
        relation = equipment[0] if equipment_closed else None
        terminal_elements.append({
            'record_type': 'mep_typed_duct_route_element',
            'record_version': SCHEMA_VERSION,
            'id': _stable_id('mep_typed_duct_terminal', key,
                             None if relation is None else relation.get('id')),
            'kind': 'equipment_port' if equipment_closed else 'unresolved_boundary',
            'element_role': 'analysis_boundary',
            'route_composite_ref': key[0],
            'composite_endpoint_index': key[1],
            'point_display': endpoint_rows[key]['point_display'],
            'm4_relation_refs': [] if relation is None else [str(relation.get('id'))],
            'candidate_interface_refs': sorted(row['id']
                                               for row in candidates_by_endpoint[key]),
            'candidate_m4_relation_refs': sorted(str(row.get('id'))
                                                 for row in equipment),
            'candidate': None if relation is None else deepcopy(relation.get('candidate', {})),
            'state': 'accepted' if equipment_closed else 'unknown',
            'reason': ('ambiguous_interface_or_terminal' if key in ambiguous_endpoints
                       else None if equipment_closed else 'no_unique_certified_interface'),
            'physical_cap_established': False,
            'physical_continuation_established': False,
            'quantity_eligible': False,
        })

    internal = {row['id']: row for row in accepted_interfaces
                if row['element_role'] == 'internal_interface'
                and all(key not in ambiguous_endpoints for key in
                        [(item['route_composite_ref'], item['endpoint_index'])
                         for item in row['composite_endpoint_indices']])}
    adjacency = {ref: set() for ref in selected_refs}
    for row in internal.values():
        left, right = row['route_composite_refs']
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = set()
    frontier = [selected_refs[0]]
    while frontier:
        ref = frontier.pop()
        if ref in reached:
            continue
        reached.add(ref)
        frontier.extend(adjacency[ref] - reached)
    chain_closed = (reached == set(selected_refs)
                    and all(len(neighbours) <= 2 for neighbours in adjacency.values())
                    and len(terminal_elements) == 2
                    and not ambiguous_endpoints)
    if not chain_closed:
        reasons.append('typed_route_is_not_one_unambiguous_bounded_chain')

    ordered_panel_refs = []
    ordered_interface_refs = []
    if chain_closed:
        terminal_panels = sorted(ref for ref, neighbours in adjacency.items()
                                 if len(neighbours) <= 1)
        current = terminal_panels[0]
        previous = None
        pair_interfaces = {
            frozenset(row['route_composite_refs']): row['id']
            for row in internal.values()
        }
        while current is not None:
            ordered_panel_refs.append(current)
            following = sorted(adjacency[current] - ({previous} if previous else set()))
            if not following:
                break
            next_ref = following[0]
            ordered_interface_refs.append(pair_interfaces[frozenset((current, next_ref))])
            previous, current = current, next_ref

    elements = sorted([*panels, *internal.values(), *terminal_elements],
                      key=lambda row: row['id'])
    assembly_id = _stable_id('mep_typed_duct_route_assembly', selected_refs,
                             [row['id'] for row in elements])
    return {
        'record_type': 'mep_typed_duct_route_assembly',
        'record_version': SCHEMA_VERSION,
        'id': assembly_id,
        'state': 'accepted' if not reasons else 'abstained',
        'reasons': sorted(set(reasons)),
        'source_contracts': {
            'm3_payload_sha256': _canonical_sha256(route_observations),
            'm3_5_payload_sha256': _canonical_sha256(outlined_route_composites),
            'm4_payload_sha256': _canonical_sha256(attribute_bindings),
        },
        'route_composite_refs': selected_refs,
        'elements': elements,
        'chain_integrity': {
            'state': 'accepted' if chain_closed else 'abstained',
            'complete_within_analysis_boundaries': chain_closed,
            'panel_count': len(panels),
            'internal_interface_count': len(internal),
            'analysis_boundary_count': len(terminal_elements),
            'ambiguous_endpoint_count': len(ambiguous_endpoints),
            'ordered_panel_refs': ordered_panel_refs,
            'ordered_internal_interface_refs': ordered_interface_refs,
        },
        'physical_run_identity_established': False,
        'physical_continuation_established': False,
        'installed_length_emitted': False,
        'quantity_eligible': False,
    }


def build_relative_duct_3d_assembly(evidence):
    """Materialize one evidence-closed rectangular duct assembly in relative XYZ.

    The input owns plan/section identity, dimensions, scale and topology.  This
    kernel only sweeps the supplied evidence-backed portals.  Its two closing
    faces are verification planes at analysis boundaries, never physical caps.
    """
    required = {
        'plan_to_section_identity': ('accepted', 'mutually_unique'),
        'cross_section': ('accepted', 'drawing_derived'),
        'route_topology': ('accepted', 'complete_within_analysis_boundaries'),
        'relative_z': ('accepted', 'drawing_derived_from_scaled_section'),
        'two_view_scale': ('accepted', 'plan_and_section_metric'),
    }
    reasons = []
    for name, (state, flag) in required.items():
        row = evidence.get(name, {})
        if row.get('state') != state or row.get(flag) is not True:
            reasons.append(name + '_unresolved')
    size_certificate = evidence.get('size_applicability_certificate')
    size_gate_names = {'native_width_height_text',
        'direct_leader_or_inline_association', 'unique_native_duct_interval',
        'outside_dimension_convention', 'rectangular_duct_dimension_unit'}
    if (not isinstance(size_certificate, dict)
            or size_certificate.get('record_type') !=
                'mep_duct_size_applicability_certificate'
            or size_certificate.get('state') != 'accepted'
            or not size_gate_names.issubset(
                (size_certificate.get('certificates') or {}).keys())
            or not all(size_certificate['certificates'][name]
                       for name in size_gate_names)):
        reasons.append('duct_size_applicability_certificate_unresolved')
    station_correspondence = evidence.get('plan_section_station_correspondence')
    if (not isinstance(station_correspondence, dict)
            or station_correspondence.get('record_type') !=
                'mep_duct_plan_section_station_correspondence'
            or station_correspondence.get('state') != 'accepted'
            or station_correspondence.get(
                'size_applicability_certificate_ref') !=
                (size_certificate or {}).get('id')
            or station_correspondence.get('sized_route_interval_ref') !=
                (size_certificate or {}).get('route_composite_ref')):
        reasons.append('plan_section_station_correspondence_unresolved')
    typed_route = evidence.get('typed_route_assembly')
    if (not isinstance(typed_route, dict) or
            typed_route.get('record_type') != 'mep_typed_duct_route_assembly'
            or typed_route.get('state') != 'accepted'
            or typed_route.get('chain_integrity', {}).get(
                'complete_within_analysis_boundaries') is not True
            or typed_route.get('chain_integrity', {}).get(
                'ambiguous_endpoint_count') != 0
            or typed_route.get('chain_integrity', {}).get(
                'analysis_boundary_count') != 2):
        reasons.append('typed_duct_route_assembly_unresolved')
    cross_section = evidence.get('cross_section', {})
    if (cross_section.get('shape') != 'rectangular'
            or cross_section.get('dimension_basis') not in {'outside', 'exterior'}
            or not cross_section.get('dimension_convention_ref')):
        reasons.append('exterior_rectangular_dimension_basis_unresolved')
    portal_rows = list(evidence.get('portals', []))
    portals = {str(row.get('id')): row for row in portal_rows}
    parts = list(evidence.get('parts', []))
    part_ids = [str(row.get('id')) for row in parts]
    if len(portals) < 2 or not parts:
        reasons.append('duct_portals_or_parts_missing')
    if (any(row.get('id') in {None, ''} for row in portal_rows)
            or len(portals) != len(portal_rows)):
        reasons.append('duct_portal_ids_are_not_unique')
    if (any(row.get('id') in {None, ''} for row in parts)
            or len(set(part_ids)) != len(part_ids)):
        reasons.append('duct_part_ids_are_not_unique')
    typed_elements = {str(row.get('id')): row
                      for row in (typed_route or {}).get('elements', [])}
    typed_part_refs = [str(row.get('typed_element_ref')) for row in parts]
    if (any(ref not in typed_elements for ref in typed_part_refs)
            or len(set(typed_part_refs)) != len(typed_part_refs)
            or any(typed_elements.get(ref, {}).get('kind') != part.get('kind')
                   for ref, part in zip(typed_part_refs, parts))):
        reasons.append('duct_parts_do_not_replay_typed_route_elements')
    if (size_certificate or {}).get('route_composite_ref') not in set(
            (typed_route or {}).get('route_composite_refs', [])):
        reasons.append('sized_interval_not_in_typed_duct_route')
    rings = {}
    for ref, portal in portals.items():
        try:
            rings[ref] = _portal_ring(portal)
        except (KeyError, TypeError, ValueError):
            reasons.append('invalid_duct_portal_geometry')
    connections = []
    connection_parts = {}
    for part in parts:
        refs = [str(ref) for ref in part.get('portal_refs', [])]
        kind = part.get('kind')
        if (kind not in {'straight', 'bend', 'taper'} or len(refs) < 2
                or (kind == 'bend' and len(refs) < 3)
                or (kind in {'straight', 'taper'} and len(refs) != 2)
                or any(ref not in portals for ref in refs)):
            reasons.append('invalid_duct_part_topology')
            continue
        endpoint_dimensions = [[_finite_positive(portals[ref].get(key))
                                for key in ('width_m', 'height_m')]
                               for ref in (refs[0], refs[-1])]
        if any(value is None for row in endpoint_dimensions for value in row):
            reasons.append('invalid_duct_portal_geometry')
            continue
        if kind == 'straight' and any(
                abs(endpoint_dimensions[0][index] - endpoint_dimensions[1][index]) > 1e-9
                for index in range(2)):
            reasons.append('straight_part_changes_cross_section')
        if kind == 'straight' and all(ref in rings for ref in refs):
            left_width = _unit_vector(portals[refs[0]]['width_axis_xyz'])
            right_width = _unit_vector(portals[refs[1]]['width_axis_xyz'])
            left_height = _unit_vector(portals[refs[0]]['height_axis_xyz'])
            right_height = _unit_vector(portals[refs[1]]['height_axis_xyz'])
            if (_dot(left_width, right_width) < 1 - 1e-8
                    or _dot(left_height, right_height) < 1 - 1e-8):
                reasons.append('straight_portal_orientation_changes')
        if kind == 'taper' and all(
                abs(endpoint_dimensions[0][index] - endpoint_dimensions[1][index]) <= 1e-9
                for index in range(2)):
            reasons.append('taper_has_no_size_change')
        if kind == 'bend' and all(ref in rings for ref in refs):
            directions = [_unit_vector([
                float(portals[right]['centre_xyz_m'][axis])
                - float(portals[left]['centre_xyz_m'][axis]) for axis in range(3)])
                for left, right in zip(refs, refs[1:])]
            if all(abs(_dot(left, right)) >= 1 - 1e-8
                   for left, right in zip(directions, directions[1:])):
                reasons.append('bend_has_no_resolved_turn')
        for pair in zip(refs, refs[1:]):
            connections.append(pair)
            connection_parts[tuple(sorted(pair))] = str(part.get('id'))
    if len(set(tuple(sorted(pair)) for pair in connections)) != len(connections):
        reasons.append('duplicate_duct_connection')
    degrees = {ref: 0 for ref in portals}
    adjacency = {ref: set() for ref in portals}
    for left, right in connections:
        degrees[left] += 1; degrees[right] += 1
        adjacency[left].add(right); adjacency[right].add(left)
    boundaries = sorted(ref for ref, degree in degrees.items() if degree == 1)
    if (len(boundaries) != 2 or any(degree not in {1, 2} for degree in degrees.values())
            or any(degree == 0 for degree in degrees.values())):
        reasons.append('route_is_not_one_bounded_chain')
    if portals:
        reached, frontier = set(), [next(iter(portals))]
        while frontier:
            ref = frontier.pop()
            if ref in reached:
                continue
            reached.add(ref); frontier.extend(adjacency[ref] - reached)
        if reached != set(portals):
            reasons.append('route_portals_are_disconnected')
    if reasons:
        return {'record_type': 'mep_relative_duct_3d_assembly',
            'id': _stable_id('mep_relative_duct_3d_assembly', evidence.get('id')),
            'state': 'abstained', 'reasons': sorted(set(reasons)),
            'relative_3d_established': False, 'absolute_elevation_resolved': False,
            'resolved_relative_centerline_length_m': None,
            'quantity_eligible': False}

    ordered_refs = [boundaries[0]]
    previous = None
    while len(ordered_refs) < len(portals):
        current = ordered_refs[-1]
        following = sorted(adjacency[current] - ({previous} if previous else set()))
        if len(following) != 1:
            raise ValueError('bounded duct chain ordering changed')
        previous, current = current, following[0]
        ordered_refs.append(current)
    vertices = []
    offsets = {}
    for ref in ordered_refs:
        offsets[ref] = len(vertices)
        vertices.extend(rings[ref])
    side_faces = []
    faces_by_part = {str(row.get('id')): [] for row in parts}
    length_by_part = {str(row.get('id')): 0.0 for row in parts}
    length = 0.0
    for left, right in zip(ordered_refs, ordered_refs[1:]):
        a, b = offsets[left], offsets[right]
        faces = [[a+i, a+(i+1)%4, b+(i+1)%4, b+i] for i in range(4)]
        side_faces.extend(faces)
        part_ref = connection_parts[tuple(sorted((left, right)))]
        faces_by_part[part_ref].extend(faces)
        segment_length = math.dist(portals[left]['centre_xyz_m'],
                                   portals[right]['centre_xyz_m'])
        length_by_part[part_ref] += segment_length
        length += segment_length
    analysis_faces = [list(reversed([offsets[ordered_refs[0]]+i for i in range(4)])),
                      [offsets[ordered_refs[-1]]+i for i in range(4)]]
    verification_faces = side_faces + analysis_faces
    watertight = _mesh_is_watertight(verification_faces)
    reprojections = []
    for view in evidence.get('reprojections', []):
        matrix = view.get('projection_matrix')
        observed = view.get('observed_portal_centres_m')
        observed_rings = view.get('observed_portal_rings_m')
        tolerance = _finite_positive(view.get('tolerance_m'))
        if (not isinstance(matrix, list) or len(matrix) != 2
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                or not isinstance(observed, list) or len(observed) != len(ordered_refs)
                or any(not isinstance(point, list) or len(point) != 2
                       or any(not isinstance(value, (int, float))
                              or not math.isfinite(float(value)) for value in point)
                       for point in observed)
                or not isinstance(observed_rings, list)
                or len(observed_rings) != len(ordered_refs)
                or any(not isinstance(ring, list) or len(ring) != 4
                       or any(not isinstance(point, list) or len(point) != 2
                              or any(not isinstance(value, (int, float))
                                     or not math.isfinite(float(value))
                                     for value in point) for point in ring)
                       for ring in observed_rings)
                or tolerance is None):
            raise ValueError('invalid duct reprojection evidence')
        projected = [[sum(float(matrix[j][i]) * float(portals[ref]['centre_xyz_m'][i])
                          for i in range(3)) for j in range(2)] for ref in ordered_refs]
        projected_rings = [[[sum(float(matrix[j][i]) * float(point[i])
                                 for i in range(3)) for j in range(2)]
                            for point in rings[ref]] for ref in ordered_refs]
        centre_residual = max(math.dist(left, right)
            for left, right in zip(projected, observed))
        envelope_residual = max(math.dist(left, right)
            for projected_ring, observed_ring in zip(projected_rings, observed_rings)
            for left, right in zip(projected_ring, observed_ring))
        residual = max(centre_residual, envelope_residual)
        reprojections.append({'view': view.get('view'), 'state': 'accepted' if residual <= tolerance else 'rejected',
            'maximum_residual_m': round(residual, 8), 'tolerance_m': tolerance,
            'centreline_residual_m': round(centre_residual, 8),
            'envelope_residual_m': round(envelope_residual, 8),
            'evidence_refs': sorted(str(ref) for ref in view.get('evidence_refs', []))})
    two_view = ({row.get('view') for row in reprojections if row['state'] == 'accepted'}
                >= {'plan', 'section'})
    state = 'accepted' if watertight and two_view else 'abstained'
    return {'record_type': 'mep_relative_duct_3d_assembly',
        'id': _stable_id('mep_relative_duct_3d_assembly', evidence.get('id'), ordered_refs),
        'state': state,
        'reasons': [] if state == 'accepted' else sorted(set(
            ([] if watertight else ['verification_envelope_not_watertight'])
            + ([] if two_view else ['two_view_reprojection_failed']))),
        'relative_3d_established': state == 'accepted',
        'metric_frame': {'kind': 'relative_xyz', 'absolute_z_origin_established': False},
        'ordered_portal_refs': ordered_refs,
        'parts': [{'id': str(row.get('id')), 'kind': row.get('kind'),
                   'portal_refs': [str(ref) for ref in row.get('portal_refs', [])],
                   'centreline_geometric_length_m': round(
                       length_by_part[str(row.get('id'))], 8),
                   'physical_side_faces': faces_by_part[str(row.get('id'))]}
                  for row in parts],
        'vertices_xyz_m': vertices, 'physical_side_faces': side_faces,
        'physical_end_faces': [],
        'verification_analysis_boundary_faces': [
            {'face': face, 'physical_cap': False, 'quantity_eligible': False}
            for face in analysis_faces],
        'verification_envelope_watertight': watertight,
        'internal_interfaces_gap_free': True,
        'reprojections': reprojections,
        'resolved_relative_centerline_length_m': round(length, 8)
            if state == 'accepted' else None,
        'absolute_elevation_resolved': False,
        'cross_system_placement_eligible': False,
        'bounded_clash_verification_eligible': False,
        'quantity_eligible': False}


def _page_local_duct_extension(registry, terminology, bindings):
    from src.drawing_engine.disciplines.mep.mep_duct_evidence import duct_section_evidence
    from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals
    if (validate_mep_terminology_proposals(terminology)
            or bindings['m2_contract_ref']['payload_sha256'] != _canonical_sha256(terminology)):
        raise ValueError('duct M5A requires the exact validated M4 terminology')
    sections = duct_section_evidence(registry=registry, terminology=terminology, bindings=bindings)
    composites = {c['id']: c for c in bindings['outlined_route_composites']}
    pages = {p['page_ref']: p for p in registry['pages']}
    attributes = _accepted_attributes(bindings)
    candidates, segments = [], []
    for section in sections:
        c = composites[section['route_composite_ref']]
        reasons = list(section['reasons'])
        values, refs = {}, []
        for kind in _ATTRIBUTE_TYPES:
            value, relation_refs, reason = _unique_attribute([c['id']], kind, attributes)
            values[kind] = value
            refs.extend(relation_refs)
            if reason:
                reasons.append(reason)
        system = values['route_system'] or {}
        if system.get('kind') not in {'supply_air','return_air','outside_air','exhaust_air'}:
            reasons.append('explicit_duct_system_missing')
        factor, scale_refs = _page_scale(pages[c['page_ref']])
        if factor is None:
            reasons.append('accepted_plan_scale_missing')
        elevation = values['route_elevation'] or {}
        z = _metres(elevation.get('value'), elevation.get('unit'))
        if elevation.get('basis') not in {'bottom','centreline'} or z is None:
            reasons.append('duct_elevation_basis_or_value_unresolved')
        points = c['derived_geometry']['centreline_points_display']
        if len(points) != 2 or math.dist(*points) <= 0:
            reasons.append('bounded_duct_subset_requires_one_straight_interval')
        identifier = _stable_id('mep_page_local_duct_3d_candidate', c['id'])
        candidate = {'record_type': 'mep_page_local_duct_3d_candidate', 'id': identifier,
            'page_ref': c['page_ref'], 'route_composite_ref': c['id'],
            'state': 'abstained' if reasons else 'accepted', 'reasons': sorted(set(reasons)),
            'section_evidence': section, 'm4_relation_refs': sorted(set(refs)),
            'geometry_inputs': {'centreline_points_display': points,
                'scale_m_per_display_point': factor, 'elevation': elevation},
            'quantity_eligible': False}
        candidates.append(candidate)
        if reasons:
            continue
        dimensions = section['orientation_and_unit_alternatives'][0]
        height = dimensions['vertical_height_m']
        width = dimensions['plan_width_m']
        center_z = z if elevation['basis'] == 'centreline' else z+height/2
        xyz = [[round(p[0]*factor,8), round(p[1]*factor,8), round(center_z,8)] for p in points]
        segments.append({'record_type': 'mep_page_local_duct_3d_segment',
            'id': _stable_id('mep_page_local_duct_3d_segment', c['id']),
            'candidate_ref': identifier, 'route_composite_ref': c['id'], 'page_ref': c['page_ref'],
            'state': 'accepted', 'centreline_points_xyz_m': xyz,
            'cross_section': {'shape': 'rectangular', 'basis': 'exterior',
                'width_m': width, 'height_m': height, 'evidence_ref': section['id']},
            'local_envelope': _duct_surface(xyz, width, height),
            'metric_frame': {'kind': 'page_local_plan', 'scale_m_per_display_point': factor,
                'scale_evidence_refs': scale_refs, 'axes': ['display_right','display_down','elevation_up'],
                'absolute_xy_origin_established': False},
            'elevation': elevation, 'system': system,
            'source_primitive_refs': c['member_source_primitive_refs'],
            'analysis_boundaries': [{'point_xyz_m': p, 'state': 'unresolved',
                'physical_boundary_established': False} for p in xyz],
            'physical_continuation_established': False, 'installed_length_emitted': False,
            'quantity_eligible': False})
    return {'m2_contract_ref': {'layer': terminology['layer'], 'payload_sha256': _canonical_sha256(terminology)},
            'segment_candidates': candidates, 'bounded_local_3d_segments': segments,
            'quantity_eligible': False}


def build_mep_bounded_local_3d_segments(
    *,
    sheet_registry: Mapping[str, Any],
    outlined_route_composites: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    cross_sheet_runs: Mapping[str, Any],
    duct_terminology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build M5A certificates from accepted, deduplicated local route evidence."""

    upstream_errors = _upstream_errors(
        sheet_registry, outlined_route_composites, attribute_bindings, cross_sheet_runs
    )
    if upstream_errors:
        raise ValueError("invalid upstream MEP payload:\n" + "\n".join(upstream_errors))
    document_keys = {
        str(payload.get("document", {}).get("document_key"))
        for payload in (
            sheet_registry,
            outlined_route_composites,
            attribute_bindings,
            cross_sheet_runs,
        )
        if payload.get("document", {}).get("document_key")
    }
    if len(document_keys) > 1:
        raise ValueError("M1, M3.5, M4, and M5 document keys do not match")

    pages = {str(row.get("page_ref")): row for row in sheet_registry.get("pages", [])}
    composites = {
        str(row.get("id")): row
        for row in outlined_route_composites.get("accepted_composites", [])
    }
    occurrences = {
        str(row.get("id")): row
        for row in cross_sheet_runs.get("projected_route_occurrences", [])
    }
    attributes = _accepted_attributes(attribute_bindings)
    transforms = _direct_transforms(sheet_registry)
    route_scopes = {
        str(row.get("id")): row for row in attribute_bindings.get("route_scopes", [])
    }
    riser_relations = [
        row
        for row in attribute_bindings.get("relations", [])
        if row.get("state") == "accepted" and row.get("relation_type") == "riser_drop"
    ]
    candidates = []
    accepted = []
    for canonical in sorted(
        cross_sheet_runs.get("canonical_projected_segments", []),
        key=lambda row: str(row.get("id")),
    ):
        candidate, segment = _candidate(
            canonical,
            pages=pages,
            composites=composites,
            occurrences=occurrences,
            attributes=attributes,
            transforms=transforms,
            route_scopes=route_scopes,
            riser_relations=riser_relations,
        )
        candidates.append(candidate)
        if segment is not None:
            accepted.append(segment)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(cross_sheet_runs.get("document", {}))),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _canonical_sha256(sheet_registry),
        },
        "m3_5_contract_ref": {
            "layer": outlined_route_composites.get("layer"),
            "schema_version": outlined_route_composites.get("schema_version"),
            "payload_sha256": _canonical_sha256(outlined_route_composites),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": _canonical_sha256(attribute_bindings),
        },
        "m5_contract_ref": {
            "layer": cross_sheet_runs.get("layer"),
            "schema_version": cross_sheet_runs.get("schema_version"),
            "payload_sha256": _canonical_sha256(cross_sheet_runs),
        },
        "segment_candidates": candidates,
        "bounded_local_3d_segments": accepted,
        "unresolved_vertical_spans": sorted(
            [
                deepcopy(span)
                for segment in accepted
                for span in segment.get("unresolved_vertical_spans", [])
            ],
            key=lambda row: row["id"],
        ),
        "summary": {
            "candidate_count": len(candidates),
            "accepted_bounded_local_3d_segment_count": len(accepted),
            "abstained_candidate_count": sum(row["state"] == "abstained" for row in candidates),
            "unresolved_vertical_span_count": sum(
                len(row.get("unresolved_vertical_spans", [])) for row in accepted
            ),
            "physical_run_hypothesis_count": 0,
        },
        "exchange_contract": {
            "accepted_registered_metric_plan_frame_required": True,
            "drawing_derived_outer_width_required": True,
            "nominal_size_is_not_physical_width": True,
            "duplicate_occurrence_agreement_required": True,
            "bottom_elevation_offset_uses_measured_outer_width": True,
            "analysis_caps_are_nonphysical_boundaries": True,
            "m5b_physical_run_stitching_separate": True,
            "physical_run_identity_established": False,
            "physical_continuation_established": False,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "calculated_severity_emitted": False,
            "m7_takeoff_emitted": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    if duct_terminology is not None:
        if attribute_bindings['m3_5_contract_ref']['payload_sha256'] != _canonical_sha256(outlined_route_composites):
            raise ValueError('duct M5A composite snapshot differs from frozen M4')
        payload['page_local_ducts'] = _page_local_duct_extension(
            sheet_registry, duct_terminology, attribute_bindings)
    errors = validate_mep_bounded_local_3d_segments(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_mep_bounded_local_3d_segments(payload: Mapping[str, Any]) -> list[str]:
    """Validate M5A reference closure and recursively reject downstream authority."""

    errors: list[str] = []
    if 'page_local_ducts' in payload:
        extension = payload['page_local_ducts']
        accepted = {r['id']: r for r in extension['segment_candidates'] if r['state'] == 'accepted'}
        segments = extension['bounded_local_3d_segments']
        if sorted(r['candidate_ref'] for r in segments) != sorted(accepted):
            errors.append('duct accepted candidates and segments differ')
        for candidate in extension['segment_candidates']:
            if (candidate['state'] == 'accepted') != (not candidate['reasons']):
                errors.append('duct candidate state and reasons disagree')
            if candidate['state'] == 'accepted' and candidate['section_evidence']['state'] != 'accepted':
                errors.append('duct cross-section is not resolved')
        for segment in segments:
            section = segment['cross_section']
            xyz = segment['centreline_points_xyz_m']
            if (section['basis'] != 'exterior' or len(xyz) != 2
                    or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in xyz)
                    or min(section['width_m'],section['height_m']) <= 0
                    or math.dist(xyz[0][:2],xyz[1][:2]) <= 0):
                errors.append('invalid bounded rectangular duct geometry')
                continue
            candidate = accepted.get(segment['candidate_ref'])
            if candidate is None:
                continue
            evidence = candidate['section_evidence']
            alternatives = evidence['orientation_and_unit_alternatives']
            if (len(alternatives) != 1 or evidence['dimension_basis'] != 'exterior'
                    or section['width_m'] != alternatives[0]['plan_width_m']
                    or section['height_m'] != alternatives[0]['vertical_height_m']):
                errors.append('duct dimensions differ from resolved section evidence')
            inputs = candidate['geometry_inputs']
            elevation = inputs['elevation']
            z = _metres(elevation.get('value'), elevation.get('unit'))
            if z is None or elevation.get('basis') not in {'bottom','centreline'}:
                errors.append('duct elevation has no accepted bottom or centreline basis')
                continue
            z += section['height_m']/2 if elevation['basis']=='bottom' else 0
            expected = [[round(p[0]*inputs['scale_m_per_display_point'],8),
                         round(p[1]*inputs['scale_m_per_display_point'],8),round(z,8)]
                        for p in inputs['centreline_points_display']]
            if xyz != expected or segment['route_composite_ref'] != candidate['route_composite_ref']:
                errors.append('duct XYZ does not replay from its own interval and datum')
            if segment['local_envelope'] != _duct_surface(xyz, section['width_m'],section['height_m']):
                errors.append('duct side surfaces or open analysis ends do not replay')
            if len(segment['analysis_boundaries']) != 2 or any(
                    r['state'] != 'unresolved' or r['physical_boundary_established'] is not False
                    or r['point_xyz_m'] != p for r,p in zip(segment['analysis_boundaries'],xyz)):
                errors.append('duct analysis boundary changed into a physical end')
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in ("m1_contract_ref", "m3_5_contract_ref", "m4_contract_ref", "m5_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")

    false_authority = {
        "physical_run_identity_established",
        "physical_continuation_established",
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
        "calculated_severity_emitted",
        "m7_takeoff_emitted",
        "quantity_eligible",
        "clash_authority",
    }
    forbidden = {
        "installed_length",
        "fitting_count",
        "confirmed_clash",
        "calculated_severity",
        "clash_status",
        "takeoff",
        "quantity",
        "physical_run_hypotheses",
    }
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key in forbidden:
            errors.append(f"{location}: forbidden M5B/M6/M7 output")
        if key in false_authority and value is not False:
            errors.append(f"{location}: authority flag must remain false")

    candidates = list(payload.get("segment_candidates", []))
    candidate_ids = [str(row.get("id")) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("segment candidate IDs must be unique")
    accepted_candidate_refs = {
        str(row.get("canonical_projected_segment_ref"))
        for row in candidates
        if row.get("state") == "accepted"
    }
    for row in candidates:
        if row.get("state") == "accepted":
            if row.get("reasons") or not all(row.get("certificates", {}).values()):
                errors.append(f"{row.get('id')}: accepted candidate has an open gate")
        elif row.get("state") == "abstained":
            if not row.get("reasons"):
                errors.append(f"{row.get('id')}: abstention lacks reason")
        else:
            errors.append(f"{row.get('id')}: invalid candidate state")

    segments = list(payload.get("bounded_local_3d_segments", []))
    segment_ids = [str(row.get("id")) for row in segments]
    if len(set(segment_ids)) != len(segment_ids):
        errors.append("bounded segment IDs must be unique")
    for row in segments:
        identifier = str(row.get("id"))
        if row.get("record_type") != "bounded_local_3d_segment":
            errors.append(f"{identifier}: record_type mismatch")
        if row.get("label") != LABEL:
            errors.append(f"{identifier}: required unresolved-continuation label missing")
        if row.get("state") != "accepted" or not all(row.get("certificates", {}).values()):
            errors.append(f"{identifier}: accepted segment has an open gate")
        if str(row.get("canonical_projected_segment_ref")) not in accepted_candidate_refs:
            errors.append(f"{identifier}: segment lacks accepted candidate")
        points = row.get("centreline_points_xyz_m", [])
        if not isinstance(points, list) or len(points) < 2 or any(
            not isinstance(point, list)
            or len(point) != 3
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
            for point in points
        ):
            errors.append(f"{identifier}: invalid XYZ centreline")
        caps = row.get("analysis_caps", [])
        if len(caps) != 2 or any(
            cap.get("state") != "unresolved"
            or cap.get("physical_boundary_established") is not False
            or cap.get("continuation_established") is not False
            or cap.get("clash_authority") is not False
            for cap in caps
        ):
            errors.append(f"{identifier}: analysis caps do not preserve unresolved endpoints")
        dimension = row.get("physical_envelope_dimension", {})
        if (
            dimension.get("kind") != "drawing_derived_outer_width"
            or dimension.get("nominal_size_used_as_physical_dimension") is not False
            or _finite_positive(dimension.get("representative_outer_width_m")) is None
        ):
            errors.append(f"{identifier}: physical width lacks drawing-derived certificate")
        if row.get("m1_metric_frame", {}).get("state") != "accepted":
            errors.append(f"{identifier}: metric frame is not accepted")
        if row.get("authority", {}).get("bounded_local_segment_identity_established") is not True:
            errors.append(f"{identifier}: bounded local identity not established")
        authority = row.get("authority", {})
        for key in (
            "physical_run_identity_established",
            "physical_continuation_established",
            "installed_length_emitted",
            "fitting_count_emitted",
            "confirmed_clash_established",
            "calculated_severity_emitted",
            "m7_takeoff_emitted",
            "quantity_eligible",
        ):
            if authority.get(key) is not False:
                errors.append(f"{identifier}: authority.{key} must be explicit false")

    spans = list(payload.get("unresolved_vertical_spans", []))
    for row in spans:
        identifier = str(row.get("id"))
        if (
            row.get("record_type") != "mep_unresolved_vertical_span"
            or row.get("state") != "unknown"
            or row.get("reason") != "second_endpoint_elevation_is_unresolved"
        ):
            errors.append(f"{identifier}: invalid unresolved vertical span")
        if any(key in row for key in ("vertical_extent_m", "resolved_3d_length_m", "installed_length")):
            errors.append(f"{identifier}: unresolved vertical span emitted a finite extent")

    summary = payload.get("summary", {})
    if summary.get("candidate_count") != len(candidates):
        errors.append("summary.candidate_count mismatch")
    if summary.get("accepted_bounded_local_3d_segment_count") != len(segments):
        errors.append("summary.accepted_bounded_local_3d_segment_count mismatch")
    if summary.get("unresolved_vertical_span_count") != len(spans):
        errors.append("summary.unresolved_vertical_span_count mismatch")
    if summary.get("physical_run_hypothesis_count") != 0:
        errors.append("summary.physical_run_hypothesis_count must remain zero")
    contract = payload.get("exchange_contract", {})
    for key in (
        "physical_run_identity_established",
        "physical_continuation_established",
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
        "calculated_severity_emitted",
        "m7_takeoff_emitted",
        "schedule_values_used",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be explicit false")
    return errors
