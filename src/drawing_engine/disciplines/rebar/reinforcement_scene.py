"""Drawing-neutral reinforcement centerline scene records.

This module deliberately knows nothing about sheet names or drawing layouts.
An upstream semantic adapter supplies one or more 3D polylines for each bar
group and states whether their placement is observed, inferred, or unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class RebarPath:
    mark: str
    role: str
    instance: int
    diameter_mm: float | None
    placement_status: str
    points_xyz_mm: tuple[Point3D, ...]
    closed: bool
    evidence: tuple[str, ...]


def reinforcement_from_semantics(data: dict[str, Any]) -> tuple[RebarPath, ...]:
    """Validate and normalize generic rebar centerline records."""

    paths: list[RebarPath] = []
    for item in data.get("reinforcement", []):
        points = tuple(tuple(float(value) for value in point) for point in item["points_xyz_mm"])
        if len(points) < 2 or any(len(point) != 3 for point in points):
            raise ValueError("each reinforcement path needs at least two XYZ points")
        diameter = item.get("diameter_mm")
        if diameter is not None and float(diameter) <= 0:
            raise ValueError("reinforcement diameter must be positive or null")
        status = item.get("placement_status", "unknown")
        if status not in {"drawing_constrained", "reviewed_schematic", "unknown"}:
            raise ValueError(f"unsupported reinforcement placement_status: {status}")
        paths.append(
            RebarPath(
                mark=str(item["mark"]),
                role=str(item["role"]),
                instance=int(item["instance"]),
                diameter_mm=None if diameter is None else float(diameter),
                placement_status=status,
                points_xyz_mm=points,
                closed=bool(item.get("closed", False)),
                evidence=tuple(item.get("evidence", [])),
            )
        )
    return tuple(paths)


def reinforcement_payload(paths: tuple[RebarPath, ...]) -> dict[str, Any]:
    marks = sorted({path.mark for path in paths}, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    return {
        "path_count": len(paths),
        "marks": marks,
        "counts_by_mark": {
            mark: sum(path.mark == mark for path in paths)
            for mark in marks
        },
        "status_counts": {
            status: sum(path.placement_status == status for path in paths)
            for status in ("drawing_constrained", "reviewed_schematic", "unknown")
        },
        "paths": [
            {
                "mark": path.mark,
                "role": path.role,
                "instance": path.instance,
                "diameter_mm": path.diameter_mm,
                "placement_status": path.placement_status,
                "points_xyz_mm": [list(point) for point in path.points_xyz_mm],
                "closed": path.closed,
                "evidence": list(path.evidence),
            }
            for path in paths
        ],
    }
