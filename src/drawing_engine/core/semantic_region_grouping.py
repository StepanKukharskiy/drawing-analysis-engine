"""Drawing-neutral primitive graph and seed-grown semantic regions.

Region rectangles are outputs of grouping. The segmenter first builds nodes
for native text, paths, path items, images, and accepted dimensions. Semantic
seeds then grow components through dimension membership, containment, touching
geometry, and compatible local style. No nominal drawing-column rectangle is
used as an input region.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions
from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader


@dataclass(frozen=True)
class PrimitiveNode:
    node_id: str
    kind: str
    bbox: tuple[float, float, float, float]
    primitive_refs: tuple[str, ...]
    text: str | None = None
    layer: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    orientation: str = "mixed"


@dataclass(frozen=True)
class PrimitiveEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class PrimitiveGraph:
    nodes: tuple[PrimitiveNode, ...]
    edges: tuple[PrimitiveEdge, ...]

    @property
    def by_id(self) -> dict[str, PrimitiveNode]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True)
class RegionProposal:
    proposal_id: str
    kind: str
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    evidence: tuple[str, ...]
    seed_node_ids: tuple[str, ...] = ()
    component_node_ids: tuple[str, ...] = ()
    features: dict[str, Any] = field(default_factory=dict)


SECTION_RE = re.compile(r"^\s*(?P<n>[1-9]\d*)\s*-\s*(?P=n)\s*$")


def _rect_from_points(points: Iterable[fitz.Point], fallback: fitz.Rect) -> fitz.Rect:
    selected = list(points)
    if not selected:
        return fitz.Rect(fallback)
    rect = fitz.Rect(selected[0], selected[0])
    for point in selected[1:]:
        rect |= fitz.Rect(point, point)
    if rect.width < 0.1:
        rect.x0 -= 0.05
        rect.x1 += 0.05
    if rect.height < 0.1:
        rect.y0 -= 0.05
        rect.y1 += 0.05
    return rect


def _item_rect(item: tuple[Any, ...], fallback: fitz.Rect) -> fitz.Rect:
    if item[0] == "re":
        return fitz.Rect(item[1])
    points: list[fitz.Point] = []
    for value in item[1:]:
        if isinstance(value, fitz.Point):
            points.append(value)
        elif isinstance(value, fitz.Quad):
            points.extend((value.ul, value.ur, value.ll, value.lr))
    return _rect_from_points(points, fallback)


def _orientation(rect: fitz.Rect) -> str:
    if rect.width >= max(4.0, rect.height * 4):
        return "horizontal"
    if rect.height >= max(4.0, rect.width * 4):
        return "vertical"
    return "mixed"


def _dimension_rect(item: DimensionAttachment) -> fitz.Rect:
    rect = fitz.Rect(item.text_bbox)
    for segment in (item.baseline, *item.extension_lines):
        rect |= fitz.Rect(fitz.Point(*segment.start), fitz.Point(*segment.end))
    for point in (*item.dimension_points, *item.measured_points):
        x, y = point
        rect |= fitz.Rect(x - 0.1, y - 0.1, x + 0.1, y + 0.1)
    return rect


def build_primitive_graph(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
) -> PrimitiveGraph:
    """Create native primitive nodes plus explicit membership relations."""

    if dimensions is None:
        dimensions = tuple(attach_dimensions(page))
    nodes: list[PrimitiveNode] = []
    edges: list[PrimitiveEdge] = []
    ref_to_item_node: dict[str, str] = {}

    for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            rect = fitz.Rect(line["bbox"])
            nodes.append(
                PrimitiveNode(
                    node_id=f"text[{block_index}].line[{line_index}]",
                    kind="text",
                    bbox=tuple(rect),
                    primitive_refs=(f"text_block[{block_index}].line[{line_index}]",),
                    text=text,
                    style={
                        "font_sizes": sorted({round(float(span.get("size", 0)), 2) for span in line.get("spans", [])}),
                        "fonts": sorted({str(span.get("font", "")) for span in line.get("spans", [])}),
                    },
                    orientation=_orientation(rect),
                )
            )

    for drawing_index, drawing in enumerate(page.get_drawings()):
        path_id = f"drawing[{drawing_index}]"
        rect = fitz.Rect(drawing["rect"])
        layer = drawing.get("layer") or drawing.get("oc")
        style = {
            "stroke": drawing.get("color"),
            "fill": drawing.get("fill"),
            "width": None if drawing.get("width") is None else float(drawing["width"]),
            "dashes": drawing.get("dashes"),
            "stroke_opacity": drawing.get("stroke_opacity"),
            "fill_opacity": drawing.get("fill_opacity"),
            "close_path": bool(drawing.get("closePath", False)),
            "drawing_type": drawing.get("type"),
        }
        nodes.append(
            PrimitiveNode(
                node_id=path_id,
                kind="path",
                bbox=tuple(rect),
                primitive_refs=(path_id,),
                layer=None if layer is None else str(layer),
                style=style,
                orientation=_orientation(rect),
            )
        )
        for item_index, item in enumerate(drawing["items"]):
            item_id = f"{path_id}.item[{item_index}]"
            item_rect = _item_rect(item, rect)
            ref_to_item_node[item_id] = item_id
            nodes.append(
                PrimitiveNode(
                    node_id=item_id,
                    kind="line_segment" if item[0] == "l" else "path_item",
                    bbox=tuple(item_rect),
                    primitive_refs=(item_id,),
                    layer=None if layer is None else str(layer),
                    style={**style, "operator": item[0]},
                    orientation=_orientation(item_rect),
                )
            )
            edges.append(PrimitiveEdge(path_id, item_id, "path_contains_item"))

    for image_index, info in enumerate(page.get_image_info(xrefs=True)):
        rect = fitz.Rect(info["bbox"])
        nodes.append(
            PrimitiveNode(
                node_id=f"image[{image_index}]",
                kind="image",
                bbox=tuple(rect),
                primitive_refs=(f"image_xref[{info.get('xref', 0)}]",),
                style={"width_px": info.get("width"), "height_px": info.get("height")},
                orientation=_orientation(rect),
            )
        )

    for item in dimensions:
        if item.status != "accepted":
            continue
        node_id = item.attachment_id
        nodes.append(
            PrimitiveNode(
                node_id=node_id,
                kind="dimension",
                bbox=tuple(_dimension_rect(item)),
                primitive_refs=(
                    item.baseline.primitive_ref,
                    *(line.primitive_ref for line in item.extension_lines),
                    *item.terminal_refs,
                ),
                text=item.text,
                style={
                    "value_mm": item.value_mm,
                    "scale_points_per_mm": item.scale_points_per_mm,
                    "score": item.score,
                },
                orientation=item.orientation,
            )
        )
        for ref in (
            item.baseline.primitive_ref,
            *(line.primitive_ref for line in item.extension_lines),
            *item.terminal_refs,
        ):
            target = ref_to_item_node.get(ref)
            if target is not None:
                edges.append(PrimitiveEdge(node_id, target, "dimension_contains_primitive"))
    return PrimitiveGraph(tuple(nodes), tuple(edges))


def augment_graph_with_text_roles(
    graph: PrimitiveGraph,
    text_roles: Iterable[dict[str, Any]],
) -> PrimitiveGraph:
    """Add OCR text observations without replacing native PDF primitives."""

    existing = {node.node_id for node in graph.nodes}
    supplemental = []
    for item in text_roles:
        if item["id"] in existing or item.get("text_method") != "geometry_gated_ocr":
            continue
        supplemental.append(
            PrimitiveNode(
                node_id=item["id"],
                kind="text",
                bbox=tuple(item["bbox_display"]),
                primitive_refs=tuple(item.get("primitive_refs", (item["id"],))),
                text=item.get("text"),
                style={
                    "text_method": "geometry_gated_ocr",
                    "confidence": item.get("confidence"),
                    "resolved_role": item.get("resolved_role"),
                },
                orientation=_orientation(fitz.Rect(item["bbox_display"])),
            )
        )
    return PrimitiveGraph((*graph.nodes, *supplemental), graph.edges)


def _distance_between(left: fitz.Rect, right: fitz.Rect) -> float:
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return hypot(dx, dy)


def _union(nodes: Iterable[PrimitiveNode], padding: float = 2.0) -> fitz.Rect:
    selected = list(nodes)
    if not selected:
        raise ValueError("cannot bound an empty semantic component")
    rect = fitz.Rect(selected[0].bbox)
    for node in selected[1:]:
        rect |= fitz.Rect(node.bbox)
    rect += (-padding, -padding, padding, padding)
    return rect


def _section_seed_nodes(graph: PrimitiveGraph) -> list[tuple[str, PrimitiveNode]]:
    seeds = []
    for node in graph.nodes:
        if node.kind != "text" or node.text is None:
            continue
        match = SECTION_RE.match(node.text.replace("–", "-").replace("—", "-"))
        if match:
            seeds.append((match.group("n"), node))
    return sorted(seeds, key=lambda item: (item[1].bbox[1], item[1].bbox[0]))


def _dimension_scale_cluster(
    seed: PrimitiveNode,
    next_seed_y: float,
    dimensions: tuple[DimensionAttachment, ...],
    page: fitz.Page,
) -> list[DimensionAttachment]:
    seed_rect = fitz.Rect(seed.bbox)
    seed_x = (seed_rect.x0 + seed_rect.x1) / 2
    candidates = []
    for item in dimensions:
        if item.status != "accepted":
            continue
        rect = _dimension_rect(item)
        center_x = (rect.x0 + rect.x1) / 2
        if rect.y1 < seed_rect.y0 - 8 or rect.y0 > next_seed_y + 4:
            continue
        if abs(center_x - seed_x) > 0.22 * page.rect.width:
            continue
        candidates.append(item)
    if not candidates:
        return []
    bins: dict[int, list[DimensionAttachment]] = defaultdict(list)
    for item in candidates:
        bins[round(item.scale_points_per_mm / 0.015)].append(item)
    cluster = max(
        bins.values(),
        key=lambda rows: (
            4 * sum(item.orientation == "horizontal" for item in rows) + len(rows),
            -statistics.median(_distance_between(seed_rect, _dimension_rect(item)) for item in rows),
        ),
    )
    return sorted(cluster, key=lambda item: (item.text_bbox[1], item.text_bbox[0]))


def _component_features(nodes: list[PrimitiveNode], label: str, dimensions: list[DimensionAttachment]) -> dict[str, Any]:
    rect = _union(nodes, padding=0)
    paths = [node for node in nodes if node.kind == "path"]
    closed = 0
    for node in paths:
        node_rect = fitz.Rect(node.bbox)
        ratio = min(node_rect.width, node_rect.height) / max(node_rect.width, node_rect.height, 0.1)
        closed += bool(node.style.get("close_path") or (ratio >= 0.65 and float(node.style.get("width") or 0) >= 1.0))
    rebar = sum(float(node.style.get("width") or 0) >= 1.5 or node.style.get("fill") is not None for node in paths)
    long_axis = sum(
        node.orientation in {"horizontal", "vertical"}
        and max(fitz.Rect(node.bbox).width, fitz.Rect(node.bbox).height) > 0.70 * max(rect.width, rect.height)
        for node in paths
    )
    return {
        "aspect_ratio": rect.width / max(rect.height, 0.1),
        "closed_concrete_outlines": closed,
        "rebar_paths": rebar,
        "dimension_chains": len(dimensions),
        "section_labels": [label],
        "table_grid_density": long_axis / max(len(paths), 1),
        "primitive_node_count": len(nodes),
    }


def _section_regions(
    page: fitz.Page,
    graph: PrimitiveGraph,
    dimensions: tuple[DimensionAttachment, ...],
) -> list[RegionProposal]:
    seeds = _section_seed_nodes(graph)
    path_nodes = [node for node in graph.nodes if node.kind == "path"]
    path_by_ref = {node.node_id: node for node in path_nodes}
    text_nodes = [node for node in graph.nodes if node.kind == "text"]
    thin_segments = extract_thin_segments(page)
    proposals: list[RegionProposal] = []
    for index, (number, seed) in enumerate(seeds):
        next_y = fitz.Rect(seeds[index + 1][1].bbox).y0 if index + 1 < len(seeds) else page.rect.height
        dimension_group = _dimension_scale_cluster(seed, next_y, dimensions, page)
        dimension_nodes = [
            PrimitiveNode(
                node_id=item.attachment_id,
                kind="dimension",
                bbox=tuple(_dimension_rect(item)),
                primitive_refs=(item.attachment_id,),
                text=item.text,
                style={"value_mm": item.value_mm, "scale_points_per_mm": item.scale_points_per_mm},
                orientation=item.orientation,
            )
            for item in dimension_group
        ]
        skeleton = [seed, *dimension_nodes]
        if not dimension_nodes:
            seed_rect = fitz.Rect(seed.bbox)
            seed_x = (seed_rect.x0 + seed_rect.x1) / 2
            host_candidates = []
            for node in path_nodes:
                rect = fitz.Rect(node.bbox)
                ratio = min(rect.width, rect.height) / max(rect.width, rect.height, 0.1)
                if rect.y0 < seed_rect.y1 or rect.y1 > next_y + 6 or min(rect.width, rect.height) < 35 or ratio < 0.25:
                    continue
                if abs((rect.x0 + rect.x1) / 2 - seed_x) > 0.20 * page.rect.width:
                    continue
                host_candidates.append(node)
            if not host_candidates:
                continue
            host = min(host_candidates, key=lambda node: _distance_between(seed_rect, fitz.Rect(node.bbox)))
            skeleton.append(host)
        corridor = _union(skeleton, padding=max(10.0, page.rect.width * 0.006))
        component_paths = []
        for node in path_nodes:
            rect = fitz.Rect(node.bbox)
            if not corridor.intersects(rect):
                continue
            if rect.width > 0.55 * page.rect.width or rect.height > 0.55 * page.rect.height:
                continue
            component_paths.append(node)
        grown = _union([*skeleton, *component_paths], padding=6.0)
        component_text = [
            node
            for node in text_nodes
            if grown.intersects(fitz.Rect(node.bbox))
            and fitz.Rect(node.bbox).y0 >= fitz.Rect(seed.bbox).y0 - 3
            and fitz.Rect(node.bbox).y1 <= next_y + 6
        ]
        leader_paths: list[PrimitiveNode] = []
        for node in text_nodes:
            node_rect = fitz.Rect(node.bbox)
            if node in component_text or node_rect.y0 < fitz.Rect(seed.bbox).y0 - 3 or node_rect.y1 > next_y + 6:
                continue
            if _distance_between(grown, node_rect) > 0.07 * page.rect.height:
                continue
            trace = trace_leader(node_rect, thin_segments, search_radius=170, max_hops=10)
            if not any(fitz.Point(*terminal) in grown + (-12, -12, 12, 12) for terminal in trace["terminals"]):
                continue
            component_text.append(node)
            for segment in trace["segments"]:
                drawing_ref = segment["drawing_ref"].split(".item[")[0]
                path_node = path_by_ref.get(drawing_ref)
                if path_node is not None:
                    leader_paths.append(path_node)
        component = list({node.node_id: node for node in [*skeleton, *component_paths, *component_text, *leader_paths]}.values())
        bbox = _union(component, padding=3.0) & page.rect
        features = _component_features(component, f"{number}-{number}", dimension_group)
        proposals.append(
            RegionProposal(
                proposal_id=f"section_{number}_{number}.{index + 1:02d}",
                kind="section_view",
                label=f"section {number}-{number}",
                bbox=tuple(bbox),
                confidence=0.94 if features["dimension_chains"] >= 3 and features["rebar_paths"] else 0.82,
                evidence=(
                    f"semantic seed {seed.node_id}: {number}-{number}",
                    f"grew through {len(dimension_group)} compatible dimension nodes",
                    f"component contains {len(component_paths)} native path nodes",
                    "bbox is the union of grouped primitive nodes",
                ),
                seed_node_ids=(seed.node_id,),
                component_node_ids=tuple(sorted({node.node_id for node in component})),
                features=features,
            )
        )
    return proposals


def _elevation_region(
    page: fitz.Page,
    graph: PrimitiveGraph,
    dimensions: tuple[DimensionAttachment, ...],
    *,
    kind: str,
    label: str,
    patterns: tuple[str, ...],
) -> RegionProposal | None:
    seeds = [
        node
        for node in graph.nodes
        if node.kind == "text" and node.text and any(pattern.casefold() in node.text.casefold() for pattern in patterns)
    ]
    vertical = [item for item in dimensions if item.status == "accepted" and item.orientation == "vertical"]
    if not seeds or not vertical:
        return None
    best: tuple[float, PrimitiveNode, DimensionAttachment] | None = None
    for seed in seeds:
        for item in vertical:
            score = _distance_between(fitz.Rect(seed.bbox), _dimension_rect(item)) - item.value_mm / 500
            if best is None or score < best[0]:
                best = (score, seed, item)
    assert best is not None
    _, seed, dimension = best
    dimension_node = PrimitiveNode(
        node_id=dimension.attachment_id,
        kind="dimension",
        bbox=tuple(_dimension_rect(dimension)),
        primitive_refs=(dimension.attachment_id,),
        text=dimension.text,
        style={"value_mm": dimension.value_mm},
        orientation="vertical",
    )
    skeleton = [seed, dimension_node]
    corridor = _union(skeleton, padding=0.065 * page.rect.width)
    paths = [
        node
        for node in graph.nodes
        if node.kind == "path"
        and corridor.intersects(fitz.Rect(node.bbox))
        and fitz.Rect(node.bbox).height <= 0.95 * page.rect.height
        and fitz.Rect(node.bbox).width <= 0.35 * page.rect.width
    ]
    component = [*skeleton, *paths]
    bbox = _union(component, padding=3.0) & page.rect
    features = {
        "aspect_ratio": bbox.width / max(bbox.height, 0.1),
        "rebar_paths": sum(float(node.style.get("width") or 0) >= 1.5 for node in paths),
        "dimension_chains": 1,
        "semantic_seed": seed.text,
        "primitive_node_count": len(component),
    }
    return RegionProposal(
        proposal_id=f"{kind}.01",
        kind=kind,
        label=label,
        bbox=tuple(bbox),
        confidence=0.89,
        evidence=(
            f"semantic seed {seed.node_id}: {seed.text}",
            f"overall vertical dimension {dimension.value_mm:g}",
            f"component contains {len(paths)} native path nodes",
            "bbox is the union of grouped primitive nodes",
        ),
        seed_node_ids=(seed.node_id,),
        component_node_ids=tuple(sorted(node.node_id for node in component)),
        features=features,
    )


def _detail_sketch_region(page: fitz.Page, graph: PrimitiveGraph) -> RegionProposal | None:
    width, height = page.rect.width, page.rect.height
    images = [node for node in graph.nodes if node.kind == "image"]
    images = [
        node
        for node in images
        if 0.025 * width <= fitz.Rect(node.bbox).width <= 0.25 * width
        and 0.025 * height <= fitz.Rect(node.bbox).height <= 0.20 * height
    ]
    if len(images) < 4:
        return None
    height_bins: dict[int, list[PrimitiveNode]] = defaultdict(list)
    for node in images:
        height_bins[round(fitz.Rect(node.bbox).height / (0.01 * height))].append(node)
    repeated = max(height_bins.values(), key=len)
    if len(repeated) < 4:
        return None
    bbox = _union(repeated, padding=max(4.0, 0.004 * width)) & page.rect
    return RegionProposal(
        proposal_id="repeated_detail_sketches.01",
        kind="bar_detail_group",
        label="repeated bar-detail sketches",
        bbox=tuple(bbox),
        confidence=0.93,
        evidence=(
            f"{len(repeated)} similarly sized embedded-image nodes",
            "vertical repetition component",
            "bbox is the union of grouped primitive nodes",
        ),
        seed_node_ids=tuple(node.node_id for node in repeated),
        component_node_ids=tuple(node.node_id for node in repeated),
        features={
            "embedded_images": len(repeated),
            "vertical_repetition": True,
            "table_grid_density": 0.0,
            "primitive_node_count": len(repeated),
        },
    )


def propose_semantic_regions(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
    graph: PrimitiveGraph | None = None,
) -> tuple[RegionProposal, ...]:
    """Grow semantic components from labels/images and return their unions."""

    if dimensions is None:
        dimensions = tuple(attach_dimensions(page))
    if graph is None:
        graph = build_primitive_graph(page, dimensions)
    proposals: list[RegionProposal] = []
    for candidate in (
        _elevation_region(
            page,
            graph,
            dimensions,
            kind="formwork_elevation",
            label="formwork elevation",
            patterns=("опалуб", "formwork"),
        ),
        _elevation_region(
            page,
            graph,
            dimensions,
            kind="reinforcement_elevation",
            label="reinforcement elevation",
            patterns=("армирован", "reinforcement"),
        ),
    ):
        if candidate is not None:
            proposals.append(candidate)
    proposals.extend(_section_regions(page, graph, dimensions))
    detail = _detail_sketch_region(page, graph)
    if detail is not None:
        proposals.append(detail)
    return tuple(proposals)
