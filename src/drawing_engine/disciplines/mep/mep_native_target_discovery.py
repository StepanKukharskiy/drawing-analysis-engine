"""Automatic, bounded native target search behind MEP annotation observations.

This is the M3 discovery input, not M4 binding. Every nearby native stroke is
an alternative; proximity, a familiar tag or a single surviving candidate never
establishes applicability or physical identity. Search limits stay explicit.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile
from time import perf_counter

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_item_ocr_observations import build_mep_item_ocr_proposals
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import propose_mep_interpretations, _stable_id
from src.drawing_engine.disciplines.mep.mep_text_observations import build_mep_document_text_proposals
from src.drawing_engine.core.vector_topology import iter_native_segments


LAYER = "mep_native_target_discovery"
VERSION = "0.1.0"
SEARCH_BOUNDS_METHOD = "cubic_control_hull_and_linear_points"
_TAG = re.compile(r"(?<![A-Z0-9])([A-Z]{1,8})\s*-\s*(\d+[A-Z]?)(?![A-Z0-9.])", re.I)
_EXCLUDED = {"schedule", "table", "cut_sheet", "legend", "note", "title"}
_FLOAT32 = struct.Struct("<f")


def _intersects(left, right):
    # Inclusive intersections preserve horizontal/vertical zero-area paths.
    return left[0] <= right[2] and right[0] <= left[2] and left[1] <= right[3] and right[1] <= left[3]


def _cells(box, size):
    for x in range(math.floor(box[0] / size), math.floor(box[2] / size) + 1):
        for y in range(math.floor(box[1] / size), math.floor(box[3] / size) + 1):
            yield x, y


def _search_rect(box, page_size, radius_in_heights):
    radius = max(1.0, min(box[2] - box[0], box[3] - box[1])) * radius_in_heights
    return [max(0, box[0] - radius), max(0, box[1] - radius),
            min(page_size[0], box[2] + radius), min(page_size[1], box[3] + radius)]


def _point_bounds(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def _native_display_geometry(native, rotation):
    def transformed(point):
        # Avoid constructing two SWIG Point objects for every line.  On dense
        # real pages this conversion dominated the supposedly compact scan.
        x, y = point
        # fitz.Point stores float32 coordinates. Preserve that canonical value
        # without allocating SWIG objects for every primitive.
        return [_FLOAT32.unpack(_FLOAT32.pack(
                    x * rotation.a + y * rotation.c + rotation.e))[0],
                _FLOAT32.unpack(_FLOAT32.pack(
                    x * rotation.b + y * rotation.d + rotation.f))[0]]

    points = native["sample_points_display"] or [native["start_display"], native["end_display"]]
    points = [transformed(point) for point in points]
    box = _point_bounds(points)
    controls = ([transformed(point) for point in native["control_points_display"]]
                if native["kind"] == "cubic" else [])
    # A Bezier stays inside its control hull, but not necessarily inside the
    # finite sample bounds. This box is only a conservative search envelope;
    # neither its corners nor its area are new observed route geometry.
    return points, box, _point_bounds([*points, *controls]) if controls else list(box)


def _drawing_search_box(drawing, rotation):
    rect = fitz.Rect(drawing["rect"]) * rotation
    controls = [list(fitz.Point(point) * rotation) for item in drawing.get("items", [])
                if item[0] == "c" for point in item[1:5]]
    return _point_bounds([[rect.x0, rect.y0], [rect.x1, rect.y1], *controls])


def _seeds(observations, invalid_refs):
    seeds = []
    for row in observations:
        if row["id"] in invalid_refs or row.get("region_role") in _EXCLUDED:
            continue
        proposals = propose_mep_interpretations(row)
        kinds = sorted({proposal["proposal_type"] for proposal in proposals})
        tags = sorted({f"{match[1].upper()}-{match[2].upper()}" for match in _TAG.finditer(row["text"])})
        if not kinds and not tags:
            continue
        seeds.append({
            "id": _stable_id("mep_native_target_search", row["page_ref"], row["id"]),
            "observation_ref": row["id"], "page_ref": row["page_ref"],
            "text": row["text"], "bbox_display": deepcopy(row["bbox_display"]),
            "proposal_refs": [proposal["id"] for proposal in proposals],
            "proposal_types": kinds, "unclassified_tag_tokens": tags,
            "tag_tokens_are_not_equipment_identity": True,
            "primitive_candidate_refs": [], "cross_boundary_primitive_refs": [],
            "candidate_count": 0, "unretained_candidate_count": 0,
            "state": "not_processed", "unresolved_reasons": [],
            "quantity_eligible": False,
        })
    return seeds


def _region_search(page_ref, box):
    return {"id": _stable_id("mep_native_target_region", page_ref, box),
            "page_ref": page_ref, "seed_kind": "geometry_first_region",
            "search_bounds_method": SEARCH_BOUNDS_METHOD,
            "observation_ref": None, "text": "", "bbox_display": box,
            "search_rect_display": box, "proposal_refs": [], "proposal_types": [],
            "unclassified_tag_tokens": [], "tag_tokens_are_not_equipment_identity": True,
            "primitive_candidate_refs": [], "cross_boundary_primitive_refs": [],
            "candidate_count": 0, "unretained_candidate_count": 0,
            "unsupported_native_item_kinds": {}, "unsupported_native_item_refs": [],
            "quantity_eligible": False, "unique_target_established": False}


def _finish_search(row):
    row["state"] = "budget_limited" if row["unretained_candidate_count"] else "searched_unbound"
    row["unique_target_established"] = False
    row["unresolved_reasons"] = ["geometric_target_applicability_not_certified"]
    if not row["relevant_competitor_search_complete"]:
        row["unresolved_reasons"].append("relevant_competitor_search_incomplete")
    if row["unretained_candidate_count"]:
        row["unresolved_reasons"].append("candidate_budget_exhausted")
    if row["cross_boundary_primitive_refs"]:
        row["unresolved_reasons"].append("candidate_geometry_extends_beyond_search_window")
    if row.get("native_overlap_alternatives") or row.get("ocr_overlap_alternatives"):
        row["unresolved_reasons"].append("overlapping_text_observations_unresolved")
    row["primitive_candidate_refs"].sort()
    row["cross_boundary_primitive_refs"].sort()


def _bounded_page_records(page, page_ref, size, budget, annotations=(), radius=8.0,
                          candidate_sink=None, include_unretained=False,
                          stream_native_drawings=False, sink_all_candidates=False):
    """One canonical native scan; retention is independently bounded per tile.

    A stroke crossing a tile boundary is an alternative in every touched tile,
    not clipped geometry. Annotation windows count even unretained competitors.
    Neither a complete tile nor a single candidate certifies an item.
    """
    if not math.isfinite(size) or size <= 0:
        raise ValueError("region size must be finite and positive")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("region candidate budget must be positive")
    width, height = page.rect.width, page.rect.height
    regions, by_cell = [], {}
    for x in range(math.ceil(width / size)):
        for y in range(math.ceil(height / size)):
            box = [x * size, y * size, min(width, (x + 1) * size), min(height, (y + 1) * size)]
            row = _region_search(page_ref, box)
            if include_unretained:
                row["_all_candidate_refs"] = []
            regions.append(row)
            by_cell[x, y] = row

    def touched(box):
        bounded = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        if bounded[0] > bounded[2] or bounded[1] > bounded[3]:
            return []
        # Include both neighbours when a zero-area stroke lies on a boundary.
        bounded[0] = math.nextafter(bounded[0], -math.inf)
        bounded[1] = math.nextafter(bounded[1], -math.inf)
        return [by_cell[cell] for cell in _cells(bounded, size) if cell in by_cell]

    annotations = deepcopy(list(annotations))
    annotation_index = defaultdict(set)
    annotation_by_id = {row["id"]: row for row in annotations}
    for row in annotations:
        row["search_rect_display"] = _search_rect(row["bbox_display"], [width, height], radius)
        row["region_refs"] = [region["id"] for region in touched(row["search_rect_display"])]
        for ref in row["region_refs"]:
            annotation_index[ref].add(row["id"])

    def add(row, candidate, box, retained):
        row["candidate_count"] += 1
        if not retained:
            row["unretained_candidate_count"] += 1
            return
        row["primitive_candidate_refs"].append(candidate)
        window = row["search_rect_display"]
        if any((box[0] < window[0], box[1] < window[1], box[2] > window[2], box[3] > window[3])):
            row["cross_boundary_primitive_refs"].append(candidate)

    rotation = page.rotation_matrix
    unsupported = Counter()
    drawing_count = 0
    candidates = []

    def consume_drawing(drawing):
        nonlocal drawing_count
        drawing_index = drawing_count
        drawing_count += 1
        kinds = Counter(str(item[0]) for item in drawing.get("items", []) if item[0] not in {"l", "re", "qu", "c"})
        if kinds:
            unsupported.update(kinds)
            for region in touched(list(fitz.Rect(drawing["rect"]) * rotation)):
                region["unsupported_native_item_kinds"] = dict(Counter(region["unsupported_native_item_kinds"]) + kinds)
                region["unsupported_native_item_refs"].extend(
                    f"drawing[{drawing_index}].item[{item_index}]"
                    for item_index, item in enumerate(drawing.get("items", [])) if item[0] not in {"l", "re", "qu", "c"})
        for native in iter_native_segments((drawing,), drawing_index_offset=drawing_index):
            points, geometry_box, box = _native_display_geometry(native, rotation)
            matches = touched(box)
            if not matches:
                continue
            retained_regions = [region for region in matches if len(region["primitive_candidate_refs"]) < budget]
            identifier = (_stable_id("mep_native_target_primitive", page_ref, native["id"])
                          if retained_regions or include_unretained else None)
            retained_ids = {region["id"] for region in retained_regions}
            for region in matches:
                add(region, identifier, box, region["id"] in retained_ids)
                if include_unretained:
                    region["_all_candidate_refs"].append(identifier)
            possible = {ref for region in matches for ref in annotation_index[region["id"]]}
            linked = [annotation_by_id[ref] for ref in sorted(possible)
                      if _intersects(box, annotation_by_id[ref]["search_rect_display"])]
            for row in linked:
                add(row, identifier, box, bool(retained_regions))
            if retained_regions or include_unretained or sink_all_candidates:
                candidate = {"id": identifier, "page_ref": page_ref,
                    "source_primitive_ref": native["id"], "source_native_segment": native,
                    "source_drawing_close_path": bool(drawing.get("closePath")),
                    "points_display": points, "bbox_display": geometry_box,
                    "search_bbox_display": box, "search_bbox_is_geometry": False,
                    "pdf_to_display_matrix": list(rotation),
                    "search_refs": sorted([*retained_ids, *(row["id"] for row in linked)]),
                    "state": "observed", "role": "unclassified_native_geometry_candidate",
                    "geometry_is_not_a_route_or_equipment": True, "quantity_eligible": False}
                if candidate_sink is None:
                    candidates.append(candidate)
                else:
                    candidate_sink(candidate)

    if stream_native_drawings:
        # PyMuPDF's callback API releases each path after it is consumed.
        # Calling get_drawings() would first construct millions of converted
        # path objects, which is the measured page-5 cold-path RSS spike.
        # The list-returning API also coalesces an immediately adjacent fill
        # and stroke of the same path into one ``fs`` record. Preserve that
        # drawing ordinal contract with a one-path lookahead.
        pending = None
        def consume_compact_drawing(drawing):
            nonlocal pending
            if (pending is not None and pending.get("type") == "f"
                    and drawing.get("type") == "s"
                    and drawing.get("seqno") == pending.get("seqno") + 1
                    and drawing.get("items") == pending.get("items")
                    and drawing.get("rect") == pending.get("rect")):
                merged = {**pending, **drawing, "type": "fs", "seqno": pending["seqno"]}
                consume_drawing(merged)
                pending = None
                return True
            if pending is not None:
                consume_drawing(pending)
            pending = drawing
            return True
        page.get_cdrawings(callback=consume_compact_drawing)
        if pending is not None:
            consume_drawing(pending)
    else:
        # Exact compatibility path for public materialized discovery APIs.
        for drawing in page.get_drawings():
            consume_drawing(drawing)
    for region in regions:
        region["relevant_competitor_search_complete"] = not region["unretained_candidate_count"] and not region["unsupported_native_item_kinds"]
        region["native_scan_complete"] = True
    by_id = {region["id"]: region for region in regions}
    for row in annotations:
        row["relevant_competitor_search_complete"] = all(by_id[ref]["relevant_competitor_search_complete"] for ref in row["region_refs"])
    for row in [*regions, *annotations]:
        _finish_search(row)
    return {"regions": regions, "searches": [*regions, *annotations], "primitive_candidates": candidates,
            "native_drawing_record_count": drawing_count, "unsupported_native_item_kinds": dict(unsupported)}


def _refined_bounded_native_page_regions(page, page_ref, *, region_size_display_points,
                                         max_candidates_per_region,
                                         minimum_region_size_display_points,
                                         candidate_sink, candidate_reader,
                                         candidate_reader_many=None,
                                         performance_profile=None,
                                         stream_native_drawings=False):
    """Scan/refine once using caller-owned exact candidate storage.

    Source IDs stay page-global. Repeated IDs in neighbouring tiles are the same
    source stroke, not additive items. The page must remain open while iterating.
    The scan retains at most one region budget per tile, not an unlimited page.
    Optional refinement subdivides only overloaded tiles, using all spooled
    sources, until the independent region budget closes or the minimum size
    prevents a further split. Source geometry is never clipped or renumbered.
    """
    minimum = minimum_region_size_display_points
    if minimum is not None and (not math.isfinite(minimum) or minimum <= 0 or minimum > region_size_display_points):
        raise ValueError("minimum region size must be positive and no greater than initial region size")
    started = perf_counter()
    scanned = _bounded_page_records(page, page_ref, region_size_display_points,
                                    max_candidates_per_region, candidate_sink=candidate_sink,
                                    include_unretained=minimum is not None,
                                    stream_native_drawings=stream_native_drawings)
    if performance_profile is not None:
        performance_profile["native_descriptor_scan_seconds"] = perf_counter() - started
        performance_profile["native_drawing_record_count"] = scanned["native_drawing_record_count"]

    refinement_started = perf_counter()
    decoded_for_refinement = 0
    def refine(region):
        nonlocal decoded_for_refinement
        all_refs = region.pop("_all_candidate_refs")
        x0, y0, x1, y1 = region["bbox_display"]
        xs = [x0, (x0 + x1) / 2, x1] if x1 - x0 >= 2 * minimum else [x0, x1]
        ys = [y0, (y0 + y1) / 2, y1] if y1 - y0 >= 2 * minimum else [y0, y1]
        if not region["unretained_candidate_count"] or len(xs) == len(ys) == 2:
            if region["unretained_candidate_count"]:
                region["unresolved_reasons"].append("minimum_region_size_reached_with_unretained_competitors")
            return [region]
        children = []
        ancestry = [*region.get("refinement_ancestry", []), {
            key: deepcopy(region[key]) for key in ("id", "bbox_display", "candidate_count", "unretained_candidate_count")}]
        for left, right in zip(xs, xs[1:]):
            for top, bottom in zip(ys, ys[1:]):
                child = _region_search(page_ref, [left, top, right, bottom])
                child.update(_all_candidate_refs=[], refinement_ancestry=deepcopy(ancestry),
                             native_scan_complete=True,
                             unsupported_native_item_kinds=deepcopy(region["unsupported_native_item_kinds"]),
                             unsupported_native_item_refs=list(region["unsupported_native_item_refs"]))
                children.append(child)
        candidates = (candidate_reader_many(all_refs) if candidate_reader_many is not None
                      else ((ref, candidate_reader(ref)) for ref in all_refs))
        for ref, candidate in candidates:
            decoded_for_refinement += 1
            box = candidate.get("search_bbox_display", candidate["bbox_display"])
            for child in children:
                window = child["bbox_display"]
                if not _intersects(box, window):
                    continue
                child["_all_candidate_refs"].append(ref)
                child["candidate_count"] += 1
                if len(child["primitive_candidate_refs"]) == max_candidates_per_region:
                    child["unretained_candidate_count"] += 1
                    continue
                child["primitive_candidate_refs"].append(ref)
                if box[0] < window[0] or box[1] < window[1] or box[2] > window[2] or box[3] > window[3]:
                    child["cross_boundary_primitive_refs"].append(ref)
        leaves = []
        for child in children:
            child["relevant_competitor_search_complete"] = not child["unretained_candidate_count"] and not child["unsupported_native_item_kinds"]
            _finish_search(child)
            leaves.extend(refine(child))
        return leaves

    regions = ([leaf for region in scanned["regions"] for leaf in refine(region)]
               if minimum is not None else scanned["regions"])
    if performance_profile is not None:
        performance_profile["region_refinement_seconds"] = perf_counter() - refinement_started
        performance_profile["refinement_candidate_decode_count"] = decoded_for_refinement
        performance_profile["region_count"] = len(regions)
    return regions


def _refined_regions_from_descriptor_pack(initial_regions, descriptor_reader, *, page_ref,
                                            page_size, max_candidates_per_region,
                                            minimum_region_size_display_points,
                                            candidate_cells=None,
                                            candidate_reader_at=None,
                                            initial_retention_max_ordinal=None):
    """Rebuild the exact adaptive partition from sequential descriptors.

    Unlike the legacy refiner, this never keeps every candidate ID in Python
    lists and needs no random lookup table.  Each refinement depth is one
    sequential pack scan.  Final retained references remain byte-for-byte
    compatible with the previous evidence; excluded sources receive separate
    count/hash/bounds receipts.
    """
    minimum = minimum_region_size_display_points
    width, height = page_size
    roots = []
    for ordinal, source in enumerate(initial_regions):
        row = deepcopy(source)
        row['_order_key'] = (ordinal,)
        row['_retention_max_ordinal'] = (initial_retention_max_ordinal or {}).get(
            row['id'])
        roots.append(row)
    active, finished = roots, []
    decoded_for_refinement = 0

    def can_split(region):
        x0, y0, x1, y1 = region['bbox_display']
        return (region['unretained_candidate_count']
                and (x1 - x0 >= 2 * minimum or y1 - y0 >= 2 * minimum))

    while True:
        split = [row for row in active if can_split(row)]
        split_ids = {id(row) for row in split}
        for row in active:
            if id(row) not in split_ids:
                if row['unretained_candidate_count'] and (
                        'minimum_region_size_reached_with_unretained_competitors'
                        not in row['unresolved_reasons']):
                    row['unresolved_reasons'].append(
                        'minimum_region_size_reached_with_unretained_competitors')
                finished.append(row)
        if not split:
            break
        children, by_cell = [], defaultdict(list)
        for parent in split:
            x0, y0, x1, y1 = parent['bbox_display']
            xs = [x0, (x0 + x1) / 2, x1] if x1 - x0 >= 2 * minimum else [x0, x1]
            ys = [y0, (y0 + y1) / 2, y1] if y1 - y0 >= 2 * minimum else [y0, y1]
            ancestry = [*parent.get('refinement_ancestry', []), {
                key: deepcopy(parent[key]) for key in
                ('id', 'bbox_display', 'candidate_count', 'unretained_candidate_count')}]
            child_ordinal = 0
            for left, right in zip(xs, xs[1:]):
                for top, bottom in zip(ys, ys[1:]):
                    child = _region_search(page_ref, [left, top, right, bottom])
                    child.update(
                        refinement_ancestry=deepcopy(ancestry), native_scan_complete=True,
                        unsupported_native_item_kinds=deepcopy(
                            parent['unsupported_native_item_kinds']),
                        unsupported_native_item_refs=list(
                            parent['unsupported_native_item_refs']),
                        _order_key=(*parent['_order_key'], child_ordinal),
                        _excluded_hasher=hashlib.sha256(), _excluded_bounds=None)
                    child_ordinal += 1
                    children.append(child)
                    clipped = [max(0, left), max(0, top), min(width, right), min(height, bottom)]
                    for cell in _cells(clipped, minimum):
                        by_cell[cell].append(child)
        def add_candidate(child, candidate_id, box, descriptor_ordinal):
            nonlocal decoded_for_refinement
            window = child['bbox_display']
            if not _intersects(box, window):
                return
            decoded_for_refinement += 1
            child['candidate_count'] += 1
            if len(child['primitive_candidate_refs']) >= max_candidates_per_region:
                child['unretained_candidate_count'] += 1
                child['_excluded_hasher'].update(candidate_id.encode('ascii') + b'\n')
                excluded = child['_excluded_bounds']
                child['_excluded_bounds'] = (list(box) if excluded is None else [
                    min(excluded[0], box[0]), min(excluded[1], box[1]),
                    max(excluded[2], box[2]), max(excluded[3], box[3])])
                return
            child['primitive_candidate_refs'].append(candidate_id)
            child['_retention_max_ordinal'] = descriptor_ordinal
            if (box[0] < window[0] or box[1] < window[1]
                    or box[2] > window[2] or box[3] > window[3]):
                child['cross_boundary_primitive_refs'].append(candidate_id)

        if candidate_cells is not None and candidate_reader_at is not None:
            for child in children:
                ordinals = set()
                for cell in _cells(child['bbox_display'], minimum):
                    ordinals.update(candidate_cells.get(cell, ()))
                for ordinal in sorted(ordinals):
                    candidate_id, box = candidate_reader_at(ordinal)
                    add_candidate(child, candidate_id, box, ordinal)
        else:
            for descriptor in descriptor_reader():
                box = descriptor['search_bbox_display']
                bounded = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
                if bounded[0] > bounded[2] or bounded[1] > bounded[3]:
                    continue
                bounded[0] = math.nextafter(bounded[0], -math.inf)
                bounded[1] = math.nextafter(bounded[1], -math.inf)
                possible = {}
                for cell in _cells(bounded, minimum):
                    for child in by_cell.get(cell, ()):
                        possible[id(child)] = child
                candidate_id = descriptor['candidate_id']
                for child in possible.values():
                    add_candidate(child, candidate_id, box,
                                  descriptor['descriptor_ordinal'])
        for child in children:
            child['relevant_competitor_search_complete'] = (
                not child['unretained_candidate_count']
                and not child['unsupported_native_item_kinds'])
            _finish_search(child)
        active = children

    regions, coverage, retention_max_ordinals = [], [], {}
    for region in sorted(finished, key=lambda row: row['_order_key']):
        region.pop('_order_key', None)
        retention_max_ordinal = region.pop('_retention_max_ordinal', None)
        if retention_max_ordinal is not None:
            retention_max_ordinals[region['id']] = retention_max_ordinal
        excluded_hasher = region.pop('_excluded_hasher', None)
        excluded_bounds = region.pop('_excluded_bounds', None)
        coverage.append({
            'region_ref': region['id'], 'bbox_display': region['bbox_display'],
            'candidate_count': region['candidate_count'],
            'retained_candidate_count': len(region['primitive_candidate_refs']),
            'retained_candidate_refs_sha256': _sha256(region['primitive_candidate_refs']),
            'excluded_candidate_count': region['unretained_candidate_count'],
            'excluded_candidate_refs_stream_sha256': (
                excluded_hasher.hexdigest() if excluded_hasher is not None
                else hashlib.sha256().hexdigest()),
            'excluded_bounds_display': excluded_bounds,
        })
        regions.append(region)
    return regions, coverage, decoded_for_refinement, retention_max_ordinals


def iter_bounded_native_page_regions(page, page_ref, *, region_size_display_points=128,
                                     max_candidates_per_region=4000,
                                     minimum_region_size_display_points=None,
                                     performance_profile=None):
    """Yield exact region packets using a bounded temporary JSON spool.

    Production uses ``RegionIndex.from_native_page`` to write the same scan
    directly to its compact SQLite index.  This compatibility iterator remains
    for APIs and tests that explicitly request materialized region packets.
    """
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as spool:
        positions = {}
        def retain(row):
            positions[row["id"]] = spool.tell()
            spool.write(json.dumps(row, separators=(",", ":")) + "\n")
        def read(ref):
            spool.seek(positions[ref])
            return json.loads(spool.readline())
        regions = _refined_bounded_native_page_regions(
            page, page_ref, region_size_display_points=region_size_display_points,
            max_candidates_per_region=max_candidates_per_region,
            minimum_region_size_display_points=minimum_region_size_display_points,
            candidate_sink=retain, candidate_reader=read,
            performance_profile=performance_profile)
        memberships = defaultdict(list)
        if minimum_region_size_display_points is not None:
            for region in regions:
                for ref in region["primitive_candidate_refs"]:
                    memberships[ref].append(region["id"])
        for region in regions:
            candidates = []
            for ref in region["primitive_candidate_refs"]:
                candidate = read(ref)
                if minimum_region_size_display_points is not None:
                    candidate["search_refs"] = sorted(memberships[ref])
                candidates.append(candidate)
            yield {"region": region, "primitive_candidates": candidates}


def _discover_bounded(source, registry, native_text, seeds_by_page, size, budget, radius, ocr,
                      page_numbers, progress):
    selected = set(range(1, len(registry["pages"]) + 1)) if page_numbers is None else set(page_numbers)
    if not selected or any(type(number) is not int or not 1 <= number <= len(registry["pages"]) for number in selected):
        raise ValueError("invalid page_numbers execution scope")
    pages, searches, primitives, regions = [], [], [], []
    with fitz.open(source) as pdf:
        if len(pdf) != len(registry["pages"]):
            raise ValueError("source page coverage mismatch")
        for scope, page in zip(registry["pages"], pdf):
            started = perf_counter()
            processed = scope["page_number"] in selected
            scan = (_bounded_page_records(page, scope["page_ref"], size, budget,
                                          seeds_by_page[scope["page_ref"]], radius) if processed else
                    {"searches": [], "regions": [], "primitive_candidates": [],
                     "native_drawing_record_count": None, "unsupported_native_item_kinds": {}})
            limited = any(row["unretained_candidate_count"] for row in scan["searches"])
            annotations = [row for row in scan["searches"] if row.get("seed_kind") != "geometry_first_region"]
            pages.append({"page_ref": scope["page_ref"], "page_number": scope["page_number"],
                "page_size_display": [page.rect.width, page.rect.height],
                "native_drawing_record_count": scan["native_drawing_record_count"],
                "drawing_records_intersecting_search_windows": scan["native_drawing_record_count"],
                "unsupported_native_item_kinds": scan["unsupported_native_item_kinds"],
                "search_refs": [row["id"] for row in scan["searches"]],
                "region_refs": [row["id"] for row in scan["regions"]],
                "primitive_candidate_refs": [row["id"] for row in scan["primitive_candidates"]],
                "stage_states": {"native_drawing_bounds_scan": "processed" if processed else "not_processed",
                    "annotation_seeded_target_search": ("budget_limited" if any(row["unretained_candidate_count"] for row in annotations)
                        else "processed" if annotations else "no_eligible_text_seeds") if processed else "not_processed",
                    "unlabelled_symbol_discovery": ("geometry_proposals_budget_limited" if limited else "geometry_proposals_only") if processed else "not_processed",
                    "route_topology_and_continuation": "not_processed", "m4_target_certification": "not_processed"},
                "candidate_search_complete": processed and all(row["relevant_competitor_search_complete"] for row in scan["regions"]),
                "search_completeness_scope": "bounded_page_regions_supported_native_geometry_only",
                "item_inventory_complete": False, "quantity_eligible": False})
            searches.extend(scan["searches"])
            primitives.extend(scan["primitive_candidates"])
            regions.extend(scan["regions"])
            if progress is not None:
                progress({"page_number": scope["page_number"], "elapsed_seconds": round(perf_counter() - started, 3),
                          "search_count": len(scan["searches"]), "primitive_candidate_count": len(scan["primitive_candidates"]),
                          "budget_limited": limited, "processed": processed})
    payload = {"schema_version": VERSION, "layer": LAYER, "document": deepcopy(registry["document"]),
        "input_hashes": {"m1": _sha256(registry), "native_text": _sha256(native_text)},
        "method": {"name": "bounded_native_region_candidate_search", "version": "2.0.0",
                   "search_bounds": SEARCH_BOUNDS_METHOD, "engine_version": fitz.VersionBind},
        "parameters": {"region_size_display_points": size, "max_candidates_per_region": budget,
                       "search_radius_in_text_heights": radius, "page_numbers": sorted(selected)},
        "pages": pages, "searches": searches, "primitive_candidates": primitives, "regions": regions,
        "summary": {"page_count": len(pages), "search_count": len(searches), "region_count": len(regions),
                    "primitive_candidate_count": len(primitives),
                    "searches_with_geometry": sum(row["candidate_count"] > 0 for row in searches),
                    "budget_limited_search_count": sum(row["state"] == "budget_limited" for row in searches)},
        "authority": {"reviewed_selectors_used": False, "accepted_target_emitted": False,
                      "physical_identity_established": False, "quantity_eligible": False,
                      "document_completeness_established": False}}
    if ocr is not None:
        payload["input_hashes"]["item_ocr"] = _sha256(ocr)
    errors = validate_native_target_discovery(payload)
    if errors:
        raise ValueError("invalid native discovery: " + "; ".join(errors))
    return payload


def discover_native_targets(*, pdf_path, sheet_registry, text_observations,
                            search_radius_in_text_heights=8.0,
                            max_candidates_per_page=20000, progress_callback=None,
                            region_size_display_points=None, max_candidates_per_region=4000,
                            item_ocr_observations=None, page_numbers=None):
    """Scan every registered page, retaining all alternatives within each window.

    Native indexing is shared with vector_topology. No reviewed selectors,
    fixed sheet coordinates, or discipline-specific colors are read here.
    Unlabelled symbols and route continuation outside the windows remain
    explicitly unprocessed; completed local search is not package completeness.
    """
    if not math.isfinite(search_radius_in_text_heights) or search_radius_in_text_heights <= 0:
        raise ValueError("search radius must be finite and positive")
    if not isinstance(max_candidates_per_page, int) or max_candidates_per_page < 1:
        raise ValueError("candidate budget must be positive")
    errors = validate_mep_sheet_registry(sheet_registry)
    if errors:
        raise ValueError("invalid M1 registry: " + "; ".join(errors))
    source = Path(pdf_path)
    if _file_sha256(source) != sheet_registry["document"]["source_pdf_sha256"]:
        raise ValueError("source PDF hash does not match registry")
    if text_observations["document"] != sheet_registry["document"] or text_observations["m1_payload_sha256"] != _sha256(sheet_registry):
        raise ValueError("native text observations do not match registry")
    validated = build_mep_document_text_proposals(text_observations)
    invalid = {row["observation_ref"] for row in validated["observation_diagnostics"]}
    seeds_by_page = defaultdict(list)
    for seed in _seeds(text_observations["observations"], invalid):
        seeds_by_page[seed["page_ref"]].append(seed)
    if item_ocr_observations is not None:
        ocr = item_ocr_observations
        if (ocr.get("document") != sheet_registry["document"]
                or ocr.get("m1_payload_sha256") != _sha256(sheet_registry)
                or ocr.get("native_text_payload_sha256") != _sha256(text_observations)):
            raise ValueError("OCR observations do not match registry/native text")
        validated_ocr = build_mep_item_ocr_proposals(ocr)
        invalid_ocr = {row["observation_ref"] for row in validated_ocr["observation_diagnostics"]}
        source_by_ref = {row["id"]: row for row in ocr["observations"]}
        for seed in _seeds(ocr["observations"], invalid_ocr):
            row = source_by_ref[seed["observation_ref"]]
            seed.update(seed_kind="ocr_annotation", source_observation=deepcopy(row),
                        native_overlap_alternatives=deepcopy(row["native_overlap_alternatives"]),
                        ocr_overlap_alternatives=deepcopy(row["ocr_overlap_alternatives"]),
                        overlapping_observations_merged=False)
            seeds_by_page[seed["page_ref"]].append(seed)
    if region_size_display_points is not None:
        return _discover_bounded(source, sheet_registry, text_observations, seeds_by_page,
                                 region_size_display_points, max_candidates_per_region,
                                 search_radius_in_text_heights, item_ocr_observations,
                                 page_numbers, progress_callback)
    if page_numbers is not None:
        raise ValueError("page_numbers requires bounded region discovery")
    pages, all_candidates, all_searches = [], [], []
    with fitz.open(source) as pdf:
        if len(pdf) != len(sheet_registry["pages"]):
            raise ValueError("source page coverage mismatch")
        for scope, page in zip(sheet_registry["pages"], pdf):
            page_started = perf_counter()
            rotation = page.rotation_matrix
            page_rect = page.rect
            seeds = seeds_by_page[scope["page_ref"]]
            searches = {row["id"]: row for row in seeds}
            cell_size = 128.0
            spatial = defaultdict(set)
            for row in seeds:
                window = _search_rect(row["bbox_display"], [page_rect.width, page_rect.height], search_radius_in_text_heights)
                row["search_rect_display"] = window
                for cell in _cells(window, cell_size):
                    spatial[cell].add(row["id"])

            def overlapping(box):
                possible = set()
                # Huge off-page paths must not enumerate unbounded empty cells.
                bounded = [max(box[0], page_rect.x0), max(box[1], page_rect.y0),
                           min(box[2], page_rect.x1), min(box[3], page_rect.y1)]
                if bounded[0] > bounded[2] or bounded[1] > bounded[3]:
                    return []
                for cell in _cells(bounded, cell_size):
                    possible.update(spatial.get(cell, ()))
                return [key for key in sorted(possible) if _intersects(box, searches[key]["search_rect_display"])]

            # Scan drawing bounds across the entire page before local selection.
            drawings = page.get_drawings()
            unsupported = Counter()
            selected_drawing_count = 0
            def drawing_filter(drawing):
                nonlocal selected_drawing_count
                if drawing.get("color") is None:
                    return False
                box = _drawing_search_box(drawing, rotation)
                keep = bool(overlapping(box))
                if keep:
                    selected_drawing_count += 1
                    unsupported.update(str(item[0]) for item in drawing.get("items", []) if item[0] not in {"l", "re", "qu", "c"})
                return keep

            candidates = []
            for native in iter_native_segments(drawings, drawing_filter):
                points, geometry_box, box = _native_display_geometry(native, rotation)
                matches = overlapping(box)
                if not matches:
                    continue
                retain = len(candidates) < max_candidates_per_page
                identifier = _stable_id("mep_native_target_primitive", scope["page_ref"], native["id"]) if retain else None
                for key in matches:
                    row = searches[key]
                    row["candidate_count"] += 1
                    if not retain:
                        row["unretained_candidate_count"] += 1
                        continue
                    row["primitive_candidate_refs"].append(identifier)
                    window = row["search_rect_display"]
                    if box[0] < window[0] or box[1] < window[1] or box[2] > window[2] or box[3] > window[3]:
                        row["cross_boundary_primitive_refs"].append(identifier)
                if retain:
                    candidates.append({
                        "id": identifier, "page_ref": scope["page_ref"],
                        "source_primitive_ref": native["id"],
                        "source_native_segment": native,
                        "points_display": points, "bbox_display": geometry_box,
                        "search_bbox_display": box, "search_bbox_is_geometry": False,
                        "pdf_to_display_matrix": list(rotation),
                        "search_refs": matches, "state": "observed",
                        "role": "unclassified_native_geometry_candidate",
                        "geometry_is_not_a_route_or_equipment": True,
                        "quantity_eligible": False,
                    })
            for row in seeds:
                row["state"] = "budget_limited" if row["unretained_candidate_count"] else "searched_unbound"
                row["unresolved_reasons"] = ["geometric_target_applicability_not_certified"]
                if row["candidate_count"] == 0:
                    row["unresolved_reasons"].append("no_native_candidate_in_configured_search_window")
                if row["unretained_candidate_count"]:
                    row["unresolved_reasons"].append("candidate_budget_exhausted")
                if row["cross_boundary_primitive_refs"]:
                    row["unresolved_reasons"].append("candidate_geometry_extends_beyond_search_window")
                row["primitive_candidate_refs"].sort()
                row["cross_boundary_primitive_refs"].sort()
                row["unique_target_established"] = False
            limited = any(row["unretained_candidate_count"] for row in seeds)
            pages.append({
                "page_ref": scope["page_ref"], "page_number": scope["page_number"],
                "page_size_display": [page_rect.width, page_rect.height],
                "native_drawing_record_count": len(drawings),
                "drawing_records_intersecting_search_windows": selected_drawing_count,
                "unsupported_native_item_kinds": dict(unsupported),
                "search_refs": [row["id"] for row in seeds],
                "primitive_candidate_refs": [row["id"] for row in candidates],
                "stage_states": {
                    "native_drawing_bounds_scan": "processed",
                    "annotation_seeded_target_search": "budget_limited" if limited else "processed" if seeds else "no_eligible_native_text_seeds",
                    "unlabelled_symbol_discovery": "not_processed",
                    "route_topology_and_continuation": "not_processed",
                    "m4_target_certification": "not_processed",
                },
                "candidate_search_complete": bool(seeds) and not limited and not unsupported,
                "search_completeness_scope": "configured_annotation_windows_and_supported_strokes_only",
                "item_inventory_complete": False, "quantity_eligible": False,
            })
            all_candidates.extend(candidates)
            all_searches.extend(seeds)
            if progress_callback is not None:
                progress_callback({"page_number": scope["page_number"],
                    "elapsed_seconds": round(perf_counter() - page_started, 3),
                    "search_count": len(seeds), "primitive_candidate_count": len(candidates),
                    "budget_limited": limited})
    payload = {
        "schema_version": VERSION, "layer": LAYER,
        "document": deepcopy(sheet_registry["document"]),
        "input_hashes": {"m1": _sha256(sheet_registry), "native_text": _sha256(text_observations)},
        "method": {"name": "native_annotation_window_candidate_search", "version": "2.0.0",
                   "search_bounds": SEARCH_BOUNDS_METHOD, "engine_version": fitz.VersionBind},
        "parameters": {"search_radius_in_text_heights": search_radius_in_text_heights, "max_candidates_per_page": max_candidates_per_page},
        "pages": pages, "searches": all_searches, "primitive_candidates": all_candidates,
        "summary": {"page_count": len(pages), "search_count": len(all_searches),
                    "primitive_candidate_count": len(all_candidates),
                    "searches_with_geometry": sum(row["candidate_count"] > 0 for row in all_searches),
                    "budget_limited_search_count": sum(row["state"] == "budget_limited" for row in all_searches)},
        "authority": {"reviewed_selectors_used": False, "accepted_target_emitted": False,
                      "physical_identity_established": False, "quantity_eligible": False,
                      "document_completeness_established": False},
    }
    if item_ocr_observations is not None:
        payload["input_hashes"]["item_ocr"] = _sha256(item_ocr_observations)
    errors = validate_native_target_discovery(payload)
    if errors:
        raise ValueError("invalid native discovery: " + "; ".join(errors))
    return payload


def validate_native_target_discovery(payload):
    """Check search evidence without granting target closure.

    Saved v1 methods replay their legacy sampled-point search bounds only; they
    are not v2 control-hull coverage certificates. Fresh v2 methods and regions
    declare the stronger search-bound semantics and must preserve every field.
    """
    errors = []
    if payload.get("layer") != LAYER or payload.get("schema_version") != VERSION:
        errors.append("discovery contract mismatch")
    pages = payload.get("pages", [])
    page_by_ref = {row["page_ref"]: row for row in pages}
    if len(page_by_ref) != len(pages):
        errors.append("duplicate registered page IDs")
    if [row.get("page_number") for row in pages] != list(range(1, payload.get("document", {}).get("page_count", 0) + 1)):
        errors.append("registered page coverage mismatch")
    searches = {row["id"]: row for row in payload.get("searches", [])}
    primitives = {row["id"]: row for row in payload.get("primitive_candidates", [])}
    bounded = payload.get("method", {}).get("name") == "bounded_native_region_candidate_search"
    conservative = payload.get("method", {}).get("search_bounds") == SEARCH_BOUNDS_METHOD
    if payload.get("method", {}).get("version") == "2.0.0" and not conservative:
        errors.append("conservative search bounds method missing")
    regions = {row["id"]: row for row in payload.get("regions", [])}
    if bounded:
        expected_regions = {ref: row for ref, row in searches.items() if row.get("seed_kind") == "geometry_first_region"}
        if regions != expected_regions or len(regions) != len(payload.get("regions", [])):
            errors.append("region evidence membership mismatch")
    if any(row["page_ref"] not in page_by_ref for row in [*searches.values(), *primitives.values()]):
        errors.append("evidence references unregistered page")
    if len(searches) != len(payload.get("searches", [])) or len(primitives) != len(payload.get("primitive_candidates", [])):
        errors.append("duplicate evidence IDs")
    search_memberships = {key: set(row["primitive_candidate_refs"]) for key, row in searches.items()}
    boundary_memberships = {key: set(row["cross_boundary_primitive_refs"]) for key, row in searches.items()}
    primitive_memberships = {key: set(row["search_refs"]) for key, row in primitives.items()}
    source_refs = {(row["page_ref"], row["source_primitive_ref"]) for row in primitives.values()}
    if len(source_refs) != len(primitives):
        errors.append("duplicate native primitive provenance")
    for page in pages:
        if page.get("item_inventory_complete") is not False or page.get("quantity_eligible") is not False:
            errors.append("page acquired unsupported authority")
        for key, records in (("search_refs", searches), ("primitive_candidate_refs", primitives)):
            if sorted(page[key]) != sorted(ref for ref, row in records.items() if row["page_ref"] == page["page_ref"]):
                errors.append("page evidence membership mismatch")
        owned = [searches[ref] for ref in page["search_refs"] if ref in searches]
        limited = any(row["unretained_candidate_count"] for row in owned)
        complete = bool(page["search_refs"]) and not limited and not page["unsupported_native_item_kinds"]
        if bounded:
            selected = page["page_number"] in payload["parameters"]["page_numbers"]
            owned_regions = [row for row in owned if row.get("seed_kind") == "geometry_first_region"]
            annotations = [row for row in owned if row.get("seed_kind") != "geometry_first_region"]
            size = payload["parameters"]["region_size_display_points"]
            width, height = page["page_size_display"]
            expected_boxes = [[x * size, y * size, min(width, (x + 1) * size), min(height, (y + 1) * size)]
                              for x in range(math.ceil(width / size)) for y in range(math.ceil(height / size))] if selected else []
            if ([row["bbox_display"] for row in owned_regions] != expected_boxes
                    or page.get("region_refs") != [row["id"] for row in owned_regions]):
                errors.append("bounded region coverage replay mismatch")
            complete = selected and all(row["relevant_competitor_search_complete"] for row in owned_regions)
        if page.get("candidate_search_complete") is not complete:
            errors.append("page candidate search completeness mismatch")
        expected = {"native_drawing_bounds_scan": "processed",
                    "annotation_seeded_target_search": "budget_limited" if limited else "processed" if page["search_refs"] else "no_eligible_native_text_seeds",
                    "unlabelled_symbol_discovery": "not_processed",
                    "route_topology_and_continuation": "not_processed",
                    "m4_target_certification": "not_processed"}
        if bounded:
            expected.update(native_drawing_bounds_scan="processed" if selected else "not_processed",
                annotation_seeded_target_search=("budget_limited" if any(row["unretained_candidate_count"] for row in annotations)
                    else "processed" if annotations else "no_eligible_text_seeds") if selected else "not_processed",
                unlabelled_symbol_discovery=("geometry_proposals_budget_limited" if limited else "geometry_proposals_only") if selected else "not_processed")
        if page.get("stage_states") != expected:
            errors.append("unsupported page stage closure")
    for row in searches.values():
        refs = row["primitive_candidate_refs"]
        page = page_by_ref.get(row["page_ref"])
        is_region = bounded and row.get("seed_kind") == "geometry_first_region"
        if page is not None:
            expected_window = row["bbox_display"] if is_region else _search_rect(row["bbox_display"], page["page_size_display"], payload["parameters"]["search_radius_in_text_heights"])
            if row["search_rect_display"] != expected_window:
                errors.append("source search window replay mismatch")
        expected_id = (_stable_id("mep_native_target_region", row["page_ref"], row["bbox_display"]) if is_region else
                       _stable_id("mep_native_target_search", row["page_ref"], row["observation_ref"]))
        if row["id"] != expected_id:
            errors.append("unstable search provenance ID")
        if bounded:
            if is_region:
                if conservative and row.get("search_bounds_method") != SEARCH_BOUNDS_METHOD:
                    errors.append("region search bounds semantics mismatch")
                expected_complete = not row["unretained_candidate_count"] and not row["unsupported_native_item_kinds"]
                if len(refs) > payload["parameters"]["max_candidates_per_region"]:
                    errors.append("region retention budget exceeded")
                if row.get("native_scan_complete") is not True:
                    errors.append("region native scan incomplete")
                if (sum(row["unsupported_native_item_kinds"].values()) != len(row["unsupported_native_item_refs"])
                        or len(set(row["unsupported_native_item_refs"])) != len(row["unsupported_native_item_refs"])):
                    errors.append("unsupported native item provenance mismatch")
            else:
                relevant = [region for region in regions.values() if region["page_ref"] == row["page_ref"]
                            and _intersects(region["bbox_display"], row["search_rect_display"])]
                if row.get("region_refs") != [region["id"] for region in relevant]:
                    errors.append("annotation competitor region coverage mismatch")
                expected_complete = all(region["relevant_competitor_search_complete"] for region in relevant)
            if row.get("relevant_competitor_search_complete") is not expected_complete:
                errors.append("relevant competitor completeness mismatch")
        if row.get("seed_kind") == "ocr_annotation":
            source = row.get("source_observation", {})
            if (source.get("handoff_state") != "proposal_only" or source.get("id") != row["observation_ref"]
                    or source.get("bbox_display") != row["bbox_display"] or source.get("text") != row["text"]
                    or source.get("page_ref") != row["page_ref"] or row.get("overlapping_observations_merged") is not False
                    or not payload["input_hashes"].get("item_ocr")
                    or source.get("native_overlap_alternatives") != row.get("native_overlap_alternatives")
                    or source.get("ocr_overlap_alternatives") != row.get("ocr_overlap_alternatives")):
                errors.append("OCR seed provenance/ambiguity mismatch")
        if row["candidate_count"] != len(refs) + row["unretained_candidate_count"]:
            errors.append("candidate budget accounting mismatch")
        if len(refs) != len(search_memberships[row["id"]]) or row["unretained_candidate_count"] < 0:
            errors.append("invalid candidate budget membership")
        if row["unretained_candidate_count"] and row["state"] != "budget_limited":
            errors.append("truncated search must remain budget limited")
        if row.get("unique_target_established") is not False or row.get("quantity_eligible") is not False:
            errors.append("search acquired unsupported target authority")
        if not set(row["cross_boundary_primitive_refs"]) <= set(refs):
            errors.append("boundary primitive reference mismatch")
        for ref in refs:
            target = primitives.get(ref)
            if not target or target["page_ref"] != row["page_ref"] or row["id"] not in primitive_memberships[ref]:
                errors.append("search-to-primitive evidence mismatch")
            if target:
                box, window = target.get("search_bbox_display", target["bbox_display"]), row["search_rect_display"]
                if not _intersects(box, window):
                    errors.append("candidate does not intersect search window")
                crosses = box[0] < window[0] or box[1] < window[1] or box[2] > window[2] or box[3] > window[3]
                if crosses != (ref in boundary_memberships[row["id"]]):
                    errors.append("search boundary membership mismatch")
    for row in primitives.values():
        if row.get("quantity_eligible") is not False or row.get("geometry_is_not_a_route_or_equipment") is not True:
            errors.append("primitive acquired unsupported item authority")
        if row["source_primitive_ref"] != row["source_native_segment"]["id"]:
            errors.append("native primitive provenance mismatch")
        if row["id"] != _stable_id("mep_native_target_primitive", row["page_ref"], row["source_primitive_ref"]):
            errors.append("unstable native primitive ID")
        for ref in row["search_refs"]:
            search = searches.get(ref)
            if not search or row["id"] not in search_memberships[ref]:
                errors.append("primitive-to-search evidence mismatch")
        if any(not math.isfinite(value) for point in row["points_display"] for value in point):
            errors.append("invalid native geometry")
        native = row["source_native_segment"]
        try:
            matrix = fitz.Matrix(row["pdf_to_display_matrix"])
            expected_points, expected_box, expected_search_box = _native_display_geometry(native, matrix)
            if expected_points != row["points_display"] or expected_box != row["bbox_display"]:
                errors.append("native coordinate replay mismatch")
            if conservative and "search_bbox_display" not in row:
                errors.append("conservative search bounds missing")
            if "search_bbox_display" in row and (row["search_bbox_display"] != expected_search_box
                    or row.get("search_bbox_is_geometry") is not False):
                errors.append("native conservative search bounds replay mismatch")
        except (ValueError, TypeError):
            errors.append("invalid native coordinate transform")
    expected_authority = {"reviewed_selectors_used": False, "accepted_target_emitted": False,
                          "physical_identity_established": False, "quantity_eligible": False,
                          "document_completeness_established": False}
    if payload.get("authority") != expected_authority:
        errors.append("discovery authority boundary changed")
    return errors
