"""Opt-in geometry experiment. No production import or persistent style cache.

The process-local patch is for isolated, single-threaded experiments/tests only.
Only point searches and pure style preparation change; traversal and evidence
stay in the reference tracer. A missing requested native library is an error.
"""
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import src.drawing_engine.disciplines.mep.mep_native_bend_connections as bend
from src.drawing_engine.core.vector_topology import point_distance_to_segment

_REFERENCE_BATCH = bend._points_on_segments_exhaustive
_REFERENCE_STYLES = bend._source_style_compatibility_exhaustive
_PRODUCTION_BATCH = bend._points_on_segments
precomputed_styles = bend._source_style_compatibility


def guard_for(points, edges, tolerance):
    guard = bend._point_index_guard(points, edges, tolerance)
    if guard is None:
        raise ValueError('experiment requires finite coordinates/tolerance bounded by 1e12')
    # Outward broad-phase padding also covers interpolation roundoff. Rust
    # rechecks this entire tolerance band with the Python reference predicate.
    return guard


def indexed_point_hits(points, edges, tolerance, stats=None):
    """Two sorted coordinate indexes; exact reference predicate after pruning."""
    guard_for(points, edges, tolerance)  # Keep the Rust comparison domain explicit.
    return bend._indexed_point_hits(points, edges, tolerance, stats)


def load_native(path):
    path = Path(path).resolve(strict=True)
    spec = spec_from_file_location('_mep_boundary_adjacency', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'not a Python extension: {path}')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rust_point_hits(native, points, edges, tolerance, stats=None):
    guard = guard_for(points, edges, tolerance)
    # One PyO3 call for the complete graph, never a call per distance. Native
    # results reference input positions, not new evidence or geometric IDs.
    hits, uncertain, tested = native.point_hits(points, edges, tolerance, guard)
    for edge_index, candidates in enumerate(uncertain):
        p, q = edges[edge_index]
        hits[edge_index].extend(i for i in candidates
            if point_distance_to_segment(points[i], p, q) <= tolerance)
        hits[edge_index].sort()
    if stats is not None:
        stats.update(distance_tests=tested, python_threshold_rechecks=sum(map(len, uncertain)))
    return hits


@contextmanager
def backend(name, native=None, batches=None):
    if name not in ('exhaustive', 'indexed', 'rust'):
        raise ValueError(f'unknown adjacency backend: {name}')
    if name == 'rust' and native is None:
        raise ValueError('Rust requested without an explicit compiled library')
    if name == 'indexed' and batches is None and bend._points_on_segments is _PRODUCTION_BATCH:
        yield  # Direct production path: no monkey-patching or experiment dependency.
        return
    reference = _REFERENCE_BATCH

    def batch(points, edges):
        # Stable primitive positions for a single call; never deduplicate here.
        if batches is not None:
            batches.append((sorted(points), list(edges)))
        if name == 'exhaustive':
            return reference(points, edges)
        if name == 'indexed':
            return _PRODUCTION_BATCH(points, edges)
        points, edges = sorted(points), list(edges)
        hits = rust_point_hits(native, points, edges, bend.TOLERANCE)
        return ([points[i] for i in indices] for indices in hits)

    styles = _REFERENCE_STYLES if name == 'exhaustive' else precomputed_styles
    with patch.object(bend, '_points_on_segments', batch), patch.object(bend, '_source_style_compatibility', styles):
        yield
