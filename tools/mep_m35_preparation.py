"""Unprepared M3.5 metric reference for isolated differential tests only.

Copied from the pre-preparation helper. Production never imports this module.
Only source preparation differs; candidate enumeration and certificates still
run through the production builder. The reference context is single-threaded.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import src.drawing_engine.disciplines.mep.mep_outlined_route_composites as m35


def parallel_metrics_reference(left, right, **_prepared):
    left_points = left["geometry"]["points_display"]
    right_points = right["geometry"]["points_display"]
    left_samples = m35._samples(left_points)
    forward = m35._samples(right_points)
    reverse = list(reversed(forward))
    forward_cost = sum(math.dist(a, b) for a, b in zip(left_samples, forward))
    reverse_cost = sum(math.dist(a, b) for a, b in zip(left_samples, reverse))
    right_samples = reverse if reverse_cost < forward_cost else forward
    reversed_right = reverse_cost < forward_cost
    separations = [math.dist(a, b) for a, b in zip(left_samples, right_samples)]
    mean = sum(separations) / len(separations)
    maximum_deviation = max(abs(value - mean) for value in separations)
    tangent_differences = [
        math.degrees(m35._angle_difference(m35._angle(a, b), m35._angle(c, d)))
        for a, b, c, d in zip(left_samples, left_samples[1:], right_samples, right_samples[1:])
    ]
    left_length = m35._length(left_points)
    right_length = m35._length(right_points)
    width = max(
        float(left.get("style", {}).get("width_display_points") or 0.0),
        float(right.get("style", {}).get("width_display_points") or 0.0),
    )
    return {
        "right_reversed": reversed_right,
        "left_samples": left_samples,
        "right_samples": right_samples,
        "left_length_display_points": left_length,
        "right_length_display_points": right_length,
        "length_ratio": min(left_length, right_length) / max(left_length, right_length, 1e-9),
        "mean_separation_display_points": mean,
        "maximum_separation_deviation_display_points": maximum_deviation,
        "maximum_tangent_difference_degrees": max(tangent_differences, default=180.0),
        "endpoint_separations_display_points": [separations[0], separations[-1]],
        "member_width_display_points": width,
    }


@contextmanager
def reference_preparation():
    # Omit the new per-row preparation too: measure the original algorithm,
    # not an exhaustive algorithm with unused preparation added on top.
    with patch.object(m35, '_parallel_metrics', parallel_metrics_reference), \
            patch.object(m35, '_prepare_polyline', return_value=None, create=True):
        yield
