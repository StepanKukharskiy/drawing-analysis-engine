import copy
from contextlib import nullcontext
import inspect
import json
import math
import random
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.mep_m35_preparation import parallel_metrics_reference, reference_preparation

import src.drawing_engine.disciplines.mep.mep_outlined_route_composites as composites_module
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import (
    build_mep_outlined_route_composites,
    validate_mep_outlined_route_composites,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph


PAGE_REF = "page.outlined"


def _segment(index, start, end, *, color=(0.2, 0.2, 0.2), width=1.0):
    source_ref = f"drawing[{index}].item[0].segment[0]"
    return {
        "id": source_ref,
        "drawing_ref": f"drawing[{index}]",
        "primitive_ref": f"drawing[{index}].item[0]",
        "item_index": 0,
        "part_index": 0,
        "kind": "line",
        "start_display": list(start),
        "end_display": list(end),
        "control_points_display": [],
        "sample_points_display": [],
        "style": {"stroke": color, "fill": None, "width": width, "dash": None},
    }


def _graph(*segments):
    registry = {
        "schema_version": "0.1.0",
        "layer": "mep_sheet_coordinate_registry",
        "document": {"document_key": "pdf-sha256:outlined", "page_count": 1},
        "pages": [{
            "record_type": "mep_sheet_page_record",
            "record_version": "0.1.0",
            "id": "m1.scope.outlined",
            "page_ref": PAGE_REF,
            "page_number": 1,
            "role": "mechanical_piping_plan",
            "fields": {
                "sheet_number": {"value": "M-1", "evidence_refs": ["title.sheet"]},
                "scale": {"drawing_inches_per_paper_inch": 96.0},
            },
        }],
        "packages": [],
    }
    return build_mep_route_graph(
        sheet_registry=registry,
        page_inputs={PAGE_REF: {"native_topology": {
            "schema_version": "0.1.0", "segments": list(segments), "vertices": []
        }}},
    )


class MepOutlinedRouteCompositesTest(unittest.TestCase):
    def test_wide_duct_and_exact_overpaint_aliases_preserve_every_native_id(self):
        segments = [
            _segment(0, (0, 0), (100, 0), width=1.5),
            _segment(1, (0, 15), (100, 15), width=1.5),
            _segment(2, (100, 0), (0, 0), width=1.5),
            _segment(3, (100, 15), (0, 15), width=1.5),
            _segment(4, (0, 0), (0, 15), width=1.5),
        ]
        graph = _graph(*segments)
        payload = build_mep_outlined_route_composites(
            route_graph=graph, maximum_separation_to_length_ratio=.2,
            collapse_exact_overpaint_representations=True)
        self.assertEqual(validate_mep_outlined_route_composites(payload), [])
        self.assertEqual(payload['summary']['accepted_composite_count'], 1)
        composite = payload['accepted_composites'][0]
        self.assertEqual(composite['exact_duplicate_representation_count'], 2)
        self.assertEqual([len(group) for group in
                          composite['equivalent_member_fragment_groups']], [2, 2])
        self.assertEqual(len(composite['member_source_primitive_refs']), 4)
        self.assertAlmostEqual(composite['derived_geometry']
                               ['corridor_width_display_points'], 15)
        self.assertEqual(composite['method']['parameters'], {
            'maximum_separation_to_length_ratio': .2,
            'collapse_exact_overpaint_representations': True})

        competing = _graph(
            *segments,
            _segment(5, (0, 30), (100, 30), width=1.5),
            _segment(6, (0, 15), (0, 30), width=1.5),
        )
        self.assertEqual(build_mep_outlined_route_composites(
            route_graph=competing, maximum_separation_to_length_ratio=.2,
            collapse_exact_overpaint_representations=True
        )['accepted_composites'], [])

    def test_closed_parallel_envelope_publishes_one_derived_centreline(self):
        graph = _graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 4)),
            _segment(2, (0, 0), (0, 4)),
        )
        payload = build_mep_outlined_route_composites(route_graph=graph)

        self.assertEqual(validate_mep_outlined_route_composites(payload), [])
        self.assertEqual(payload["summary"]["accepted_composite_count"], 1)
        composite = payload["accepted_composites"][0]
        self.assertEqual(composite["closure_kind"], "start_cap_path")
        self.assertEqual(
            composite["derived_geometry"]["centreline_points_display"],
            [[0.0, 2.0], [100.0, 2.0]],
        )
        self.assertTrue(composite["native_strokes_preserved"])
        self.assertFalse(composite["physical_route_identity_established"])
        self.assertFalse(composite["quantity_eligible"])

    def test_parallel_distance_without_closure_never_merges_adjacent_pipes(self):
        graph = _graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 4)),
        )
        payload = build_mep_outlined_route_composites(route_graph=graph)

        self.assertEqual(payload["accepted_composites"], [])
        candidate = next(
            row for row in payload["candidates"]
            if len(row["member_fragment_refs"]) == 2
        )
        self.assertIn("no_envelope_closing_feature", candidate["reasons"])

    def test_equally_supported_pairing_abstains(self):
        graph = _graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 4)),
            _segment(2, (0, 8), (100, 8)),
            _segment(3, (0, 0), (0, 4)),
            _segment(4, (0, 4), (0, 8)),
        )
        payload = build_mep_outlined_route_composites(route_graph=graph)

        self.assertEqual(payload["accepted_composites"], [])
        self.assertGreaterEqual(
            sum("equally_supported_alternative_pairing" in row["reasons"] for row in payload["candidates"]),
            2,
        )

    def test_style_and_synchronized_geometry_are_hard_gates(self):
        style_graph = _graph(
            _segment(0, (0, 0), (100, 0), width=1.0),
            _segment(1, (0, 4), (100, 4), width=3.0),
            _segment(2, (0, 0), (0, 4), width=1.0),
        )
        style_payload = build_mep_outlined_route_composites(route_graph=style_graph)
        self.assertEqual(style_payload["accepted_composites"], [])
        self.assertTrue(any("incompatible_member_style" in row["reasons"] for row in style_payload["candidates"]))

        turn_graph = _graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 8)),
            _segment(2, (0, 0), (0, 4)),
        )
        turn_payload = build_mep_outlined_route_composites(route_graph=turn_graph)
        self.assertEqual(turn_payload["accepted_composites"], [])
        self.assertTrue(any("turns_or_endpoints_are_not_synchronized" in row["reasons"] for row in turn_payload["candidates"]))

    def test_validator_preserves_quantity_and_authority_boundary(self):
        payload = build_mep_outlined_route_composites(route_graph=_graph(
            _segment(0, (0, 0), (100, 0)),
            _segment(1, (0, 4), (100, 4)),
            _segment(2, (0, 0), (0, 4)),
        ))
        changed = copy.deepcopy(payload)
        changed["accepted_composites"][0]["quantity_eligible"] = True
        self.assertTrue(validate_mep_outlined_route_composites(changed))
        source = inspect.getsource(composites_module)
        self.assertNotIn("M&P mark-up", source)
        self.assertNotIn("page_number ==", source)
        self.assertNotIn("schedule_comparison", source)


class M35PreparationTest(unittest.TestCase):
    def test_each_source_prepared_once_and_every_pair_still_compared(self):
        graph = _graph(*[_segment(i, (0, i*4), (100, i*4)) for i in range(20)])
        for _ in range(2):  # No preparation survives a builder invocation.
            with patch.object(composites_module, '_samples', wraps=composites_module._samples) as sample_calls, \
                    patch.object(composites_module, '_parallel_metrics', wraps=composites_module._parallel_metrics) as pair_calls:
                actual = build_mep_outlined_route_composites(route_graph=graph)
                self.assertEqual(20, sample_calls.call_count)
                self.assertEqual(190, pair_calls.call_count)
        with reference_preparation(), patch.object(composites_module, '_samples', wraps=composites_module._samples) as sample_calls:
            expected = build_mep_outlined_route_composites(route_graph=graph)
            self.assertEqual(380, sample_calls.call_count)
        self.assertEqual(expected, actual)

    def assert_replays(self, graph):
        before = copy.deepcopy(graph)
        with reference_preparation():
            expected = build_mep_outlined_route_composites(route_graph=graph)
        actual = build_mep_outlined_route_composites(route_graph=graph)
        self.assertEqual(expected, actual)
        self.assertEqual(composites_module._canonical_sha256(expected),
                         composites_module._canonical_sha256(actual))
        self.assertEqual(before, graph)
        return actual

    def test_unrounded_metrics_preserve_multisegment_threshold_and_direction_cases(self):
        rng = random.Random(351722722)
        paths = [[[0., 0.], [0., 0.]], [[0., 0.], [1e-12, 0.]],
                 [[0., 0.], [30., 0.], [30., 40.]],
                 [[100., 0.], [0., 0.]], [[0., 0.], [100., 0.]]]
        paths += [[[offset+rng.uniform(-20, 20), offset+rng.uniform(-20, 20)]
                   for _ in range(4)] for offset in (0., -10000., 1e9)]
        for points in paths:
            for separation in (.1, math.nextafter(.1, 0), math.nextafter(.1, math.inf), 4., 12.):
                a = {'geometry': {'points_display': points}, 'style': {'width_display_points': 1.}}
                b = {'geometry': {'points_display': [[x, y+separation] for x, y in reversed(points)]},
                     'style': {'width_display_points': .85}}
                self.assertEqual(parallel_metrics_reference(a, b), composites_module._parallel_metrics(a, b))

    def test_mutated_geometry_style_and_ownership_never_reuse_a_certificate(self):
        graph = _graph(_segment(0, (0, 0), (100, 0)), _segment(1, (0, 4), (100, 4)),
                       _segment(2, (0, 0), (0, 4)))
        self.assertEqual(1, len(self.assert_replays(graph)['accepted_composites']))
        for kind in ('geometry', 'style', 'provenance', 'ownership'):
            changed = copy.deepcopy(graph)
            row = next(r for r in changed['pages'][0]['fragments']
                       if r['source_primitive_ref'] == 'drawing[1].item[0].segment[0]')
            if kind == 'geometry':
                row['geometry']['points_display'][-1][1] += 10
            elif kind == 'style':
                row['style']['width_display_points'] = 3.
            elif kind == 'provenance':
                row['provenance']['method'] = 'different_source_method'
            else:
                row['ownership']['view_scope_ref'] = 'different_view_scope'
            with self.subTest(kind=kind):
                result = self.assert_replays(changed)
                self.assertFalse(result['accepted_composites'])
                self.assertTrue(result['candidates'])  # Failed gates remain evidence.
        self.assertEqual(1, len(self.assert_replays(graph)['accepted_composites']))

    def test_competing_strokes_and_missing_closure_remain_abstentions(self):
        for segments in (
            [_segment(0, (0, 0), (100, 0)), _segment(1, (0, 4), (100, 4))],
            [_segment(0, (0, 0), (100, 0)), _segment(1, (0, 4), (100, 4)),
             _segment(2, (0, 8), (100, 8)), _segment(3, (0, 0), (0, 4)),
             _segment(4, (0, 4), (0, 8))],
        ):
            self.assertFalse(self.assert_replays(_graph(*segments))['accepted_composites'])

    def test_order_input_validation_and_output_aliasing_are_unchanged(self):
        graph = _graph(_segment(0, (0, 0), (100, 0)), _segment(1, (0, 4), (100, 4)),
                       _segment(2, (0, 0), (0, 4)))
        # Keep each input's original orientation; do not impose a new canonical
        # member order on metric arithmetic when the source order changes.
        for reverse in (False, True):
            order = copy.deepcopy(graph)
            if reverse:
                order['pages'][0]['fragments'].reverse()
            result = self.assert_replays(order)
            result['accepted_composites'][0]['derived_geometry']['centreline_points_display'][0][0] = 999.
            self.assertNotEqual(result['accepted_composites'][0], result['candidates'][0])
            self.assert_replays(order)
        graph['pages'][0]['fragments'][0]['quantity_eligible'] = True
        for context in (reference_preparation(), nullcontext()):
            with context, self.assertRaises(ValueError):
                build_mep_outlined_route_composites(route_graph=graph)


class M35PreparationReplayTest(unittest.TestCase):
    assert_replays = M35PreparationTest.assert_replays
    def test_complete_frozen_outputs_match_without_dropping_failed_candidates(self):
        root = Path(__file__).resolve().parents[1]
        run = root/'output/mep-stroke-ownership-2026-08-31'
        graph = json.loads((run/'route-observations.json').read_bytes())
        expected = json.loads((run/'outlined-route-composites.json').read_bytes())
        self.assertEqual(expected, self.assert_replays(graph))
        del graph, expected
        for number in (12, 16):
            frozen = json.loads((root/f'output/mep-package-native-verified-2026-08-31/page-{number:03d}.automatic-targets.json').read_bytes())
            self.assertEqual(frozen['composites'], self.assert_replays(frozen['route_graph']))
            del frozen


if __name__ == "__main__":
    unittest.main()
