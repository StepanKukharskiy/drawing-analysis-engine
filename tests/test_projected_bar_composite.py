from __future__ import annotations

from copy import deepcopy
import unittest

from src.drawing_engine.disciplines.rebar.projected_bar_composite import build_composite_projected_bar_proposals


def _fragment(identifier: str, x: float, mark: str = "9") -> dict:
    return {
        "id": identifier,
        "primitive_ref": f"drawing.{identifier}.item[0]",
        "source_path_ref": f"drawing.{identifier}",
        "geometry": {
            "kind": "line",
            "points_display": [[x, 10.0], [x, 110.0]],
            "length_points": 100.0,
            "angle_deg": 90.0,
        },
        "style": {
            "width_pt": 2.76,
            "stroke": [0.0, 0.0, 0.0],
            "fill": None,
            "dashes": "[] 0",
        },
        "mark_hypotheses": [mark],
    }


def _component(identifier: str, fragment: str, mark: str = "9") -> dict:
    return {
        "id": identifier,
        "fragment_ids": [fragment],
        "view_ids": ["view.1"],
        "mark_hypotheses": [mark],
        "topology": "single_fragment",
    }


class ProjectedBarCompositeTests(unittest.TestCase):
    def test_unique_near_coincident_pair_is_accepted_without_mutation(self) -> None:
        fragments = [_fragment("fragment.a", 10.0), _fragment("fragment.b", 12.28)]
        components = [
            _component("component.a", "fragment.a"),
            _component("component.b", "fragment.b"),
        ]
        original = deepcopy((fragments, components))
        result = build_composite_projected_bar_proposals(fragments, components)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["state"], "accepted")
        self.assertEqual(
            proposal["certificate"],
            "unique_near_coincident_parallel_same_mark_double_stroke",
        )
        self.assertEqual(proposal["fragment_ids"], ["fragment.a", "fragment.b"])
        self.assertEqual((fragments, components), original)
        self.assertFalse(result["contract"]["native_fragments_mutated"])
        self.assertFalse(result["contract"]["quantities_changed"])

    def test_distant_same_mark_pair_stays_review_candidate(self) -> None:
        fragments = [_fragment("fragment.a", 10.0), _fragment("fragment.b", 110.0)]
        components = [
            _component("component.a", "fragment.a"),
            _component("component.b", "fragment.b"),
        ]
        result = build_composite_projected_bar_proposals(fragments, components)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["state"], "review_candidate")
        self.assertFalse(proposal["local_double_stroke_candidate"])
        self.assertIn("same mark is candidate generation only", proposal["reason_not_accepted"])

    def test_non_unique_local_pairing_abstains(self) -> None:
        fragments = [
            _fragment("fragment.a", 10.0),
            _fragment("fragment.b", 12.0),
            _fragment("fragment.c", 14.0),
        ]
        components = [
            _component("component.a", "fragment.a"),
            _component("component.b", "fragment.b"),
            _component("component.c", "fragment.c"),
        ]
        result = build_composite_projected_bar_proposals(fragments, components)
        local = [item for item in result["proposals"] if item["local_double_stroke_candidate"]]
        self.assertEqual(len(local), 2)
        self.assertTrue(all(item["state"] == "review_candidate" for item in local))
        self.assertEqual(result["summary"]["accepted_count"], 0)

    def test_different_marks_do_not_generate_a_pair(self) -> None:
        fragments = [_fragment("fragment.a", 10.0, "9"), _fragment("fragment.b", 12.0, "10")]
        components = [
            _component("component.a", "fragment.a", "9"),
            _component("component.b", "fragment.b", "10"),
        ]
        result = build_composite_projected_bar_proposals(fragments, components)
        self.assertEqual(result["proposals"], [])


if __name__ == "__main__":
    unittest.main()
