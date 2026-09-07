import copy
import inspect
import json
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.rebar.rebar_takeoff_adapter import build_rebar_takeoff_adapter
from src.drawing_engine.project.takeoff_intelligence import validate_takeoff_adapter


ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "output" / "object_agnostic"


def _graph(name: str) -> dict:
    return json.loads((GRAPHS / f"{name}.engineering-graph.json").read_text(encoding="utf-8"))


class RebarTakeoffAdapterTest(unittest.TestCase):
    def test_resolved_detail_takeoff_maps_scoped_count_and_length_but_not_mass(self):
        result = build_rebar_takeoff_adapter(_graph("3179 ЛС (2)-2"))

        self.assertEqual(result["summary"], {
            "occurrence_count": 20,
            "physical_item_count": 12,
            "calculated_line_count": 24,
            "unresolved_occurrence_count": 8,
        })
        counts = [row for row in result["calculated_lines"] if row["metric_kind"] == "physical_count"]
        lengths = [row for row in result["calculated_lines"] if row["metric_kind"] == "fabrication_length"]
        self.assertEqual(sum(row["value"] for row in counts), 181)
        self.assertAlmostEqual(sum(row["value"] for row in lengths), 187.985)
        self.assertFalse(any(row["metric_kind"] == "mass" for row in result["calculated_lines"]))
        self.assertTrue(all(
            row["diameter_mm"]["state"] == "convention_dependent"
            for row in result["physical_items"]
        ))
        self.assertEqual(validate_takeoff_adapter(result), [])

    def test_mixed_family_fixture_preserves_accepted_and_unresolved_separately(self):
        result = build_rebar_takeoff_adapter(_graph("1"))

        self.assertEqual(len(result["occurrences"]), 18)
        self.assertEqual(len(result["physical_items"]), 10)
        self.assertEqual(result["summary"]["unresolved_occurrence_count"], 8)
        self.assertEqual(
            [(row["metric_kind"], row["value"]) for row in result["calculated_lines"]],
            [("physical_count", 4), ("fabrication_length", 37.04), ("physical_count", 67)],
        )
        self.assertFalse(any(row["metric_kind"] == "mass" for row in result["calculated_lines"]))

    def test_unresolved_drawing_emits_observation_without_item_or_quantity(self):
        result = build_rebar_takeoff_adapter(_graph("candidate-08-staircase-page"))

        self.assertEqual(len(result["occurrences"]), 1)
        self.assertEqual(result["occurrences"][0]["item_subtype"], "unresolved_observation_scope")
        self.assertTrue(result["occurrences"][0]["unresolved_reasons"])
        self.assertEqual(result["physical_items"], [])
        self.assertEqual(result["calculated_lines"], [])
        self.assertEqual(result["declared_lines"], [])

    def test_profile_estimates_never_enter_calculated_channel(self):
        graph = _graph("v24")
        result = build_rebar_takeoff_adapter(graph)
        changed = copy.deepcopy(graph)
        changed["pages"][0]["rebar_program"]["estimated_quantity_takeoff"]["totals"] = {
            "physical_bar_count": 999,
            "fabrication_length_m": 999.0,
            "mass_kg": 999.0,
        }
        replay = build_rebar_takeoff_adapter(changed)

        self.assertEqual(len(result["occurrences"]), 9)
        self.assertEqual(result["physical_items"], [])
        self.assertEqual(result["calculated_lines"], [])
        self.assertEqual(replay["occurrences"], result["occurrences"])
        self.assertEqual(replay["physical_items"], result["physical_items"])
        self.assertEqual(replay["calculated_lines"], result["calculated_lines"])
        self.assertNotEqual(
            replay["native_payload_ref"]["payload_sha256"],
            result["native_payload_ref"]["payload_sha256"],
        )

    def test_missing_document_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "document_key"):
            build_rebar_takeoff_adapter({"pages": []})

    def test_unclosed_detail_elements_do_not_disappear_or_promote(self):
        result = build_rebar_takeoff_adapter({
            "document_key": "doc.synthetic",
            "pages": [{
                "page": 1,
                "rebar_program": {
                    "native_vector_detail_linking": {
                        "status": "unresolved",
                        "physical_families": [],
                        "reason": "family identity unresolved",
                    },
                    "unknowns": ["object binding is not unique"],
                },
                "reinforcement_quantities": {
                    "status": "partial",
                    "elements": [{
                        "object_instance_id": "object.1",
                        "mark": "1",
                        "count": 4,
                        "total_length_m": 10.0,
                    }],
                },
            }],
        })

        self.assertEqual(len(result["occurrences"]), 1)
        self.assertTrue(result["occurrences"][0]["unresolved_reasons"])
        self.assertEqual(result["physical_items"], [])
        self.assertEqual(result["calculated_lines"], [])

    def test_adapter_is_drawing_neutral_and_has_no_declaration_or_approval_path(self):
        source = inspect.getsource(__import__("src.drawing_engine.disciplines.rebar.rebar_takeoff_adapter", fromlist=["*"])).lower()
        for forbidden in (
            "candidate-08",
            "3179",
            ".pdf",
            "estimated_quantity_takeoff",
            "convention_dependent_mass_kg",
        ):
            self.assertNotIn(forbidden, source)
        result = build_rebar_takeoff_adapter({
            "document_key": "doc.synthetic",
            "pages": [],
        })
        self.assertEqual(result["declared_lines"], [])
        self.assertNotIn("approval", result)


if __name__ == "__main__":
    unittest.main()
