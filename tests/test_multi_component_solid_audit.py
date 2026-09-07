import unittest
from pathlib import Path

import fitz

from src.drawing_engine.audit.render_object_agnostic_audit import (
    _solid_replay_validation_text,
    draw_3d_or_status,
)
from src.drawing_engine.disciplines.concrete.multi_component_solid_replay import replay_multi_component_solid
from tests.test_multi_component_solid_replay import GRAPH_HASH, fixture, replay_fixture


def audit_record(replay):
    return {
        "page": 1,
        "contours": [],
        "engineering_graph": {
            "solid_replay": replay,
            "solid_preview": replay.get("solid_preview"),
            "quantities": [],
            "solid_hypotheses": [],
            "view_hypotheses": [],
            "object_instance_graph": {"summary": {}},
            "view_frame_graph": {"summary": {}},
            "dimension_ownership": {"summary": {}},
            "metric_equation_graph": {"summary": {}},
            "rebar_program": {"physical_path_graph": {"summary": {}}},
            "unresolved": [],
        },
    }


def normalized_page_text(page):
    return page.get_text().replace("\xa0", " ").replace("\xad", "-")


class MultiComponentSolidAuditTest(unittest.TestCase):
    def test_accepted_replay_exposes_every_validation_category(self):
        replay = replay_fixture()
        text = _solid_replay_validation_text(replay, "en")

        for heading in ("COMPONENT VALIDATION", "INTERFACES", "OVERLAP", "VOLUME", "REPROJECTION"):
            self.assertIn(heading, text)
        self.assertIn("component.a", text)
        self.assertIn("interface.a_b", text)
        self.assertIn("Preview only. Replay wrote no quantity.", text)

        output = fitz.open()
        draw_3d_or_status(output, Path("synthetic-solid.pdf"), audit_record(replay), "en", 0.01)
        rendered_text = normalized_page_text(output[0])
        output.close()
        self.assertIn("CERTIFIED SOLID REPLAY", rendered_text)
        self.assertIn("COMPONENT VALIDATION", rendered_text)
        self.assertIn("REPROJECTION", rendered_text)

    def test_abstaining_replay_shows_exact_reason_and_no_mesh(self):
        constraints, native_entities, pages, coordinate = fixture()
        pages[0]["solid_hypothesis_replay_records"][0]["physical_component_transforms"][0][
            "axis_signs"
        ] = "unresolved"
        replay = replay_multi_component_solid(
            canonical_graph_sha256=GRAPH_HASH,
            constraints=constraints,
            native_entities=native_entities,
            engineering_pages=pages,
            shared_coordinate_reclosure=coordinate,
        )

        self.assertEqual(replay["status"], "insufficient_constraints")
        self.assertIsNone(replay["solid_preview"])
        text = _solid_replay_validation_text(replay, "en")
        self.assertIn("axis signs are unresolved", text)
        self.assertIn("No assembly mesh was published", text)

        output = fitz.open()
        draw_3d_or_status(output, Path("synthetic-solid.pdf"), audit_record(replay), "en", 0.01)
        rendered_text = normalized_page_text(output[0])
        output.close()
        self.assertIn("AXONOMETRIC MODEL - NOT GENERATED", rendered_text)
        self.assertIn("axis signs are unresolved", rendered_text)

    def test_every_negative_replay_category_stays_meshless_in_audit(self):
        cases = (
            (
                "stale graph hash",
                lambda record: record.__setitem__("base_canonical_graph_sha256", "stale"),
            ),
            (
                "missing evidence",
                lambda record: record["components"][0].__setitem__("evidence_refs", []),
            ),
            (
                "presentation-only transform",
                lambda record: record["physical_component_transforms"][0].__setitem__(
                    "placement_role", "presentation_only"
                ),
            ),
            (
                "incomplete projections",
                lambda record: record["supplied_views"][0]["component_projections"].pop(),
            ),
        )
        for label, mutation in cases:
            with self.subTest(label=label):
                constraints, native_entities, pages, coordinate = fixture()
                mutation(pages[0]["solid_hypothesis_replay_records"][0])
                replay = replay_multi_component_solid(
                    canonical_graph_sha256=GRAPH_HASH,
                    constraints=constraints,
                    native_entities=native_entities,
                    engineering_pages=pages,
                    shared_coordinate_reclosure=coordinate,
                )
                self.assertIn(replay["status"], {"insufficient_constraints", "reclosed_fail"})
                self.assertIsNone(replay["solid_preview"])
                self.assertIn("No assembly mesh was published", _solid_replay_validation_text(replay, "en"))


if __name__ == "__main__":
    unittest.main()
