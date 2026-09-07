import json
import sqlite3
import unittest
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_sheet_region_ownership import (
    build_mep_sheet_region_ownership, validate_mep_sheet_region_ownership,
)


ROOT = Path(__file__).resolve().parents[1]


class MepSheetRegionOwnershipUnitTest(unittest.TestCase):
    def test_repeated_edge_grid_and_border_are_non_route_but_interior_view_survives(self):
        pages = []
        candidates = []
        for page_number in range(1, 4):
            page_ref = f"page.{page_number}"
            pages.append({
                "page_number": page_number, "page_ref": page_ref,
                "page_rect_display": [0, 0, 1000, 700],
                "registry_role": "mechanical_piping_plan",
                "text_blocks": [{"bbox_display": [920, 100, 990, 650],
                                 "text": "DRAWING REVISION SHEET SCALE"}],
            })
            for index, y in enumerate((100, 180, 260, 340, 420, 500)):
                candidates.append({
                    "route_ledger_ref": f"edge-row.{page_number}.{index}",
                    "occurrence_ref": f"edge-occ.{page_number}.{index}",
                    "page_ref": page_ref, "points_display": [[920, y], [990, y]],
                    "projected_2d_length_m": 1.0, "source_primitive_refs": [],
                })
            candidates.extend([
                {"route_ledger_ref": f"border.{page_number}",
                 "occurrence_ref": f"border-occ.{page_number}", "page_ref": page_ref,
                 "points_display": [[0, 10], [1000, 10]], "projected_2d_length_m": 10.0,
                 "source_primitive_refs": []},
                {"route_ledger_ref": f"route.{page_number}",
                 "occurrence_ref": f"route-occ.{page_number}", "page_ref": page_ref,
                 "points_display": [[200, 250], [600, 250]], "projected_2d_length_m": 4.0,
                 "source_primitive_refs": []},
            ])
        payload = build_mep_sheet_region_ownership(
            document={"source_pdf_sha256": "source"}, pages=pages,
            route_candidates=candidates, source_pdf_sha256="source")
        self.assertEqual(validate_mep_sheet_region_ownership(payload), [])
        memberships = {row["occurrence_ref"]: row for row in payload["candidate_ownership"]}
        self.assertEqual(memberships["edge-occ.1.0"]["region_role"], "title_block_revision_stamp")
        self.assertEqual(memberships["border-occ.1"]["region_role"], "border")
        self.assertTrue(memberships["route-occ.1"]["route_certification_eligible"])
        self.assertEqual(len(payload["non_route_drawing_content"]), 21)

    def test_ruled_schedule_text_excludes_local_grid_but_not_nearby_route(self):
        page = {"page_number": 1, "page_ref": "page.1", "page_rect_display": [0, 0, 1000, 700],
                "registry_role": "mechanical_piping_plan",
                "text_blocks": [{"bbox_display": [100, 100, 400, 125], "text": "EQUIPMENT SCHEDULE"}]}
        candidates = [{
            "route_ledger_ref": "grid.horizontal", "occurrence_ref": "grid.horizontal.occ",
            "page_ref": "page.1", "points_display": [[100, 120], [400, 120]],
            "projected_2d_length_m": 3.0, "source_primitive_refs": [],
        }]
        for index, x in enumerate((120, 180, 240, 300)):
            candidates.append({
                "route_ledger_ref": f"grid.vertical.{index}",
                "occurrence_ref": f"grid.vertical.{index}.occ", "page_ref": "page.1",
                "points_display": [[x, 100], [x, 180]], "projected_2d_length_m": .8,
                "source_primitive_refs": [],
            })
        candidates.append({
            "route_ledger_ref": "real.route", "occurrence_ref": "real.route.occ",
            "page_ref": "page.1", "points_display": [[500, 300], [800, 300]],
            "projected_2d_length_m": 3.0, "source_primitive_refs": [],
        })
        payload = build_mep_sheet_region_ownership(
            document={}, pages=[page], route_candidates=candidates, source_pdf_sha256="source")
        memberships = {row["occurrence_ref"]: row for row in payload["candidate_ownership"]}
        self.assertEqual(memberships["grid.horizontal.occ"]["region_role"], "schedule_table")
        self.assertTrue(memberships["real.route.occ"]["route_certification_eligible"])


class MepSheetRegionOwnershipRealOutputTest(unittest.TestCase):
    def setUp(self):
        self.ownership = json.loads((ROOT / "output/mep-sheet-region-ownership-2026-09-03/sheet-region-ownership.json").read_text())

    def test_repeated_title_block_family_is_fully_excluded(self):
        self.assertEqual(validate_mep_sheet_region_ownership(self.ownership), [])
        summary = self.ownership["summary"]
        self.assertEqual(summary["excluded_non_route_candidate_count"], 104)
        self.assertAlmostEqual(summary["excluded_projected_length_m_by_role"]["title_block_revision_stamp"],
                               351.32486208)
        self.assertEqual(len(self.ownership["regions"]), 13)
        self.assertTrue(all(row["region_role"] == "title_block_revision_stamp"
                            and row["route_length_counted"] is False
                            for row in self.ownership["non_route_drawing_content"]))

    def test_non_route_geometry_is_preserved_in_sqlite(self):
        database = ROOT / "output/projects/mep/sheet-region-ownership-v1.sqlite"
        with sqlite3.connect(database) as connection:
            count, length = connection.execute(
                "SELECT count(*),sum(projected_length_m) FROM non_route_drawing_content").fetchone()
            self.assertEqual(count, 104)
            self.assertAlmostEqual(length, 351.32486208)
            snapshot = json.loads(connection.execute(
                "SELECT value_json FROM metadata WHERE key='snapshot_id'").fetchone()[0])
            self.assertRegex(snapshot, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
