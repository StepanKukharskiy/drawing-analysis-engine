import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.dimension_attachment import attach_dimensions, attachment_payload


ROOT = Path(__file__).resolve().parents[1]


def add_horizontal_dimension(page: fitz.Page, y: float, value: str = "100") -> None:
    x0, x1 = 100.0, 200.0
    page.insert_text((140, y - 5), value, fontsize=10)
    page.draw_line((x0, y), (x1, y), width=0.5)
    page.draw_line((x0, y), (x0, y + 30), width=0.5)
    page.draw_line((x1, y), (x1, y + 30), width=0.5)
    for x in (x0, x1):
        page.draw_line((x, y), (x + 3, y + 3), width=0.5)
        page.draw_line((x, y), (x - 3, y - 3), width=0.5)


def add_inset_baseline_dimension(page: fitz.Page, y: float, value: str = "100") -> None:
    x0, x1 = 100.0, 200.0
    inset = 6.0
    page.insert_text((140, y - 5), value, fontsize=10)
    page.draw_line((x0 + inset, y), (x1 - inset, y), width=0.5)
    page.draw_line((x0, y), (x0, y + 30), width=0.5)
    page.draw_line((x1, y), (x1, y + 30), width=0.5)
    for x in (x0, x1):
        page.draw_line((x, y), (x + 3, y + 3), width=0.5)
        page.draw_line((x, y), (x - 3, y - 3), width=0.5)


def add_arrow_dimension(page: fitz.Page, y: float, value: str = "100") -> None:
    x0, x1 = 100.0, 200.0
    page.insert_text((140, y - 5), value, fontsize=10)
    page.draw_line((x0, y), (x1, y), width=0.5)
    page.draw_line((x0, y), (x0, y + 30), width=0.5)
    page.draw_line((x1, y), (x1, y + 30), width=0.5)
    for endpoint, direction in ((x0, 1.0), (x1, -1.0)):
        page.draw_line((endpoint, y), (endpoint + direction * 6, y - 3), width=0.5)
        page.draw_line((endpoint, y), (endpoint + direction * 6, y + 3), width=0.5)
    page.draw_line((x0, y + 30), (x1, y + 30), width=1.2)


class DimensionAttachmentTest(unittest.TestCase):
    def test_repeated_scale_and_complete_native_geometry_are_accepted(self):
        document = fitz.open()
        page = document.new_page(width=400, height=300)
        for y in (50.0, 130.0, 210.0):
            add_horizontal_dimension(page, y)

        attachments = attach_dimensions(page)
        accepted = [item for item in attachments if item.status == "accepted"]
        self.assertEqual(len(accepted), 3)
        for item in accepted:
            self.assertAlmostEqual(item.scale_points_per_mm, 1.0, places=3)
            self.assertEqual(item.scale_support, 3)
            self.assertAlmostEqual(item.endpoint_alignment_residual, 0.0, places=3)
            self.assertGreaterEqual(len(item.terminal_refs), 4)
            self.assertEqual(item.measured_points[0][1], item.measured_points[1][1])

        payload = attachment_payload(attachments)
        self.assertEqual(payload["accepted_count"], 3)
        self.assertEqual(payload["ambiguous_count"], 0)
        self.assertTrue(all(row["primitive_refs"] for row in payload["attachments"]))

    def test_numeric_text_without_dimension_geometry_abstains(self):
        document = fitz.open()
        page = document.new_page(width=300, height=200)
        page.insert_text((100, 100), "400", fontsize=10)
        self.assertEqual(attach_dimensions(page), ())

    def test_extension_lines_outside_inset_baseline_are_accepted(self):
        document = fitz.open()
        page = document.new_page(width=400, height=300)
        for y in (50.0, 130.0, 210.0):
            add_inset_baseline_dimension(page, y)

        accepted = [item for item in attach_dimensions(page) if item.status == "accepted"]
        self.assertEqual(len(accepted), 3)
        self.assertTrue(all(item.scale_support == 3 for item in accepted))

    def test_arrowed_chains_share_the_native_proposal_schema(self):
        document = fitz.open()
        page = document.new_page(width=400, height=320)
        for y in (50.0, 140.0, 230.0):
            add_arrow_dimension(page, y)

        accepted = [item for item in attach_dimensions(page) if item.status == "accepted"]
        self.assertEqual(len(accepted), 3)
        self.assertTrue(all(item.terminal_style == "arrow" for item in accepted))
        self.assertTrue(all(all(item.terminal_refs_by_endpoint) for item in accepted))
        payload = attachment_payload(tuple(accepted))
        self.assertTrue(all(row["terminal_style"] == "arrow" for row in payload["attachments"]))

    def test_midpoint_ticks_and_rotated_numeric_text_are_recognized(self):
        source = fitz.open(ROOT / "КЖ0-2-4.pdf")
        document = fitz.open()
        document.insert_pdf(source)
        document[0].remove_rotation()
        attachments = attach_dimensions(document[0])
        accepted = {(item.orientation, item.value_mm) for item in attachments if item.status == "accepted"}

        self.assertIn(("horizontal", 3600.0), accepted)
        self.assertIn(("vertical", 2880.0), accepted)
        self.assertIn(("vertical", 200.0), accepted)
        cross_section_widths = [
            item
            for item in attachments
            if item.status == "accepted"
            and item.orientation == "horizontal"
            and item.value_mm == 3600.0
            and abs(item.scale_points_per_mm / 0.0945 - 1.0) < 0.01
        ]
        self.assertTrue(cross_section_widths)


if __name__ == "__main__":
    unittest.main()
