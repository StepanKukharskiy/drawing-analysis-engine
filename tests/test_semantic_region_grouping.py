import unittest

import fitz

from src.drawing_engine.core.semantic_region_grouping import build_primitive_graph, propose_semantic_regions


class SemanticRegionGroupingTest(unittest.TestCase):
    def test_section_proposals_use_page_relative_geometry(self):
        document = fitz.open()
        page = document.new_page(width=1000, height=1400)
        page.insert_text((470, 100), "1 - 1", fontsize=18)
        page.insert_text((470, 500), "2 - 2", fontsize=18)
        page.draw_rect(fitz.Rect(390, 130, 610, 450))
        page.draw_rect(fitz.Rect(380, 530, 620, 1200))

        proposals = propose_semantic_regions(page)
        sections = [proposal for proposal in proposals if proposal.kind == "section_view"]
        self.assertEqual([proposal.label for proposal in sections], ["section 1-1", "section 2-2"])
        self.assertLess(sections[0].bbox[1], 100)
        self.assertLess(sections[0].bbox[3], sections[1].bbox[3])
        self.assertTrue(all(0 <= value <= 1400 for proposal in sections for value in proposal.bbox))
        self.assertTrue(all(proposal.seed_node_ids for proposal in sections))
        self.assertTrue(all(proposal.component_node_ids for proposal in sections))
        self.assertTrue(all("bbox is the union" in " ".join(proposal.evidence) for proposal in sections))

    def test_graph_contains_native_text_paths_and_line_items(self):
        document = fitz.open()
        page = document.new_page(width=500, height=500)
        page.insert_text((50, 50), "1 - 1", fontsize=12)
        page.draw_rect(fitz.Rect(100, 100, 250, 250))
        page.draw_line((100, 275), (250, 275))
        graph = build_primitive_graph(page)
        kinds = {node.kind for node in graph.nodes}
        self.assertTrue({"text", "path", "line_segment"}.issubset(kinds))
        self.assertIn("path_contains_item", {edge.relation for edge in graph.edges})

    def test_page_without_strong_anchors_abstains(self):
        document = fitz.open()
        page = document.new_page(width=500, height=500)
        page.insert_text((40, 40), "GENERAL NOTES", fontsize=12)
        self.assertEqual(propose_semantic_regions(page), ())


if __name__ == "__main__":
    unittest.main()
