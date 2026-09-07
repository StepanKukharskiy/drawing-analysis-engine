import unittest

import fitz

from src.drawing_engine.core.pdf_native_metadata import get_native_drawings, inventory_page_metadata
from src.drawing_engine.core.vector_topology import extract_page_topology, iter_native_segments
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_page


class PdfNativeMetadataTest(unittest.TestCase):
    def test_layers_visibility_and_duplicate_names_do_not_supply_object_identity(self):
        doc = fitz.open()
        page = doc.new_page()
        first = doc.add_ocg('Piping', on=True)
        second = doc.add_ocg('Piping', on=False)
        page.draw_line((10, 10), (50, 10), oc=first)
        page.draw_line((10, 30), (50, 30), oc=second)
        inventory = inventory_page_metadata(page)
        self.assertEqual([r['default_configuration_visibility'] for r in inventory['optional_content_groups']], [True, False])
        drawings, enriched_inventory = get_native_drawings(page, extended_metadata=True)
        segments = list(iter_native_segments(drawings))
        self.assertEqual([r['id'] for r in segments], ['drawing[0].item[0].segment[0]', 'drawing[1].item[0].segment[0]'])
        for row in segments:
            self.assertEqual(row['native_metadata']['layer_name'], 'Piping')
            self.assertEqual(row['native_metadata']['layer_xref_candidates'], [first, second])
            self.assertIs(row['native_metadata']['object_identity_established'], False)
        self.assertEqual(inventory['default_configuration'], enriched_inventory['default_configuration'])
        selected = list(iter_native_segments(drawings, lambda d: d['seqno'] == 1))
        self.assertEqual(selected[0]['id'], segments[1]['id'])
        scope = {'record_type': 'mep_sheet_page_record', 'page_ref': 'page.1', 'page_number': 1, 'fields': {}}
        route = build_mep_route_page(page_scope=scope,
            native_topology=extract_page_topology(page, include_native_metadata=True))
        self.assertEqual(len(route['fragments']), 2)
        self.assertEqual(len(route['vertices']), 4)
        self.assertTrue(all(r['provenance']['native_metadata']['layer_name'] == 'Piping' for r in route['fragments']))
        self.assertEqual(route['native_metadata_inventory']['default_configuration'], inventory['default_configuration'])

    def test_layerless_legacy_ids_and_geometry_are_unchanged(self):
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect((20, 20, 70, 70))
        legacy = list(iter_native_segments(page.get_drawings()))
        topology = extract_page_topology(page)
        self.assertNotIn('native_metadata_inventory', topology)
        self.assertTrue(all('native_metadata' not in row for row in legacy))
        enriched = extract_page_topology(page, include_native_metadata=True)
        for before, after in zip(topology['segments'], enriched['segments']):
            self.assertEqual(before, {k: v for k, v in after.items() if k != 'native_metadata'})
        self.assertEqual(enriched['native_metadata_inventory']['optional_content_groups'], [])

    def test_rendering_groups_do_not_renumber_paths_or_claim_form_membership(self):
        source = fitz.open()
        inner = source.new_page(width=100, height=100)
        inner.draw_rect((10, 10, 90, 90), fill=(1, 0, 0), fill_opacity=.4)
        doc = fitz.open()
        page = doc.new_page()
        page.show_pdf_page(fitz.Rect(10, 10, 110, 110), source, 0)
        plain = list(iter_native_segments(page.get_drawings()))
        drawings, metadata = get_native_drawings(page, extended_metadata=True)
        enriched = list(iter_native_segments(drawings))
        self.assertTrue(metadata['form_xobjects'])
        self.assertEqual(metadata['form_to_primitive_membership'], 'not_established')
        self.assertTrue(metadata['rendering_contexts'])
        self.assertEqual(plain, [{k: v for k, v in row.items() if k != 'native_metadata'} for row in enriched])
        self.assertTrue(any(row['native_metadata']['rendering_context_refs'] for row in enriched))


if __name__ == '__main__':
    unittest.main()
