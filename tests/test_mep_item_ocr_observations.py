from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.mep.mep_item_ocr_observations import (
    build_mep_item_ocr_proposals, extract_mep_item_ocr_observations,
    page_ocr_selection, validate_mep_item_ocr_observations,
)
from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_mep_sheet_registry, build_sheet_page_record
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations


ROOT = Path(__file__).resolve().parents[1]


def _data():
    words = [
        (1, '2"', 20, 20, 30, 20, '98'), (1, 'HWR', 55, 20, 70, 20, '96'),
        (2, '0"', 20, 50, 30, 20, '98'), (3, 'VALVE', 20, 80, 70, 20, '20'),
    ]
    return {key: [row[i] for row in words] for i, key in enumerate(
        ('line_num', 'text', 'left', 'top', 'width', 'height', 'conf'))} | {
        'block_num': [1] * len(words), 'par_num': [1] * len(words),
    }


def _inputs(directory, rotation=0):
    source = Path(directory) / 'drawing.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        page.insert_text((10, 20), '2" HWR', fontsize=10)
        page.set_rotation(rotation)
        size = list(page.rect.br)
        pdf.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    quality = {'route': 'hybrid_text_ocr', 'reason': 'synthetic outlined text', 'metrics': {
        'native_text_chars': 0, 'native_path_item_count': 12, 'embedded_image_area_ratio': 0,
    }}
    page_record = build_sheet_page_record(page_ref='test.page', page_number=1,
        page_width=size[0], page_height=size[1], native_tokens=[], quality=quality)
    registry = build_mep_sheet_registry(document={'document_key': 'pdf-sha256:' + digest,
        'source_pdf_sha256': digest, 'source_bytes': source.stat().st_size, 'page_count': 1}, pages=[page_record])
    native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
    return dict(pdf_path=source, sheet_registry=registry, native_text_observations=native)


class MepItemOcrObservationsTest(unittest.TestCase):
    def test_quality_selection_is_graphic_density_based_not_sheet_name(self):
        registry = json.loads((ROOT / 'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json').read_text())
        self.assertEqual([page['page_number'] for page in registry['pages'] if page_ocr_selection(page)['selected']], [16, 17, 18])
        sparse = deepcopy(registry['pages'][16])
        sparse['fields'] = {}
        sparse['role'] = 'arbitrary'
        sparse['page_number'] = 1
        self.assertTrue(page_ocr_selection(sparse)['selected'])
        self.assertFalse(page_ocr_selection(registry['pages'][0])['selected'])

    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations.pytesseract.get_tesseract_version', return_value='test-engine')
    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations._run_ocr', return_value=_data())
    def test_raw_words_null_authority_and_native_alternatives_survive(self, ocr, version):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _inputs(directory)
            before = deepcopy(inputs['native_text_observations'])
            payload = extract_mep_item_ocr_observations(**inputs, render_scale=2, tile_pixels=1024)
            self.assertEqual(inputs['native_text_observations'], before)
            self.assertEqual(validate_mep_item_ocr_observations(payload), [])
            self.assertEqual(payload['crops'][0]['raw_tsv_rows'][0]['text'], '2"')
            self.assertEqual(payload['crops'][0]['pixel_to_display_matrix'], [.5, 0, 0, .5, 0, 0])
            line = next(row for row in payload['observations'] if row['text'] == '2" HWR')
            self.assertEqual(line['bbox_display'], [10, 10, 62.5, 20])
            self.assertEqual(line['confidence'], .96)
            self.assertTrue(line['native_overlap_alternatives'])
            self.assertFalse(line['native_overlap_alternatives'][0]['automatic_merge_established'])
            self.assertEqual(payload['summary']['handoff_state_counts'], {'proposal_only': 1, 'quarantined': 2})
            proposals = build_mep_item_ocr_proposals(payload)
            self.assertTrue(proposals['proposals'])
            self.assertTrue(all(row['state'] == 'abstained' for row in proposals['proposals']))
            self.assertEqual(len(proposals['observation_diagnostics']), 2)
            self.assertFalse(payload['pages'][0]['item_inventory_complete'])
            corrupted = deepcopy(payload)
            corrupted['observations'][0]['bbox_display'][0] += 4
            self.assertTrue(validate_mep_item_ocr_observations(corrupted))
            with self.assertRaisesRegex(ValueError, 'invalid OCR evidence'):
                build_mep_item_ocr_proposals(corrupted)

    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations.pytesseract.get_tesseract_version', return_value='test-engine')
    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations._run_ocr', return_value=_data())
    def test_tile_offsets_rotation_and_budget_remain_explicit(self, ocr, version):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _inputs(directory, rotation=90)
            payload = extract_mep_item_ocr_observations(**inputs, render_scale=3, tile_pixels=256, max_crops=3)
            self.assertEqual(payload['pages'][0]['state'], 'partial_or_failed_raster_scan')
            self.assertLess(payload['pages'][0]['processed_raster_area_fraction'], 1)
            completed = [crop for crop in payload['crops'] if crop['state'] == 'completed']
            self.assertEqual(len(completed), 3)
            self.assertGreater(completed[1]['pixel_to_display_matrix'][4], 0)
            self.assertEqual(completed[2]['pixel_to_display_matrix'][4], 448 / 3)
            row = payload['observations'][0]
            crop = completed[0]
            self.assertEqual(row['bbox_pdf'], list(fitz.Rect(row['bbox_display']) * fitz.Matrix(crop['display_to_pdf_matrix'])))
            corrupted = deepcopy(payload)
            corrupted['pages'][0]['processed_raster_area_fraction'] = 1
            self.assertIn('page raster coverage does not replay', validate_mep_item_ocr_observations(corrupted))

    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations.pytesseract.get_tesseract_version', return_value='test-engine')
    @patch('src.drawing_engine.disciplines.mep.mep_item_ocr_observations._run_ocr', side_effect=RuntimeError('OCR timeout'))
    def test_failed_crop_never_claims_empty_page_or_complete_coverage(self, ocr, version):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _inputs(directory)
            payload = extract_mep_item_ocr_observations(**inputs, max_crops=1)
            self.assertEqual(payload['crops'][0]['state'], 'failed')
            self.assertEqual(payload['pages'][0]['processed_raster_area_fraction'], 0)
            self.assertFalse(payload['pages'][0]['item_absence_established'])
            self.assertEqual(payload['observations'], [])
            inputs['sheet_registry']['document']['source_pdf_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'source PDF does not match M1'):
                extract_mep_item_ocr_observations(**inputs)
