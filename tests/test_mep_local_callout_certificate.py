"""Real native callout and bend replay, with incomplete-evidence negatives."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import fitz

from tools.generate_mep_local_callout_certificate import replay_certificate
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256


class LocalCalloutCertificateReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((Path(__file__).resolve().parents[1] /
            'output/mep-local-callout-certificate-2026-09-04/certificate-v2.json').read_text())

    def test_real_local_binding_and_system_only_bend_replay(self):
        p = self.payload
        self.assertEqual([], replay_certificate(p))
        local = p['summary']['local_target']
        destination = p['summary']['extended_target']
        self.assertEqual({'route_system', 'route_size'}, {
            r['relation_type'] for r in p['local_M4_bindings']['relations']
            if r['state'] == 'accepted' and local in r['target_refs']})
        self.assertEqual({'route_system'}, {
            r['relation_type'] for r in p['extended_M4_bindings']['relations']
            if r['state'] == 'accepted' and destination in r['target_refs']})
        self.assertFalse(p['summary']['source_page_coverage_established'])
        self.assertIsNone(p['summary']['installed_length'])
        self.assertIsNone(p['summary']['purchase_length'])

    def test_missing_native_query_or_changed_bend_cannot_replay(self):
        # Copy only the changed branch; replay must not mutate frozen inputs.
        p = dict(self.payload)
        p['leader_native_queries'] = p['leader_native_queries'][1:]
        with self.assertRaisesRegex(ValueError, 'native leader query'):
            replay_certificate(p)
        p = dict(self.payload)
        p['bend'] = deepcopy(p['bend'])
        p['bend']['search']['complete'] = False
        with self.assertRaises(ValueError):
            replay_certificate(p)


class LocalCalloutCertificateRenderTest(unittest.TestCase):
    def test_one_raster_base_vector_overlay_and_exact_certificate_hash(self):
        root = Path(__file__).resolve().parents[1]
        certificate = root / 'output/mep-local-callout-certificate-2026-09-04/certificate-v2.json'
        with fitz.open(root / 'output/pdf/mep_local_callout_and_bend_2026-09-04.pdf') as pdf:
            self.assertEqual(1, len(pdf))
            page = pdf[0]
            self.assertEqual(1, len(page.get_images()))
            self.assertGreater(len(page.get_drawings()), 0)
            text = page.get_text()
            self.assertIn(_file_sha256(certificate), text)
            self.assertIn('Size and elevation not propagated', text)
            self.assertIn('not a complete network audit', text)
