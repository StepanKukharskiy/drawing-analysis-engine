from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.mep_equipment_preparation import load_equipment_inputs_reference
import src.drawing_engine.disciplines.mep.mep_equipment_identity as identity
import src.drawing_engine.disciplines.mep.mep_projected_identity_inputs as inputs
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals
from test_mep_equipment_identity import identity_fixture


def write_fixture(directory):
    """Same native bytes in two roles, with different valid frozen M2 payloads."""
    root = Path(directory)
    native, typography = identity_fixture()
    observations = [page['items'][0]['inputs']['observation'] for page in native['pages']]
    terms = build_mep_terminology_proposals(document=native['document'], observations=observations)
    for page in native['pages']:
        item = page['items'][0]['inputs']
        item['proposal'] = next(row for row in terms['proposals'] if row['anchor_ref'] == item['observation']['id'])
    raw = gzip.compress(json.dumps(native).encode(), mtime=0)
    native_sha = hashlib.sha256(raw).hexdigest()
    native_path = root / 'native.json.gz'
    native_path.write_bytes(raw)
    earlier = None
    earlier_path = None
    for role in ('calibration', 'current'):
        role_dir = root / role
        role_dir.mkdir(exist_ok=True)
        role_terms = terms if earlier is None else build_mep_terminology_proposals(
            document=native['document'], observations=observations + [
                {'id': 'outside', 'page_ref': 'page.outside', 'text': 'HUH-99', 'region_role': 'unknown'}])
        terms_path = role_dir / 'terms.json'
        terms_path.write_text(json.dumps(role_terms))
        payload = identity.build_equipment_2d_identity(native_inputs=native, typography=typography,
            calibration=earlier['drawing_convention'] if earlier else None)
        payload.update(input_payload_sha256={'terminology-proposals': _sha256(role_terms)},
            native_replay_sha256=native_sha, native_payload_sha256=_sha256(native),
            typography_payload_sha256=_sha256(typography),
            calibration_payload_sha256=_sha256(earlier) if earlier else None,
            coverage={'execution_page_numbers': [p['page_number'] for p in native['pages']],
                'calibration_page_refs': sorted({s['page_ref'] for g in payload['drawing_convention']['groups']
                                                for s in g['members']}),
                'calibration_consumes_current_scope': earlier is None,
                'whole_package_equipment_coverage_established': False, 'engineer_approved': False,
                'reviewed_selectors_consumed': False})
        path = role_dir / 'equipment-2d-identities.json'
        path.write_text(json.dumps(payload))
        manifest = {'layer': 'mep_equipment_identity_replay_inputs', 'quantity_eligible': False,
            'identity_payload_sha256': _sha256(payload), 'typography': typography,
            'native_replay': {'path': str(native_path), 'file_sha256': native_sha},
            'terminology': {'path': str(terms_path), 'payload_sha256': _sha256(role_terms)},
            'calibration_payload': {'path': str(earlier_path), 'payload_sha256': _sha256(earlier)} if earlier else None}
        path.with_name('equipment-2d-replay.json').write_text(json.dumps(manifest))
        earlier, earlier_path = payload, path
    return path, role_terms


class EquipmentInputPreparationTest(unittest.TestCase):
    def test_exact_reference_parity_without_archive_round_trip_or_role_aliasing(self):
        with tempfile.TemporaryDirectory() as directory:
            path, terms = write_fixture(directory)
            with patch.object(inputs, '_decode_archive', wraps=inputs._decode_archive) as decode:
                expected = load_equipment_inputs_reference(path, terminology=terms)
                self.assertEqual(decode.call_count, 2)
            with patch.object(inputs, '_decode_archive', wraps=inputs._decode_archive) as decode, \
                    patch.object(identity, '_check_frozen_terms', wraps=identity._check_frozen_terms) as check:
                actual = inputs.load_equipment_identity_inputs(path, terminology=terms)
                self.assertEqual(decode.call_count, 0)
                self.assertEqual(check.call_count, 2)
                self.assertNotEqual(check.call_args_list[0].args[1], check.call_args_list[1].args[1])
            self.assertEqual(expected, actual)
            self.assertEqual(_sha256(expected), _sha256(actual))
            kwargs = actual[1]
            self.assertEqual(kwargs['native_inputs'], kwargs['calibration_native_inputs'])
            self.assertIsNot(kwargs['native_inputs'], kwargs['calibration_native_inputs'])
            kwargs['native_inputs']['pages'][0]['items'].clear()
            self.assertTrue(kwargs['calibration_native_inputs']['pages'][0]['items'])
            with self.assertRaisesRegex(ValueError, 'competitor search'):
                identity.replay_equipment_2d_identity(actual[0], terminology=terms, **kwargs)
            # Returned objects are not a persistent cache or the portable source.
            self.assertEqual(expected, inputs.load_equipment_identity_inputs(path, terminology=terms))
            self.assertEqual(expected[:2], inputs.replay_equipment_identity_artifacts(actual[2], terminology=terms))

    def test_portable_mutation_and_current_m2_still_require_independent_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path, terms = write_fixture(directory)
            _, _, artifacts = inputs.load_equipment_identity_inputs(path, terminology=terms)
            native_name = artifacts['equipment-2d-replay']['portable']['native_archive_artifact']
            for field, value in [('data_base64', ''), ('payload_sha256', '0'*64),
                                 ('file_sha256', '0'*64), ('document', {}), ('quantity_eligible', True)]:
                changed = deepcopy(artifacts)
                changed[native_name][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    inputs.replay_equipment_identity_artifacts(changed, terminology=terms)
            calibration_terms = artifacts[artifacts['equipment-2d-calibration-replay']['portable']['terminology_artifact']]
            for load in (load_equipment_inputs_reference, inputs.load_equipment_identity_inputs):
                with self.assertRaisesRegex(ValueError, 'differs from current M2'):
                    load(path, terminology=calibration_terms)
            # Rebinding calibration to current M2, even with updated manifest
            # hashes and identical native bytes, cannot bypass interpretation.
            changed = deepcopy(artifacts)
            current = changed['equipment-2d-replay']
            earlier = changed['equipment-2d-calibration-replay']
            earlier['portable']['terminology_artifact'] = current['portable']['terminology_artifact']
            earlier['terminology']['payload_sha256'] = current['terminology']['payload_sha256']
            with self.assertRaisesRegex(ValueError, 'different frozen M2'):
                inputs.replay_equipment_identity_artifacts(changed, terminology=terms)

    def test_prepared_loader_rechecks_source_bytes_document_and_competitors_each_operation(self):
        for mutation, reason in [('bytes', 'file hash mismatch'), ('document', 'source document mismatch'),
                                 ('source_pdf', 'source PDF mismatch'), ('competitor', 'competitor search')]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, terms = write_fixture(directory)
                expected = inputs.load_equipment_identity_inputs(path, terminology=terms)
                raw = (Path(directory) / 'native.json.gz').read_bytes()
                native = json.loads(gzip.decompress(raw))
                if mutation == 'document':
                    native['document'] = {}
                elif mutation == 'source_pdf':
                    native['source_pdf_sha256'] = 'b'*64
                elif mutation == 'competitor':
                    native['pages'][0]['items'].clear()
                else:
                    native['unexpected'] = True
                changed = gzip.compress(json.dumps(native).encode(), mtime=0)
                (Path(directory) / 'native.json.gz').write_bytes(changed)
                if mutation != 'bytes':
                    # Rebind file/claim/manifest hashes so these negatives exercise
                    # source and interpretation checks, not just changed raw bytes.
                    earlier_sha = None
                    for role in ('calibration', 'current'):
                        role_path = Path(directory) / role / path.name
                        payload = json.loads(role_path.read_text())
                        manifest_path = role_path.with_name('equipment-2d-replay.json')
                        manifest = json.loads(manifest_path.read_text())
                        payload['native_replay_sha256'] = hashlib.sha256(changed).hexdigest()
                        payload['calibration_payload_sha256'] = earlier_sha
                        manifest['native_replay']['file_sha256'] = payload['native_replay_sha256']
                        if earlier_sha:
                            manifest['calibration_payload']['payload_sha256'] = earlier_sha
                        earlier_sha = _sha256(payload)
                        manifest['identity_payload_sha256'] = earlier_sha
                        role_path.write_text(json.dumps(payload))
                        manifest_path.write_text(json.dumps(manifest))
                for load in (load_equipment_inputs_reference, inputs.load_equipment_identity_inputs):
                    with self.subTest(load=load.__name__), self.assertRaisesRegex(ValueError, reason):
                        load(path, terminology=terms)
                # Recreating the same paths must not leave stale decoded inputs.
                path, terms = write_fixture(directory)
                self.assertEqual(expected, inputs.load_equipment_identity_inputs(path, terminology=terms))

    def test_archive_modified_inside_preparation_cannot_use_saved_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path, terms = write_fixture(directory)
            original = inputs._equipment_artifacts

            def changed(*args, **kwargs):
                artifacts = original(*args, **kwargs)
                if args[2] == 'equipment-2d':
                    name = artifacts['equipment-2d-replay']['portable']['native_archive_artifact']
                    artifacts[name]['quantity_eligible'] = True
                return artifacts

            with patch.object(inputs, '_equipment_artifacts', side_effect=changed), \
                    self.assertRaisesRegex(ValueError, 'archive changed during preparation'):
                inputs.load_equipment_identity_inputs(path, terminology=terms)


class EquipmentInputPreparationReplayTest(unittest.TestCase):
    def test_frozen_pilot_and_heldout_calibration_match_complete_reference(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('pilot', 'holdout'):
            path = root / f'output/mep-equipment-2d-{name}-2026-08-31/equipment-2d-identities.json'
            expected = load_equipment_inputs_reference(path)
            expected_sha = _sha256(expected)
            del expected  # Do not hold four independent decoded native trees.
            actual = inputs.load_equipment_identity_inputs(path)
            self.assertEqual(expected_sha, _sha256(actual))
            self.assertEqual(actual[0], json.loads(path.read_text()))
            del actual


if __name__ == '__main__':
    unittest.main()
