from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import fitz
import numpy as np
import trimesh

from src.drawing_engine.disciplines.concrete.generic_prismatic_solver import _box_mesh
from src.drawing_engine.cli import resolve_export_source
from src.drawing_engine.exports.estimation_project_export import (
    FrozenProject, add_project_database, artifact_record, export_optional)


ROOT = Path(__file__).resolve().parents[1]


def assembly():
    return {'mark': 'EP14', 'state': 'derived',
            'linear_unit_evidence': {'state': 'direct', 'unit': 'mm'},
            'child_parts': [{'id': 'part:7', 'mark': '7', 'kind': 'plate', 'state': 'derived',
                'reprojection': {'passed': True}, 'solid': {
                    **_box_mesh((120, 80, 8)), 'coordinate_basis': 'relative orthographic millimetre frame'},
                'calculated': {'volume_m3': .0000768}, 'approved': None}]}


def frozen_project(directory, payload, *, artifact='assembly'):
    """A synthetic snapshot using the actual portable SQLite writer/resolver."""
    directory.mkdir(parents=True)
    with fitz.open() as pdf:
        pdf.new_page().insert_text((30, 30), 'Synthetic frozen source')
        pdf.save(directory / 'source.pdf')
    (directory / f'{artifact}.json').write_text(json.dumps(payload))
    (directory / 'declarations.json').write_text(json.dumps({'declared_volume': 999999}))
    manifest = {'schema_version': 'estimation_cli.v1', 'execution_status': 'succeeded',
        'source_sha256': hashlib.sha256((directory / 'source.pdf').read_bytes()).hexdigest(),
        'mode': 'native_detail_assembly' if artifact == 'assembly' else 'fresh_analysis',
        'contract': {'fixture': 'synthetic export test'}, 'artifacts': {}}
    for key, filename in [('source_pdf', 'source.pdf'), (artifact, f'{artifact}.json'),
                           ('declarations', 'declarations.json')]:
        manifest['artifacts'][key] = artifact_record(directory, directory / filename)
    add_project_database(directory, manifest)
    (directory / 'result.json').write_text(json.dumps(manifest))
    # Prove exports depend on SQLite, not the intermediate JSON files.
    (directory / f'{artifact}.json').unlink()
    (directory / 'declarations.json').unlink()
    return FrozenProject(directory, manifest)


class ObjExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, *args, cwd=None):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'rebar.py'), 'export', *args,
            '--reserve-gib', '0.05', '--max-growth-gib', '0.05'],
            cwd=cwd or self.root, env={**os.environ, 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1',
                'REBAR_DISK_USAGE_LOG': str(self.root / 'disk.jsonl')}, capture_output=True, text=True)

    def test_bare_export_finds_hidden_project_and_chooses_new_output_on_repeat(self):
        project = frozen_project(self.root / '.estimation-output', assembly())
        before = project.database.read_bytes()
        first = self.cli('--format', 'obj')
        self.assertEqual(first.returncode, 0, first.stderr)
        original = (self.root / 'obj-export' / 'EP14-part-7.mm.obj').read_bytes()
        second = self.cli('--format', 'obj')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((self.root / 'obj-export-2' / 'EP14-part-7.mm.obj').read_bytes(), original)
        self.assertEqual((self.root / 'obj-export' / 'EP14-part-7.mm.obj').read_bytes(), original)
        self.assertEqual(project.database.read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.temporary-*')))

    def test_discovery_supports_timestamp_and_batch_document_layouts(self):
        for index, relative in enumerate(('estimation-output/20260907',
                'estimation-output/20260907/001-drawing', '.estimation-output/001-drawing')):
            with self.subTest(relative=relative):
                folder = self.root / str(index)
                project = frozen_project(folder / relative, assembly())
                self.assertEqual(resolve_export_source(folder), (project.root / 'result.json').resolve())

    def test_project_folder_takes_precedence_and_explicit_source_still_works(self):
        project = frozen_project(self.root / 'project', assembly())
        frozen_project(project.root / '.estimation-output', assembly())
        self.assertEqual(resolve_export_source(project.root), (project.root / 'result.json').resolve())
        result = self.cli(str(project.root), '--format', 'obj', '--output', './chosen')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'chosen' / 'EP14-part-7.mm.obj').is_file())

    def test_batch_folder_resolves_document_but_batch_index_is_not_a_snapshot(self):
        project = frozen_project(self.root / 'batch' / '001-drawing', assembly())
        index = self.root / 'batch' / 'result.json'
        index.write_text(json.dumps({'schema_version': 'estimation_batch.v1', 'execution_status': 'succeeded'}))
        self.assertEqual(resolve_export_source(index.parent), (project.root / 'result.json').resolve())
        with self.assertRaisesRegex(ValueError, 'batch index'):
            resolve_export_source(index)

    def test_ambiguous_discovery_lists_projects_without_starting_export(self):
        a = frozen_project(self.root / 'estimation-output/a', assembly())
        b = frozen_project(self.root / 'estimation-output/b', assembly())
        result = self.cli('--format', 'obj')
        self.assertEqual(result.returncode, 1)
        self.assertIn('multiple projects', result.stderr)
        self.assertIn(str((a.root / 'result.json').resolve()), result.stderr)
        self.assertIn(str((b.root / 'result.json').resolve()), result.stderr)
        self.assertFalse((self.root / 'obj-export').exists())
        self.assertFalse(list(self.root.glob('*.temporary-*')))

    def test_missing_project_and_derivative_index_are_not_project_inputs(self):
        folder = self.root / '.estimation-output'
        folder.mkdir()
        (folder / 'result.json').write_text(json.dumps({'format': 'obj', 'state': 'exported'}))
        result = self.cli('--format', 'obj')
        self.assertEqual(result.returncode, 1)
        self.assertIn('no completed project', result.stderr)
        self.assertFalse((self.root / 'obj-export').exists())
        self.assertFalse(list(self.root.glob('*.temporary-*')))

    def test_stale_scratch_does_not_block_explicit_export_or_get_deleted(self):
        frozen_project(self.root / '.estimation-output', assembly())
        old = self.root / 'obj-export.temporary'
        old.mkdir()
        (old / 'diagnostic.txt').write_text('retain earlier job evidence')
        result = self.cli('.estimation-output/result.json', '--format', 'obj', '--output', './obj-export')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'obj-export' / 'EP14-part-7.mm.obj').is_file())
        self.assertEqual((old / 'diagnostic.txt').read_text(), 'retain earlier job evidence')
        self.assertFalse(list(self.root.glob('*.temporary-*')))

    def test_failed_export_cleans_empty_owned_scratch_and_can_retry(self):
        frozen_project(self.root / '.estimation-output', assembly())
        failed = self.cli('--format', 'obj', '--artifact', 'declarations', '--output', './obj-export')
        self.assertEqual(failed.returncode, 1)
        self.assertFalse((self.root / 'obj-export').exists())
        self.assertFalse(list(self.root.glob('*.temporary-*')))
        retry = self.cli('--format', 'obj', '--output', './obj-export')
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertTrue((self.root / 'obj-export' / 'EP14-part-7.mm.obj').is_file())

    def test_invalid_explicit_project_fails_before_allocating_scratch(self):
        project = frozen_project(self.root / '.estimation-output', assembly())
        project.database.write_bytes(project.database.read_bytes() + b'tamper')
        result = self.cli('.estimation-output/result.json', '--format', 'obj')
        self.assertEqual(result.returncode, 1)
        self.assertIn('hash mismatch', result.stderr)
        self.assertFalse((self.root / 'obj-export').exists())
        self.assertFalse(list(self.root.glob('*.temporary-*')))

    def export(self, payload=None, *, artifact='assembly', name='case'):
        project = frozen_project(self.root / name, assembly() if payload is None else payload, artifact=artifact)
        before = project.database.read_bytes()
        source = project.source().read_bytes()
        output = export_optional(project, self.root / f'{name}-obj', format='obj')
        self.assertEqual(project.database.read_bytes(), before)
        self.assertEqual(project.source().read_bytes(), source)
        return json.loads(output.read_text()), output.parent

    def test_obj_roundtrip_preserves_vertices_faces_volume_units_and_hashes(self):
        payload = assembly()
        before = deepcopy(payload)
        report, output = self.export(payload)
        self.assertEqual(payload, before)
        self.assertEqual(report['state'], 'exported')
        part, = report['parts']
        self.assertEqual(part['path'], 'EP14-part-7.mm.obj')
        self.assertEqual(part['units'], 'mm')
        self.assertEqual(part['source_pointer'], '/child_parts/0/solid')
        self.assertFalse(report['assembly_placement_established'])
        self.assertFalse(report['quantity_authority_granted'])
        self.assertFalse(report['fabrication_release'])
        self.assertIsNone(report['approved'])
        path = output / part['path']
        loaded = trimesh.load(path, force='mesh', process=False)
        expected = payload['child_parts'][0]['solid']
        np.testing.assert_array_equal(loaded.vertices, expected['vertices_xyz_mm'])
        np.testing.assert_array_equal(loaded.faces, expected['triangles'])
        self.assertTrue(loaded.is_watertight)
        self.assertTrue(loaded.is_winding_consistent)
        self.assertAlmostEqual(loaded.volume, 120*80*8)
        self.assertEqual(report['artifacts']['solid_1'], artifact_record(output, path))
        self.assertNotIn('999999', path.read_text())

    def test_unknown_units_preserve_coordinates_without_assuming_mm(self):
        payload = assembly()
        payload['linear_unit_evidence'] = {'state': 'unknown', 'unit': None, 'observations': []}
        solid = payload['child_parts'][0]['solid']
        solid['vertices_xyz_drawing_units'] = solid.pop('vertices_xyz_mm')
        solid['coordinate_basis'] = 'relative drawing units'
        report, output = self.export(payload)
        part, = report['parts']
        self.assertIsNone(part['units'])
        self.assertTrue(part['path'].endswith('.unitless.obj'))
        loaded = trimesh.load(output / part['path'], force='mesh', process=False)
        np.testing.assert_array_equal(loaded.vertices, solid['vertices_xyz_drawing_units'])

    def test_missing_or_uncertified_geometry_is_not_reconstructed(self):
        mutations = [
            lambda p: p['child_parts'][0].update(solid=None),
            lambda p: p['child_parts'][0].update(state='unknown'),
            lambda p: p['child_parts'][0]['reprojection'].update(passed=False),
            lambda p: p['child_parts'][0]['solid']['validation'].update(watertight=False),
            lambda p: p['linear_unit_evidence'].update(unit='in', observations=['conflict']),
            lambda p: p['linear_unit_evidence'].update(state='unknown', unit=None),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                payload = assembly()
                mutate(payload)
                report, output = self.export(payload, name=f'negative-{index}')
                self.assertEqual(report['state'], 'abstained')
                self.assertTrue(report['parts'][0]['reasons'])
                self.assertEqual(list(output.glob('*.obj')), [])

    def test_corrupt_meshes_abstain_without_repair(self):
        mutations = [
            lambda m: m['vertices_xyz_mm'][0].__setitem__(0, float('nan')),
            lambda m: m['triangles'][0].__setitem__(0, 99),
            lambda m: m['triangles'][0].__setitem__(0, -1),
            lambda m: m['triangles'][0].__setitem__(0, True),
            lambda m: m['triangles'][0].__setitem__(0, []),
            lambda m: m['triangles'].pop(),
            lambda m: m['triangles'][0].reverse(),
            lambda m: m['triangles'].append(m['triangles'][0]),
            lambda m: m['vertices_xyz_mm'].__setitem__(0, m['vertices_xyz_mm'][1]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                payload = assembly()
                mutate(payload['child_parts'][0]['solid'])
                report, output = self.export(payload, name=f'invalid-{index}')
                self.assertEqual(report['state'], 'abstained')
                self.assertTrue(report['parts'][0]['reasons'])
                self.assertEqual(list(output.glob('*.obj')), [])

    def test_partial_export_keeps_parts_separate_with_safe_unique_names(self):
        payload = assembly()
        payload['mark'] = '../../ЭП14\n'
        second = deepcopy(payload['child_parts'][0])
        second['id'] = 'part:other'
        payload['child_parts'].extend([second, {'id': 'bar:8', 'kind': 'welded_bar_family'}])
        report, output = self.export(payload)
        self.assertEqual(report['state'], 'partial')
        self.assertEqual(len(list(output.glob('*.obj'))), 2)
        paths = [p['path'] for p in report['parts'] if p['path']]
        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(all(Path(p).name == p and p.startswith('ЭП14-part-7') for p in paths))
        self.assertEqual(report['parts'][-1]['state'], 'abstained')

    def test_structural_export_ignores_overlays_and_unresolved_preview(self):
        page = {'page': 2, 'specialised_solver': {'status': 'resolved'},
                'solid_hypotheses': [{'id': 'solid.001', 'state': 'derived'}],
                'solid_preview': {'mesh': _box_mesh((300, 200, 100)),
                    'rebar_paths': [{'not': 'a solid'}],
                    'construction_region_overlays': [{'mesh': _box_mesh((1, 1, 1))}]}}
        unknown = deepcopy(page)
        unknown.update(page=3, specialised_solver={'status': 'gated_unavailable'})
        report, output = self.export({'pages': [page, unknown]}, artifact='engineering_graph')
        self.assertEqual(report['state'], 'partial')
        self.assertEqual([p.name for p in output.glob('*.obj')], ['page-2-solid.mm.obj'])
        self.assertEqual(report['parts'][1]['reasons'], ['no_resolved_structural_solid'])

    def test_presentation_only_collections_and_empty_records_are_explicit(self):
        mesh = _box_mesh((1, 2, 3))
        mesh['components'] = [{'display_offset_state': 'presentation_only_not_physical_placement'}]
        page = {'page': 1, 'specialised_solver': {'status': 'resolved'},
                'solid_hypotheses': [{'id': 'solid.001'}], 'solid_preview': {'mesh': mesh}}
        report, _ = self.export({'pages': [page]}, artifact='engineering_graph')
        self.assertEqual(report['state'], 'abstained')
        self.assertIn('presentation_only', report['parts'][0]['reasons'][0])
        report, _ = self.export({'pages': []}, artifact='engineering_graph', name='empty')
        self.assertEqual(report['reasons'], ['no_supported_solid_records'])

    def test_unsupported_artifact_tampering_and_overwrite_fail(self):
        project = frozen_project(self.root / 'source', assembly())
        for artifact in ('declarations', 'missing'):
            with self.assertRaisesRegex(ValueError, 'OBJ export requires'):
                export_optional(project, self.root / 'bad', format='obj', artifact=artifact)
        self.assertFalse((self.root / 'bad').exists())
        result = export_optional(project, self.root / 'obj', format='obj')
        before = result.read_bytes()
        with self.assertRaises(FileExistsError):
            export_optional(project, result.parent, format='obj')
        self.assertEqual(result.read_bytes(), before)
        project.source().write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            export_optional(project, self.root / 'tampered', format='obj')
        self.assertFalse((self.root / 'tampered').exists())

    def test_cli_exports_from_relocated_sqlite_without_workspace_fixtures(self):
        project = frozen_project(self.root / 'original', assembly())
        moved = self.root / 'moved'
        shutil.move(project.root, moved)
        before = (moved / 'project.sqlite').read_bytes()
        output = self.root / 'cli-obj'
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'rebar.py'), 'export',
            str(moved / 'result.json'), '--format', 'obj', '--output', str(output),
            '--reserve-gib', '0.05', '--max-growth-gib', '0.05'],
            cwd=self.root, env={**os.environ, 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1',
                'REBAR_DISK_USAGE_LOG': str(self.root / 'disk.jsonl')},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        event = json.loads(result.stdout)
        self.assertEqual(event['export_state'], 'exported')
        self.assertEqual(event['exported_file_count'], 1)
        self.assertEqual(event['omitted_part_count'], 0)
        self.assertTrue((output / 'EP14-part-7.mm.obj').is_file())
        self.assertEqual(json.loads((output / 'result.json').read_text())['state'], 'exported')
        self.assertEqual((moved / 'project.sqlite').read_bytes(), before)
        self.assertFalse(output.with_name(output.name + '.temporary').exists())


if __name__ == '__main__':
    unittest.main()
