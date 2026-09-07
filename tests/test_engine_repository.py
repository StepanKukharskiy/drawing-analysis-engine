import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from deploy.source.engine_repository import check_index
from deploy.source.stage_source import stage

ROOT = Path(__file__).resolve().parents[1]


class EngineRepositoryTest(unittest.TestCase):
    def test_stage_has_all_engine_files_and_no_workspace_owners(self):
        manifest = ROOT / 'deploy/source/engine-manifest.json'
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp).resolve() / 'engine'
            report = stage(ROOT, destination, manifest, profile='engine')
            selected = {r['path'] for r in report['files']}
            engine = {str(p.relative_to(ROOT)) for p in (ROOT / 'src/drawing_engine').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
            self.assertTrue(engine <= selected, sorted(engine - selected))
            self.assertFalse(any(p.startswith(('apps/', 'research/', 'fixtures/', 'data/', 'archive/')) for p in selected))
            self.assertFalse((destination / 'deploy/source/private-fixtures.json').exists())
            self.assertTrue((destination / 'docs/estimation-cli.md').is_file())

    def test_index_requires_exact_files_bytes_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'deploy/source').mkdir(parents=True)
            (root / 'README.md').write_text('engine')
            (root / 'deploy/source/engine-manifest.json').write_text(json.dumps({
                'schema_version': 1, 'groups': {'runtime': ['README.md', 'deploy/source/engine-manifest.json']}}))
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
            git('init', '--quiet')
            git('add', '.')
            self.assertEqual(check_index(root)['index_files'], 2)
            (root / 'README.md').chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'bytes/mode'):
                check_index(root)
            (root / 'README.md').chmod(0o644)
            (root / 'README.md').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'bytes/mode'):
                check_index(root)
            git('add', '.')
            (root / 'extra.pdf').write_bytes(b'private')
            with self.assertRaisesRegex(ValueError, 'unreviewed untracked'):
                check_index(root)
            git('add', '.')
            with self.assertRaisesRegex(ValueError, 'index differs'):
                check_index(root)

    def test_ignore_rules_cover_both_unicode_modes_and_nested_private_pdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / '.gitignore').write_bytes((ROOT / '.gitignore').read_bytes())
            for directory in ('Пример_чертежей_для_декомпозиции', 'Пример_чертежей_для_декомпозиции',
                              'audit_examples', 'docs/content-series', 'review-app'):
                folder = root / directory
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'private.pdf').write_bytes(b'private')
            subprocess.run(['git', '-C', str(root), 'init', '--quiet'], check=True)
            for mode in ('true', 'false'):
                result = subprocess.check_output(['git', '-C', str(root), '-c', 'core.precomposeUnicode='+mode,
                                                  'ls-files', '--others', '--exclude-standard', '-z'])
                self.assertEqual(result, b'.gitignore\0')
