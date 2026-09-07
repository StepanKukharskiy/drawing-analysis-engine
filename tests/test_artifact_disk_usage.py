import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.drawing_engine.pipelines.generate_mep_stroke_ownership import write
from src.drawing_engine.operations.run_artifact_job import run
from src.drawing_engine.operations.artifact_disk_usage import (DiskBudgetError, artifact_disk_usage,
                                     check_capacity, measure_paths)


class ArtifactDiskUsageTest(unittest.TestCase):
    def test_scoped_sizes_deduplicate_roots_and_hardlinks_skip_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root/'outputs'; directory.mkdir()
            file = directory/'one'; file.write_bytes(b'abc')
            os.link(file, directory/'hardlink')
            (directory/'cycle').symlink_to(directory, target_is_directory=True)
            (directory/'external').symlink_to(root/'unrelated')
            (root/'unrelated').write_bytes(b'x'*1000)
            result = measure_paths([directory, file, directory/'missing'])
            self.assertEqual(result['logical_bytes'], 3)
            self.assertEqual(result['allocated_bytes'], file.stat().st_blocks*512)
            self.assertEqual(result['file_count'], 1)
            self.assertEqual(result['symlinks_skipped'], 2)
            self.assertTrue(result['complete'])

    def test_failure_records_partial_outputs_and_temporary_files_without_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'out'; output.write_bytes(b'old')
            staging=root/'staging'; staging.mkdir(); ledger=root/'ledger.jsonl'
            with self.assertRaisesRegex(ValueError, 'producer failed'):
                with artifact_disk_usage('failure', [output], temporary_paths=[staging],
                                         ledger=ledger, min_free_bytes=0):
                    output.write_bytes(b'partial')
                    (staging/'retained').write_bytes(b'temporary')
                    raise ValueError('producer failed')
            records=[json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(records[-1]['status'], 'failed')
            self.assertEqual(records[-1]['delta']['logical_bytes'], 4)
            self.assertEqual(records[-1]['temporary_after']['logical_bytes'], 9)
            self.assertEqual(output.read_bytes(), b'partial')
            self.assertFalse(records[-1]['automatic_cleanup'])
            self.assertFalse(records[-1]['engineering_authority'])
            self.assertIn('free_bytes', next(iter(records[-1]['filesystems_after'].values())))

    def test_floor_and_reserve_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('src.drawing_engine.operations.artifact_disk_usage.free_space', return_value={
                    'device': {'probe_path': tmp, 'free_bytes': 120, 'total_bytes': 200}}):
                check_capacity([tmp], min_free_bytes=100, reserve_bytes=20)
                with self.assertRaises(DiskBudgetError):
                    check_capacity([tmp], min_free_bytes=100, reserve_bytes=21)

    def test_low_space_prevents_write_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); file=root/'output'; ledger=root/'ledger.jsonl'
            with patch('src.drawing_engine.operations.artifact_disk_usage.check_capacity', side_effect=DiskBudgetError('low')):
                with self.assertRaises(DiskBudgetError):
                    with artifact_disk_usage('blocked', [file], ledger=ledger):
                        file.write_bytes(b'not written')
            self.assertFalse(file.exists())
            self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['status'], 'failed')

    def test_json_and_gzip_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); payload={'z': 2, 'a': ['native-ref']}
            expected=(json.dumps(payload, sort_keys=True, separators=(',', ':'))+'\n').encode()
            with patch.dict(os.environ, {'REBAR_DISK_USAGE_LOG': str(root/'ledger.jsonl')}):
                write(root/'value.json', payload)
                write(root/'value.json.gz', payload)
            self.assertEqual((root/'value.json').read_bytes(), expected)
            self.assertEqual((root/'value.json.gz').read_bytes(), gzip.compress(expected, mtime=0))
            self.assertEqual(len((root/'ledger.jsonl').read_text().splitlines()), 4)

    def test_operational_ledger_is_excluded_from_output_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/'ledger.jsonl'
            with artifact_disk_usage('empty', [root], ledger=ledger, min_free_bytes=0):
                pass
            result=json.loads(ledger.read_text().splitlines()[-1])
            self.assertEqual(result['delta']['logical_bytes'], 0)

    def test_overlapping_output_and_staging_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                with artifact_disk_usage('bad', [tmp], temporary_paths=[Path(tmp)/'staging']):
                    self.fail('should not run')

    def test_output_symlink_cannot_hide_an_unmeasured_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); target=root/'original'; target.write_bytes(b'preserved')
            alias=root/'alias'; alias.symlink_to(target)
            with self.assertRaisesRegex(DiskBudgetError, 'must not be symlinks'):
                with artifact_disk_usage('alias', [alias], ledger=root/'ledger.jsonl'):
                    alias.write_bytes(b'changed')
            self.assertEqual(target.read_bytes(), b'preserved')

    def test_wrapper_counts_arbitrary_binary_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'artifact'; ledger=root/'ledger.jsonl'
            run([sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"artifact")', str(output)],
                outputs=[output], ledger=ledger, min_free_bytes=0)
            result=json.loads(ledger.read_text().splitlines()[-1])
            self.assertEqual(result['after']['logical_bytes'], 8)
            self.assertEqual(result['status'], 'completed')

    def test_wrapper_failure_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'artifact'; ledger=root/'ledger.jsonl'
            with self.assertRaises(subprocess.CalledProcessError):
                run([sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"partial"); sys.exit(2)', str(output)],
                    outputs=[output], ledger=ledger, min_free_bytes=0)
            self.assertEqual(output.read_bytes(), b'partial')
            self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['status'], 'failed')

    def test_wrapper_enforces_growth_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'artifact'; ledger=root/'ledger.jsonl'
            with self.assertRaisesRegex(DiskBudgetError, 'growth budget'):
                run([sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"x"*65536)', str(output)],
                    outputs=[output], ledger=ledger, min_free_bytes=0, max_growth_bytes=0)
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['status'], 'failed')

    def test_wrapper_stops_owned_job_when_free_space_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/'ledger.jsonl'
            with patch('src.drawing_engine.operations.run_artifact_job.check_capacity', side_effect=DiskBudgetError('space dropped')):
                with self.assertRaisesRegex(DiskBudgetError, 'space dropped'):
                    run([sys.executable, '-c', 'import time; time.sleep(30)'],
                        outputs=[root/'output'], ledger=ledger, min_free_bytes=0,
                        poll_seconds=.1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['status'], 'failed')

    def test_missing_required_artifact_is_a_failed_stage_not_empty_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/'ledger.jsonl'; output=root/'missing'
            with self.assertRaisesRegex(ValueError, 'required artifacts'):
                run([sys.executable, '-c', 'pass'], outputs=[output],
                    required_outputs=[output], ledger=ledger, min_free_bytes=0)
            self.assertEqual(json.loads(ledger.read_text().splitlines()[-1])['status'], 'failed')

    def test_v3_monitor_keeps_canonical_bytes_and_includes_sidecars(self):
        from src.drawing_engine.project.project_packed_store import PackedProjectStore
        from src.drawing_engine.project.project_v3_direct_writer import V3DirectArtifactWriter
        from src.drawing_engine.project.project_knowledge_store import _json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); database=root/'package.sqlite'; ledger=root/'ledger.jsonl'
            payload={'schema_version': '1', 'layer': 'test', 'rows': [{'id': 'native.1'}]}
            with patch.dict(os.environ, {'REBAR_DISK_USAGE_LOG': str(ledger)}):
                with PackedProjectStore(database, create=True) as store:
                    writer=V3DirectArtifactWriter(store)
                    result=writer.write('mep-automatic-page-cache', payload)
                    self.assertEqual(b''.join(writer.iter_bytes(result['artifact_key'])), _json(payload).encode())
            record=json.loads(ledger.read_text().splitlines()[-1])
            self.assertIn(str(database)+'-wal', record['output_paths'])
            self.assertGreater(record['after']['logical_bytes'], 0)
