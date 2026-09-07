import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore
from src.drawing_engine.project.project_store_pointer import read_pointer, rollback_pointer, switch_pointer


class ProjectStorePointerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pointer = self.root/'project-store-active.json'
        for name,revision in (('v1.sqlite',1),('v3.sqlite',3)):
            with ProjectKnowledgeStore(self.root/name) as store:
                store.import_snapshot(project_id='project',document_id='document',
                    source_sha256=str(revision)*64,context={'revision':revision},
                    artifacts={'artifact':{'schema_version':'1','layer':'test',
                               'records':[{'id':f'record-{revision}','record_type':'test'}]}})

    def test_atomic_switch_and_rollback(self):
        first = switch_pointer(self.pointer,active='v3.sqlite',rollback='v1.sqlite',
                               project_id='project',document_id='document')
        self.assertEqual(first['active'],'v3.sqlite')
        second = rollback_pointer(self.pointer,project_id='project',document_id='document')
        self.assertEqual(second['active'],'v1.sqlite')
        self.assertEqual(second['rollback'],'v3.sqlite')
        self.assertEqual(second['generation'],2)

    def test_failed_replace_preserves_previous_pointer_and_unsafe_names_fail(self):
        switch_pointer(self.pointer,active='v1.sqlite',rollback='v3.sqlite',
                       project_id='project',document_id='document')
        before = self.pointer.read_bytes()
        with patch('src.drawing_engine.project.project_store_pointer.os.replace',side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError,'interrupted'):
                switch_pointer(self.pointer,active='v3.sqlite',rollback='v1.sqlite',
                               project_id='project',document_id='document')
        self.assertEqual(self.pointer.read_bytes(),before)
        self.assertEqual(read_pointer(self.pointer)['active'],'v1.sqlite')
        with self.assertRaisesRegex(ValueError,'filename'):
            switch_pointer(self.pointer,active='../v3.sqlite',rollback='v1.sqlite',
                           project_id='project',document_id='document')


if __name__ == '__main__': unittest.main()
