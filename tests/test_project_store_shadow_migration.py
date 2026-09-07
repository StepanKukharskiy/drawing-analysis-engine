import copy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.migrate_project_store_v2 import ShadowMigration
from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore


def legacy_database(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript("""
        CREATE TABLE snapshots (id TEXT PRIMARY KEY,project_id TEXT NOT NULL,document_id TEXT NOT NULL,
          source_sha256 TEXT NOT NULL,context_json TEXT NOT NULL,manifest_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE active_documents (project_id TEXT NOT NULL,document_id TEXT NOT NULL,
          snapshot_id TEXT NOT NULL REFERENCES snapshots(id),PRIMARY KEY(project_id,document_id));
        CREATE TABLE artifacts (sha256 TEXT PRIMARY KEY,payload_zlib BLOB NOT NULL);
        CREATE TABLE snapshot_artifacts (snapshot_id TEXT NOT NULL REFERENCES snapshots(id),name TEXT NOT NULL,
          artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),PRIMARY KEY(snapshot_id,name));
        CREATE TABLE nodes (id TEXT PRIMARY KEY,artifact_sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
          pointer TEXT NOT NULL,source_id TEXT,record_type TEXT NOT NULL,page_ref TEXT,state TEXT,payload_json TEXT,
          UNIQUE(artifact_sha256,pointer));
        CREATE INDEX nodes_source ON nodes(source_id,page_ref);
        CREATE INDEX nodes_kind ON nodes(record_type,state,page_ref);
        CREATE TABLE edges (node_id TEXT NOT NULL REFERENCES nodes(id),field TEXT NOT NULL,target_ref TEXT NOT NULL,
          PRIMARY KEY(node_id,field,target_ref));
        CREATE INDEX edges_target ON edges(target_ref);
        CREATE TABLE reviews (id TEXT PRIMARY KEY,snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
          node_id TEXT NOT NULL REFERENCES nodes(id),payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        PRAGMA user_version=1;
        """)


class ProjectStoreShadowMigrationTest(unittest.TestCase):
    def test_active_first_then_history_proves_chunk_query_neighbor_export_review_and_fts_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'legacy.sqlite'
            shadow = Path(directory) / 'shadow.sqlite'
            legacy_database(source)
            scope = {'project_id':'project','document_id':'document'}
            artifact = {'schema_version':'0.1.0','layer':'mep_route_observation_graph',
                'document':{'source_pdf_sha256':'a'*64},'pages':[{'id':'page-record','page_ref':'page.1',
                'record_type':'page_route_observations','observations':[
                    {'id':'route.1','record_type':'route_fragment','page_ref':'page.1','state':'observed',
                     'text':'AHU supply','target_ref':'annotation.1'},
                    {'id':'annotation.1','record_type':'text','page_ref':'page.1','text':'AHU-1'}]}]}
            inputs = {**scope,'source_sha256':'a'*64,'context':{'version':1},
                      'artifacts':{'route-observations':artifact}}
            with ProjectKnowledgeStore(source) as store:
                historical = store.import_snapshot(**inputs)
                node = store.query(**scope,source_id='route.1')[0]
                store.add_review(**scope,snapshot_id=historical,node_id=node['id'],
                                 review={'reviewer':'engineer','decision':'retain_observation'})
                changed = copy.deepcopy(inputs)
                changed['context']['version'] = 2
                changed['artifacts']['route-observations']['pages'][0]['observations'][0]['state']='abstained'
                active = store.import_snapshot(**changed)
            events=[]
            with ShadowMigration(source,shadow,progress=events.append) as migration:
                migration.migrate('active')
                active_report=migration.verify('active')
                self.assertTrue(active_report['all_parity'])
                self.assertFalse(active_report['cutover_ready'])
                with ProjectKnowledgeStore(shadow) as store:
                    row=store.query(**scope,source_id='route.1')[0]
                    self.assertIsNone(row['payload'])
                    self.assertEqual(store.node_payload(**scope,node_id=row['id'])['state'],'abstained')
                    neighbors=store.neighbors(**scope,node_id=row['id'])
                    self.assertEqual(next(item for item in neighbors if item['target_ref']=='annotation.1')['resolution'],'unique_record')
                    self.assertEqual(store.artifact(**scope,name='route-observations'),changed['artifacts']['route-observations'])
                    search=store.search(**scope,text='AHU',limit=10)
                    self.assertTrue(any(item['source_id']=='route.1' for item in search))
                    self.assertTrue(all(item['retrieval_authority']=='proposal_only' for item in search))
                migration.migrate('history')
                estimate=migration.estimate()
                self.assertEqual(estimate['remaining_counts'],{'artifacts':0,'nodes':0,'edges':0})
                self.assertFalse(estimate['estimated_target_passed'])
                self.assertEqual(estimate['projection_indexes']['maximum_bytes'],2 * 1024 ** 3)
                self.assertEqual(estimate['capacity_now']['minimum_transaction_free_bytes'],6 * 1024 ** 3)
                report=migration.verify('all')
                self.assertTrue(report['correctness_gate_passed'])
                self.assertFalse(report['cutover_ready'])
                self.assertFalse(report['storage_efficiency']['whole_store']['cutover_gate_passed'])
                with self.assertRaisesRegex(ValueError,'whole-store efficiency'):
                    migration.cutover(Path(directory) / 'backup.sqlite')
                self.assertEqual(report['vector_retrieval']['status'],'not_implemented')
            with ProjectKnowledgeStore(shadow) as store:
                self.assertEqual(len(store.reviews(**scope,snapshot_id=historical)),1)
                self.assertEqual(store.snapshot(**scope)['id'],active)


if __name__ == '__main__':
    unittest.main()
