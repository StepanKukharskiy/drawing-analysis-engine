from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from test_mep_automatic_target_binding import automatic_fixture
from tools.mep_project import build
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import build_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore


class MepBindingOutcomesTest(unittest.TestCase):
    def test_stale_and_unearned_outcomes_cannot_replace_active_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, terms, targets, _, _, bindings, catalog, _ = automatic_fixture(directory)
            runs = build_mep_cross_sheet_runs(sheet_registry=registry,
                route_graph=targets['route_graph'], attribute_bindings=bindings)
            payloads = {'sheet-registry': registry, 'route-observations': targets['route_graph'],
                'terminology-proposals': terms, 'attribute-bindings': bindings,
                'cross-sheet-runs': runs, 'item-catalog': catalog}
            relation = next(r for r in bindings['relations'] if r['state'] == 'accepted')
            outcome = {'id': 'binding-outcome', 'record_type': 'mep_bounded_review_outcome',
                'page_ref': 'page.1', 'subject_kind': 'annotation_binding', 'state': 'accepted',
                'relation_type': relation['relation_type'], 'accepted_relation_refs': [relation['id']],
                'subject_refs': relation['target_refs'], 'expected_candidate': relation['candidate'],
                'evaluation_only': True, 'quantity_eligible': False,
                'physical_continuation_established': False, 'physical_item_identity_established': False}
            review = {'schema_version': '0.1.0', 'layer': 'mep_bounded_review_outcomes', 'document': registry['document'],
                'input_payload_sha256': {'attribute-bindings': _sha256(bindings)},
                'outcomes': [outcome], 'quantity_eligible': False}
            for name, payload in {**payloads, 'review-outcomes': review}.items():
                (root / (name + '.json')).write_text(json.dumps(payload))
            args = dict(source=root/'neutral.pdf', registry_path=root/'sheet-registry.json', run_dir=root,
                output_dir=root/'project-output', database=root/'project.sqlite', project_id='test', document_id='drawing')
            original = build(**args)['snapshot_id']
            mutations = []
            stale = deepcopy(review)
            stale['input_payload_sha256']['attribute-bindings'] = '0' * 64
            mutations.append(stale)
            for field, value in (('accepted_relation_refs', ['unknown']),
                                 ('expected_candidate', {'kind': 'invented'}),
                                 ('subject_refs', ['unrelated']), ('page_ref', 'other-page'),
                                 ('subject_kind', 'equipment'), ('subject_kind', 'connection'),
                                 ('engineer_approved', True),
                                 ('physical_item_identity_established', True)):
                changed = deepcopy(review)
                changed['outcomes'][0][field] = value
                mutations.append(changed)
            for changed in mutations:
                with self.subTest(changed=changed['outcomes'][0]):
                    (root/'review-outcomes.json').write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        build(**args)
                    with ProjectKnowledgeStore(args['database']) as store:
                        self.assertEqual(store.snapshot(project_id='test', document_id='drawing')['id'], original)
                        self.assertEqual(store.artifact(project_id='test', document_id='drawing',
                            name='item-catalog'), catalog)


if __name__ == '__main__':
    unittest.main()
