from pathlib import Path
import unittest

from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style


class NativeSubpathReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        cls.shadow=cls.root/'output/mep-gemini-boundary-shadow-v2-2026-09-04'
        cls.cases={};needed=set()
        for n in (1,2,3,16):
            folder=cls.shadow/f'case-{n:02d}'
            q=read(folder/'native-query.json.gz');baseline=read(folder/'baseline.json')['partition']
            cls.cases[n]=(q,baseline)
            needed.update(r['source_native_segment']['drawing_ref'] for r in q['source_rows'])
        info=read(cls.root/'output/mep-source-denominator-page5-v2-2026-09-03/manifest.json')['native_authored_path_pack']
        pack=NativePathPack(info['path'],info)
        if not pack.verify_hash():raise ValueError('native path pack changed')
        cls.context={}
        for p in pack.records():
            ref=f"drawing[{p['drawing_ordinal']}]"
            if ref in needed:cls.context[ref]={k:p[k] for k in ('drawing_ordinal','source_segment_count')}

    def replay(self,n):
        q,b=self.cases[n];refs=set(b['native_member_source_refs'])
        native=next(r for r in q['source_rows'] if r['source_primitive_ref'] in refs)
        paths={r['source_native_segment']['drawing_ref']:self.context[r['source_native_segment']['drawing_ref']]
               for r in q['source_rows'] if r['source_native_segment']['drawing_ref'] in self.context}
        return partition_corridor(b['original_candidate'],q,refs,_normalise_style(native['source_native_segment']['style']),authored_paths=paths)

    def test_six_microsegments_recover_known_false_exclusion_without_identity(self):
        r=self.replay(2)
        expected={f'drawing[{n}].item[0].segment[0]' for n in range(1886471,1886477)}
        c=next(c for c in r['crossing_path_certificates'] if 'drawing[1886473].item[0].segment[0]' in c.get('source_primitive_refs',[]))
        self.assertTrue(expected<=set(c['supporting_chain_source_refs']))
        self.assertTrue(all(j['exact_seam_residual']==0 for j in c['joins']))
        old=self.cases[2][1]
        gain=sum(i['projected_path_display_points'] for i in r['retained_intervals'])-sum(i['projected_path_display_points'] for i in old['retained_intervals'])
        self.assertAlmostEqual(6.546571192214575,gain)
        self.assertFalse(r['system_identity_established']);self.assertFalse(r['physical_continuity_established'])

    def test_other_two_false_exclusions_remain_explicit_not_false_passes(self):
        for n in (1,3):
            r=self.replay(n);q,b=self.cases[n]
            self.assertEqual([i['parameter_interval'] for i in b['body_intervals']],
                             [i['parameter_interval'] for i in r['body_intervals']])
            self.assertEqual(len(q['source_rows']),r['native_subpath_inventory']['accounted_source_primitive_count'])

    def test_gemini_false_symbol_retention_is_still_blocked(self):
        r=self.replay(16);old=self.cases[16][1]
        self.assertEqual([i['parameter_interval'] for i in old['body_intervals']],
                         [i['parameter_interval'] for i in r['body_intervals']])
        self.assertTrue(r['body_intervals']);self.assertIsNone(r['installed_length'])

    def test_all_373_inputs_and_protected_artifacts_preserved(self):
        p=read(self.root/'output/mep-native-subpaths-v3-2026-09-04/replay.json')
        self.assertEqual(373,len(p['records']));self.assertEqual(373,len({r['candidate_ref'] for r in p['records']}))
        self.assertEqual(0,p['counts']['new_exclusions_candidates'])
        self.assertFalse(p['boundary_changes_published']);self.assertEqual(0,p['new_identity_accepts'])
        for row in p['records']:
            self.assertEqual(row['partition_sha256'],_file_sha256(Path(row['partition_path'])))
            if row['accounted_source_primitive_count'] is not None:
                self.assertEqual(row['source_primitive_count'],row['accounted_source_primitive_count'])
        frozen=read(self.shadow/'protocol.json')
        for field in ('M4','audit_pdf'):
            self.assertEqual(frozen[field+'_sha256'],_file_sha256(Path(frozen[field+'_path'])))

    def test_four_invalid_model_responses_never_enter_certificate_replay(self):
        responses=[read(p) for p in self.shadow.glob('case-*/shadow.json')]
        invalid=[r for r in responses if r.get('state')=='invalid_response']
        self.assertEqual(15,len(responses));self.assertEqual(4,len(invalid))
        self.assertTrue(all('native_replay' not in r and 'proposal' not in r for r in invalid))
        # Invalid JSON is a pipeline failure, not a semantic abstention.
        summary=read(self.shadow/'results.json')
        self.assertEqual(4,sum(summary[k]['counts']['invalid_responses'] for k in ('development','evaluation')))

    def test_frozen_source_evaluation_keeps_misses_and_endpoint_gap_explicit(self):
        from types import SimpleNamespace
        from tempfile import TemporaryDirectory
        from src.drawing_engine.pipelines.generate_mep_stroke_ownership import write
        from tools.review_mep_native_subpath_changes import frozen_evaluation
        folder=self.root/'output/mep-native-subpaths-v3-2026-09-04'
        replay=read(folder/'replay.json');expected=read(folder/'frozen-evaluation.json')
        with TemporaryDirectory() as directory:
            output=Path(directory);write(output/'replay.json',replay)
            args=SimpleNamespace(output=output,shadow=self.shadow,
                frozen_evaluation_review=self.root/'fixtures/mep/page5-gemini-shadow-review-2026-09-04.json')
            frozen_evaluation(args,replay)
            self.assertEqual(expected,read(output/'frozen-evaluation.json'))
        self.assertEqual(8,expected['counts']['reviewed_cases'])
        self.assertEqual(2,expected['counts']['remaining_false_exclusions'])
        self.assertEqual(0,expected['counts']['retained_contamination'])
        self.assertIsNone(expected['genuine_endpoint_recall'])
        self.assertFalse(expected['model_answers_consumed'])
        self.assertFalse(expected['publication_gate_passed'])
