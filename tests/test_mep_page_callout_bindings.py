from copy import deepcopy
from pathlib import Path
import json
import unittest

import fitz

from tools.generate_mep_page_callout_bindings import bind_page, connection_applicability, replay_page
from tools.render_mep_page_callout_bindings import overlay_rows
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _intersects, _native_display_geometry
from src.drawing_engine.core.vector_topology import iter_native_segments
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read
from src.drawing_engine.disciplines.mep.mep_component_evidence_classification import anchored_style_hypotheses


class PageCalloutBindingTest(unittest.TestCase):
    def test_frontier_review_is_separate_complete_sample_not_population_truth(self):
        root=Path(__file__).resolve().parents[1]
        out=root/'output/mep-interpretation-frontier-2026-09-04'
        study=read(out/'study.json');result=read(out/'source-review.json')
        review=root/'fixtures/mep/page5-frontier-source-review-2026-09-04.json'
        self.assertEqual(_file_sha256(review),result['review_input_sha256'])
        self.assertEqual({r['candidate_ref'] for r in study['sample']},
                         {r['candidate_ref'] for r in result['sample_decisions']})
        self.assertEqual({'discrete_body_or_interface_contamination':13,'visible_pipe_interval':12},result['sample_class_counts'])
        self.assertEqual(146,len(result['neutral_ranked_by_candidate_length']))
        self.assertEqual(list(range(1,147)),[r['impact_rank'] for r in result['neutral_ranked_by_candidate_length']])
        self.assertIsNone(result['population_precision'])
        self.assertIsNone(result['population_recall'])
        self.assertFalse(result['review_changes_M4'])
        self.assertFalse(result['complete_sheet_coverage'])
        arc=[r for r in result['missing_geometry_source_witnesses'] if 303293<=r['drawing_ordinal']<=303302]
        self.assertEqual(10,len(arc))
        self.assertTrue(all(not r['recovery_memberships'] for r in arc))

    def test_style_hypothesis_is_scoped_supported_and_not_an_M4_binding(self):
        key=('page','main_plan_view','full-native-style')
        profiles={ref:{'key':key,'route_role_checks_passed':True} for ref in ('a','b','route','leader','symbol','other-view','other-width','conflict')}
        relations=[{'id':ref,'state':'accepted','relation_type':'route_system',
                    'target_refs':[ref],'candidate':{'kind':'chilled_water_return'}} for ref in ('a','b')]
        rows=[{'id':ref,'state':'supported_unidentified_mep_candidate','candidate_systems':[]}
              for ref in ('route','leader','symbol','other-view','other-width','conflict')]
        profiles['leader']['route_role_checks_passed']=False
        rows[2]['state']='unclassified_view_geometry'
        profiles['other-view']['key']=('page','detail_view','full-native-style')
        profiles['other-width']['key']=('page','main_plan_view','different-native-width')
        rows[-1]['candidate_systems']=['heating_hot_water_supply']
        original=deepcopy((rows,relations,profiles))
        result=anchored_style_hypotheses(candidates=rows,local_bindings=relations,profiles=profiles)
        self.assertEqual(['route'],[r['candidate_ref'] for r in result['hypotheses'] if r['system_hypothesis']])
        self.assertTrue(all(not r['system_identity_established'] for r in result['hypotheses']))
        self.assertEqual(original,(rows,relations,profiles))
        self.assertEqual(result,anchored_style_hypotheses(candidates=rows,local_bindings=relations[::-1],profiles=profiles))
        replay=anchored_style_hypotheses(candidates=rows,local_bindings=relations,
                                       profiles=json.loads(json.dumps(profiles)))
        self.assertEqual(_sha256(result),_sha256(replay))
        rows[0]['route_candidate_support']={'transverse_attachment_negative':True}
        rejected=anchored_style_hypotheses(candidates=rows,local_bindings=relations,profiles=profiles)
        self.assertIsNone(rejected['hypotheses'][0]['system_hypothesis'])

    def test_duplicate_anchor_and_conflicting_callouts_cannot_colour_candidates(self):
        profiles={ref:{'key':('page','view','style'),'route_role_checks_passed':True} for ref in ('a','b','r')}
        relations=[{'id':ref,'state':'accepted','relation_type':'route_system','target_refs':['a'],
                    'candidate':{'kind':'chilled_water_return'}} for ref in ('x','y')]
        rows=[{'id':'r','state':'supported_unidentified_mep_candidate'}]
        result=anchored_style_hypotheses(candidates=rows,local_bindings=relations,profiles=profiles)
        self.assertEqual('insufficient_local_anchors',result['mappings'][0]['state'])
        relations[1]['target_refs']=['b'];relations[1]['candidate']['kind']='heating_hot_water_supply'
        result=anchored_style_hypotheses(candidates=rows,local_bindings=relations,profiles=profiles)
        self.assertEqual('conflicting_anchors',result['mappings'][0]['state'])
        self.assertIsNone(result['hypotheses'][0]['system_hypothesis'])

    def test_indexed_query_preserves_full_iterator_ids_geometry_and_order(self):
        with fitz.open() as pdf:
            page=pdf.new_page(width=300,height=200)
            for x in (10,100,200):
                page.draw_line((x,20),(x+10,60))
            page.draw_circle((110,40),3,color=(0,0,0),fill=(0,0,0))
            index=NativeBoundaryQueries(page,'test-page')
            for box in ([90,10,130,80],[0,0,300,200],[250,150,280,190]):
                actual,complete,_=index.query(box)
                expected=[]
                for native in iter_native_segments(index.drawings):
                    points,bounds,search=_native_display_geometry(native,page.rotation_matrix)
                    if _intersects(box,search):
                        expected.append((native,points,bounds,search))
                self.assertTrue(complete)
                self.assertEqual(expected,[(r['source_native_segment'],r['points_display'],r['bbox_display'],r['search_bbox_display']) for r in actual])

    def test_legacy_solid_is_only_hypothesis_without_current_M4(self):
        row={'id':'old','state':'identified_mep_route','outlined_composite_ref':'legacy',
             'polyline_display':[[0,0],[10,0]],'projected_path_display_points':10,
             'attributes':{'route_system':{'state':'accepted','value':'chilled_water_return'}}}
        recovery={'outlined_corridor_components':[row],'single_centreline_components':[]}
        original=deepcopy(recovery)
        solids,candidates,_=overlay_rows(recovery,{'relations':[],'outlined_route_composites':[]})
        self.assertEqual([],solids)
        self.assertEqual('supported_unidentified_mep_candidate',candidates[0]['state'])
        self.assertEqual(original,recovery)

    def test_multiport_interfaces_cannot_extend_system(self):
        branch={'id':'branch','relation_type':'projected_native_branch','state':'accepted','reasons':[]}
        result=connection_applicability({'boundary_connections':[branch]})
        self.assertEqual('abstained',result[0]['state'])
        self.assertEqual('accepted',branch['state'])


class PageCalloutBindingReplayTest(unittest.TestCase):
    def test_version_two_frontier_replays_one_new_system_and_retains_port_warnings(self):
        from tools.generate_mep_page_callout_bindings import OwnershipQueries
        root=Path(__file__).resolve().parents[1]
        run=root/'output/mep-page5-native-callouts-v2-2026-09-04'
        result=root/'output/mep-interpretation-frontier-2026-09-04'
        targets=read(run/'targets.json.gz');targets['junction_interior_version']=2
        direct,extended,_=bind_page(read(run/'terminology.json.gz'),targets,
            OwnershipQueries(read(run/'capture-index.json')['ownership']))
        self.assertEqual(_sha256(read(run/'local-M4.json.gz')),_sha256(direct))
        extended=json.loads(json.dumps(extended))
        self.assertEqual(read(result/'connection-replay/extended-M4.json.gz'),extended)
        old=read(run/'extended-M4.json.gz')
        changed=[r for r in extended['relations'] if r['state']=='accepted' and r not in old['relations']]
        self.assertEqual(['route_system'],[r['relation_type'] for r in changed])
        self.assertEqual(['mep_outlined_route_composite.631b9862053e4cd19034'],changed[0]['target_refs'])
        refs=lambda m:{ref for r in m['relations'] if r['state']=='accepted' and r['relation_type']=='route_system' for ref in r['target_refs']}
        self.assertEqual(15,len(refs(extended)))
        self.assertEqual(2,len(refs(old)-refs(extended)))

    def test_page_five_native_chains_and_current_M4_replay(self):
        root=Path(__file__).resolve().parents[1]
        self.assertEqual([],replay_page(root/'output/mep-page5-native-callouts-v2-2026-09-04'))

    def test_page_three_positive_passes_same_page_mechanism(self):
        root=Path(__file__).resolve().parents[1]
        p=json.loads((root/'output/mep-local-callout-certificate-2026-09-04/certificate-v2.json').read_text())
        targets=dict(p['targets']);targets['boundary_connections']=[p['bend']]
        direct,extended,unresolved=bind_page(p['local_terminology'],targets,
            {p['observation']['id']:p['ownership_query']})
        self.assertEqual(p['local_M4_bindings'],direct)
        self.assertEqual(p['extended_M4_bindings'],extended)
        self.assertEqual(p['nearby_unresolved_proposals'],unresolved)


class PageCalloutBindingRenderTest(unittest.TestCase):
    def test_boundary_corrected_audit_preserves_replay_and_source_geometry(self):
        from tools.render_mep_page_callout_bindings import apply_boundary_corrections
        root=Path(__file__).resolve().parents[1]
        folder=root/'output/mep-route-boundaries-2026-09-04'
        path=root/'output/pdf/mep_page5_boundary_corrected_2026-09-04.pdf'
        manifest=read(path.with_suffix('.manifest.json'))
        integration=read(folder/'replay/boundary-integration.json')
        partitions=read(folder/'partitions.json')
        bends=read(folder/'bends/recovered-bends.json.gz')
        recovery=read(root/'output/mep-page5-page-wide-recovery-v20-2026-09-03/recovery.json')
        corrected,bodies,warnings,new=apply_boundary_corrections(recovery,partitions,bends)
        bindings=read(folder/'replay/extended-M4.json.gz')
        identified,candidates,_=overlay_rows(corrected,bindings)
        self.assertEqual(15,len(identified))
        self.assertEqual({r['id'] for r in identified},set(manifest['identified_component_refs']))
        self.assertEqual({r['id'] for r in candidates},set(manifest['candidate_component_refs']))
        self.assertTrue(set(manifest['corrected_source_candidate_refs']).isdisjoint(manifest['candidate_component_refs']))
        self.assertTrue(set(integration['withdrawn_extension_refs']).isdisjoint(manifest['identified_component_refs']))
        self.assertEqual(13,len(manifest['corrected_source_candidate_refs']))
        self.assertEqual(12,len(bodies));self.assertEqual(3,len(new))
        self.assertEqual(len(warnings),manifest['body_boundary_warning_count'])
        self.assertEqual(_file_sha256(folder/'partitions.json'),manifest['boundary_partitions_sha256'])
        self.assertEqual(_file_sha256(folder/'bends/recovered-bends.json.gz'),manifest['bend_recovery_sha256'])
        self.assertEqual(integration['current_M4_sha256'],manifest['current_M4_sha256'])
        self.assertEqual(_file_sha256(path),manifest['pdf_sha256'])
        self.assertIsNone(integration['installed_length']);self.assertIsNone(integration['purchase_length'])
        self.assertEqual([],manifest['failed_labels'])
        with fitz.open(path) as pdf:
            self.assertEqual(1,len(pdf))
            self.assertEqual(1,len(pdf[0].get_images()))
            pix=fitz.Pixmap(pdf,pdf[0].get_images()[0][0])
            self.assertEqual(3,pix.colorspace.n)
            self.assertIn('BEND ? 0.33m',pdf[0].get_text())
            self.assertIn('BODY / SYMBOL ?',pdf[0].get_text())

    def test_colour_audit_preserves_source_RGB_and_all_current_identities(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'output/pdf/mep_page5_native_callout_bindings_colour_2026-09-04.pdf'
        manifest=read(path.with_suffix('.manifest.json'))
        previous=read(root/'output/pdf/mep_page5_native_callout_bindings_2026-09-04.manifest.json')
        self.assertEqual(previous['identified_component_refs'],manifest['identified_component_refs'])
        self.assertEqual(previous['candidate_component_refs'],manifest['candidate_component_refs'])
        self.assertEqual(previous['current_M4_sha256'],manifest['current_M4_sha256'])
        self.assertEqual(_file_sha256(path),manifest['pdf_sha256'])
        evidence=read(Path(manifest['style_hypotheses_path']))
        self.assertEqual(_file_sha256(Path(manifest['style_hypotheses_path'])),manifest['style_hypotheses_sha256'])
        self.assertGreater(manifest['colour_style_supported_candidates'],0)
        self.assertTrue(all(not r['system_identity_established'] for r in evidence['hypotheses']))
        recovery=read(root/'output/mep-page5-page-wide-recovery-v20-2026-09-03/recovery.json')
        bindings=read(Path(manifest['run'])/'extended-M4.json.gz')
        direct=read(Path(manifest['run'])/'local-M4.json.gz')
        _,candidates,_=overlay_rows(recovery,bindings)
        profiles={r['candidate_ref']:r['source_profile'] for r in evidence['hypotheses']}
        for mapping in evidence['mappings']:
            profiles.update(mapping['anchor_source_profiles'])
        replay=anchored_style_hypotheses(candidates=candidates,local_bindings=direct['relations'],profiles=profiles)
        self.assertEqual(_sha256(replay),_sha256({k:v for k,v in evidence.items() if k!='inputs'}))
        self.assertEqual([],manifest['failed_labels'])
        with fitz.open(path) as pdf:
            self.assertEqual(1,len(pdf))
            images=pdf[0].get_images()
            self.assertEqual(1,len(images))
            pix=fitz.Pixmap(pdf,images[0][0]); data=pix.samples
            # MuPDF may embed RGB through an ICCBased colour profile.
            self.assertEqual(3,pix.colorspace.n)
            self.assertEqual(3,pix.n)
            self.assertTrue(any(data[i]!=data[i+1] or data[i]!=data[i+2] for i in range(0,len(data),3)))
            self.assertIn('Colour/style-supported hypothesis',pdf[0].get_text())

    def test_page_five_is_raster_backed_with_current_solid_bindings_only(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'output/pdf/mep_page5_native_callout_bindings_2026-09-04.pdf'
        manifest=read(path.with_suffix('.manifest.json'))
        bindings=read(Path(manifest['run'])/'extended-M4.json.gz')
        expected={t for r in bindings['relations'] if r['state']=='accepted'
                  and r['relation_type']=='route_system' for t in r['target_refs']}
        self.assertEqual(expected,set(manifest['identified_component_refs']))
        self.assertEqual(_file_sha256(path),manifest['pdf_sha256'])
        self.assertEqual([],manifest['failed_labels'])
        self.assertFalse(manifest['complete_page_route_coverage'])
        self.assertIsNone(manifest['wrong_accepts'])
        with fitz.open(path) as pdf:
            self.assertEqual(1,len(pdf))
            self.assertEqual(1,len(pdf[0].get_images()))
            self.assertGreater(len(pdf[0].get_drawings()),len(expected))
            text=pdf[0].get_text()
            self.assertIn('Connection / scope warning',text)
            self.assertNotIn('purchase',text.lower())
