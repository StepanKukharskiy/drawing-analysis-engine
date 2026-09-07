from copy import deepcopy
import gzip
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import (certify_trace, transverse_cubic_crossing,
    certify_junction_interior, connected_trace_scopes, bounded_inline_detail_components,
    bound_annotation_stroke_roles)


def update_source_hashes(query):
    query['search']['all_source_refs_sha256'] = _sha256(sorted(r['source_primitive_ref'] for r in query['source_rows']))
    query['search']['source_rows_sha256'] = _sha256(query['source_rows'])


class MepProjectedTraceCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination/projected-trace-completion-replay.json.gz'
        cls.fixture = json.loads(gzip.decompress(path.read_bytes()))
        cls.positive = next(r for r in cls.fixture['expected_completion']['scoped_traces'] if r['projected_scope_complete'])
        cls.scope = cls.positive['scope']
        cls.query = next(q for q in cls.fixture['source_queries'] if q['id'] == cls.positive['source_query_ref'])

    def altered_line(self, ref, points, kind='line'):
        source = deepcopy(next(r for r in self.query['source_rows'] if r['source_native_segment']['kind'] == 'line'))
        source.update(source_primitive_ref=ref, points_display=points,
            bbox_display=[min(p[i] for p in points) for i in (0, 1)] + [max(p[i] for p in points) for i in (0, 1)])
        source['search_bbox_display'] = source['bbox_display']
        source['source_native_segment']['kind'] = kind
        return source

    def test_real_complete_trace_has_two_certified_interfaces_and_no_physical_authority(self):
        before = deepcopy((self.scope, self.query))
        result = certify_trace(self.scope, self.query)
        self.assertEqual(result, self.positive)
        self.assertEqual(len(result['endpoint_classifications']), 2)
        self.assertTrue(result['coverage']['complete_source_query'])
        self.assertTrue(all(r['classification'] == 'certified_scope_interface' for r in result['endpoint_classifications']))
        self.assertTrue(result['closed_native_line_motifs'])
        self.assertTrue(all(not m['additional_open_port_support_refs'] for m in result['closed_native_line_motifs']))
        self.assertEqual(result['epistemic_state'], 'inferred')
        self.assertFalse(result['physical_continuation_established'])
        self.assertFalse(result['physical_run_complete'])
        self.assertFalse(result['quantity_eligible'])
        self.assertEqual((self.scope, self.query), before)

    def test_real_trace_loses_completion_when_a_native_sidewall_edge_is_missing(self):
        query = deepcopy(self.query)
        missing = set(self.positive['native_sidewall_coverage'][0]['source_primitive_refs'])
        query['source_rows'] = [r for r in query['source_rows'] if r['source_primitive_ref'] not in missing]
        update_source_hashes(query)
        result = certify_trace(self.scope, query)
        self.assertFalse(result['projected_scope_complete'])
        self.assertIn('missing_native_sidewall_coverage', result['reasons'])

    def test_hidden_open_branch_and_through_body_competitor_are_not_glyph_details(self):
        ports = self.scope['ports']
        center = [(a + b) / 2 for a, b in zip(ports[0]['center'], ports[1]['center'])]
        width = ports[0]['width']
        for role, points in [('hidden_branch', [center, [center[0], center[1] - 10 * width]]),
                             ('through_body', [[center[0] - 5 * width, center[1] + width / 8],
                                               [center[0] + 5 * width, center[1] + width / 8]])]:
            query = deepcopy(self.query)
            query['source_rows'].append(self.altered_line(role, points))
            update_source_hashes(query)
            result = certify_trace(self.scope, query)
            self.assertFalse(result['projected_scope_complete'], role)
            self.assertIn('unresolved_incident_endpoint_or_interior_feature', result['reasons'])

    def test_unexplained_endpoint_and_competing_destination_prevent_completion(self):
        scope = deepcopy(self.scope)
        scope['interfaces'].pop()
        result = certify_trace(scope, self.query)
        self.assertFalse(result['projected_scope_complete'])
        self.assertIn('unexplained_endpoint_or_missing_certified_scope_interface', result['reasons'])
        scope = deepcopy(self.scope)
        scope['interfaces'][0]['unresolved_competing_candidate_refs'] = ['unresolved_native_port_destination']
        result = certify_trace(scope, self.query)
        self.assertFalse(result['projected_scope_complete'])
        self.assertIn('unresolved_competing_native_interface_destination', result['reasons'])

    def test_cropped_competitor_missing_source_and_exhausted_search_cannot_close_scope(self):
        variants = []
        query = deepcopy(self.query)
        query['search']['bbox_display'][0] += 1
        variants.append((query, 'complete_corridor_and_competitor_margin_not_queried'))
        query = deepcopy(self.query)
        query['source_rows'].pop()
        variants.append((query, 'native_query_source_inventory_changed'))
        query = deepcopy(self.query)
        query['search']['budget_exhausted'] = True
        variants.append((query, 'native_source_search_incomplete_or_exhausted'))
        query = deepcopy(self.query)
        query['search']['classification_pair_budget'] = 1
        variants.append((query, 'native_straight_support_work_budget_exhausted'))
        for query, reason in variants:
            result = certify_trace(self.scope, query)
            self.assertFalse(result['projected_scope_complete'])
            self.assertIn(reason, result['reasons'])

    def test_unsupported_curve_remains_a_blocker_even_inside_a_closed_body(self):
        ports = self.scope['ports']
        center = [(a + b) / 2 for a, b in zip(ports[0]['center'], ports[1]['center'])]
        query = deepcopy(self.query)
        query['source_rows'].append(self.altered_line('unsupported_curve', [center,
            [center[0] + .1, center[1] + .1], [center[0] + .2, center[1]]], 'cubic'))
        update_source_hashes(query)
        result = certify_trace(self.scope, query)
        self.assertFalse(result['projected_scope_complete'])
        self.assertIn('unsupported_native_class_inside_trace', result['reasons'])

    def test_native_curve_crossing_uses_control_monotonicity_not_samples(self):
        source = {'source_primitive_ref': 'curve', 'source_native_segment': {
            'kind': 'cubic', 'start_display': [4,-4], 'end_display': [6,4],
            'control_points_display': [[4,-1],[6,1]], 'style': {'stroke': [0,0,0], 'fill': None}}}
        positive = transverse_cubic_crossing(source, lambda p:p, 10, 1)
        self.assertIsNotNone(positive)
        self.assertFalse(positive['route_identity_established'])
        noisy = deepcopy(source)
        noisy['source_native_segment']['control_points_display'][-1][1] = 4.0000004
        self.assertIsNotNone(transverse_cubic_crossing(noisy, lambda p:p, 10, 1))
        for name, value in [('control_points_display', [[4,3],[6,-3]]),
                             ('start_display', [4,0]), ('end_display', [10,0])]:
            bad = deepcopy(source);bad['source_native_segment'][name] = value
            self.assertIsNone(transverse_cubic_crossing(bad, lambda p:p, 10, 1))
        source['source_native_segment']['style']['fill'] = [0,0,0]
        self.assertIsNone(transverse_cubic_crossing(source, lambda p:p, 10, 1))

    def test_bounded_inline_detail_requires_dense_finite_nonroute_ink(self):
        def row(index, a, b):
            return {'source_primitive_ref':f'detail-{index}', 'points_display':[a,b],
                    'source_native_segment':{'kind':'line'}}
        sources = [row(i, [5, -1 + i / 8 * 2], [5, -1 + (i + 1) / 8 * 2]) for i in range(8)]
        classifications = [{'source_primitive_ref':r['source_primitive_ref'],
                            'classification':'unresolved_incident_endpoint_or_interior_feature'} for r in sources]
        result = bounded_inline_detail_components(classifications, sources, lambda p:p, 10, 1, set())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['state'], 'accepted')
        self.assertFalse(result[0]['physical_continuity_established'])
        for changed, reason in [({sources[0]['source_primitive_ref']}, 'component_ink_also_supports_certified_route')]:
            blocked = bounded_inline_detail_components(classifications, sources, lambda p:p, 10, 1, changed)
            self.assertEqual(blocked[0]['state'], 'abstained')
            self.assertIn(reason, blocked[0]['reasons'])
        sparse = bounded_inline_detail_components(classifications[:1], sources[:1], lambda p:p, 10, 1, set())
        self.assertEqual(sparse[0]['state'], 'abstained')

    def test_bound_annotation_role_reuses_accepted_contact_but_excludes_route_ink(self):
        annotation_ref = next(r['source_primitive_ref'] for r in self.query['source_rows']
                              if r['source_primitive_ref'] not in self.scope['input_composite']['member_source_primitive_refs'])
        route_ref = self.scope['input_composite']['member_source_primitive_refs'][0]
        evidence = {'id':'evidence', 'state':'observed', 'page_ref':self.scope['page_ref'],
            'method':{'name':'complete_native_dot_leader_contact','version':'1.0.0'},
            'target_kind':'route_composite', 'target_refs':[self.scope['composite_ref']],
            'geometric_evidence_refs':[route_ref, annotation_ref],
            'automatic_search_certificate':{'leader_observation_ref':'text',
                'source_searches_sha256':'a'*64, 'reviewed_selectors_used':False}}
        bindings = {'relations':[{'state':'accepted','target_kind':'route_composite',
            'target_refs':[self.scope['composite_ref']], 'binding_evidence_refs':['evidence']}],
            'binding_evidence':[evidence]}
        roles = bound_annotation_stroke_roles(self.scope, self.query, bindings, {route_ref})
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0]['source_primitive_refs'], [annotation_ref])
        self.assertEqual(roles[0]['excluded_route_boundary_source_refs'], [route_ref])
        self.assertFalse(roles[0]['attribute_applicability_established'])
        self.assertFalse(roles[0]['physical_continuation_established'])
        blocked = deepcopy(bindings)
        blocked['binding_evidence'][0]['automatic_search_certificate']['reviewed_selectors_used'] = True
        self.assertEqual(bound_annotation_stroke_roles(self.scope, self.query, blocked, {route_ref}), [])

    def test_junction_interior_retains_hidden_branch_and_crossing_as_different_results(self):
        ports = [{'center':[0,1],'outward':[1,0]}, {'center':[10,1],'outward':[-1,0]}]
        def row(ref,a,b):
            return {'source_primitive_ref':ref,'points_display':[a,b], 'source_native_segment':{'kind':'line'}}
        source = [row('wall0',[0,0],[10,0]),row('wall1',[0,2],[10,2])]
        connection = {'id':'junction','page_ref':'page','state':'accepted','ports':ports,
            'search':{'complete':True},'source_rows':source,
            'boundary_paths':[{'points_display':r['points_display'],'source_primitive_refs':[r['source_primitive_ref']]} for r in source]}
        self.assertTrue(certify_junction_interior(connection)['projected_scope_complete'])
        crossing = deepcopy(connection);crossing['source_rows'].append(row('crossing',[5,-4],[5,5]))
        self.assertTrue(certify_junction_interior(crossing)['projected_scope_complete'])
        hidden = deepcopy(connection);hidden['source_rows'].append(row('hidden',[5,1],[5,5]))
        self.assertFalse(certify_junction_interior(hidden)['projected_scope_complete'])

    def test_partial_straight_sidewall_is_not_an_interior_branch(self):
        def row(ref,a,b):
            return {'source_primitive_ref':ref,'points_display':[a,b], 'source_native_segment':{'kind':'line'}}
        walls=[row('lower',[-.1,0],[.1,0]),row('upper',[-.1,2],[.1,2])]
        c={'id':'join','page_ref':'page','state':'accepted','relation_type':'projected_collinear_boundary_join',
           'ports':[{'center':[0,1],'outward':[1,0]},{'center':[0,1],'outward':[-1,0]}],
           'search':{'complete':True},'source_rows':walls+[row('long',[-10,0],[0,0])],
           'boundary_paths':[{'points_display':r['points_display'],'source_primitive_refs':[r['source_primitive_ref']]} for r in walls]}
        self.assertFalse(certify_junction_interior(c,version=1)['projected_scope_complete'])
        self.assertTrue(certify_junction_interior(c,version=2)['projected_scope_complete'])
        for ref,points in [('branch',[[0,0],[0,5]]),('interior',[[0,1],[5,1]]),
                           ('near_sidewall',[[0,.01],[5,.01]])]:
            other=deepcopy(c);other['source_rows'].append(row(ref,*points))
            self.assertFalse(certify_junction_interior(other)['projected_scope_complete'],ref)
        crossing=deepcopy(c);crossing['source_rows'].append(row('crossing',[.04,-5],[.04,5]))
        self.assertTrue(certify_junction_interior(crossing)['projected_scope_complete'])
        self.assertEqual('transverse_crossing_without_native_endpoint_attachment',
            certify_junction_interior(crossing)['source_classifications'][-1]['classification'])
        c['search']['complete']=False
        self.assertFalse(certify_junction_interior(c)['projected_scope_complete'])

    def test_connected_composition_cannot_skip_an_unsearched_middle_member(self):
        composites, fragments, traces, joins = [], [], [], []
        for i in range(3):
            ref = str(i)
            composites.append({'id':ref, 'page_ref':'page', 'member_fragment_refs':[ref],
                'derived_geometry':{'centreline_points_display':[[12*i,1],[12*i+10,1]]}})
            fragments.append({'id':ref,'local_metric_observation':{'drawing_inches_per_paper_inch':1}})
            traces.append({'id':'trace'+ref,'scope_ref':'scope'+ref,'composite_refs':[ref],
                'projected_scope_complete':True,'endpoint_classifications':[],'reasons':[]})
            if i:
                a,b=12*i-2,12*i
                sources=[{'source_primitive_ref':f'{i}-{y}', 'points_display':[[a,y],[b,y]],
                          'source_native_segment':{'kind':'line'}} for y in (0,2)]
                joins.append({'id':'join'+ref,'page_ref':'page','state':'accepted',
                    'relation_type':'projected_collinear_boundary_join', 'composite_refs':[str(i-1),ref],
                    'ports':[{'id':f'{i}-left','composite_ref':str(i-1),'center':[a,1],'outward':[1,0]},
                             {'id':f'{i}-right','composite_ref':ref,'center':[b,1],'outward':[-1,0]}],
                    'search':{'complete':True},'source_rows':sources,
                    'boundary_paths':[{'points_display':s['points_display'],
                                       'source_primitive_refs':[s['source_primitive_ref']]} for s in sources]})
        def build(rows):
            return connected_trace_scopes(rows, joins, {'accepted_composites':composites}, {'pages':[{'fragments':fragments}]})
        missing=build([traces[0],traces[2]])
        self.assertEqual(missing['parent_scopes'][0]['uncovered_members'][0]['composite_ref'],'1')
        self.assertEqual(len(missing['complete_subtraces']),2)
        self.assertTrue(all(r['member_corridor_count']==1 for r in missing['complete_subtraces']))
        complete=build(traces)
        self.assertEqual(len(complete['complete_subtraces']),1)
        self.assertEqual(complete['complete_subtraces'][0]['member_corridor_count'],3)
        self.assertEqual(complete['complete_subtraces'][0]['projected_length_display_points'],34)
        self.assertIsNone(complete['complete_subtraces'][0]['installed_length_m'])


if __name__ == '__main__':
    unittest.main()
