import copy
import unittest

from tools.validate_mep_subpath_endpoints import contains, intersects, review_gate, expand_review


class SubpathEndpointGateTest(unittest.TestCase):
    def setUp(self):
        self.protocol={'cards':[{'case_id':'delta-01','group':'delta'}]}
        self.review={'decisions':[{'case_id':'delta-01','source_role':'pipe_continues',
            'source_refs':['native.a'],'note':'visible continuous walls', 'boundary_verdict':'preserved',
            'end_reviews':[{'endpoint_index':i,'verdict':'visually_continuous','note':'walls continue'} for i in (0,1)]}]}

    def test_clean_interiors_do_not_supply_terminal_truth(self):
        result=review_gate(self.protocol,self.review)
        self.assertFalse(result['publication_gate_passed'])
        self.assertIn('physical_cap',result['missing_endpoint_truth_classes'])
        self.assertIsNone(result['genuine_endpoint_recall'])

    def test_two_distinct_interval_ends_required(self):
        self.review['decisions'][0]['end_reviews'][1]['endpoint_index']=0
        with self.assertRaisesRegex(ValueError,'both recovered'):review_gate(self.protocol,self.review)

    def test_boundary_uncertainty_cannot_hide_under_clean_interior(self):
        self.review['decisions'][0]['end_reviews'][1]['verdict']='unresolved'
        self.assertEqual(['delta-01'],review_gate(self.protocol,self.review)['unresolved_cases'])

    def test_unsafe_end_is_counted(self):
        self.review['decisions'][0]['end_reviews'][1]['verdict']='unsafe'
        self.assertEqual(['delta-01'],review_gate(self.protocol,self.review)['unsafe_cases'])

    def test_every_case_exactly_once(self):
        self.review['decisions']*=2
        with self.assertRaises(ValueError):review_gate(self.protocol,self.review)

    def test_no_native_refs_no_review(self):
        self.review['decisions'][0]['source_refs']=[]
        with self.assertRaises(ValueError):review_gate(self.protocol,self.review)

    def test_visual_gold_alone_cannot_publish(self):
        for role in ('genuine_termination','physical_cap','equipment_contact','crossing_near_endpoint'):
            self.protocol['cards'].append({'case_id':role,'group':'endpoint-pool'})
            row=copy.deepcopy(self.review['decisions'][0]);row.update(case_id=role,source_role=role,end_reviews=[])
            self.review['decisions'].append(row)
        result=review_gate(self.protocol,self.review)
        self.assertEqual([],result['missing_endpoint_truth_classes'])
        self.assertFalse(result['publication_gate_passed'])

    def test_query_crop_coverage_not_bbox_overlap(self):
        self.assertTrue(intersects([0,0,10,10],[9,9,11,11]))
        self.assertFalse(contains([0,0,10,10],[9,9,11,11]))
        self.assertTrue(contains([0,0,10,10],[1,1,9,9]))

    def test_spatial_references_never_assign_semantic_role(self):
        evidence={'cases':[{'case_id':'x','source_refs':['wall.stroke'],'reference_meaning':'crop context'}]}
        review={'role_key':{'N':'non_route_drawing_content'},'delta_reviews':[],
            'endpoint_reviews':[['x','N','architectural corner']], 'native_endpoint_gate_passed':True}
        expanded=expand_review(review,evidence)
        self.assertEqual('non_route_drawing_content',expanded['decisions'][0]['source_role'])
        self.assertFalse(expanded['native_endpoint_gate_passed'])
