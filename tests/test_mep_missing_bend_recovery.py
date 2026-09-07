import math
import unittest
from pathlib import Path

from tools.recover_mep_missing_bends import quarter_chains, pair_chains, recover_window
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style


class MissingBendRecoveryTest(unittest.TestCase):
    def fixture(self):
        style={'stroke':[.2,.7,.3],'fill':None,'width':.24,'dash':'[] 0'}
        rows=[];paths=[]
        def add(a,b):
            ref=f'drawing[{len(rows)}].item[0].segment[0]'
            rows.append({'source_primitive_ref':ref,'points_display':[a,b],
                'source_native_segment':{'id':ref,'kind':'line','style':style},
                'bbox_display':[min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])]})
        for radius in (10,14):
            points=[[radius*math.cos(i*math.pi/24),radius*math.sin(i*math.pi/24)] for i in range(13)]
            points[1][0]=radius;points[-2][1]=radius;points[-1][0]=0
            for a,b in zip(points,points[1:]):add(a,b)
            paths.append(points)
        for end in (0,-1):add(paths[0][end],paths[1][end])
        style=_normalise_style(style)
        window={'bbox_display':[-5,-5,20,20],'nominating_ports':[
            {'candidate_ref':'one','style':style,'width':4},
            {'candidate_ref':'two','style':style,'width':4}]}
        return rows,style,window,add,paths

    def test_two_native_microsegment_sides_close_geometry_only(self):
        rows,style,window,add,paths=self.fixture()
        self.assertEqual(2,len(quarter_chains(rows,style)))
        records=recover_window(window,rows,'page',True)
        self.assertEqual(1,len(records))
        self.assertEqual('accepted',records[0]['geometry_state'])
        self.assertFalse(records[0]['system_identity_accepted'])
        self.assertIsNone(records[0]['installed_length'])
        self.assertEqual(24,len(records[0]['source_segment_refs']))

    def test_branch_missing_side_and_incomplete_inventory_do_not_accept(self):
        rows,style,window,add,paths=self.fixture()
        add(paths[0][6],[8,8])
        self.assertFalse(any(r['geometry_state']=='accepted' for r in recover_window(window,rows,'page',True)))
        rows,style,window,add,paths=self.fixture()
        del rows[4]
        self.assertFalse(any(r['geometry_state']=='accepted' for r in recover_window(window,rows,'page',True)))
        rows,style,window,add,paths=self.fixture()
        self.assertFalse(any(r['geometry_state']=='accepted' for r in recover_window(window,rows,'page',False)))

    def test_crossing_is_not_a_branch_or_connection(self):
        rows,style,window,add,paths=self.fixture()
        add([-4,7],[19,7])
        records=recover_window(window,rows,'page',True)
        self.assertEqual(1,len(records))
        self.assertEqual('accepted',records[0]['geometry_state'])
        self.assertEqual('transverse_crossing_without_native_endpoint_attachment',
            records[0]['junction_interior_certificate']['source_classifications'][-1]['classification'])
        for r in records:
            self.assertFalse(r['physical_continuity_established'])
            self.assertNotIn(rows[-1]['source_primitive_ref'],r['source_segment_refs'])

    def test_circle_is_not_a_bend(self):
        rows,style,window,add,paths=self.fixture()
        rows.clear()
        points=[[10*math.cos(i*math.pi/12),10*math.sin(i*math.pi/12)] for i in range(24)]
        for a,b in zip(points,points[1:]+points[:1]):add(a,b)
        self.assertEqual([],quarter_chains(rows,style))

    def test_real_green_microsegments_reach_geometry_candidacy_not_identity(self):
        from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read
        root=Path(__file__).resolve().parents[1]
        artifact=root/'output/mep-route-boundaries-2026-09-04/bends/recovered-bends.json.gz'
        records=read(artifact)['records']
        known={f'drawing[{i}].item[0].segment[0]' for i in range(303293,303303)}
        matching=[r for r in records if known<=set(r['source_segment_refs'])]
        self.assertEqual(1,len(matching))
        row=matching[0]
        self.assertEqual('accepted',row['geometry_state'])
        self.assertEqual('abstained',row['full_interface_certificate']['state'])
        self.assertTrue(row['analysis_cuts_are_not_physical_ports'])
        self.assertEqual(4,len(row['terminal_boundary_fragments']))
        self.assertEqual(24,len(row['source_segment_refs']))
        self.assertEqual(20,len(row['core_source_segment_refs']))
        self.assertFalse(row['system_identity_accepted'])
        self.assertIsNone(row['installed_length'])
