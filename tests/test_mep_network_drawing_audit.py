from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import fitz

from src.drawing_engine.audit.render_mep_network_drawing_audit import render_network_drawing_audit, _windows


def fixture(directory, count=2, attributed=True, page_count=1):
    source=Path(directory)/"native.pdf"
    pdf=fitz.open()
    for number in range(page_count):
        page=pdf.new_page(width=800,height=600)
        page.insert_text((50,90),"KEEP NATIVE TEXT",fontsize=16)
        page.add_freetext_annot(fitz.Rect(50,120,230,150),"KEEP AUTHOR NOTE",fontsize=12)
    pdf.save(source)
    pdf.close()
    rows=[]
    for index in range(count):
        number=index%page_count+1
        points=[[30,84+index*1.3],[700,84+index*1.3]]
        values={"system":{"state":"accepted","text":"HHWS"},
                "size":{"state":"accepted","text":"1 in" if index%2==0 else "2 in"},
                "elevation":{"state":"accepted","text":"BOP +9 ft" if index%2==0 else "BOP +10 ft"}}
        rows.append({"id":f"segment.{index}","kind":"segment","label":"Route", "state":"derived",
            "payload":{"geometry_only":not attributed},"artifact":"network-hierarchy","pointer":f"/segments/{index}",
            "artifact_sha256":"frozen","display_values":values if attributed else {},
            "marks":[{"page_ref":f"p{number}","page_number":number,"source_ref":f"occurrence.{index}",
                      "points_display":points,"source_primitive_refs":[f"drawing[{index}].item[0].segment[0]"],
                      "display_values":{"projected_length":{"state":"observed","text":f"{index+1}.25 m",
                          "value":index+1.25,"unit":"m","basis":"projected_2d","scope":"page_local_occurrence"}}}]})
    review={"snapshot_id":"frozen-snapshot","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "project_id":"test","document_id":"test-document","rows":rows,"networks":[],"junctions":[],
            "pages":[{"page_number":n,"page_ref":f"p{n}","page_size_display":[800,600],"role":"mechanical_piping_plan"} for n in range(1,page_count+1)],
            "processing_scope":{"execution_page_numbers":list(range(1,page_count+1)),"registered_page_count":page_count}}
    return source,review


class MepNetworkDrawingAuditTest(unittest.TestCase):
    def test_single_page_sqlite_presentation_preserves_source_two_and_omits_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source, review = fixture(directory, count=4, page_count=2)
            output = Path(directory) / 'one.pdf'
            frozen = deepcopy(review)
            report = render_network_drawing_audit(source, output, review, preview_pages=[2],
                maximum_closeups=0, include_isolated_geometry=True, single_page=True, write_manifest=False)
            self.assertEqual(review, frozen)
            self.assertEqual(report['coverage']['overview_source_page_numbers'], [2])
            self.assertEqual(report['page_count'], 1)
            self.assertFalse(output.with_suffix('.manifest.json').exists())
            with fitz.open(output) as pdf:
                self.assertEqual(len(pdf), 1)
                self.assertIn('DRAWING 2', pdf[0].get_text().replace('\xa0', ' '))
                self.assertIn('Frozen SQLite', pdf[0].get_text().replace('\xa0', ' '))
                self.assertTrue(any(a.info.get('content') == 'KEEP AUTHOR NOTE' for a in pdf[0].annots()))
                self.assertTrue(any('segment.1' in a.info.get('content', '') for a in pdf[0].annots()))
                self.assertTrue(all(a.opacity == 0 for a in pdf[0].annots() if 'segment.' in a.info.get('content', '')))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                render_network_drawing_audit(source, Path(directory) / 'bad.pdf', review, maximum_closeups=0, single_page=True)

    def test_small_presentation_scope_can_use_its_requested_closeup_budget(self):
        entries = [{'key': str(i), 'page_number': 1, 'priority': 1,
                    'anchor_display': point, 'closeup_focus_display': point,
                    'priority_reasons': ['known_size'], 'priority_branch_junction_refs': []}
                   for i, point in enumerate(([250, 250], [900, 900], [1650, 1650]))]
        pages = [{'page_number': 1, 'page_size_display': [2000, 2000]}]
        self.assertEqual(len(_windows(entries, pages, 3)), 3)
        self.assertEqual(len(_windows(entries, pages, 2)), 2)
        self.assertEqual(_windows(entries, pages, 0), [])

    def test_local_readouts_never_merge_size_length_or_elevation(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory)
            frozen=deepcopy(review)
            output=Path(directory)/"audit.pdf"
            manifest=render_network_drawing_audit(source,output,review,maximum_closeups=1)
            self.assertEqual(review,frozen)
            self.assertEqual([e["display_values"]["size"] for e in manifest["entries"]],["1 in","2 in"])
            self.assertEqual([e["display_values"]["projected_length"] for e in manifest["entries"]],["1.25 m","2.25 m"])
            self.assertEqual([e["display_values"]["elevation"] for e in manifest["entries"]],["BOP +9 ft","BOP +10 ft"])
            self.assertFalse(manifest["authority"]["installed_length_established"])
            self.assertTrue(all(e["printed_readouts"] for e in manifest["entries"]))
            with fitz.open(output) as pdf:
                text="\n".join(p.get_text() for p in pdf).replace("\xa0"," ")
                self.assertIn("1.25 m",text)
                self.assertIn("2.25 m",text)
                self.assertNotIn("3.50 m",text)
                self.assertTrue(any(l["page"]==1 for l in pdf[-1].get_links()))

    def test_many_unknown_segments_are_marked_without_register_page_explosion(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=120,attributed=False)
            manifest=render_network_drawing_audit(source,Path(directory)/"audit.pdf",review,maximum_closeups=2,include_isolated_geometry=True)
            self.assertEqual(manifest["page_count"],2)
            self.assertEqual(manifest["coverage"]["marked_entry_count"],120)
            self.assertEqual(len(manifest["coverage"]["unlabelled_entry_keys"]),120)
            self.assertTrue(all(e["display_values"]["size"]=="Unknown" for e in manifest["entries"]))
            self.assertEqual(len(manifest["pages"][0]["marked_entry_keys"]),120)

    def test_default_filter_preserves_connected_unknown_known_isolated_and_equipment(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=7,attributed=False)
            review['rows'][1]['display_values']={'size':{'state':'accepted','text':'2 in'}}
            review['runs']=[{'id':'run','segment_refs':['segment.2','segment.3']}]
            review['networks']=[{'id':'network','segment_refs':['segment.3','segment.6']}]
            review['rows'][4].update(kind='hvac',label='Unresolved HVAC tag')
            review['rows'][5]['payload']['projected_scope_complete']=True
            frozen=deepcopy(review)
            manifest=render_network_drawing_audit(source,Path(directory)/'audit.pdf',review,maximum_closeups=0)
            self.assertEqual(review,frozen)
            self.assertEqual(len(manifest['entries']),7)
            self.assertEqual(manifest['coverage']['mapped_entry_count'],7)
            self.assertEqual(manifest['coverage']['marked_entry_count'],6)
            self.assertEqual(manifest['coverage']['presentation_omitted_entry_keys'],['segment.0:0'])
            self.assertFalse(manifest['entries'][0]['presentation_included'])
            self.assertTrue(all(e['presentation_included'] for e in manifest['entries'][1:]))
            self.assertEqual(manifest['entries'][4]['kind'],'unresolved_hvac_observation')
            self.assertEqual(manifest['pages'][0]['omitted_isolated_geometry_entry_keys'],['segment.0:0'])

    def test_source_text_annotation_pixels_and_source_bytes_survive_transparent_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=1)
            digest=review["source_sha256"]
            output=Path(directory)/"audit.pdf"
            render_network_drawing_audit(source,output,review,maximum_closeups=0)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            with fitz.open(source) as before,fitz.open(output) as after:
                self.assertEqual(sum(1 for _ in before[0].annots()),sum(1 for _ in after[1].annots()))
                for box in [before[0].search_for("KEEP NATIVE TEXT")[0],fitz.Rect(50,120,230,150)]:
                    original=before[0].get_pixmap(clip=box,annots=True).samples
                    rendered=after[1].get_pixmap(clip=box,annots=True).samples
                    self.assertEqual(len(original),len(rendered))
                    # A transparent form can round antialias pixels by one
                    # channel level; source lettering must not be obscured.
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(original,rendered)),1)

    def test_two_source_pages_graft_complete_overlay_and_retain_page_local_readouts(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,page_count=2)
            manifest=render_network_drawing_audit(source,Path(directory)/"audit.pdf",review,maximum_closeups=0)
            self.assertEqual([p["audit_page_number"] for p in manifest["pages"]],[2,3])
            self.assertEqual([e["printed_readouts"][0]["audit_page_number"] for e in manifest["entries"]],[2,3])

    def test_unordered_branch_ports_never_become_an_invented_polyline(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=1)
            review["junctions"]=[{"id":"branch","page_ref":"p1","state":"accepted",
                                  "port_points_display":[[100,100],[200,200],[100,200]],
                                  "centreline_points_display":[],"body_paths":[]}]
            manifest=render_network_drawing_audit(source,Path(directory)/"audit.pdf",review,maximum_closeups=0)
            self.assertEqual(manifest["junction_paths"],[])

    def test_fitting_context_never_duplicates_a_segment_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=1)
            context=deepcopy(review["rows"][0]["marks"][0])
            context.update(role="inferred_fitting_body_and_port_closeup",source_ref="fitting.context")
            review["rows"][0]["marks"].append(context)
            manifest=render_network_drawing_audit(source,Path(directory)/"audit.pdf",review,maximum_closeups=0)
            self.assertEqual(len(manifest["entries"]),1)
            self.assertTrue(any(row["reason"]=="supplemental_fitting_context_not_another_segment"
                                for row in manifest["omitted_rows"]))

    def test_unresolved_tag_callout_preserves_native_text_without_inventing_body(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=1)
            row=review['rows'][0]
            row.update(kind='hvac',label='HUH-X',display_values={})
            with fitz.open(source) as pdf:
                box=list(pdf[0].search_for('KEEP NATIVE TEXT')[0])
            mark=row['marks'][0]
            mark.pop('points_display')
            mark.update(bbox_display=box,display_values={})
            manifest=render_network_drawing_audit(source,Path(directory)/'audit.pdf',review,maximum_closeups=0)
            entry=manifest['entries'][0]
            self.assertEqual(entry['kind'],'unresolved_hvac_observation')
            self.assertEqual(entry['printed_readouts'][0]['kind'],'unresolved_tag_callout')
            self.assertEqual(entry['source_mark']['bbox_display'],box)
            self.assertEqual(manifest['coverage']['visible_overview_marker_count'],1)

    def test_unknown_completed_trace_and_accepted_branch_receive_closeups(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=5,attributed=False)
            review['rows'][0]['payload']['projected_scope_complete']=True
            review['rows'][0]['marks'][0]['points_display']=[[20,40],[80,40]]
            for index in range(1,4):
                review['rows'][index]['marks'][0]['points_display']=[[10,460+index*10],[760,460+index*10]]
            review['junctions']=[{'id':'accepted.branch','state':'accepted','page_ref':'p1',
                'segment_refs':['segment.1','segment.2','segment.3'],
                'port_points_display':[[760,470],[760,480],[760,490]],'centreline_points_display':[]},
                {'id':'unresolved.branch','state':'abstained','page_ref':'p1','segment_refs':['segment.4'],
                 'port_points_display':[[300,200],[300,220],[320,220]],'centreline_points_display':[]}]
            frozen=deepcopy(review)
            manifest=render_network_drawing_audit(source,Path(directory)/'audit.pdf',review,maximum_closeups=2)
            self.assertEqual(review,frozen)
            self.assertEqual(len(manifest['closeups']),2)
            self.assertTrue(any('completed_projected_scope' in c['priority_reasons'] for c in manifest['closeups']))
            branch=next(c for c in manifest['closeups'] if c['priority_branch_junction_refs'])
            self.assertEqual(branch['priority_branch_junction_refs'],['accepted.branch'])
            self.assertEqual(set(branch['readout_entry_keys']),{'segment.1:0','segment.2:0','segment.3:0'})
            self.assertTrue(all(e['display_values']['size']=='Unknown' for e in manifest['entries']))
            self.assertEqual(manifest['entries'][4]['priority'],0)
            self.assertEqual(manifest['junction_paths'],[])

    def test_source_revision_and_geometry_scope_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source,review=fixture(directory,count=1)
            output=Path(directory)/"audit.pdf"
            with self.assertRaisesRegex(ValueError,"overwrite"):
                render_network_drawing_audit(source,source,review)
            wrong=deepcopy(review);wrong["source_sha256"]="wrong"
            with self.assertRaisesRegex(ValueError,"differs"):
                render_network_drawing_audit(source,output,wrong)
            review['rows'][0]['display_values']={}  # Filtering never hides invalid evidence geometry.
            review["rows"][0]["marks"][0]["points_display"][0][0]=900
            with self.assertRaisesRegex(ValueError,"outside"):
                render_network_drawing_audit(source,output,review)


if __name__=="__main__":
    unittest.main()
