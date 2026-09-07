import argparse
from pathlib import Path
import tempfile
import unittest

import fitz

from src.drawing_engine.pipelines.generate_mep_stroke_ownership import write,read
from tools.run_mep_diagnostic_package import receipt_valid
from tools.render_mep_diagnostic_package import metric_length,page_rows,render,length_consistent
from tools.verify_mep_diagnostic_package import verify
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256


class DiagnosticPackageTest(unittest.TestCase):
    def test_all_diagnostic_subprocesses_use_current_tool_owners(self):
        from unittest.mock import patch
        from tools.run_mep_diagnostic_package import process, ROOT
        import json
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(output=Path(temporary), phase='inventory', source='source.pdf',
                database='project.sqlite', ownership='ownership.json', registry='registry.json', trace='trace.json')
            commands = []
            def capture(folder, name, command, required):
                commands.append(command)
                script = Path(command[2])
                self.assertEqual(script.parent, ROOT/'tools')
                self.assertTrue(script.is_file())
                for path in required:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'acceptance_gate': {'errors': []}, 'native_descriptor_pack': {'record_count': 0}}))
            scope = {'page_number': 1, 'page_ref': 'page:1', 'fields': {'scale': {'drawing_inches_per_paper_inch': 1}}}
            with patch('tools.run_mep_diagnostic_package.stage', side_effect=capture):
                self.assertEqual(process(args, scope, {})['state'], 'native_inventory_frozen')
                args.phase = 'diagnostic'
                result = process(args, scope, {})
                self.assertEqual(result['state'], 'native_callouts_replayed')
                self.assertIsNone(result['installed_length'])
            self.assertEqual([Path(c[2]).name for c in commands], [
                'generate_mep_source_denominator.py', 'generate_mep_source_denominator_v2.py',
                'generate_mep_page_callout_bindings.py', 'run_mep_diagnostic_package.py', 'partition_mep_diagnostic_page.py'])

    def test_unscaled_is_null_not_zero(self):
        self.assertIsNone(metric_length(500,None))
        self.assertIsNone(metric_length(None,48))
        self.assertAlmostEqual(metric_length(72,48),1.2192)

    def test_conflicting_candidate_length_is_detected_not_silently_replaced(self):
        row={'polyline_display':[[0,0],[3,4]],'projected_path_display_points':8}
        self.assertFalse(length_consistent(row))
        self.assertEqual(row['projected_path_display_points'],8)
        row['projected_path_display_points']=5
        self.assertTrue(length_consistent(row))

    def test_failed_sheet_has_explicit_no_interpretation(self):
        found,candidates,warnings,evidence=page_rows({'error':'inventory failed'},'hash')
        self.assertEqual((found,candidates,warnings),([],[],[]))
        self.assertEqual(evidence['state'],'no_current_interpretation')
        self.assertEqual(evidence['reason'],'inventory failed')

    def test_receipt_rejects_changed_output_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'value.json';write(f,{'a':1})
            receipt={'context':{'input':'one'},'artifacts':{str(f):_file_sha256(f)}}
            self.assertTrue(receipt_valid(receipt,{'input':'one'}))
            self.assertFalse(receipt_valid(receipt,{'input':'two'}))
            write(f,{'a':2})
            self.assertFalse(receipt_valid(receipt,{'input':'one'}))

    def test_every_sheet_rendered_as_rgb_without_false_empty_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original.pdf';output=root/'audit.pdf'
            with fitz.open() as pdf:
                for height in (792,1500):
                    page=pdf.new_page(width=1000,height=height)
                    page.draw_line((50,50),(500,500),color=(1,0,0))
                    page.add_freetext_annot(fitz.Rect(60,70,300,140),'SOURCE DIVIDER',fontsize=24,text_color=(1,0,0))
                pdf.save(source)
            registry=root/'registry.json'
            write(registry,{'pages':[{'page_number':n,'fields':{'scale':{'state':'unknown'}}} for n in (1,2)]})
            digest=_file_sha256(source)
            write(root/'protocol.json',{'source_sha256':digest,'registry_sha256':_file_sha256(registry),'protected':{}})
            execution={'source_sha256':digest,'pages':[{'page':n,'state':'failed','error':'test fixture'} for n in (1,2)]}
            write(root/'interpret.json',execution)
            args=argparse.Namespace(batch=root,output=output,source=source,registry=registry)
            render(args)
            manifest=read(output.with_suffix('.manifest.json'))
            self.assertEqual(len(manifest['pages']),2)
            self.assertTrue(all(r['original_annotation_count']==1 for r in manifest['pages']))
            self.assertIsNone(manifest['package_length_total'])
            verify(argparse.Namespace(pdf=output,source=source,output=root/'checks'))
            self.assertTrue(read(root/'checks'/'checks.json')['mechanical_checks_passed'])
            with fitz.open(output) as pdf:
                self.assertEqual(len(pdf),2)
                for page in pdf:
                    raster=fitz.Pixmap(pdf,page.get_images()[0][0])
                    self.assertEqual(raster.colorspace.n,3)
                    self.assertIn('Processing incomplete',page.get_text())
                    self.assertIn('No accepted overlay is not evidence of no pipes.',page.get_text())
            with self.assertRaisesRegex(ValueError,'overwrite'):render(args)

    def test_duplicate_or_missing_sheets_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';write(source,{'source':'test'})
            registry=root/'registry.json';write(registry,{'pages':[{'page_number':1},{'page_number':2}]})
            digest=_file_sha256(source)
            write(root/'protocol.json',{'source_sha256':digest,'registry_sha256':_file_sha256(registry),'protected':{}})
            write(root/'interpret.json',{'source_sha256':digest,'pages':[{'page':1},{'page':1}]})
            with self.assertRaisesRegex(ValueError,'missing or duplicate'):
                render(argparse.Namespace(batch=root,source=source,registry=registry,output=root/'out.pdf'))
