import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.cli import discover_pdfs, export_audits, inspect_result, main, write_json, resolve_audit_task

ROOT = Path(__file__).resolve().parents[1]


class FolderCliTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "drawing.PDF"
        with fitz.open() as pdf:
            for text in ("UNSELECTED FIRST PAGE", "SELECTED SECOND PAGE"):
                page = pdf.new_page(width=1000, height=700)
                page.insert_text((50, 50), text)
                page.draw_rect(fitz.Rect(100, 100, 200, 180))
            pdf.save(self.source)

    def test_launcher_from_pdf_folder_and_page_shorthand(self):
        result = subprocess.run([str(ROOT / "bin/estimation"), "audit", "--help"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--page", result.stdout)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        manifest = self.root / "result.json"
        write_json(manifest, {"execution_status": "succeeded"})
        with patch("src.drawing_engine.cli.export_audits", return_value=manifest) as export, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit", "page", "2", "--task", "hvac", "--worker"]), 0)
            self.assertEqual(export.call_args.kwargs["page_number"], 2)
            self.assertEqual(export.call_args.kwargs["task"], "hvac")
            first_output = export.call_args.args[1]
            self.assertEqual(main(["audit", "--task", "structural", "--worker"]), 0)
            self.assertNotEqual(first_output, export.call_args.args[1])
            self.assertEqual(export.call_args.args[1].parent, self.root / "estimation-output")

    def test_discovery_is_case_insensitive_nonrecursive_and_empty_is_error(self):
        folder = self.root / "estimation-output"
        folder.mkdir()
        (folder / "old-audit.pdf").write_bytes(b"do not consume generated outputs")
        (self.root / "notes.txt").write_text("ignored")
        self.assertEqual(discover_pdfs(self.root), [self.source])
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "no PDF"):
            discover_pdfs(empty)

    def mechanical_source(self):
        source = self.root / "neutral-name.pdf"
        with fitz.open() as pdf:
            for title in ("UNCLASSIFIED DRAWING", "Level 01 Area 1A Mechanical Piping Plan"):
                pdf.new_page().insert_text((50, 50), title)
            pdf.save(source)
        return source

    def test_auto_task_uses_only_selected_native_page_title(self):
        import src.drawing_engine.disciplines.mep.mep_sheet_registry as m1
        source = self.mechanical_source()
        visited = []
        native_tokens = m1._native_tokens
        def read(page, ref):
            visited.append(page.number + 1)
            return native_tokens(page, ref)
        with patch.object(m1, "_native_tokens", side_effect=read), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(resolve_audit_task([source], 2), "mep")
        self.assertEqual(visited, [2])
        self.assertEqual(json.loads(stderr.getvalue())["basis"], "native_sheet_titles")
        with self.assertRaisesRegex(ValueError, "specify --task"):
            resolve_audit_task([source])
        with self.assertRaisesRegex(ValueError, "specify --task"):
            resolve_audit_task([source, self.source], 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_audit_task([source], 3)

    def test_unknown_task_prompts_and_validates_selection(self):
        with patch("builtins.input", side_effect=["", "auto", " Rebar "]) as prompt, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(resolve_audit_task([self.source], 2, interactive=True), "rebar")
        self.assertEqual(prompt.call_count, 3)
        self.assertEqual(json.loads(stderr.getvalue().splitlines()[-1])["basis"], "user_selection")

    def test_unknown_task_without_terminal_never_starts_extraction(self):
        with patch("sys.stdin.isatty", return_value=False), \
             patch("src.drawing_engine.cli.export_audits") as export, \
             patch("src.drawing_engine.operations.run_artifact_job.run") as run, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["audit", str(self.source), "--page", "2"]), 1)
        self.assertIn("specify --task", stderr.getvalue())
        export.assert_not_called()
        run.assert_not_called()

    def test_parent_resolves_task_before_starting_worker(self):
        source = self.mechanical_source()
        for name, pdf, response, expected in (("detected", source, None, "mep"),
                                               ("prompted", self.source, "hvac", "hvac")):
            with self.subTest(name=name), patch("sys.stdin.isatty", return_value=True), \
                 patch("builtins.input", return_value=response) as prompt, \
                 patch("src.drawing_engine.operations.run_artifact_job.run") as run, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["audit", str(pdf), "--page", "2",
                                       "--output", str(self.root / name)]), 0)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--task") + 1], expected)
                self.assertEqual(command[command.index("--page") + 1], "2")
                self.assertEqual(prompt.call_count, int(response is not None))

    def test_explicit_task_bypasses_detection(self):
        manifest = self.root / "result.json"
        write_json(manifest, {"execution_status": "succeeded"})
        with patch("src.drawing_engine.cli.resolve_audit_task") as detect, \
             patch("src.drawing_engine.cli.export_audits", return_value=manifest) as export, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit", str(self.source), "--task", "concrete", "--worker"]), 0)
        detect.assert_not_called()
        self.assertEqual(export.call_args.kwargs["task"], "concrete")

    def test_interrupt_reports_cancellation_without_traceback(self):
        with patch("src.drawing_engine.cli.export_audits", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["audit", str(self.source), "--task", "mep", "--worker"]), 130)
        self.assertEqual(json.loads(stderr.getvalue())["event"], "cancelled")
        for code in (130, -2):
            with patch("src.drawing_engine.operations.run_artifact_job.run", side_effect=subprocess.CalledProcessError(code, "worker")), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(main(["audit", str(self.source), "--task", "mep",
                                       "--output", str(self.root / f"cancel-{code}")]), 130)
            self.assertEqual(json.loads(stderr.getvalue().splitlines()[-1])["event"], "cancelled")
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_multiple_documents_keep_successes_and_report_failures(self):
        (self.root / "second.pdf").write_bytes(b"invalid PDF")
        calls = []
        def produce(source, output, **kwargs):
            calls.append((source, kwargs))
            if source.name == "second.pdf":
                raise ValueError("selected page is outside the source PDF")
            output.mkdir()
            write_json(output / "result.json", {"execution_status": "succeeded"})
            return output / "result.json"
        with patch("src.drawing_engine.cli.export_comparison", side_effect=produce):
            result = export_audits(self.root, self.root / "batch", page_number=2, task="structural")
        report = json.loads(result.read_text())
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(options["page_number"] == 2 for _, options in calls))
        self.assertEqual(report["execution_status"], "partially_failed")
        self.assertFalse(report["quantities_aggregated"])
        self.assertEqual(report["failed_document_count"], 1)
        self.assertIn("sha256", report["documents"][0]["result"])

    def test_invalid_page_and_ambiguous_syntax_do_not_start_extraction(self):
        with patch("src.drawing_engine.cli.export_audits") as export, contextlib.redirect_stderr(io.StringIO()):
            for args in (["audit", "page", "0"], ["audit", str(self.source), "2"],
                         ["audit", "page", "2", "--page", "1"]):
                self.assertEqual(main([*args, "--worker"]), 1)
            export.assert_not_called()

    def test_declaration_extraction_does_not_scan_unselected_page(self):
        from src.drawing_engine.disciplines.concrete.schedule_comparison import extract_declared_schedules
        visited = []
        def read(page):
            visited.append(page.number + 1)
            return [{"value": 1}]
        with patch("src.drawing_engine.disciplines.concrete.schedule_comparison._concrete_declarations", side_effect=read), \
             patch("src.drawing_engine.disciplines.concrete.schedule_comparison._steel_declarations", side_effect=read):
            result = extract_declared_schedules(self.source, page_numbers=[2])
        self.assertEqual(visited, [2, 2])
        self.assertEqual([p["page"] for p in result["pages"]], [2])

    def test_mep_selection_preserves_ids_and_skips_unselected_content(self):
        from src.drawing_engine.disciplines.mep.mep_sheet_registry import extract_mep_sheet_registry
        from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations
        from src.drawing_engine.disciplines.mep.mep_annotation_observations import extract_pdf_annotation_observations
        import src.drawing_engine.disciplines.mep.mep_sheet_registry as m1
        with patch.object(m1, "_native_tokens", wraps=m1._native_tokens) as tokens, \
             patch.object(m1, "_assess_page_quality_bounded", wraps=m1._assess_page_quality_bounded) as quality:
            registry = extract_mep_sheet_registry(self.source, page_numbers=[2], ocr_unknown_pages=False)
        self.assertEqual(tokens.call_count, 1)
        self.assertEqual(quality.call_count, 1)
        self.assertEqual(registry["pages"][0]["quality_route"]["mep_registry_route"], "not_processed")
        text = extract_mep_text_observations(pdf_path=self.source, sheet_registry=registry)
        self.assertEqual({r["page_number"] for r in text["observations"]}, {2})
        self.assertFalse(text["pages"][0]["native_text_scan_complete"])
        annotations = extract_pdf_annotation_observations(self.source, page_numbers=[2])
        self.assertFalse(annotations["pages"][0]["annotation_scan_complete"])
        self.assertEqual(registry["pages"][1]["page_ref"], annotations["pages"][1]["id"])

    def test_selected_structural_audit_runs_only_requested_page_and_replays(self):
        from src.drawing_engine.cli import export_comparison
        from src.drawing_engine.exports.estimation_project_export import FrozenProject, render_project_audit
        import src.drawing_engine.pipelines.generate_object_agnostic_bundle as engine
        visited = []
        understand = engine.understand_page
        def selected(page, **kwargs):
            visited.append(page.number + 1)
            return understand(page, **kwargs)
        with patch.object(engine, "understand_page", side_effect=selected):
            result = export_comparison(self.source, self.root / "structural", page_number=2)
        self.assertEqual(visited, [2])
        manifest = inspect_result(result)
        self.assertEqual(manifest["contract"]["source_page_numbers"], [2])
        graph = inspect_result(result, "engineering_graph")
        self.assertEqual([p["page"] for p in graph["pages"]], [2])
        self.assertEqual([p["page"] for p in inspect_result(result, "comparison")["pages"]], [2])
        with fitz.open(result.parent / "audit.pdf") as audit:
            self.assertEqual(len(audit), 3)
            self.assertIn("SELECTED SECOND PAGE", audit[0].get_text())
            self.assertNotIn("UNSELECTED FIRST PAGE", "".join(p.get_text() for p in audit))
        with patch.object(engine, "understand_page", side_effect=AssertionError("re-extracted")):
            render_project_audit(FrozenProject(result.parent, manifest), self.root / "replayed.pdf")
        self.assertEqual((result.parent / "source" / self.source.name).read_bytes(), self.source.read_bytes())

    def test_selected_mep_delivery_is_fresh_portable_and_keeps_search_evidence(self):
        from src.drawing_engine.exports.estimation_project_export import export_fresh_mep, FrozenProject, render_project_audit
        result = export_fresh_mep(self.source, self.root / "mep", page_number=2, task="hvac")
        manifest = inspect_result(result)
        self.assertFalse(manifest["contract"]["historical_inputs_used"])
        self.assertEqual(manifest["contract"]["source_page_numbers"], [2])
        self.assertIn("hvac-inventory", manifest["records"])
        self.assertIn("page-002.automatic-targets", manifest["records"])
        self.assertNotIn("page-001.automatic-targets", manifest["records"])
        performance = inspect_result(result, "page-002.performance")
        self.assertEqual(performance["engine"], "page_endpoint_topology")
        self.assertFalse(manifest["contract"]["approval_granted_by_execution"])
        with fitz.open(result.parent / "audit.pdf") as audit:
            self.assertNotIn("UNSELECTED FIRST PAGE", "".join(p.get_text() for p in audit))
            presentation = json.loads(audit.embfile_get("audit-manifest.json"))["presentation"]
            self.assertEqual(presentation["processed_source_page_numbers"], [2])
            self.assertEqual(presentation["source_rendering"], "raster")
            self.assertEqual(len(audit), 1)
            self.assertTrue(audit[0].get_images())
            self.assertFalse(audit[0].get_links())
            self.assertEqual(presentation["source_opacity"], .47)
            self.assertEqual(manifest["audit"]["page_count"], len(audit))
        with patch("src.drawing_engine.pipelines.generate_mep_automatic_items.run", side_effect=AssertionError("re-extracted")):
            render_project_audit(FrozenProject(result.parent, manifest), self.root / "mep-replayed.pdf")
