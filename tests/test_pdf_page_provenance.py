import tempfile
import unittest
from pathlib import Path

import fitz

from src.drawing_engine.core.pdf_page_provenance import (
    build_extracted_page_provenance,
    validate_extracted_page_provenance,
)


class PdfPageProvenanceTest(unittest.TestCase):
    def test_extracted_page_records_identical_native_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.pdf"
            extracted_path = Path(directory) / "extracted.pdf"
            source = fitz.open()
            source.new_page(width=200, height=100).insert_text((20, 30), "first")
            page = source.new_page(width=200, height=100)
            page.insert_text((20, 30), "Section A-A")
            page.draw_rect((20, 40, 180, 80), width=1)
            source.save(source_path)
            extracted = fitz.open()
            extracted.insert_pdf(source, from_page=1, to_page=1)
            extracted.save(extracted_path)
            source.close()
            extracted.close()

            record = build_extracted_page_provenance(source_path, 2, extracted_path)
            validate_extracted_page_provenance(record, extracted_path)

        self.assertTrue(record["canonical_page_content_identical"])
        self.assertEqual(record["source"]["word_count"], 2)
        self.assertEqual(record["source"]["drawing_object_count"], 1)
        self.assertEqual(
            record["source"]["canonical_page_content_sha256"],
            record["extracted_fixture"]["canonical_page_content_sha256"],
        )

    def test_nonidentical_page_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "left.pdf"
            right_path = Path(directory) / "right.pdf"
            for path, text in ((left_path, "left"), (right_path, "right")):
                document = fitz.open()
                document.new_page().insert_text((20, 30), text)
                document.save(path)
                document.close()
            with self.assertRaisesRegex(ValueError, "identical canonical content"):
                build_extracted_page_provenance(left_path, 1, right_path)


if __name__ == "__main__":
    unittest.main()
