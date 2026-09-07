"""Prove that an extracted PDF page preserves canonical native content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fitz


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (fitz.Point, fitz.Rect, fitz.Quad, fitz.Matrix)):
        return [_json_value(item) for item in value]
    return value


def canonical_page_signature(page: fitz.Page) -> dict[str, Any]:
    """Hash native words and drawing objects in display coordinates."""

    words = page.get_text("words")
    drawings = page.get_drawings()
    payload = _json_value(
        {
            "page_rect_display": page.rect,
            "page_rotation": page.rotation,
            "words": words,
            "drawing_objects": drawings,
        }
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "canonicalization": "pymupdf-native-page-content-v1",
        "canonical_page_content_sha256": hashlib.sha256(encoded).hexdigest(),
        "word_count": len(words),
        "drawing_object_count": len(drawings),
        "page_rect_display": list(page.rect),
        "page_rotation": page.rotation,
    }


def build_extracted_page_provenance(
    source_pdf: Path,
    source_page_number: int,
    extracted_pdf: Path,
    extracted_page_number: int = 1,
) -> dict[str, Any]:
    """Return a fail-closed equivalence record for two PDF pages."""

    source_pdf = source_pdf.resolve()
    extracted_pdf = extracted_pdf.resolve()
    with fitz.open(source_pdf) as source_document, fitz.open(extracted_pdf) as extracted_document:
        if not 1 <= source_page_number <= source_document.page_count:
            raise ValueError("source page number is outside the PDF")
        if not 1 <= extracted_page_number <= extracted_document.page_count:
            raise ValueError("extracted page number is outside the PDF")
        source_signature = canonical_page_signature(source_document[source_page_number - 1])
        extracted_signature = canonical_page_signature(extracted_document[extracted_page_number - 1])
        identical = source_signature == extracted_signature
        if not identical:
            raise ValueError("source and extracted pages do not have identical canonical content")
        return {
            "schema_version": "0.1.0",
            "layer": "extracted_pdf_page_provenance",
            "status": "verified_identical_canonical_page_content",
            "source": {
                "pdf": str(source_pdf),
                "pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
                "page_number": source_page_number,
                **source_signature,
            },
            "extracted_fixture": {
                "pdf": str(extracted_pdf),
                "pdf_sha256": hashlib.sha256(extracted_pdf.read_bytes()).hexdigest(),
                "page_number": extracted_page_number,
                **extracted_signature,
            },
            "canonical_page_content_identical": True,
            "contract": {
                "native_words_compared": True,
                "native_drawing_objects_compared": True,
                "page_geometry_and_rotation_compared": True,
                "schedule_values_used": False,
            },
        }


def validate_extracted_page_provenance(
    record: dict[str, Any],
    extracted_pdf: Path,
) -> None:
    """Verify a stored record still matches both PDFs exactly."""

    source = record.get("source", {})
    extracted = record.get("extracted_fixture", {})
    expected_fixture = extracted_pdf.resolve()
    if Path(str(extracted.get("pdf", ""))).resolve() != expected_fixture:
        raise ValueError("provenance fixture path does not match the analyzed PDF")
    rebuilt = build_extracted_page_provenance(
        Path(str(source.get("pdf", ""))),
        int(source.get("page_number", 0)),
        expected_fixture,
        int(extracted.get("page_number", 0)),
    )
    if rebuilt != record:
        raise ValueError("stored extracted-page provenance is stale")
