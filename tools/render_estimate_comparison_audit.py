#!/usr/bin/env python3
"""Render frozen calculation versus declared schedule discrepancy audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.audit.audit_presentation import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    audit_font_file,
    choose,
    duration_text,
    quantity_text,
    review_action,
    term,
    validate_language,
)


COLORS = {
    "ink": (0.08, 0.12, 0.20),
    "muted": (0.35, 0.40, 0.48),
    "paper": (0.97, 0.98, 0.99),
    "white": (1.0, 1.0, 1.0),
    "calculated": (0.05, 0.55, 0.68),
    "declared": (0.90, 0.56, 0.07),
    "pass": (0.08, 0.58, 0.32),
    "review": (0.82, 0.20, 0.18),
    "unknown": (0.43, 0.47, 0.54),
}


def insert_text(page: fitz.Page, rect: fitz.Rect, text: str, size: float, color: tuple[float, float, float], *, bold: bool = False) -> None:
    page.insert_textbox(rect, text, fontsize=size, fontname="AuditSansBold" if bold else "AuditSans", fontfile=audit_font_file(bold), color=color, lineheight=1.22)


def _fmt(value: float | None, unit: str, language: str) -> str:
    precision = 4 if unit == "m3" else 1
    return quantity_text(language, value, unit, precision)


def _fmt_delta(value: float | None, unit: str, language: str) -> str:
    precision = 4 if unit == "m3" else 1
    return quantity_text(language, value, unit, precision, signed=True)


def _percent_text(value: float | None, language: str) -> str:
    return choose(language, "UNKNOWN", "НЕ ОПРЕДЕЛЕНО") if value is None else f"{value:+.3f}%"


def _status_color(comparison: dict[str, Any]) -> tuple[float, float, float]:
    if comparison["severity"] == "pass":
        return COLORS["pass"]
    if comparison["severity"] in {"review", "high"}:
        return COLORS["review"]
    return COLORS["unknown"]


def _combined_status_color(*comparisons: dict[str, Any]) -> tuple[float, float, float]:
    if any(item["severity"] in {"review", "high"} for item in comparisons):
        return COLORS["review"]
    if comparisons and all(item["severity"] == "pass" for item in comparisons):
        return COLORS["pass"]
    return COLORS["unknown"]


def _declaration_boxes(page_record: dict[str, Any], language: str) -> list[tuple[fitz.Rect, str, str]]:
    boxes = []
    declared = page_record["declared_by_designer"]
    for item in declared["concrete"]:
        for index, rect in enumerate(_exact_declaration_boxes(item)):
            value = quantity_text(language, item["value"], "m3", 4)
            label = choose(language, f"DECLARED CONCRETE {value}", f"БЕТОН ПО ВЕДОМОСТИ {value}") if index == 0 else ""
            boxes.append((rect, label, "concrete"))
    for item in declared["reinforcement"]:
        for index, rect in enumerate(_exact_declaration_boxes(item)):
            value = quantity_text(language, item["value"], "kg", 1)
            label = choose(language, f"DECLARED STEEL {value}", f"СТАЛЬ ПО ВЕДОМОСТИ {value}") if index == 0 else ""
            boxes.append((rect, label, "steel"))
    return boxes


def _exact_declaration_boxes(item: dict[str, Any]) -> list[fitz.Rect]:
    """Return only evidence localized to exact value cells.

    A broad schedule row or search region is useful provenance, but must never
    be rendered with the same treatment as an exact declared-total cell.
    """

    boxes = []
    for key in ("total_cell_bbox_display", "cell_bbox_display"):
        if item.get(key):
            boxes.append(fitz.Rect(item[key]))
    for cell in item.get("evidence_cells", []):
        if cell.get("bbox_display"):
            boxes.append(fitz.Rect(cell["bbox_display"]))
    boxes.extend(fitz.Rect(box) for box in item.get("cell_bboxes_display", []))
    component_cells_are_exact = str(item.get("basis", "")).endswith("_component_cell_ocr_sum")
    for component in item.get("components", []):
        if component.get("total_cell_bbox_display"):
            boxes.append(fitz.Rect(component["total_cell_bbox_display"]))
        elif component.get("cell_bbox_display"):
            boxes.append(fitz.Rect(component["cell_bbox_display"]))
        elif component_cells_are_exact and component.get("bbox_display"):
            boxes.append(fitz.Rect(component["bbox_display"]))
        elif component.get("label") in {"overall", "total", "declared_total", "class_subtotal"} and component.get("bbox_display"):
            boxes.append(fitz.Rect(component["bbox_display"]))
    return list({tuple(round(value, 2) for value in box): box for box in boxes if not box.is_empty}.values())


def draw_overlay(
    output: fitz.Document,
    display: fitz.Document,
    source_page: fitz.Page,
    page_record: dict[str, Any],
    index: int,
    ocgs: dict[str, int],
    frozen: dict[str, Any],
    language: str,
    understanding_seconds: float | None,
) -> None:
    sidebar = 430
    page = output.new_page(width=source_page.rect.width + sidebar, height=source_page.rect.height)
    page.show_pdf_page(source_page.rect, display, index, oc=ocgs["SOURCE"])
    comparisons = {
        "concrete": page_record["comparison"]["concrete"],
        "steel": page_record["comparison"]["reinforcement_mass"],
    }
    concrete_comparison = comparisons["concrete"]
    for rect, label, kind in _declaration_boxes(page_record, language):
        page.draw_rect(rect + (-3, -3, 3, 3), color=COLORS["declared"], width=2.2, fill=COLORS["declared"], fill_opacity=0.07, oc=ocgs["DECLARED_SCHEDULE"])
        if label:
            tag = fitz.Rect(rect.x0, max(0, rect.y0 - 18), min(source_page.rect.x1, rect.x0 + 220), rect.y0)
            insert_text(page, tag, label, 7.2, COLORS["declared"], bold=True)
        if comparisons[kind]["status"] == "discrepancy":
            page.draw_rect(rect + (-7, -7, 7, 7), color=COLORS["review"], width=2.5, dashes="[8 4]", oc=ocgs["DISCREPANCIES"])

    x0 = source_page.rect.width
    page.draw_rect(fitz.Rect(x0, 0, page.rect.x1, page.rect.y1), color=COLORS["paper"], fill=COLORS["paper"], width=0)
    margin = 26
    insert_text(page, fitz.Rect(x0 + margin, 26, page.rect.x1 - margin, 92), choose(language, "QUANTITY AND MASS\nCOMPARISON", "ПРОВЕРКА\nОБЪЕМОВ И МАССЫ"), 20, COLORS["ink"], bold=True)
    note = choose(
        language,
        "Drawing quantities were calculated before the schedule was read. Highlighted schedule cells are comparison evidence only.",
        "Количества по чертежу рассчитаны до чтения ведомости. Выделенные ячейки используются только для сравнения.",
    )
    insert_text(page, fitz.Rect(x0 + margin, 110, page.rect.x1 - margin, 190), f"{note}\n{duration_text(language, understanding_seconds)}", 9.5, COLORS["muted"], bold=True)
    y = 210
    cards = [
        (choose(language, "AUDIT ID", "КОД ПРОВЕРКИ"), frozen["sha256"][:16], COLORS["calculated"]),
        (choose(language, "CONCRETE - CALCULATED", "БЕТОН - РАСЧЕТ"), _fmt(concrete_comparison["calculated"], "m3", language), COLORS["calculated"]),
        (choose(language, "CONCRETE - DECLARED", "БЕТОН - ВЕДОМОСТЬ"), _fmt(concrete_comparison["declared"], "m3", language), COLORS["declared"]),
        (choose(language, "CONCRETE - DRAWING MINUS SCHEDULE", "БЕТОН - ЧЕРТЕЖ МИНУС ВЕДОМОСТЬ"), _fmt_delta(concrete_comparison["delta"], "m3", language), _status_color(concrete_comparison)),
    ]
    steel = page_record["comparison"]["reinforcement_mass"]
    centerline = page_record["calculated_from_drawing"].get("reinforcement_centerline_m")
    fabrication = page_record["calculated_from_drawing"].get("reinforcement_fabrication_length_m")
    cards.extend(
        [
            (choose(language, "REBAR - FABRICATION LENGTH", "АРМАТУРА - ЗАГОТОВИТЕЛЬНАЯ ДЛИНА") if fabrication is not None else choose(language, "REBAR - PLACED CENTERLINE (PARTIAL)", "АРМАТУРА - ОСИ В КОНСТРУКЦИИ (ЧАСТИЧНО)"), _fmt(fabrication if fabrication is not None else centerline, "m", language), COLORS["calculated"]),
            (choose(language, "STEEL - CALCULATED", "СТАЛЬ - РАСЧЕТ"), _fmt(steel["calculated"], "kg", language), COLORS["calculated"]),
            (choose(language, "STEEL - DECLARED", "СТАЛЬ - ВЕДОМОСТЬ"), _fmt(steel["declared"], "kg", language), COLORS["declared"]),
            (choose(language, "STEEL - DRAWING MINUS SCHEDULE", "СТАЛЬ - ЧЕРТЕЖ МИНУС ВЕДОМОСТЬ"), _fmt_delta(steel["delta"], "kg", language), _status_color(steel)),
        ]
    )
    status_height = min(280.0, max(240.0, page.rect.height * 0.22))
    status_y = page.rect.height - 25 - status_height
    card_step = (status_y - 12 - y) / len(cards)
    card_height = min(64.0, max(46.0, card_step - 7))
    for label, value, color in cards:
        page.draw_rect(fitz.Rect(x0 + margin, y, page.rect.x1 - margin, y + card_height), color=color, fill=COLORS["white"], width=1.2)
        insert_text(page, fitz.Rect(x0 + margin + 13, y + 7, page.rect.x1 - margin - 13, y + min(25, card_height * 0.45)), label, 7.5, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x0 + margin + 13, y + card_height * 0.50, page.rect.x1 - margin - 13, y + card_height - 5), value, 11.5, COLORS["ink"], bold=True)
        y += card_step
    status_items = (("concrete", concrete_comparison), ("steel", steel))
    box_height = 94.0
    for status_index, (kind, comparison) in enumerate(status_items):
        top = status_y + status_index * (box_height + 8)
        status_color = _status_color(comparison)
        page.draw_rect(fitz.Rect(x0 + margin, top, page.rect.x1 - margin, top + box_height), color=status_color, fill=COLORS["white"], width=1.5)
        heading = choose(language, "CONCRETE", "БЕТОН") if kind == "concrete" else choose(language, "STEEL", "СТАЛЬ")
        unit = "m3" if kind == "concrete" else "kg"
        delta_text = _fmt_delta(comparison["delta"], unit, language) if comparison["delta"] is not None else choose(language, "NOT COMPARABLE", "НЕЛЬЗЯ СРАВНИТЬ")
        insert_text(page, fitz.Rect(x0 + margin + 12, top + 9, page.rect.x1 - margin - 12, top + 31), f"{heading}: {term(language, comparison['status']).upper()} / {term(language, comparison['severity']).upper()} | {delta_text}", 8.8, status_color, bold=True)
        insert_text(page, fitz.Rect(x0 + margin + 12, top + 38, page.rect.x1 - margin - 12, top + box_height - 8), review_action(language, kind, comparison["status"]), 7.7, COLORS["ink"], bold=True)
    quote_y = status_y + 2 * (box_height + 8) + 3
    insert_text(
        page,
        fitz.Rect(x0 + margin, quote_y, page.rect.x1 - margin, page.rect.height - 25),
        choose(language, "QUOTE STATUS: NOT APPROVED - engineer review required", "СТАТУС КП: НЕ УТВЕРЖДЕНО - требуется проверка инженера"),
        8.3,
        COLORS["review"],
        bold=True,
    )


def draw_summary(output: fitz.Document, source: Path, comparison: dict[str, Any], language: str) -> None:
    page = output.new_page(width=1600, height=1000)
    page.draw_rect(page.rect, color=COLORS["paper"], fill=COLORS["paper"], width=0)
    page.draw_rect(fitz.Rect(0, 0, 1600, 120), color=COLORS["ink"], fill=COLORS["ink"], width=0)
    insert_text(page, fitz.Rect(48, 28, 1520, 70), choose(language, "DRAWING VS DESIGNER SCHEDULE", "ЧЕРТЕЖ И ВЕДОМОСТЬ ПРОЕКТИРОВЩИКА"), 24, COLORS["white"], bold=True)
    insert_text(page, fitz.Rect(48, 77, 1520, 106), f"{source.name} | {duration_text(language, comparison.get('timing', {}).get('drawing_understanding_seconds'))}", 10, COLORS["white"])
    page_record = comparison["pages"][0]
    concrete = page_record["comparison"]["concrete"]
    steel = page_record["comparison"]["reinforcement_mass"]
    cards = [
        (choose(language, "CALCULATED CONCRETE", "БЕТОН ПО РАСЧЕТУ"), _fmt(concrete["calculated"], "m3", language), COLORS["calculated"]),
        (choose(language, "DECLARED CONCRETE", "БЕТОН ПО ВЕДОМОСТИ"), _fmt(concrete["declared"], "m3", language), COLORS["declared"]),
        (choose(language, "DRAWING MINUS SCHEDULE", "ЧЕРТЕЖ МИНУС ВЕДОМОСТЬ"), _fmt_delta(concrete["delta"], "m3", language), _status_color(concrete)),
        (choose(language, "DECLARED STEEL", "СТАЛЬ ПО ВЕДОМОСТИ"), _fmt(steel["declared"], "kg", language), COLORS["declared"]),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 48 + index * 385
        page.draw_rect(fitz.Rect(x, 160, x + 350, 300), color=color, fill=COLORS["white"], width=1.5)
        insert_text(page, fitz.Rect(x + 20, 184, x + 330, 220), label, 9, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x + 20, 235, x + 330, 280), value, 22, COLORS["ink"], bold=True)
    insert_text(page, fitz.Rect(48, 350, 760, 390), choose(language, "PROCESSING SEQUENCE", "ПОСЛЕДОВАТЕЛЬНОСТЬ ОБРАБОТКИ"), 15, COLORS["ink"], bold=True)
    sequence_labels = {
        "read_and_hash_calculated_engineering_graph": choose(language, "lock the drawing calculation", "зафиксировать расчет по чертежу"),
        "extract_calculated_quantities": choose(language, "record quantities calculated from the drawing", "записать количества, рассчитанные по чертежу"),
        "open_pdf_and_extract_declared_schedules": choose(language, "read the designer's schedule totals independently", "независимо прочитать итоги ведомости"),
        "compare_without_feedback": choose(language, "flag differences for engineer review", "передать разницы на проверку инженеру"),
    }
    sequence = "\n".join(f"{index}. {sequence_labels.get(step, step.replace('_', ' '))}" for index, step in enumerate(comparison["processing_sequence"], start=1))
    insert_text(page, fitz.Rect(48, 410, 760, 620), sequence, 12, COLORS["ink"])
    insert_text(page, fitz.Rect(820, 350, 1520, 390), choose(language, "REVIEW STATUS", "СТАТУС ПРОВЕРКИ"), 15, _combined_status_color(concrete, steel), bold=True)
    detail = choose(
        language,
        f"Concrete: {term(language, concrete['status'])} / {term(language, concrete['severity'])}\nDrawing - schedule: {_fmt_delta(concrete['delta'], 'm3', language)} ({_percent_text(concrete['delta_percent_of_declared'], language)})\nAction: {review_action(language, 'concrete', concrete['status'])}\n\nSteel: {term(language, steel['status'])} / {term(language, steel['severity'])}\nDrawing - schedule: {_fmt_delta(steel['delta'], 'kg', language)} ({_percent_text(steel['delta_percent_of_declared'], language)})\nAction: {review_action(language, 'steel', steel['status'])}",
        f"Бетон: {term(language, concrete['status'])} / {term(language, concrete['severity'])}\nЧертеж - ведомость: {_fmt_delta(concrete['delta'], 'm3', language)} ({_percent_text(concrete['delta_percent_of_declared'], language)})\nДействие: {review_action(language, 'concrete', concrete['status'])}\n\nСталь: {term(language, steel['status'])} / {term(language, steel['severity'])}\nЧертеж - ведомость: {_fmt_delta(steel['delta'], 'kg', language)} ({_percent_text(steel['delta_percent_of_declared'], language)})\nДействие: {review_action(language, 'steel', steel['status'])}",
    )
    insert_text(page, fitz.Rect(820, 410, 1520, 650), detail, 12, COLORS["ink"])
    frozen = comparison["frozen_calculation"]
    boundary = choose(
        language,
        f"REVIEW CONTROL\nAudit ID: {frozen['sha256']}\nDrawing values, schedule values and approved values remain separate. Unknown is never converted to zero. Quote status remains not approved until an engineer confirms both checks.",
        f"КОНТРОЛЬ ПРОВЕРКИ\nКод проверки: {frozen['sha256']}\nЗначения по чертежу, ведомости и утвержденные значения хранятся отдельно. Неизвестное не заменяется нулем. КП не утверждается до проверки обоих разделов инженером.",
    )
    insert_text(page, fitz.Rect(48, 730, 1520, 910), boundary, 13, COLORS["ink"], bold=True)


def build_audit(source: Path, comparison_path: Path, output_pdf: Path, language: str = DEFAULT_LANGUAGE) -> Path:
    language = validate_language(language)
    comparison = json.loads(comparison_path.read_text())
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    binding = comparison.get("result_binding", {})
    if binding.get("source_pdf_sha256") != source_sha256:
        raise ValueError("comparison result binding does not match the audit source PDF")
    if binding.get("engineering_graph_artifact_sha256") != comparison.get("frozen_calculation", {}).get("sha256"):
        raise ValueError("comparison result binding does not match its frozen engineering graph")
    understanding_seconds = comparison.get("timing", {}).get("drawing_understanding_seconds")
    raw = fitz.open(source)
    display = fitz.open()
    display.insert_pdf(raw)
    for page in display:
        page.remove_rotation()
    output = fitz.open()
    ocgs = {
        "SOURCE": output.add_ocg("SOURCE", on=1),
        "DECLARED_SCHEDULE": output.add_ocg("DECLARED_SCHEDULE", on=1),
        "DISCREPANCIES": output.add_ocg("DISCREPANCIES", on=1),
    }
    for index, source_page in enumerate(display):
        reference = output.new_page(width=source_page.rect.width, height=source_page.rect.height)
        reference.show_pdf_page(reference.rect, display, index)
        record = next(item for item in comparison["pages"] if item["page"] == index + 1)
        draw_overlay(output, display, source_page, record, index, ocgs, comparison["frozen_calculation"], language, understanding_seconds)
    draw_summary(output, source, comparison, language)
    output.set_metadata({
        "title": choose(language, f"{source.name} - drawing versus designer schedule", f"{source.name} - сравнение чертежа и ведомости"),
        "author": choose(language, "Drawing-first estimate comparison pipeline", "Конвейер сравнения с приоритетом чертежа"),
        "subject": choose(language, "Frozen drawing calculation, independent declared schedule, discrepancy, and engineer approval boundary", "Зафиксированный расчет, независимая ведомость, расхождения и граница утверждения инженером"),
        "creator": "PyMuPDF",
    })
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_pdf, garbage=4, deflate=True, clean=True)
    output.close()
    display.close()
    raw.close()
    manifest = {
        "schema_version": "0.1.0",
        "source_pdf": str(source.resolve()),
        "comparison_json": str(comparison_path.resolve()),
        "output_pdf": str(output_pdf.resolve()),
        "presentation_language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_binding": binding,
        "timing": comparison.get("timing", {}),
        "layers": list(ocgs),
        "page_structure": [*(item for index in range(len(comparison["pages"])) for item in (f"source_{index + 1}", f"comparison_overlay_{index + 1}")), "comparison_summary"],
        "frozen_graph_sha256": comparison["frozen_calculation"]["sha256"],
    }
    output_pdf.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--comparison-dir", type=Path, default=ROOT / "output" / "estimates")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "pdf")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE)
    args = parser.parse_args()
    for source in args.inputs:
        comparison = args.comparison_dir / f"{source.stem}.estimate-comparison.json"
        if not comparison.exists():
            parser.error(f"comparison record not found: {comparison}")
        output = args.output_dir / f"{source.stem}_estimate_comparison_audit.pdf"
        print(build_audit(source, comparison, output, language=args.language))


if __name__ == "__main__":
    main()
