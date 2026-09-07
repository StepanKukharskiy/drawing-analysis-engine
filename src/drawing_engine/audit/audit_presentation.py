"""Shared language and Unicode-font support for human audit artifacts."""

from __future__ import annotations

import colorsys
from functools import lru_cache
from pathlib import Path


DEFAULT_LANGUAGE = "ru"
SUPPORTED_LANGUAGES = ("ru", "en")


def item_palette_color(index: int) -> tuple[float, float, float]:
    """The shared rebar-detail / item palette; color conveys identity, not truth."""
    hue = (0.07 + index * 0.618033988749895) % 1.0
    return tuple(round(value, 6) for value in colorsys.hsv_to_rgb(hue, 0.72, 0.74))


def validate_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported audit language: {language!r}")
    return language


def choose(language: str, english: str, russian: str) -> str:
    return english if validate_language(language) == "en" else russian


def term(language: str, value: object) -> str:
    text = str(value)
    if language == "en":
        return text
    return {
        "accepted": "принято",
        "calculation_unavailable": "расчет недоступен",
        "declaration_unavailable": "декларация недоступна",
        "derived": "вычислено",
        "discrepancy": "расхождение",
        "high": "высокая важность",
        "match": "совпадение",
        "partial": "частично",
        "pass": "норма",
        "pending": "ожидает проверки",
        "resolved": "решено",
        "review": "проверить",
        "unknown": "неизвестно",
        "unresolved": "не решено",
    }.get(text, text)


def duration_text(language: str, seconds: float | None) -> str:
    english_value = "not recorded" if seconds is None else f"{seconds:.2f} s"
    russian_value = "не зафиксировано" if seconds is None else f"{seconds:.2f} с"
    return choose(language, f"Drawing understanding time: {english_value}", f"Время анализа чертежа: {russian_value}")


def unit_text(language: str, unit: str) -> str:
    """Return the presentation unit without exposing storage conventions."""

    units = {
        "en": {"m3": "m³", "m2": "m²", "m": "m", "mm": "mm", "kg": "kg"},
        "ru": {"m3": "м³", "m2": "м²", "m": "м", "mm": "мм", "kg": "кг"},
    }
    validate_language(language)
    return units[language].get(unit, unit)


def quantity_text(
    language: str,
    value: float | None,
    unit: str,
    digits: int = 3,
    *,
    signed: bool = False,
) -> str:
    """Format an audit quantity, preserving unknown rather than inventing zero."""

    if value is None:
        return choose(language, "UNKNOWN", "НЕ ОПРЕДЕЛЕНО")
    number = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f"{number} {unit_text(language, unit)}"


def review_action(language: str, quantity: str, status: object) -> str:
    """Translate a comparison state into the next engineering action."""

    status_text = str(status or "unknown")
    actions = {
        "concrete": {
            "match": (
                "No quantity difference detected; engineer confirmation is still required.",
                "Разница не выявлена; перед КП требуется подтверждение инженера.",
            ),
            "pass": (
                "No quantity difference detected; engineer confirmation is still required.",
                "Разница не выявлена; перед КП требуется подтверждение инженера.",
            ),
            "discrepancy": (
                "Verify openings, recesses and embedded-item deductions before quotation.",
                "Проверить проемы, углубления и вычет закладных деталей до подготовки КП.",
            ),
            "calculation_unavailable": (
                "Complete the concrete geometry before comparing or quoting this quantity.",
                "Завершить геометрию бетона до сравнения и подготовки КП.",
            ),
            "declaration_unavailable": (
                "Locate and confirm the designer's concrete total.",
                "Найти и подтвердить итоговый объем бетона в ведомости.",
            ),
        },
        "steel": {
            "match": (
                "No mass difference detected; engineer confirmation is still required.",
                "Разница по массе не выявлена; перед КП требуется подтверждение инженера.",
            ),
            "pass": (
                "No mass difference detected; engineer confirmation is still required.",
                "Разница по массе не выявлена; перед КП требуется подтверждение инженера.",
            ),
            "discrepancy": (
                "Verify bar counts, diameters and cutting lengths before quotation.",
                "Проверить количество, диаметры и заготовительные длины стержней до подготовки КП.",
            ),
            "calculation_unavailable": (
                "Resolve bar counts, diameters and cutting lengths before comparing steel mass.",
                "Определить количество, диаметры и заготовительные длины до сравнения массы стали.",
            ),
            "declaration_unavailable": (
                "Locate and confirm the designer's steel total.",
                "Найти и подтвердить итоговую массу стали в ведомости.",
            ),
        },
    }
    english, russian = actions.get(quantity, {}).get(
        status_text,
        (
            "Review the highlighted evidence before quotation.",
            "Проверить выделенные данные до подготовки КП.",
        ),
    )
    return choose(language, english, russian)


def solver_text(language: str, value: object) -> str:
    text = str(value or "unknown")
    if language == "en":
        return text
    return {
        "cross_view_semantic": "семантическая реконструкция по видам",
        "generic_cross_view_prismatic": "универсальная призматическая реконструкция по видам",
        "generic_dimensioned_profile_extrusion": "универсальная экструзия размерного профиля",
        "generic_prismatic": "универсальная призматическая реконструкция",
        "generic_profile_extrusion": "универсальная экструзия профиля",
        "object_instance_profile_extrusions": "экструзии профилей отдельных объектов",
        "procedural_object_instance_profile_extrusions": "процедурные экструзии профилей отдельных объектов",
        "strict_procedural": "строгий процедурный решатель",
        "unknown": "неизвестно",
    }.get(text, text)


def reason_text(language: str, value: object) -> str:
    text = str(value or "unknown")
    actions = {
        "at least two section proposals are required": (
            "Confirm a second dimensioned view of the same item.",
            "Подтвердить второй размерный вид этого же изделия.",
        ),
        "no cross-view thin-prism closure": (
            "Confirm which two views show the same thin item and confirm its thickness.",
            "Подтвердить, какие два вида показывают одно тонкостенное изделие, и уточнить его толщину.",
        ),
        "no dimensioned cross-view profile extrusion closure": (
            "Confirm the dimensioned profile and the view that provides its depth.",
            "Подтвердить размерный профиль и вид, задающий его глубину.",
        ),
        "no procedural object-instance pairs for scoped extrusion": (
            "Confirm which views belong to each physical item.",
            "Подтвердить, какие виды относятся к каждому физическому изделию.",
        ),
        "cross-view correspondence is not unique": (
            "Choose which highlighted views belong to the same physical item.",
            "Указать, какие выделенные виды относятся к одному изделию.",
        ),
        "solid topology is not emitted without a closed constraint system": (
            "Confirm all dimensions and view links needed to bound the concrete volume.",
            "Подтвердить все размеры и связи видов, необходимые для определения объема бетона.",
        ),
        "remaining object views are not yet fused": (
            "Confirm how the remaining views relate to the current model.",
            "Подтвердить, как оставшиеся виды относятся к текущей модели.",
        ),
        "non-primary reinforcement topology": (
            "Confirm the highlighted reinforcement links between views.",
            "Подтвердить выделенные связи арматуры между видами.",
        ),
        "gated_unavailable": (
            "Required view or dimension evidence is still missing.",
            "Не хватает необходимого вида или размера.",
        ),
        "resolved": (
            "No additional geometry confirmation is required.",
            "Дополнительное подтверждение геометрии не требуется.",
        ),
        "unknown": (
            "Additional drawing evidence needs engineer confirmation.",
            "Требуется подтверждение дополнительных данных чертежа.",
        ),
    }
    english, russian = actions.get(
        text,
        (
            "Additional drawing evidence needs engineer confirmation.",
            "Требуется подтверждение дополнительных данных чертежа.",
        ),
    )
    return choose(language, english, russian)


@lru_cache(maxsize=2)
def audit_font_file(bold: bool = False) -> str:
    names = (
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
        if bold
        else ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    )
    portable = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in (*names, *portable):
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("a Unicode sans-serif audit font was not found")
