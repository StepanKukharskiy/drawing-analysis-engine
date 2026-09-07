"""Versioned, proposal-only MEP terminology and graphical interpretation.

M2 translates text, legend entries, and already-observed graphical signatures
into neutral candidates.  It does not bind a candidate to a route or physical
component.  In particular, legend membership, colour, equal text, and symbol
proximity cannot establish system identity, a connection, an elevation, a
clash, or a quantity here.

Inputs are deliberately small evidence records rather than PDF pages.  Native
and raster extractors can therefore share this adapter while retaining their
own immutable primitive IDs and coordinate provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_terminology_and_symbol_proposals"
TERMINOLOGY_PACK_ID = "mep-core-en"
TERMINOLOGY_PACK_VERSION = "0.1.0"
METHOD_VERSION = "1.0.0"

_AUTHORITY_FLAGS = (
    "route_identity_established",
    "system_identity_established",
    "physical_connection_established",
    "physical_continuation_established",
    "elevation_established",
    "clash_established",
    "installed_length_emitted",
    "quantity_eligible",
)
_PROPOSAL_STATES = {"proposed", "abstained"}
_EPISTEMIC_STATES = {"observed", "derived", "inferred", "convention_dependent", "unknown"}


def _entry(
    entry_id: str,
    proposal_type: str,
    canonical_kind: str,
    terms: Iterable[str],
    *,
    category: str | None = None,
    ambiguous: bool = False,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "proposal_type": proposal_type,
        "canonical_kind": canonical_kind,
        "category": category,
        "terms": list(terms),
        "ambiguous_without_context": ambiguous,
    }


# This is interpretation vocabulary, not a global ontology.  Updating it is an
# explicit version change so stored proposals remain replayable.
_TERMINOLOGY_ENTRIES = (
    _entry("system.chws", "system", "chilled_water_supply", ("CHWS", "CHW SUPPLY")),
    _entry("system.chwr", "system", "chilled_water_return", ("CHWR", "CHW RETURN")),
    _entry("system.hws", "system", "heating_hot_water_supply", ("HWS", "HHWS")),
    _entry("system.hwr", "system", "heating_hot_water_return", ("HWR", "HHWR")),
    _entry("system.cws", "system", "condenser_water_supply", ("CWS",)),
    _entry("system.cwr", "system", "condenser_water_return", ("CWR",)),
    _entry("system.lube", "system", "lube_fluid", ("LUBE", "LUBE OIL", "LO")),
    _entry("system.vehicle_exhaust", "system", "vehicle_exhaust", ("VE", "VEH EXH", "VEHICLE EXHAUST")),
    _entry("valve.ball", "component", "ball_valve", ("BV", "BALL VALVE"), category="valve"),
    _entry("valve.gate", "component", "gate_valve", ("GV", "GATE VALVE"), category="valve"),
    _entry("valve.check", "component", "check_valve", ("CV", "CHECK VALVE"), category="valve"),
    _entry("valve.butterfly", "component", "butterfly_valve", ("BFV", "BUTTERFLY VALVE"), category="valve"),
    _entry("fitting.elbow", "component", "elbow", ("ELBOW", "ELL"), category="fitting"),
    _entry("fitting.tee", "component", "tee", ("TEE",), category="fitting"),
    _entry("fitting.reducer", "component", "reducer", ("REDUCER", "RED"), category="fitting"),
    _entry("damper.fire", "component", "fire_damper", ("FIRE DAMPER", "FD"), category="damper", ambiguous=True),
    _entry("other.floor_drain", "component", "floor_drain", ("FLOOR DRAIN", "FD"), category="drain", ambiguous=True),
    _entry("zone.service", "service_zone", "service_access_zone", ("SERVICE ZONE", "ACCESS ZONE", "SERVICE CLEARANCE")),
    _entry("continuation.matchline", "continuation", "match_line_continuation", ("MATCH LINE", "MATCHLINE", "CONTINUATION")),
    _entry("vertical.riser", "vertical_transition", "riser", ("RISE", "RISER", "UP")),
    _entry("vertical.drop", "vertical_transition", "drop", ("DROP", "DOWN")),
)

_PACK_BY_TERM: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
for _terminology_entry in _TERMINOLOGY_ENTRIES:
    for _term in _terminology_entry["terms"]:
        _PACK_BY_TERM[_term].append(_terminology_entry)

_PIPE_SIZE = re.compile(
    r"(?<![A-Z0-9./-])(?:DN\s*(?P<dn>\d+(?:\.\d+)?)|"
    r"(?:[Ø⌀]\s*|SIZE\s+)(?P<diameter>\d+(?:\.\d+)?)\s*(?P<diameter_unit>MM|IN|\"|INCH(?:ES)?)?|"
    r"(?P<inch>\d+(?:\.\d+|\s+\d+\s*/\s*\d+|\s*/\s*\d+)?)\s*(?:\"|IN(?:CH(?:ES)?)?\b))",
    re.I,
)
_DUCT_SIZE = re.compile(
    r"(?<![A-Z0-9])(?P<width>\d+(?:\.\d+)?)\s*[X×]\s*"
    r"(?P<height>\d+(?:\.\d+)?)\s*(?P<unit>MM|IN|\"|INCH(?:ES)?)?",
    re.I,
)
_DUCT_CALLOUT = re.compile(
    r"(?<![A-Z0-9./])(?P<width>\d+(?:\.\d+)?)\s*(?P<separator>[/X×])\s*"
    r"(?P<height>\d+(?:\.\d+)?)\s*(?P<unit>MM|IN|\")?\s+"
    r"(?P<system>SA|RA|OA|EA|EXH)\b", re.I,
)
_ELEVATION = re.compile(
    r"(?<![A-Z0-9])(?P<datum>BOP|BOD|BOT|BOTTOM|BE|CL|C/L|C\.L\.|CENTRELINE|CENTERLINE|EL|ELEV)"
    r"\s*(?:EL(?:EV(?:ATION)?)?\.?\s*)?[:=@]?\s*"
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>MM|M|FT|FEET|'|IN|\"|INCH(?:ES)?)?",
    re.I,
)
_ARCHITECTURAL_ELEVATION = re.compile(
    r"(?<![A-Z0-9])(?P<datum>BOP|BOD|BOT|BOTTOM|BE|CL|C/L|C\.L\.|CENTRELINE|CENTERLINE|EL|ELEV)"
    r"\s*(?:EL(?:EV(?:ATION)?)?\.?\s*)?[:=@]?\s*"
    r"(?P<feet>[+-]?\d+)\s*'\s*(?:-\s*)?"
    r"(?:(?P<fraction_numerator>\d+)\s*/\s*(?P<fraction_denominator>\d+)"
    r"|(?P<inches>\d+)(?:\s+(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+))?)?\s*\"?",
    re.I,
)
_EQUIPMENT_TAG = re.compile(
    r"(?<![A-Z0-9])(?P<prefix>AHU|FCU|EF|SF|RF|VFD|HUH|P|PUMP|FAN)"
    r"\s*[-_]\s*(?P<number>[A-Z0-9]+(?:[-_.][A-Z0-9]+)*)",
    re.I,
)

_ARCHITECTURAL_DISTANCE = re.compile(
    r"(?<![A-Z0-9./])(?P<feet>[+-]?\d+)\s*'\s*(?:-\s*)?"
    r"(?P<inches>\d+\s*/\s*\d+|\d+(?:\s+\d+\s*/\s*\d+)?)?\s*\"?",
    re.I,
)
_SCALE = re.compile(
    r"\b(?:SCALE\s*[:=]?|N\.?T\.?S\.?)\b|"
    r"\d+(?:\s+\d+\s*/\s*\d+|\s*/\s*\d+|\.\d+)?\s*\"\s*=\s*\d+\s*'|"
    r"^\s*1\s*:\s*\d+\s*$",
    re.I,
)
_NON_DRAWING_TEXT_ROLES = {"schedule", "table", "cut_sheet", "title", "note"}


def interpret_mep_text(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Separate lexical roles before any inch token becomes a size candidate.

    Offsets refer to the retained normalized text, never PDF coordinates. Region
    roles are evidence supplied by the observation adapter; lexical roles do not
    establish a region boundary, physical dimension, or target applicability.
    """

    text = _normalise_text(observation.get("text"))
    region_role = str(observation.get("region_role") or observation.get("context") or "unknown").lower()
    units = []
    architectural_elevations = list(_ARCHITECTURAL_ELEVATION.finditer(text))
    distances = list(_ARCHITECTURAL_DISTANCE.finditer(text))
    for match in distances:
        is_elevation = any(start.start() <= match.start() < start.end() for start in architectural_elevations)
        feet = float(match.group("feet"))
        inches = _number(match.group("inches")) if match.group("inches") else 0.0
        units.append({
            "role": "elevation" if is_elevation else "architectural_distance",
            "span_normalized": list(match.span()), "raw_text": match.group(0),
            "value": round(feet + (-1 if match.group("feet").startswith("-") else 1) * inches / 12, 8),
            "unit": "ft", "epistemic_state": "derived",
        })
    if region_role in _NON_DRAWING_TEXT_ROLES:
        role, proposal_text = region_role, ""
    elif _SCALE.search(text):
        role, proposal_text = "scale", ""
        for unit in units:
            unit["role"] = "scale_operand"
    else:
        # Mask entire distances, including positive inch tails. Elevations retain
        # their datum and are masked separately from size parsing below.
        masked = list(text)
        for unit in units:
            if unit["role"] == "architectural_distance":
                start, end = unit["span_normalized"]
                masked[start:end] = " " * (end - start)
        proposal_text = "".join(masked)
        if region_role == "legend" or observation.get("legend_membership"):
            role = "legend"
        elif architectural_elevations or _ELEVATION.search(text):
            role = "elevation"
        elif distances and not proposal_text.strip():
            role = "architectural_distance"
        elif _PIPE_SIZE.search(proposal_text) or _DUCT_SIZE.search(proposal_text):
            role = "nominal_size_candidate"
        else:
            role = "unknown"
    elevation_spans = [match.span() for match in _ELEVATION.finditer(text)]
    occupied = [tuple(unit["span_normalized"]) for unit in units]
    for match in _ELEVATION.finditer(text):
        if any(start <= match.start() < end or match.start() <= start < match.end() for start, end in occupied):
            continue
        units.append({
            "role": "elevation", "span_normalized": list(match.span()),
            "raw_text": match.group(0), "value": float(match.group("value")),
            "unit": _unit(match.group("unit")), "epistemic_state": "derived",
        })
    for match in _PIPE_SIZE.finditer(proposal_text):
        if any(start.start() < match.end() and match.start() < start.end()
               for pattern in (_DUCT_SIZE, _DUCT_CALLOUT) for start in pattern.finditer(proposal_text)):
            continue
        if any(start < match.end() and match.start() < end for start, end in [*occupied, *elevation_spans]):
            continue
        value = match.group("dn") or match.group("diameter") or match.group("inch")
        unit = "mm" if match.group("dn") else "in" if match.group("inch") else _unit(match.group("diameter_unit"))
        units.append({
            "role": "nominal_size_candidate", "span_normalized": list(match.span()),
            "raw_text": match.group(0), "value": _number(value), "unit": unit,
            "epistemic_state": "derived",
        })
    return {
        "method": "mep_text_role_unit_boundary", "version": "1.0.0",
        "normalized_text": text, "text_role": role, "region_role": region_role,
        "unit_interpretations": units, "proposal_text": proposal_text,
        "quantity_eligible": False,
    }

_SYMBOL_KINDS = {
    "equipment": ("equipment", "equipment"),
    "equipment_outline": ("equipment", "equipment"),
    "valve": ("component", "valve"),
    "gate_valve": ("component", "gate_valve"),
    "ball_valve": ("component", "ball_valve"),
    "check_valve": ("component", "check_valve"),
    "fitting": ("component", "fitting"),
    "elbow": ("component", "elbow"),
    "tee": ("component", "tee"),
    "reducer": ("component", "reducer"),
    "coupling": ("component", "coupling"),
    "pipe_coupling": ("component", "coupling"),
    "damper": ("component", "damper"),
    "fire_damper": ("component", "fire_damper"),
    "service_zone": ("service_zone", "service_access_zone"),
    "continuation": ("continuation", "continuation_symbol"),
    "match_line": ("continuation", "match_line_continuation"),
    "cap": ("terminal", "capped_physical_terminal"),
    "pipe_cap": ("terminal", "capped_physical_terminal"),
    "capped_terminal": ("terminal", "capped_physical_terminal"),
    "drawing_boundary": ("terminal", "drawing_scope_boundary"),
    "scope_boundary": ("terminal", "drawing_scope_boundary"),
    "detail_section_interface": ("terminal", "detail_section_interface"),
    "riser": ("vertical_transition", "riser"),
    "rise": ("vertical_transition", "riser"),
    "drop": ("vertical_transition", "drop"),
}


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def terminology_pack() -> dict[str, Any]:
    """Return a JSON-safe copy of the exact vocabulary used by this module."""

    return {
        "id": TERMINOLOGY_PACK_ID,
        "version": TERMINOLOGY_PACK_VERSION,
        "language": "en",
        "entries": [dict(item, terms=list(item["terms"])) for item in _TERMINOLOGY_ENTRIES],
        "contract": {
            "legend_membership_is_evidence_only": True,
            "colour_is_evidence_only": True,
            "full_pack_arbitration_deferred": True,
        },
    }


def _normalise_text(value: object) -> str:
    text = str(value or "").upper().replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.translate(str.maketrans({"′": "'", "’": "'", "″": '"', "“": '"', "”": '"'}))
    return " ".join(text.split())


def _normalise_channels(observation: Mapping[str, Any]) -> list[str]:
    channels = [str(item).lower() for item in observation.get("evidence_channels", []) if str(item)]
    if not channels:
        if observation.get("text"):
            channels.append("text")
        if observation.get("symbol_kind") or observation.get("symbol_candidates"):
            channels.append("symbol_geometry")
        if observation.get("color") is not None or observation.get("colour") is not None:
            channels.append("color")
    return sorted(set(channels))


def _number(value: str) -> float:
    text = re.sub(r"\s*/\s*", "/", " ".join(value.split()))
    if " " in text and "/" in text:
        whole, fraction = text.split(" ", 1)
        numerator, denominator = fraction.split("/", 1)
        return float(whole) + float(numerator) / float(denominator)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _unit(raw: object, *, default: str | None = None) -> str | None:
    value = str(raw or "").upper()
    if value == "MM":
        return "mm"
    if value == "M":
        return "m"
    if value in {"FT", "FEET", "'"}:
        return "ft"
    if value in {"IN", '"', "INCH", "INCHES"}:
        return "in"
    return default


def _elevation_basis(raw: object) -> tuple[str, bool]:
    datum = str(raw or "").upper().replace(".", "").replace("/", "")
    if datum in {"BOP", "BOD", "BOT", "BOTTOM", "BE"}:
        return "bottom", False
    if datum in {"CL", "CENTRELINE", "CENTERLINE"}:
        return "centreline", False
    return "unspecified", True


def _term_matches(text: str) -> list[dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for term, entries in _PACK_BY_TERM.items():
        if re.search(rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])", text):
            for item in entries:
                matches[item["id"]] = item
    return [matches[key] for key in sorted(matches)]


def _candidate_rows(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    interpretation = interpret_mep_text(observation)
    if interpretation["region_role"] in _NON_DRAWING_TEXT_ROLES:
        return []
    text = interpretation["proposal_text"]
    rows: list[dict[str, Any]] = []
    architectural_matches = list(_ARCHITECTURAL_ELEVATION.finditer(text))
    architectural_spans = [match.span() for match in architectural_matches]
    elevation_spans = [*architectural_spans, *(match.span() for match in _ELEVATION.finditer(text))]

    def inside_architectural_elevation(match: re.Match[str]) -> bool:
        return any(
            start < match.end() and match.start() < end
            for start, end in elevation_spans
        )

    for item in _term_matches(text):
        rows.append(
            {
                "proposal_type": item["proposal_type"],
                "candidate": {
                    "kind": item["canonical_kind"],
                    "category": item.get("category"),
                    "raw_text": text,
                    "terminology_entry_ref": item["id"],
                },
                "method_name": "versioned_exact_term_match",
                "epistemic_state": "inferred",
                "ambiguous_without_context": bool(item["ambiguous_without_context"]),
            }
        )

    # Slash notation is a duct size only with an explicit air-service suffix.
    # Unit and width/height orientation remain unresolved for later geometry
    # replay; an architectural fraction alone is never a duct annotation.
    for match in _DUCT_CALLOUT.finditer(text):
        if inside_architectural_elevation(match):
            continue
        rows.append({
            "proposal_type": "system",
            "candidate": {"kind": {"SA": "supply_air", "RA": "return_air",
                "OA": "outside_air", "EA": "exhaust_air", "EXH": "exhaust_air"}[
                    match.group("system").upper()], "raw_text": match.group(0)},
            "method_name": "bounded_air_service_size_callout_v1",
            "epistemic_state": "derived",
        })
        if match.group("separator") == "/":
            rows.append({
                "proposal_type": "inline_size",
                "candidate": {"kind": "rectangular_duct_size",
                    "width": float(match.group("width")), "height": float(match.group("height")),
                    "unit": _unit(match.group("unit")), "raw_text": match.group(0)},
                "method_name": "bounded_air_service_size_callout_v1",
                "epistemic_state": "derived",
            })

    for match in _DUCT_SIZE.finditer(text):
        if inside_architectural_elevation(match):
            continue
        rows.append(
            {
                "proposal_type": "inline_size",
                "candidate": {
                    "kind": "rectangular_duct_size",
                    "width": float(match.group("width")),
                    "height": float(match.group("height")),
                    "unit": _unit(match.group("unit"), default=None),
                    "raw_text": match.group(0),
                },
                "method_name": "bounded_inline_size_parser",
                "epistemic_state": "derived",
            }
        )
    for match in _PIPE_SIZE.finditer(text):
        if inside_architectural_elevation(match):
            continue
        if any(duct.start() < match.end() and match.start() < duct.end()
               for pattern in (_DUCT_SIZE, _DUCT_CALLOUT) for duct in pattern.finditer(text)):
            continue
        if match.group("dn") is not None:
            candidate = {
                "kind": "nominal_diameter",
                "value": float(match.group("dn")),
                "unit": "mm",
                "designation": "DN",
                "raw_text": match.group(0),
            }
        elif match.group("diameter") is not None:
            candidate = {
                "kind": "diameter",
                "value": float(match.group("diameter")),
                "unit": _unit(match.group("diameter_unit"), default=None),
                "designation": "diameter",
                "raw_text": match.group(0),
            }
        else:
            candidate = {
                "kind": "nominal_size",
                "value": _number(match.group("inch")),
                "unit": "in",
                "designation": "inch_size",
                "raw_text": match.group(0),
            }
        rows.append(
            {
                "proposal_type": "inline_size",
                "candidate": candidate,
                "method_name": "bounded_inline_size_parser",
                "epistemic_state": "derived",
            }
        )

    for match in architectural_matches:
        basis, ambiguous = _elevation_basis(match.group("datum"))
        feet = int(match.group("feet"))
        inches = int(match.group("inches") or 0)
        if match.group("numerator"):
            inches += int(match.group("numerator")) / int(match.group("denominator"))
        elif match.group("fraction_numerator"):
            inches += int(match.group("fraction_numerator")) / int(
                match.group("fraction_denominator")
            )
        sign = -1.0 if match.group("feet").startswith("-") else 1.0
        rows.append(
            {
                "proposal_type": "elevation",
                "candidate": {
                    "kind": "elevation",
                    "basis": basis,
                    "value": round(feet + sign * inches / 12.0, 8),
                    "unit": "ft",
                    "raw_text": match.group(0),
                },
                "method_name": "bounded_architectural_elevation_parser",
                "epistemic_state": "derived" if not ambiguous else "unknown",
                "ambiguous_without_context": ambiguous,
            }
        )

    for match in _ELEVATION.finditer(text):
        if any(start <= match.start() < end for start, end in architectural_spans):
            continue
        basis, ambiguous = _elevation_basis(match.group("datum"))
        rows.append(
            {
                "proposal_type": "elevation",
                "candidate": {
                    "kind": "elevation",
                    "basis": basis,
                    "value": float(match.group("value")),
                    "unit": _unit(match.group("unit"), default=None),
                    "raw_text": match.group(0),
                },
                "method_name": "bounded_elevation_parser",
                "epistemic_state": "derived" if not ambiguous else "unknown",
                "ambiguous_without_context": ambiguous,
            }
        )

    for match in _EQUIPMENT_TAG.finditer(text):
        prefix = match.group("prefix").upper()
        number = match.group("number").upper()
        rows.append(
            {
                "proposal_type": "equipment",
                "candidate": {
                    "kind": "equipment_tag",
                    "equipment_class_token": prefix,
                    "tag": f"{prefix}-{number}",
                    "raw_text": match.group(0),
                },
                "method_name": "bounded_equipment_tag_parser",
                "epistemic_state": "derived",
            }
        )

    raw_symbol_candidates = observation.get("symbol_candidates", [])
    if isinstance(raw_symbol_candidates, str):
        raw_symbol_candidates = [raw_symbol_candidates]
    symbol_candidates = [
        str(item).lower().strip().replace(" ", "_") for item in raw_symbol_candidates
    ]
    if observation.get("symbol_kind"):
        symbol_candidates.append(str(observation["symbol_kind"]).lower().strip().replace(" ", "_"))
    for symbol in sorted(set(symbol_candidates)):
        mapped = _SYMBOL_KINDS.get(symbol)
        if mapped is None:
            continue
        proposal_type, canonical_kind = mapped
        rows.append(
            {
                "proposal_type": proposal_type,
                "candidate": {
                    "kind": canonical_kind,
                    "observed_symbol_kind": symbol,
                },
                "method_name": "observed_symbol_signature_mapping",
                "epistemic_state": "inferred",
            }
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = json.dumps(
            [row["proposal_type"], row["candidate"]],
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _authority() -> dict[str, bool]:
    return {key: False for key in _AUTHORITY_FLAGS}


def propose_mep_interpretations(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create replayable neutral candidates from one evidence observation."""

    evidence_ref = str(observation.get("id") or "")
    if not evidence_ref:
        raise ValueError("observation.id is required")
    page_ref = str(observation.get("page_ref") or "")
    channels = _normalise_channels(observation)
    colour_only = bool(channels) and set(channels) <= {"color", "colour"}
    legend_membership = bool(observation.get("legend_membership")) or str(
        observation.get("region_role") or observation.get("context") or ""
    ).lower() == "legend"
    candidates = _candidate_rows(observation)
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[item["proposal_type"]].append(item)

    proposals: list[dict[str, Any]] = []
    for row in candidates:
        siblings = [
            item["candidate"]
            for item in by_type[row["proposal_type"]]
            if item["candidate"] != row["candidate"]
        ]
        reasons = []
        if legend_membership:
            reasons.append("legend_membership_is_evidence_only")
        if colour_only:
            reasons.append("color_only_evidence")
        if observation.get("region_role") == "unknown":
            reasons.append("region_applicability_unresolved")
        if row.get("ambiguous_without_context") or siblings:
            reasons.append("multiple_interpretations_require_scope_or_target_evidence")
        state = "abstained" if reasons else "proposed"
        anchor_ref = str(observation.get("anchor_ref") or evidence_ref)
        evidence_refs = sorted(
            {evidence_ref, *(str(item) for item in observation.get("evidence_refs", []))}
        )
        proposal_id = _stable_id(
            "mep_interpretation_proposal",
            TERMINOLOGY_PACK_VERSION,
            evidence_refs,
            row["proposal_type"],
            row["candidate"],
        )
        proposals.append(
            {
                "record_type": "mep_interpretation_proposal",
                "record_version": SCHEMA_VERSION,
                "id": proposal_id,
                "page_ref": page_ref,
                "anchor_ref": anchor_ref,
                "interpretation_scope_ref": str(
                    observation.get("interpretation_scope_ref") or anchor_ref
                ),
                "proposal_type": row["proposal_type"],
                "candidate": row["candidate"],
                "state": state,
                "epistemic_state": row["epistemic_state"],
                "method": {
                    "name": row["method_name"],
                    "version": METHOD_VERSION,
                    "terminology_pack_id": TERMINOLOGY_PACK_ID,
                    "terminology_pack_version": TERMINOLOGY_PACK_VERSION,
                },
                "evidence_refs": evidence_refs,
                "evidence_channels": channels,
                "legend_membership": legend_membership,
                "alternatives": siblings,
                "conflicts": [],
                "reasons": sorted(set(reasons)),
                "acceptance": _authority(),
            }
        )
    return sorted(proposals, key=lambda item: item["id"])


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    """Return semantic identity without label/matcher provenance spelling."""

    semantic = {
        key: value
        for key, value in candidate.items()
        if key not in {"raw_text", "observed_symbol_kind", "terminology_entry_ref"}
    }
    return json.dumps(semantic, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _walk_mappings(value: object, path: str = "proposal") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        yield path, value
        for key, child in value.items():
            yield from _walk_mappings(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_mappings(child, f"{path}[{index}]")


def _apply_scope_conflicts(proposals: list[dict[str, Any]]) -> None:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        groups[(proposal["interpretation_scope_ref"], proposal["proposal_type"])].append(proposal)
    for group in groups.values():
        identities = {_candidate_identity(item["candidate"]) for item in group}
        evidence_sets = {tuple(item["evidence_refs"]) for item in group}
        if len(identities) <= 1 or len(evidence_sets) <= 1:
            continue
        for proposal in group:
            conflicts = [
                {
                    "proposal_ref": other["id"],
                    "candidate": other["candidate"],
                    "evidence_refs": other["evidence_refs"],
                }
                for other in group
                if _candidate_identity(other["candidate"])
                != _candidate_identity(proposal["candidate"])
            ]
            proposal["conflicts"] = sorted(conflicts, key=lambda item: item["proposal_ref"])
            proposal["alternatives"] = sorted(
                {
                    _candidate_identity(item): item
                    for item in [*proposal["alternatives"], *(conflict["candidate"] for conflict in conflicts)]
                }.values(),
                key=_candidate_identity,
            )
            proposal["state"] = "abstained"
            proposal["reasons"] = sorted(
                set([*proposal["reasons"], "conflicting_evidence_in_interpretation_scope"])
            )


def build_mep_terminology_proposals(
    *,
    document: Mapping[str, Any],
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the complete isolated M2 exchange layer."""

    retained_observations = [dict(item) for item in observations]
    proposals = [
        proposal
        for observation in retained_observations
        for proposal in propose_mep_interpretations(observation)
    ]
    _apply_scope_conflicts(proposals)
    proposals.sort(key=lambda item: (item["page_ref"], item["proposal_type"], item["id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": dict(document),
        "terminology_pack": terminology_pack(),
        "source_observations": retained_observations,
        "proposals": proposals,
        "summary": {
            "source_observation_count": len(retained_observations),
            "proposal_count": len(proposals),
            "proposal_type_counts": dict(sorted(Counter(item["proposal_type"] for item in proposals).items())),
            "state_counts": dict(sorted(Counter(item["state"] for item in proposals).items())),
        },
        "exchange_contract": {
            "proposal_layer_only": True,
            "legend_membership_is_evidence_only": True,
            "color_only_evidence_must_abstain": True,
            "conflicts_must_abstain": True,
            "route_or_system_identity_established": False,
            "physical_connection_or_continuation_established": False,
            "physical_elevation_established": False,
            "confirmed_clash_established": False,
            "installed_length_emitted": False,
            "quantity_eligible": False,
        },
    }


def validate_mep_terminology_proposals(payload: Mapping[str, Any]) -> list[str]:
    """Return contract errors without changing or completing proposals."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    pack = payload.get("terminology_pack", {})
    if pack.get("id") != TERMINOLOGY_PACK_ID or pack.get("version") != TERMINOLOGY_PACK_VERSION:
        errors.append("terminology pack identity/version mismatch")
    observations = list(payload.get("source_observations", []))
    observation_refs = [str(item.get("id") or "") for item in observations]
    if "" in observation_refs or len(set(observation_refs)) != len(observation_refs):
        errors.append("source observation IDs must be present and unique")
    known_evidence = set(observation_refs)
    observations_by_ref = dict(zip(observation_refs, observations))
    proposals = list(payload.get("proposals", []))
    proposal_refs = [str(item.get("id") or "") for item in proposals]
    if "" in proposal_refs or len(set(proposal_refs)) != len(proposal_refs):
        errors.append("proposal IDs must be present and unique")
    known_proposals = set(proposal_refs)

    forbidden = {
        "accepted_route_ref",
        "accepted_system_ref",
        "accepted_target_ref",
        "physical_continuation_ref",
        "installed_length",
        "quantity",
        "confirmed_clash",
    }
    for proposal in proposals:
        prefix = str(proposal.get("id") or "proposal")
        if proposal.get("record_type") != "mep_interpretation_proposal":
            errors.append(f"{prefix}: invalid record_type")
        if proposal.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}: invalid record_version")
        if proposal.get("state") not in _PROPOSAL_STATES:
            errors.append(f"{prefix}: invalid proposal state")
        if proposal.get("epistemic_state") not in _EPISTEMIC_STATES:
            errors.append(f"{prefix}: invalid epistemic state")
        for path, mapping in _walk_mappings(proposal, prefix):
            forbidden_here = forbidden.intersection(mapping)
            if forbidden_here:
                errors.append(
                    f"{path}: forbidden engineering authority field(s) "
                    f"{', '.join(sorted(forbidden_here))}"
                )
            elevated_here = [
                key
                for key in _AUTHORITY_FLAGS
                if key in mapping and mapping.get(key) is not False
            ]
            if elevated_here:
                errors.append(
                    f"{path}: engineering authority flag(s) must remain false: "
                    f"{', '.join(sorted(elevated_here))}"
                )
        method = proposal.get("method", {})
        if not method.get("name") or method.get("version") != METHOD_VERSION:
            errors.append(f"{prefix}: missing method/version provenance")
        if method.get("terminology_pack_id") != TERMINOLOGY_PACK_ID or method.get(
            "terminology_pack_version"
        ) != TERMINOLOGY_PACK_VERSION:
            errors.append(f"{prefix}: wrong terminology provenance")
        evidence_refs = [str(item) for item in proposal.get("evidence_refs", [])]
        if not evidence_refs or any(item not in known_evidence for item in evidence_refs):
            errors.append(f"{prefix}: evidence refs are not closed")
        for ref in evidence_refs:
            source = observations_by_ref.get(ref, {})
            role = str(source.get("region_role") or source.get("context") or "").lower()
            if role in _NON_DRAWING_TEXT_ROLES:
                errors.append(f"{prefix}: non-drawing text role cannot supply proposals")
            if role == "unknown" and proposal.get("state") != "abstained":
                errors.append(f"{prefix}: unresolved region applicability did not abstain")
        authority = proposal.get("acceptance", {})
        if set(authority) != set(_AUTHORITY_FLAGS) or any(authority.get(key) is not False for key in _AUTHORITY_FLAGS):
            errors.append(f"{prefix}: proposal acquired engineering authority")
        channels = {str(item).lower() for item in proposal.get("evidence_channels", [])}
        if channels and channels <= {"color", "colour"} and proposal.get("state") != "abstained":
            errors.append(f"{prefix}: color-only proposal did not abstain")
        if proposal.get("legend_membership") and proposal.get("state") != "abstained":
            errors.append(f"{prefix}: legend proposal did not abstain")
        conflicts = list(proposal.get("conflicts", []))
        if conflicts and proposal.get("state") != "abstained":
            errors.append(f"{prefix}: conflicting proposal did not abstain")
        for conflict in conflicts:
            if str(conflict.get("proposal_ref")) not in known_proposals:
                errors.append(f"{prefix}: conflict references unknown proposal")
        candidate = proposal.get("candidate", {})
        if proposal.get("proposal_type") == "inline_size":
            for key in ("value", "width", "height"):
                if key in candidate and (
                    not isinstance(candidate[key], (int, float))
                    or not math.isfinite(float(candidate[key]))
                    or float(candidate[key]) <= 0
                ):
                    errors.append(f"{prefix}: inline size must be finite and positive")

    for diagnostic in payload.get("observation_diagnostics", []):
        ref = diagnostic.get("observation_ref")
        if ref not in known_evidence or not diagnostic.get("errors"):
            errors.append("invalid observation diagnostic evidence")
        if diagnostic.get("downstream_acceptance_closed") is not True or diagnostic.get("quantity_eligible") is not False:
            errors.append("quarantined observation acquired authority")
        if any(ref in proposal.get("evidence_refs", []) for proposal in proposals):
            errors.append("quarantined observation supplied a canonical proposal")

    expected_summary = {
        "source_observation_count": len(observations),
        "proposal_count": len(proposals),
        "proposal_type_counts": dict(sorted(Counter(item.get("proposal_type") for item in proposals).items())),
        "state_counts": dict(sorted(Counter(item.get("state") for item in proposals).items())),
    }
    if payload.get("summary") != expected_summary:
        errors.append("summary does not replay from records")
    contract = payload.get("exchange_contract", {})
    required_true = (
        "proposal_layer_only",
        "legend_membership_is_evidence_only",
        "color_only_evidence_must_abstain",
        "conflicts_must_abstain",
    )
    required_false = (
        "route_or_system_identity_established",
        "physical_connection_or_continuation_established",
        "physical_elevation_established",
        "confirmed_clash_established",
        "installed_length_emitted",
        "quantity_eligible",
    )
    for key in required_true:
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in required_false:
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors
