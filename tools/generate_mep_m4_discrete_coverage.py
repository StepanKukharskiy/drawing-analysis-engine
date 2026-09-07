#!/usr/bin/env python3
"""Freeze full-package M4 discrete-target review without promoting proximity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals


FIXTURE_DIR = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_DIR = FIXTURE_DIR / "real_m2_m5_checkpoint"
OUTPUT_DIR = FIXTURE_DIR / "real_m4_discrete_coverage"
DEFAULT_M0 = FIXTURE_DIR / "m_and_p_coordination.annotation-observations.json"
DEFAULT_M1 = FIXTURE_DIR / "m_and_p_coordination.sheet-registry.json"
DEFAULT_TRUTH = FIXTURE_DIR / "m_and_p_coordination_m7b_coverage_truth.json"
DEFAULT_M2 = CHECKPOINT_DIR / "pages_1a_1b.terminology-proposals.json"
DEFAULT_M3 = CHECKPOINT_DIR / "pages_1a_1b.route-observations.json"
DEFAULT_M35 = CHECKPOINT_DIR / "pages_1a_1b.outlined-route-composites.json"
DEFAULT_M4 = CHECKPOINT_DIR / "pages_1a_1b.attribute-bindings.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _coverage_observations(m0: dict, truth: dict) -> list[dict]:
    pairs = {str(row["id"]): row for row in m0.get("annotation_pairs", [])}
    annotations = {str(row["id"]): row for row in m0.get("annotations", [])}
    symbol_by_category = {
        "equipment": "equipment",
        "valve": "valve",
        "fitting": "fitting",
        "damper": "damper",
        "fixture": "equipment",
    }
    output = []
    for finding in truth.get("reviewed_findings", []):
        pair_ref = str(finding["annotation_pair_ref"])
        pair = pairs[pair_ref]
        callout = annotations[str(pair["callout_annotation_ref"])]
        cloud = annotations[str(pair["cloud_annotation_ref"])]
        categories = sorted(set(str(value) for value in finding["categories"]))
        output.append({
            "id": f"m4_discrete_coverage_observation.{finding['id']}",
            "page_ref": str(pair["page_ref"]),
            "page_number": int(pair["page_number"]),
            "text": str(callout.get("text") or ""),
            "bbox_display": deepcopy(cloud.get("geometry", {}).get("rect_display")),
            "symbol_candidates": sorted(
                {symbol_by_category[category] for category in categories}
            ),
            "anchor_ref": pair_ref,
            "interpretation_scope_ref": str(finding["id"]),
            "evidence_channels": [
                "pdf_annotation_markup",
                "reviewed_discrete_category_classification",
            ],
            "review_provenance": {
                "reviewed_finding_ref": str(finding["id"]),
                "annotation_pair_ref": pair_ref,
                "cloud_annotation_ref": str(pair["cloud_annotation_ref"]),
                "callout_annotation_ref": str(pair["callout_annotation_ref"]),
                "markup_claim_only": True,
                "target_identity_established": False,
            },
        })
    return sorted(output, key=lambda row: (row["page_number"], row["id"]))


def _negative_binding_evidence(m2: dict, coverage_observation_ids: set[str]) -> list[dict]:
    reason_by_relation = {
        "equipment": [
            "nearby_equipment_without_port_connectivity",
            "no_unique_geometric_target",
        ],
        "component": ["no_unique_geometric_target"],
    }
    output = []
    for proposal in m2.get("proposals", []):
        observation_refs = sorted(
            coverage_observation_ids.intersection(
                str(ref) for ref in proposal.get("evidence_refs", [])
            )
        )
        if len(observation_refs) != 1:
            continue
        proposal_type = str(proposal.get("proposal_type"))
        candidate = proposal.get("candidate", {})
        category = str(candidate.get("category") or "")
        kind = str(candidate.get("kind") or "")
        if proposal_type == "equipment":
            reasons = reason_by_relation["equipment"]
        elif proposal_type == "component" and (
            category == "valve" or "valve" in kind
        ):
            reasons = ["ambiguous_inline_symbol_target", "no_unique_geometric_target"]
        elif proposal_type == "component" and (
            category == "damper" or "damper" in kind
        ):
            reasons = ["broad_region_without_unique_target", "no_unique_geometric_target"]
        elif proposal_type == "component":
            reasons = reason_by_relation["component"]
        else:
            continue
        observation_ref = observation_refs[0]
        output.append({
            "id": f"m4_discrete_negative_target.{proposal['id']}",
            "proposal_ref": str(proposal["id"]),
            "page_ref": str(proposal["page_ref"]),
            "state": "observed",
            "method": {
                "name": "reviewed_discrete_target_coverage",
                "version": "1.0.0",
            },
            "disposition": "abstain",
            "target_kind": None,
            "target_refs": [],
            "geometric_evidence_refs": [observation_ref],
            "negative_gate_reasons": reasons,
            "reviewed_observation_ref": observation_ref,
        })
    return sorted(output, key=lambda row: row["id"])


def _review_freeze(registry: dict, m2: dict, m4: dict, coverage_ids: set[str]) -> dict:
    page_by_ref = {str(row["page_ref"]): row for row in registry.get("pages", [])}
    proposal_by_id = {str(row["id"]): row for row in m2.get("proposals", [])}
    relations = [
        row
        for row in m4.get("relations", [])
        if coverage_ids.intersection(
            str(ref)
            for ref in proposal_by_id.get(str(row.get("proposal_ref")), {}).get(
                "evidence_refs", []
            )
        )
    ]
    relations_by_page: dict[str, list[dict]] = {}
    for relation in relations:
        relations_by_page.setdefault(str(relation["page_ref"]), []).append(relation)
    accepted_route_pages = {
        str(relation["page_ref"])
        for relation in m4.get("relations", [])
        if relation.get("state") == "accepted"
        and str(relation.get("relation_type", "")).startswith("route_")
    }
    rows = []
    for page in registry.get("pages", []):
        if page.get("role") == "divider" or str(page["page_ref"]) in accepted_route_pages:
            continue
        page_ref = str(page["page_ref"])
        page_relations = sorted(
            relations_by_page.get(page_ref, []), key=lambda row: str(row["id"])
        )
        rows.append({
            "page_ref": page_ref,
            "pdf_page_number": int(page["page_number"]),
            "drawing_sheet_number": page.get("fields", {}).get("sheet_number", {}).get("value"),
            "review_state": (
                "reviewed_no_acceptable_discrete_target"
                if page_relations
                else "candidate_inventory_not_established"
            ),
            "proposal_refs": sorted(str(row["proposal_ref"]) for row in page_relations),
            "relation_refs": sorted(str(row["id"]) for row in page_relations),
            "accepted_discrete_target_count": sum(
                row.get("state") == "accepted" for row in page_relations
            ),
            "document_occurrence_completeness_established": False,
        })
    state_counts = dict(sorted(Counter(row["review_state"] for row in rows).items()))
    return {
        "schema_version": "0.1.0",
        "layer": "mep_m4_discrete_target_review_freeze",
        "document": deepcopy(registry.get("document", {})),
        "m2_contract_ref": {"payload_sha256": _sha256(m2)},
        "m4_contract_ref": {"payload_sha256": _sha256(m4)},
        "pages": rows,
        "summary": {
            "uncovered_item_bearing_page_count": len(rows),
            "review_state_counts": state_counts,
            "reviewed_discrete_proposal_count": len(relations),
            "accepted_discrete_target_count": sum(
                row.get("state") == "accepted" for row in relations
            ),
        },
        "coverage_conclusion": {
            "reviewed_markup_regions_with_no_acceptable_target": state_counts.get(
                "reviewed_no_acceptable_discrete_target", 0
            ),
            "pages_with_candidate_inventory_not_established": state_counts.get(
                "candidate_inventory_not_established", 0
            ),
            "document_occurrence_completeness": "not_established",
        },
        "authority": {
            "item_occurrence_established": False,
            "physical_identity_established": False,
            "calculated_count_established": False,
            "quantity_eligible": False,
        },
    }


def generate(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    m0 = _load(DEFAULT_M0)
    registry = _load(DEFAULT_M1)
    truth = _load(DEFAULT_TRUTH)
    base_m2 = _load(DEFAULT_M2)
    route_graph = _load(DEFAULT_M3)
    composites = _load(DEFAULT_M35)
    base_m4 = _load(DEFAULT_M4)

    coverage_observations = _coverage_observations(m0, truth)
    coverage_ids = {str(row["id"]) for row in coverage_observations}
    terminology = build_mep_terminology_proposals(
        document=base_m2["document"],
        observations=[*base_m2["source_observations"], *coverage_observations],
    )
    binding_evidence = [
        *base_m4["binding_evidence"],
        *_negative_binding_evidence(terminology, coverage_ids),
    ]
    bindings = build_mep_attribute_bindings(
        terminology_proposals=terminology,
        route_graph=route_graph,
        outlined_route_composites=composites,
        binding_evidence=binding_evidence,
    )
    review = _review_freeze(registry, terminology, bindings, coverage_ids)

    outputs = {
        "full_package.terminology-proposals.json": terminology,
        "full_package.attribute-bindings.json": bindings,
        "full_package.discrete-target-review.json": review,
    }
    paths = {}
    for name, payload in outputs.items():
        path = output_dir / name
        _write(path, payload)
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    for path in generate(output_dir=args.output_dir).values():
        print(path)


if __name__ == "__main__":
    main()
