import copy
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import src.drawing_engine.disciplines.mep.mep_claim_grounding as claim_grounding_module
from src.drawing_engine.disciplines.mep.mep_annotation_observations import (
    pair_annotation_observations,
    validate_pdf_annotation_observations,
)
from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_claim_grounding import (
    build_mep_claim_grounding,
    validate_mep_claim_grounding,
)
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import validate_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import build_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph


SOURCE_SHA = "0" * 64
DOCUMENT_KEY = f"pdf-sha256:{SOURCE_SHA}"
PAGE_REF = "page.1"
ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
REAL_CHECKPOINT = REAL_ROOT / "real_m2_m5_checkpoint"


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _segment(index, start, end):
    source_ref = f"drawing[{index}].item[0].segment[0]"
    return {
        "id": source_ref,
        "drawing_ref": f"drawing[{index}]",
        "primitive_ref": f"drawing[{index}].item[0]",
        "item_index": 0,
        "part_index": 0,
        "kind": "line",
        "start_display": list(start),
        "end_display": list(end),
        "control_points_display": [],
        "sample_points_display": [],
        "style": {
            "stroke": [0.2, 0.2, 0.2],
            "fill": None,
            "width": 1.0,
            "dash": None,
        },
    }


def _upstream_target_payloads(*, target_count=1):
    segments = [
        _segment(0, (0, 0), (100, 0)),
        _segment(1, (0, 4), (100, 4)),
        _segment(2, (0, 0), (0, 4)),
    ]
    if target_count == 2:
        segments.extend(
            [
                _segment(3, (0, 10), (100, 10)),
                _segment(4, (0, 14), (100, 14)),
                _segment(5, (0, 10), (0, 14)),
            ]
        )
    registry = {
        "schema_version": "0.1.0",
        "layer": "mep_sheet_coordinate_registry",
        "document": {"document_key": DOCUMENT_KEY, "page_count": 1},
        "pages": [
            {
                "record_type": "mep_sheet_page_record",
                "record_version": "0.1.0",
                "id": "m1.scope.1",
                "page_ref": PAGE_REF,
                "page_number": 1,
                "role": "mechanical_piping_plan",
                "fields": {
                    "sheet_number": {
                        "value": "M-1",
                        "evidence_refs": ["title.sheet"],
                    },
                    "scale": {"drawing_inches_per_paper_inch": 96.0},
                },
            }
        ],
        "packages": [],
    }
    graph = build_mep_route_graph(
        sheet_registry=registry,
        page_inputs={
            PAGE_REF: {
                "native_topology": {
                    "schema_version": "0.1.0",
                    "segments": segments,
                    "vertices": [],
                }
            }
        },
    )
    composites = build_mep_outlined_route_composites(route_graph=graph)
    assert len(composites["accepted_composites"]) == target_count
    binding_evidence = [
        {"id": f"binding.evidence.{index}", "state": "observed"}
        for index in range(1, target_count + 1)
    ]
    route_scopes = []
    relations = []
    for index, composite in enumerate(composites["accepted_composites"], start=1):
        scope_ref = f"m4.scope.{index}"
        route_scopes.append(
            {
                "record_type": "mep_page_local_route_scope",
                "id": scope_ref,
                "page_ref": PAGE_REF,
                "fragment_refs": composite["member_fragment_refs"],
                "route_composite_ref": composite["id"],
                "page_local_only": True,
                "quantity_eligible": False,
            }
        )
        relations.append(
            {
                "record_type": "mep_page_local_binding_relation",
                "id": f"m4.relation.{index}",
                "page_ref": PAGE_REF,
                "relation_type": "route_system",
                "target_kind": "route_composite",
                "target_refs": [composite["id"]],
                "route_scope_refs": [scope_ref],
                "applicability_scope_refs": [],
                "binding_evidence_refs": [f"binding.evidence.{index}"],
                "authority": {"page_local_binding_established": True},
                "certificates": {"unique_geometric_target": True},
                "state": "accepted",
                "reasons": [],
                "quantity_eligible": False,
            }
        )
    bindings = {
        "schema_version": "0.1.0",
        "layer": "mep_page_local_attribute_bindings",
        "document": {"document_key": DOCUMENT_KEY, "page_count": 1},
        "m2_contract_ref": {"payload_sha256": "2" * 64},
        "m3_contract_ref": {
            "payload_sha256": composites["m3_contract_ref"]["payload_sha256"]
        },
        "m3_5_contract_ref": {
            "payload_sha256": _canonical_sha256(composites)
        },
        "outlined_route_composites": copy.deepcopy(
            composites["accepted_composites"]
        ),
        "binding_evidence_sha256": _canonical_sha256(binding_evidence),
        "binding_evidence": binding_evidence,
        "route_scopes": route_scopes,
        "attribute_applicability_scopes": [],
        "relations": relations,
        "summary": {},
        "exchange_contract": {
            "page_local_binding_only": True,
            "branch_and_change_point_applicability_split": True,
            "continuations_are_endpoint_relations_only": True,
            "cross_sheet_join_established": False,
            "physical_continuation_established": False,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    assert validate_mep_attribute_bindings(bindings) == []
    partial = []
    for index, composite in enumerate(composites["accepted_composites"], start=1):
        partial.append(
            {
                "record_type": "mep_partial_2_5d_centreline_segment",
                "record_version": "0.1.0",
                "id": f"m5.partial.{index}",
                "page_ref": PAGE_REF,
                "route_target_ref": composite["id"],
                "source_fragment_refs": composite["member_fragment_refs"],
                "system": {"value": "CHWS"},
                "size": {"value": 100, "unit": "mm"},
                "points_xy_m": [[0.0, float(index)], [1.0, float(index)]],
                "elevation_reference_basis": "bottom",
                "elevation_reference_m": 3.0,
                "centreline_elevation_m": None,
                "projected_2d_length_m": 1.0,
                "resolved_3d_length_m": None,
                "relation_refs": [f"relation.{index}"],
                "reason": "elevation_reference_does_not_resolve_centreline_offset",
                "state": "unknown",
                "quantity_eligible": False,
            }
        )
    m5 = {
        "schema_version": "0.1.0",
        "layer": "mep_cross_sheet_run_hypotheses",
        "document": {"document_key": DOCUMENT_KEY, "page_count": 1},
        "m1_contract_ref": {"payload_sha256": "1" * 64},
        "m3_contract_ref": {
            "payload_sha256": composites["m3_contract_ref"]["payload_sha256"]
        },
        "m4_contract_ref": {"payload_sha256": _canonical_sha256(bindings)},
        "continuation_candidates": [],
        "overlap_duplicate_candidates": [],
        "projected_route_occurrences": [],
        "canonical_projected_segments": [],
        "partial_2_5d_centreline_segments": partial,
        "resolved_3d_centreline_segments": [],
        "unresolved_vertical_spans": [],
        "physical_run_hypotheses": [],
        "summary": {},
        "exchange_contract": {
            "mutual_unique_continuation_required": True,
            "accepted_m1_registration_required": True,
            "compatible_system_size_elevation_required": True,
            "endpoint_bound_continuation_evidence_required": True,
            "transform_cycle_consistency_required": True,
            "overlap_canonicalized_before_run_assembly": True,
            "duplicate_interval_alone_does_not_establish_run": True,
            "source_occurrences_separate_from_canonical_segments": True,
            "projected_2d_and_resolved_3d_lengths_separate": True,
            "unresolved_vertical_spans_preserved": True,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    assert validate_mep_cross_sheet_runs(m5) == []
    return composites, bindings, m5


def _annotation(record_id, xref, kind, rect):
    is_cloud = kind == "cloud"
    x0, y0, x1, y1 = rect
    return {
        "record_type": "pdf_annotation_observation",
        "record_version": "0.1.0",
        "id": record_id,
        "document_key": DOCUMENT_KEY,
        "page_ref": PAGE_REF,
        "page_number": 1,
        "annotation_subtype": "Polygon" if is_cloud else "FreeText",
        "intent": "PolygonCloud" if is_cloud else "FreeTextCallout",
        "subject": "Cloud" if is_cloud else "Callout",
        "author": "Reviewer",
        "text": "Review route coordination" if not is_cloud else "",
        "epistemic_state": "observed",
        "immutable_source_observation": True,
        "geometry": {
            "coordinate_space": "page_display_points_top_left",
            "rect_display": list(rect),
            "rect_pdf": list(rect),
            "vertices_display": (
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                if is_cloud
                else [[x0, y0]]
            ),
        },
        "source_object": {"xref": xref, "object_name": f"object-{xref}"},
        "in_reply_to": {"xref": 20, "generation": 0} if is_cloud else None,
        "reply_type": "Group" if is_cloud else None,
    }


def _markup(cloud_rect):
    cloud = _annotation("cloud.1", 10, "cloud", cloud_rect)
    callout = _annotation("callout.1", 20, "callout", (25, 2, 50, 8))
    pairs, abstentions = pair_annotation_observations([cloud, callout])
    payload = {
        "schema_version": "0.1.0",
        "layer": "mep_pdf_annotation_observations",
        "document": {
            "document_key": DOCUMENT_KEY,
            "source_filename": "synthetic.pdf",
            "source_pdf_sha256": SOURCE_SHA,
            "source_bytes": 1,
            "page_count": 1,
        },
        "pages": [
            {
                "record_type": "mep_page_observation",
                "record_version": "0.1.0",
                "id": PAGE_REF,
                "document_key": DOCUMENT_KEY,
                "page_number": 1,
                "source_page_object": {"xref": 1},
                "media_box_pdf": [0, 0, 200, 200],
                "crop_box_pdf": [0, 0, 200, 200],
                "page_rect_display": [0, 0, 200, 200],
                "rotation_degrees": 0,
                "pdf_to_display_matrix": [1, 0, 0, 1, 0, 0],
                "display_to_pdf_matrix": [1, 0, 0, 1, 0, 0],
                "source_content_stream_xrefs": [],
                "annotation_refs": ["callout.1", "cloud.1"],
                "sheet_role": "unknown",
                "native_raster_route": "unclassified",
            }
        ],
        "annotations": [cloud, callout],
        "annotation_pairs": pairs,
        "pairing_abstentions": abstentions,
        "summary": {
            "page_count": 1,
            "annotation_count": 2,
            "annotation_pair_count": 1,
            "pairing_abstention_count": 0,
            "annotation_subtype_counts": {"FreeText": 1, "Polygon": 1},
            "pair_relationship_counts": {"pdf_group_relationship": 1},
        },
        "exchange_contract": {
            "page_record_type": "mep_page_observation",
            "annotation_record_type": "pdf_annotation_observation",
            "pair_record_type": "annotation_pair_observation",
            "record_version": "0.1.0",
            "page_transforms_are_explicit": True,
            "source_pdf_objects_are_preserved": True,
            "annotation_observations_are_immutable": True,
            "annotation_pairs_are_markup_claims_only": True,
            "sheet_role_and_native_raster_route_are_deferred_to_m1": True,
            "target_grounding_is_deferred_to_m6": True,
            "route_identity_or_connectivity_established": False,
            "confirmed_clash_or_clearance_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
        "extractor": {"name": "synthetic", "version": "0.1.0"},
    }
    assert validate_pdf_annotation_observations(payload) == []
    return payload


class MepClaimGroundingTest(unittest.TestCase):
    def test_real_1a_1b_checkpoint_replays_all_m0_claims_as_page_mismatch_abstentions(self):
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        markup = load(REAL_ROOT / "m_and_p_coordination.annotation-observations.json")
        composites = load(
            REAL_CHECKPOINT / "pages_1a_1b.outlined-route-composites.json"
        )
        bindings = load(REAL_CHECKPOINT / "pages_1a_1b.attribute-bindings.json")
        runs = load(REAL_CHECKPOINT / "pages_1a_1b.cross-sheet-runs.json")
        stored = load(REAL_CHECKPOINT / "pages_1a_1b.claim-grounding.json")

        replay = build_mep_claim_grounding(
            markup_observations=markup,
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=runs,
        )

        self.assertEqual(replay, stored)
        self.assertEqual(validate_mep_claim_grounding(stored), [])
        self.assertEqual(stored["summary"]["claim_count"], 15)
        self.assertEqual(stored["summary"]["eligible_partial_2_5d_target_count"], 4)
        self.assertEqual(stored["summary"]["accepted_unique_grounding_count"], 0)
        self.assertEqual(stored["summary"]["ambiguous_2d_overlap_candidate_count"], 0)
        self.assertEqual(stored["summary"]["abstention_count"], 15)
        target_pages = {row["page_ref"] for row in stored["eligible_targets"]}
        self.assertTrue(
            all(row["page_ref"] not in target_pages for row in stored["grounding_records"])
        )
        self.assertTrue(
            all(row["state"] == "abstained" for row in stored["grounding_records"])
        )
        self.assertTrue(
            all(
                row["reasons"] == ["no_eligible_partial_2_5d_target_on_claim_page"]
                for row in stored["grounding_records"]
            )
        )

    def test_unique_composite_overlap_accepts_one_hash_bound_claim_target(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=1)
        markup = _markup((20, -1, 30, 5))

        payload = build_mep_claim_grounding(
            markup_observations=markup,
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=m5,
        )

        self.assertEqual(validate_mep_claim_grounding(payload), [])
        self.assertEqual(payload["summary"]["accepted_unique_grounding_count"], 1)
        record = payload["grounding_records"][0]
        self.assertEqual(record["record_type"], "mep_claim_target_grounding")
        self.assertEqual(record["state"], "accepted")
        self.assertEqual(len(record["alternatives"]), 1)
        self.assertEqual(record["target_ref"], record["alternatives"][0]["target_ref"])
        self.assertEqual(
            record["claim_provenance"]["m0_annotation_pair_ref"],
            markup["annotation_pairs"][0]["id"],
        )
        self.assertEqual(
            payload["m0_contract_ref"]["payload_sha256"],
            hashlib.sha256(
                json.dumps(
                    markup,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    def test_multiple_route_corridors_emit_2d_overlap_candidate_with_alternatives(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=2)
        payload = build_mep_claim_grounding(
            markup_observations=_markup((20, -1, 30, 15)),
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=m5,
        )

        record = payload["grounding_records"][0]
        self.assertEqual(record["record_type"], "2d_overlap_candidate")
        self.assertEqual(record["state"], "candidate")
        self.assertIsNone(record["target_ref"])
        self.assertEqual(len(record["alternatives"]), 2)
        self.assertFalse(record["ambiguity"]["resolved"])

    def test_missing_route_target_abstains_without_inventing_an_identity(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=1)
        payload = build_mep_claim_grounding(
            markup_observations=_markup((150, 150, 170, 170)),
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=m5,
        )

        record = payload["grounding_records"][0]
        self.assertEqual(record["record_type"], "mep_claim_grounding_abstention")
        self.assertEqual(record["state"], "abstained")
        self.assertEqual(record["alternatives"], [])
        self.assertIsNone(record["target_ref"])
        self.assertEqual(
            record["reasons"],
            ["claim_polygon_does_not_overlap_eligible_page_target"],
        )

    def test_composite_without_accepted_m4_binding_is_not_an_eligible_target(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=1)
        unbound = copy.deepcopy(bindings)
        unbound["relations"] = []
        self.assertEqual(validate_mep_attribute_bindings(unbound), [])
        replay_runs = copy.deepcopy(m5)
        replay_runs["m4_contract_ref"]["payload_sha256"] = _canonical_sha256(unbound)

        payload = build_mep_claim_grounding(
            markup_observations=_markup((20, -1, 30, 5)),
            outlined_route_composites=composites,
            attribute_bindings=unbound,
            cross_sheet_runs=replay_runs,
        )

        self.assertEqual(payload["eligible_targets"], [])
        self.assertEqual(payload["grounding_records"][0]["state"], "abstained")
        self.assertEqual(
            payload["grounding_records"][0]["reasons"],
            ["no_eligible_partial_2_5d_target_on_claim_page"],
        )

    def test_resolved_3d_only_target_is_outside_m6a_and_partial_target_has_no_authority(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=1)
        partial = m5["partial_2_5d_centreline_segments"][0]
        resolved_only = copy.deepcopy(m5)
        resolved_only["partial_2_5d_centreline_segments"] = []
        resolved_only["resolved_3d_centreline_segments"] = [
            {
                "record_type": "mep_resolved_3d_centreline_segment",
                "id": "m5.resolved.1",
                "route_target_ref": partial["route_target_ref"],
                "quantity_eligible": False,
            }
        ]
        self.assertEqual(validate_mep_cross_sheet_runs(resolved_only), [])
        markup = _markup((20, -1, 30, 5))
        resolved_payload = build_mep_claim_grounding(
            markup_observations=markup,
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=resolved_only,
        )
        self.assertEqual(resolved_payload["eligible_targets"], [])
        self.assertEqual(resolved_payload["grounding_records"][0]["state"], "abstained")

        partial_payload = build_mep_claim_grounding(
            markup_observations=markup,
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=m5,
        )
        keys = {key for _path, key, _value in claim_grounding_module._walk_items(partial_payload)}
        self.assertFalse(
            keys
            & {
                "confirmed_clash",
                "calculated_severity",
                "severity",
                "installed_length",
                "quantity",
                "m7_output",
            }
        )
        self.assertFalse(partial_payload["grounding_records"][0]["three_dimensional_authority"])

    def test_recursive_validator_rejects_authority_and_ambiguity_promotions(self):
        composites, bindings, m5 = _upstream_target_payloads(target_count=1)
        payload = build_mep_claim_grounding(
            markup_observations=_markup((20, -1, 30, 5)),
            outlined_route_composites=composites,
            attribute_bindings=bindings,
            cross_sheet_runs=m5,
        )
        for key in (
            "confirmed_clash",
            "calculated_severity",
            "severity",
            "installed_length_m",
            "quantity",
            "m7_output_record",
        ):
            changed = copy.deepcopy(payload)
            changed["grounding_records"][0]["nested"] = {key: 1}
            self.assertTrue(validate_mep_claim_grounding(changed), key)
        for key in (
            "three_dimensional_authority",
            "clash_authority_enabled",
            "calculated_severity_enabled",
            "installed_length_emitted",
            "m7_outputs_enabled",
            "quantity_eligible",
        ):
            changed = copy.deepcopy(payload)
            changed["grounding_records"][0][key] = True
            self.assertTrue(validate_mep_claim_grounding(changed), key)
        changed = copy.deepcopy(payload)
        changed["grounding_records"][0]["alternatives"].append(
            copy.deepcopy(changed["grounding_records"][0]["alternatives"][0])
        )
        self.assertTrue(validate_mep_claim_grounding(changed))

        changed = copy.deepcopy(payload)
        changed["m5_contract_ref"]["m4_payload_sha256"] = "f" * 64
        self.assertTrue(validate_mep_claim_grounding(changed))

        changed = copy.deepcopy(payload)
        changed["grounding_records"][0]["page_ref"] = "page.2"
        self.assertTrue(validate_mep_claim_grounding(changed))

        changed = copy.deepcopy(payload)
        changed["grounding_records"][0]["claim_provenance"][
            "claim_polygon_display"
        ] = [[150, 150], [160, 150], [160, 160], [150, 160]]
        self.assertTrue(validate_mep_claim_grounding(changed))

    def test_production_source_has_no_fixture_dispatch_or_forbidden_layer_imports(self):
        source = inspect.getsource(claim_grounding_module)
        self.assertNotIn("M&P mark-up", source)
        self.assertNotIn("L01-MP-P.1A", source)
        self.assertNotIn("page_number ==", source)
        self.assertNotIn("step1c", source.casefold())
        self.assertNotIn("licensed", source.casefold())


if __name__ == "__main__":
    unittest.main()
