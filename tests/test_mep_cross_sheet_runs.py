import copy
import inspect
import unittest

import src.drawing_engine.disciplines.mep.mep_cross_sheet_runs as mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import (
    build_mep_cross_sheet_runs,
    validate_mep_cross_sheet_runs,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import build_mep_terminology_proposals


DOCUMENT_KEY = "pdf-sha256:m5-synthetic"


def _page(page_ref, number):
    return {
        "record_type": "mep_sheet_page_record",
        "record_version": "0.1.0",
        "id": f"m1.scope.{page_ref}",
        "page_ref": page_ref,
        "page_number": number,
        "role": "mechanical_piping_plan",
        "fields": {
            "sheet_number": {"value": f"SYN-{number}", "evidence_refs": []},
            "scale": {"drawing_inches_per_paper_inch": 48.0},
        },
    }


def _registration(identifier, source, target, dx, dy=0.0):
    return {
        "record_type": "mep_adjoining_sheet_transform",
        "record_version": "0.1.0",
        "id": identifier,
        "source_page_ref": source,
        "target_page_ref": target,
        "state": "accepted",
        "matrix_source_display_to_target_display": [1.0, 0.0, 0.0, 1.0, dx, dy],
        "maximum_residual_tolerance_display": 0.25,
        "physical_continuation_established": False,
        "route_identity_established": False,
        "quantity_eligible": False,
    }


def _segment(index, start, end, *, color=(1.0, 0.0, 0.0)):
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
            "stroke": list(color),
            "fill": None,
            "width": 1.0,
            "dash": None,
        },
    }


def _build_graph(page_segments, registrations):
    pages = [_page(page_ref, index + 1) for index, page_ref in enumerate(page_segments)]
    registry = {
        "schema_version": "0.1.0",
        "layer": "mep_sheet_coordinate_registry",
        "document": {"document_key": DOCUMENT_KEY, "page_count": len(pages)},
        "pages": pages,
        "packages": [],
        "adjoining_sheet_transforms": list(registrations),
    }
    graph = build_mep_route_graph(
        sheet_registry=registry,
        page_inputs={
            page_ref: {
                "native_topology": {
                    "schema_version": "0.1.0",
                    "segments": segments,
                    "vertices": [],
                }
            }
            for page_ref, segments in page_segments.items()
        },
    )
    return registry, graph


def _fragment(page, source_ref):
    return next(row for row in page["fragments"] if row["source_primitive_ref"] == source_ref)


def _endpoint(page, fragment_ref, role):
    return next(
        row
        for row in page["endpoints"]
        if row["fragment_ref"] == fragment_ref and row["role"] == role
    )


def _bindings(graph, specs):
    observations = []
    requested = []
    for index, spec in enumerate(specs):
        page_ref = spec["page_ref"]
        base = f"obs.{index}"
        observations.extend(
            [
                {
                    "id": f"{base}.system-size",
                    "page_ref": page_ref,
                    "text": f'{spec.get("size", "DN 50")} {spec.get("system", "CHWS")}',
                    "evidence_channels": ["native_pdf_text"],
                },
                {
                    "id": f"{base}.elevation",
                    "page_ref": page_ref,
                    "text": spec.get("elevation", "CL +2.500 M"),
                    "evidence_channels": ["native_pdf_text"],
                },
            ]
        )
        if spec.get("continuation_role"):
            continuation = {
                "id": f"{base}.continuation",
                "page_ref": page_ref,
                "evidence_channels": ["native_vector_geometry"],
            }
            if spec.get("continuation_text"):
                continuation["text"] = spec["continuation_text"]
                continuation["evidence_channels"].append("native_pdf_text")
            else:
                continuation["symbol_kind"] = "continuation"
            observations.append(continuation)
        requested.append((base, spec))
    terminology = build_mep_terminology_proposals(
        document={"document_key": DOCUMENT_KEY}, observations=observations
    )
    proposals_by_evidence = {}
    for proposal in terminology["proposals"]:
        for evidence_ref in proposal["evidence_refs"]:
            proposals_by_evidence.setdefault(evidence_ref, []).append(proposal)
    pages = {page["page_ref"]: page for page in graph["pages"]}
    evidence = []
    for base, spec in requested:
        page = pages[spec["page_ref"]]
        fragment = _fragment(page, spec["source_ref"])
        targets = spec.get("target_fragment_refs", [fragment["id"]])
        for observation_ref in (f"{base}.system-size", f"{base}.elevation"):
            if observation_ref.endswith("elevation") and spec.get("omit_elevation"):
                continue
            for proposal in proposals_by_evidence[observation_ref]:
                evidence.append(
                    {
                        "id": f"bind.{len(evidence)}",
                        "proposal_ref": proposal["id"],
                        "page_ref": spec["page_ref"],
                        "state": "observed",
                        "method": {"name": "synthetic_exact_target", "version": "1.0.0"},
                        "target_kind": "route_fragment",
                        "target_refs": targets,
                        "geometric_evidence_refs": targets,
                    }
                )
        role = spec.get("continuation_role")
        if role:
            endpoint = _endpoint(page, fragment["id"], role)
            proposal = proposals_by_evidence[f"{base}.continuation"][0]
            evidence.append(
                {
                    "id": f"bind.{len(evidence)}",
                    "proposal_ref": proposal["id"],
                    "page_ref": spec["page_ref"],
                    "state": "observed",
                    "method": {"name": "synthetic_endpoint_symbol", "version": "1.0.0"},
                    "target_kind": "route_endpoint",
                    "target_refs": [endpoint["id"]],
                    "geometric_evidence_refs": [endpoint["id"]],
                }
            )
    return build_mep_attribute_bindings(
        terminology_proposals=terminology,
        route_graph=graph,
        binding_evidence=evidence,
    )


class MepCrossSheetRunsTest(unittest.TestCase):
    def test_mutual_unique_registered_continuation_closes_but_stays_quantity_ineligible(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right]},
            [_registration("registration.ab", "page.a", "page.b", 90)],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"], "continuation_role": "end"},
                {"page_ref": "page.b", "source_ref": right["id"], "continuation_role": "start"},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )

        self.assertEqual(validate_mep_cross_sheet_runs(payload), [])
        self.assertEqual(payload["summary"]["accepted_continuation_count"], 1)
        self.assertEqual(payload["summary"]["physical_run_hypothesis_count"], 1)
        self.assertEqual(payload["summary"]["resolved_3d_centreline_segment_count"], 2)
        self.assertFalse(payload["quantity_eligible"])
        self.assertFalse(payload["exchange_contract"]["installed_length_emitted"])

    def test_incompatible_attributes_and_missing_elevation_abstain_independently(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right]},
            [_registration("registration.ab", "page.a", "page.b", 90)],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"], "continuation_role": "end"},
                {
                    "page_ref": "page.b",
                    "source_ref": right["id"],
                    "continuation_role": "start",
                    "system": "CHWR",
                    "omit_elevation": True,
                },
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        candidate = payload["continuation_candidates"][0]
        self.assertEqual(candidate["state"], "abstained")
        self.assertIn("incompatible_system", candidate["reasons"])
        self.assertIn("target_missing_elevation", candidate["reasons"])

    def test_missing_registration_abstains(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (10, 0), (20, 0))
        registry, graph = _build_graph({"page.a": [left], "page.b": [right]}, [])
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"], "continuation_role": "end"},
                {"page_ref": "page.b", "source_ref": right["id"], "continuation_role": "start"},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertIn(
            "missing_accepted_m1_registration",
            payload["continuation_candidates"][0]["reasons"],
        )

    def test_incompatible_continuation_tags_abstain(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right]},
            [_registration("registration.ab", "page.a", "page.b", 90)],
        )
        bindings = _bindings(
            graph,
            [
                {
                    "page_ref": "page.a",
                    "source_ref": left["id"],
                    "continuation_role": "end",
                    "continuation_text": "MATCH LINE A",
                },
                {
                    "page_ref": "page.b",
                    "source_ref": right["id"],
                    "continuation_role": "start",
                    "continuation_text": "MATCH LINE B",
                },
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertIn(
            "ambiguous_continuation_evidence",
            payload["continuation_candidates"][0]["reasons"],
        )

    def test_bidirectional_uniqueness_rejects_two_coincident_targets(self):
        left = _segment(0, (0, 0), (10, 0))
        right_a = _segment(1, (100, -0.8), (110, -0.8))
        right_b = _segment(2, (100, 0.8), (110, 0.8))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right_a, right_b]},
            [_registration("registration.ab", "page.a", "page.b", 90)],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"], "continuation_role": "end"},
                {"page_ref": "page.b", "source_ref": right_a["id"], "continuation_role": "start"},
                {"page_ref": "page.b", "source_ref": right_b["id"], "continuation_role": "start"},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        coincident = [
            row
            for row in payload["continuation_candidates"]
            if row["certificates"]["transformed_endpoint_coincidence"]
        ]
        self.assertEqual(len(coincident), 2)
        self.assertTrue(
            all("continuation_pair_is_not_mutual_unique" in row["reasons"] for row in coincident)
        )

    def test_transform_cycle_residual_abstains(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right], "page.c": []},
            [
                _registration("registration.ab", "page.a", "page.b", 90),
                _registration("registration.ac", "page.a", "page.c", 40),
                _registration("registration.cb", "page.c", "page.b", 40),
            ],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"], "continuation_role": "end"},
                {"page_ref": "page.b", "source_ref": right["id"], "continuation_role": "start"},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertIn(
            "transform_cycle_residual_exceeds_tolerance",
            payload["continuation_candidates"][0]["reasons"],
        )

    def test_overlap_is_canonicalised_but_duplicate_interval_is_not_a_run(self):
        left = _segment(0, (0, 0), (10, 0))
        right = _segment(1, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right]},
            [_registration("registration.ab", "page.a", "page.b", 100)],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"]},
                {"page_ref": "page.b", "source_ref": right["id"]},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertEqual(payload["summary"]["accepted_overlap_duplicate_count"], 1)
        self.assertEqual(payload["summary"]["canonical_projected_segment_count"], 1)
        segment = payload["canonical_projected_segments"][0]
        self.assertEqual(segment["source_occurrence_count"], 2)
        self.assertTrue(segment["physical_segment_identity_established"])
        self.assertEqual(payload["summary"]["physical_run_hypothesis_count"], 0)
        self.assertTrue(
            payload["exchange_contract"][
                "duplicate_interval_alone_does_not_establish_run"
            ]
        )

    def test_duplicate_overlap_ambiguity_and_missing_elevation_abstain(self):
        left = _segment(0, (0, 0), (10, 0))
        right_a = _segment(1, (100, -0.8), (110, -0.8))
        right_b = _segment(2, (100, 0.8), (110, 0.8))
        right_c = _segment(3, (100, 0), (110, 0))
        registry, graph = _build_graph(
            {"page.a": [left], "page.b": [right_a, right_b, right_c]},
            [_registration("registration.ab", "page.a", "page.b", 100)],
        )
        bindings = _bindings(
            graph,
            [
                {"page_ref": "page.a", "source_ref": left["id"]},
                {"page_ref": "page.b", "source_ref": right_a["id"]},
                {"page_ref": "page.b", "source_ref": right_b["id"]},
                {"page_ref": "page.b", "source_ref": right_c["id"], "omit_elevation": True},
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        reasons = {reason for row in payload["overlap_duplicate_candidates"] for reason in row["reasons"]}
        self.assertIn("duplicate_overlap_pair_is_not_mutual_unique", reasons)
        self.assertIn("target_missing_elevation", reasons)
        self.assertEqual(payload["summary"]["accepted_overlap_duplicate_count"], 0)

    def test_unresolved_vertical_span_is_preserved(self):
        segment = _segment(0, (0, 0), (10, 0))
        registry, graph = _build_graph({"page.a": [segment]}, [])
        page = graph["pages"][0]
        fragment = page["fragments"][0]
        endpoint = _endpoint(page, fragment["id"], "end")
        terminology = build_mep_terminology_proposals(
            document={"document_key": DOCUMENT_KEY},
            observations=[
                {
                    "id": "obs.riser",
                    "page_ref": "page.a",
                    "symbol_kind": "riser",
                    "evidence_channels": ["native_vector_geometry"],
                }
            ],
        )
        proposal = terminology["proposals"][0]
        bindings = build_mep_attribute_bindings(
            terminology_proposals=terminology,
            route_graph=graph,
            binding_evidence=[
                {
                    "id": "bind.riser",
                    "proposal_ref": proposal["id"],
                    "page_ref": "page.a",
                    "state": "observed",
                    "method": {"name": "synthetic_endpoint_symbol", "version": "1.0.0"},
                    "target_kind": "route_endpoint",
                    "target_refs": [endpoint["id"]],
                    "geometric_evidence_refs": [endpoint["id"]],
                }
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertEqual(payload["summary"]["unresolved_vertical_span_count"], 1)
        self.assertIsNone(payload["unresolved_vertical_spans"][0]["vertical_extent_m"])

    def test_bottom_elevation_remains_partial_2_5d_until_centreline_offset_closes(self):
        segment = _segment(0, (0, 0), (10, 0))
        registry, graph = _build_graph({"page.a": [segment]}, [])
        bindings = _bindings(
            graph,
            [
                {
                    "page_ref": "page.a",
                    "source_ref": segment["id"],
                    "elevation": 'BE= 10\' - 0"',
                }
            ],
        )
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        self.assertEqual(payload["summary"]["partial_2_5d_centreline_segment_count"], 1)
        self.assertEqual(payload["summary"]["resolved_3d_centreline_segment_count"], 0)
        partial = payload["partial_2_5d_centreline_segments"][0]
        self.assertIsNone(partial["centreline_elevation_m"])
        self.assertIsNone(partial["resolved_3d_length_m"])
        self.assertEqual(partial["elevation_reference_basis"], "bottom")

    def test_validator_rejects_m7_outputs_and_production_has_no_fixture_dispatch(self):
        left = _segment(0, (0, 0), (10, 0))
        registry, graph = _build_graph({"page.a": [left]}, [])
        bindings = _bindings(graph, [{"page_ref": "page.a", "source_ref": left["id"]}])
        payload = build_mep_cross_sheet_runs(
            sheet_registry=registry, route_graph=graph, attribute_bindings=bindings
        )
        changed = copy.deepcopy(payload)
        changed["installed_length"] = 12.0
        self.assertTrue(validate_mep_cross_sheet_runs(changed))
        changed = copy.deepcopy(payload)
        changed["quantity_eligible"] = True
        self.assertTrue(validate_mep_cross_sheet_runs(changed))
        source = inspect.getsource(mep_cross_sheet_runs)
        self.assertNotIn("L01-MP-P.1A", source)
        self.assertNotIn("M&P mark-up", source)


if __name__ == "__main__":
    unittest.main()
