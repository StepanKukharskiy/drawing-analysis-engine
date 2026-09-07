import copy
import inspect
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import fitz

import src.drawing_engine.disciplines.mep.mep_annotation_observations as mep_annotation_observations

from src.drawing_engine.disciplines.mep.mep_annotation_observations import (
    extract_pdf_annotation_observations,
    pair_annotation_observations,
    validate_pdf_annotation_observations,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
SOURCE_PDF = ROOT / "M&P mark-up against shop systems piping.pdf"
TRUTH_PATH = FIXTURE_ROOT / "m_and_p_coordination_truth.json"
OBSERVATIONS_PATH = FIXTURE_ROOT / "m_and_p_coordination.annotation-observations.json"


def _annotation(
    record_id,
    xref,
    kind,
    *,
    rect=(0, 0, 10, 10),
    leader_target=None,
    in_reply_to=None,
    reply_type=None,
):
    is_cloud = kind == "cloud"
    return {
        "id": record_id,
        "page_ref": "page.1",
        "page_number": 1,
        "annotation_subtype": "Polygon" if is_cloud else "FreeText",
        "intent": "PolygonCloud" if is_cloud else "FreeTextCallout",
        "subject": "Cloud" if is_cloud else "Callout",
        "text": "" if is_cloud else "Coordination comment",
        "geometry": {
            "rect_display": list(rect),
            "vertices_display": (
                [[1, 1], [9, 1], [9, 9], [1, 9]]
                if is_cloud
                else [list(leader_target or (20, 20))]
            ),
        },
        "source_object": {"xref": xref, "object_name": f"object-{xref}"},
        "in_reply_to": in_reply_to,
        "reply_type": reply_type,
    }


class MepAnnotationObservationsTest(unittest.TestCase):
    def test_small_pdf_preserves_annotation_fields_and_page_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.pdf"
            document = fitz.open()
            page = document.new_page(width=300, height=200)
            annotation = page.add_freetext_annot(
                fitz.Rect(20, 30, 180, 70),
                "Authored coordination note",
                text_color=(1, 0, 0),
            )
            annotation.set_info(
                title="Reviewer",
                subject="Text Box",
                content="Authored coordination note",
            )
            annotation.update()
            document.save(path)
            document.close()

            payload = extract_pdf_annotation_observations(path)

        self.assertEqual(validate_pdf_annotation_observations(payload), [])
        self.assertEqual(payload["summary"]["page_count"], 1)
        self.assertEqual(payload["summary"]["annotation_count"], 1)
        observation = payload["annotations"][0]
        self.assertEqual(observation["annotation_subtype"], "FreeText")
        self.assertEqual(observation["author"], "Reviewer")
        self.assertEqual(observation["text"], "Authored coordination note")
        self.assertEqual(observation["geometry"]["coordinate_space"], "page_display_points_top_left")
        self.assertEqual(len(observation["geometry"]["rect_display"]), 4)
        self.assertEqual(len(payload["pages"][0]["pdf_to_display_matrix"]), 6)
        self.assertIsInstance(observation["source_object"]["xref"], int)
        self.assertIn("1 0 0 rg", observation["appearance"]["default_appearance"])
        display_rect = fitz.Rect(observation["geometry"]["rect_pdf"]) * fitz.Matrix(
            *payload["pages"][0]["pdf_to_display_matrix"]
        )
        for actual, expected in zip(display_rect, observation["geometry"]["rect_display"]):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertTrue(payload["exchange_contract"]["annotation_observations_are_immutable"])
        self.assertFalse(payload["exchange_contract"]["confirmed_clash_or_clearance_established"])
        self.assertFalse(payload["exchange_contract"]["quantity_eligible"])

    def test_pdf_group_relationship_takes_precedence_over_geometry(self):
        cloud = _annotation(
            "cloud.1",
            10,
            "cloud",
            in_reply_to={"xref": 20, "generation": 0},
            reply_type="Group",
        )
        callout = _annotation("callout.1", 20, "callout", leader_target=(100, 100))

        pairs, abstentions = pair_annotation_observations([cloud, callout])

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["relationship"], "pdf_group_relationship")
        self.assertEqual(pairs[0]["epistemic_state"], "direct")
        self.assertEqual(abstentions, [])
        self.assertTrue(pairs[0]["markup_claim_only"])
        self.assertIsNone(pairs[0]["target_identity_ref"])
        self.assertFalse(pairs[0]["confirmed_clash"])

        text_box = _annotation("text-box.1", 40, "callout", leader_target=(100, 100))
        text_box["intent"] = None
        text_box["subject"] = "Text Box"
        text_box["in_reply_to"] = {"xref": 50, "generation": 0}
        text_box["reply_type"] = "Group"
        text_box_cloud = _annotation("cloud.2", 50, "cloud")
        text_box_pairs, text_box_abstentions = pair_annotation_observations(
            [text_box, text_box_cloud]
        )
        self.assertEqual(len(text_box_pairs), 1)
        self.assertEqual(text_box_pairs[0]["comment_form"], "text_box")
        self.assertEqual(text_box_abstentions, [])

    def test_unique_leader_target_pairs_but_ambiguous_geometry_abstains(self):
        callout = _annotation("callout.1", 20, "callout", leader_target=(5, 5))
        cloud = _annotation("cloud.1", 10, "cloud")
        pairs, abstentions = pair_annotation_observations([cloud, callout])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["relationship"], "unique_geometric_leader_target")
        self.assertTrue(pairs[0]["evidence"]["mutual_unique"])
        self.assertEqual(abstentions, [])

        second_cloud = _annotation("cloud.2", 30, "cloud")
        pairs, abstentions = pair_annotation_observations([cloud, second_cloud, callout])
        self.assertEqual(pairs, [])
        self.assertEqual(len(abstentions), 1)
        self.assertEqual(abstentions[0]["reason"], "ambiguous_geometric_pairing")

    def test_validator_rejects_engineering_authority_on_source_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.pdf"
            document = fitz.open()
            page = document.new_page(width=100, height=100)
            page.add_text_annot((20, 20), "claim")
            document.save(path)
            document.close()
            payload = extract_pdf_annotation_observations(path)

        changed = copy.deepcopy(payload)
        changed["annotations"][0]["accepted_target_ref"] = "route.1"
        errors = validate_pdf_annotation_observations(changed)
        self.assertTrue(any("accepted_target_ref is forbidden" in item for item in errors))

    def test_reviewed_truth_catalog_is_internally_closed(self):
        truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
        roles = Counter(item["role"] for item in truth["page_truth"])
        categories = Counter(item["category"] for item in truth["issue_pairs"])
        counts = truth["expected_counts"]
        self.assertEqual(len(truth["page_truth"]), counts["pages"])
        self.assertEqual(roles["divider"], counts["divider_pages"])
        self.assertEqual(roles["mechanical_piping_plan"], counts["mechanical_piping_plans"])
        self.assertEqual(roles["lube_system_shop_plan"], counts["lube_system_shop_plans"])
        self.assertEqual(len(truth["issue_pairs"]), counts["issue_pairs"])
        for category in (
            "vfd_clearance",
            "vehicle_exhaust_clash",
            "fire_damper_access",
            "huh_valve_access",
            "vent_conflict",
        ):
            self.assertEqual(categories[category], counts[category])
        self.assertEqual(
            sum(bool(item["missing_elevation_explicit"]) for item in truth["issue_pairs"]),
            counts["missing_elevation_comments"],
        )
        self.assertTrue(truth["contract"]["production_dispatch_by_fixture_fields_forbidden"])
        self.assertTrue(truth["contract"]["markup_claims_do_not_establish_clash_or_clearance"])

    def test_production_reader_contains_no_fixture_dispatch(self):
        source = inspect.getsource(mep_annotation_observations)
        self.assertNotIn("M&P mark-up", source)
        for reviewed_xref in (97, 115, 155, 164, 167, 169, 218, 221):
            self.assertNotIn(f"== {reviewed_xref}", source)

    def test_live_fixture_matches_frozen_source_and_reviewed_truth(self):
        truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
        stored = json.loads(OBSERVATIONS_PATH.read_text(encoding="utf-8"))
        extracted = extract_pdf_annotation_observations(SOURCE_PDF)

        self.assertEqual(extracted, stored)
        self.assertEqual(validate_pdf_annotation_observations(stored, SOURCE_PDF), [])
        self.assertEqual(extracted["document"]["source_pdf_sha256"], truth["source"]["sha256"])
        self.assertEqual(extracted["document"]["source_bytes"], truth["source"]["bytes"])
        self.assertEqual(extracted["summary"]["page_count"], truth["expected_counts"]["pages"])
        self.assertEqual(
            extracted["summary"]["annotation_count"],
            truth["expected_counts"]["pdf_annotations"],
        )
        self.assertEqual(
            extracted["summary"]["annotation_pair_count"],
            truth["expected_counts"]["issue_pairs"],
        )
        annotations_by_id = {item["id"]: item for item in extracted["annotations"]}
        observed_pairs = {
            (
                item["page_number"],
                annotations_by_id[item["cloud_annotation_ref"]]["source_object"]["xref"],
                annotations_by_id[item["callout_annotation_ref"]]["source_object"]["xref"],
                annotations_by_id[item["callout_annotation_ref"]]["text"],
            )
            for item in extracted["annotation_pairs"]
        }
        truth_pairs = {
            (
                item["page_number"],
                item["cloud_source_xref"],
                item["callout_source_xref"],
                item["text_exact"],
            )
            for item in truth["issue_pairs"]
        }
        self.assertEqual(observed_pairs, truth_pairs)
        self.assertEqual(extracted["pairing_abstentions"], [])


if __name__ == "__main__":
    unittest.main()
