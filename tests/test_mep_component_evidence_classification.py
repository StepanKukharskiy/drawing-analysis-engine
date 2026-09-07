import unittest

from src.drawing_engine.disciplines.mep.mep_component_evidence_classification import (
    classify_projected_components, validate_component_evidence_classification,
)


class MepComponentEvidenceClassificationTest(unittest.TestCase):
    def test_outlined_pair_alone_stays_unclassified_and_colour_only_supports(self):
        components = []
        composites = []
        fragments = []
        bindings = []
        dispositions = {}
        colors = ([1, 0, .25], [.4, .4, .4], [1, 0, .25], [.4, .4, .4])
        systems = ("heating_hot_water_supply", None, None, None)
        for index in range(4):
            component_ref = f"component.{index}"
            composite_ref = f"composite.{index}"
            fragment_ref = f"fragment.{index}"
            source_ref = f"source.{index}"
            components.append({
                "id": component_ref, "page_ref": "page.5",
                "composite_refs": [composite_ref], "source_segment_refs": [source_ref],
            })
            composites.append({
                "id": composite_ref, "page_ref": "page.5",
                "member_fragment_refs": [fragment_ref],
            })
            fragments.append({"id": fragment_ref, "style": {"stroke": colors[index]}})
            bindings.append({
                "component_ref": component_ref,
                "attributes": {
                    "route_system": {"state": "accepted" if systems[index] else "unknown",
                                     "value": systems[index], "relation_refs": ["m4.system"] if systems[index] else []},
                    "route_size": {"state": "accepted" if index == 2 else "unknown",
                                   "value": 2 if index == 2 else None,
                                   "relation_refs": ["m4.size"] if index == 2 else []},
                    "route_elevation": {"state": "unknown", "value": None,
                                        "relation_refs": []},
                },
            })
            dispositions[source_ref] = {
                "primary_disposition": "drawing_furniture" if index == 3 else "route_evidence",
                "candidate_roles": ["drawing_furniture"] if index == 3 else ["route_evidence"],
                "region_role": "main_plan_view",
            }
        recovery = {
            "page_ref": "page.5", "geometry_components": components,
            "geometry_components_sha256": "frozen", "post_geometry_M4_bindings": bindings,
        }
        payload = classify_projected_components(
            recovery=recovery, route_page={"fragments": fragments}, composites=composites,
            m4_relations=[], colour_mapping_certificates=[{
                "id": "colour.red", "state": "accepted_one_to_one_review_correlation",
                "native_stroke_rgb": [1, 0, .25], "system": "heating_hot_water_supply",
            }], source_dispositions=dispositions)
        self.assertEqual([
            "identified_mep_route", "unclassified_view_geometry",
            "supported_unidentified_mep_candidate", "non_route_drawing_content",
        ], [row["state"] for row in payload["classifications"]])
        self.assertIsNone(payload["classifications"][2]["identified_system"])
        self.assertFalse(payload["classifications"][1]["outlined_pair_alone_establishes_mep_route"])
        self.assertIsNone(payload["negative_promotion_gates"]["hatching"]
                          ["promoted_source_reference_count"])
        self.assertEqual([], validate_component_evidence_classification(payload))

    def test_negative_gate_counts_come_from_promoted_source_references(self):
        components = [
            {"id": "component.left", "page_ref": "page.5",
             "composite_refs": ["composite.left"], "source_segment_refs": ["source.left"]},
            {"id": "component.right", "page_ref": "page.5",
             "composite_refs": ["composite.right"], "source_segment_refs": ["source.right"]},
        ]
        composites = [
            {"id": "composite.left", "page_ref": "page.5",
             "member_fragment_refs": ["fragment.left"],
             "member_source_primitive_refs": ["source.left"]},
            {"id": "composite.right", "page_ref": "page.5",
             "member_fragment_refs": ["fragment.right"],
             "member_source_primitive_refs": ["source.right"]},
        ]
        fragments = [
            {"id": "fragment.left", "style": {"stroke": [1, 0, .25],
                                                   "width_display_points": .5,
                                                   "dash_pattern": "[] 0"}},
            {"id": "fragment.right", "style": {"stroke": [1, 0, .25],
                                                    "width_display_points": .8,
                                                    "dash_pattern": "[] 0"}},
        ]
        bindings = [{
            "component_ref": component["id"],
            "attributes": {
                "route_system": {"state": "accepted", "value": "heating_hot_water_supply",
                                 "relation_refs": [f"m4.{index}"]},
                "route_size": {"state": "unknown", "value": None, "relation_refs": []},
                "route_elevation": {"state": "unknown", "value": None, "relation_refs": []},
            },
        } for index, component in enumerate(components)]
        recovery = {
            "page_ref": "page.5", "geometry_components": components,
            "geometry_components_sha256": "frozen", "post_geometry_M4_bindings": bindings,
            "accepted_transitions": [{"composite_refs": ["composite.left", "composite.right"]}],
        }
        payload = classify_projected_components(
            recovery=recovery,
            route_page={"fragments": fragments, "crossings": [{
                "fragment_refs": ["fragment.left", "fragment.right"]}]},
            composites=composites, m4_relations=[], colour_mapping_certificates=[],
            source_dispositions={
                "source.left": {"primary_disposition": "route_evidence",
                                "candidate_roles": ["annotation_dimension"],
                                "region_role": "main_plan_view"},
                "source.right": {"primary_disposition": "route_evidence",
                                 "candidate_roles": ["route_evidence"],
                                 "region_role": "main_plan_view"},
            })
        gates = payload["negative_promotion_gates"]
        self.assertEqual(1, gates["dimension_or_annotation"]["promoted_source_reference_count"])
        self.assertEqual(2, gates["crossing_only_contact"]["promoted_source_reference_count"])
        self.assertEqual(2, gates["incompatible_style_near_join"]["promoted_source_reference_count"])
        self.assertEqual([], validate_component_evidence_classification(payload))


if __name__ == "__main__":
    unittest.main()
