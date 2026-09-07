"""HVAC review register: lexical classes and accepted M4 targets stay separate.

This adapter consumes a separately replayed 2D body/tag gate when present;
the existing equipment-port gate remains independent. Native and OCR
source records remain unchanged, including confidence, transforms and roles.
"""

from collections import Counter, defaultdict
from copy import deepcopy

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import _canonical_sha256, _stable_id
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals


VERSION = "0.1.0"
LAYER = "mep_hvac_review_inventory"
EQUIPMENT_CLASSES = {"AHU": "air_handling_unit", "FCU": "fan_coil_unit",
    "HUH": "unit_heater", "EF": "exhaust_fan", "SF": "supply_fan", "RF": "return_fan"}
HVAC_SYSTEMS = {"chilled_water_supply", "chilled_water_return", "heating_hot_water_supply",
    "heating_hot_water_return", "condenser_water_supply", "condenser_water_return", "vehicle_exhaust"}


def build_mep_hvac_inventory(*, terminology, attribute_bindings, projected_identity_bindings=None):
    errors = validate_mep_terminology_proposals(terminology) + validate_mep_attribute_bindings(attribute_bindings)
    if errors:
        raise ValueError("invalid HVAC upstream inputs: " + "; ".join(errors))
    if (terminology["document"].get("document_key") != attribute_bindings["document"].get("document_key") or
            attribute_bindings["m2_contract_ref"]["payload_sha256"] != _canonical_sha256(terminology)):
        raise ValueError("HVAC inputs do not reference the same frozen M2 payload")
    observations = {r["id"]: r for r in terminology["source_observations"]}
    projected = defaultdict(list)
    assessments = defaultdict(list)
    if projected_identity_bindings is not None:
        if (projected_identity_bindings.get('layer') != 'mep_projected_identity_bindings'
                or projected_identity_bindings['document'] != attribute_bindings['document']
                or projected_identity_bindings['input_payload_sha256'].get('attribute-bindings') != _canonical_sha256(attribute_bindings)
                or projected_identity_bindings['input_payload_sha256'].get('terminology-proposals') != _canonical_sha256(terminology)):
            raise ValueError('HVAC projected identities do not reference this frozen M2/M4')
        for identity in projected_identity_bindings['equipment_bindings']:
            assessments[identity['proposal_ref']].append(identity)
            if identity['state'] == 'accepted':
                projected[identity['proposal_ref']].append(identity)
    relations = defaultdict(list)
    for relation in attribute_bindings["relations"]:
        relations[relation["proposal_ref"]].append(relation)
    systems = [r for r in attribute_bindings["relations"]
               if r["state"] == "accepted" and r["relation_type"] == "route_system"]
    entries = []
    for proposal in terminology["proposals"]:
        candidate = proposal["candidate"]
        kind, category = candidate.get("kind"), candidate.get("category")
        if category is None and "damper" in str(kind):
            category = "damper"
        if category is None and kind in {"fitting", "elbow", "tee", "reducer"}:
            category = "fitting"
        supported = True
        if kind == "equipment_tag":
            item_type = "equipment"
            item_class = EQUIPMENT_CLASSES.get(candidate.get("equipment_class_token"))
            supported = item_class is not None
            item_class = item_class or "unsupported_equipment_class"
        elif kind == "rectangular_duct_size":
            item_type, item_class = "duct", "rectangular_duct"
        elif category in {"damper", "fitting", "valve"}:
            item_type, item_class = "accessory", kind
        else:
            continue
        bound = [r for r in relations[proposal["id"]] if r["state"] == "accepted"]
        fragment_refs = {ref for r in bound for ref in r["target_fragment_refs"]}
        system_relations = [r for r in systems if r["page_ref"] == proposal["page_ref"]
                            and fragment_refs.intersection(r["target_fragment_refs"])]
        hvac_known = item_type in {"equipment", "duct"} or category == "damper" or any(
            r["candidate"].get("kind") in HVAC_SYSTEMS for r in system_relations)
        reasons = set(proposal["reasons"])
        reasons.update(reason for r in relations[proposal["id"]] if r["state"] != "accepted"
                       for reason in r["reasons"])
        if not supported:
            reasons.add("equipment_class_not_supported")
        if not bound:
            reasons.add("equipment_body_and_port_identity_unresolved" if item_type == "equipment"
                        else "geometric_item_target_unresolved")
        if not hvac_known:
            reasons.add("hvac_system_applicability_unresolved")
        entries.append({"id": _stable_id("mep_hvac_review_item", proposal["id"]),
            "record_type": "mep_hvac_review_item", "record_version": VERSION,
            "page_ref": proposal["page_ref"], "item_type": item_type, "item_class": item_class,
            "class_supported": supported, "class_basis": "frozen_m2_interpretation",
            "proposal_ref": proposal["id"], "candidate": deepcopy(candidate),
            "source_observation_refs": sorted(ref for ref in proposal["evidence_refs"] if ref in observations),
            "source_relation_refs": sorted(r["id"] for r in bound),
            "target_refs": sorted({ref for r in bound for ref in r["target_refs"]}),
            "source_fragment_refs": sorted(fragment_refs),
            "system_relation_refs": sorted(r["id"] for r in system_relations),
            "state": "bound_to_m4_target" if bound and supported and hvac_known else "unresolved",
            "epistemic_state": "derived" if bound else proposal["epistemic_state"],
            "unresolved_reasons": sorted(reasons),
            "authority": {"page_local_binding_established": bool(bound),
                "equipment_port_binding_established": any(r["relation_type"] == "equipment_endpoint" for r in bound),
                "physical_item_identity_established": False, "physical_count_established": False,
                "quantity_eligible": False}, "quantity_eligible": False})
        if assessments[proposal['id']]:
            # Failed searches remain evidence, never an accepted body or port.
            entries[-1]['projected_identity_assessment_refs'] = sorted(r['id'] for r in assessments[proposal['id']])
            entries[-1]['unresolved_reasons'] = sorted(set(entries[-1]['unresolved_reasons']) |
                {reason for r in assessments[proposal['id']] if r['state'] != 'accepted' for reason in r['reasons']})
        identities = projected.get(proposal['id'], [])
        if identities:
            if len(identities) != 1 or item_type != 'equipment' or not supported:
                raise ValueError('HVAC projected body identity is not mutually unique or supported')
            identity = identities[0]
            if (identity['page_ref'] != proposal['page_ref'] or identity['equipment_tag'] != candidate.get('tag')
                    or identity['authority'].get('projected_body_tag_identity_established') is not True
                    or any(identity['authority'].get(key) is not False for key in
                        ('equipment_port_binding_established', 'physical_item_identity_established', 'physical_placement_established', 'physical_count_established', 'quantity_eligible'))):
                raise ValueError('HVAC projected body certificate has invalid ownership or authority')
            entry = entries[-1]
            entry.update(state='identified_2d' if not bound else entry['state'], epistemic_state='inferred',
                projected_identity_relation_refs=[identity['id']], projected_body_motif_ref=identity['body_motif_ref'],
                projected_body_bbox_display=identity['body_bbox_display'],
                projected_body_source_primitive_refs=identity['source_primitive_refs'],
                source_proposal_reasons=deepcopy(proposal['reasons']))
            # The new certificate resolves 2D ownership only. Original M2 facts
            # remain immutable and equipment_endpoint / M7 inputs stay unchanged.
            entry['unresolved_reasons'] = sorted((set(entry['unresolved_reasons'])
                - set(identity['resolved_context_abstentions'])
                - {'equipment_body_and_port_identity_unresolved', 'no_valid_geometric_target', 'upstream_proposal_abstained_or_conflicted'})
                | {'equipment_ports_unresolved', 'physical_item_identity_unresolved'})
            entry['authority']['projected_body_tag_identity_established'] = True
    entries.sort(key=lambda r: r["id"])
    used_observations = {ref for r in entries for ref in r["source_observation_refs"]}
    return {"schema_version": VERSION, "layer": LAYER, "document": deepcopy(attribute_bindings["document"]),
        "input_payload_sha256": {"m2": _canonical_sha256(terminology), "m4": _canonical_sha256(attribute_bindings),
            **({'projected_identity_bindings': _canonical_sha256(projected_identity_bindings)} if projected_identity_bindings is not None else {})},
        "source_observations": [deepcopy(observations[ref]) for ref in sorted(used_observations)],
        "items": entries,
        "coverage": {"supported_equipment_tokens": sorted(EQUIPMENT_CLASSES),
            "rectangular_duct_annotation_supported": True,
            "symbol_only_equipment_identity_supported": False, "air_terminal_identity_supported": False,
            "round_duct_identity_supported": False, "complete_item_inventory_established": False},
        "summary": {"review_entry_count": len(entries),
            "item_type_counts": dict(Counter(r["item_type"] for r in entries)),
            "state_counts": dict(Counter(r["state"] for r in entries)), "physical_item_count": None,
            **({'projected_equipment_identity_count': sum(r.get('authority', {}).get('projected_body_tag_identity_established', False) for r in entries)}
               if projected_identity_bindings is not None else {})},
        "quantity_eligible": False}
