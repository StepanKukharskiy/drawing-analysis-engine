#!/usr/bin/env python3
"""Build a frozen MEP network/HVAC checkpoint, persist it, and query its evidence.

Consumes existing automatic or assisted artifacts; this is not a PDF extraction
runner. The input execution mode remains visible in the persisted context.
"""

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.pipelines.generate_mep_automatic_items import _write
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import _canonical_sha256
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import build_mep_hvac_inventory
from src.drawing_engine.disciplines.mep.mep_item_catalog import validate_mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_network_assembly import build_mep_networks
from src.drawing_engine.disciplines.mep.mep_projected_identity_bindings import build_projected_identity_bindings
from src.drawing_engine.disciplines.mep.mep_projected_identity_inputs import load_equipment_identity_inputs, load_fitting_identity_inputs, _archive, _decode_archive
from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore


def build(*, source, registry_path, run_dir, output_dir, database, project_id, document_id):
    def progress(event):
        print(json.dumps(event), file=sys.stderr, flush=True)

    progress({'phase': 'snapshot_input_load'})
    implementation = {name: _file_sha256(ROOT / name) for name in (
        "src/drawing_engine/disciplines/mep/mep_network_assembly.py", "src/drawing_engine/disciplines/mep/mep_hvac_inventory.py", "src/drawing_engine/project/project_knowledge_store.py",
        "src/drawing_engine/disciplines/mep/mep_cross_sheet_runs.py", "tools/mep_project.py",
        "src/drawing_engine/disciplines/mep/mep_boundary_connection_replay.py", "src/drawing_engine/disciplines/mep/mep_native_boundary_connections.py",
        "src/drawing_engine/disciplines/mep/mep_native_bend_connections.py", "src/drawing_engine/disciplines/mep/mep_native_branch_connections.py",
        "src/drawing_engine/disciplines/mep/mep_projected_identity_bindings.py", "src/drawing_engine/disciplines/mep/mep_projected_identity_inputs.py",
        "src/drawing_engine/disciplines/mep/mep_equipment_identity.py", "src/drawing_engine/disciplines/mep/mep_fitting_hypotheses.py", "src/drawing_engine/core/pdf_native_metadata.py",
        'src/drawing_engine/disciplines/mep/mep_outlined_route_composites.py', 'src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py',
        'src/drawing_engine/disciplines/mep/mep_automatic_target_binding.py', 'src/drawing_engine/disciplines/mep/mep_native_boundary_queries.py',
        'src/drawing_engine/disciplines/mep/mep_native_equipment_observations.py', 'src/drawing_engine/core/dimension_attachment.py', 'src/drawing_engine/disciplines/mep/mep_attribute_binding.py')}
    if (run_dir / 'package-binding-replay-manifest.json').exists():
        implementation['src/drawing_engine/disciplines/mep/mep_native_cap_replay.py'] = _file_sha256(ROOT / 'src/drawing_engine/disciplines/mep/mep_native_cap_replay.py')
    registry = json.loads(registry_path.read_text())
    digest = _file_sha256(source)
    if digest != registry["document"].get("source_pdf_sha256"):
        raise ValueError("source PDF does not match the frozen registry")
    artifacts = {"sheet-registry": registry}
    for name in ("route-observations", "terminology-proposals", "attribute-bindings",
                 "cross-sheet-runs", "item-catalog"):
        artifacts[name] = json.loads((run_dir / (name + ".json")).read_text())
    for name in ("automatic-discovery", "bounded-local-3d", "outlined-route-composites", "outlined-route-connections",
                 "review-outcomes", 'package-policy-manifest', 'package-interpreters-manifest',
                 'package-binding-replay-manifest', 'stroke-ownership-queries'):
        path = run_dir / (name + ".json")
        if path.exists():
            artifacts[name] = json.loads(path.read_text())
    outcomes = artifacts.get('review-outcomes')
    if outcomes is not None:
        if (outcomes.get('layer') != 'mep_bounded_review_outcomes' or outcomes.get('document') != registry['document']
                or outcomes.get('quantity_eligible') is not False):
            raise ValueError('review outcomes do not match the frozen document or authority boundary')
        for name, dependency_digest in outcomes.get('input_payload_sha256', {}).items():
            if name not in artifacts or _canonical_sha256(artifacts[name]) != dependency_digest:
                raise ValueError('review outcomes do not reference this frozen artifact: ' + name)
        if 'elevation_applicability' in outcomes:
            from src.drawing_engine.disciplines.mep.mep_attribute_binding import elevation_applicability_outcomes
            expected = elevation_applicability_outcomes(terminology=artifacts['terminology-proposals'],
                bindings=artifacts['attribute-bindings'], graph=artifacts['route-observations'],
                boundary_connections=artifacts['outlined-route-connections'])
            if _canonical_sha256(expected) != _canonical_sha256(outcomes['elevation_applicability']):
                raise ValueError('elevation applicability does not replay from this frozen M4/source scope')
        ids = [row['id'] for row in outcomes['outcomes']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate review outcome IDs')
        if any(row.get('quantity_eligible') is not False
               or row.get('physical_continuation_established') is not False
               or row.get('physical_item_identity_established') is not False
               or row.get('engineer_approved') not in (None, False) for row in outcomes['outcomes']):
            raise ValueError('diagnostic review outcomes cannot grant physical or quantity authority')
        relations = {r['id']: r for r in artifacts['attribute-bindings']['relations']}
        connections = {r['id']: r for r in artifacts.get('outlined-route-connections', {}).get('connections', [])}
        elevation_applicability = {r['id']: r for r in outcomes.get('elevation_applicability', {}).get('outcomes', [])}
        for row in outcomes['outcomes']:
            if row['state'] != 'accepted':
                continue
            if row['subject_kind'] == 'connection':
                connection = connections.get(row.get('source_boundary_connection_ref'))
                if (connection is None or connection['state'] != 'accepted'
                        or connection['page_ref'] != row.get('page_ref')
                        or connection['id'] not in row.get('subject_refs', [])):
                    raise ValueError('accepted connection outcome lacks matching frozen boundary evidence')
                continue  # The network builder below independently replays this certificate.
            if row['subject_kind'] != 'annotation_binding':
                raise ValueError('accepted diagnostic subject has no supported identity certificate')
            refs = row.get('accepted_relation_refs', [])
            semantic = lambda value: {k: v for k, v in value.items() if k not in {'raw_text', 'terminology_entry_ref'}}
            direct = bool(refs) and not any(ref not in relations or relations[ref]['state'] != 'accepted'
                or relations[ref]['relation_type'] != row.get('relation_type')
                or relations[ref]['page_ref'] != row.get('page_ref')
                or not set(relations[ref]['target_refs']).intersection(row['subject_refs'])
                or semantic(relations[ref]['candidate']) != semantic(row.get('expected_candidate', {})) for ref in refs)
            applicability_refs = row.get('accepted_applicability_outcome_refs', [])
            applicability = row.get('relation_type') == 'route_elevation' and bool(applicability_refs)
            for ref in applicability_refs:
                outcome = elevation_applicability.get(ref)
                accepted_targets = [] if outcome is None else [target for target in outcome['candidate_extents']
                    if target['state'] == 'accepted_straight_annotation_extent'
                    and target['target_ref'] in row['subject_refs']]
                if (outcome is None or outcome['page_ref'] != row.get('page_ref')
                        or semantic(outcome['candidate']) != semantic(row.get('expected_candidate', {}))
                        or not accepted_targets
                        or not set(row.get('accepted_extent_connection_refs', [])).issubset({connection
                            for target in accepted_targets for connection in target['projected_path_connection_refs']})):
                    applicability = False
            if not direct and not applicability:
                raise ValueError('accepted annotation outcome lacks matching frozen M4 authority')
    errors = validate_mep_item_catalog(artifacts["item-catalog"])
    if (errors or artifacts["item-catalog"]["m4_contract_ref"]["payload_sha256"] != _canonical_sha256(artifacts["attribute-bindings"])
            or artifacts["item-catalog"]["m1_contract_ref"]["payload_sha256"] != _canonical_sha256(registry)):
        raise ValueError("catalog does not match this frozen M4 payload: " + "; ".join(errors))
    companion, fitting_network_replay = None, None
    identity_inputs = {}
    if (run_dir / 'equipment-2d-identities.json').exists():
        payload, replay, portable = load_equipment_identity_inputs(run_dir / 'equipment-2d-identities.json',
            terminology=artifacts['terminology-proposals'])
        identity_inputs.update(equipment_identity=payload, equipment_replay=replay)
        artifacts.update(portable)
    if (run_dir / 'fitting-hypotheses.json.gz').exists():
        payload, replay, portable = load_fitting_identity_inputs(run_dir / 'fitting-hypotheses.json.gz')
        identity_inputs.update(fitting_hypotheses=payload, fitting_replay=replay)
        fitting_network_replay = {'payload': payload, **replay}
        artifacts.update(portable)
    if identity_inputs:
        companion = build_projected_identity_bindings(attribute_bindings=artifacts['attribute-bindings'],
            terminology=artifacts['terminology-proposals'], **identity_inputs)
        artifacts['projected-identity-bindings'] = companion
        # Equipment's complete native query archive is no longer needed after
        # its replay. Keep its compressed portable artifact, not a second large
        # live object tree while the independent network replay runs.
        identity_inputs.clear()
        del payload, replay, portable
        progress({'phase': 'projected_equipment_and_fitting_replayed'})
    if (run_dir / 'native-metadata.json').exists():
        metadata = json.loads((run_dir / 'native-metadata.json').read_text())
        if metadata.get('document', {}).get('source_pdf_sha256') != digest:
            raise ValueError('native metadata inventory belongs to another source')
        artifacts['native-metadata'] = metadata
    trace_replay = None
    if (run_dir / 'trace-completion.json').exists():
        payload = json.loads((run_dir / 'trace-completion.json').read_text())
        queries = json.loads(gzip.decompress((run_dir / 'trace-source-queries.json.gz').read_bytes()))
        manifest = json.loads((run_dir / 'trace-manifest.json').read_text())
        if (manifest['source_queries_file_sha256'] != _file_sha256(run_dir / 'trace-source-queries.json.gz')
                or manifest['payload_sha256'] != _canonical_sha256(payload)):
            raise ValueError('trace completion archive differs from its frozen manifest')
        artifacts.update({'trace-completion': payload,
            'trace-source-queries': {'schema_version': '0.1.0', 'layer': 'mep_projected_trace_native_inputs',
                'original_payload_sha256': _canonical_sha256(queries), **queries},
            'trace-manifest': {'schema_version': '0.1.0', 'layer': 'mep_projected_trace_replay_manifest',
                'original_payload_sha256': _canonical_sha256(manifest), **manifest}})
        trace_replay = dict(graph=artifacts['route-observations'], composites=artifacts['outlined-route-composites'],
            boundary_connections=artifacts['outlined-route-connections'], source_queries=queries['source_queries'])
        if payload.get('attribute_bindings_sha256') is not None:
            if payload['attribute_bindings_sha256'] != _canonical_sha256(artifacts['attribute-bindings']):
                raise ValueError('trace annotation ownership differs from the frozen M4 artifact')
            trace_replay['attribute_bindings'] = artifacts['attribute-bindings']
    progress({'phase': 'projected_network_replay'})
    networks = build_mep_networks(sheet_registry=registry, route_graph=artifacts["route-observations"],
        attribute_bindings=artifacts["attribute-bindings"], cross_sheet_runs=artifacts["cross-sheet-runs"],
        boundary_connections=artifacts.get('outlined-route-connections'),
        projected_identity_bindings=companion, fitting_replay=fitting_network_replay,
        trace_completion=artifacts.get('trace-completion'), trace_replay=trace_replay)
    progress({'phase': 'projected_network_replayed', 'summary': networks['summary']})
    hvac = build_mep_hvac_inventory(terminology=artifacts["terminology-proposals"],
        attribute_bindings=artifacts["attribute-bindings"], projected_identity_bindings=companion)
    artifacts.update({"network-hierarchy": networks, "hvac-inventory": hvac})
    context = {"execution_mode": artifacts.get("automatic-discovery", {}).get("execution_mode", "frozen_artifact_replay"),
        "implementation_sha256": implementation}
    upstream_context = run_dir / "run-context.json"
    if upstream_context.exists():
        context["upstream_run_context"] = json.loads(upstream_context.read_text())
    ownership = artifacts.get('stroke-ownership-queries')
    if ownership is not None:
        expected = context.get('upstream_run_context', {}).get('stroke_ownership_rebind', {})
        if (ownership.get('document') != registry['document']
                or _canonical_sha256(ownership) != expected.get('capture_payload_sha256')):
            raise ValueError('stroke ownership queries differ from the frozen binding inputs')
        queries = {q['id']: q for page in ownership['pages'] for q in page['queries']}
        for evidence in artifacts['attribute-bindings']['binding_evidence']:
            roles = evidence.get('scoped_stroke_ownership')
            if roles and (roles.get('source_query_ref') not in queries
                    or _canonical_sha256(queries[roles['source_query_ref']]) != roles['source_query_sha256']):
                raise ValueError('binding stroke role lost its exact native query evidence')
    policy = artifacts.get('package-policy-manifest')
    if policy is not None:
        upstream = context.get('upstream_run_context')
        expected_pages = sorted(p['page_number'] for p in registry['pages'])
        if (policy.get('document') != registry['document'] or not upstream
                or policy.get('current_interpretation_context') != upstream
                or policy.get('current_interpretation_context_sha256') != _canonical_sha256(upstream)
                or policy.get('execution_page_numbers') != expected_pages
                or upstream.get('parameters', {}).get('page_numbers') != expected_pages
                or policy.get('pilot_or_prior_results_unioned') is not False):
            raise ValueError('package policy manifest differs from this uniform frozen execution')
    rebind = artifacts.get('package-binding-replay-manifest')
    if rebind is not None:
        from src.drawing_engine.disciplines.mep.mep_native_cap_replay import load_native_cap_manifest, replay_native_cap_rejections, _artifact_path
        upstream = context.get('upstream_run_context', {})
        stage = upstream.get('attribute_rebind', {})
        cap_directory = run_dir / 'native-cap-replay'
        prior_terms = json.loads((cap_directory / 'upstream-terminology.json').read_text())
        cap_manifest = load_native_cap_manifest(cap_directory, registry=registry, terminology=prior_terms,
                                               upstream_context=rebind['source_run_context'])
        if (rebind.get('document') != registry['document']
                or cap_manifest.get('authority') != {'negative_cap_witness_only': True,
                    'positive_acceptance_authority': False, 'quantity_eligible': False}
                or rebind.get('current_run_context_sha256') != _canonical_sha256(upstream)
                or rebind.get('source_run_context_sha256') != _canonical_sha256(rebind['source_run_context'])
                or stage.get('source_phase_context') != rebind['source_run_context']
                or rebind.get('native_cap_manifest_sha256') != _canonical_sha256(cap_manifest)
                or stage.get('native_cap_manifest_sha256') != _canonical_sha256(cap_manifest)
                or rebind.get('binding_result_rows_reused') is not False
                or rebind.get('reviewed_selectors_consumed') is not False
                or rebind.get('physical_continuation_established') is not False
                or rebind.get('quantity_eligible') is not False
                or policy.get('attribute_rebind_manifest_sha256') != _canonical_sha256(rebind)):
            raise ValueError('attribute rebind lineage differs from its native cap replay')
        for name, expected in {**rebind['unchanged_geometry_file_sha256'], **rebind['derived_file_sha256']}.items():
            if name not in artifacts or _file_sha256(run_dir / (name + '.json')) != expected:
                raise ValueError('attribute rebind output changed: ' + name)
        artifacts['native-cap-replay-manifest'] = cap_manifest
        artifacts['native-cap-upstream-terminology'] = prior_terms
        portable_pages = []
        for page in cap_manifest['pages']:
            record = page['native_archive']
            archive = _archive(_artifact_path(cap_directory, record), record['file_sha256'], registry['document'])
            payload = json.loads(_artifact_path(cap_directory, page['outcomes']).read_text())
            if (archive['payload_sha256'] != record['payload_sha256']
                    or _canonical_sha256(payload) != page['outcomes']['payload_sha256']):
                raise ValueError('native cap replay artifact changed')
            native = _decode_archive(archive, registry['document'])
            replay_native_cap_rejections(payload, native_page=native, registry=registry)
            progress({'phase': 'native_cap_page_replayed', 'page_number': page['page_number']})
            prefix = 'native-cap-page-' + str(page['page_number'])
            artifacts[prefix + '-archive'] = archive
            artifacts[prefix + '-outcomes'] = payload
            portable_pages.append({'page_ref': page['page_ref'], 'page_number': page['page_number'],
                'native_archive_artifact': prefix + '-archive', 'outcomes_artifact': prefix + '-outcomes'})
            del native
        artifacts['native-cap-portable-inputs'] = {'schema_version': '0.1.0',
            'layer': 'mep_native_cap_portable_inputs', 'document': registry['document'],
            'manifest_artifact': 'native-cap-replay-manifest',
            'manifest_payload_sha256': _canonical_sha256(cap_manifest),
            'upstream_terminology_artifact': 'native-cap-upstream-terminology',
            'pages': portable_pages, 'quantity_eligible': False}
    interpreters = artifacts.get('package-interpreters-manifest')
    if interpreters is not None:
        equipment = artifacts.get('equipment-2d-identities')
        fitting = artifacts.get('fitting-hypotheses')
        calibration = artifacts.get('equipment-2d-calibration-identities')
        fitting_inputs = artifacts.get('fitting-replay', {}).get('source_queries', [])
        if (interpreters.get('document') != registry['document']
                or interpreters.get('execution_page_numbers') != sorted(p['page_number'] for p in registry['pages'])
                or interpreters.get('source_run_context_sha256') != _canonical_sha256(context.get('upstream_run_context'))
                or interpreters.get('pilot_or_prior_result_rows_unioned') is not False
                or interpreters.get('quantity_eligible') is not False
                or not equipment or not fitting or not calibration
                or interpreters.get('equipment_payload_sha256') != _canonical_sha256(equipment)
                or interpreters.get('fitting_payload_sha256') != _canonical_sha256(fitting)
                or interpreters.get('equipment_calibration_payload_sha256') != _canonical_sha256(calibration)
                or interpreters.get('equipment_native_file_sha256') != equipment.get('native_replay_sha256')
                or [entry['sha256'] for entry in fitting_inputs] != [interpreters.get('fitting_input_queries_file_sha256')]):
            raise ValueError('package interpreter manifest differs from its frozen replay inputs')
    # Fail before either persistence or file writes if output would overwrite history.
    generated_names = ['network-hierarchy', 'hvac-inventory'] + (['projected-identity-bindings'] if companion else [])
    for name in generated_names:
        path = output_dir / (name + ".json")
        if path.exists() and json.loads(path.read_text()) != artifacts[name]:
            raise ValueError("checkpoint output differs; choose a new output directory")
    if implementation != {name: _file_sha256(ROOT / name) for name in implementation}:
        raise RuntimeError("implementation changed during replay; rerun before importing")
    with ProjectKnowledgeStore(database) as store:
        progress({'phase': 'snapshot_transaction_start'})
        snapshot = store.import_snapshot(project_id=project_id, document_id=document_id,
            source_sha256=digest, context=context, artifacts=artifacts, progress=progress)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in generated_names:
        _write(output_dir / (name + ".json"), artifacts[name])
    report = {"schema_version": "0.1.0", "layer": "mep_project_checkpoint",
        "snapshot_id": snapshot, "project_id": project_id, "document_id": document_id,
        "source_pdf_sha256": digest, "execution_mode": context["execution_mode"],
        "network_summary": networks["summary"], "hvac_summary": hvac["summary"],
        "quantity_eligible": False}
    _write(output_dir / "checkpoint.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="replay frozen artifacts and import one complete project snapshot")
    for name in ("source", "registry", "run-dir", "output-dir"):
        build_parser.add_argument("--" + name, required=True, type=Path)
    query_parser = sub.add_parser("query", help="query immutable records; no state inference")
    for name in ("record-type", "state", "page-ref", "source-id"):
        query_parser.add_argument("--" + name)
    query_parser.add_argument("--limit", type=int, default=100)
    query_parser.add_argument("--offset", type=int, default=0)
    evidence_parser = sub.add_parser("evidence", help="one-hop provenance links with unresolved/ambiguous targets")
    evidence_parser.add_argument("--node-id", required=True)
    node_parser = sub.add_parser('node', help='read the exact immutable body at an indexed evidence pointer')
    node_parser.add_argument('--node-id', required=True)
    for child in (build_parser, query_parser, evidence_parser, node_parser):
        child.add_argument("--database", required=True, type=Path)
        child.add_argument("--project", required=True)
        child.add_argument("--document", required=True)
    for child in (query_parser, evidence_parser, node_parser):
        child.add_argument("--snapshot", help="explicit historical snapshot; defaults to the active revision")
    args = parser.parse_args()
    if args.command == "build":
        result = build(source=args.source, registry_path=args.registry, run_dir=args.run_dir,
            output_dir=args.output_dir, database=args.database, project_id=args.project, document_id=args.document)
    else:
        if not args.database.is_file():
            parser.error("database does not exist")
        with ProjectKnowledgeStore(args.database) as store:
            scope = {"project_id": args.project, "document_id": args.document, "snapshot_id": args.snapshot}
            if args.command == "query":
                result = store.query(**scope, record_type=args.record_type, state=args.state,
                    page_ref=args.page_ref, source_id=args.source_id, limit=args.limit, offset=args.offset)
                for row in result:
                    row.pop("payload_json")
            elif args.command == 'node':
                result = store.node_payload(**scope, node_id=args.node_id)
            else:
                result = store.neighbors(**scope, node_id=args.node_id)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
