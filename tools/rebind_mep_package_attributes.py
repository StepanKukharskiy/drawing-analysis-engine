#!/usr/bin/env python3
"""Rebind attributes after fresh negative cap evidence; keep source geometry frozen."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drawing_engine.pipelines.generate_mep_automatic_items import run, _write
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registry', 'native-text', 'annotations', 'ocr', 'upstream',
                 'native-cap-replay', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    inputs = {name: load(getattr(args, name)) for name in ('registry', 'native_text', 'annotations', 'ocr')}
    upstream_context = load(args.upstream / 'run-context.json')
    upstream_policy = load(args.upstream / 'package-policy-manifest.json')
    pages = sorted(row['page_number'] for row in inputs['registry']['pages'])
    if (upstream_policy['document'] != inputs['registry']['document']
            or upstream_policy['current_interpretation_context_sha256'] != _sha256(upstream_context)
            or upstream_policy['current_interpretation_context'] != upstream_context
            or upstream_policy['execution_page_numbers'] != pages
            or upstream_context['parameters']['page_numbers'] != pages
            or upstream_policy['pilot_or_prior_results_unioned'] is not False):
        raise ValueError('attribute rebind requires one verified all-page source policy')
    if args.output_dir.exists():
        raise ValueError('attribute rebind must use a new output directory')
    params = upstream_context['parameters']
    run(source=args.source, output_dir=args.output_dir, rebind_from=args.upstream,
        native_cap_replay_dir=args.native_cap_replay, page_numbers=pages,
        region_size=params['region_size'], region_budget=params['region_budget'],
        minimum_member_length=params['minimum_member_length'], **inputs)
    context = load(args.output_dir / 'run-context.json')
    unchanged = {}
    for name in ('route-observations', 'outlined-route-composites', 'outlined-route-connections'):
        old = _file_sha256(args.upstream / (name + '.json'))
        if _file_sha256(args.output_dir / (name + '.json')) != old:
            raise ValueError('attribute rebind changed frozen geometry: ' + name)
        unchanged[name] = old
        # Identical frozen geometry remains a single immutable disk body. All
        # producers replace files atomically, so later runs cannot edit its peer.
        path = args.output_dir / (name + '.json')
        linked = path.with_suffix('.json.linked')
        os.link(args.upstream / path.name, linked)
        linked.replace(path)
    shutil.copytree(args.native_cap_replay, args.output_dir / 'native-cap-replay')
    shutil.copy2(args.upstream / 'terminology-proposals.json',
                 args.output_dir / 'native-cap-replay/upstream-terminology.json')
    metadata = args.upstream / 'native-metadata.json'
    if metadata.exists():
        shutil.copy2(metadata, args.output_dir / metadata.name)
    trace_names = ['trace-completion.json', 'trace-source-queries.json.gz', 'trace-manifest.json']
    reused_trace = {}
    if any((args.upstream / name).exists() for name in trace_names):
        if not all((args.upstream / name).exists() for name in trace_names):
            raise ValueError('upstream trace stage is incomplete')
        for name in trace_names:
            shutil.copy2(args.upstream / name, args.output_dir / name)
            reused_trace[name] = _file_sha256(args.upstream / name)
    derived = {name: _file_sha256(args.output_dir / (name + '.json')) for name in
        ('terminology-proposals', 'attribute-bindings', 'cross-sheet-runs',
         'bounded-local-3d', 'item-catalog', 'automatic-discovery')}
    manifest = {'schema_version': '0.1.0', 'layer': 'mep_package_attribute_rebind',
        'document': inputs['registry']['document'], 'execution_page_numbers': pages,
        'source_run_context': upstream_context, 'source_run_context_sha256': _sha256(upstream_context),
        'current_run_context_sha256': _sha256(context),
        'source_phase_policy_sha256': _sha256(upstream_policy),
        'unchanged_geometry_file_sha256': unchanged, 'derived_file_sha256': derived,
        'reused_trace_files_sha256': reused_trace,
        'trace_reuse_basis': 'byte-identical M3/M3.5/boundary inputs; independent M5C replay required',
        'native_cap_manifest_sha256': context['attribute_rebind']['native_cap_manifest_sha256'],
        'binding_result_rows_reused': False, 'reviewed_selectors_consumed': False,
        'physical_continuation_established': False, 'quantity_eligible': False}
    _write(args.output_dir / 'package-binding-replay-manifest.json', manifest)
    policy = {**upstream_policy, 'current_interpretation_context': context,
        'current_interpretation_context_sha256': _sha256(context),
        'source_phase_policy': upstream_policy,
        'attribute_rebind_manifest_sha256': _sha256(manifest),
        'policy_scope': 'frozen all-page source geometry; fresh native cap negatives and uniform attribute rebind'}
    _write(args.output_dir / 'package-policy-manifest.json', policy)
    print(json.dumps({'phase': 'attribute_rebind_complete',
        'summary': load(args.output_dir / 'automatic-discovery.json')['summary']}), flush=True)


if __name__ == '__main__':
    main()
