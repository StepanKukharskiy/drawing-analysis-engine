#!/usr/bin/env python3
"""Fresh per-sheet diagnostic execution; never reuse a legacy route cache.

Subprocess isolation bounds native-PDF memory. Immutable stage receipts permit
resume only against identical inputs/code and output bytes. An explicit preserved
run is copied by reference, not recomputed or granted additional authority.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.operations.run_artifact_job import run as run_artifact_job


def receipt_valid(receipt, context):
    return (receipt['context'] == context and all(
        Path(p).is_file() and _file_sha256(Path(p)) == digest
        for p, digest in receipt['artifacts'].items()))


def stage(folder, name, command, required):
    context = {'command': list(map(str, command)),
               'code': _file_sha256(Path(command[2]))}
    marker = folder / (name + '.receipt.json')
    if marker.exists():
        if not receipt_valid(read(marker), context):
            raise ValueError('stale stage receipt: ' + str(marker))
        return
    started = time.time()
    print(json.dumps({'page': folder.name, 'stage': name, 'state': 'running'}), flush=True)
    with (folder / (name + '.log')).open('w') as log:
        run_artifact_job(list(map(str, command)), outputs=[folder], cwd=ROOT,
                         stdout=log, stderr=subprocess.STDOUT, required_outputs=required)
    write(marker, {'context': context, 'elapsed_seconds': time.time()-started,
                   'artifacts': {str(p.resolve()): _file_sha256(p) for p in required}})
    print(json.dumps({'page': folder.name, 'stage': name, 'state': 'finished'}), flush=True)


def process(args, scope, frozen):
    number = scope['page_number']
    folder = args.output / f'page-{number:03d}'
    folder.mkdir(exist_ok=True)
    result_path = folder / (args.phase + '.json')
    result = {'page': number, 'page_ref': scope['page_ref'], 'state': 'running',
              'complete_pipe_coverage': False, 'engineer_accepted': False,
              'installed_length': None, 'purchase_length': None}
    cmd = lambda script: [sys.executable, '-B', str(ROOT/'tools'/script)]
    try:
        preserved = frozen.get(str(number))
        if preserved:
            result.update(state='preserved_frozen_page', **preserved)
        elif args.phase == 'inventory':
            capture = folder/'native-source'
            denominator = folder/'denominator'
            common = ['--page', number, '--source', args.source, '--database', args.database,
                      '--ownership', args.ownership]
            stage(folder, 'source-scan', cmd('generate_mep_source_denominator.py') + common +
                  ['--output-dir', capture], [capture/'manifest.json'])
            stage(folder, 'authored-paths', cmd('generate_mep_source_denominator_v2.py') + common +
                  ['--reuse-denominator-manifest', capture/'manifest.json', '--output-dir', denominator],
                  [denominator/'manifest.json'])
            manifest = read(denominator/'manifest.json')
            if manifest['acceptance_gate']['errors']:
                raise ValueError('native inventory failed')
            result.update(state='native_inventory_frozen', denominator=str((denominator/'manifest.json').resolve()),
                          native_segment_count=manifest['native_descriptor_pack']['record_count'])
        else:
            inventory = read(folder/'inventory.json')
            if inventory['state'] == 'failed':
                raise ValueError('source inventory failed; no fallback geometry allowed')
            denominator = Path(inventory['denominator'])
            run = folder/'callouts'
            stage(folder, 'callouts', cmd('generate_mep_page_callout_bindings.py') +
                  ['--page', number, '--source', args.source, '--registry', args.registry,
                   '--denominator', denominator, '--output', run],
                  [run/p for p in ('source.json','extended-M4.json.gz','local-M4.json.gz','summary.json')])
            result.update(state='native_callouts_replayed', denominator=str(denominator), run=str(run.resolve()))
            scale = scope['fields']['scale'].get('drawing_inches_per_paper_inch')
            if not scale:
                result.update(candidate_state='not_run_missing_view_scale', recovery=None)
            else:
                recovery = folder/'recovery'
                stage(folder, 'candidate-recovery', cmd('run_mep_diagnostic_package.py') +
                      ['--worker-recovery', '--source', args.source, '--database', args.database,
                       '--trace', args.trace, '--page', number, '--denominator', denominator,
                       '--output', recovery], [recovery/'recovery.json'])
                result.update(recovery=str((recovery/'recovery.json').resolve()),
                              candidate_state='diagnostic_boundary_review_pending')
                partitions = folder/'boundaries'
                stage(folder, 'candidate-boundaries', cmd('partition_mep_diagnostic_page.py') +
                      ['--page', number, '--source', args.source, '--recovery', recovery/'recovery.json',
                       '--denominator', denominator, '--output', partitions],
                      [partitions/'partitions.json.gz'])
                result['partitions'] = str((partitions/'partitions.json.gz').resolve())
    except Exception as exc:
        result.update(state='failed', error=str(exc))
    write(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--database', type=Path, default=ROOT/'data/projects/mep/project-v3.sqlite')
    p.add_argument('--ownership', type=Path, default=ROOT/'output/mep-sheet-region-ownership-2026-09-03/sheet-region-ownership.json')
    p.add_argument('--registry', type=Path, default=ROOT/'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json')
    p.add_argument('--trace', type=Path, default=ROOT/'output/mep-audit-trace-network-2026-09-03/trace-network.json')
    p.add_argument('--preserve-run', type=Path)
    p.add_argument('--preserve-recovery', type=Path)
    p.add_argument('--preserve-partitions', type=Path)
    p.add_argument('--preserve-bends', type=Path)
    p.add_argument('--preserve-pdf', type=Path)
    p.add_argument('--phase', choices=['inventory','interpret'], default='inventory')
    p.add_argument('--workers', type=int, choices=[1,2], default=1)
    p.add_argument('--worker-recovery', action='store_true')
    p.add_argument('--page', type=int)
    p.add_argument('--denominator', type=Path)
    args = p.parse_args()
    if args.worker_recovery:
        from tools.generate_mep_page5_page_wide_recovery import generate
        generate(source=args.source, database=args.database, denominator_path=args.denominator,
                 trace_path=args.trace, output_dir=args.output, page_number=args.page,
                 reuse_path_role_manifest=None)
        return
    registry = read(args.registry)
    source_hash = _file_sha256(args.source)
    if source_hash != registry['document']['source_pdf_sha256']:
        raise ValueError('source/registry snapshot mismatch')
    frozen = {}
    protected = {}
    if args.preserve_run:
        source = read(args.preserve_run/'source.json')
        if source['source_pdf_sha256'] != source_hash:
            raise ValueError('preserved run is from another source')
        frozen[str(source['page'])] = {'run': str(args.preserve_run.resolve()),
            'denominator': source['denominator_path'],
            **{name: str(getattr(args, 'preserve_'+name).resolve()) if getattr(args, 'preserve_'+name) else None
               for name in ('recovery','partitions','bends','pdf')}}
        files = [args.preserve_run/'extended-M4.json.gz', args.preserve_run/'local-M4.json.gz',
                 args.preserve_run/'source.json', Path(source['denominator_path'])]
        files += [getattr(args, 'preserve_'+k) for k in ('recovery','partitions','bends','pdf') if getattr(args, 'preserve_'+k)]
        protected = {str(f.resolve()): _file_sha256(f) for f in files}
    context = {'source_sha256': source_hash, 'registry_sha256': _file_sha256(args.registry),
               'ownership_sha256': _file_sha256(args.ownership), 'trace_sha256': _file_sha256(args.trace),
               'database_sha256': _file_sha256(args.database), 'preserved': frozen, 'protected': protected,
               'implementation_sha256': {str(f.relative_to(ROOT)): _file_sha256(f) for f in
                    sorted((ROOT/'src').glob('*.py'))},
               'complete_network_claimed': False, 'review_status': 'diagnostic_unreviewed'}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = args.output/'protocol.json'
    if protocol.exists() and read(protocol) != context:
        raise ValueError('batch input/code/protected snapshot changed; choose a new output')
    if not protocol.exists(): write(protocol, context)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda scope: process(args, scope, frozen), registry['pages']))
    if any(_file_sha256(Path(f)) != digest for f,digest in protected.items()):
        raise ValueError('protected page changed')
    write(args.output/(args.phase+'.json'), {'source_sha256': source_hash, 'pages': rows,
        'page_count': len(rows), 'failures': sum(r['state']=='failed' for r in rows),
        'all_pages_attempted': True, 'complete_pipe_coverage': False})
    print(json.dumps({'phase': args.phase, 'pages': len(rows), 'failures': sum(r['state']=='failed' for r in rows)}), flush=True)


if __name__ == '__main__': main()
