#!/usr/bin/env python3
"""Replay automatic composite ports against original native multi-port bodies."""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = {'mep_native_boundary_connections.py': 'src/drawing_engine/disciplines/mep/mep_native_boundary_connections.py', 'mep_native_boundary_queries.py': 'src/drawing_engine/disciplines/mep/mep_native_boundary_queries.py', 'mep_native_branch_connections.py': 'src/drawing_engine/disciplines/mep/mep_native_branch_connections.py', 'mep_native_target_discovery.py': 'src/drawing_engine/disciplines/mep/mep_native_target_discovery.py', 'mep_outlined_route_composites.py': 'src/drawing_engine/disciplines/mep/mep_outlined_route_composites.py', 'mep_route_observations.py': 'src/drawing_engine/disciplines/mep/mep_route_observations.py', 'vector_topology.py': 'src/drawing_engine/core/vector_topology.py'}

sys.path.insert(0, str(ROOT))

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_branch_connections import discover_native_branches, branch_port_candidates, _ports


def read_payload(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def write_payload(path, metadata, rows):
    """Stream one record at a time; publish only the complete artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.partial')
    with ExitStack() as stack:
        raw = stack.enter_context(staging.open('wb'))
        stream = stack.enter_context(gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=1)) if path.suffix == '.gz' else raw
        stream.write(json.dumps(metadata, separators=(',', ':'), sort_keys=True).encode()[:-1])
        stream.write(b',"connections":[')
        for ordinal, row in enumerate(rows):
            if ordinal:
                stream.write(b',')
            stream.write(json.dumps(row, separators=(',', ':'), sort_keys=True).encode())
        stream.write(b']}\n')
    staging.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Required for live extraction; unused by source-free replay')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay-from', type=Path, help='A prior page-checkpoint directory; no source extraction')
    args = parser.parse_args()
    graph = json.loads((args.run / 'route-observations.json').read_text())
    composites = json.loads((args.run / 'outlined-route-composites.json').read_text())
    if not args.replay_from and args.source is None:
        parser.error('--source is required without --replay-from')
    if not args.replay_from and _file_sha256(args.source) != graph['document']['source_pdf_sha256']:
        raise ValueError('source PDF differs from frozen route graph')
    implementation = {name: _file_sha256(ROOT / IMPLEMENTATION_PATHS[name]) for name in (
        'mep_native_branch_connections.py', 'mep_native_boundary_connections.py',
        'mep_native_boundary_queries.py', 'mep_native_target_discovery.py',
        'mep_outlined_route_composites.py', 'mep_route_observations.py', 'vector_topology.py')}
    implementation['tools/discover_mep_native_branches.py'] = _file_sha256(Path(__file__))
    metadata = {'document': graph['document'], 'm3_payload_sha256': _sha256(graph),
        'm35_payload_sha256': _sha256(composites), 'implementation_sha256': implementation,
        'candidate_scope': 'all accepted composite ports on processed pages; bounded orthogonal T groups',
        'quantity_eligible': False, 'engineer_approved': False}
    context_hash = _sha256(metadata)
    checkpoints = args.out.parent / 'native-branch-pages' / context_hash[:16]
    checkpoint_paths = []
    coverage = []
    reason_counts = Counter()
    fragments = {r['id']: r for page in graph['pages'] for r in page['fragments']}
    with ExitStack() as stack:
        pdf = None if args.replay_from else stack.enter_context(fitz.open(args.source))
        for page in graph['pages']:
            checkpoint = checkpoints / f"page-{page['page_number']:03d}.json.gz"
            if checkpoint.exists():
                cached = read_payload(checkpoint)
                if cached['context_sha256'] != context_hash:
                    raise ValueError('branch checkpoint context changed')
                coverage.append(cached['page_coverage'])
                reason_counts.update(reason for row in cached['connections'] for reason in row['reasons'])
                print(f"page {page['page_number']}: verified checkpoint ({len(cached['connections'])} queries)", flush=True)
                checkpoint_paths.append(checkpoint)
                del cached
                continue
            local = [r for r in composites['accepted_composites'] if r['page_ref'] == page['page_ref']]
            candidates = branch_port_candidates(list(_ports(local, fragments)))
            coverage.append({'page_ref': page['page_ref'], 'page_number': page['page_number'],
                'accepted_composite_count': len(local), 'candidate_count': len(candidates),
                'native_query_count': 0, 'accepted_branch_count': 0,
                'scope_complete': False, 'status': 'no_qualifying_port_group' if local else 'no_accepted_composites'})
            if not candidates:
                print(f"page {page['page_number']}: no qualifying port group", flush=True)
                write_payload(checkpoint, {'context_sha256': context_hash, 'page_coverage': coverage[-1]}, [])
                checkpoint_paths.append(checkpoint)
                continue
            if args.replay_from:
                prior = read_payload(args.replay_from / checkpoint.name)
                if (prior.get('m3_payload_sha256') != metadata['m3_payload_sha256']
                        or prior.get('m35_payload_sha256') != metadata['m35_payload_sha256']
                        or prior['page_coverage']['page_ref'] != page['page_ref']):
                    raise ValueError('frozen queries do not belong to these M3/M3.5 inputs and page')
                by_box = {tuple(row['search']['bbox_display']): row for row in prior['connections']}

                class FrozenQuery:
                    def query(self, box):
                        if tuple(box) not in by_box:
                            raise ValueError('replay needs a new live native query')
                        row = by_box[tuple(box)]
                        if _sha256(sorted(r['source_primitive_ref'] for r in row['source_rows'])) != row['search']['all_source_refs_sha256']:
                            raise ValueError('incomplete frozen native source query')
                        return row['source_rows'], row['search']['complete'], row['search']['region_refs']

                index = FrozenQuery()
            else:
                index = NativeBoundaryQueries(pdf[page['page_number'] - 1], page['page_ref'])
            found = discover_native_branches(index=index, composites=composites, graph=graph, page_ref=page['page_ref'])
            del index
            coverage[-1].update(native_query_count=len(found), accepted_branch_count=sum(r['state'] == 'accepted' for r in found),
                status='bounded_candidate_queries_complete')
            print(f"page {page['page_number']}: {dict(Counter(r['state'] for r in found))}", flush=True)
            reason_counts.update(reason for row in found for reason in row['reasons'])
            write_payload(checkpoint, {'context_sha256': context_hash, 'page_coverage': coverage[-1],
                'm3_payload_sha256': metadata['m3_payload_sha256'], 'm35_payload_sha256': metadata['m35_payload_sha256']}, found)
            checkpoint_paths.append(checkpoint)
            del found
            if args.replay_from:
                del prior, by_box

    def frozen_rows():
        for path in checkpoint_paths:
            frozen = read_payload(path)
            yield from frozen['connections']
            del frozen

    metadata.update(page_coverage=coverage, context_sha256=context_hash,
        checkpoint_directory=str(checkpoints), execution_mode='frozen_query_replay' if args.replay_from else 'live_native_queries_or_exact_checkpoints')
    write_payload(args.out, metadata, frozen_rows())
    print(dict(reason_counts), flush=True)
    pilot = [row for row in coverage if row['page_number'] in (2, 3)]
    report = ['# Native branch discovery boundary', '',
        'This extension uses the existing registered 18-page baseline. It is separate from the pages 2–3 annotation pilot.', '',
        f"Complete source query records: {sum(row['native_query_count'] for row in coverage)}; accepted projected branches: {sum(row['accepted_branch_count'] for row in coverage)}.",
        f"The pages 2–3 subset contains {sum(row['native_query_count'] for row in pilot)} queries. All other pages remain a separate discovery scope.",
        'No physical continuation, quantity, fitting count, or engineer approval is established. A page with no qualifying group is not a claim of no branches.', '',
        '| Page | Accepted composite inputs | Candidate groups | Source query records | Accepted branches |',
        '| --- | ---: | ---: | ---: | ---: |']
    report.extend(f"| {r['page_number']} | {r['accepted_composite_count']} | {r['candidate_count']} | {r['native_query_count']} | {r['accepted_branch_count']} |" for r in coverage)
    report.extend(['', 'Abstention reason counts (one candidate can have several reasons):', ''])
    report.extend(f'- `{reason}`: {count}' for reason, count in sorted(reason_counts.items()))
    report.extend(['', f"Execution: `{metadata['execution_mode']}`.", f"Context SHA-256: `{context_hash}`.",
        f"M3 SHA-256: `{metadata['m3_payload_sha256']}`.", f"M3.5 SHA-256: `{metadata['m35_payload_sha256']}`.",
        f"Artifact SHA-256: `{_file_sha256(args.out)}`.", '',
        'Each source query retains every native row, its original primitive IDs, bounds, query hash, port diagnostics and competing destinations.',
        'Checkpoints are isolated by the input and startup implementation hashes; final artifacts stream one record at a time and publish atomically.', ''])
    args.out.with_name(args.out.name.replace('.json.gz', '').replace('.json', '') + '.REPORT.md').write_text('\n'.join(report))


if __name__ == '__main__':
    main()
