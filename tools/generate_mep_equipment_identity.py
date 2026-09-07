#!/usr/bin/env python3
"""Generate independent 2D equipment certificates from frozen native searches."""

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_equipment_identity import build_equipment_2d_identity, _check_frozen_terms
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


def native_typography(document, frozen):
    results = []
    for scope in frozen['pages']:
        page = document[scope['page_number']-1]
        lines = [line for block in page.get_text('dict')['blocks'] for line in block.get('lines', [])]
        for item in scope['items']:
            obs = item['inputs']['observation']
            matches = [line for line in lines if ''.join(s['text'] for s in line['spans']) == obs['text']
                       and list(line['bbox']) == obs['bbox_pdf']]
            if len(matches) != 1:
                continue
            line = matches[0]; spans = line['spans']; fonts = sorted({s['font'] for s in spans})
            result = {'record_type': 'mep_equipment_native_typography', 'page_ref': obs['page_ref'],
                'source_observation_ref': obs['id'], 'source_native_ref': obs['source_native_ref'],
                'source_pdf_sha256': frozen['source_pdf_sha256'], 'native_text': obs['text'],
                'bbox_display': obs['bbox_display'], 'font': fonts[0] if len(fonts) == 1 else None,
                'font_sizes': [s['size'] for s in spans], 'font_flags': [s['flags'] for s in spans],
                'bold': all(s['flags'] & 16 for s in spans), 'horizontal': list(line['dir']) == [1., 0.],
                'state': 'observed', 'quantity_eligible': False}
            result['id'] = _stable_id('mep_equipment_native_typography', result)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-replay', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--terminology', type=Path, required=True,
                        help='Exact M2 snapshot to bind; every consumed original tag/observation must agree')
    parser.add_argument('--calibration', type=Path, help='Frozen earlier identity payload; this input cannot update its convention')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(gzip.decompress(args.native_replay.read_bytes()))
    terminology = json.loads(args.terminology.read_text())
    _check_frozen_terms(frozen, terminology)
    if _file_sha256(args.source) != frozen['source_pdf_sha256']:
        raise ValueError('typography PDF differs from frozen native geometry source')
    with fitz.open(args.source) as document:
        typography = native_typography(document, frozen)
    earlier = json.loads(args.calibration.read_text()) if args.calibration else None
    result = build_equipment_2d_identity(native_inputs=frozen, typography=typography,
        calibration=earlier['drawing_convention'] if earlier else None)
    result['input_payload_sha256'] = {'terminology-proposals': _sha256(terminology)}
    result['native_replay_sha256'] = _file_sha256(args.native_replay)
    result['native_payload_sha256'] = _sha256(frozen)
    result['typography_payload_sha256'] = _sha256(typography)
    result['calibration_payload_sha256'] = _sha256(earlier) if earlier else None
    result['coverage'] = {'execution_page_numbers': [p['page_number'] for p in frozen['pages']],
        'calibration_page_refs': sorted({s['page_ref'] for g in result['drawing_convention']['groups'] for s in g['members']}),
        'calibration_consumes_current_scope': earlier is None,
        'whole_package_equipment_coverage_established': False, 'engineer_approved': False,
        'reviewed_selectors_consumed': False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'equipment-2d-identities.json').write_text(json.dumps(result, indent=2)+'\n')
    evidence = {'schema_version': '0.1.0', 'layer': 'mep_equipment_identity_replay_inputs',
        'native_replay': {'path': str(args.native_replay.resolve()), 'file_sha256': _file_sha256(args.native_replay)},
        'terminology': {'path': str(args.terminology.resolve()), 'payload_sha256': _sha256(terminology)},
        'typography': typography,
        'calibration_payload': {'path': str(args.calibration.resolve()), 'payload_sha256': _sha256(earlier)} if earlier else None,
        'identity_payload_sha256': _sha256(result), 'quantity_eligible': False}
    (args.output_dir/'equipment-2d-replay.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(result['summary']))
    for row in result['identities']:
        print(row['equipment_tag'],row['state'],row['reasons'],row['body_bbox_display'])


if __name__ == '__main__':
    main()
