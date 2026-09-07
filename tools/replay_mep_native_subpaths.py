#!/usr/bin/env python3
"""Replay frozen candidates into a separate path-aware experiment, never a PDF/M4."""
import argparse
from collections import Counter
from array import array
from bisect import bisect_left
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256,_file_sha256
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style


def difference(intervals, previous):
    result=[]
    for a,b in intervals:
        pieces=[(a,b)]
        for c,d in previous:
            pieces=[p for x,y in pieces for p in ([(x,min(y,c))] if x<c else [])+([(max(x,d),y)] if y>d else []) if p[1]>p[0]+1e-9]
        result.extend(pieces)
    return result


def replay(args):
    if args.output.exists():raise ValueError('choose a new output directory')
    baseline=read(args.baseline/'partitions.json');selection=read(args.baseline/'selection.json')
    if _file_sha256(args.baseline/'selection.json')!=baseline['inputs']['selection_sha256']:raise ValueError('selection changed')
    protocol=read(args.shadow/'protocol.json')
    for key in ('M4','audit_pdf'):
        if _file_sha256(Path(protocol[key+'_path']))!=protocol[key+'_sha256']:raise ValueError('protected baseline changed')
    old=baseline['partitions']
    for r in old:
        path=Path(r['query_ref']['path'])
        if _file_sha256(path)!=r['query_ref']['sha256']:raise ValueError('native query changed')
    info=read(args.denominator)['native_authored_path_pack'];pack=NativePathPack(info['path'],info)
    if not pack.verify_hash():raise ValueError('authored path pack changed')
    ordinals=array('I');segment_counts=array('I')
    for r in pack.records():
        ordinals.append(r['drawing_ordinal']);segment_counts.append(r['source_segment_count'])
    args.output.mkdir(parents=True)
    write(args.output/'authored-context.json',{'denominator_sha256':_file_sha256(args.denominator),
          'native_path_pack_sha256':info['sha256'],'native_path_pack_path':info['path'],
          'lookup_method':'binary search over compact drawing ordinal/count arrays; per-query evidence kept in source pack'})
    outputs=[];counts=Counter()
    for i,r in enumerate(old):
        q=read(Path(r['query_ref']['path']));members=set(r['native_member_source_refs'])
        if r['original_candidate'].get('corridor_width_display_points'):
            native=next(row for row in q['source_rows'] if row['source_primitive_ref'] in members)
            local_context={}
            for row in q['source_rows']:
                ref=row['source_native_segment']['drawing_ref'];ordinal=int(ref[8:-1]);index=bisect_left(ordinals,ordinal)
                if index<len(ordinals) and ordinals[index]==ordinal:
                    local_context[ref]={'drawing_ordinal':ordinal,'source_segment_count':segment_counts[index]}
            new=partition_corridor(r['original_candidate'],q,members,_normalise_style(native['source_native_segment']['style']),authored_paths=local_context)
        else:new={**r,'path_aware_replay_state':'not_applicable_without_straight_corridor'}
        path=args.output/'partitions'/f'{i:03d}.json.gz';write(path,new)
        before=[a['parameter_interval'] for a in r['retained_intervals']]
        after=[a['parameter_interval'] for a in new['retained_intervals']]
        gained=difference(after,before);lost=difference(before,after)
        counts['replayed_candidates']+=1;counts['changed_candidates']+=bool(gained or lost)
        counts['new_exclusions_candidates']+=bool(lost)
        counts['recovered_display_points']+=sum(b-a for a,b in gained)
        counts['newly_excluded_display_points']+=sum(b-a for a,b in lost)
        certs=new.get('crossing_path_certificates',[])
        counts.update('certificate:'+c['method'] for c in certs)
        record={'candidate_ref':r['candidate_ref'],'partition_path':str(path.resolve()),'partition_sha256':_file_sha256(path),
            'query_ref':r['query_ref'],'state':new['state'],'gained_parameter_intervals':gained,
            'lost_parameter_intervals':lost,'certificate_methods':sorted({c['method'] for c in certs}),
            'source_primitive_count':len(q['source_rows']),
            'accounted_source_primitive_count':new.get('native_subpath_inventory',{}).get('accounted_source_primitive_count'),
            'retained_interval_count':len(new['retained_intervals']),'unknown_interval_count':len(new['body_intervals'])}
        if new.get('native_subpath_inventory') and record['source_primitive_count']!=record['accounted_source_primitive_count']:raise ValueError('lost source primitives')
        outputs.append(record)
        if i%10==0:print(i+1,dict(counts),flush=True)
    excluded={r['candidate_ref'] for r in read(args.previous_review)['decisions']}
    excluded.update(r['candidate_ref'] for r in read(args.baseline/'generalization.json')['sample'])
    for case in protocol['cases']:
        p=read(args.shadow/case['case_id']/'baseline.json').get('partition')
        if p:excluded.add(p['candidate_ref'])
    # One fresh parent per stratum; min/median/max changed extent is explicit.
    sample=[]
    for kind in ('exact_native_subpath_transverse_v2',):
        pool=[r for r in outputs if r['gained_parameter_intervals'] and kind in r['certificate_methods'] and r['candidate_ref'] not in excluded]
        pool.sort(key=lambda r:(sum(b-a for a,b in r['gained_parameter_intervals']),r['candidate_ref']))
        for index in sorted({0,len(pool)//2,len(pool)-1}) if pool else []:
            r=pool[index];excluded.add(r['candidate_ref'])
            a,b=max(r['gained_parameter_intervals'],key=lambda p:p[1]-p[0])
            parent=next(p for p in old if p['candidate_ref']==r['candidate_ref']);start,end=parent['original_candidate']['polyline_display']
            import math
            length=math.dist(start,end);points=[[start[k]+t*(end[k]-start[k])/length for k in (0,1)] for t in (a,b)]
            sample.append({'candidate_ref':r['candidate_ref'],'parameter_interval':[a,b],'polyline_display':points,
                'bbox_display':[min(p[k] for p in points)-18 for k in (0,1)]+[max(p[k] for p in points)+18 for k in (0,1)],
                'partition_path':r['partition_path'],'query_ref':r['query_ref'],'review_state':'pending'})
    if len(outputs)!=len(selection['candidates']) or {r['candidate_ref'] for r in outputs}!={r['id'] for r in selection['candidates']}:raise ValueError('candidate coverage changed')
    write(args.output/'replay.json',{'baseline_sha256':_file_sha256(args.baseline/'partitions.json'),
        'selection_sha256':_file_sha256(args.baseline/'selection.json'),'denominator_sha256':_file_sha256(args.denominator),
        'source_pdf_sha256':selection['source_pdf_sha256'],'protected_protocol_sha256':_file_sha256(args.shadow/'protocol.json'),
        'code_hashes':{p:_file_sha256(ROOT/p) for p in ('src/drawing_engine/disciplines/mep/mep_native_subpaths.py','src/drawing_engine/disciplines/mep/mep_route_body_partition.py',
            'src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py','tools/replay_mep_native_subpaths.py')},
        'counts':dict(counts),'records':outputs,'fresh_sample':sample,'review_status':'pending',
        'boundary_changes_published':False,'new_identity_accepts':0,'complete_sheet_coverage':False,
        'installed_length':None,'purchase_length':None})
    print(dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','denominator','shadow','previous-review','output'):p.add_argument('--'+name,type=Path,required=True)
    replay(p.parse_args())
