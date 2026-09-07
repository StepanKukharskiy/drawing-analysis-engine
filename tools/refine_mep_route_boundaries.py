#!/usr/bin/env python3
"""Investigate reviewed route/body contamination against complete native queries.

Reviewed IDs nominate searches, never certify body geometry or pipe identity.
All old candidates and raw source strokes remain immutable in the input pack.
"""
import argparse
from pathlib import Path
import sys
import math
from collections import defaultdict

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256,_file_sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor,nonlinear_chord_defect


def refine(args):
    recovery=read(args.recovery);manifest=read(args.output/'capture-index.json')
    if manifest['recovery_sha256']!=_file_sha256(args.recovery):raise ValueError('recovery changed')
    def queries():
        # Large page-wide windows must not all be expanded in RAM together.
        for p in manifest['queries']:
            if _file_sha256(Path(p['path']))!=p['sha256']:raise ValueError('query changed')
            yield read(Path(p['path']))
    requested=set()
    candidates={}
    if getattr(args,'candidates',None):
        if _file_sha256(args.candidates)!=manifest['selection_sha256']:raise ValueError('selection changed')
        selected=read(args.candidates)['candidates']
    else:selected=(q['candidate'] for q in queries())
    for r in selected:
        candidates[r['id']]=r
        requested.update(i for a,b in r.get('source_path_ordinal_intervals',r.get('path_ordinal_intervals',[])) for i in range(a,b+1))
    denominator=read(args.denominator);info=denominator['native_authored_path_pack'];pack=NativePathPack(info['path'],info)
    if not pack.verify_hash():raise ValueError('native paths changed')
    paths={r['path_ordinal']:r for r in pack.records() if r['path_ordinal'] in requested}
    if set(paths)!=requested:raise ValueError('missing native member path')
    members={}
    for q in queries():
        r=q['candidate'];ords={paths[i]['drawing_ordinal'] for a,b in r.get('source_path_ordinal_intervals',r.get('path_ordinal_intervals',[])) for i in range(a,b+1)}
        if r!=candidates[r['id']]:raise ValueError('captured candidate differs from selection')
        members[r['id']]=[a for a in q['source_rows'] if int(a['source_native_segment']['drawing_ref'][8:-1]) in ords]
    outputs=[]
    for q in queries():
        r=q['candidate'];native=members[r['id']];refs={a['source_primitive_ref'] for a in native}
        if not native:raise ValueError('candidate source membership unavailable')
        if not q['search']['complete'] or q['search']['source_rows_sha256']!=_sha256(q['source_rows']):raise ValueError('incomplete native query')
        if r.get('corridor_width_display_points'):
            row=partition_corridor(r,q,refs,_normalise_style(native[0]['source_native_segment']['style']))
        else:
            defect=nonlinear_chord_defect(r,native)
            parents=[a['id'] for a in candidates.values() if a.get('corridor_width_display_points')
                and refs<={s['source_primitive_ref'] for s in members[a['id']]}]
            row={'candidate_ref':r['id'],'state':'representation_withdrawn' if defect or len(parents)==1 else 'boundary_rule_not_applicable',
                'representation_defect':defect,'duplicate_outline_parent_refs':parents,
                'retained_intervals':[],'body_intervals':[],
                'reasons':['nonlinear_chord'] if defect else ['duplicate_sidewall'] if len(parents)==1 else ['no_straight_corridor_boundary_certificate']}
        row['original_candidate']=r;row['native_member_source_refs']=sorted(refs)
        row['query_ref']=next(p for p in manifest['queries'] if p['candidate_ref']==r['id'])
        outputs.append(row)
        print(len(outputs),row['candidate_ref'],row['state'],len(row['retained_intervals']),len(row['body_intervals']),flush=True)
    write(args.output/'partitions.json',{'inputs':manifest,'denominator_sha256':_file_sha256(args.denominator),
        'partitions':outputs,'all_source_strokes_preserved':True,'new_system_bindings':0,
        'body_type_identity_established':False,'installed_length':None,'purchase_length':None})


def capture(args):
    recovery=read(args.recovery)
    if getattr(args,'candidates',None):
        selection=read(args.candidates)
        review={'baseline':{'recovery_sha256':selection['recovery_sha256']},
                'source_pdf_sha256':selection['source_pdf_sha256']}
        selected=selection['candidates']
    else:
        if not args.review:raise ValueError('capture requires a review or explicit frozen candidate selection')
        review=read(args.review)
        rows={r['id']:r for name in ('outlined_corridor_components','single_centreline_components') for r in recovery[name]}
        selected=[rows[r['candidate_ref']] for r in review['decisions'] if r['review_class']=='discrete_body_or_interface_contamination']
    if _file_sha256(args.recovery)!=review['baseline']['recovery_sha256']:
        raise ValueError('review recovery snapshot changed')
    if _file_sha256(args.source)!=review['source_pdf_sha256']:
        raise ValueError('source PDF changed')
    captures=[]
    with fitz.open(args.source) as pdf:
        page=pdf[args.page-1];index=None
        for number,row in enumerate(selected):
            path=args.output/'queries'/f'{number:03d}.json.gz'
            if path.exists():
                query=read(path)
                if query['candidate']!=row or query['source_pdf_sha256']!=review['source_pdf_sha256']:
                    raise ValueError('cached source capture changed')
            else:
                if index is None:index=NativeBoundaryQueries(page,recovery['page_ref'])
                points=row['polyline_display'];pad=max(12,min(40,row['projected_path_display_points']))
                box=[min(p[i] for p in points)-pad for i in (0,1)]+[max(p[i] for p in points)+pad for i in (0,1)]
                sources,complete,refs=index.query(box)
                query={'candidate':row,'page_ref':recovery['page_ref'],'source_rows':sources,
                    'source_pdf_sha256':review['source_pdf_sha256'],
                    'search':{'bbox_display':box,'complete':complete,'query_refs':refs,
                        'source_rows_sha256':_sha256(sources),
                        'all_source_refs_sha256':_sha256(sorted(r['source_primitive_ref'] for r in sources))}}
                write(path,query)
            captures.append({'path':str(path.resolve()),'sha256':_file_sha256(path),'candidate_ref':row['id']})
            print(number,row['id'],len(query['source_rows']),flush=True)
    write(args.output/'capture-index.json',{'recovery_sha256':_file_sha256(args.recovery),
        'review_sha256':_file_sha256(args.review) if args.review else None,
        'selection_sha256':_file_sha256(args.candidates) if getattr(args,'candidates',None) else None,'queries':captures})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery',type=Path,required=True);p.add_argument('--review',type=Path)
    p.add_argument('--candidates',type=Path)
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--page',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--denominator',type=Path);p.add_argument('--refine',action='store_true')
    args=p.parse_args()
    refine(args) if args.refine else capture(args)
