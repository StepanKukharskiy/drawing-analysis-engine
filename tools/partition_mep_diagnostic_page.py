#!/usr/bin/env python3
"""Apply the current native-subpath boundary mechanism to every route candidate.

This is unreviewed diagnostic output, not permission to publish new identities.
All original candidates and native query rows remain in the evidence sidecars.
"""
import argparse
from array import array
from bisect import bisect_left
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from tools.refine_mep_route_boundaries import capture
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor,nonlinear_chord_defect
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style


def run(args):
    recovery=read(args.recovery)
    selected=[r for k in ('outlined_corridor_components','single_centreline_components')
              for r in recovery[k] if r['state'] in
              {'identified_mep_route','supported_unidentified_mep_candidate'}]
    args.output.mkdir(parents=True,exist_ok=True)
    selection={'recovery_sha256':_file_sha256(args.recovery),
               'source_pdf_sha256':_file_sha256(args.source),'candidates':selected}
    args.candidates=args.output/'selection.json';args.review=None
    if args.candidates.exists() and read(args.candidates)!=selection:raise ValueError('selection changed')
    write(args.candidates,selection)
    capture(args)
    index=read(args.output/'capture-index.json')
    info=read(args.denominator)['native_authored_path_pack'];pack=NativePathPack(info['path'],info)
    if not pack.verify_hash():raise ValueError('path pack changed')
    wanted={i for r in selected for a,b in r.get('source_path_ordinal_intervals',r.get('path_ordinal_intervals',[])) for i in range(a,b+1)}
    drawings=array('I');counts=array('I');paths={}
    for p in pack.records():
        drawings.append(p['drawing_ordinal']);counts.append(p['source_segment_count'])
        if p['path_ordinal'] in wanted:paths[p['path_ordinal']]=p['drawing_ordinal']
    if set(paths)!=wanted:raise ValueError('candidate path not in complete inventory')
    members={r['id']:{paths[i] for a,b in r.get('source_path_ordinal_intervals',r.get('path_ordinal_intervals',[]))
                      for i in range(a,b+1)} for r in selected}
    records=[]
    for n,qref in enumerate(index['queries']):
        path=Path(qref['path'])
        if _file_sha256(path)!=qref['sha256']:raise ValueError('native query changed')
        q=read(path);r=q['candidate'];native=[s for s in q['source_rows']
            if int(s['source_native_segment']['drawing_ref'][8:-1]) in members[r['id']]]
        refs={s['source_primitive_ref'] for s in native}
        if not native:raise ValueError('candidate has no native members')
        if r.get('corridor_width_display_points'):
            context={}
            for s in q['source_rows']:
                ref=s['source_native_segment']['drawing_ref'];ordinal=int(ref[8:-1]);i=bisect_left(drawings,ordinal)
                if i>=len(drawings) or drawings[i]!=ordinal:raise ValueError('query path absent from inventory')
                context[ref]={'drawing_ordinal':ordinal,'source_segment_count':counts[i]}
            row=partition_corridor(r,q,refs,_normalise_style(native[0]['source_native_segment']['style']),authored_paths=context)
        else:
            defect=nonlinear_chord_defect(r,native)
            parents=[p['id'] for p in selected if p.get('corridor_width_display_points')
                     and members[r['id']]<=members[p['id']]]
            row={'candidate_ref':r['id'],'state':'representation_withdrawn' if defect or len(parents)==1 else 'abstained',
                 'representation_defect':defect,'duplicate_outline_parent_refs':parents,
                 'retained_intervals':[],'body_intervals':[],
                 'reasons':['nonlinear_chord'] if defect else ['duplicate_sidewall'] if len(parents)==1 else ['boundary_certificate_not_applicable']}
        row.update(original_candidate=r,native_member_source_refs=sorted(refs),query_ref=qref)
        record_path=args.output/'partitions'/f'{n:04d}.json.gz';write(record_path,row)
        # The full native-subpath inventory stays compressed in its own record.
        records.append({k:v for k,v in row.items() if k not in
            {'native_subpath_inventory','crossing_path_certificates'}} | {
                'partition_path':str(record_path.resolve()),'partition_sha256':_file_sha256(record_path)})
        if n%25==0:print(n+1,len(selected),row['state'],flush=True)
    write(args.output/'partitions.json.gz',{'inputs':index,'partitions':records,
          'source_pdf_sha256':selection['source_pdf_sha256'],'all_source_strokes_preserved':True,
          'independent_review':'pending','new_system_bindings':0,'installed_length':None,'purchase_length':None})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('recovery','denominator','output','source'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--page',type=int,required=True)
    run(p.parse_args())
