#!/usr/bin/env python3
"""Freeze source-first endpoint and full-delta review; never change M4 or PDF."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256,_sha256


def checked(path,digest):
    if _file_sha256(path)!=digest:raise ValueError(f'changed frozen input: {path}')
    return read(path)


def prepare(args):
    from tools.review_mep_interpretation_frontier import crop_cards
    import fitz
    if args.output.exists():raise ValueError('use a new review directory')
    replay=read(args.replay/'replay.json')
    checked(args.baseline/'partitions.json',replay['baseline_sha256'])
    baseline=read(args.baseline/'partitions.json')['partitions']
    originals={r['candidate_ref']:r for r in baseline}
    for name,digest in replay['code_hashes'].items():
        if _file_sha256(ROOT/name)!=digest:raise ValueError('classifier changed before review')
    if _file_sha256(args.source)!=replay['source_pdf_sha256']:raise ValueError('source changed')
    protected=read(args.shadow/'protocol.json')
    for key in ('M4','audit_pdf'):
        if _file_sha256(Path(protected[key+'_path']))!=protected[key+'_sha256']:raise ValueError('protected artifact changed')
    cards=[];changed=set()
    def card(group,ref,points,**extra):
        record=originals[ref]
        cards.append({'case_id':f'{group}-{1+sum(c["group"]==group for c in cards):02d}',
            'group':group,'candidate_ref':ref,'polyline_display':points,
            'bbox_display':[min(p[i] for p in points)-30 for i in (0,1)]+[max(p[i] for p in points)+30 for i in (0,1)],
            'query_ref':record['query_ref'],'review_state':'pending',**extra})
    for row in replay['records']:
        if not row['gained_parameter_intervals']:continue
        changed.add(row['candidate_ref']);p=originals[row['candidate_ref']]
        a,b=p['original_candidate']['polyline_display'];length=math.dist(a,b)
        for interval in row['gained_parameter_intervals']:
            points=[[a[k]+t*(b[k]-a[k])/length for k in (0,1)] for t in interval]
            card('delta',row['candidate_ref'],points,parameter_interval=interval,
                 partition_path=row['partition_path'],partition_sha256=row['partition_sha256'])
        for end,point in enumerate((a,b)):
            card('parent-end',row['candidate_ref'],[point],endpoint_index=end,
                 partition_path=row['partition_path'],partition_sha256=row['partition_sha256'])
    excluded=set(changed)
    for path in (args.prior_review,args.baseline/'generalization.json'):
        r=read(path);excluded.update(c['candidate_ref'] for c in r.get('decisions',r.get('sample',[])))
    for c in protected['cases']:
        p=read(args.shadow/c['case_id']/'baseline.json').get('partition')
        if p:excluded.add(p['candidate_ref'])
    # Endpoint nominations, not genuine-endpoint gold. Freeze before source review.
    pool=sorted((p for p in baseline if p['candidate_ref'] not in excluded and p['state']=='partitioned'),
                key=lambda p:_sha256(['endpoint-source-review-v1',p['candidate_ref']]))
    for p in pool[:12]:
        end=int(_sha256(p['candidate_ref'])[-1],16)%2
        card('endpoint-pool',p['candidate_ref'],[p['original_candidate']['polyline_display'][end]],endpoint_index=end)
    native_search=[]
    import re
    with fitz.open(args.source) as pdf:
        for n,page in enumerate(pdf):
            text=page.get_text('text')
            matches=[m.group(0) for m in re.finditer(r'\b(?:cap(?:ped)?|plug(?:ged)?|terminat\w*|end of)\b',text,re.I)]
            native_search.append({'page':n+1,'native_text_sha256':_sha256(text),'matches':matches})
    args.output.mkdir(parents=True)
    protocol={'replay_path':str((args.replay/'replay.json').resolve()),'replay_sha256':_file_sha256(args.replay/'replay.json'),
        'source_pdf_sha256':replay['source_pdf_sha256'],'source_page':args.page,'cards':cards,
        'counts':dict(Counter(c['group'] for c in cards)),
        'changed_candidate_count':len(changed),'originals_path':str((args.baseline/'partitions.json').resolve()),
        'originals_sha256':replay['baseline_sha256'],'protected_artifacts':{k:protected[k] for k in ('M4_path','M4_sha256','audit_pdf_path','audit_pdf_sha256')},
        'native_terminal_text_search':native_search,'native_text_no_match_proves_no_graphical_caps':False,
        'evaluation_status':'new source-first nominations; previous held-out cases are exposed diagnostic regressions, never tuning truth',
        'publication_gate_passed':False,'new_identity_accepts':0,'installed_length':None,'purchase_length':None}
    write(args.output/'protocol.json',protocol)
    for group in ('delta','parent-end','endpoint-pool'):
        crop_cards(args.source,args.page,[c for c in cards if c['group']==group],ROOT/'tmp/pdfs'/args.output.name,group,mark_endpoints=True)
    print(protocol['counts'])


def review_gate(protocol,review):
    cards={c['case_id']:c for c in protocol['cards']};rows=review['decisions']
    if len(rows)!=len(cards) or {r['case_id'] for r in rows}!=set(cards):raise ValueError('review must cover each frozen case once')
    roles={'pipe_continues','genuine_termination','physical_cap','equipment_contact','crossing_near_endpoint','symbol_or_support','non_route_drawing_content','unresolved'}
    for r in rows:
        if r['source_role'] not in roles:raise ValueError('invalid source review role')
        if r['boundary_verdict'] not in {'preserved','unsafe','unresolved'}:raise ValueError('invalid boundary verdict')
        if not r.get('source_refs') or not r.get('note'):raise ValueError('review requires native references and explanation')
        if cards[r['case_id']]['group']=='delta':
            ends=r.get('end_reviews',[])
            if len(ends)!=2 or {e.get('endpoint_index') for e in ends}!={0,1}:raise ValueError('both recovered interval ends must be reviewed')
            for e in ends:
                if e.get('verdict') not in {'visually_continuous','unresolved','unsafe'} or not e.get('note'):
                    raise ValueError('each boundary requires a verdict and observation')
    counts=Counter(r['source_role'] for r in rows if cards[r['case_id']]['group']!='delta')
    missing=sorted({'genuine_termination','physical_cap','equipment_contact','crossing_near_endpoint'}-set(counts))
    unresolved=[r['case_id'] for r in rows if r['boundary_verdict']=='unresolved' or any(e['verdict']=='unresolved' for e in r.get('end_reviews',[]))]
    unsafe=[r['case_id'] for r in rows if r['boundary_verdict']=='unsafe' or any(e['verdict']=='unsafe' for e in r.get('end_reviews',[]))]
    native=review.get('native_endpoint_gate_passed') is True
    return {'source_role_counts':dict(counts),'missing_endpoint_truth_classes':missing,
        'unresolved_cases':unresolved,'unsafe_cases':unsafe,
        'native_endpoint_gate_passed':native,
        'genuine_endpoint_recall':None if missing else review.get('genuine_endpoint_recall'),
        'publication_gate_passed':native and not(missing or unresolved or unsafe),
        'boundary_changes_published':False,'new_identity_accepts':0,
        'installed_length':None,'purchase_length':None}


def contains(outer,inner):
    return all(outer[k]<=inner[k] and inner[k+2]<=outer[k+2] for k in (0,1))


def intersects(a,b):
    return all(a[k]<=b[k+2] and b[k]<=a[k+2] for k in (0,1))


def native_measurements(protocol):
    """Spatial evidence references are context, NOT machine-assigned semantic gold."""
    from collections import defaultdict
    groups=defaultdict(list)
    for c in protocol['cards']:groups[c['query_ref']['path']].append(c)
    cases=[];joins=[]
    for path,cards in groups.items():
        q=checked(Path(path),cards[0]['query_ref']['sha256'])
        if q['source_pdf_sha256']!=protocol['source_pdf_sha256']:raise ValueError('query source changed')
        rows=q['source_rows'];by_ref={r['source_primitive_ref']:r for r in rows}
        for c in cards:
            crop=c['bbox_display']
            complete=q['search']['complete'] and contains(q['search']['bbox_display'],crop)
            # Include all ink in the reviewed source crop, never only route members.
            refs=sorted(r['source_primitive_ref'] for r in rows if intersects(r['bbox_display'],crop))
            cases.append({'case_id':c['case_id'],'query_sha256':c['query_ref']['sha256'],
                'complete_review_crop_search':complete,'source_refs':refs,
                'reference_meaning':'all intersecting crop context; role ownership independently reviewed',
                'focus_points':c['polyline_display'],'physical_terminal_established':False})
        partition_card=next((c for c in cards if 'partition_path' in c),None)
        if partition_card:
            part=checked(Path(partition_card['partition_path']),partition_card['partition_sha256'])
            for path_record in part.get('native_subpath_inventory',{}).get('subpaths',[]):
                for join in path_record['joins']:
                    left,right=(by_ref[r] for r in join['source_refs_in_order'])
                    same=left['source_native_segment']['style']==right['source_native_segment']['style']
                    residual=math.dist(left['points_display'][-1],right['points_display'][0])
                    joins.append({'candidate_ref':part['candidate_ref'],'join_ref':join['id'],
                        'source_refs':join['source_refs_in_order'],'composition':path_record['composition'],
                        'measured_seam_residual':residual,'matching_native_style':same,
                        'physical_connection_established':join['physical_connection_established']})
        print('checked native endpoint context',len(cases),'/',len(protocol['cards']),flush=True)
    return {'cases':cases,'joins':joins,'scope':'all stored regrouped subpath joins in the 16 changed-parent queries; repeated occurrences not physical counts',
        'nonzero_seams':sum(j['measured_seam_residual']!=0 for j in joins),
        'style_mismatches':sum(not j['matching_native_style'] for j in joins),
        'physical_connection_claims':sum(j['physical_connection_established'] for j in joins),
        'genuine_endpoint_gate_passed':False,'coordinate_parity_proves_terminal_preservation':False}


def expand_review(review,measurements):
    by_case={r['case_id']:r for r in measurements['cases']};decisions=[]
    for case,role,note,*ends in review['delta_reviews']+review['endpoint_reviews']:
        source_role=review['role_key'][role]
        decisions.append({'case_id':case,'source_role':source_role,'note':note,
            'source_refs':by_case[case]['source_refs'],
            'source_refs_meaning':by_case[case]['reference_meaning'],
            'boundary_verdict':'unresolved' if role in {'U','E','N'} else 'preserved',
            'boundary_verdict_scope':'visual local extent only; not physical terminal certification',
            'end_reviews':[{'endpoint_index':i,'verdict':ends[2*i],'note':ends[2*i+1]} for i in range(len(ends)//2)]})
    return {**review,'decisions':decisions,'native_endpoint_gate_passed':False}


def diagnose_misses(protocol,shadow):
    """Inspect frozen failures; no thresholds or classifier decisions are changed."""
    replay=checked(Path(protocol['replay_path']),protocol['replay_sha256'])
    records={r['candidate_ref']:r for r in replay['records']}
    originals=checked(Path(protocol['originals_path']),protocol['originals_sha256'])
    originals={r['candidate_ref']:r for r in originals['partitions']}
    shadow_protocol=read(shadow/'protocol.json');cases={c['case_id']:c for c in shadow_protocol['cases']}
    result=[]
    for n in (1,3,9,17):
        case_id=f'case-{n:02d}';case=cases[case_id]
        baseline=checked(shadow/case_id/'baseline.json',case['baseline_sha256'])['partition']
        record=records[baseline['candidate_ref']]
        part=checked(Path(record['partition_path']),record['partition_sha256'])
        query_ref=originals[baseline['candidate_ref']]['query_ref'];q=checked(Path(query_ref['path']),query_ref['sha256'])
        sources={r['source_primitive_ref']:r for r in q['source_rows']}
        a,b=baseline['original_candidate']['polyline_display'];length=math.dist(a,b)
        box=case['images'][1]['bbox_display'];focus=[(box[k]+box[k+2])/2 for k in (0,1)]
        t=sum((focus[k]-a[k])*(b[k]-a[k])/length for k in (0,1))
        nearest_t=max(0,min(length,t))
        intervals=[r for r in part['body_intervals'] if r['parameter_interval'][0]<=nearest_t<=r['parameter_interval'][1]]
        refs=sorted({ref for r in intervals for ref in r['source_primitive_refs']})
        examples=[]
        for ref in refs:
            row=sources[ref];pts=row['points_display'];native=row['source_native_segment']
            axial=[sum((p[k]-a[k])*(b[k]-a[k])/length for k in (0,1)) for p in pts]
            examples.append({'source_ref':ref,'kind':native['kind'],'authored_path':native['drawing_ref'],
                'endpoints_display':[pts[0],pts[-1]],'axial_endpoint_span':max(axial)-min(axial),
                'bbox_intersects_review_detail':intersects(row['bbox_display'],box)})
        result.append({'case_id':case_id,'split':'development' if n in (1,3) else 'former_held_out_exposed_diagnostic',
            'candidate_ref':baseline['candidate_ref'],'partition_sha256':record['partition_sha256'],
            'query_sha256':query_ref['sha256'],'crop_focus_parameter':t,'nearest_scope_parameter':nearest_t,
            'focus_outside_scope_due_to_coordinate_rounding':t<0 or t>length,
            'body_intervals_at_nearest_focus':[{'parameter_interval':r['parameter_interval'],'reason':r['reason']} for r in intervals],
            'blocking_source_kind_counts':dict(Counter(e['kind'] for e in examples)),
            'blocking_source_refs':refs,'largest_axial_blockers':sorted(examples,key=lambda e:-e['axial_endpoint_span'])[:5],
            'before_after_body_parameters_equal':[r['parameter_interval'] for r in baseline['body_intervals']]==[r['parameter_interval'] for r in part['body_intervals']],
            'classifier_tuned':False,'untouched_evaluation':False})
    return {'records':result,'shadow_protocol_sha256':_file_sha256(shadow/'protocol.json'),
        'model_results_consumed':False,'evaluation_policy':'Former held-out cases are exposed diagnostics, not new untouched evaluation or tuning inputs.'}


def export(args):
    protocol=read(args.output/'protocol.json');review=read(args.review)
    if review['protocol_sha256']!=_file_sha256(args.output/'protocol.json'):raise ValueError('review snapshot changed')
    checked(Path(protocol['replay_path']),protocol['replay_sha256'])
    evidence=native_measurements(protocol)
    write(args.output/'native-measurements.json',evidence)
    if 'delta_reviews' in review:review=expand_review(review,evidence)
    for context in evidence['cases']:
        decision=next((r for r in review['decisions'] if r['case_id']==context['case_id']),None)
        if decision and not set(decision.get('source_refs',[]))<=set(context['source_refs']):raise ValueError('review source refs outside frozen crop')
    protected=protocol['protected_artifacts']
    for key in ('M4','audit_pdf'):
        if _file_sha256(Path(protected[key+'_path']))!=protected[key+'_sha256']:raise ValueError('protected artifact changed')
    result=review_gate(protocol,review)
    result['incomplete_review_crop_searches']=[c['case_id'] for c in evidence['cases'] if not c['complete_review_crop_search']]
    if result['incomplete_review_crop_searches']:result['publication_gate_passed']=False
    images=sorted((ROOT/'tmp/pdfs'/args.output.name).glob('*.png'))
    expected_images=2*sum(math.ceil(n/4) for n in protocol['counts'].values())
    if len(images)!=expected_images:raise ValueError('incomplete frozen review contact sheets')
    write(args.output/'review-results.json',{'protocol_sha256':review['protocol_sha256'],
        'review_sha256':_file_sha256(args.review),'native_measurements_sha256':_file_sha256(args.output/'native-measurements.json'),
        'review_images':[{'path':str(p),'sha256':_file_sha256(p)} for p in images],
        'crop_renderer_sha256':_file_sha256(ROOT/'tools/review_mep_interpretation_frontier.py'),
        'code_sha256':_file_sha256(Path(__file__)),'reviewer':review['reviewer'],'decisions':review['decisions'],**result})
    if args.shadow:write(args.output/'remaining-misses.json',diagnose_misses(protocol,args.shadow))
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','baseline','shadow','prior-review','output','review'):p.add_argument('--'+name,type=Path)
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--page',type=int,default=5)
    args=p.parse_args();export(args) if args.review else prepare(args)
