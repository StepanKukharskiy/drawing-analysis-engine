#!/usr/bin/env python3
"""Source-first review of a frozen path-aware delta; no publication authority."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256


def frozen_evaluation(args,replay):
    """Repeat frozen source gold at its crop focus, never consume model answers."""
    review=read(args.frozen_evaluation_review);protocol=read(args.shadow/'protocol.json')
    if review['protocol_sha256']!=_file_sha256(args.shadow/'protocol.json'):raise ValueError('evaluation protocol changed')
    if protocol['source_pdf_sha256']!=replay['source_pdf_sha256']:raise ValueError('evaluation source changed')
    records={r['candidate_ref']:r for r in replay['records']};cases={c['case_id']:c for c in protocol['cases']}
    rows=[];counts=Counter()
    def state(part,t):
        for name in ('retained_intervals','body_intervals'):
            if any(a['parameter_interval'][0]<t<a['parameter_interval'][1] for a in part[name]):return name
        return 'unresolved_or_analysis_boundary'
    for d in review['decisions']:
        if d['split']!='evaluation':continue
        case=cases[d['case_id']];path=args.shadow/d['case_id']/'baseline.json'
        if _file_sha256(path)!=case['baseline_sha256']:raise ValueError('evaluation baseline changed')
        before=read(path)['partition'];record=records[before['candidate_ref']]
        path=Path(record['partition_path'])
        if _file_sha256(path)!=record['partition_sha256']:raise ValueError('replayed partition changed')
        after=read(path);a,b=before['original_candidate']['polyline_display'];length=math.dist(a,b)
        box=case['images'][1]['bbox_display'];point=[(box[i]+box[i+2])/2 for i in (0,1)]
        t=sum((point[i]-a[i])*(b[i]-a[i])/length for i in (0,1))
        old,new=state(before,t),state(after,t)
        rows.append({'case_id':d['case_id'],'candidate_ref':before['candidate_ref'],
            'gold':d['gold'],'stratum':d['stratum'],'focus_parameter':t,'before':old,'after':new,
            'partition_sha256':record['partition_sha256'],'scope':'reviewed crop-centre probe only'})
        counts['reviewed_cases']+=1
        counts['false_exclusions_recovered']+=d['gold']=='pipe' and old=='body_intervals' and new=='retained_intervals'
        counts['new_false_exclusions']+=d['gold']=='pipe' and old=='retained_intervals' and new=='body_intervals'
        counts['retained_contamination']+=d['gold']=='non_pipe' and new=='retained_intervals'
        counts['remaining_false_exclusions']+=d['gold']=='pipe' and new!='retained_intervals'
        counts['unresolved_gold_cases']+=d['gold']=='unresolved'
    result={'replay_sha256':_file_sha256(args.output/'replay.json'),
        'review_sha256':_file_sha256(args.frozen_evaluation_review),'records':rows,'counts':dict(counts),
        'scope':'same frozen evaluation, crop-centre probes; not full-interval or population metrics',
        'genuine_endpoint_recall':None,'independently_confirmed_genuine_endpoint_cases':0,
        'model_answers_consumed':False,'publication_gate_passed':False,
        'publication_blockers':['genuine endpoint gold unavailable; independent engineer adjudication pending'],
        'new_identity_accepts':0,'installed_length':None,'purchase_length':None}
    write(args.output/'frozen-evaluation.json',result);print(dict(counts))


def main(args):
    replay=read(args.output/'replay.json')
    if args.frozen_evaluation_review:
        frozen_evaluation(args,replay);return
    if args.review is None:
        from tools.review_mep_interpretation_frontier import crop_cards
        if _file_sha256(args.source)!=replay['source_pdf_sha256']:raise ValueError('source PDF changed')
        crop_cards(args.source,args.page,replay['fresh_sample'],ROOT/'tmp/pdfs'/args.output.name,'subpath')
        print({'fresh_changed_source_samples':len(replay['fresh_sample'])});return
    review=read(args.review)
    if review['replay_sha256']!=_file_sha256(args.output/'replay.json'):raise ValueError('review snapshot changed')
    decisions=review['decisions'];sample=replay['fresh_sample']
    if len(decisions)!=len(sample) or {d['candidate_ref'] for d in decisions}!={d['candidate_ref'] for d in sample}:raise ValueError('review must cover each sample exactly once')
    if any(d['classification'] not in {'pipe_interval','symbol_or_body','non_route_drawing_content','unresolved'} for d in decisions):raise ValueError('invalid review classification')
    contaminated=[d['candidate_ref'] for d in decisions if d['classification'] in {'symbol_or_body','non_route_drawing_content'}]
    unresolved=[d['candidate_ref'] for d in decisions if d['classification']=='unresolved']
    write(args.output/'review-results.json',{'replay_sha256':review['replay_sha256'],
        'review_sha256':_file_sha256(args.review),'reviewer':review['reviewer'],'decisions':decisions,
        'reviewed_pipe_intervals':sum(d['classification']=='pipe_interval' for d in decisions),
        'residual_contamination_in_sample':contaminated,'unresolved_review_intervals':unresolved,
        'new_exclusions_geometry_count':replay['counts']['new_exclusions_candidates'],
        'missed_pipe_intervals_population':None,'residual_contamination_population':None,
        'independent_human_engineer_review':False,'publication_gate_passed':False,
        'publication_blockers':['independently reviewed genuine endpoints and symbol/body negatives are still required']+
            (['fresh changed outcomes include contamination'] if contaminated else [])+
            (['fresh changed outcomes include unresolved geometry'] if unresolved else []),
        'boundary_changes_published':False,'new_identity_accepts':0,
        'installed_length':None,'purchase_length':None})
    print({'reviewed':len(decisions),'contamination':len(contaminated),'unresolved':len(unresolved),'published':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--page',type=int,default=5);p.add_argument('--review',type=Path)
    p.add_argument('--frozen-evaluation-review',type=Path)
    p.add_argument('--shadow',type=Path)
    main(p.parse_args())
