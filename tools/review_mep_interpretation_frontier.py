#!/usr/bin/env python3
"""Freeze a review sample, neutral backlog and one-interface frontier.

Review selection never accepts geometry or alters M4. Source crops are rendered
before a separate locator layer; manual decisions remain a separate overlay.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import sys

import fitz
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256


def replay_connections(run, output):
    from tools.generate_mep_page_callout_bindings import bind_page, OwnershipQueries, connection_applicability
    targets=read(run/'targets.json.gz');targets['junction_interior_version']=2
    direct,extended,abstentions=bind_page(read(run/'terminology.json.gz'),targets,
        OwnershipQueries(read(run/'capture-index.json')['ownership']))
    old=read(run/'extended-M4.json.gz')
    if _sha256(direct)!=_sha256(read(run/'local-M4.json.gz')):
        raise ValueError('connection-only replay changed local callout bindings')
    old_relation_hashes={_sha256(r) for r in old['relations']}
    changed_types={r['relation_type'] for r in extended['relations'] if r['state']=='accepted'
        and _sha256(r) not in old_relation_hashes}
    if changed_types-{'route_system'}:
        raise ValueError('connection-only replay changed non-system attributes')
    refs=lambda m:{ref for r in m['relations'] if r['state']=='accepted' and r['relation_type']=='route_system' for ref in r['target_refs']}
    added=refs(extended)-refs(old)
    composites={r['id']:r for r in extended['outlined_route_composites']}
    registry=read(Path(read(run/'source.json')['registry_path']))
    scope=next(p for p in registry['pages'] if p['page_ref']==targets['composites']['accepted_composites'][0]['page_ref'])
    scale=scope['fields']['scale']['drawing_inches_per_paper_inch']
    rows=[{'target_ref':ref,'polyline_display':composites[ref]['derived_geometry']['centreline_points_display'],
           'projected_length_m':composites[ref]['derived_geometry']['projected_path_display_points']/72*scale*.0254,
           'system_relations':[r for r in extended['relations'] if r['state']=='accepted' and r['relation_type']=='route_system' and ref in r['target_refs']]}
          for ref in sorted(added)]
    result={'base_run':str(run.resolve()),'base_targets_sha256':_file_sha256(run/'targets.json.gz'),
        'base_M4_sha256':_file_sha256(run/'extended-M4.json.gz'),'junction_interior_version':2,
        'before_identified_count':len(refs(old)),'after_identified_count':len(refs(extended)),
        'newly_identified_targets':rows,'removed_target_refs':sorted(refs(old)-refs(extended)),
        'newly_identified_projected_length_m':sum(r['projected_length_m'] for r in rows),
        'installed_length':None,'purchase_length':None,
        'size_propagated':'route_size' in changed_types,'elevation_propagated':'route_elevation' in changed_types}
    write(output/'connection-replay/local-M4.json.gz',direct)
    write(output/'connection-replay/extended-M4.json.gz',extended)
    write(output/'connection-replay/abstentions.json.gz',abstentions)
    write(output/'connection-replay/connection-applicability.json.gz',connection_applicability(targets))
    write(output/'connection-improvement.json',result)
    print({k:v for k,v in result.items() if k.endswith('_count') or k=='newly_identified_projected_length_m'})
    return result


def select_sample(rows, hypotheses):
    groups=defaultdict(list)
    for h in hypotheses:
        if h['state']=='colour_style_supported':
            row=rows[h['candidate_ref']]
            groups[(h['system_hypothesis'],row['channel'])].append(row)
    selected=[]
    for key, members in sorted(groups.items()):
        members=sorted(members,key=lambda r:(r['projected_path_display_points'],r['id']))
        indexes=(range(len(members)) if key[1]=='single_centreline_or_microsegment' else
                 sorted({round(i*(len(members)-1)/4) for i in range(5)}))
        selected.extend({'candidate_ref':members[i]['id'],'stratum':list(key),
                         'length_rank':i+1,'stratum_count':len(members)} for i in indexes)
    return selected


def export_review(args):
    """Bind authored review decisions to frozen inputs; never feed them into M4."""
    from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
    from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import PageWidePathRolePack
    study=read(args.output/'study.json');review=read(args.review_input)
    if review['baseline']!=study['inputs'] or review['source_pdf_sha256']!=study['source_pdf_sha256']:
        raise ValueError('review baseline changed')
    for path_key,hash_key in [('recovery_path','recovery_sha256'),('hypotheses_path','hypotheses_sha256')]:
        if _file_sha256(Path(study['inputs'][path_key]))!=study['inputs'][hash_key]:
            raise ValueError('review input changed')
    if _file_sha256(args.run/'extended-M4.json.gz')!=study['inputs']['M4_sha256']:
        raise ValueError('review M4 changed')
    decisions={r['candidate_ref']:r for r in review['decisions']}
    if len(decisions)!=len(review['decisions']) or set(decisions)!={r['candidate_ref'] for r in study['sample']}:
        raise ValueError('review must cover exact frozen sample once')
    binding_review=review.get('new_binding_review')
    if binding_review:
        if binding_review['M4_sha256']!=_file_sha256(args.output/'connection-replay/extended-M4.json.gz'):
            raise ValueError('reviewed connection replay changed')
        added=read(args.output/'connection-improvement.json')['newly_identified_targets']
        if {r['target_ref'] for r in added}!={r['target_ref'] for r in binding_review['decisions']}:
            raise ValueError('new target review is incomplete')
    from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import certify_junction_interior
    targets=read(args.run/'targets.json.gz')
    certificates=[certify_junction_interior(c,version=2) for c in targets['boundary_connections']
        if c['state']=='accepted' and c['relation_type'] in {'projected_collinear_boundary_join','projected_native_bend'}]
    write(args.output/'junction-certificates.json.gz',{'version':2,
        'targets_sha256':_file_sha256(args.run/'targets.json.gz'),'certificates':certificates})
    recovery=read(args.recovery)
    rows=[r for name in ('outlined_corridor_components','single_centreline_components') for r in recovery[name]]
    source=read(args.run/'source.json');registry=read(Path(source['registry_path']))
    scale=next(p for p in registry['pages'] if p['page_ref']==recovery['page_ref'])['fields']['scale']['drawing_inches_per_paper_inch']/72*.0254
    ranked=[]
    for rank,r in enumerate(sorted(study['neutral_candidates'],key=lambda r:(-r['projected_length_points'],r['candidate_ref'])),1):
        ranked.append({**r,'impact_rank':rank,'candidate_projected_length_m':r['projected_length_points']*scale,
                       'length_is_certified_system_total':False})
    witnesses=[];matched_paths=0
    if args.denominator:
        denominator=read(args.denominator);p=denominator['native_authored_path_pack'];q=recovery['path_role_pack']
        pack=NativePathPack(p['path'],p);roles=PageWidePathRolePack(q['path'],q)
        if not pack.verify_hash() or not roles.verify_hash() or len(pack)!=q['record_count']:
            raise ValueError('source path/role pack changed')
        styles=denominator['native_descriptor_pack']['styles']
        for path,role in zip(pack.records(),roles.records()):
            style=styles[path['style_id']] if path['style_id'] is not None else {}
            stroke=style.get('stroke');b=path['bbox_display']
            for region in review.get('missing_geometry_review_regions',[]):
                box=region['bbox_display']
                if (not stroke or [round(v,6) for v in stroke]!=region['observed_native_stroke'] or
                    b[2]<box[0] or b[0]>box[2] or b[3]<box[1] or b[1]>box[3]):continue
                matched_paths+=1
                # Keep the complete query count, but expand only long witnesses
                # or explicitly reviewed paths. The denominator stays compact.
                if (path['source_segment_length_points']<5 and
                    path['drawing_ordinal'] not in region.get('reviewed_drawing_ordinals',[])):continue
                memberships=[{'candidate_ref':r['id'],'state':r['state']} for r in rows
                    if any(start<=path['path_ordinal']<=end for start,end in r.get('source_path_ordinal_intervals',r.get('path_ordinal_intervals',[])))]
                witnesses.append({'review_region':region['id'],**path,'measured_path_role':role,
                    'recovery_memberships':memberships,'visual_role_not_inferred_from_colour':True})
    result={'inputs':study['inputs'],'review_input_sha256':_file_sha256(args.review_input),
        'reviewer':review['reviewer'],'engineer_review':False,
        'sample_class_counts':dict(Counter(r['review_class'] for r in decisions.values())),
        'sample_decisions':review['decisions'],'sample_size':len(decisions),
        'new_binding_review':binding_review,
        'reviewed_new_binding_count':len(binding_review['decisions']) if binding_review else 0,
        'wrong_accepts_observed_in_reviewed_new_bindings':sum(r['wrong_accept_observed'] for r in binding_review['decisions']) if binding_review else None,
        'population_precision':None,'population_recall':None,'population_wrong_accepts':None,
        'neutral_primary_reason_counts':study['neutral_reason_counts'],'neutral_ranked_by_candidate_length':ranked,
        'missing_geometry_source_witnesses':witnesses,'complete_sheet_coverage':False,
        'missing_geometry_query':{'matched_native_path_count':matched_paths,
            'witness_selection':'length >= 5 display points or explicitly reviewed native drawing ordinal',
            'denominator_manifest_sha256':_file_sha256(args.denominator) if args.denominator else None},
        'review_changes_M4':False,'installed_length':None,'purchase_length':None}
    write(args.output/'source-review.json',result)
    print({'sample_class_counts':result['sample_class_counts'],'neutral_ranked':len(ranked),'source_witnesses':len(witnesses)})


def make_study(run, recovery_path, hypotheses_path):
    recovery=read(recovery_path);evidence=read(hypotheses_path)
    rows={r['id']:r for name in ('outlined_corridor_components','single_centreline_components') for r in recovery[name]}
    neutral=[]
    for h in evidence['hypotheses']:
        if h['state']=='colour_style_supported':continue
        r=rows[h['candidate_ref']]
        category=('conflicting_evidence' if h['conflicting_systems'] else
                  'missing_current_anchor_or_supported_style_mapping' if h['reason']=='unmapped_native_style' else
                  'unclosed_role_evidence')
        neutral.append({'candidate_ref':r['id'],'blocking_reason':h['reason'],'category':category,
            'conflicting_systems':h['conflicting_systems'],'path_roles':h['source_profile'].get('measured_path_roles',[]),
            'projected_length_points':r['projected_path_display_points'],
            'missing_implementation_proven':False,'source_profile':h['source_profile']})
    neutral.sort(key=lambda r:(r['category'],-r['projected_length_points'],r['candidate_ref']))
    m4=read(run/'extended-M4.json.gz')
    accepted={ref for r in m4['relations'] if r['state']=='accepted' and r['relation_type']=='route_system' for ref in r['target_refs']}
    by_composite=defaultdict(list)
    for r in rows.values():
        if r.get('outlined_composite_ref'):by_composite[r['outlined_composite_ref']].append(r['id'])
    frontier=[]
    for c in read(run/'connection-applicability.json.gz'):
        known=set(c['composite_refs']) & accepted
        if len(known)!=1:continue
        other=next(ref for ref in c['composite_refs'] if ref not in accepted)
        frontier.append({'connection_ref':c['id'],'accepted_target_ref':next(iter(known)),
            'candidate_target_ref':other,'recovery_candidate_refs':by_composite[other],
            'state':c['state'],'reasons':c['reasons'],'bbox_display':c['search']['bbox_display']})
    return {'inputs':{'run':str(run.resolve()),'recovery_path':str(recovery_path.resolve()),
        'hypotheses_path':str(hypotheses_path.resolve()),'hypotheses_sha256':_file_sha256(hypotheses_path),
        'recovery_sha256':_file_sha256(recovery_path),'M4_sha256':_file_sha256(run/'extended-M4.json.gz')},
        'sample_selection':'All supported single-centreline candidates; five length-rank quantiles per outlined system; frozen before visual decisions',
        'sample':select_sample(rows,evidence['hypotheses']), 'neutral_candidates':neutral,
        'neutral_reason_counts':dict(Counter(r['blocking_reason'] for r in neutral)),
        'neutral_category_counts':dict(Counter(r['category'] for r in neutral)),
        'one_interface_frontier':frontier,'complete_sheet_coverage':False,
        'population_wrong_accepts':None,'engineering_authority_changed':False},rows


def crop_cards(source, page_number, cards, out, prefix, *, mark_endpoints=False):
    """One native rendering, with separate native-only and locator contact sheets."""
    with fitz.open(source) as pdf:
        pix=pdf[page_number-1].get_pixmap(matrix=fitz.Matrix(3,3),annots=False)
    base=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    out.mkdir(parents=True,exist_ok=True)
    for locator in (False,True):
        for batch in range((len(cards)+3)//4):
            image=Image.new('RGB',(1500,1200),'white');draw=ImageDraw.Draw(image)
            for i,card in enumerate(cards[batch*4:batch*4+4]):
                x,y=(i%2)*750,(i//2)*600
                box=card['bbox_display'];box=[round(v*3) for v in box]
                crop=base.crop(box);scale=min(730/crop.width,550/crop.height)
                crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
                image.paste(crop,(x+10,y+40))
                draw.text((x+10,y+8),f"{prefix}{batch*4+i+1:02d} | native source"+(' + locator' if locator else ''),fill='black')
                if locator:
                    pts=[(x+10+(p[0]*3-box[0])*scale,y+40+(p[1]*3-box[1])*scale) for p in card.get('polyline_display',[])]
                    if len(pts)>=2:draw.line(pts,fill='#b600ce',width=3)
                    if mark_endpoints:
                        for j,(px,py) in enumerate(pts):
                            draw.ellipse((px-5,py-5,px+5,py+5),outline='#b600ce',width=2)
                            draw.text((px+7,py-12 if j==0 else py+5),str(j+1),fill='#b600ce')
            image.save(out/f"{prefix}-{batch+1:02d}-{'locator' if locator else 'native'}.png")


def main(args):
    if args.review_input:
        export_review(args)
        return
    if args.replay_connections:
        replay_connections(args.run,args.output)
        return
    study,rows=make_study(args.run,args.recovery,args.hypotheses)
    cards=[]
    for row in study['sample']:
        route=rows[row['candidate_ref']];points=route['polyline_display']
        box=[min(p[i] for p in points)-30 for i in (0,1)]+[max(p[i] for p in points)+30 for i in (0,1)]
        row['bbox_display']=box
        cards.append({**row,'polyline_display':points})
    source=read(args.run/'source.json')
    if _file_sha256(args.source)!=source['source_pdf_sha256']:raise ValueError('source changed')
    study['source_pdf_sha256']=source['source_pdf_sha256']
    crop_dir=ROOT/'tmp/pdfs'/args.output.name
    study['crop_directory']=str(crop_dir)
    write(args.output/'study.json',study)
    crop_cards(args.source,source['page'],cards,crop_dir,'sample')
    frontier=[{**r,'bbox_display':[r['bbox_display'][i]+(-28 if i<2 else 28) for i in range(4)]}
              for r in study['one_interface_frontier']]
    crop_cards(args.source,source['page'],frontier,crop_dir,'frontier')
    print({'sample':len(cards),'neutral':len(study['neutral_candidates']),'one_interface_frontier':len(frontier)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--hypotheses',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--replay-connections',action='store_true')
    p.add_argument('--review-input',type=Path)
    p.add_argument('--denominator',type=Path)
    main(p.parse_args())
