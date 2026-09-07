#!/usr/bin/env python3
"""One diagnostic page per original sheet, RGB raster base/vector overlays.

No legacy render is copied. Page-local lengths are observation sums, not a
deduplicated package quantity. Failures/unscaled/raster sheets remain visible.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read,write
from tools.render_mep_page_callout_bindings import (
    overlay_rows,apply_boundary_corrections,native_style_profiles)
from tools.render_mep_page5_page_wide_recovery import SYSTEM_LABELS,SYSTEM_COLOURS,_label
from src.drawing_engine.disciplines.mep.mep_component_evidence_classification import anchored_style_hypotheses
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256

INK=(.10,.14,.20)
NEUTRAL=(.43,.46,.50)


def metric_length(points,scale):
    return None if not scale or points is None else points/72*scale*.0254


def length_consistent(row):
    measured=sum(math.dist(a,b) for a,b in zip(row['polyline_display'],row['polyline_display'][1:]))
    return math.isclose(measured,row['projected_path_display_points'],abs_tol=1e-4,rel_tol=1e-5)


def page_rows(record,source_hash):
    empty={'outlined_corridor_components':[],'single_centreline_components':[]}
    if not record.get('run'):
        return [],[],[],{'state':'no_current_interpretation','reason':record.get('error','not processed')}
    run=Path(record['run']);source=read(run/'source.json')
    if source['source_pdf_sha256']!=source_hash or source['page']!=record['page']:
        raise ValueError('source/page mismatch')
    bindings=read(run/'extended-M4.json.gz')
    direct=read(run/'local-M4.json.gz')
    denominator_path=Path(source['denominator_path'])
    if _file_sha256(denominator_path)!=source['denominator_sha256']:raise ValueError('denominator changed')
    recovery=read(Path(record['recovery'])) if record.get('recovery') else empty
    warnings=[]
    if record.get('recovery'):
        if recovery['inputs']['source_pdf_sha256']!=source_hash or recovery['page_ref']!=source['page_ref']:
            raise ValueError('candidate source/page mismatch')
        if recovery['validation']['errors']:raise ValueError('invalid recovery')
    if record.get('partitions'):
        partitions=read(Path(record['partitions']))
        if partitions['inputs']['recovery_sha256']!=_file_sha256(Path(record['recovery'])):
            raise ValueError('partition recovery mismatch')
        bends=read(Path(record['bends'])) if record.get('bends') else {}
        recovery,_,warnings,_=apply_boundary_corrections(recovery,partitions,bends)
    identified,candidates,suppressed=overlay_rows(recovery,bindings)
    if any(not length_consistent(row) for row in identified):
        raise ValueError('accepted route length differs from its rendered centreline')
    for row in candidates:
        row['diagnostic_length_state']='consistent' if length_consistent(row) else 'unknown_reported_and_rendered_length_conflict'
    style={}
    if record.get('recovery'):
        profiles=native_style_profiles(recovery,direct,read(denominator_path))
        style=anchored_style_hypotheses(candidates=candidates,local_bindings=direct['relations'],profiles=profiles)
        hypotheses={r['candidate_ref']:r for r in style['hypotheses']}
        for row in candidates:row['system_hypothesis']=hypotheses[row['id']]['system_hypothesis']
    stops=read(run/'connection-stops.json.gz')
    accepted={r['id'] for r in identified}
    analysis_boundaries=warnings
    warnings = [s['port']['center'] for s in stops if s['port']['composite_ref'] in accepted]
    connections=read(run/'connection-applicability.json.gz')
    return identified,candidates,warnings,{
        'M4_sha256':_file_sha256(run/'extended-M4.json.gz'),
        'local_M4_sha256':_file_sha256(run/'local-M4.json.gz'),
        'denominator_sha256':source['denominator_sha256'],
        'boundary_partitions_sha256':_file_sha256(Path(record['partitions'])) if record.get('partitions') else None,
        'bend_recovery_sha256':_file_sha256(Path(record['bends'])) if record.get('bends') else None,
        'recovery_sha256':_file_sha256(Path(record['recovery'])) if record.get('recovery') else None,
        'style_hypotheses':style,'suppressed_duplicate_observations':suppressed,
        'candidate_analysis_boundaries_diagnostic_only':analysis_boundaries,
        'projected_bends':sum(r['state']=='accepted' and r['relation_type']=='projected_native_bend' for r in connections),
        'projected_straight_interfaces':sum(r['state']=='accepted' and r['relation_type']=='projected_collinear_boundary_join' for r in connections),
        'physical_connector_counts':None,'independent_review':'pending'}


def render(args):
    protocol=read(args.batch/'protocol.json')
    selected=getattr(args,'pages',None)
    execution=({'source_sha256':protocol['source_sha256'],'pages':[
        read(args.batch/f'page-{n:03d}'/'interpret.json') for n in selected]}
        if selected else read(args.batch/'interpret.json'))
    source_hash=_file_sha256(args.source)
    if source_hash!=protocol['source_sha256'] or source_hash!=execution['source_sha256']:raise ValueError('source changed')
    for p,digest in protocol['protected'].items():
        if _file_sha256(Path(p))!=digest:raise ValueError('protected baseline changed')
    registry=read(args.registry)
    if _file_sha256(args.registry)!=protocol['registry_sha256']:raise ValueError('registry changed')
    scopes={r['page_number']:r for r in registry['pages']}
    records={r['page']:r for r in execution['pages']}
    expected=set(selected) if selected else set(scopes)
    if (len(records)!=len(execution['pages']) or set(records)!=expected
            or not expected<=set(scopes)):raise ValueError('missing or duplicate page records')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('do not overwrite a diagnostic package')
    pages=[]
    with fitz.open(args.source) as source,fitz.open() as pdf:
        if not selected and len(source)!=len(records):raise ValueError('not all original sheets accounted')
        for number in sorted(records):
            record=records[number];scope=scopes[number]
            identified,candidates,warnings,evidence=page_rows(record,source_hash)
            original=source[number-1];rect=original.rect
            scale=scope['fields']['scale'].get('drawing_inches_per_paper_inch')
            # Authored PDF markups belong to the source appearance, not M4.
            # In particular, some divider pages contain only FreeText annots.
            source_annotations=list(original.annots() or [])
            pix=original.get_pixmap(matrix=fitz.Matrix(2,2),colorspace=fitz.csRGB,annots=True)
            page=pdf.new_page(width=rect.width+660,height=max(rect.height,1100))
            page.insert_image(rect,stream=pix.tobytes('jpeg',jpg_quality=88))
            page.draw_rect(rect,color=None,fill=(1,1,1),fill_opacity=.42 if identified or candidates else .10)
            for collection,key,width,dash in ((candidates,'system_hypothesis',4,'[12 8] 0'),(identified,'system',7.5,None)):
                grouped=defaultdict(list)
                for row in collection:grouped[row.get(key)].append(row)
                for system,rows in grouped.items():
                    shape=page.new_shape()
                    for row in rows:
                        if len(row['polyline_display'])<2:raise ValueError('unrenderable route')
                        shape.draw_polyline(row['polyline_display'])
                    shape.finish(color=SYSTEM_COLOURS.get(system,NEUTRAL),width=width,dashes=dash,lineCap=1)
                    shape.commit()
            # Warnings denote unresolved interfaces, never counted connectors.
            seen=set();occupied=[]
            for point in warnings:
                key=tuple(round(v,2) for v in point)
                if key in seen:continue
                seen.add(key);x,y=point
                page.draw_rect(fitz.Rect(x-4,y-4,x+4,y+4),color=(.8,.4,.05),fill=(1,1,1),width=1.5)
                occupied.append(fitz.Rect(x-7,y-7,x+7,y+7))
            unlabelled=[]
            for i,row in enumerate(identified,1):
                sizes={(s['value'],s['unit']) for s in row['sizes']}
                size=f'{next(iter(sizes))[0]:g}in' if len(sizes)==1 and next(iter(sizes))[1]=='in' else 'size?'
                length=metric_length(row['projected_path_display_points'],scale)
                tag=f"R{i:02d} {SYSTEM_LABELS.get(row['system'],row['system'])} {size} " + (f'{length:.2f}m' if length is not None else 'length?')
                if not _label(page,row['polyline_display'],tag,SYSTEM_COLOURS.get(row['system'],INK),rect,occupied,size=20,required=True):
                    unlabelled.append(row['id'])
            x=rect.width+30;y=58
            def text(s,size=18,bold=False,color=INK):
                nonlocal y
                if y+size>page.rect.height-70:raise ValueError('sidebar overflow')
                if fitz.get_text_length(s,fontname='hebo' if bold else 'helv',fontsize=size)>600:raise ValueError('sidebar line too wide: '+s)
                page.insert_text((x,y),s,fontsize=size,fontname='hebo' if bold else 'helv',color=color);y+=size*1.6
            text(f'SHEET {number:02d} / {len(source):02d}',30,True)
            text('DIAGNOSTIC - ENGINEER REVIEW REQUIRED',19,True)
            text('Pipe coverage remains incomplete.',17)
            if record['state']=='preserved_frozen_page':text('Existing local identities preserved.',16)
            if record['state']=='failed':text('Processing incomplete - see evidence manifest.',16,color=(.8,.2,.1))
            if not scale:text('Scale unresolved: metric lengths UNKNOWN.',16)
            y+=30
            for title,color,dash in [('Locally accepted system',(.1,.3,.85),None),('Colour/style hypothesis',(.1,.3,.85),'[12 8] 0'),('Unresolved system candidate',NEUTRAL,'[12 8] 0')]:
                page.draw_line((x,y),(x+64,y),color=color,width=5,dashes=dash)
                page.insert_text((x+82,y+5),title,fontsize=18,color=INK);y+=40
            text('Small amber square: unresolved interface.',16)
            y+=20
            text('PROJECTED OBSERVATIONS',23,True)
            text('System              Identified       Candidate',18,True)
            totals=defaultdict(lambda:[0.,0.])
            for rows,key,col in ((identified,'system',0),(candidates,'system_hypothesis',1)):
                for row in rows:
                    bucket=totals[row.get(key)]
                    if row.get('diagnostic_length_state','consistent')!='consistent':bucket[col]=None
                    elif bucket[col] is not None:bucket[col]+=row['projected_path_display_points']
            schedules=[]
            for system in sorted(totals,key=lambda s:s or ''):
                lengths=[metric_length(v,scale) for v in totals[system]]
                values=[f'{v:.2f} m' if v is not None else 'UNKNOWN' for v in lengths]
                label=SYSTEM_LABELS.get(system,'UNKNOWN')
                text(f'{label:<12} {values[0]:>12}   {values[1]:>12}',19,True,SYSTEM_COLOURS.get(system,INK))
                schedules.append({'system':system,'identified_projected_m':lengths[0],'candidate_projected_m':lengths[1]})
            if not totals:text('No measured routes published on this sheet.',17)
            text('Candidate sums may contain overlapping observations.',15)
            invalid_lengths=sum(r['diagnostic_length_state']!='consistent' for r in candidates)
            if invalid_lengths:text(f'{invalid_lengths} candidate length conflicts: affected sum UNKNOWN.',15)
            text('Repeated sheets are not added into a package total.',15)
            y+=25
            text('CONNECTOR / INTERFACE TYPES',22,True)
            text(f"L bends (certified projected): {evidence.get('projected_bends','UNKNOWN')}",18)
            text(f"Straight interfaces: {evidence.get('projected_straight_interfaces','UNKNOWN')}",18)
            text('T connectors / equipment ports: UNKNOWN',18)
            text('Physical connector quantities remain unresolved.',15)
            if not identified and not candidates:
                y+=25;text('No accepted overlay is not evidence of no pipes.',16)
                text('Original drawing retained for source-first review.',16)
            if unlabelled:
                y+=20;text(f'{len(unlabelled)} labels omitted to avoid collisions.',16)
                text('Their route geometry remains visible.',16)
            page.insert_text((x,page.rect.height-42),'SOURCE SHA256 '+source_hash[:24],fontsize=12,color=INK)
            pages.append({'page':number,'execution':record,'identified':identified,'candidates':candidates,
                          'schedule':schedules,'evidence':evidence,'unplaced_labels':unlabelled,
                          'drawing_inches_per_paper_inch':scale,
                          'rendered_route_count':len(identified)+len(candidates),
                          'source_raster':'RGB, 144 dpi, JPEG 88','review_state':'pending',
                          'original_annotation_count':len(source_annotations),
                          'original_annotations_rendered_as_background_only':True,
                          'installed_length':None,'purchase_length':None})
            print(number,len(identified),len(candidates),flush=True)
        pdf.save(args.output,garbage=4,deflate=True)
    write(args.output.with_suffix('.manifest.json'),{'source_sha256':source_hash,'pdf_sha256':_file_sha256(args.output),
          'renderer_sha256':_file_sha256(Path(__file__)),
          'protocol_sha256':_file_sha256(args.batch/'protocol.json'),'pages':pages,
          'status':'partial_render_preview' if selected else 'diagnostic_unreviewed',
          'source_pages_rendered':sorted(records),'complete_pipe_coverage':False,'package_length_total':None,
          'installed_length':None,'purchase_length':None})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pages',type=int,nargs='+',help='Bounded QA preview only; omit for the full package')
    p.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    p.add_argument('--registry',type=Path,default=ROOT/'fixtures/mep/m_and_p_coordination/m_and_p_coordination.sheet-registry.json')
    render(p.parse_args())
