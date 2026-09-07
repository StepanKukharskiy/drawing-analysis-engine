#!/usr/bin/env python3
"""Render current local M4 identities without promoting the recovery proposals."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from tools.render_mep_page5_page_wide_recovery import (
    SYSTEM_COLOURS, SYSTEM_LABELS, _draw_polyline, _label,
)
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_component_evidence_classification import anchored_style_hypotheses
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_page_wide_path_classification import PageWidePathRolePack
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import VIEW_ROLES


def native_style_profiles(recovery, bindings, denominator):
    """Resolve styles/roles from compact source packs, never rendered colours."""
    import re
    requests = defaultdict(set)
    drawing_requests = defaultdict(set)
    for row in [*recovery['outlined_corridor_components'], *recovery['single_centreline_components']]:
        if row['state'] not in {'identified_mep_route','supported_unidentified_mep_candidate'}:
            continue
        intervals = row.get('source_path_ordinal_intervals', row.get('path_ordinal_intervals', []))
        for start,end in intervals:
            for ordinal in range(start,end+1):
                requests[ordinal].add(row['id'])
    accepted = {ref for r in bindings['relations'] if r['state']=='accepted'
                and r['relation_type']=='route_system' for ref in r['target_refs']}
    for c in bindings['outlined_route_composites']:
        if c['id'] in accepted:
            for ref in c['member_source_primitive_refs']:
                drawing_requests[int(re.match(r'drawing\[(\d+)\]',ref)[1])].add(c['id'])
    path_info,role_info = denominator['native_authored_path_pack'],recovery['path_role_pack']
    paths = NativePathPack(path_info['path'],path_info)
    roles = PageWidePathRolePack(role_info['path'],role_info)
    if len(paths)!=role_info['record_count'] or not paths.verify_hash() or not roles.verify_hash():
        raise ValueError('native path/role source pack changed')
    if any(ordinal<0 or ordinal>=len(paths) for ordinal in requests):
        raise ValueError('candidate native path reference outside source inventory')
    styles=denominator['native_descriptor_pack']['styles']
    profiles=defaultdict(lambda:{'styles':set(),'regions':set(),'roles':set(),'source_path_ordinals':[]})
    found_drawings=set()
    for path,role in zip(paths.records(),roles.records()):
        ordinal=path['path_ordinal']
        if path['drawing_ordinal'] in drawing_requests:
            found_drawings.add(path['drawing_ordinal'])
        for ref in requests.get(ordinal,set()) | drawing_requests.get(path['drawing_ordinal'],set()):
            p=profiles[ref]
            p['styles'].add(path['style_id']);p['regions'].add(path['region_role'])
            p['roles'].add(role['role']);p['source_path_ordinals'].append(ordinal)
    if found_drawings!=set(drawing_requests):
        raise ValueError('local anchor source path missing from native inventory')
    prohibited={'annotation_dimension','text_associated_stroke_candidate','measured_hatch_candidate',
                'architectural_boundary_candidate','drawing_furniture','excluded_non_view_content',
                'equipment_fitting_evidence'}
    result={}
    for ref,p in profiles.items():
        owned=len(p['regions'])==1 and p['regions']<=VIEW_ROLES
        homogeneous=len(p['styles'])==1 and None not in p['styles']
        native=styles[next(iter(p['styles']))] if homogeneous else None
        # Full native style, not RGB alone; serialization rounds extraction noise
        # only. No nearest-colour or width matching is performed.
        style = ({k:([round(v,6) for v in value] if isinstance(value,list) else
                    round(value,6) if isinstance(value,float) else value)
                  for k,value in native.items()} if native else None)
        key=(recovery['page_ref'],next(iter(p['regions'])),_sha256(style)) if owned and style and style.get('stroke') else None
        result[ref]={'key':key,'native_style':style,'region_roles':sorted(p['regions'],key=str),
            'measured_path_roles':sorted(p['roles']), 'source_path_ordinals':p['source_path_ordinals'],
            'route_role_checks_passed':bool(key and not p['roles'] & prohibited)}
    return result


def review_crops(run, source_path, folder):
    """Native-only contact crops for source-first inspection, not truth labels."""
    from PIL import Image, ImageDraw
    from io import BytesIO
    source=read(run/'source.json')
    bindings=read(run/'local-M4.json.gz')
    targets=read(run/'targets.json.gz')
    terms=read(run/'terminology.json.gz')
    observations={o['id']:o for o in terms['source_observations']}
    evidence={e['id']:e for e in bindings['binding_evidence']}
    rows=[]
    for r in bindings['relations']:
        if r['state']!='accepted' or r['relation_type']!='route_system':
            continue
        e=evidence[r['binding_evidence_refs'][0]]
        ref=e['automatic_search_certificate']['leader_observation_ref']
        observation=observations[ref]
        contact=e['automatic_search_certificate']['contact_point_display']
        box=fitz.Rect(observation['bbox_display']);box.include_point(contact)
        box+=(-18,-18,18,18)
        rows.append({'relation_ref':r['id'],'target_ref':r['target_refs'][0],
                     'observation':observation,'bbox_display':list(box)})
    with fitz.open(source_path) as pdf:
        pix=pdf[source['page']-1].get_pixmap(matrix=fitz.Matrix(2,2),annots=False)
    base=Image.open(BytesIO(pix.tobytes('png'))).convert('RGB')
    folder.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(rows,1):
        box=row['bbox_display']
        crop=base.crop(tuple(round(v*2) for v in box))
        crop.thumbnail((1100,580))
        image=Image.new('RGB',(1120,650),'white')
        image.paste(crop,(10,50))
        draw=ImageDraw.Draw(image)
        draw.text((10,10),f"{index:02d} | {row['observation']['text']} | {row['relation_ref']}",fill='black')
        image.save(folder/f'contact-{index:02d}.png')
    write(folder/'index.json',{'M4_sha256':_file_sha256(run/'local-M4.json.gz'),'rows':rows,
                             'review_decisions_are_not_source_evidence':True})
    return rows


def overlay_rows(recovery, bindings):
    """Only exact current composite identity replaces an old observation row.

    Equal source colours and nearby lines are never semantic mappings.
    """
    accepted = defaultdict(list)
    for relation in bindings['relations']:
        if relation['state']=='accepted' and relation['relation_type']=='route_system':
            for ref in relation['target_refs']:
                accepted[ref].append(relation)
    current = {c['id']:c for c in bindings['outlined_route_composites']}
    sizes=defaultdict(list)
    for relation in bindings['relations']:
        if relation['state']=='accepted' and relation['relation_type']=='route_size':
            for ref in relation['target_refs']:
                sizes[ref].append(relation['candidate'])
    identified = []
    for ref, relations in sorted(accepted.items()):
        kinds = {r['candidate']['kind'] for r in relations}
        if len(kinds)!=1:
            raise ValueError('conflicting accepted systems cannot render solid')
        composite = current[ref]
        identified.append({'id':ref,'polyline_display':composite['derived_geometry']['centreline_points_display'],
            'projected_path_display_points':composite['derived_geometry']['projected_path_display_points'],
            'system':next(iter(kinds)), 'sizes':sizes[ref], 'relation_refs':[r['id'] for r in relations]})
    def signature(points):
        points = tuple(tuple(round(v,5) for v in p) for p in points)
        return min(points,tuple(reversed(points)))
    signatures = {signature(row['polyline_display']) for row in identified}
    candidates, suppressed = [], []
    for source in [*recovery['outlined_corridor_components'],*recovery['single_centreline_components']]:
        if source['state'] not in {'identified_mep_route','supported_unidentified_mep_candidate'}:
            continue
        if source.get('outlined_composite_ref') in accepted or signature(source['polyline_display']) in signatures:
            suppressed.append(source['id'])
            continue
        row = deepcopy(source)
        # Presentation identity is assigned only by current anchored-style
        # replay below, not stale accepted flags or source colour alone.
        row['system_hypothesis'] = None
        row['state'] = 'supported_unidentified_mep_candidate'
        candidates.append(row)
    return identified,candidates,suppressed


def apply_boundary_corrections(recovery,partitions,bends):
    """Presentation consumes current geometry dispositions, never review IDs."""
    result=deepcopy(recovery);changes={r['candidate_ref']:r for r in partitions['partitions']}
    bodies=[];warnings=[];new=[]
    for name in ('outlined_corridor_components','single_centreline_components'):
        kept=[]
        for row in result[name]:
            correction=changes.get(row['id'])
            if correction is None:kept.append(row);continue
            if correction['original_candidate']!=row:raise ValueError('partition candidate changed')
            for interval in correction['retained_intervals']:
                retained=deepcopy(row)
                retained.update(id=interval['id'],polyline_display=interval['polyline_display'],
                    projected_path_display_points=interval['projected_path_display_points'],
                    outlined_composite_ref=None,state='supported_unidentified_mep_candidate',
                    attributes={},boundary_partition_ref=correction.get('id'))
                kept.append(retained);new.append(retained)
                warnings.extend(p['center_display'] for p in interval['boundary_ports'])
            if not correction.get('duplicate_outline_parent_refs'):
                bodies.append(correction)
        result[name]=kept
    for row in bends.get('records',[]):
        if row.get('state')!='supported_unidentified_mep_candidate':continue
        if row.get('geometry_state')!='accepted':raise ValueError('bend geometry certificate not accepted')
        result['outlined_corridor_components'].append(deepcopy(row));new.append(row)
        warnings.extend(row.get('endpoint_warning_points_display',[]))
    return result,bodies,warnings,new


def render(args):
    run = args.run
    source = read(run/'source.json')
    recovery = read(args.recovery)
    bindings = read(run/'extended-M4.json.gz')
    direct = read(run/'local-M4.json.gz')
    summary = read(run/'summary.json')
    if _file_sha256(args.source)!=source['source_pdf_sha256'] or recovery['inputs']['source_pdf_sha256']!=source['source_pdf_sha256']:
        raise ValueError('source snapshot changed')
    if recovery['page_ref']!=source['page_ref']:
        raise ValueError('recovery page differs')
    registry = read(Path(source['registry_path']))
    scope = next(p for p in registry['pages'] if p['page_ref']==source['page_ref'])
    scale_field = scope['fields']['scale']
    scale = float(scale_field['drawing_inches_per_paper_inch'])
    metres = lambda length: length/72*scale*.0254
    identified,candidates,suppressed = overlay_rows(recovery,bindings)
    denominator_path=Path(source['denominator_path'])
    if _file_sha256(denominator_path)!=source['denominator_sha256']:
        raise ValueError('source denominator changed')
    denominator=read(denominator_path)
    boundary_bodies=[];boundary_warnings=[];new_geometry=[]
    partitions_path=getattr(args,'partitions',None);bends_path=getattr(args,'bend_recovery',None)
    if partitions_path:
        partitions=read(partitions_path)
        if partitions['inputs']['recovery_sha256']!=_file_sha256(args.recovery):raise ValueError('boundary recovery snapshot changed')
        bends=read(bends_path) if bends_path else {}
        recovery,boundary_bodies,boundary_warnings,new_geometry=apply_boundary_corrections(recovery,partitions,bends)
        identified,candidates,suppressed=overlay_rows(recovery,bindings)
    profiles=native_style_profiles(recovery,direct,denominator)
    style_evidence=anchored_style_hypotheses(candidates=candidates,
        local_bindings=direct['relations'],profiles=profiles)
    hypotheses={r['candidate_ref']:r for r in style_evidence['hypotheses']}
    for row in candidates:
        row['system_hypothesis']=hypotheses[row['id']]['system_hypothesis']
    previous = {r['outlined_composite_ref'] for r in recovery['outlined_corridor_components'] if r['state']=='identified_mep_route'}
    local_refs = {ref for r in direct['relations'] if r['state']=='accepted' and r['relation_type']=='route_system' for ref in r['target_refs']}
    stops = read(run/'connection-stops.json.gz')
    accepted_refs = {r['id'] for r in identified}
    visible_stops = [s for s in stops if s['port']['composite_ref'] in accepted_refs]
    with fitz.open(args.source) as original:
        base = original[source['page']-1]
        rect = base.rect
        pix = base.get_pixmap(matrix=fitz.Matrix(2,2), colorspace=fitz.csRGB, annots=False)
        jpeg = pix.tobytes('jpeg',jpg_quality=88)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    occupied, failed_labels = [],[]
    with fitz.open() as pdf:
        page = pdf.new_page(width=rect.width+680,height=rect.height)
        page.insert_image(rect,stream=jpeg)
        page.draw_rect(rect,color=None,fill=(1,1,1),fill_opacity=.5)
        grouped = defaultdict(list)
        for row in candidates:
            grouped[row['system_hypothesis']].append(row)
        for system, rows in grouped.items():
            shape=page.new_shape()
            for row in rows:
                if len(row['polyline_display'])>=2:
                    shape.draw_polyline(row['polyline_display'])
            shape.finish(color=SYSTEM_COLOURS.get(system,(.46,.48,.51)),width=4,
                         dashes='[12 8] 0',stroke_opacity=.8,lineCap=1)
            shape.commit()
        for row in identified:
            _draw_polyline(page,row['polyline_display'],colour=SYSTEM_COLOURS[row['system']],width=7.5)
        body_colour=(.50,.18,.62)
        for body in boundary_bodies:
            defect=body.get('representation_defect')
            if defect:
                shape=page.new_shape()
                for native in defect['native_paths']:
                    if len(native['points_display'])>=2:shape.draw_polyline(native['points_display'])
                shape.finish(color=body_colour,width=2.5);shape.commit()
            else:
                for interval in body['body_intervals']:
                    a,b=interval['polyline_display'];box=fitz.Rect(min(a[0],b[0])-3,min(a[1],b[1])-3,max(a[0],b[0])+3,max(a[1],b[1])+3)
                    page.draw_rect(box,color=body_colour,width=1.5,fill=body_colour,fill_opacity=.08)
        for point in boundary_warnings:
            p=fitz.Point(point)
            page.draw_rect(fitz.Rect(p.x-4,p.y-4,p.x+4,p.y+4),color=body_colour,fill=(1,1,1),width=1.5)
            occupied.append(fitz.Rect(p.x-7,p.y-7,p.x+7,p.y+7))
        for stop in visible_stops:
            point=fitz.Point(stop['port']['center'])
            page.draw_circle(point,7,color=(.84,.43,.02),fill=(1,1,1),width=2)
            occupied.append(fitz.Rect(point.x-10,point.y-10,point.x+10,point.y+10))
        for i,row in enumerate(identified,1):
            dimensions={(r['value'],r['unit']) for r in row['sizes']}
            size_label=(f'{next(iter(dimensions))[0]:g}in' if len(dimensions)==1 and next(iter(dimensions))[1]=='in' else 'size?')
            tag=f"R{i:02d} {SYSTEM_LABELS[row['system']]} {size_label} {metres(row['projected_path_display_points']):.2f}m"
            if not _label(page,row['polyline_display'],tag,SYSTEM_COLOURS[row['system']],rect,occupied,size=18,required=True):
                failed_labels.append(row['id'])
        for body in boundary_bodies:
            if not _label(page,body['original_candidate']['polyline_display'],'BODY / SYMBOL ?',body_colour,rect,occupied,size=14,required=True):
                failed_labels.append(body['candidate_ref'])
        for row in new_geometry:
            if row.get('channel')=='native_microsegmented_bend':
                if not _label(page,row['polyline_display'],f"BEND ? {metres(row['projected_path_display_points']):.2f}m",(.4,.42,.46),rect,occupied,size=14,required=True):
                    failed_labels.append(row['id'])
        x,y=rect.width+34,65
        ink=(.09,.13,.19)
        def text(value,size=18,color=ink,bold=False):
            nonlocal y
            page.insert_text((x,y),value,fontsize=size,fontname='hebo' if bold else 'helv',color=color)
            y+=size*1.6
        text(f"PAGE {source['page']} | LOCAL M4",28,bold=True)
        text('Bounded interpretation - coverage remains open',17)
        y+=24
        for label,color,dash in [('Accepted system',(.05,.25,.9),None),('Colour/style-supported hypothesis',(.05,.25,.9),'[12 8] 0'),('Conflicting / unmapped system',(.46,.48,.51),'[12 8] 0')]:
            page.draw_line((x,y),(x+65,y),color=color,width=7,dashes=dash)
            page.insert_text((x+82,y+5),label,fontsize=18,color=ink)
            y+=43
        page.draw_circle((x+25,y),7,color=(.84,.43,.02),fill=(1,1,1),width=2)
        page.insert_text((x+82,y+5),'Connection / scope warning',fontsize=18,color=ink)
        y+=67
        if boundary_bodies:
            text('Purple: body/symbol UNKNOWN; square: analysis cut',16,body_colour)
            y+=15
        text('PROJECTED LENGTHS',23,bold=True)
        text('System          Accepted        Candidate*',17,bold=True)
        length_by_system=defaultdict(float)
        candidate_lengths=defaultdict(float)
        for r in identified:
            length_by_system[r['system']]+=metres(r['projected_path_display_points'])
        for r in candidates:
            candidate_lengths[r['system_hypothesis']]+=metres(r['projected_path_display_points'])
        for system in [*SYSTEM_LABELS,None]:
            if not length_by_system[system] and not candidate_lengths[system]:
                continue
            text(f"{SYSTEM_LABELS.get(system,'UNKNOWN'):<10} {length_by_system[system]:>9.2f} m     {candidate_lengths[system]:>9.2f} m",20,
                 SYSTEM_COLOURS.get(system,ink),True)
        text('*Candidate observations; overlaps may remain.',15)
        text('Not an installed-length or physical-run total.',15)
        y+=30
        text('INTERPRETATION RESULT',23,bold=True)
        for label,key in [('Local system bindings','accepted_local_system_bindings'),('Locally identified segments','locally_identified_segments'),('Extended segments','system_extended_segments')]:
            text(f'{label}: {summary[key]}',20)
        text(f'New local segments vs prior audit: {len(local_refs-previous)}',18)
        text('Size and elevation are not propagated.',17)
        y+=30
        text('CONNECTORS',23,bold=True)
        connections=read(run/'connection-applicability.json.gz')
        bend_count=sum(r['state']=='accepted' and r['relation_type']=='projected_native_bend' for r in connections)
        straight_count=sum(r['state']=='accepted' and r['relation_type']=='projected_collinear_boundary_join' for r in connections)
        text(f'Certified projected bends (L): {bend_count}',20)
        text(f'Certified straight interfaces: {straight_count}',20)
        text('T branches / equipment ports: unresolved',18)
        text('Physical connector types and counts: UNKNOWN',17)
        if boundary_bodies:
            text(f'Body/symbol observations for review: {len(boundary_bodies)}',17)
            text(f'Recovered bounded bend shapes: {sum(r.get("channel")=="native_microsegmented_bend" for r in new_geometry)}',17)
        y+=30
        text('REVIEW STATUS',23,bold=True)
        text('Wrong accepts: independent review required',17)
        text('Unclassified drawing geometry is not overlaid.',16)
        text('Unmarked source content is not certified absent.',16)
        text('Background preserves faded original colours.',16)
        text('Style correlation never accepts M4 identity.',16)
        y+=40
        text('Evidence hashes',16,bold=True)
        for name,digest in [('Source',source['source_pdf_sha256']),('Current M4',_file_sha256(run/'extended-M4.json.gz'))]:
            text(name,12)
            text(digest[:32],11)
            text(digest[32:],11)
        pdf.save(args.output,garbage=4,deflate=True)
    evidence_path=args.output.with_suffix('.style-hypotheses.json')
    style_evidence['inputs']={'local_M4_sha256':_file_sha256(run/'local-M4.json.gz'),
        'recovery_sha256':_file_sha256(args.recovery), 'denominator_sha256':source['denominator_sha256'],
        'path_role_pack_sha256':recovery['path_role_pack']['sha256']}
    write(evidence_path,style_evidence)
    manifest={'source':source,'run':str(run.resolve()),'recovery_sha256':_file_sha256(args.recovery),
        'current_M4_sha256':_file_sha256(run/'extended-M4.json.gz'),'pdf_sha256':_file_sha256(args.output),
        'identified_component_refs':[r['id'] for r in identified], 'candidate_component_refs':[r['id'] for r in candidates],
        'replaced_observation_refs':suppressed,'connection_warning_count':len(visible_stops),
        'failed_labels':failed_labels,'new_local_segments_vs_prior_audit':len(local_refs-previous),
        'source_raster_colourspace':'RGB',
        'style_hypotheses_path':str(evidence_path.resolve()),'style_hypotheses_sha256':_file_sha256(evidence_path),
        'colour_style_supported_candidates':sum(r['state']=='colour_style_supported' for r in hypotheses.values()),
        'conflicting_candidate_count':sum(bool(r['conflicting_systems']) for r in hypotheses.values()),
        'wrong_accepts':None,'complete_page_route_coverage':False}
    if partitions_path:
        manifest.update(boundary_partitions_sha256=_file_sha256(partitions_path),
            bend_recovery_sha256=_file_sha256(bends_path) if bends_path else None,
            corrected_source_candidate_refs=[r['candidate_ref'] for r in partitions['partitions']],
            body_symbol_observation_count=len(boundary_bodies),
            new_geometry_candidate_refs=[r['id'] for r in new_geometry],
            body_boundary_warning_count=len(boundary_warnings))
    write(args.output.with_suffix('.manifest.json'),manifest)
    print({k:v for k,v in manifest.items() if k in {'connection_warning_count','failed_labels','new_local_segments_vs_prior_audit'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--recovery',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'M&P mark-up against shop systems piping.pdf')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partitions',type=Path)
    parser.add_argument('--bend-recovery',type=Path)
    render(parser.parse_args())
