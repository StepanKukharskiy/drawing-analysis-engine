#!/usr/bin/env python3
"""Append drawing-first interpretation gates to an exact snapshot-backed audit.

The presentation selects reviewed cases only after inference. The ZIP exports
every immutable snapshot artifact and exact review overlay, including records
omitted from readable PDF marks. No interpretation or approval is added here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import zlib

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore
from src.drawing_engine.audit.render_mep_network_drawing_audit import _draw_paths, _clip_line
from src.drawing_engine.audit.render_mep_partial_audit import _source_text_boxes

INK, GRAY = (.08,.15,.22), (.35,.4,.45)
GREEN, AMBER, RED, BLUE = (.02,.5,.32), (.8,.47,.04), (.76,.16,.17), (.03,.37,.7)


def text(page, box, value, size=12, color=INK):
    value = str(value).replace('\u2013','-').replace('\u2014','-').replace('ø',' dia ').replace('Ø',' dia ')
    if page.insert_textbox(fitz.Rect(box),value,fontsize=size,fontname='helv',color=color) < 0:
        raise ValueError('acceptance audit text exceeds its box: '+value)


def bounds(points, margin=20):
    return fitz.Rect(min(p[0] for p in points)-margin,min(p[1] for p in points)-margin,
                     max(p[0] for p in points)+margin,max(p[1] for p in points)+margin)


def native_path(row):
    """Sample native cubic controls for display only; never replace their evidence."""
    native = row['source_native_segment']
    controls = native.get('control_points_display', [])
    if native['kind'] != 'cubic' or len(controls) != 2:
        return row['points_display']
    points = [native['start_display'], *controls, native['end_display']]
    matrix = fitz.Matrix(*row.get('pdf_to_display_matrix', [1,0,0,1,0,0]))
    result = []
    for i in range(33):
        t = i / 32
        weights = [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3]
        result.append(list(fitz.Point(*(sum(w*p[k] for w,p in zip(weights,points)) for k in (0,1))) * matrix))
    return result


def panel(page, source, box, destination, overlays=(), marks=()):
    box = fitz.Rect(box) & source.rect
    destination = fitz.Rect(destination)
    scale = min(destination.width/box.width,destination.height/box.height)
    target = fitz.Rect(0,0,box.width*scale,box.height*scale)
    target += (destination.x0+(destination.width-target.width)/2,destination.y0+(destination.height-target.height)/2,
               destination.x0+(destination.width-target.width)/2,destination.y0+(destination.height-target.height)/2)
    pix = source.get_pixmap(matrix=fitz.Matrix(min(24,max(2,scale*1.5)),min(24,max(2,scale*1.5))),clip=box,alpha=False,annots=True)
    page.insert_image(target,stream=pix.tobytes('png'))
    matrix = box.torect(target)
    boxes = [r*matrix for r in _source_text_boxes(source) if r.intersects(box)]
    for paths,color,width in overlays:
        clipped = [_clip_line(fitz.Point(a), fitz.Point(b), box) for path in paths for a,b in zip(path,path[1:])]
        _draw_paths(page,[piece for piece in clipped if piece],boxes,color,width,matrix)
    for point,color,label in marks:
        pt=fitz.Point(point)*matrix
        if target.contains(pt):
            page.draw_circle(pt,3,color=color,fill=(1,1,1),width=1)
            if label:
                left = min(pt.x+5, destination.x1-160)
                top = max(destination.y0, min(pt.y-28, destination.y1-26))
                text(page,(left,top,left+155,top+26),label,9,color)
    page.draw_rect(destination,color=(.85,.88,.9),width=.5)
    return {'source_bbox_display':list(box),'destination_bbox':list(target)}


def checked_review(directory, manifest_name):
    manifest = json.loads((directory / manifest_name).read_text())
    for name, entry in manifest['files'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('independent review evidence changed: ' + name)
    return manifest


def export_snapshot(store, scope, output, review_directories=(), implementation=None, source=None, comparison_files=(), base_audit=None):
    snapshot=store.snapshot(**scope)
    manifest={'snapshot':snapshot,'artifacts':[], 'scope_complete_for_snapshot':True,
              'drawing_discovery_complete':False,'engineer_approved':False}
    staging=output.with_suffix('.partial.zip')
    with zipfile.ZipFile(staging,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        cursor=store.connection.execute('SELECT sa.name,sa.artifact_sha256 FROM snapshot_artifacts sa WHERE sa.snapshot_id=? ORDER BY sa.name',(snapshot['id'],))
        for row in cursor:
            if Path(row['name']).name != row['name']:
                raise ValueError('invalid artifact export name')
            # Native-query artifacts exceed a GiB. Export their exact bytes
            # without keeping a second uncompressed copy beside the audit.
            digest, byte_count = hashlib.sha256(), 0
            with archive.open('artifacts/'+row['name']+'.json','w',force_zip64=True) as target:
                for data in store.iter_artifact_bytes(**scope,name=row['name']):
                    target.write(data); digest.update(data); byte_count += len(data)
            if digest.hexdigest()!=row['artifact_sha256']:
                raise ValueError('snapshot artifact content hash changed')
            manifest['artifacts'].append({'name':row['name'],'sha256':row['artifact_sha256'],'bytes':byte_count})
            print(json.dumps({'phase':'snapshot_artifact_exported','artifact':row['name'],'bytes':byte_count}),flush=True)
        reviews=[dict(r) for r in store.connection.execute('SELECT * FROM reviews WHERE snapshot_id=? ORDER BY id',(snapshot['id'],))]
        archive.writestr('reviews.json',json.dumps(reviews,sort_keys=True))
        manifest['independent_review_files'] = []
        for prefix, directory in review_directories:
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                name = prefix + '/' + path.name
                archive.writestr(name, data)
                manifest['independent_review_files'].append({'name': name, 'sha256': hashlib.sha256(data).hexdigest()})
        manifest['comparison_baseline_files'] = []
        for name, path, expected in comparison_files:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('comparison baseline changed during rendering: '+str(path))
            archive.writestr(name,data)
            manifest['comparison_baseline_files'].append({'name':name,'sha256':expected,'bytes':len(data)})
        if implementation:
            frozen = json.loads((implementation/'implementation-manifest.json').read_text())
            context = json.loads(snapshot['manifest_json'])['context']
            for name, expected in context['implementation_sha256'].items():
                if frozen['files'].get(name) != expected:
                    raise ValueError('export implementation differs from snapshot: '+name)
            for name, expected in frozen['files'].items():
                path = implementation/name
                if not path.resolve().is_relative_to(implementation.resolve()):
                    raise ValueError('implementation export path escapes frozen runtime')
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('frozen implementation changed: '+name)
                archive.writestr('implementation/'+name, data)
            archive.writestr('implementation/implementation-manifest.json',json.dumps(frozen,sort_keys=True,indent=2))
            manifest['frozen_interpretation_implementation_included'] = True
        presentation=Path(__file__).read_bytes()
        archive.writestr('presentation/render_mep_interpretation_acceptance.py',presentation)
        manifest['presentation_implementation_sha256']=hashlib.sha256(presentation).hexdigest()
        if base_audit:
            data=json.dumps(base_audit,sort_keys=True,indent=2).encode()
            archive.writestr('presentation/base-drawing-audit.manifest.json',data)
            manifest['base_drawing_audit_manifest_sha256']=hashlib.sha256(data).hexdigest()
        if source:
            data=source.read_bytes()
            if hashlib.sha256(data).hexdigest()!=snapshot['source_sha256']:
                raise ValueError('export source PDF differs from the frozen snapshot')
            archive.writestr('source/original-drawing.pdf',data)
            manifest['original_source_pdf_included']=True
        archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True,indent=2))
    staging.replace(output)
    return {'path':str(output.resolve()),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'artifact_count':len(manifest['artifacts']),'review_overlay_count':len(reviews)}


def render(args):
    scope={'project_id':args.project,'document_id':args.document,'snapshot_id':args.snapshot}
    base_manifest=json.loads(args.base_pdf.with_suffix('.manifest.json').read_text())
    if base_manifest['snapshot_id']!=args.snapshot or base_manifest['pdf_sha256']!=hashlib.sha256(args.base_pdf.read_bytes()).hexdigest():
        raise ValueError('base drawing audit is not this exact frozen snapshot')
    comparison_files = []
    for cohort, directory, names in (
            ('pilot',args.pilot_baseline,('trace-completion.json',)),
            ('package',args.package_baseline,('trace-completion.json','attribute-bindings.json'))):
        if directory:
            comparison_files.extend(('comparison-baselines/'+cohort+'/'+name,directory/name,
                hashlib.sha256((directory/name).read_bytes()).hexdigest()) for name in names)
    with ProjectKnowledgeStore(args.database) as store:
        read=lambda name:store.artifact(**scope,name=name)
        registry,review,trace,bindings,equipment=map(read,('sheet-registry','review-outcomes','trace-completion','attribute-bindings','equipment-2d-identities'))
        snapshot=store.snapshot(**scope)
        digest=hashlib.sha256(args.source.read_bytes()).hexdigest()
        if digest!=registry['document']['source_pdf_sha256'] or digest!=base_manifest['source_pdf_sha256']:
            raise ValueError('acceptance source PDF differs from frozen evidence')
        review_directories = []
        independent = None
        if args.source_matching or args.source_review:
            if not args.source_matching or not args.source_review:
                raise ValueError('both frozen source review and matching overlay are required')
            original = checked_review(args.source_review, 'freeze-manifest.json')
            matching = checked_review(args.source_matching, 'matching-manifest.json')
            independent = json.loads((args.source_matching / 'matching-overlay.json').read_text())
            if (matching['snapshot_id'] != args.snapshot or independent['snapshot_id'] != args.snapshot
                    or independent['source_pdf_sha256'] != digest
                    or independent['source_review_manifest_sha256'] != hashlib.sha256((args.source_review / 'freeze-manifest.json').read_bytes()).hexdigest()):
                raise ValueError('independent review does not refer to this source and snapshot')
            for name, expected in independent['artifact_sha256'].items():
                actual = store.connection.execute('SELECT artifact_sha256 FROM snapshot_artifacts WHERE snapshot_id=? AND name=?', (args.snapshot, name)).fetchone()
                if not actual or actual[0] != expected:
                    raise ValueError('independent matching references another artifact: ' + name)
            review_directories = [('independent-source-review', args.source_review), ('independent-matching', args.source_matching)]
        pages={p['page_ref']:p['page_number'] for p in registry['pages']}
        baseline=json.loads((args.pilot_baseline/'trace-completion.json').read_text())
        pilot_ids={r['composite_refs'][0] for r in baseline['scoped_traces']}
        current={r['composite_refs'][0]:r for r in trace['scoped_traces']}
        if not pilot_ids.issubset(current):
            raise ValueError('current snapshot is missing original pilot corridor scopes')
        audit={'snapshot_id':args.snapshot,'source_pdf_sha256':digest,'base_pdf_sha256':base_manifest['pdf_sha256'],
            'comparison_baseline_file_sha256':{name:digest for name,_,digest in comparison_files},
            'presentation_implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'base_page_count':base_manifest['page_count'],'acceptance_pages':[],
            'pilot_scope_complete_count':sum(current[ref]['projected_scope_complete'] for ref in pilot_ids),
            'pilot_scope_count':len(pilot_ids),'package_scope_complete_count':trace['complete_scoped_trace_count'],
            'package_scope_count':len(trace['scoped_traces']),'readable_selection_is_exhaustive':False,
            'classification_budget_limited_scope_count':sum('native_straight_support_work_budget_exhausted' in r['reasons'] for r in trace['scoped_traces']),
            'installed_length_established':False,'equipment_port_positive_established':False}
        if args.package_baseline:
            prior_trace = json.loads((args.package_baseline/'trace-completion.json').read_text())
            prior = {r['composite_refs'][0]:r for r in prior_trace['scoped_traces']}
            audit['package_corridor_delta'] = {'before_complete':prior_trace['complete_scoped_trace_count'],
                'after_complete':trace['complete_scoped_trace_count'],'denominator_before':len(prior),'denominator_after':len(current),
                'newly_complete_refs':sorted(ref for ref in prior if ref in current and not prior[ref]['projected_scope_complete'] and current[ref]['projected_scope_complete']),
                'regressed_refs':sorted(ref for ref in prior if prior[ref]['projected_scope_complete'] and (ref not in current or not current[ref]['projected_scope_complete'])),
                'missing_scope_refs':sorted(set(prior)-set(current)), 'not_network_recall':True}
            del prior_trace,prior
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with fitz.open(args.source) as source,fitz.open(args.base_pdf) as pdf:
            def new(title,subtitle):
                page=pdf.new_page(width=1190,height=842)
                text(page,(32,26,1160,66),title,23)
                text(page,(32,70,1155,103),subtitle,12,GRAY)
                text(page,(32,806,1155,834),f'Snapshot {args.snapshot[:16]} | Appendix page {len(pdf)-base_manifest["page_count"]} | Projected drawing interpretation; no installed quantity or engineer approval',9,GRAY)
                return page
            page=new('Interpretation acceptance - original pilot, then package','The pages 2-3 six-corridor benchmark remains separate from the retained 18-page geometry.')
            lines=[f'Benchmark: {review["summary"]["correct_bindings"]}/{review["summary"]["expected_bindings"]} correct attributes; {review["summary"]["correctly_identified_and_markable_items"]}/{review["summary"]["reviewed_item_denominator"]} fully identified items; {review["summary"]["wrong_bindings_in_reviewed_scope"]} wrong accepted attributes in this benchmark.',
                f'Original pilot: {audit["pilot_scope_complete_count"]}/{len(pilot_ids)} complete nominated corridors; before: 2/6.',
                f'Retained package: {audit["package_scope_complete_count"]}/{audit["package_scope_count"]} complete nominated corridors (before: {audit.get("package_corridor_delta",{}).get("before_complete","unmeasured")}). This is not network recall.',
                'Green marks preserve supported local scopes. Amber marks show proposed applicability or partial geometry. Red marks show retained blockers.',
                'All four elevation extents remain undetermined. Literal bottom values and direct targets are known; projected connections do not extend elevation.',
                'Equipment: existing HUH body/tag identities are retained. No defensible equipment-port positive was found under the supported interpretation.',
                f'Unresolved: {audit["package_scope_count"]-audit["package_scope_complete_count"]} nominated corridors; {audit["classification_budget_limited_scope_count"]} retain a classification budget limit. Fresh discovery is still needed outside retained candidates and unsearched interiors.',
                'The companion ZIP exports every snapshot artifact and evidence record. Selected PDF pages do not imply exhaustive visual review.']
            y=125
            for line in lines:text(page,(40,y,1145,y+62),line,17);y+=78
            audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'coverage_boundary'})
            if args.package_baseline:
                previous = json.loads((args.package_baseline/'attribute-bindings.json').read_text())
                old = {r['id']:r for r in previous['relations'] if r['state']=='accepted'}
                now = {r['id']:r for r in bindings['relations'] if r['state']=='accepted'}
                added = sorted(set(now)-set(old))
                audit['package_relation_delta'] = {'before':len(old),'after':len(now),'added_refs':added,
                    'missing_refs':sorted(set(old)-set(now)), 'changed_refs':[ref for ref in old if ref in now and old[ref]!=now[ref]],
                    'independent_recall_measurement':False}
                pilot_pages = {r['page_ref'] for r in baseline['scoped_traces']}
                targets = {ref for key in added for ref in now[key]['target_refs'] if now[key]['page_ref'] not in pilot_pages}
                terms = read('terminology-proposals')
                props = {r['id']:r for r in terms['proposals']};obs = {r['id']:r for r in terms['source_observations']}
                comps = {r['id']:r for r in bindings['outlined_route_composites']}
                for target in sorted(targets):
                    if target not in comps:continue
                    relations = [now[key] for key in added if target in now[key]['target_refs']]
                    annotations = {}
                    for relation in relations:
                        annotation = obs[props[relation['proposal_ref']]['anchor_ref']]
                        while annotation.get('source_observation_ref') in obs:
                            annotation = obs[annotation['source_observation_ref']]
                        annotations[annotation['id']] = annotation
                    comp = comps[target];number=pages[comp['page_ref']]
                    path = comp['derived_geometry']['centreline_points_display']
                    points = list(path)+[list(p) for a in annotations.values() for p in (fitz.Rect(a['bbox_display']).tl,fitz.Rect(a['bbox_display']).br)]
                    page=new(f'Additional package binding - drawing {number}','Outside the pages 2-3 pilot | Newly accepted local attributes on retained geometry; no new route or physical quantity.')
                    panel(page,source[number-1],bounds(points,30),(32,115,1158,615),[([path],GREEN,3)])
                    text(page,(35,640,1140,702),'Source callout: '+' / '.join(a['text'] for a in annotations.values())+'\nNew bindings: '+', '.join(r['relation_type'].replace('route_','') for r in relations),18)
                    text(page,(35,716,1140,790),'The native leader targets the highlighted route. A bounded two-anchor body explains competing body strokes. Elevation applicability and physical continuation are not added.\nPost-inference review case; outside the independent source inventory.',14,GRAY)
                    audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'additional_package_binding','source_page_number':number,
                        'target_ref':target,'accepted_relation_refs':[r['id'] for r in relations],
                        'independent_source_inventory_case':False,'engineer_approved':False})
                del terms,props,obs,comps
            elevations=review['elevation_applicability']['outcomes']
            for benchmark in review['outcomes']:
                if benchmark.get('subject_kind')!='annotation_binding' or benchmark.get('relation_type')!='route_elevation':continue
                nominated={r['source_observation_ref'] for r in benchmark['annotation_candidates']}
                cases=[r for r in elevations if r['source_observation_ref'] in nominated and any(t['target_ref'] in benchmark['subject_refs'] for t in r['candidate_extents'])]
                if len(cases)!=1:raise ValueError('benchmark elevation does not have one independently recorded source scope')
                row=cases[0];number=pages[row['page_ref']]
                page=new(f'Elevation applicability - drawing {number}', 'Original four-item benchmark | Direct ownership is accepted; extension across projected joins is undetermined.')
                paths=[t['points_display'] for t in row['candidate_extents']]
                accepted=[t['points_display'] for t in row['candidate_extents'] if t['state']=='existing_direct_binding']
                annotation=fitz.Rect(row['annotation_bbox_display'])
                points=[p for path in paths for p in path]+[list(annotation.tl),list(annotation.br)]
                direct_points=[p for path in accepted for p in path]+[list(annotation.tl),list(annotation.br)]
                overlays=[(paths,AMBER,3),(accepted,GREEN,3)]
                contacts=[(e['automatic_search_certificate']['contact_point_display'],BLUE,'Direct contact') for e in row['direct_contact_evidence']]
                text(page,(32,106,570,132),'Annotation and directly bound target',15)
                panel(page,source[number-1],bounds(direct_points,18),(32,138,545,550),overlays,contacts)
                text(page,(574,106,1155,132),'Proposed extent and scope boundaries',15)
                stop_labels={'unresolved_projected_destination':'Unresolved destination',
                    'retained_geometry_scope_exit':'Retained scope exit'}
                stops=[(b['point_display'],RED,stop_labels.get(b['reason_codes'][0],'Scope boundary'))
                    for b in row['boundaries'] if b['candidate_extent_stops_here']]
                panel(page,source[number-1],bounds(points,25),(575,138,1158,550),overlays,stops)
                text(page,(35,579,1140,618),f'Literal: {row["literal_text"]}   |   Basis: {row["candidate"].get("basis","unknown")}   |   Datum: not independently established',18)
                text(page,(35,636,1140,707),'Accepted extent: the existing local M4 target only. No downstream elevation extent is accepted.\nStop: positive evidence for a constant elevation extent or an applicable section/detail is missing.',16)
                reasons=sorted({stop_labels.get(code,code.replace('_',' ')) for b in row['boundaries'] if b['candidate_extent_stops_here'] for code in b['reason_codes']})
                text(page,(35,727,1140,789),'Candidate path stops: '+', '.join(reasons)+'.\nLeader ownership: supported. Change/riser search: retained observations only. No applicable section/detail established; the proposed path does not prove a level pipe.',13,GRAY)
                audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'elevation_applicability','source_page_number':number,
                    'outcome_ref':row['id'],'benchmark_outcome_ref':benchmark['id']})
            # All six original pilot scopes, plus every newly completed package-only scope.
            comparisons = [(r, 'Pilot corridor', 'Original six-corridor benchmark') for r in baseline['scoped_traces']]
            if args.package_baseline:
                prior_trace = json.loads((args.package_baseline/'trace-completion.json').read_text())
                comparisons.extend((r, 'Additional package corridor', 'Retained package; outside the original pilot')
                    for r in prior_trace['scoped_traces'] if r['composite_refs'][0] not in pilot_ids
                    and not r['projected_scope_complete'] and current[r['composite_refs'][0]]['projected_scope_complete'])
            needed_scopes={current[r['composite_refs'][0]]['scope_ref'] for r,_,_ in comparisons}
            native_by_scope={q['scope_ref']:{r['source_primitive_ref']:r for r in q['source_rows']}
                for q in read('trace-source-queries')['source_queries'] if q['scope_ref'] in needed_scopes}
            for before, title, cohort in comparisons:
                ref=before['composite_refs'][0];after=current[ref];number=pages[after['page_ref']]
                old_bad={r['source_primitive_ref'] for r in before['source_classifications'] if r['classification'].startswith(('unresolved_','unsupported_'))}
                new_bad={r['source_primitive_ref'] for r in after['source_classifications'] if r['classification'].startswith(('unresolved_','unsupported_'))}
                resolved=old_bad-new_bad;regressed=new_bad-old_bad
                # Source geometry is from the exact imported query artifact.
                native=native_by_scope[after['scope_ref']]
                path=after['scope']['input_composite']['derived_geometry']['centreline_points_display']
                box=bounds(path,30)
                page=new(f'{title} - drawing {number}',f'{cohort} | Before: {before["state"]} / {len(old_bad)} blockers. After: {after["state"]} / {len(new_bad)} blockers.')
                inset_destinations=None
                if box.width > 2*box.height:
                    text(page,(32,109,230,138),'BEFORE',16);text(page,(32,344,230,373),'AFTER',16)
                    destinations=[(32,141,1158,329),(32,376,1158,564)]
                else:
                    text(page,(32,109,560,138),'BEFORE',16);text(page,(608,109,1155,138),'AFTER',16)
                    destinations=[(32,150,567,562),(608,150,1158,562)]
                    if box.height > 2*box.width and resolved:
                        destinations=[(32,150,275,562),(608,150,851,562)]
                        inset_destinations=[(295,230,567,490),(871,230,1158,490)]
                        detail_box=bounds([p for ref in resolved for p in native_path(native[ref])],6)
                for index,(dest,bad,known) in enumerate(zip(destinations,(old_bad,new_bad),(set(),resolved))):
                    overlays=[([path],BLUE,2),([native_path(native[r]) for r in sorted(bad)[:60]],RED,.9),
                              ([native_path(native[r]) for r in sorted(known)],GREEN,2)]
                    panel(page,source[number-1],box,dest,overlays)
                    if inset_destinations:
                        detail=inset_destinations[index]
                        text(page,(detail[0],198,detail[2],224),'Explained-stroke detail',12,GRAY)
                        panel(page,source[number-1],detail_box,detail,overlays)
                text(page,(35,583,1140,633),f'{len(resolved)} blockers explained; {len(regressed)} new blockers. Original primitives and alternatives retained.\nBlue: reviewed centreline. Red: unresolved stroke. Green: explained stroke.',15)
                classifications=Counter(r.get('native_stroke_role',r['classification']) for r in after['source_classifications'] if r['source_primitive_ref'] in resolved)
                explanations={'named_grid_axis_dash_sequence':'Grid strokes: opposing named bubbles and a complete dash sequence',
                    'transverse_curve_without_native_endpoint_attachment':'Native curves cross the corridor without endpoint attachment; no join is added'}
                description='; '.join(f'{value} - {explanations.get(key,key.replace("_"," "))}' for key,value in classifications.items())
                if not description:
                    description = ('The earlier complete corridor is preserved; no additional blocker needed classification.'
                        if after['projected_scope_complete'] else 'No blocker removed. The evidence remains insufficient.')
                text(page,(35,645,1140,702),description,15)
                refs=', '.join(sorted(resolved)) if resolved else '; '.join(after['reasons']) or 'Both earlier completion checks remain closed.'
                if max(len(old_bad),len(new_bad))>60:
                    refs='At most 60 unresolved strokes are marked per panel; all blockers remain in the evidence ZIP.\n'+refs
                text(page,(35,722,1140,788),refs,12,GRAY)
                audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'corridor_before_after','source_page_number':number,
                    'cohort':cohort,'composite_ref':ref,'before_blockers':sorted(old_bad),'after_blockers':sorted(new_bad),
                    'resolved_blockers':sorted(resolved),'regressed_blockers':sorted(regressed),'maximum_displayed_unresolved_strokes':60})
            del native_by_scope,native
            connected=trace.get('connected_trace_completion',{})
            pilots=[r for r in connected.get('complete_subtraces',[]) if set(r['composite_refs']).intersection(pilot_ids)]
            for row in pilots:
                number=pages[row['page_ref']];parts=[p['points_display'] for p in row['centreline_parts']]
                page=new(f'Extended projected trace - drawing {number}','Complete only inside the named native port exits. External continuation remains unresolved.')
                panel(page,source[number-1],bounds([p for path in parts for p in path],35),(32,115,1158,574),
                    [(parts,GREEN,3)],[(p['point_display'],RED,'Scope exit') for p in row['scope_exits']])
                text(page,(35,594,1140,639),f'Projected length: {row["projected_length_m"]:.3f} m   |   {row["member_corridor_count"]} certified corridor(s) + {row["covered_interface_count"]} covered interface(s)',19)
                parents=[p for p in connected['parent_scopes'] if set(p['composite_refs']).intersection(row['composite_refs'])]
                missing=[m['composite_ref'] for p in parents for m in p['uncovered_members']]
                junctions=sorted({ref for p in parents for ref in p['uncovered_junction_refs']})
                text(page,(35,656,1140,704),f'Parent trace: partial. Uncovered members: {len(missing)}; uncovered junctions: {len(junctions)}. No installed length, constant elevation or physical fitting continuity is established.',16)
                text(page,(35,718,1140,791),'Uncovered members: '+', '.join(missing)+'\nUncovered junctions: '+(', '.join(junctions) or 'none inside this parent scope'),10,GRAY)
                audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'extended_projected_trace','trace_ref':row['id'],
                    'source_page_number':number,'uncovered_parent_members':missing,'uncovered_parent_junctions':junctions})
            chosen=[]
            for state in ('accepted','abstained'):
                selected=next((r for r in equipment['identities'] if r['state']==state),None)
                if selected:chosen.append(selected)
            observations={r['id']:r for r in equipment['source_observations']}
            for row in chosen:
                number=row['page_number'];boxes=[fitz.Rect(observations[ref]['bbox_display']) for ref in row['source_observation_refs']]
                if row.get('body_bbox_display'):boxes.append(fitz.Rect(row['body_bbox_display']))
                box=bounds([list(p) for b in boxes for p in (b.tl,b.br)],55)
                page=new(f'Equipment port boundary - {row["equipment_tag"]}',f'Drawing {number} | Body/tag: {row["state"]}. Port: unresolved. No physical connection is accepted.')
                body_paths = []
                if row['state']=='accepted' and row.get('body_bbox_display'):
                    body = fitz.Rect(row['body_bbox_display'])
                    body_paths = [[list(p) for p in (body.tl,body.tr,body.br,body.bl,body.tl)]]
                candidates=row['port_outcome'].get('port_geometry_candidates',[])
                terminal=next((c for c in candidates if c.get('contacts') and all(p['contact_kind']=='native_endpoint_incidence' for p in c['contacts'])),None)
                port_marks=[] if terminal is None else [(terminal['contacts'][0]['point_display'],AMBER,'Terminal pair: unresolved')]
                panel(page,source[number-1],box,(32,115,1158,585),[(body_paths,GREEN,2)],port_marks)
                explanation = ('Accepted HUH body/tag identity does not establish a port. Repeated casing tabs and pipes crossing the closed casing do not supply an independent connector or opening.'
                    if row['state']=='accepted' else
                    'The equipment tag is observed, but its body association remains unresolved. Neither the nearby route nor the matching tag supplies a body-owned connector or port.')
                text(page,(35,610,1140,687),explanation,18)
                text(page,(35,714,1140,787),'Result: port positive remains unmet for this supported source interpretation. An explicitly applicable service detail or independent native interface is still required; unresolved candidates remain in the exported evidence.',15,GRAY)
                audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'equipment_port_boundary','identity_ref':row['id'],'source_page_number':number,
                    'displayed_unresolved_port_candidate_ref':terminal['id'] if terminal else None})
            if independent:
                summary = independent['summary']
                equipment_check = independent['equipment_check']
                page = new('Independent source-first review', 'Two frozen source regions were inventoried before engine candidates were shown to an independent AI reviewer. No human approval.')
                counts = summary['route_status_counts']
                lines = [
                    f'Routes: {counts.get("fully",0)} fully covered within the frozen 2-point endpoint tolerance; {counts.get("partly",0)} partial; {counts.get("missed",0)} missed; {counts.get("ambiguous",0)} ambiguous. Denominator: {summary["route_denominator"]}.',
                    f'Bends / branches: {summary["join_positive_denominator"]} source positives. Outcomes: ' + '; '.join(f'{k.replace("_"," ")}: {v}' for k,v in summary['join_positive_counts'].items()),
                    f'Crossing negatives: {summary["crossing_negative_denominator"]}. ' + '; '.join(f'{k.replace("_"," ")}: {v}' for k,v in summary['crossing_negative_counts'].items()) + '. Missing geometry is not a successful negative.',
                    f'Local attributes: {summary["local_attribute_counts"].get("correct_local_binding",0)}/{summary["local_attribute_denominator"]} correctly bound. Local attachment and constant elevation applicability remain separate assessments.',
                    f'HUH-9: tag {"observed" if equipment_check["tag_text_recovered"] else "not recovered"}; reviewed body {equipment_check["body_recovery"]}. Port assessment remains separate. Body omission does not justify accepting a port.',
                    'These bounded source inventories measure omissions beyond retained candidates. They do not establish whole-package recall, precision or physical quantities.',
                    'The source inventory was not used to tune this implementation. Exact mappings, tolerated and untolerated remainders, competing records and frozen hashes are included in the evidence ZIP.'
                ]
                for i,line in enumerate(lines):text(page,(40,125+i*91,1145,204+i*91),line,17)
                audit['independent_review'] = {'summary': summary, 'source_review_manifest_sha256': independent['source_review_manifest_sha256'],
                    'matching_manifest_sha256': hashlib.sha256((args.source_matching/'matching-manifest.json').read_bytes()).hexdigest()}
                audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'independent_source_first_summary'})
                for region, box in original['regions'].items():
                    rows = [r for r in independent['route_matches'] if r['region_id'] == region]
                    number = rows[0]['page_number']
                    page = new(f'Source-first region - drawing {number}', 'Green: covered within reviewed endpoint tolerance. Amber: partial / disputed. Red: missed source interval. Projected geometry only.')
                    colors = {'fully':GREEN, 'partly':AMBER, 'ambiguous':AMBER, 'missed':RED}
                    overlays = [([r['source_points_display'] for r in rows if r['status']==state],color,2) for state,color in colors.items()]
                    panel(page,source[number-1],box,(32,115,1158,646),overlays)
                    text(page,(35,669,1140,732),'The denominator is the frozen source inventory, not engine candidates. Inline coupling partitions and competing duplicate outlines remain explicit. No physical length is inferred from a missed interval.',17)
                    text(page,(35,747,1140,792),'Region: '+region+' | '+', '.join(f'{r["source_inventory_id"]}: {r["status"]}' for r in rows),11,GRAY)
                    audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'independent_source_first_region','region_id':region,'source_page_number':number})
                by_inventory = {r['source_inventory_id']: r for r in independent['route_matches']}
                for finding in independent['connection_matches']:
                    if finding['outcome'] != 'partly_degree_mismatch':
                        continue
                    members = [by_inventory[ref] for ref in finding['source_members']]
                    number = members[0]['page_number']
                    point = finding['source_point_display']
                    page = new(f'Independent branch finding - drawing {number}', 'An accepted two-way pass-through does not recover the visible three-way source branch.')
                    covered = {ref for r in finding['engine_candidates']
                        if r['state']=='accepted' and r['geometric_certificate_state']=='accepted'
                        for ref in r['covered_source_members']}
                    overlays = [([r['source_points_display'] for r in members if (r['source_inventory_id'] in covered)==accepted], color, 3)
                        for accepted,color in ((True,GREEN),(False,RED))]
                    panel(page,source[number-1],bounds([point],55),(32,115,1158,599),overlays,[(point,BLUE,'Branch under review')])
                    text(page,(35,626,1140,703),'Green: engine pass-through. Red: the independently inventoried third arm is missing. The original source and engine alternatives remain unchanged. This is not a fully recovered branch.',18)
                    text(page,(35,729,1140,788),finding['source_inventory_id']+' | '+', '.join(r['engine_ref'] for r in finding['engine_candidates'])+'\nIndependent AI finding; human adjudication and physical continuity remain open.',12,GRAY)
                    audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'independent_branch_degree_mismatch',
                        'source_inventory_ref':finding['source_inventory_id'],'source_page_number':number})
            page=new('Adversarial controls - keep uncertainty visible','Synthetic regression controls, not additional recovered drawing content. Each control is checked independently of the reviewed source inventory.')
            labels=[('Hidden branch','A native arm ending inside the corridor keeps the scope partial.'),('Unconnected crossing','A through-stroke remains a crossing; it does not create a route join.'),('Missing sidewall','Removing a source wall prevents completion. No gap is filled.')]
            for i,(title,description) in enumerate(labels):
                x=45+i*380;text(page,(x,132,x+345,170),title,19)
                for y in (270,310):
                    if i==2 and y==310:
                        page.draw_line((x+20,y),(x+120,y),color=BLUE,width=3);page.draw_line((x+185,y),(x+320,y),color=BLUE,width=3)
                    else:page.draw_line((x+20,y),(x+320,y),color=BLUE,width=3)
                if i<2:page.draw_line((x+165,290 if i==0 else 220),(x+165,370),color=RED,width=3)
                text(page,(x,417,x+345,540),description,17)
            text(page,(40,624,1140,735),'Regression tests also retain non-monotone/filled curves, missing grid labels or dashes, protected route ink, unexplained junction interiors, and unsearched middle corridors. These controls measure fail-closed behavior, not package recall.',17,GRAY)
            audit['acceptance_pages'].append({'page_number':len(pdf),'kind':'synthetic_adversarial_controls'})
            toc=pdf.get_toc()
            if not toc:
                toc=[[1,'Drawing overviews and local measurements',1]]
                toc.extend([2,f'Drawing {r["source_page_number"]}',r['audit_page_number']] for r in base_manifest['pages'])
                if base_manifest['closeups']:
                    toc.append([1,'Selected drawing close-ups',base_manifest['closeups'][0]['audit_page_number']])
                    toc.extend([2,f'Drawing {r["source_page_number"]} close-up',r['audit_page_number']] for r in base_manifest['closeups'])
            toc.append([1,'Interpretation acceptance',base_manifest['page_count']+1])
            for entry in audit['acceptance_pages'][1:]:
                label=entry['kind'].replace('_',' ').capitalize()
                if entry.get('source_page_number'):
                    label+=f' - drawing {entry["source_page_number"]}'
                toc.append([2,label,entry['page_number']])
            pdf.set_toc(toc)
            if args.output.exists():
                old=hashlib.sha256(args.output.read_bytes()).hexdigest()[:12]
                args.output.replace(args.output.with_name(args.output.stem+'-'+old+'.pdf'))
            pdf.save(args.output,garbage=3,deflate=True)
            audit['page_count']=len(pdf)
        audit['pdf_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
        if args.export:
            audit['complete_snapshot_export']=export_snapshot(store,scope,args.export,review_directories,args.implementation,args.source,comparison_files,base_manifest)
        args.output.with_suffix('.manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({k:v for k,v in audit.items() if k!='acceptance_pages'},indent=2))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('database','source','base-pdf','pilot-baseline','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('project','document','snapshot'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--source-review',type=Path)
    parser.add_argument('--source-matching',type=Path)
    parser.add_argument('--implementation',type=Path)
    parser.add_argument('--package-baseline',type=Path)
    parser.add_argument('--export',type=Path)
    render(parser.parse_args())


if __name__=='__main__':main()
