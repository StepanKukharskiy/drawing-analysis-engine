#!/usr/bin/env python3
"""Recover bounded two-sided bends from exhaustive native microsegments.

Existing supported corridor ends nominate search extents, not geometry or
identity. Native chains and a complete two-boundary replay own bend geometry.
Endpoint gaps remain explicit; this adapter never publishes an M4 binding.
"""
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import argparse
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read, write
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256, _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack
from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE, _dot, _sub
from src.drawing_engine.disciplines.mep.mep_native_bend_connections import trace_bend_boundaries
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible, _samples
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import certify_junction_interior
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.core.vector_topology import point_distance_to_segment


def intersect(a,b):
    return all(a[i]<=b[i+2] and b[i]<=a[i+2] for i in (0,1))


def nominated_windows(recovery,styles):
    ports=[]
    for r in recovery['outlined_corridor_components']:
        if (r['state'] not in {'identified_mep_route','supported_unidentified_mep_candidate'}
                or not r['certificate_flags'].get('accepted_drawing_view_ownership')
                or len(r['polyline_display'])!=2):continue
        for end in (0,1):
            center=r['polyline_display'][end]; delta=_sub(center,r['polyline_display'][1-end]);length=math.hypot(*delta)
            if not length:continue
            ports.append({'candidate_ref':r['id'],'center':center,'outward':[v/length for v in delta],
                'width':r['corridor_width_display_points'],'style':_normalise_style(styles[r['style_id']])})
    windows=[]
    for a,b in combinations(ports,2):
        w=max(a['width'],b['width'])
        if (a['candidate_ref']==b['candidate_ref'] or math.dist(a['center'],b['center'])>4*w
                or abs(_dot(a['outward'],b['outward']))>.001
                or abs(a['width']-b['width'])>max(.15,.05*w)
                or _dot(_sub(b['center'],a['center']),a['outward'])<=0
                or _dot(_sub(a['center'],b['center']),b['outward'])<=0
                or not _style_compatible(a['style'],b['style'])):continue
        centers=[a['center'],b['center']]
        box=[min(p[i] for p in centers)-2*w for i in (0,1)]+[max(p[i] for p in centers)+2*w for i in (0,1)]
        windows.append({'bbox_display':box,'nominating_ports':[a,b]})
    return windows


def quarter_chains(rows,style):
    """Native endpoint chains; ambiguous degree, circles and non-turns stop."""
    edges={};cells=defaultdict(list)
    def key(point):
        cell=tuple(math.floor(v/TOLERANCE) for v in point)
        near={p for x in (-1,0,1) for y in (-1,0,1) for p in cells[(cell[0]+x,cell[1]+y)] if math.dist(point,p)<=TOLERANCE}
        if len(near)>1:return None
        p=next(iter(near)) if near else tuple(point)
        if not near:cells[cell].append(p)
        return p
    for r in rows:
        n=r['source_native_segment']
        if n['kind']!='line' or not _style_compatible(style,_normalise_style(n['style'])):continue
        a,b=[key(p) for p in (r['points_display'][0],r['points_display'][-1])]
        if a is None or b is None:return []
        if a==b:continue
        edges.setdefault(tuple(sorted((a,b))),set()).add(r['source_primitive_ref'])
    adjacency=defaultdict(set)
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    stops=set()
    for point,neighbors in adjacency.items():
        if len(neighbors)!=2:stops.add(point);continue
        a,b=[_sub(n,point) for n in neighbors]
        if _dot(a,b)/(math.hypot(*a)*math.hypot(*b))>-math.sqrt(.5):stops.add(point)
    visited=set();out=[]
    for start in sorted(adjacency):
        if start not in stops:continue
        for other in sorted(adjacency[start]):
            e=tuple(sorted((start,other)))
            if e in visited:continue
            points=[start,other];refs=set(edges[e]);visited.add(e)
            while points[-1] not in stops:
                nxt=next(iter(adjacency[points[-1]]-{points[-2]}));e=tuple(sorted((points[-1],nxt)))
                if e in visited:break
                visited.add(e);refs.update(edges[e]);points.append(nxt)
            if len(points)<5 or points[0]==points[-1]:continue
            directions=[_sub(b,a) for a,b in zip(points,points[1:])]
            turns=[math.atan2(a[0]*b[1]-a[1]*b[0],_dot(a,b)) for a,b in zip(directions,directions[1:])]
            if (abs(abs(sum(turns))-math.pi/2)>.015 or
                    any(t*sum(turns)<-1e-6 for t in turns) or max(map(abs,turns))>math.pi/4):continue
            out.append({'points_display':[list(p) for p in points],'source_primitive_refs':sorted(refs)})
    return out


def pair_chains(chains,style,page_ref):
    pairs=[]
    for a,b0 in combinations(chains,2):
        for reverse in (False,True):
            b={**b0,'points_display':list(reversed(b0['points_display'])) if reverse else b0['points_display']}
            aa,bb=a['points_display'],b['points_display'];widths=[math.dist(aa[i],bb[i]) for i in (0,-1)]
            w=sum(widths)/2
            if w<=TOLERANCE or abs(widths[0]-widths[1])>max(.15,.05*w):continue
            ports=[]
            for end,inside in ((0,1),(-1,-2)):
                tangent=_sub(aa[inside],aa[end]);length=math.hypot(*tangent);tangent=[v/length for v in tangent]
                if abs(_dot(_sub(bb[end],aa[end]),tangent))>TOLERANCE:break
                ports.append({'center':[(x+y)/2 for x,y in zip(aa[end],bb[end])],
                    'sides':[aa[end],bb[end]],'outward':tangent,'width':w,'style':style,'page_ref':page_ref})
            if len(ports)!=2 or abs(_dot(ports[0]['outward'],ports[1]['outward']))>.001:continue
            samples=[_samples(p['points_display'],17) for p in (a,b)]
            if any(abs(math.dist(x,y)-w)>max(.15,.08*w) for x,y in zip(*samples)):continue
            pairs.append((ports,[a,b]))
    incidence=Counter(tuple(p['source_primitive_refs']) for _,paths in pairs for p in paths)
    return [(ports,paths) for ports,paths in pairs
            if all(incidence[tuple(p['source_primitive_refs'])]==1 for p in paths)]


def recover_window(window,rows,page_ref,complete):
    style=window['nominating_ports'][0]['style'];chains=quarter_chains(rows,style);output=[]
    for ports,proposal in pair_chains(chains,style,page_ref):
        width=ports[0]['width']
        if abs(width-window['nominating_ports'][0]['width'])>max(.15,.05*width):continue
        paths,reasons=trace_bend_boundaries(ports,rows)
        if not complete:reasons.append('incomplete_native_inventory_or_view_ownership')
        if not paths:continue
        refs=sorted({r for p in paths for r in p['source_primitive_refs']})
        identifier=_stable_id('mep_inventory_bend_candidate',page_ref,refs)
        connection={'id':identifier,'page_ref':page_ref,'state':'accepted' if not reasons else 'abstained','relation_type':'projected_native_bend','ports':ports,'boundary_paths':paths,
            'source_rows':rows,'source_primitive_refs':refs,'search':{'bbox_display':window['bbox_display'],
            'complete':complete,'all_source_refs_sha256':_sha256(sorted(r['source_primitive_ref'] for r in rows))}}
        full_connection=connection
        interior=certify_junction_interior(connection)
        full_interior=interior
        terminal_boundary_fragments=[]
        # Endpoint glyphs may enter the tangent stubs. Bound the recovered
        # interval at the next authored vertices, retaining the omitted stubs
        # and the original failed interface search. These are analysis cuts,
        # never typed connector ports or evidence of continuity.
        if interior['state']!='accepted' and all(len(p['points_display'])>=5 for p in paths):
            core=[{**p,'points_display':p['points_display'][1:-1]} for p in paths]
            core_ports=[]
            for end,inside in ((0,1),(-1,-2)):
                sides=[p['points_display'][end] for p in core];across=_sub(sides[1],sides[0]);w=math.hypot(*across)
                normal=[-across[1]/w,across[0]/w]
                tangent=_sub(core[0]['points_display'][inside],sides[0])
                if _dot(tangent,normal)<0:normal=[-v for v in normal]
                core_ports.append({'center':[(a+b)/2 for a,b in zip(*sides)],'sides':sides,
                    'outward':normal,'width':w,'style':style,'page_ref':page_ref})
            traced,core_reasons=trace_bend_boundaries(core_ports,rows)
            if not core_reasons:
                attempt={**connection,'ports':core_ports,'boundary_paths':traced}
                checked=certify_junction_interior(attempt)
                if checked['state']=='accepted':
                    for path in paths:
                        p=path['points_display']
                        for points in (p[:2],p[-2:]):
                            stub_refs=[r['source_primitive_ref'] for r in rows if r['source_primitive_ref'] in path['source_primitive_refs']
                                and all(point_distance_to_segment(v,r['points_display'][0],r['points_display'][-1])<=TOLERANCE for v in points)]
                            terminal_boundary_fragments.append({'points_display':points,'state':'unresolved_interface_stub',
                                'source_primitive_refs':stub_refs})
                    paths=traced;ports=core_ports;connection=attempt;interior=checked
        if interior['state']!='accepted':reasons.extend(interior['reasons'])
        sampled=[_samples(p['points_display'],33) for p in paths]
        line=[[(a+b)/2 for a,b in zip(x,y)] for x,y in zip(*sampled)]
        native_styles={_sha256(_normalise_style(r['source_native_segment']['style'])):
            _normalise_style(r['source_native_segment']['style']) for r in rows if r['source_primitive_ref'] in refs}
        output.append({'id':identifier,'record_type':'mep_native_microsegment_bend_candidate','page_ref':page_ref,
            'channel':'native_microsegmented_bend','candidate_systems':[],
            'state':'supported_unidentified_mep_candidate' if not reasons else 'unresolved_bend_geometry',
            'geometry_state':'accepted' if not reasons else 'abstained','reasons':sorted(set(reasons)),
            'polyline_display':line,'projected_path_display_points':sum(math.dist(a,b) for a,b in zip(line,line[1:])),
            'corridor_width_display_points':width,'source_segment_refs':refs,'boundary_paths':paths,
            'core_source_segment_refs':sorted({ref for p in paths for ref in p['source_primitive_refs']}),
            'ports':ports,'nominating_candidate_refs':sorted(p['candidate_ref'] for p in window['nominating_ports']),
            'native_style':next(iter(native_styles.values())) if len(native_styles)==1 else None,
            'native_styles':list(native_styles.values()),'system':None,'system_identity_accepted':False,'physical_continuity_established':False,
            'installed_length':None,'purchase_length':None,'quantity_eligible':False,
            'boundary_interfaces':[{'center':p['center'],'side_points':p['sides'],'state':'unresolved_physical_interface',
                'source_port_plane_only':True} for p in ports],
            'endpoint_warning_points_display':[p['center'] for p in ports],
            'junction_interior_certificate':interior,'full_interface_certificate':full_interior,
            'complete_bend_boundary_paths':full_connection['boundary_paths'],
            'terminal_boundary_fragments':terminal_boundary_fragments,
            'analysis_cuts_are_not_physical_ports':bool(terminal_boundary_fragments),
            'native_search':connection['search'],
            'source_rows':rows})
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--denominator',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifest=read(args.denominator);recovery=read(args.recovery);dm=manifest['native_descriptor_pack'];pack=NativeDescriptorPack(dm['path'],dm)
    if not pack.verify_hashes() or _file_sha256(Path(manifest['source']['pdf_path']))!=manifest['source']['pdf_sha256']:raise ValueError('native source identity changed')
    windows=nominated_windows(recovery,dm['styles']);print('nominated windows',len(windows),flush=True)
    selections=[[] for _ in windows];drawings=set();ordinal_by_ref={};cells=defaultdict(set)
    def box_cells(box):
        return ((x,y) for x in range(math.floor(box[0]/64),math.floor(box[2]/64)+1)
                for y in range(math.floor(box[1]/64),math.floor(box[3]/64)+1))
    for i,w in enumerate(windows):
        for cell in box_cells(w['bbox_display']):cells[cell].add(i)
    # One complete compact descriptor scan, without two million expanded rows.
    for ordinal,raw in enumerate(pack.raw_records()):
        bbox=raw[27:31]
        candidates={i for cell in box_cells(bbox) for i in cells.get(cell,())}
        hits=[i for i in candidates if intersect(bbox,windows[i]['bbox_display'])]
        if not hits:continue
        d=pack._decode_descriptor(raw);d['descriptor_ordinal']=ordinal;row=pack.row(d);ordinal_by_ref[row['source_primitive_ref']]=ordinal
        drawings.add(d['drawing_ordinal'])
        for i in hits:selections[i].append(row)
    path_manifest=manifest['native_authored_path_pack'];paths=NativePathPack(path_manifest['path'],path_manifest)
    if not paths.verify_hash():raise ValueError('native path pack changed')
    source_paths={p['drawing_ordinal']:p for p in paths.records() if p['drawing_ordinal'] in drawings}
    ownership={drawing:p['region_role'] for drawing,p in source_paths.items()}
    write(args.output/'native-window-capture.json.gz',{'windows':windows,'selections':selections,
        'ownership':ownership,'ordinal_by_ref':ordinal_by_ref,'source_pdf_sha256':manifest['source']['pdf_sha256']})
    result={};diagnostics=[]
    allowed={'main_plan_view','section_detail_riser_view'}
    complete_inventory=not dm.get('unsupported_native_item_kinds') and manifest['acceptance_gate']['unaccounted_segment_count']==0
    for w,rows in zip(windows,selections):
        owned=all(ownership.get(int(r['source_native_segment']['drawing_ref'][8:-1])) in allowed for r in rows)
        found=recover_window(w,rows,recovery['page_ref'],complete_inventory and owned)
        diagnostics.append({'bbox_display':w['bbox_display'],'nominating_candidate_refs':[p['candidate_ref'] for p in w['nominating_ports']],
            'native_row_count':len(rows),'accepted_view_only':owned,'recovered_count':len(found)})
        for r in found:
            r['source_descriptor_ordinals']=[ordinal_by_ref[ref] for ref in r['source_segment_refs']]
            ordinals=sorted({source_paths[int(ref.split(']')[0][8:])]['path_ordinal'] for ref in r['source_segment_refs']})
            intervals=[]
            for ordinal in ordinals:
                if intervals and ordinal==intervals[-1][1]+1:intervals[-1][1]=ordinal
                else:intervals.append([ordinal,ordinal])
            r['source_path_ordinal_intervals']=intervals
            result[r['id']]=r
    records=list(result.values());summary={'candidate_count':len(records),'states':dict(Counter(r['geometry_state'] for r in records)),
        'nominated_window_count':len(windows),'source_primitive_count':len(pack),'complete_sheet_route_coverage':False,
        'accepted_system_identities':0,'installed_length':None,'purchase_length':None}
    write(args.output/'recovered-bends.json.gz',{'inputs':{'denominator_sha256':_file_sha256(args.denominator),'recovery_sha256':_file_sha256(args.recovery)},
        'records':records,'windows':diagnostics,'summary':summary})
    write(args.output/'summary.json',summary);print(summary)


if __name__=='__main__':main()
