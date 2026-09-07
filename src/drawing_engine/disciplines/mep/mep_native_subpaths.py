"""Exact native stroke composition, never pipe identity or physical connection.

Authored paint membership is retained even when a serialized microsegment chain
is nominated across paint records. Coordinate coincidence alone never joins it.
"""
from collections import defaultdict

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


def reconstruct_native_subpaths(sources, search, authored_paths=None):
    authored_paths=authored_paths or {}
    by_ref={r['source_primitive_ref']:r for r in sources}
    if len(by_ref)!=len(sources):raise ValueError('duplicate native primitive IDs')
    endpoints=defaultdict(list)
    for ref,row in by_ref.items():
        for end,p in enumerate((row['points_display'][0],row['points_display'][-1])):
            endpoints[tuple(p)].append((ref,end))
    transitions=[];forward={};backward={}
    for point,contacts in sorted(endpoints.items()):
        if len(contacts)<2:continue
        record={'id':_stable_id('mep_exact_native_vertex',point),'point_display':list(point),'incident_source_refs':sorted({r for r,e in contacts}),
                'state':'branch_or_coincident_endpoint_alternatives' if len(contacts)>2 else 'unresolved_endpoint_contact',
                'physical_connection_established':False}
        if len(contacts)==2 and {e for r,e in contacts}=={0,1}:
            left=next(r for r,e in contacts if e==1);right=next(r for r,e in contacts if e==0)
            a,b=by_ref[left],by_ref[right];na,nb=a['source_native_segment'],b['source_native_segment']
            pa,pb=na.get('drawing_ref'),nb.get('drawing_ref')
            same_style=na['style']==nb['style'] and na['style'].get('fill') is None
            same_path=(pa is not None and pa==pb and
                ((na.get('item_index')==nb.get('item_index') and nb.get('part_index',-2)==na.get('part_index',-4)+1)
                 or (nb.get('item_index',-2)==na.get('item_index',-4)+1 and nb.get('part_index')==0)))
            serial=False
            if pa in authored_paths and pb in authored_paths:
                ca,cb=authored_paths[pa],authored_paths[pb]
                serial=(cb['drawing_ordinal']==ca['drawing_ordinal']+1 and
                        ca['source_segment_count']==cb['source_segment_count']==1)
            u=[a['points_display'][-1][i]-a['points_display'][0][i] for i in (0,1)]
            v=[b['points_display'][-1][i]-b['points_display'][0][i] for i in (0,1)]
            smooth=sum(x*y for x,y in zip(u,v))>0
            if left!=right and same_style and na['kind']==nb['kind']=='line' and (same_path or serial and smooth):
                record.update(state='internal_native_path_vertex' if same_path else 'serialized_fragment_vertex_candidate',
                    source_refs_in_order=[left,right],authored_path_refs=[pa,pb],exact_seam_residual=0.0)
                forward[left]=right;backward[right]=left
        transitions.append(record)
    transition_by_pair={tuple(r['source_refs_in_order']):r for r in transitions if 'source_refs_in_order' in r}
    remaining=set(by_ref);paths=[];singletons=[];singleton_ends=[]
    seeds=sorted(ref for ref in remaining if ref not in backward)+sorted(remaining)
    for seed in seeds:
        if seed not in remaining:continue
        refs=[];ref=seed
        while ref in remaining:
            remaining.remove(ref);refs.append(ref);ref=forward.get(ref)
        joins=[transition_by_pair[(a,b)] for a,b in zip(refs,refs[1:])]
        rows=[by_ref[r] for r in refs];points=[rows[0]['points_display'][0]]+[r['points_display'][-1] for r in rows]
        closed=points[0]==points[-1]
        box=search['bbox_display']
        terminal=[]
        for point in (points[0],points[-1]):
            inside=all(box[i]<point[i]<box[i+2] for i in (0,1))
            contacts=endpoints[tuple(point)]
            terminal.append({'point_display':point,'state':'closed_path' if closed else
                'query_boundary_or_outside_search' if not inside else
                'native_path_endpoint_only' if len(contacts)==1 else 'unresolved_endpoint_contact',
                'endpoint_transition_ref':_stable_id('mep_exact_native_vertex',point) if len(contacts)>1 else None,
                'physical_terminal_established':False})
        if len(refs)==1:
            # Provenance/style/points already live in the hash-bound native query.
            # Avoid expanding hundreds of thousands of one-item paint records twice.
            singletons.append(refs[0]);singleton_ends.append([e['state'] for e in terminal]);continue
        paths.append({'id':_stable_id('mep_native_subpath',refs),'source_primitive_refs':refs,
            'authored_path_refs':[r['source_native_segment'].get('drawing_ref') for r in rows],
            'native_style':rows[0]['source_native_segment']['style'],'joins':joins,'endpoints':terminal,
            'closed':closed,'all_line':all(r['source_native_segment']['kind']=='line' for r in rows),
            'composition':'serialized_paint_fragment_candidate' if any(j['state']=='serialized_fragment_vertex_candidate' for j in joins)
                else 'authored_paint_item_subpath',
            'route_identity_established':False})
    return {'subpaths':paths,'singleton_source_refs':singletons,'singleton_endpoint_states':singleton_ends,
        'singleton_provenance':'authored membership, native style and coordinates remain in the exact source query row',
        'endpoint_transitions':transitions,
        'source_primitive_count':len(sources),'accounted_source_primitive_count':len(singletons)+sum(len(p['source_primitive_refs']) for p in paths),
        'source_rows_sha256':_sha256(sources),'authored_path_context_sha256':_sha256(authored_paths),
        'physical_connection_established':False}


def transverse_subpath_passages(inventory, sources, search, local, length, half, protected_refs, margin):
    """Only explain a bounded monotone through-passage, not the entire path.

Closed symbols, reversals, caps on walls, query-edge joins, and coordinate-only
contacts do not qualify. Every unexcused stroke remains an independent blocker.
"""
    by_ref={r['source_primitive_ref']:r for r in sources};certificates=[]
    box=search['bbox_display']
    for path in inventory['subpaths']:
        refs=path['source_primitive_refs']
        if len(refs)<2 or path['closed'] or not path['all_line'] or set(refs)&protected_refs:continue
        # Two unrelated paint records cannot establish a serialized curve chain.
        if path['composition']=='serialized_paint_fragment_candidate' and len(refs)<3:continue
        rows=[by_ref[r] for r in refs]
        points=[rows[0]['points_display'][0]]+[r['points_display'][-1] for r in rows]
        projected=[local(p) for p in points]
        i=0
        while i<len(refs):
            side=-1 if projected[i][1]<-half-margin else 1 if projected[i][1]>half+margin else 0
            if not side:i+=1;continue
            j=i+1
            while j<len(points) and -half-margin<=projected[j][1]<=half+margin:j+=1
            if j>=len(points):break
            crossed=side*projected[j][1]<-half-margin
            window=projected[i:j+1]
            monotone=all(side*(b[1]-a[1])<0 for a,b in zip(window,window[1:]))
            inside=all(box[k]+margin<p[k]<box[k+2]-margin for p in points[i+1:j] for k in (0,1))
            # The whole local line hull is bounded away from scope exits.
            axial=all(margin<p[0]<length-margin for p in window)
            if crossed and monotone and inside and axial and j-i>1:
                window_refs=refs[i:j]
                certificates.append({'method':'exact_native_subpath_transverse_v2','subpath_ref':path['id'],
                    'source_primitive_refs':window_refs,'supporting_chain_source_refs':refs,
                    'authored_path_refs':path['authored_path_refs'][i:j],
                    'joins':path['joins'][i:j-1],'points_local':window,
                    'internal_vertices_are_not_pipe_endpoints':True,
                    'crossing_connection_established':False,'role_identity_established':False,
                    'all_other_strokes_remain_competitors':True})
            i=max(i+1,j)
    return certificates
