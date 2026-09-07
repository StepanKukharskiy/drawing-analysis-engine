"""Conservative projected corridor/body partitions, with no new M4 authority.

Native body identity is not inferred from a bounding box. Unexplained ink cuts
an analysis interval; the retained pieces require continuous original sidewalls.
Nonlinear source paths cannot be replaced by their endpoint chord.
"""
from copy import deepcopy
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE, boundary_coverage


def _clip(a,b,box):
    lo,hi=0.,1.
    for axis in (0,1):
        delta=b[axis]-a[axis]
        if abs(delta)<1e-12:
            if not box[axis]<=a[axis]<=box[axis+2]:return None
        else:
            p,q=sorted(((box[axis]-a[axis])/delta,(box[axis+2]-a[axis])/delta))
            lo,hi=max(lo,p),min(hi,q)
    if lo>hi:return None
    return [[a[i]+t*(b[i]-a[i]) for i in (0,1)] for t in (lo,hi)]


def partition_corridor(candidate, query, member_refs, style, *, version=2, authored_paths=None):
    """Return an exact parameter partition; body spans are explicitly unknown."""
    points=candidate['polyline_display'];sources=query['source_rows'];search=query['search']
    if version not in (1,2):raise ValueError('unsupported body partition version')
    result={'id':_stable_id('mep_route_body_partition',candidate['id'],_sha256(sources)),
        'candidate_ref':candidate['id'],'page_ref':query['page_ref'],'state':'abstained',
        'source_rows_sha256':_sha256(sources),'member_source_primitive_refs':sorted(member_refs),
        'retained_intervals':[],'body_intervals':[],'crossing_source_refs':[],
        'reasons':[],'system_identity_established':False,'physical_continuity_established':False,
        'installed_length':None,'purchase_length':None,'quantity_eligible':False}
    if (not search.get('complete') or search.get('source_rows_sha256')!=_sha256(sources)
        or search.get('all_source_refs_sha256')!=_sha256(sorted(r['source_primitive_ref'] for r in sources))):
        result['reasons']=['incomplete_or_stale_native_query'];return result
    if len(points)!=2 or not candidate.get('corridor_width_display_points'):
        result['reasons']=['no_two_sided_straight_corridor'];return result
    origin,end=points;length=math.dist(origin,end);half=candidate['corridor_width_display_points']/2
    if length<=TOLERANCE or half<=TOLERANCE:
        result['reasons']=['degenerate_corridor'];return result
    direction=[(end[i]-origin[i])/length for i in (0,1)];normal=[-direction[1],direction[0]]
    local=lambda p:[sum((p[i]-origin[i])*axis[i] for i in (0,1)) for axis in (direction,normal)]
    world=lambda x,y:[origin[i]+x*direction[i]+y*normal[i] for i in (0,1)]
    bounds=search['bbox_display']
    if not all(bounds[i]+TOLERANCE<p[i]<bounds[i+2]-TOLERANCE for x in (0,length) for y in (-half,half) for p in [world(x,y)] for i in (0,1)):
        result['reasons']=['corridor_outside_complete_query'];return result
    members=[r for r in sources if r['source_primitive_ref'] in member_refs]
    walls=[boundary_coverage(world(0,y),world(length,y),members,style) for y in (-half,half)]
    if any(not refs for refs in walls):
        result['reasons']=['original_sidewall_coverage_missing'];return result
    path_crossings=set()
    if version==2:
        from src.drawing_engine.disciplines.mep.mep_native_subpaths import reconstruct_native_subpaths,transverse_subpath_passages
        inventory=reconstruct_native_subpaths(sources,search,authored_paths)
        certificates=transverse_subpath_passages(inventory,sources,search,local,length,half,member_refs,TOLERANCE)
        for certificate in certificates:path_crossings.update(certificate['source_primitive_refs'])
        result.update(id=_stable_id('mep_route_body_partition',candidate['id'],_sha256(sources),'native_subpaths_v2',_sha256(authored_paths or {})),
            classifier_version=2,native_subpath_inventory=inventory,crossing_path_certificates=certificates)
    blocked=[]
    for row in sources:
        native=row['source_native_segment'];a,b=map(local,(row['points_display'][0],row['points_display'][-1]))
        if version==2 and row['source_primitive_ref'] in path_crossings:
            result['crossing_source_refs'].append(row['source_primitive_ref']);continue
        if native['kind']=='line':
            if abs(a[1]-b[1])<=TOLERANCE and abs(abs(a[1])-half)<=TOLERANCE:
                continue
            clipped=_clip(a,b,[0,-half+TOLERANCE,length,half-TOLERANCE])
            if clipped is None:continue
            # Crossing-only lines remain disconnected, never route joins.
            if ((a[1]<-half and b[1]>half) or (b[1]<-half and a[1]>half)):
                result['crossing_source_refs'].append(row['source_primitive_ref']);continue
            low,high=sorted(p[0] for p in clipped)
        else:
            box=row['search_bbox_display']
            corners=[local([x,y]) for x in (box[0],box[2]) for y in (box[1],box[3])]
            if max(p[1] for p in corners)<-half or min(p[1] for p in corners)>half:continue
            low,high=max(0,min(p[0] for p in corners)),min(length,max(p[0] for p in corners))
            if low>high:continue
        # Native paint width is an observed uncertainty margin, not a snap.
        margin=abs(float(native['style'].get('width') or 0))/2
        low,high=max(0,low-margin),min(length,high+margin)
        if high-low>TOLERANCE:blocked.append((low,high,[row['source_primitive_ref']]))
    merged=[]
    for low,high,refs in sorted(blocked):
        if merged and low<=merged[-1][1]+TOLERANCE:
            merged[-1][1]=max(merged[-1][1],high);merged[-1][2].extend(refs)
        else:merged.append([low,high,list(refs)])
    cuts=sorted({0.,length,*[v for a,b,_ in merged for v in (a,b)]})
    for low,high in zip(cuts,cuts[1:]):
        if high-low<=1e-12:continue
        overlaps=[r for a,b,refs in merged if a<(low+high)/2<b for r in refs]
        # Tiny remainders remain unknown rather than false isolated pipes.
        is_body=bool(overlaps) or high-low<max(TOLERANCE,2*float(style.get('width_display_points') or 0))
        row={'parameter_interval':[low,high],'polyline_display':[world(low,0),world(high,0)],
            'projected_path_display_points':high-low,'sidewall_source_refs':walls,
            'source_primitive_refs':sorted(set(overlaps)),
            'boundary_ports':[{'center_display':world(x,0),'sides_display':[world(x,-half),world(x,half)],
                'state':'unresolved_analysis_boundary','native_equipment_port_established':False,
                'source_sidewall_refs':walls} for x in (low,high)]}
        row['id']=_stable_id('mep_body_interval' if is_body else 'mep_pipe_interval',result['id'],low,high)
        row['reason']='unexplained_native_ink' if overlaps else 'short_unclosed_remainder' if is_body else 'continuous_native_sidewalls_without_interior_endpoint_ink'
        result['body_intervals' if is_body else 'retained_intervals'].append(row)
    result.update(state='partitioned',partition_length_display_points=length,
        partition_residual_display_points=length-sum(r['projected_path_display_points'] for k in ('body_intervals','retained_intervals') for r in result[k]))
    return result


def nonlinear_chord_defect(candidate, sources):
    """A measurable representation failure, not a claim about symbol type."""
    pts=candidate['polyline_display']
    if len(pts)!=2:return None
    a,b=pts;length=math.dist(a,b)
    if length<=TOLERANCE:return None
    offsets=[abs((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))/length
        for r in sources for p in r['points_display']]
    if not offsets or max(offsets)<=TOLERANCE:return None
    return {'state':'observed_representation_defect','reason':'nonlinear_native_path_replaced_by_unsupported_endpoint_chord',
        'maximum_chord_deviation_display_points':max(offsets),
        'source_primitive_refs':sorted(r['source_primitive_ref'] for r in sources),
        'native_paths':[{'source_primitive_ref':r['source_primitive_ref'],'points_display':r['points_display']} for r in sources],
        'boundary_ports':[{'center_display':p,'state':'native_path_endpoint_only','typed_port_established':False} for p in pts],
        'generic_element_type':'UNKNOWN','system_identity_established':False,'quantity_eligible':False}
