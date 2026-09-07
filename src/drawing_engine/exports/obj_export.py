"""Export frozen solid meshes without solving geometry or inventing placement."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from src.drawing_engine.exports.detail_dxf_export import _filename_mark


def _validate_mesh(mesh, vertex_key):
    """Check serialization safety and closed oriented topology; never repair ink."""
    if mesh.get('validation', {}).get('watertight') is not True:
        raise ValueError('stored_watertight_validation_required')
    vertices, faces = mesh.get(vertex_key), mesh.get('triangles')
    if not isinstance(vertices, list) or not vertices or any(
            not isinstance(p, list) or len(p) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in vertices):
        raise ValueError('finite_xyz_vertices_required')
    if not isinstance(faces, list) or not faces or any(
            not isinstance(f, list) or len(f) != 3 or any(
                type(i) is not int or not 0 <= i < len(vertices) for i in f)
            or len(set(f)) != 3 for f in faces):
        raise ValueError('valid_triangle_indices_required')
    if len({tuple(sorted(f)) for f in faces}) != len(faces):
        raise ValueError('duplicate_triangles')
    edges = Counter()
    volumes = []
    origin = vertices[0]
    for face in faces:
        a, b, c = (vertices[i] for i in face)
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        if not all(math.isfinite(x) for x in cross) or not any(cross):
            raise ValueError('degenerate_triangle')
        volumes.append(sum((a[i]-origin[i])*cross[i] for i in range(3)))
        edges.update(zip(face, face[1:] + face[:1]))
    if any(n != 1 or edges[b, a] != 1 for (a, b), n in edges.items()):
        raise ValueError('closed_consistently_oriented_mesh_required')
    if {i for face in faces for i in face} != set(range(len(vertices))):
        raise ValueError('unused_vertices')
    if not all(math.isfinite(v) for v in volumes) or math.fsum(volumes) <= 0:
        raise ValueError('positive_oriented_volume_required')
    return vertices, faces


def _detail_candidates(assembly):
    units = assembly.get('linear_unit_evidence', {})
    native_mm = units.get('state') == 'direct' and units.get('unit') == 'mm'
    for index, part in enumerate(assembly.get('child_parts', [])):
        mesh = part.get('solid') or {}
        key = 'vertices_xyz_mm' if native_mm else 'vertices_xyz_drawing_units'
        reason = None
        if part.get('kind') != 'plate':
            reason = 'unsupported_part_kind_without_solid_adapter'
        elif not mesh:
            reason = 'no_resolved_solid_mesh'
        elif part.get('state') != 'derived' or part.get('reprojection', {}).get('passed') is not True:
            reason = 'resolved_part_and_passed_reprojection_required'
        elif not native_mm and (units.get('observations') or units.get('unit') is not None):
            reason = 'unsupported_or_conflicting_native_units'
        elif key not in mesh or ('vertices_xyz_mm' in mesh and 'vertices_xyz_drawing_units' in mesh):
            reason = 'mesh_unit_evidence_mismatch'
        yield {'ref': part.get('id'), 'mark': part.get('mark'),
               'name': f"{_filename_mark(assembly.get('mark'), 'detail')}-part-{_filename_mark(part.get('mark'), str(index+1))}",
               'source_pointer': f'/child_parts/{index}/solid', 'mesh': mesh, 'vertex_key': key,
               'units': 'mm' if native_mm else None,
               'coordinate_basis': mesh.get('coordinate_basis'),
               'reason': reason}


def _structural_candidates(graph):
    for index, page in enumerate(graph.get('pages', [])):
        if type(page.get('page')) is not int or page['page'] < 1:
            raise ValueError('positive source page number required for OBJ export')
        preview = page.get('solid_preview') or {}
        mesh = preview.get('mesh') or {}
        pointer = f'/pages/{index}/solid_preview/mesh'
        base = {'ref': f"page:{page.get('page')}", 'mark': None,
                'name': f"page-{page.get('page', index+1)}-solid", 'source_pointer': pointer,
                'mesh': mesh, 'vertex_key': 'vertices_xyz_mm', 'units': 'mm',
                'coordinate_basis': 'frozen_relative_xyz_mm', 'reason': None}
        if (page.get('specialised_solver', {}).get('status') != 'resolved'
                or not page.get('solid_hypotheses') or not mesh):
            yield {**base, 'reason': 'no_resolved_structural_solid'}
        elif mesh.get('components'):
            # These meshes include display-only offsets. Until a component
            # adapter replays their ranges, never present them as an assembly.
            yield {**base, 'reason': 'presentation_only_component_collection_not_supported'}
        else:
            yield base


def export_project_obj(project, output, *, artifact=None):
    from src.drawing_engine.exports.estimation_project_export import artifact_record
    output = Path(output)
    default = 'assembly' if project.manifest.get('mode') == 'native_detail_assembly' else 'engineering_graph'
    artifact = artifact or default
    if artifact not in {'assembly', 'engineering_graph'} or artifact not in project.manifest['records']:
        raise ValueError('OBJ export requires a frozen detail assembly or structural engineering_graph')
    # Validate the original source and read only its frozen geometry record.
    project.source()
    payload = project.load(artifact)
    candidates = list(_detail_candidates(payload) if artifact == 'assembly' else _structural_candidates(payload))
    report = {'schema_version': 'solid_obj_export.v1', 'format': 'obj',
              'source_snapshot': project.scope, 'source_sha256': project.manifest['source_sha256'],
              'source_artifact': artifact, 'source_artifact_sha256': project.manifest['records'][artifact]['sha256'],
              'scope': 'separate_frozen_solid_definitions_in_relative_frames',
              'assembly_placement_established': False, 'absolute_orientation_established': False,
              'fabrication_release': False, 'approved': None, 'quantity_authority_granted': False,
              'excluded_channels': ['reinforcement_paths', 'preview_overlays', 'declared_values'],
              'parts': [], 'artifacts': {}}
    output.mkdir(parents=True, exist_ok=False)
    used = set()
    for index, candidate in enumerate(candidates):
        record = {k: v for k, v in candidate.items() if k not in {'name', 'mesh', 'vertex_key', 'reason'}}
        record.update(state='abstained', path=None, reasons=[])
        report['parts'].append(record)
        try:
            if candidate['reason']:
                raise ValueError(candidate['reason'])
            vertices, faces = _validate_mesh(candidate['mesh'], candidate['vertex_key'])
        except ValueError as error:
            record['reasons'].append(str(error))
            continue
        suffix = 'mm' if candidate['units'] == 'mm' else 'unitless'
        stem = candidate['name']
        name = f'{stem}.{suffix}.obj'
        count = 1
        while name.casefold() in used:
            count += 1
            name = f'{stem}-{count}.{suffix}.obj'
        used.add(name.casefold())
        path = output / name
        with path.open('x', encoding='ascii', newline='\n') as stream:
            stream.write('# Drawing engine frozen solid; no fabrication approval\n')
            stream.write(f'# units: {suffix}; relative coordinates; no assembly placement\n')
            stream.write('# source: ' + json.dumps({'sha256': report['source_sha256'],
                'artifact': artifact, 'pointer': candidate['source_pointer'], 'mark': candidate['mark']}, ensure_ascii=True) + '\n')
            stream.write(f'o solid_{index+1}\ns off\n')
            for point in vertices:
                stream.write('v ' + ' '.join(format(v, '.17g') for v in point) + '\n')
            for face in faces:
                stream.write('f ' + ' '.join(str(i+1) for i in face) + '\n')
        record.update(state='exported', path=name, vertex_count=len(vertices), triangle_count=len(faces))
        report['artifacts'][f'solid_{index+1}'] = artifact_record(output, path)
    count = len(report['artifacts'])
    report['state'] = 'exported' if count and count == len(candidates) else 'partial' if count else 'abstained'
    report['reasons'] = ['no_supported_solid_records'] if not candidates else []
    (output / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    return output / 'result.json'
