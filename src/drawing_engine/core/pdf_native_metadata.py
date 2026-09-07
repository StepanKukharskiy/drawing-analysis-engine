"""Native PDF rendering metadata, never CAD object or route identity.

Capture visibility before extracting drawings: extraction may report hidden
content too. OCG names are not unique identifiers; duplicate names stay explicit.
"""

from copy import deepcopy

import fitz


def inventory_page_metadata(page):
    document = page.parent
    groups = document.get_ocgs() if document.is_pdf else {}
    configuration = document.get_layer() if document.is_pdf else {}
    layers = []
    for xref, group in sorted(groups.items()):
        on, off = xref in configuration.get('on', []), xref in configuration.get('off', [])
        default = True if on and not off else False if off and not on else None
        if not on and not off and configuration.get('basestate') in {'ON', 'OFF'}:
            default = configuration['basestate'] == 'ON'
        layers.append({'xref': xref, 'name': group.get('name'),
            'default_configuration_visibility': default,
            'api_reported_current_visibility': group.get('on'),
            'intent': group.get('intent'), 'usage': group.get('usage')})
    forms = [{'xref': row[0], 'resource_name': row[1], 'invoker_xref': row[2],
              'bbox_pdf': list(row[3]), 'optional_content_xref': document.get_oc(row[0]) or None}
             for row in page.get_xobjects()] if document.is_pdf else []
    return {'schema_version': '0.1.0', 'page_number': page.number + 1,
        'method': 'pymupdf_native_pdf_metadata', 'extractor_version': fitz.VersionBind,
        'optional_content_groups': layers, 'default_configuration': configuration,
        'optional_content_configurations': list(document.get_layers()) if document.is_pdf else [],
        'form_xobjects': forms, 'form_to_primitive_membership': 'not_established',
        'marked_content_to_primitive_membership': 'not_exposed_by_this_extractor',
        'original_cad_layer_names_recovered': False, 'object_identity_established': False,
        'connection_established': False, 'quantity_eligible': False}


def drawing_metadata(drawing):
    """Preserve a raw layer name even for callers using ordinary get_drawings."""
    metadata = deepcopy(drawing.get('native_metadata', {}))
    if drawing.get('layer'):
        metadata['layer_name'] = drawing['layer']
        metadata.setdefault('membership_basis', 'api_reported_optional_content_name')
        metadata.setdefault('object_identity_established', False)
    return metadata


def get_native_drawings(page, *, extended_metadata=False):
    """Keep ordinary drawing ordinals while optionally retaining rendering groups.

    Clip and transparency-group records never become engineering primitives.
    Their nesting is rendering context, not a CAD block or equipment identity.
    """
    inventory = inventory_page_metadata(page)
    raw = page.get_drawings(extended=extended_metadata)
    if not extended_metadata:
        return raw, inventory
    paths, contexts, stack = [], [], []
    names = {}
    for layer in inventory['optional_content_groups']:
        names.setdefault(layer['name'], []).append(layer['xref'])
    for ordinal, drawing in enumerate(raw):
        level = drawing.get('level', 0)
        stack = [entry for entry in stack if entry['level'] < level]
        if drawing['type'] in {'clip', 'group'}:
            context = {'id': f'native_rendering_context.{ordinal}', 'level': level,
                'kind': drawing['type'],
                'bbox_pdf': list(drawing['rect']) if drawing.get('rect') is not None else None,
                'scissor_bbox_pdf': list(drawing['scissor']) if drawing.get('scissor') is not None else None,
                'parent_refs': [entry['id'] for entry in stack],
                'properties': {key: drawing[key] for key in
                    ('layer', 'isolated', 'knockout', 'blendmode', 'opacity', 'even_odd') if key in drawing}}
            contexts.append(context)
            stack.append(context)
            continue
        metadata = drawing_metadata(drawing)
        metadata.update({'paint_sequence_number': drawing.get('seqno'),
            'rendering_context_refs': [entry['id'] for entry in stack],
            'layer_xref_candidates': names.get(drawing.get('layer'), []),
            'graphics_state': {key: drawing[key] for key in
                ('lineCap', 'lineJoin', 'stroke_opacity', 'fill_opacity', 'closePath', 'even_odd') if key in drawing},
            'object_identity_established': False})
        paths.append({**drawing, 'native_metadata': metadata})
    inventory['rendering_contexts'] = contexts
    inventory['primitive_membership_scope'] = 'paint-path rendering context only'
    return paths, inventory


if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with fitz.open(args.source) as document:
        payload = {'schema_version': '0.1.0', 'layer': 'pdf_native_metadata_inventory',
            'document': {'source_pdf_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                         'filename': args.source.name},
            'pages': [inventory_page_metadata(page) for page in document], 'quantity_eligible': False}
    if args.output.exists() and json.loads(args.output.read_text()) != payload:
        raise ValueError('metadata output differs; choose a new output path to preserve history')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
