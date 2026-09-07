"""Replay frozen native-boundary certificates before projected run assembly."""

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import discover_native_boundary_connections
from src.drawing_engine.disciplines.mep.mep_native_bend_connections import discover_native_bends
from src.drawing_engine.disciplines.mep.mep_native_branch_connections import discover_native_branches


def replay_boundary_connections(payload, graph, bindings):
    if (payload['m3_payload_sha256'] != _sha256(graph)
            or payload['m35_payload_sha256'] != (bindings.get('m3_5_contract_ref') or {}).get('payload_sha256')):
        raise ValueError('boundary connections do not reference the frozen M3/M3.5 inputs')
    composites = {r['id']: r for r in bindings['outlined_route_composites']}
    records = payload['connections']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate native boundary connection IDs')
    accepted_ports = []
    for record in records:
        if record['state'] != 'accepted':
            continue
        sources = record['source_rows']
        if (len({r['source_primitive_ref'] for r in sources}) != len(sources)
                or _sha256(sorted(r['source_primitive_ref'] for r in sources)) != record['search']['all_source_refs_sha256']):
            raise ValueError('boundary replay is missing complete native query sources')
        expected_port_count = 3 if record['relation_type'] == 'projected_native_branch' else 2
        if len(record['composite_refs']) != expected_port_count or any(ref not in composites for ref in record['composite_refs']):
            raise ValueError('boundary replay references an unknown composite')

        class FrozenQuery:
            def query(self, box):
                if box != record['search']['bbox_display']:
                    return [], False, []
                return sources, record['search']['complete'], record['search']['region_refs']

            def with_initial_search_refs(self, rows):
                # Frozen rows already contain the exact serialized first-scan
                # memberships being replayed; hydration is a cold-index-only
                # compatibility operation.
                return rows

        method = {'projected_collinear_boundary_join': discover_native_boundary_connections,
                  'projected_native_bend': discover_native_bends,
                  'projected_native_branch': discover_native_branches}.get(record['relation_type'])
        if method is None:
            raise ValueError('unsupported boundary connection relation')
        replay = method(index=FrozenQuery(), graph=graph, page_ref=record['page_ref'],
            composites={'accepted_composites': [composites[ref] for ref in record['composite_refs']]})
        if record not in replay:
            raise ValueError('accepted native boundary connection does not replay')
        accepted_ports.extend(record['port_refs'])
    if len(accepted_ports) != len(set(accepted_ports)):
        raise ValueError('native boundary connection ports are not mutually unique')
    return records
