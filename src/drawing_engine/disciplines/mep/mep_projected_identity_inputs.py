"""Load replayable projected certificates and archive their exact native inputs.

Opaque gzip archives retain complete searches without creating millions of
SQLite geometry nodes. Portable replay follows artifact names, never local
paths. Loading or archiving evidence grants no additional engineering authority.
"""

import base64
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_equipment_identity import replay_equipment_2d_identity


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw), raw


def _path(value, base):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _archive_and_payload(path, expected_sha, document):
    payload, raw = _read(path)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha, 'native archive file hash mismatch')
    _require(raw.startswith(b'\x1f\x8b'), 'native evidence archive must preserve original gzip bytes')
    archive = {'schema_version': '0.1.0', 'layer': 'mep_opaque_native_replay_archive',
        'document': deepcopy(document), 'encoding': 'base64+gzip',
        'file_sha256': expected_sha, 'payload_sha256': _sha256(payload),
        'data_base64': base64.b64encode(raw).decode('ascii'), 'quantity_eligible': False}
    return archive, payload


def _archive(path, expected_sha, document):
    return _archive_and_payload(path, expected_sha, document)[0]


def _check_archive_source(payload, document):
    _require(payload.get('document') == document, 'native archive source document mismatch')
    _require(payload.get('source_pdf_sha256', document['source_pdf_sha256']) == document['source_pdf_sha256'],
             'native archive source PDF mismatch')


def _decode_archive(archive, document):
    _require(archive.get('layer') == 'mep_opaque_native_replay_archive'
        and archive.get('encoding') == 'base64+gzip' and archive.get('document') == document
        and archive.get('quantity_eligible') is False, 'native archive contract mismatch')
    raw = base64.b64decode(archive['data_base64'], validate=True)
    _require(hashlib.sha256(raw).hexdigest() == archive['file_sha256'], 'native archive bytes changed')
    payload = json.loads(gzip.decompress(raw))
    _require(_sha256(payload) == archive['payload_sha256'], 'native archive payload changed')
    _check_archive_source(payload, document)
    return payload


def _equipment_artifacts(identity_path, manifest_path, prefix, seen, *, prepared=None):
    identity_path, manifest_path = Path(identity_path), Path(manifest_path)
    _require(identity_path.resolve() not in seen, 'cyclic equipment calibration dependency')
    seen = seen | {identity_path.resolve()}
    payload, _ = _read(identity_path)
    manifest, _ = _read(manifest_path)
    _require(manifest.get('layer') == 'mep_equipment_identity_replay_inputs'
        and manifest.get('quantity_eligible') is False, 'equipment replay manifest contract mismatch')
    _require(_sha256(payload) == manifest.get('identity_payload_sha256'), 'equipment identity manifest hash mismatch')
    base, document = manifest_path.parent, payload['document']
    native = manifest['native_replay']
    native_name = 'projected-native-' + native['file_sha256']
    terms, _ = _read(_path(manifest['terminology']['path'], base))
    terms_sha = _sha256(terms)
    _require(terms_sha == manifest['terminology']['payload_sha256'], 'equipment terminology manifest hash mismatch')
    terms_name = 'projected-terminology-' + terms_sha
    archive, native_payload = _archive_and_payload(_path(native['path'], base), native['file_sha256'], document)
    if prepared is not None:
        # Owned only by this loader operation, never provided by portable callers.
        # Key by replay role, not equal archive bytes: preserve independent trees
        # and interpretation/calibration checks for the two evidence contexts.
        prepared[prefix + '-replay'] = (deepcopy(archive), native_payload)
    del native_payload
    artifacts = {prefix + '-identities': payload, terms_name: terms, native_name: archive}
    portable = {'identity_artifact': prefix + '-identities', 'native_archive_artifact': native_name,
                'terminology_artifact': terms_name, 'calibration_replay_artifact': None}
    calibration = manifest.get('calibration_payload')
    if calibration:
        _require(len(seen) == 1, 'nested equipment calibration is outside the versioned replay contract')
        calibration_path = _path(calibration['path'], base)
        earlier, _ = _read(calibration_path)
        _require(_sha256(earlier) == calibration['payload_sha256'], 'equipment calibration manifest hash mismatch')
        child_prefix = prefix + '-calibration'
        artifacts.update(_equipment_artifacts(calibration_path,
            calibration_path.with_name('equipment-2d-replay.json'), child_prefix, seen, prepared=prepared))
        portable['calibration_replay_artifact'] = child_prefix + '-replay'
    artifacts[prefix + '-replay'] = {**deepcopy(manifest), 'document': deepcopy(document), 'portable': portable}
    return artifacts


def _equipment_replay_inputs(artifacts, manifest_name, seen, *, prepared=None):
    _require(manifest_name not in seen, 'cyclic portable equipment calibration dependency')
    seen = seen | {manifest_name}
    manifest = artifacts[manifest_name]
    _require(manifest.get('layer') == 'mep_equipment_identity_replay_inputs'
        and manifest.get('quantity_eligible') is False, 'equipment replay manifest contract mismatch')
    names = manifest['portable']
    payload = artifacts[names['identity_artifact']]
    _require(_sha256(payload) == manifest['identity_payload_sha256'], 'portable equipment identity hash mismatch')
    _require(payload['document'] == manifest['document'], 'portable equipment document mismatch')
    terms = artifacts[names['terminology_artifact']]
    _require(_sha256(terms) == manifest['terminology']['payload_sha256'], 'portable terminology hash mismatch')
    archive = artifacts[names['native_archive_artifact']]
    _require(archive['file_sha256'] == manifest['native_replay']['file_sha256'] == payload['native_replay_sha256'],
             'equipment native archive binding mismatch')
    if prepared is None:
        native_inputs = _decode_archive(archive, payload['document'])
    else:
        original_archive, native_inputs = prepared[manifest_name]
        _require(archive == original_archive, 'native archive changed during preparation')
        _require(archive['document'] == payload['document'], 'native archive contract mismatch')
        _check_archive_source(native_inputs, payload['document'])
    kwargs = {'native_inputs': native_inputs, 'typography': manifest['typography']}
    earlier_name = names.get('calibration_replay_artifact')
    _require(bool(earlier_name) == bool(manifest.get('calibration_payload')), 'portable calibration dependency missing')
    if earlier_name:
        _require(len(seen) == 1, 'nested equipment calibration is outside the versioned replay contract')
        earlier, earlier_kwargs, earlier_terms = _equipment_replay_inputs(artifacts, earlier_name, seen, prepared=prepared)
        _require(_sha256(earlier) == manifest['calibration_payload']['payload_sha256'], 'portable calibration hash mismatch')
        kwargs.update(calibration_payload=earlier, calibration_native_inputs=earlier_kwargs['native_inputs'],
            calibration_typography=earlier_kwargs['typography'], calibration_terminology=earlier_terms)
    return payload, kwargs, terms


def _replay_equipment_identity_artifacts(artifacts, manifest_name, terminology, *, prepared=None):
    payload, kwargs, frozen_terms = _equipment_replay_inputs(artifacts, manifest_name, set(), prepared=prepared)
    if terminology is not None:
        _require(_sha256(terminology) == _sha256(frozen_terms), 'equipment identity differs from current M2')
    replay_equipment_2d_identity(payload, terminology=frozen_terms, **kwargs)
    return payload, kwargs


def replay_equipment_identity_artifacts(artifacts, *, manifest_name='equipment-2d-replay', terminology=None):
    """Verify archived SQLite evidence without reading any original local path.

    Return (identity payload, replay kwargs). The kwargs deliberately exclude
    terminology, which the M4 companion passes from its current snapshot.
    """
    return _replay_equipment_identity_artifacts(artifacts, manifest_name, terminology)


def load_equipment_identity_inputs(identity_path, *, replay_manifest_path=None, terminology=None):
    """Return (identity payload, replay kwargs excluding M2, portable artifacts)."""
    identity_path = Path(identity_path)
    manifest_path = Path(replay_manifest_path) if replay_manifest_path else identity_path.with_name('equipment-2d-replay.json')
    prepared = {}
    artifacts = _equipment_artifacts(identity_path, manifest_path, 'equipment-2d', set(), prepared=prepared)
    payload, kwargs = _replay_equipment_identity_artifacts(artifacts, 'equipment-2d-replay', terminology, prepared=prepared)
    return payload, kwargs, artifacts


def replay_fitting_identity_artifacts(artifacts, *, manifest_name='fitting-replay'):
    """Replay complete fitting searches against their original frozen M3/M3.5."""
    from src.drawing_engine.disciplines.mep.mep_fitting_hypotheses import replay_fitting_hypotheses
    manifest = artifacts[manifest_name]
    _require(manifest.get('layer') == 'mep_fitting_hypotheses_replay_inputs'
        and manifest.get('quantity_eligible') is False, 'fitting replay manifest contract mismatch')
    names = manifest['portable']; payload = artifacts[names['identity_artifact']]
    _require(_sha256(payload) == manifest['payload_sha256'], 'portable fitting payload hash mismatch')
    _require(manifest['document'] == payload['document'], 'portable fitting document mismatch')
    graph = artifacts[names['graph_artifact']]; composites = artifacts[names['composites_artifact']]
    _require(_sha256(graph) == payload['m3_payload_sha256'] and _sha256(composites) == payload['m35_payload_sha256'],
             'fitting frozen M3/M3.5 hash mismatch')
    archive_names = names['source_query_archives']
    _require(len(archive_names) == len(manifest['source_queries']), 'fitting native archive count mismatch')
    queries = []
    for name, entry in zip(archive_names, manifest['source_queries']):
        archive = artifacts[name]
        _require(archive['file_sha256'] == entry['sha256'], 'fitting native archive binding mismatch')
        frozen = _decode_archive(archive, payload['document'])
        _require(frozen['m3_payload_sha256'] == _sha256(graph) and frozen['m35_payload_sha256'] == _sha256(composites),
                 'fitting native query upstream hash mismatch')
        queries.extend(frozen['connections'])
    kwargs = {'source_queries': queries, 'composites': composites, 'graph': graph}
    replay_fitting_hypotheses(payload, **kwargs)
    return payload, kwargs


def load_fitting_identity_inputs(identity_path, *, replay_manifest_path=None):
    """Return (fitting payload, replay kwargs, portable artifacts), without PDF IO."""
    identity_path = Path(identity_path)
    manifest_path = Path(replay_manifest_path) if replay_manifest_path else identity_path.with_name(identity_path.name + '.manifest.json')
    payload, raw = _read(identity_path); manifest, _ = _read(manifest_path)
    _require(hashlib.sha256(raw).hexdigest() == manifest['artifact_sha256'], 'fitting artifact file hash mismatch')
    _require(_sha256(payload) == manifest['payload_sha256'], 'fitting artifact payload hash mismatch')
    run = _path(manifest['run'], manifest_path.parent)
    graph, _ = _read(run / 'route-observations.json')
    composites, _ = _read(run / 'outlined-route-composites.json')
    artifacts = {'fitting-hypotheses': payload, 'fitting-route-observations': graph,
                 'fitting-outlined-route-composites': composites}
    archive_names = []
    for entry in manifest['source_queries']:
        name = 'projected-native-' + entry['sha256']
        artifacts[name] = _archive(_path(entry['path'], manifest_path.parent), entry['sha256'], payload['document'])
        archive_names.append(name)
    artifacts['fitting-replay'] = {**deepcopy(manifest), 'schema_version': '0.1.0',
        'layer': 'mep_fitting_hypotheses_replay_inputs', 'document': deepcopy(payload['document']),
        'portable': {'identity_artifact': 'fitting-hypotheses', 'graph_artifact': 'fitting-route-observations',
            'composites_artifact': 'fitting-outlined-route-composites', 'source_query_archives': archive_names}}
    payload, kwargs = replay_fitting_identity_artifacts(artifacts)
    return payload, kwargs, artifacts
