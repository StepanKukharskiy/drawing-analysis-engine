"""Original archive-then-decode equipment loader, for differential checks only."""
from pathlib import Path

from src.drawing_engine.disciplines.mep.mep_projected_identity_inputs import _equipment_artifacts, replay_equipment_identity_artifacts


def load_equipment_inputs_reference(identity_path, *, replay_manifest_path=None, terminology=None):
    identity_path = Path(identity_path)
    manifest_path = (Path(replay_manifest_path) if replay_manifest_path else
                     identity_path.with_name('equipment-2d-replay.json'))
    artifacts = _equipment_artifacts(identity_path, manifest_path, 'equipment-2d', set())
    payload, kwargs = replay_equipment_identity_artifacts(artifacts, terminology=terminology)
    return payload, kwargs, artifacts
