"""Shared runtime helper, preserved from its historical experiment owner."""
import json
from src.drawing_engine.project.project_packed_store import PackedProjectStore

def _artifact_json(store: PackedProjectStore, *, project_id: str, document_id: str, name: str) -> dict:
    return json.loads(b"".join(store.iter_artifact_bytes(
        project_id=project_id, document_id=document_id, name=name)))
