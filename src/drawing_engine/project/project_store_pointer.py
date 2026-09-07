"""Atomic package pointer with an explicit one-step rollback target."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile


POINTER_VERSION = 1


def _database_name(directory, value):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError('package database must be a filename in the pointer directory')
    path = Path(directory) / value
    if not path.is_file():
        raise ValueError('package database does not exist: ' + value)
    return path


def validate_database(path, *, project_id, document_id):
    path = Path(path)
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version not in (1, 2, 3):
            raise ValueError('unsupported package database schema')
        check = [row[0] for row in connection.execute('PRAGMA quick_check')]
        if check != ['ok']:
            raise ValueError('package database quick_check failed')
        active = connection.execute(
            'SELECT 1 FROM active_documents WHERE project_id=? AND document_id=?',
            (project_id, document_id)).fetchone()
        if active is None:
            raise ValueError('package database lacks the active project/document')
    return {'database':path.name,'schema':version,'quick_check':'ok'}


def read_pointer(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get('schema_version') != POINTER_VERSION:
        raise ValueError('unsupported package pointer schema')
    directory = path.parent
    _database_name(directory,value.get('active'))
    if value.get('rollback'):
        _database_name(directory,value['rollback'])
    return value


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,sort_keys=True,separators=(',',':'))
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory = os.open(path.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def switch_pointer(path, *, active, rollback, project_id, document_id):
    path = Path(path)
    active_path = _database_name(path.parent,active)
    rollback_path = _database_name(path.parent,rollback) if rollback else None
    active_check = validate_database(active_path,project_id=project_id,document_id=document_id)
    rollback_check = (validate_database(rollback_path,project_id=project_id,document_id=document_id)
                      if rollback_path else None)
    previous = read_pointer(path) if path.exists() else None
    value = {'schema_version':POINTER_VERSION,'active':active_path.name,
             'rollback':rollback_path.name if rollback_path else None,
             'generation':(previous or {}).get('generation',0)+1,
             'updated_at':datetime.now(timezone.utc).isoformat(),
             'validation':{'active':active_check,'rollback':rollback_check}}
    _atomic_write(path,value)
    return value


def rollback_pointer(path, *, project_id, document_id):
    current = read_pointer(path)
    if not current.get('rollback'):
        raise ValueError('package pointer has no rollback target')
    return switch_pointer(path,active=current['rollback'],rollback=current['active'],
                          project_id=project_id,document_id=document_id)
