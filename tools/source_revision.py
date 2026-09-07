"""Verify explicit historical source bindings across the reviewed relocation."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(root, name):
    root = Path(root).resolve()
    path = PurePosixPath(name)
    candidate = root / name
    if (path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name
            or candidate.is_symlink() or not candidate.resolve().is_relative_to(root)):
        raise ValueError('unsafe historical source path')
    return candidate


def binding(root, name, expected):
    row = json.loads((Path(root) / 'deploy/source/source-revisions.json').read_text())['bindings'][name]
    path = PurePosixPath(row['owner'])
    if row['original_sha256'] != expected or path.is_absolute() or '..' in path.parts:
        raise ValueError('unrecognized historical source binding')
    return row


def verify_relocated_source(root, name, expected):
    """Accept only the exact original bytes or the exact reviewed relocated owner."""
    root = Path(root)
    original = source_path(root, name)
    if original.is_file() and digest(original) == expected:
        return
    row = binding(root, name, expected)
    if digest(source_path(root, row['owner'])) != row['owner_sha256']:
        raise ValueError('classifier changed after source relocation')


def frozen_source_path(root, name, expected):
    """Resolve historical source bytes; never import or execute the saved copy."""
    root = Path(root)
    original = source_path(root, name)
    if original.is_file() and digest(original) == expected:
        return original
    binding(root, name, expected)
    path = source_path(root, 'fixtures/source-history/' + expected + '.py')
    if digest(path) != expected:
        raise ValueError('historical source bytes differ from frozen binding')
    return path
