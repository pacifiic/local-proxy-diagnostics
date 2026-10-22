#!/usr/bin/env python3
"""Restore losslessly packed scientific artifacts without overwriting data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe artifact path: {name}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact path escapes checkout: {name}')
    return path


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_one(root: Path, record: dict, verify_only: bool = False) -> str:
    packed = safe_path(root, record['compressed_path'])
    output = safe_path(root, record['path'])
    if file_sha(packed) != record['compressed_sha256']:
        raise ValueError(f'Compressed checksum mismatch: {record["compressed_path"]}')
    if output.exists():
        if output.stat().st_size != record['size'] or file_sha(output) != record['sha256']:
            raise FileExistsError(f'Refusing to overwrite different data: {output}')
        return 'existing'
    if output.is_symlink():
        raise FileExistsError(f'Refusing a dangling output symlink: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        if not verify_only:
            descriptor, name = tempfile.mkstemp(prefix='.restore-', dir=output.parent)
            os.close(descriptor)
            temporary = Path(name)
        digest, size = hashlib.sha256(), 0
        with gzip.open(packed, 'rb') as source:
            sink = temporary.open('wb') if temporary else None
            try:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
                    if size > record['size']:
                        raise ValueError(f'Artifact larger than manifest: {output}')
                    if sink:
                        sink.write(block)
            finally:
                if sink:
                    sink.close()
        if size != record['size'] or digest.hexdigest() != record['sha256']:
            raise ValueError(f'Restored checksum mismatch: {output}')
        if temporary:
            # Link creation is exclusive, unlike replace/rename on an existing file.
            os.link(temporary, output)
        return 'verified' if verify_only else 'restored'
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/archive_manifest.json').read_text())
    if manifest['format_version'] != 1:
        raise ValueError('Unsupported artifact manifest version')
    paths = [entry['path'] for entry in manifest['entries']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate output paths in artifact manifest')
    counts = {}
    for index, record in enumerate(manifest['entries'], 1):
        status = restore_one(root, record, args.verify_only)
        counts[status] = counts.get(status, 0) + 1
        if index % 50 == 0:
            print(f'{index}/{len(paths)} artifacts checked', flush=True)
    print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()
