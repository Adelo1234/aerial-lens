"""Verify and restore the release's binary artifacts into this source checkout."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'runtime-assets.json').read_text())
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected['sha256']:
        raise SystemExit('Archive SHA-256 does not match this source snapshot.')
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read('RUNTIME_ASSETS_MANIFEST.json'))
        for item in manifest:
            relative = PurePosixPath(item['path'])
            target = (root / Path(*relative.parts)).resolve()
            if relative.is_absolute() or not target.is_relative_to(root):
                raise SystemExit('Unsafe path in runtime archive.')
            data = archive.read(item['path'])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise SystemExit(f"Integrity check failed: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f'Restored {len(manifest)} verified runtime files. Source files were not replaced.')


if __name__ == '__main__':
    main()
