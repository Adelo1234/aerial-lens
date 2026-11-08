"""All 20 Google Drive dataset links in the official VisDrone README (2026-10-06)."""
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE = 'https://github.com/VisDrone/VisDrone-Dataset'
# task, split/part, Drive file ID, approximate compressed GB from the README
CATALOG = [
    ('DET', 'train', '1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn', 1.44),
    ('DET', 'val', '1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59', .07),
    ('DET', 'test-dev', '1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V', .28),
    ('DET', 'test-challenge', '1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT', .28),
    ('VID', 'train', '1NSNapZQHar22OYzQYuXCugA3QlMndzvw', 7.53),
    ('VID', 'val', '1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B', 1.49),
    ('VID', 'test-dev', '1-BEq--FcjshTF1UwUabby_LHhYj41os5', 2.14),
    ('VID', 'test-challenge', '1Qwyp_cEpGyXGqJ8IbusEzuNHgbM403NP', 2.70),
    ('SOT', 'train-part1', '1a-AmQjYATzj8seXLlEm9Sx8aQmClrJka', 7.78),
    ('SOT', 'train-part2', '16YPyhNDQrTgW8I2HaH_HNEO-KTRA-xso', 12.59),
    ('SOT', 'val', '18SNAOlCJtApnG2m45ud-1e_OtGYill0D', 1.29),
    ('SOT', 'test-dev', '1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC', 11.27),
    ('SOT', 'test-challenge-part1', '1zxMLZrkkz4BkufbaDXQVMpDz-PRpPpxs', 17.40),
    ('SOT', 'test-challenge-part2', '13J_9rwpf2TpVVsWgzMUv4uCxpOKDNbef', 17.31),
    ('SOT', 'test-challenge-initialization', '1z7lQ0co8Dcu-ZojMN7dnIxn-i6QbGNL4', .000012),
    ('MOT', 'train', '1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh', 7.53),
    ('MOT', 'val', '1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu', 1.48),
    ('MOT', 'test-dev', '14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu', 2.14),
    ('MOT', 'test-challenge', '1I0nn6dVKctzDE5YJ3q9qOlhKLiSIDAxF', 2.70),
    ('CC', 'DroneCrowd', '1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd', 1.03),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_one(archive, target):
    """Keep separate task/part roots; refuse unsafe paths and incomplete reuse."""
    archive, target = Path(archive), Path(target)
    digest = sha256(archive)
    marker = target/'.extraction-complete.json'
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text())['sha256'] == digest:
            return target
        raise ValueError(f'Existing extraction incomplete or different: {target}. Choose a new target.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in member.filename:
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Archive symlinks are unsupported.')
        needed = sum(m.file_size for m in bundle.infolist()) + 2*10**9
        if shutil.disk_usage(target.parent).free < needed:
            raise OSError(f'Insufficient space to extract {archive.name}: need {needed/10**9:.1f} GB free.')
        with tempfile.TemporaryDirectory(prefix='extract-', dir=target.parent) as staging:
            bundle.extractall(staging)
            (Path(staging)/'.extraction-complete.json').write_text(json.dumps({'sha256': digest}))
            Path(staging).rename(target)
    return target


def download_all(root, extract=False, drive_free_gb=None):
    """Download every archive; retain successes and explicitly report failures."""
    import gdown
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'download-all-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    remaining = sum(gb for task, split, _, gb in CATALOG
                    if not (root/task/f'{split}.zip').is_file())
    # DriveFS statvfs can reflect its local cache rather than cloud quota.
    # Supply free quota verified on the account's Drive storage page when mounted.
    free = drive_free_gb if drive_free_gb is not None else shutil.disk_usage(root).free/10**9
    print(f'All five tasks: {len(CATALOG)} archives, ~{sum(x[3] for x in CATALOG):.2f} GB compressed.')
    print(f'Destination: {root}; free: {free:.1f} GB; remaining estimate: {remaining:.1f} GB.')
    if drive_free_gb is not None:
        print('Free capacity supplied from verified Google Drive storage; local cache checked per archive.')
    if free < remaining + 2:
        raise OSError('Not enough disk for all remaining archives plus 2 GB reserve. '
                      'Use a Colab runtime with more disk or a mounted Drive with enough quota.')
    failures = []
    for task, split, file_id, gb in CATALOG:
        key = f'{task}/{split}'
        archive = root/task/f'{split}.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        record = {'source_repository': SOURCE, 'file_id': file_id,
                  'url': f'https://drive.google.com/file/d/{file_id}/view',
                  'estimated_gb': gb, 'path': str(archive)}
        try:
            print(f'\n[{key}] ~{gb:.2f} GB', flush=True)
            if not archive.exists():
                if shutil.disk_usage(root).free < (gb+2)*10**9:
                    raise OSError('Disk reserve would be exhausted.')
                partial = archive.with_suffix('.zip.partial')
                result = gdown.download(id=file_id, output=str(partial), quiet=False, resume=True)
                if not result or not zipfile.is_zipfile(partial):
                    raise ValueError('Download returned no valid ZIP; check the official link or Drive quota.')
                with zipfile.ZipFile(partial) as bundle:
                    bad = bundle.testzip()
                    if bad: raise ValueError(f'ZIP CRC failure: {bad}')
                partial.replace(archive)
            previous = manifest.get(key, {})
            digest = sha256(archive)
            if previous.get('sha256') and digest != previous['sha256']:
                raise ValueError('Cached archive checksum differs from previous successful download.')
            if not zipfile.is_zipfile(archive):
                raise ValueError('Cached file is not a ZIP. Replace it with the official archive.')
            # Cached files without a prior completion record need full CRC validation too.
            if previous.get('status') != 'complete':
                with zipfile.ZipFile(archive) as bundle:
                    bad = bundle.testzip()
                    if bad: raise ValueError(f'ZIP CRC failure: {bad}')
            record.update(status='complete', sha256=digest, bytes=archive.stat().st_size)
            if extract:
                record['extracted_path'] = str(extract_one(archive, root/'extracted'/task/split))
        except Exception as error:
            record.update(status='failed', error=str(error))
            failures.append(key)
            print(f'FAILED: {key}: {error}', flush=True)
        manifest[key] = record
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temporary.replace(manifest_path)
    print(f'\nCompleted: {len(CATALOG)-len(failures)}/{len(CATALOG)}')
    if failures:
        raise RuntimeError(f'Incomplete downloads: {failures}. Rerun to reuse completed archives and retry failures. '
                           f'See {manifest_path}.')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download all five VisDrone tasks and all splits')
    parser.add_argument('--output', default='/content/visdrone-all')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    download_all(args.output, extract=args.extract)
