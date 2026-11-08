"""Download the official VisDrone DET training archive linked by its maintainers."""
import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE = 'https://github.com/VisDrone/VisDrone-Dataset'
FILE_ID = '1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn'
ARCHIVE_URL = f'https://drive.google.com/file/d/{FILE_ID}/view'
FOLDER = 'VisDrone2019-DET-train'


def extract_archive(archive, destination):
    """Validate ZIP members and required layout before exposing extracted data."""
    archive, destination = Path(archive), Path(destination)
    target = destination/FOLDER
    if target.exists():
        raise ValueError(f'Extraction target already exists: {target}')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in member.filename:
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Archive symlinks are unsupported.')
        with tempfile.TemporaryDirectory(prefix='extract-', dir=destination) as staging:
            bundle.extractall(staging)
            source = Path(staging)/FOLDER
            if not (source/'images').is_dir() or not (source/'annotations').is_dir():
                raise ValueError(f'Expected {FOLDER}/images and annotations in official archive.')
            if not any((source/'images').glob('*.jpg')) or not any((source/'annotations').glob('*.txt')):
                raise ValueError('Archive contains no expected JPG images or TXT annotations.')
            shutil.move(str(source), str(target))
    return target


def download(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target, manifest = output/FOLDER, output/'download-source.json'
    if target.exists():
        if not manifest.is_file() or not (target/'images').is_dir() or not (target/'annotations').is_dir():
            raise ValueError('Existing data has no completed download manifest. Choose a new output directory.')
        print(f'Reusing completed download: {target}')
        return target
    archive = output/f'{FOLDER}.zip'
    if not archive.is_file():
        import gdown
        partial = output/f'{FOLDER}.zip.download'
        result = gdown.download(id=FILE_ID, output=str(partial), quiet=False)
        if not result or not partial.is_file() or not zipfile.is_zipfile(partial):
            raise RuntimeError(f'Download failed or returned a non-ZIP file. Download manually from {ARCHIVE_URL}; '
                               f'save it as {archive} and rerun. Drive quotas can temporarily block downloads.')
        partial.replace(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f'Invalid archive: {archive}. Replace it with the official ZIP.')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    target = extract_archive(archive, output)
    manifest.write_text(json.dumps(dict(source_repository=SOURCE, archive_url=ARCHIVE_URL,
                          task='Task 1: Object Detection in Images', subset='VisDrone-DET trainset',
                          archive_sha256=digest.hexdigest(), archive_bytes=archive.stat().st_size,
                          note='Locally calculated digest for provenance; upstream checksum not supplied.'),
                          indent=2), encoding='utf-8')
    print(f'Downloaded and extracted: {target}')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch the official VisDrone DET train ZIP')
    parser.add_argument('--output', default='data/raw')
    download(parser.parse_args().output)
