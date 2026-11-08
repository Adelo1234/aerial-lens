import sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work/download-deps'))
import gdown

target = ROOT/'local-training/raw/DET/train.zip'
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    partial = target.with_suffix('.zip.partial')
    result = gdown.download(id='1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn', output=str(partial), quiet=False, resume=True)
    if not result or not zipfile.is_zipfile(partial):
        raise RuntimeError('Official DET train download did not return a ZIP.')
    with zipfile.ZipFile(partial) as bundle:
        bad = bundle.testzip()
        if bad:
            raise RuntimeError(f'Archive CRC failed: {bad}')
    partial.replace(target)
print('Verified DET archive:', target, target.stat().st_size, flush=True)
