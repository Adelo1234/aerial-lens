import json
from pathlib import Path

root = Path('outputs/drone-tracking')
source = (root/'drone_project/download_all.py').read_text(encoding='utf-8')
source = source.split("if __name__ == '__main__':")[0]
cells = []
def cell(kind, text):
    item = dict(cell_type=kind, metadata={}, source=text.splitlines(True), id=f'cell-{len(cells):03d}')
    if kind == 'code':
        compile(text, '<notebook>', 'exec')
        item.update(execution_count=None, outputs=[])
    cells.append(item)

cell('markdown', '''# Download all VisDrone datasets into Colab

Downloads **all 20 dataset files** listed across Tasks 1–5 in the [official repository](https://github.com/VisDrone/VisDrone-Dataset): DET, VID, SOT (both parts), MOT, and DroneCrowd, including validation, test-dev, test-challenge, and SOT initialization files. Source links checked 2026-10-06.

Choose a **CPU runtime** for downloading; no GPU is needed. Run all cells and authorize the Google Drive mount for the account whose storage you want to use. No project ZIP is required. Archives persist in `MyDrive/VisDrone-all`.

The published compressed sizes total approximately **99.02 GB** (actual bytes may differ). Colab disk is finite and temporary; inspect the printed free disk before downloading. Extracting everything requires additional space. By default this notebook retains all ZIP archives and extracts none; the full datasets remain available as archives. Optional extraction checks exact ZIP uncompressed sizes before proceeding. Challenge ground truth is not provided by the maintainers; SOT initialization is included as published.

Completed archives are reused on rerun. Partial downloads are resumed where supported by Drive. Quota-blocked or broken links are recorded as failures, and the final cell raises an error if any of the 20 files is incomplete. SHA-256 digests are calculated locally for reuse/provenance; the repository does not provide upstream checksums.
''')
cell('code', '''import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown==5.2.0'], check=True)
''')
cell('code', source)
cell('code', '''from google.colab import drive
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/VisDrone-all')
EXTRACT_ALL = False  # Keep archives in Drive; extract only required splits locally for training.
# Verified Drive storage: 11.15 GB used of 200 GB on 2026-10-06.
# Update this number from Drive's storage page before a later fresh download.
DRIVE_FREE_GB = 188.85
manifest = download_all(ROOT, extract=EXTRACT_ALL, drive_free_gb=DRIVE_FREE_GB)
''')
cell('code', '''assert len(manifest) == 20 and all(item['status'] == 'complete' for item in manifest.values())
for name, item in manifest.items():
    print(f"{name:40s} {item['bytes']/10**9:7.2f} GB  {item['path']}")
print('TOTAL:', sum(item['bytes'] for item in manifest.values())/10**9, 'GB')
print('Download manifest:', ROOT/'download-all-manifest.json')
''')
cell('markdown', '''## Extract a dataset when you need it

Keeping archives avoids spending disk on two copies of every image. Uncomment the example below to extract DET training for the existing YOLO project. Other tasks keep their own annotation formats and need task-specific training/evaluation; downloading them does not convert them to YOLO labels.

Colab deletes runtime files when the runtime is reset/deleted. Download the manifest or save it to Drive if you need a record. See [Colab storage and runtime limits](https://research.google.com/colaboratory/faq.html).
''')
cell('code', '''# det_root = extract_one(ROOT/'DET/train.zip', ROOT/'extracted/DET/train')
# print(det_root/'VisDrone2019-DET-train')
''')
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
               language_info=dict(name='python'), colab=dict(name='Colab_Download_All_VisDrone.ipynb')),
               nbformat=4, nbformat_minor=5)
total = sum(row[3] for row in __import__('runpy').run_path(str(root/'drone_project/download_all.py'))['CATALOG'])
cells[0]['source'] = ''.join(cells[0]['source']).replace('99.02', f'{total:.2f}').splitlines(True)
for path in [root/'Colab_Download_All_VisDrone.ipynb', Path('outputs/Colab_Download_All_VisDrone.ipynb')]:
    path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print('Created standalone all-data notebook:', len(cells), 'cells;', total, 'GB')
