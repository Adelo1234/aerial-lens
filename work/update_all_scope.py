from pathlib import Path

path = Path('work/build_notebook.py')
text = path.read_text()
start = text.index("md('''## 1. Download")
end = text.index("md('''## 2. Verify", start)
replacement = """md('''## 1. Download all five official VisDrone tasks

Dataset source: https://github.com/VisDrone/VisDrone-Dataset

This cell downloads **all 20 dataset files** across DET, VID, SOT, MOT, and DroneCrowd, including both SOT parts, all listed splits, and SOT challenge initialization. Published compressed sizes total approximately **98.45 GB**. Colab storage varies: the downloader checks actual free disk and refuses to start if the remaining archive estimates exceed it. Downloads are temporary in `/content`; mounting Drive does not enlarge that disk.

Archives are retained and checksummed. Only DET training is extracted for this YOLOv8 assignment. The standalone `Colab_Download_All_VisDrone.ipynb` downloads the same files without requiring the project ZIP or GPU. Other task annotations remain in their original formats and require separate training/evaluation workflows. Test-challenge ground truth is not available.

Failed links and Drive quota errors are recorded in `download-all-manifest.json`. Rerun to reuse completed archives and retry failures. If space is insufficient, use a runtime with more disk or set `RAW` to a mounted Drive folder with sufficient quota. Extracting DET training and creating the custom split needs additional disk beyond the archive total.

The assignment's custom 70/20/10 split still uses **DET training only**. Official validation/test splits and other tasks stay separate. These are custom project scores, not official benchmark scores. Supply a sequence-groups CSV with `image_stem,group_id` through `GROUPS_CSV` if appropriate.
''')
code('''from drone_project.download_all import download_all, extract_one, SOURCE
SOURCE_URL = SOURCE
RAW = Path('/content/visdrone-all')
ALL_DOWNLOADS = download_all(RAW, extract=False)
DET_EXTRACTED = extract_one(RAW/'DET/train.zip', RAW/'extracted/DET/train')
TRAIN_ROOT = DET_EXTRACTED/'VisDrone2019-DET-train'
IMAGES = TRAIN_ROOT/'images'
LABELS = TRAIN_ROOT/'annotations'
LABEL_FORMAT = 'visdrone'
GROUPS_CSV = None
PERSIST = Path('/content/drive/MyDrive/aerial-tracking-project')
DATASET = Path('/content/visdrone-project-data')
PERSIST.mkdir(parents=True, exist_ok=True)
assert IMAGES.is_dir() and LABELS.is_dir(), 'Expected images and annotations in DET training archive.'

def run(*args):
    subprocess.run([sys.executable, '-m', 'drone_project.cli', *map(str, args)], check=True)

if DATASET.exists():
    print('Dataset already prepared in this runtime. Review its manifest before reusing it.')
else:
    command = ['prepare', '--images', IMAGES, '--labels', LABELS,
               '--format', LABEL_FORMAT, '--output', DATASET]
    if GROUPS_CSV:
        command += ['--groups-csv', GROUPS_CSV]
    run(*command)
summary = json.loads((DATASET/'summary.json').read_text())
print(json.dumps(summary, indent=2))
for name in ['summary.json', 'manifest.json', 'data.yaml']:
    shutil.copy2(DATASET/name, PERSIST/name)
shutil.copy2(RAW/'download-all-manifest.json', PERSIST/'download-all-manifest.json')
(PERSIST/'dataset-source.json').write_text(json.dumps(dict(
    source_url=SOURCE_URL, image_path=str(IMAGES), label_path=str(LABELS),
    label_format=LABEL_FORMAT, split_policy='Custom 70/20/10 from official DET training only',
    note='All five tasks downloaded; DET training only used in YOLO workflow.'), indent=2))
print('Manifest copied to Drive. Re-prepare runtime-local data after a runtime reset.')
''')
"""
path.write_text(text[:start]+replacement+text[end:], encoding='utf-8')
readme = Path('outputs/drone-tracking/README.md')
text = readme.read_text(encoding='utf-8')
text = text.replace('Run the dataset cell. It downloads and extracts the official VisDrone-DET training ZIP linked by its maintainers. No dataset shortcut in Drive is required. The cell includes a manual-download fallback for Drive quota failures.', 'Run the dataset cell. It downloads all 20 official dataset files (~98.45 GB compressed) across Tasks 1–5 and extracts DET training for the YOLO workflow. It checks free disk and reports failures.')
start = text.index('Dataset scope is **Task 1:')
end = text.index('The default input is', start)
text = text[:start]+'''Dataset download scope is **all five tasks**: DET, VID, SOT, MOT, and DroneCrowd. Every dataset Google Drive link in the official README is included (20 files, including all splits, both SOT parts, and challenge initialization). The legacy `drone_project.download` command still downloads DET training alone; use `python -m drone_project.download_all --output /content/visdrone-all` for all tasks.

For downloads only, open `Colab_Download_All_VisDrone.ipynb` in Colab and run all cells on a CPU runtime. It is standalone: no ZIP upload or GPU needed. All archives are retained under `/content/visdrone-all/<task>/<split>.zip`. Their published compressed sizes total ~98.45 GB. Disk is finite and runtime-local files are temporary. Optional extraction requires additional disk, checked against exact ZIP uncompressed sizes. Completed archives have CRC checks and SHA-256 records; reruns reuse them and retry failures. Locally calculated hashes record provenance, not upstream authenticity. The downloader raises an error if any file is incomplete.

The training notebook downloads the same complete collection, but only **DET training** enters the assignment's custom 70/20/10 split. Other tasks and official validation/test datasets stay separate. Downloading SOT/MOT/DroneCrowd does not implement their separate benchmark training/evaluation methods. Challenge ground truth is unavailable; published initialization data is included.

'''+text[end:]
readme.write_text(text, encoding='utf-8')
verification = Path('outputs/drone-tracking/docs/VERIFICATION.md')
text = verification.read_text(encoding='utf-8')
start = text.index('The current notebook uses')
verification.write_text(text[:start]+'''The updated notebooks include all 20 dataset links across Tasks 1–5, checked against the official README on 2026-10-06. Downloader fixture tests cover completeness, reuse, explicit failure reporting, safe extraction, and insufficient disk. Notebook code cells passed syntax checks. Full real downloads have not been executed: the accessible Colab browser was signed out. The collection totals approximately 98.45 GB compressed; actual runtime disk determines whether it fits.
''', encoding='utf-8')
