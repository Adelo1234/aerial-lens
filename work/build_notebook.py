import json
from pathlib import Path

root = Path('outputs/drone-tracking')
cells = []
def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

md('''# Aerial detection and tracking: YOLOv8 + ByteTrack

This notebook prepares your supplied VisDrone dataset, trains YOLOv8, evaluates the held-out test set, and saves a tracked video with visual-control commands.

**Before running:** select a GPU runtime and upload `drone-tracking.zip` to `MyDrive`. Training results are written to Drive. No result is prefilled or fabricated. The synthetic camera demo at the end tests control only.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys, subprocess, shutil, json
ZIP_PATH = Path('/content/drive/MyDrive/drone-tracking.zip')
assert ZIP_PATH.is_file(), f'Upload the project ZIP to {ZIP_PATH}'
PROJECT = Path('/content/drone-tracking')
if not PROJECT.exists():
    shutil.unpack_archive(str(ZIP_PATH), '/content')
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > GPU, then restart this notebook.'
print('GPU:', torch.cuda.get_device_name(0))
''')
md('''## 1. Download all five official VisDrone tasks

Dataset source: https://github.com/VisDrone/VisDrone-Dataset

This cell downloads **all 20 dataset files** across DET, VID, SOT, MOT, and DroneCrowd, including both SOT parts, all listed splits, and SOT challenge initialization. Published compressed sizes total approximately **98.45 GB**. Archives persist in `MyDrive/VisDrone-all`; DET training is extracted locally for fast image access. Update the free Drive quota in this cell from the account's storage page before a later fresh download. Mounted Drive filesystem capacity can reflect local cache space rather than cloud quota.

Archives are retained and checksummed. Only DET training is extracted for this YOLOv8 assignment. The standalone `Colab_Download_All_VisDrone.ipynb` downloads the same files without requiring the project ZIP or GPU. Other task annotations remain in their original formats and require separate training/evaluation workflows. Test-challenge ground truth is not available.

Failed links and Drive quota errors are recorded in `download-all-manifest.json`. Rerun to reuse completed archives and retry failures. If space is insufficient, use a runtime with more disk or set `RAW` to a mounted Drive folder with sufficient quota. Extracting DET training and creating the custom split needs additional disk beyond the archive total.

The assignment's custom 70/20/10 split still uses **DET training only**. Official validation/test splits and other tasks stay separate. These are custom project scores, not official benchmark scores. Supply a sequence-groups CSV with `image_stem,group_id` through `GROUPS_CSV` if appropriate.
''')
code('''from drone_project.download_all import download_all, extract_one, SOURCE
SOURCE_URL = SOURCE
RAW = Path('/content/drive/MyDrive/VisDrone-all')
# Update from the account's Drive storage page before a later fresh download.
ALL_DOWNLOADS = download_all(RAW, extract=False, drive_free_gb=188.85)
DET_EXTRACTED = extract_one(RAW/'DET/train.zip', Path('/content/visdrone-extracted/DET/train'))
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
md('''## 2. Verify annotations visually

Check small objects, crowded scenes, clipped boxes, and class names before training. Review the class counts printed above. Do not treat missing classes as strong evaluation evidence.
''')
code('''from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from drone_project.data import NAMES
samples = sorted((DATASET/'images/train').glob('*'))[:4]
fig, axes = plt.subplots(1, len(samples), figsize=(20,6), squeeze=False)
for ax, path in zip(axes[0], samples):
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for line in (DATASET/'labels/train'/f'{path.stem}.txt').read_text().splitlines():
        if not line.strip(): continue
        c, cx, cy, bw, bh = map(float, line.split())
        box = ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)
        draw.rectangle(box, outline='lime', width=2)
        draw.text((box[0],box[1]), NAMES[int(c)], fill='yellow')
    ax.imshow(image); ax.set_title(path.name); ax.axis('off')
plt.show()
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
''')
md('''## 3. Optional quick setup check

This uses a small fraction of training images for one epoch, with validation. It validates setup, not final performance. Keep this run separate from your full training.
''')
code('''RUN_QUICK_CHECK = False
if RUN_QUICK_CHECK:
    from ultralytics import YOLO
    YOLO('yolov8n.pt').train(data=str(DATASET/'data.yaml'), fraction=0.02,
        epochs=1, imgsz=640, batch=4, device=0, workers=2,
        project=str(PERSIST/'runs'), name='quick-check', seed=42)
''')
md('''## 4. Train the baseline

Start with YOLOv8n. Reduce `BATCH` if you run out of GPU memory. Images at 960 pixels cost more compute than 640, but preserve more detail for small objects. The best checkpoint is chosen on validation. Colab runtime duration and available GPUs vary.
''')
code('''MODEL = 'yolov8n.pt'
EPOCHS = 50
IMAGE_SIZE = 960
BATCH = 8
RUN_NAME = 'yolov8n-visdrone'
RUNS = PERSIST/'runs'
run('train', '--data', DATASET/'data.yaml', '--model', MODEL,
    '--epochs', EPOCHS, '--imgsz', IMAGE_SIZE, '--batch', BATCH,
    '--device', '0', '--output', RUNS, '--name', RUN_NAME)
# If the name was already used, Ultralytics may increment the folder.
import glob
candidates = list(RUNS.glob(RUN_NAME+'*/weights/best.pt'))
assert candidates, 'Training has not produced best.pt.'
BEST = max(candidates, key=lambda p: p.stat().st_mtime)
print('Best checkpoint:', BEST)
''')
md('''### Resume after a runtime interruption

If the runtime resets, run setup/data preparation again to restore `/content/visdrone-project-data`. Use the same images, class mapping, and split seed. Set `LAST` to your Drive checkpoint, and run this cell instead of starting a new full run.
''')
code('''RESUME = False
LAST = RUNS/RUN_NAME/'weights/last.pt'
if RESUME:
    assert LAST.is_file(), f'No checkpoint at {LAST}'
    run('train', '--data', DATASET/'data.yaml', '--resume', LAST)
    BEST = LAST.parent/'best.pt'
''')
md('''## 5. Final test evaluation

Run after choosing the model configuration on validation. This writes aggregate/per-class precision, recall, mAP50 and mAP50:95, speed, and plots. Use these measured values in your report. Repeatedly tuning against test metrics invalidates a held-out test claim.
''')
code('''run('evaluate', '--data', DATASET/'data.yaml', '--weights', BEST,
    '--split', 'test', '--imgsz', IMAGE_SIZE, '--batch', BATCH,
    '--device', '0', '--output', RUNS)
metric_paths = list(RUNS.glob('eval-test*/metrics.json'))
METRICS = max(metric_paths, key=lambda p: p.stat().st_mtime)
print(METRICS.read_text())
from IPython.display import display
for name in ['results.png', 'confusion_matrix_normalized.png']:
    path = BEST.parent.parent/name if name == 'results.png' else METRICS.parent/name
    if path.exists(): display(Image.open(path))
''')
md('''## 6. Track an aerial video

Supply an actual continuous video. A sequence of unrelated still images cannot validate tracking. This cell is optional until a video is supplied. Classes `0,1` mean VisDrone pedestrian/people. The output contains ByteTrack IDs and command logs; recorded footage does not respond to those commands.
''')
code('''VIDEO = Path('/content/drive/MyDrive/aerial-video.mp4')
TRACK_CLASSES = '0,1'
TRACK_OUTPUT = PERSIST/'tracking'
if VIDEO.is_file():
    run('track', '--weights', BEST, '--source', VIDEO,
        '--classes', TRACK_CLASSES, '--device', '0', '--output', TRACK_OUTPUT)
    print('Saved:', TRACK_OUTPUT/'tracked.mp4')
    print('Commands:', TRACK_OUTPUT/'commands.csv')
else:
    print('Set VIDEO to your aerial video to run tracking.')
''')
md('''## 7. Control simulation and downloads

This synthetic camera demo uses ideal boxes, not a detector. It includes a brief occlusion and zeroes commands while the target is missing. ArduPilot command replay instructions are in `docs/SITL.md`; full 3D camera integration is separate.
''')
code('''SIM = Path('/content/control-demo')
if not SIM.exists(): run('simulate', '--output', SIM)
from IPython.display import Image as DisplayImage
display(DisplayImage(filename=str(SIM/'simulation.gif')))
print((SIM/'summary.json').read_text())
shutil.copytree(SIM, PERSIST/'control-demo', dirs_exist_ok=True)

# Optional: download your actual trained weights and measured results.
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(BEST))
    files.download(str(METRICS))
print('All persistent results:', PERSIST)
''')
md('''## Report checklist

- Record image/box counts and source/license information.
- Include actual training curves, confusion matrix, and per-class test AP.
- Describe failures on tiny targets, occlusion, and crowded scenes.
- Report video throughput separately from detector-only inference time.
- Do not claim IDF1/HOTA without labeled trajectories and a tracking evaluator.
- Describe the synthetic control model accurately; it is not a measured aircraft flight.

Source documentation: [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [training](https://docs.ultralytics.com/modes/train/), [tracking](https://docs.ultralytics.com/modes/track/), [VisDrone](https://github.com/VisDrone/VisDrone-Dataset).
''')
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                language_info=dict(name='python',version='3.12'), colab=dict(name='Colab_Training.ipynb'),accelerator='GPU'),
                nbformat=4,nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:03d}'
    if cell['cell_type']=='code':
        compile(''.join(cell['source']),f'notebook cell {i}','exec')
(root/'Colab_Training.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(f'Wrote and syntax-checked {len(cells)} cells')
