import json
from pathlib import Path

base = json.loads(Path('outputs/Colab_Part1_Training.ipynb').read_text())
def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item

intro = """# Part 1: improve VisDrone accuracy
Uses the same custom split and saved YOLOv8n baseline. Trains a separate pretrained YOLOv8s candidate at 1280 pixels for up to 100 epochs with early stopping. This is an experiment, not a verified accuracy improvement. Larger images and a larger model consume more GPU time. Select a T4 GPU and run all with adelalyamani2006@gmail.com.

Validation selects the model, image size, and deployment confidence. Test data is evaluated only after selection. Baseline files are preserved. Checkpoints and logs persist in Drive. Scores remain custom project scores; exact duplicates are grouped, but related sequence frames may share splits.
"""
prepare = ''.join(base['cells'][3]['source'])
# Check the old manifest BEFORE the preparation cell can replace its Drive copy.
prepare = """OLD_MANIFEST = Path('/content/drive/MyDrive/aerial-tracking-project/manifest.json')
assert OLD_MANIFEST.is_file(), 'The original split manifest is required for a fair comparison.'
original_manifest = json.loads(OLD_MANIFEST.read_text())
""" + prepare
prepare = prepare.replace("for name in ['summary.json', 'manifest.json', 'data.yaml']:", """new_manifest = json.loads((DATASET/'manifest.json').read_text())
identity = lambda records: sorted((r['image'], r['split'], r['sha256']) for r in records)
assert identity(original_manifest) == identity(new_manifest), 'Split differs from baseline; stop before training.'
for name in ['summary.json', 'manifest.json', 'data.yaml']:""")
prepare = prepare.replace("shutil.copy2(RAW/'download-all-manifest.json', PERSIST/'download-all-manifest.json')", """if (RAW/'download-all-manifest.json').is_file():
    shutil.copy2(RAW/'download-all-manifest.json', PERSIST/'download-all-manifest.json')""")
config = """from datetime import datetime, timezone
import numpy as np
from ultralytics import YOLO
BASELINE = PERSIST/'runs/yolov8n-visdrone/weights/best.pt'
assert BASELINE.is_file(), f'Baseline missing: {BASELINE}'
EXPERIMENT = PERSIST/'accuracy-experiments'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
EXPERIMENT.mkdir(parents=True, exist_ok=False)
IMAGE_SIZE = 1280
BATCH = 2  # Conservative T4 batch for dense scenes at 1280 pixels.
MAX_DET = 1000
EPOCHS = 100
PATIENCE = 20
# Reconnect recovery: set RESUME_CHECKPOINT to the interrupted candidate's last.pt.
# A completed, stripped checkpoint cannot be resumed; use it for evaluation.
RESUME_CHECKPOINT = None
print('Experiment output:', EXPERIMENT)

# Inspect TRAINING labels only; no validation/test-derived training weights.
from PIL import Image
from collections import Counter
counts = Counter()
small_boxes = total_boxes = crowded_images = 0
for image_path in sorted((DATASET/'images/train').glob('*')):
    if not image_path.is_file():
        continue
    with Image.open(image_path) as image:
        width, height = image.size
    rows = [row.split() for row in (DATASET/'labels/train'/f'{image_path.stem}.txt').read_text().splitlines() if row.strip()]
    crowded_images += len(rows) > 300
    resize = IMAGE_SIZE / max(width, height)
    for row in rows:
        cls, _, _, w, h = map(float, row)
        counts[int(cls)] += 1
        total_boxes += 1
        small_boxes += min(w * width * resize, h * height * resize) < 16
report = dict(train_class_counts=dict(counts), train_boxes=total_boxes,
              boxes_with_short_side_under_16_pixels_at_1280=small_boxes,
              train_images_over_300_objects=crowded_images)
(EXPERIMENT/'training-label-audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"""
helpers = """def measure(weights, size, label, split='val'):
    model = YOLO(str(weights))
    result = model.val(data=str(DATASET/'data.yaml'), split=split, imgsz=size,
                       batch=BATCH, device=0, conf=0.001, iou=0.7, max_det=MAX_DET,
                       project=str(EXPERIMENT), name=label, plots=True, verbose=False)
    metrics = dict(label=label, weights=str(weights), imgsz=size, split=split,
                   precision=float(result.box.mp), recall=float(result.box.mr),
                   map50=float(result.box.map50), map50_95=float(result.box.map),
                   speed_ms=result.speed, max_det=MAX_DET, metric_confidence_floor=0.001,
                   per_class=[dict(class_id=int(c), name=model.names[int(c)],
                                   precision=float(result.box.p[i]), recall=float(result.box.r[i]),
                                   map50=float(result.box.ap50[i]), map50_95=float(result.box.ap[i]))
                              for i, c in enumerate(result.box.ap_class_index)])
    if split == 'val':
        index = int(np.argmax(result.box.f1_curve.mean(axis=0)))
        metrics['deployment_confidence'] = float(result.box.px[index])
        metrics['macro_precision_at_deployment_confidence'] = float(result.box.p_curve[:, index].mean())
        metrics['macro_recall_at_deployment_confidence'] = float(result.box.r_curve[:, index].mean())
        metrics['confidence_selection'] = 'Maximum unsmoothed mean per-class validation F1 at IoU 0.5'
    (Path(result.save_dir)/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    del model
    torch.cuda.empty_cache()
    return metrics

# Compare using the same max_det, confidence floor and NMS IoU.
# Published old metrics used different settings and are not the comparison baseline.
baselines = [measure(BASELINE, 960, 'baseline-val-960'),
             measure(BASELINE, IMAGE_SIZE, 'baseline-val-1280')]
(EXPERIMENT/'baseline-comparison.json').write_text(json.dumps(baselines, indent=2))
"""
training_script = """import json, sys
from pathlib import Path
import torch, ultralytics
from ultralytics import YOLO
folder, dataset, resume, size, batch, epochs, patience = sys.argv[1:]
config = dict(data=dataset, epochs=int(epochs), patience=int(patience), imgsz=int(size), batch=int(batch),
              device=0, workers=2, seed=42, deterministic=True,
              optimizer='AdamW', lr0=0.001, lrf=0.01, cos_lr=True,
              weight_decay=0.0005, warmup_epochs=3, mosaic=1.0,
              close_mosaic=10, scale=0.5, fliplr=0.5, flipud=0.0,
              hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
              max_det=1000, amp=True, cache=False, save=True,
              save_period=5, plots=True, project=folder, name='candidate',
              exist_ok=False)
model = YOLO(resume if resume else 'yolov8s.pt')
if resume:
    model.train(resume=True)
else:
    model.train(**config)
actual = Path(model.trainer.save_dir)
(Path(folder)/'candidate-path.json').write_text(json.dumps({'run_dir': str(actual)}))
(actual/'environment.json').write_text(json.dumps(dict(
    torch=torch.__version__, ultralytics=ultralytics.__version__,
    gpu=torch.cuda.get_device_name(0), actual_args=vars(model.trainer.args)), indent=2, default=str))
"""
train_cell = r"""# File logging avoids flooding Colab with every progress-bar update.
import time
script = PROJECT/'train_accuracy_candidate.py'
script.write_text(TRAINING_SCRIPT)
log_path = EXPERIMENT/'training.log'
with log_path.open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(script), str(EXPERIMENT),
                                str(DATASET/'data.yaml'), str(RESUME_CHECKPOINT or ''),
                                str(IMAGE_SIZE), str(BATCH), str(EPOCHS), str(PATIENCE)],
                               stdout=log, stderr=subprocess.STDOUT)
    shown = ''
    while process.poll() is None:
        time.sleep(10)
        with log_path.open(errors='replace') as stream:
            stream.seek(max(0, log_path.stat().st_size-2048))
            tail = stream.read()
        latest = tail.replace('\r', '\n').splitlines()
        if latest and latest[-1] != shown:
            shown = latest[-1]
            print(shown[-250:], flush=True)
assert process.returncode == 0, f'Training failed; inspect {log_path}'
run_dir = Path(json.loads((EXPERIMENT/'candidate-path.json').read_text())['run_dir'])
CANDIDATE = run_dir/'weights/best.pt'
assert CANDIDATE.is_file(), 'Candidate did not produce a saved checkpoint.'
print('Saved candidate:', CANDIDATE)
"""
comparison = """candidate = measure(CANDIDATE, IMAGE_SIZE, 'candidate-val-1280')
comparisons = baselines + [candidate]
# Primary quality measure includes localization quality, not precision alone.
selected = max(comparisons, key=lambda item: item['map50_95'])
selection = dict(selected=selected, comparisons=comparisons,
                 candidate_improved=candidate['map50_95'] > max(x['map50_95'] for x in baselines),
                 criterion='Highest validation mAP50-95; ties keep the earlier baseline',
                 caveat='Single-seed custom split; differences are not a statistical significance claim.')
(EXPERIMENT/'selection.json').write_text(json.dumps(selection, indent=2))
print(json.dumps(selection, indent=2))
from IPython.display import display
for path in [run_dir/'results.png', EXPERIMENT/'candidate-val-1280'/'confusion_matrix_normalized.png']:
    if path.is_file():
        display(Image.open(path))
"""
test = """# Frozen selection from validation; test results never influence model selection.
selection = json.loads((EXPERIMENT/'selection.json').read_text())
selected = selection['selected']
final = measure(selected['weights'], selected['imgsz'], 'selected-test', split='test')
print('Deployment confidence selected on validation:', selected['deployment_confidence'])
print('AP is calculated at the low confidence floor; deployment confidence is a separate operating point.')
"""
nb = dict(metadata=base['metadata'], nbformat=4, nbformat_minor=5, cells=[
    cell('markdown', intro), cell('code', ''.join(base['cells'][1]['source'])),
    cell('markdown', '## Restore and verify the original split\nStops if the image identities, hashes or split assignments differ.'),
    cell('code', prepare), cell('code', config),
    cell('markdown', '## Measure the baseline on validation\nThe 1000-detection limit is shared by all models. Confidence is selected on validation; raising it can increase precision while reducing recall.'),
    cell('code', helpers),
    cell('markdown', '## Train a separate candidate\nYOLOv8s, 1280 pixels, up to 100 epochs, patience 20, batch 2, cosine learning-rate decay. Baseline files are preserved. Training may exceed a single Colab session. Set RESUME_CHECKPOINT for an interrupted, unstripped last.pt; actual resume arguments come from that checkpoint.'),
    cell('code', 'TRAINING_SCRIPT = '+repr(training_script)+'\n'+train_cell),
    cell('markdown', '## Select using validation results'), cell('code', comparison),
    cell('markdown', '## Final held-out test\nReports the selected model once; do not use these results to retune it.'), cell('code', test)])
nb['metadata']['colab']['name'] = 'Colab_Part1_Improve_Accuracy.ipynb'
for i, item in enumerate(nb['cells']):
    if item['cell_type'] == 'code':
        compile(''.join(item['source']), f'cell {i}', 'exec')
compile(training_script, 'training_script', 'exec')
target = Path('outputs/Colab_Part1_Improve_Accuracy.ipynb')
target.write_text(json.dumps(nb, indent=2), encoding='utf-8')
Path('outputs/drone-tracking/Colab_Part1_Improve_Accuracy.ipynb').write_text(target.read_text(), encoding='utf-8')
print(target, target.stat().st_size, 'bytes; all code cells compiled')
