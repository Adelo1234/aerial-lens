"""Collect measured Stage 1 results and update the project's stale status text."""
import json
from pathlib import Path
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
run = root/'local-training/runs/yolov8s-1280-20261008-142900'
final = run/'stage1-final'
metrics = json.loads((final/'test-metrics.json').read_text())
assert metrics['split'] == 'test'
report = final/'PART1_RESULTS.md'
text = report.read_text(encoding='utf-8')
text = text.replace('## Evidence', '''## Error analysis

Cars achieved the highest test AP50 (86.4%) and recall (82.6%). Buses also performed comparatively well (75.5% AP50). Bicycles (35.4% AP50, 31.7% recall) and people (39.7% AP50, 33.8% recall) remain weaker classes. The confusion matrix shows substantial missed detections for those classes and confusion between vans and cars. Object size has not been isolated using a size-aware evaluator, so these results do not prove a small-object-specific cause. Any new accuracy experiment should be selected on validation, preserving this test as a disclosed already-used evaluation set.

## Evidence''')
report.write_text(text, encoding='utf-8')

bundle = root/'outputs/part1-results'
assert not bundle.exists(), 'Results bundle already exists.'
shutil.copytree(final, bundle/'stage1-final')
candidate = bundle/'candidate'
(candidate/'weights').mkdir(parents=True)
for name in ['results.csv', 'results.png', 'args.yaml']:
    shutil.copy2(run/'candidate'/name, candidate/name)
shutil.copy2(run/'candidate/weights/best.pt', candidate/'weights/best.pt')
shutil.copy2(root/'local-training/dataset/summary.json', bundle/'dataset-summary.json')

project = root/'outputs/drone-tracking/docs/PROJECT_REPORT.md'
existing = project.read_text(encoding='utf-8')
existing = existing.replace('It validates label geometry and rejects exact duplicate images.',
                            'It validates label geometry and keeps exact duplicate images in the same split using SHA-256 grouping.')
existing = existing.replace('The assignment\'s 70/20/10 split is implemented with seed 42. A sequence-group CSV prevents related frames from crossing subsets. Without grouping, the split is image-level. Grouped image proportions can differ from the requested ratios. This split should not be described as the official VisDrone benchmark protocol.',
                            'The assignment\'s approximate 70/20/10 split uses seed 42: 4,529 training, 1,295 validation, and 647 test images. In the completed run, grouping covers exact duplicates, not every related sequence frame. Related scenes may cross splits. This is a custom project split, not the official VisDrone benchmark protocol.')
existing = existing.replace('## Temporal association',
                            'The completed candidate fine-tuned pretrained YOLOv8s for 100 epochs at 1280 pixels, batch 4, AdamW, cosine learning-rate decay, and patience 20. Mosaic was disabled during the final ten epochs. Final evaluation used batch 2, confidence floor .001, NMS IoU .7, and max_det 1000. The Colab baseline was not recovered for a controlled comparison.\n\n## Temporal association')
start, end = existing.index('## Results status'), existing.index('## References')
existing = existing[:start] + '''## Results status

Part 1 detector training and final custom held-out test evaluation are complete. The frozen YOLOv8s checkpoint achieved test precision 64.8%, recall 54.0%, mAP50 57.8%, and mAP50-95 35.5% on 647 images. These precision and recall values follow the Ultralytics class-mean reporting convention. Validation at the same settings achieved precision 66.9%, recall 54.3%, mAP50 58.4%, and mAP50-95 36.5%.

Validation selected a confidence threshold of approximately .278. At that fixed threshold, interpolated test curves indicate approximate class-mean precision 65.2% and recall 53.6%. The metric confidence floor is not the deployment threshold.

See [the final Part 1 report](../../part1-results/stage1-final/PART1_RESULTS.md) for the full protocol, per-class metrics, limitations, and evidence. [The results bundle](../../part1-results/) includes the trained checkpoint, training curves, confusion matrices, precision-recall curves, and labeled/predicted samples.

The split groups exact duplicates but not all related scene frames. No statistically controlled improvement over the Colab baseline or official benchmark result is claimed. Tracking IDF1/HOTA and real closed-loop SITL flight metrics have not been evaluated. The toy simulation is separate from learned detector performance.

''' + existing[end:]
project.write_text(existing, encoding='utf-8')
readme = root/'outputs/drone-tracking/README.md'
existing = readme.read_text(encoding='utf-8').replace(
    'No trained VisDrone model is included.',
    'A trained YOLOv8s checkpoint and completed Part 1 evaluation are available in [the local results bundle](../part1-results/stage1-final/PART1_RESULTS.md).')
readme.write_text(existing, encoding='utf-8')
with (root/'outputs/PART1_STATUS.md').open('a', encoding='utf-8') as stream:
    stream.write('\nStage 1 final evaluation 2026-10-27: saved YOLOv8s checkpoint frozen; SHA-256 recorded. Validation/test evaluated at 1280, batch2, max_det1000, conf floor .001, NMS IoU .7. Test: precision .647958, recall .540122, mAP50 .577535, mAP50-95 .354938 on 647 images. Validation-selected confidence .278278; approximate test macro P/R from curves at that threshold .651667/.536354. Results bundled in outputs/part1-results with best.pt, JSON metrics, plots, and report. No new training launched; no controlled Colab baseline comparison available.\n')
archive = root/'outputs/Part1_Results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            zipped.write(path, path.relative_to(bundle.parent))
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
assert (bundle/'candidate/weights/best.pt').read_bytes() == (run/'candidate/weights/best.pt').read_bytes()
print('Results bundle verified:', archive, archive.stat().st_size, 'bytes')
