"""Freeze the trained detector, select confidence on validation, then evaluate test."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'work/yolo-settings')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    dataset = ROOT/'local-training/dataset'
    run = ROOT/'local-training/runs/yolov8s-1280-20261008-142900'
    weights = run/'candidate/weights/best.pt'
    output = run/'stage1-final'
    assert weights.is_file(), 'Completed model checkpoint is missing.'
    assert not output.exists(), 'Final evaluation already exists; inspect it before rerunning.'
    manifest_path = dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    groups, counts = {}, {}
    for row in manifest:
        split = row['split']
        counts[split] = counts.get(split, 0) + 1
        assert groups.setdefault(row['sha256'], split) == split, 'Exact duplicate crosses splits.'
        assert (dataset/'images'/split/row['image']).is_file(), 'Image missing.'
        assert (dataset/'labels'/split/(Path(row['image']).stem+'.txt')).is_file(), 'Label missing.'
    assert counts == {'train': 4529, 'val': 1295, 'test': 647}, counts
    output.mkdir(parents=True)
    config = dict(data=str(dataset/'data.yaml'), imgsz=1280, batch=2, device=0,
                  workers=2, conf=.001, iou=.7, max_det=1000, plots=True,
                  project=str(output), verbose=False)
    selection = dict(weights=str(weights), weights_sha256=digest(weights),
                     manifest_sha256=digest(manifest_path), images_by_split=counts,
                     model='YOLOv8s', imgsz=1280, batch=2, confidence_floor=.001,
                     nms_iou=.7, max_det=1000, torch=torch.__version__,
                     ultralytics=ultralytics.__version__,
                     note='Checkpoint chosen by training validation fitness; no test-based tuning. Original Colab baseline unavailable for a controlled comparison.')

    def measure(split):
        print(f'Starting {split} evaluation', flush=True)
        result = YOLO(str(weights)).val(split=split, name=split, **config)
        metrics = dict(split=split, precision=float(result.box.mp), recall=float(result.box.mr),
                       map50=float(result.box.map50), map50_95=float(result.box.map),
                       speed_ms=result.speed,
                       precision_recall_note='Ultralytics class-mean precision/recall at its split-specific smoothed mean-F1 operating point, not at confidence .001.',
                       per_class=[dict(class_id=int(c), name=result.names[int(c)],
                                       precision=float(result.box.p[i]), recall=float(result.box.r[i]),
                                       map50=float(result.box.ap50[i]), map50_95=float(result.box.ap[i]))
                                  for i, c in enumerate(result.box.ap_class_index)])
        return result, metrics

    val_result, val_metrics = measure('val')
    index = int(np.argmax(val_result.box.f1_curve.mean(axis=0)))
    confidence = float(val_result.box.px[index])
    selection['validation_selected_confidence'] = confidence
    selection['confidence_criterion'] = 'Maximum unsmoothed mean class F1 on validation curve; fixed before test.'
    (output/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    (output/'validation-metrics.json').write_text(json.dumps(val_metrics, indent=2), encoding='utf-8')
    print('Selection frozen before test:', json.dumps(selection), flush=True)
    del val_result
    torch.cuda.empty_cache()
    test_result, test_metrics = measure('test')
    curve_precision = np.array([np.interp(confidence, test_result.box.px, row) for row in test_result.box.p_curve])
    curve_recall = np.array([np.interp(confidence, test_result.box.px, row) for row in test_result.box.r_curve])
    test_metrics['validation_fixed_confidence'] = dict(
        confidence=confidence, macro_precision=float(curve_precision.mean()),
        macro_recall=float(curve_recall.mean()),
        method='Approximate values interpolated from test PR curves at the validation-selected confidence; no threshold chosen on test.')
    (output/'test-metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    rows = ['| Class | Precision | Recall | AP50 | AP50-95 |', '|---|---:|---:|---:|---:|']
    for item in test_metrics['per_class']:
        rows.append(f"| {item['name']} | {item['precision']:.1%} | {item['recall']:.1%} | {item['map50']:.1%} | {item['map50_95']:.1%} |")
    report = f"""# Part 1: final detector evaluation

The frozen YOLOv8s checkpoint was evaluated at 1280 pixels after completing 100 training epochs. Validation selected the training checkpoint; test results did not change its weights or configuration.

## Data

Official VisDrone DET training archive: 6,471 images, 343,204 valid boxes. Custom seed-42 split: 4,529 train, 1,295 validation, 647 test. The archive is 1,549,875,511 bytes. Preparation excluded 10,345 ignored/other/score-zero annotations and one zero-height box. Two duplicate image occurrences share their originals' split. Related scene frames are not all grouped; this limits claims of independent-scene generalization. These are custom project scores, not official benchmark scores. The original Colab manifest and checkpoint were not available for an identical baseline comparison.

## Aggregate results

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation | {val_metrics['precision']:.1%} | {val_metrics['recall']:.1%} | {val_metrics['map50']:.1%} | {val_metrics['map50_95']:.1%} |
| Test | {test_metrics['precision']:.1%} | {test_metrics['recall']:.1%} | {test_metrics['map50']:.1%} | {test_metrics['map50_95']:.1%} |

AP uses a confidence floor of .001, NMS IoU .7, and max_det 1000. The precision/recall above follow Ultralytics' split-specific smoothed mean-F1 convention and are class means; .001 is not a deployment threshold. To describe a fixed deployment setting, validation selected confidence {confidence:.3f}; interpolating test curves at that fixed threshold gives approximate class-mean precision {curve_precision.mean():.1%} and recall {curve_recall.mean():.1%}. Increasing confidence can trade recall for precision and does not by itself improve AP.

## Test results by class

{chr(10).join(rows)}

## Evidence

- `selection.json`: frozen checkpoint/configuration, confidence selected on validation, SHA-256 hashes.
- `validation-metrics.json`, `test-metrics.json`: aggregate and per-class results.
- `val/`, `test/`: precision-recall curves, confusion matrices, and labeled/predicted sample batches.
- `../candidate/results.png`, `../candidate/results.csv`: training progress.
- `../candidate/weights/best.pt`: trained model.

## Scope

This completes evaluation of the existing Stage 1 detector. No additional training or baseline superiority is claimed. ByteTrack/video metrics and simulator flight evaluation are separate stages. Further tuning after reviewing this test requires a new independent final evaluation set for an unbiased claim.
"""
    (output/'PART1_RESULTS.md').write_text(report, encoding='utf-8')
    print('Completed:', json.dumps(test_metrics), flush=True)
    print('Report:', output/'PART1_RESULTS.md', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
