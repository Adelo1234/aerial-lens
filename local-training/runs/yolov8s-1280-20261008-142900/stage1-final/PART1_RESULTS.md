# Part 1: final detector evaluation

The frozen YOLOv8s checkpoint was evaluated at 1280 pixels after completing 100 training epochs. Validation selected the training checkpoint; test results did not change its weights or configuration.

## Data

Official VisDrone DET training archive: 6,471 images, 343,204 valid boxes. Custom seed-42 split: 4,529 train, 1,295 validation, 647 test. The archive is 1,549,875,511 bytes. Preparation excluded 10,345 ignored/other/score-zero annotations and one zero-height box. Two duplicate image occurrences share their originals' split. Related scene frames are not all grouped; this limits claims of independent-scene generalization. These are custom project scores, not official benchmark scores. The original Colab manifest and checkpoint were not available for an identical baseline comparison.

## Aggregate results

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation | 66.9% | 54.3% | 58.4% | 36.5% |
| Test | 64.8% | 54.0% | 57.8% | 35.5% |

AP uses a confidence floor of .001, NMS IoU .7, and max_det 1000. The precision/recall above follow Ultralytics' split-specific smoothed mean-F1 convention and are class means; .001 is not a deployment threshold. To describe a fixed deployment setting, validation selected confidence 0.278; interpolating test curves at that fixed threshold gives approximate class-mean precision 65.2% and recall 53.6%. Increasing confidence can trade recall for precision and does not by itself improve AP.

## Test results by class

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| pedestrian | 71.6% | 58.6% | 65.0% | 32.6% |
| people | 61.7% | 33.8% | 39.7% | 16.0% |
| bicycle | 53.4% | 31.7% | 35.4% | 16.5% |
| car | 81.4% | 82.6% | 86.4% | 60.7% |
| van | 65.4% | 62.2% | 65.7% | 48.2% |
| truck | 67.1% | 65.4% | 67.8% | 47.9% |
| tricycle | 51.2% | 40.6% | 41.9% | 24.7% |
| awning-tricycle | 54.7% | 41.3% | 44.4% | 27.6% |
| bus | 78.0% | 69.7% | 75.5% | 53.6% |
| motor | 63.4% | 54.2% | 55.9% | 27.1% |

## Error analysis

Cars achieved the highest test AP50 (86.4%) and recall (82.6%). Buses also performed comparatively well (75.5% AP50). Bicycles (35.4% AP50, 31.7% recall) and people (39.7% AP50, 33.8% recall) remain weaker classes. The confusion matrix shows substantial missed detections for those classes and confusion between vans and cars. Object size has not been isolated using a size-aware evaluator, so these results do not prove a small-object-specific cause. Any new accuracy experiment should be selected on validation, preserving this test as a disclosed already-used evaluation set.

## Evidence

- `selection.json`: frozen checkpoint/configuration, confidence selected on validation, SHA-256 hashes.
- `validation-metrics.json`, `test-metrics.json`: aggregate and per-class results.
- `val/`, `test/`: precision-recall curves, confusion matrices, and labeled/predicted sample batches.
- `../candidate/results.png`, `../candidate/results.csv`: training progress.
- `../candidate/weights/best.pt`: trained model.

## Scope

This completes evaluation of the existing Stage 1 detector. No additional training or baseline superiority is claimed. ByteTrack/video metrics and simulator flight evaluation are separate stages. Further tuning after reviewing this test requires a new independent final evaluation set for an unbiased claim.
