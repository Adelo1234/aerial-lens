# Experiment protocol

1. Inspect `summary.json` before training. Confirm class mapping, ignored-box counts, background counts, and class coverage in each split. Visually inspect converted boxes on representative images.
2. Supply sequence groups when frames share a source video. Preserve the manifest and seed for every comparison.
3. Run a one-epoch subset test to verify the GPU and dependency setup. Do not report this as final accuracy.
4. Train a baseline YOLOv8n at 960 pixels. Lower batch size if memory is insufficient. Record the GPU and run duration.
5. Compare one change at a time on validation: YOLOv8s vs YOLOv8n, image size 640 vs 960, or augmentation settings. Keep splits fixed. Native CLI flags expose model, epochs, size, and batch; other training choices are centralized in `train()`.
6. After selecting the configuration, evaluate the held-out test set once. Record aggregate and per-class metrics and discuss uncertainty on sparse classes.
7. Use a separate aerial VIDEO for tracking. VisDrone-DET still images do not supply continuous trajectories for IDF1/HOTA. The current video output demonstrates IDs, but does not measure tracking benchmark scores.
8. For controller evaluation, report normalized center error, occlusion behavior, and the exact simulator model. The provided GIF uses ideal measurements; detection noise and delay must be evaluated separately before extending conclusions.

| Run | Model | Image size | Batch | Seed | Validation mAP50 | Test mAP50 | GPU | Duration |
|---|---|---|---|---|---|---|---|---|
| baseline | YOLOv8n | 960 | 8 | 42 | Pending | Pending | Pending | Pending |

