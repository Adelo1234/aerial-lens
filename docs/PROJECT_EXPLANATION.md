# Full project explanation

## 1. Purpose and current status

The project is an aerial-vision system intended to detect objects, track a selected target, and eventually use its position to guide a simulated drone. We developed it in stages so that each component could be checked separately.

Stage 1 is complete: we prepared VisDrone image data, fine-tuned a detector and evaluated the frozen model. Stage 2 includes a working public image/video website and a separate local tracking demonstration. Stage 3 remains incomplete. Work on further drone-control integration was stopped at the owner's request.

## 2. The data we used

Our source is the official VisDrone dataset repository. Its broader download catalog covers five tasks:

| Task | Content and purpose | Published approximate compressed size |
|---|---|---:|
| DET | Individual images and object boxes for detection | 2.07 GB |
| VID | Video sequences and boxes for video detection | 13.86 GB |
| SOT | Sequences for following one designated object | 67.64 GB |
| MOT | Sequences for tracking multiple objects with identities | 13.85 GB |
| DroneCrowd | Crowd imagery and counting annotations | 1.03 GB |
| Total | 20 cataloged archives | 98.45 GB |

Those are approximate published archive sizes, not measured extracted storage. The earlier Drive downloader recorded 15 of 20 archives, about 45.4 GB; some downloads failed because of quotas. We did not train the detector on that entire downloaded collection.

The completed detector used the DET training archive: 1,549,875,511 bytes, approximately 1.55 decimal GB, containing 6,471 images. Preparation retained 343,204 valid bounding boxes before Ultralytics' duplicate-label cleanup. The prepared dataset occupied approximately 1.57 GB. Additional archives serve different tasks and are not automatically additional image-training data.

The ten detection classes are pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus and motor. `Pedestrian` and `people` are distinct source categories; `motor` denotes the dataset's motor category.

## 3. Cleaning and converting annotations

A raw VisDrone annotation stores left and top pixel coordinates, width, height, a score flag, category, truncation and occlusion. The score field is an annotation flag, not a model-generated confidence value.

Our preparation workflow:

1. Checked ZIP integrity and recorded SHA-256 provenance. These checks detect corruption and changes; they do not prove upstream authenticity.
2. Verified images could be loaded and annotation files existed. Malformed rows cause an explicit failure instead of silently disappearing.
3. Excluded ignored regions (category 0), the unsupported `others` category (11), and score-zero entries: 10,345 excluded annotations collectively.
4. Removed one zero-height box and clipped remaining boxes to image boundaries. Zero-area or fully outside boxes are not retained.
5. Converted category IDs 1–10 to YOLO's zero-based 0–9 IDs.
6. Converted clipped pixel boxes into normalized YOLO center coordinates and dimensions.

For clipped corners `(x1, y1, x2, y2)` in an image of width W and height H:

```text
center_x = (x1 + x2) / (2W)
center_y = (y1 + y2) / (2H)
width    = (x2 - x1) / W
height   = (y2 - y1) / H
```

The resulting coordinates lie within the image's normalized coordinate range. Raw data was preserved, with converted labels, manifests and summaries written separately. Truncation and occlusion were not turned into extra classes.

## 4. Dataset splitting and leakage limits

We used a reproducible seed-42 custom split:

| Subset | Images | Role |
|---|---:|---|
| Train | 4,529 | Update model weights |
| Validation | 1,295 | Select the checkpoint and deployment confidence |
| Test | 647 | Evaluate the frozen selected model |

SHA-256 grouping found two duplicate image occurrences and kept each with its original in the same subset. These copies were grouped rather than removed. All related scene frames and near duplicates were not grouped, so the split does not establish performance on fully independent scenes.

This custom split comes from DET train. It is not the official VisDrone benchmark split. The test set has now been inspected; future tuning should use validation, followed by a fresh independent final evaluation for an unbiased new performance claim.

## 5. The detector and training procedure

YOLOv8 predicts object boxes and classes in one inference pass. We fine-tuned pretrained weights rather than training every visual feature from scratch. The native Ultralytics trainer handled learning, checkpointing and validation. PyTorch Lightning and language-model training tools were reviewed earlier but are not the implemented training engine; no LLM was trained.

The initial experiment used YOLOv8n for 50 epochs at 960-pixel input resolution. Its historical validation results were about 52.1% precision, 41.9% recall, 42.7% mAP50 and 25.7% mAP50-95. The original checkpoint and split manifest were not recovered locally, so those historical figures are not a controlled identical-data comparison.

The completed candidate used the larger YOLOv8s model for 100 epochs at 1280-pixel input resolution, batch size 4, and seed 42. AdamW used an initial learning rate of 0.001, cosine decay, weight decay 0.0005 and three warmup epochs. Mixed precision was enabled. Periodic checkpoints, best/last weights, training arguments, logs and metric plots were saved.

Training augmentation included horizontal flips, translation, scaling, HSV changes and mosaic. Mosaic was disabled for the last ten epochs. These are training-time transformations, not extra permanently stored images. Higher input resolution was intended to preserve more object detail, but a size-specific evaluation would be needed to establish a small-object-specific improvement.

The best checkpoint was selected using validation fitness, then frozen before the final test evaluation.

## 6. Results and their meaning

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation | 66.9% | 54.3% | 58.4% | 36.5% |
| Test | 64.8% | 54.0% | 57.8% | 35.5% |

Precision describes how often accepted predictions are correct. Recall describes how many labeled objects are found. The reported precision and recall are class means at Ultralytics' split-specific smoothed mean-F1 operating point; they are not pooled counts of all predictions or a claim that every uploaded image has 64.8% accuracy.

IoU measures the overlap between predicted and labeled boxes. mAP50 averages class average precision using IoU 0.50. mAP50-95 also requires progressively tighter box overlap, averaging thresholds 0.50 through 0.95. Its lower value shows that accurate localization remains a challenge.

Cars performed best, with 86.4% AP50 and 82.6% recall. Buses reached 75.5% AP50. Bicycles and the `people` class were weaker at 35.4% and 39.7% AP50 respectively. There were missed objects and confusion between vans and cars. We have not isolated all errors by object size, altitude, weather or camera angle.

Validation selected confidence approximately 0.278, rounded to 0.28 in the website. A confidence threshold filters predictions; raising it can reduce false positives while also hiding real objects. It does not retrain the model or automatically improve AP. A displayed box confidence is not the model's overall measured accuracy.

## 7. Exporting the model for the website

We exported the selected PyTorch checkpoint to ONNX, using float32, opset 17 and fixed input shape `[1, 3, 1280, 1280]`. The ONNX model is approximately 45.2 MB. Four checked chunks distribute it to the browser.

On one validation sample, PyTorch-versus-ONNX checking found maximum class-score difference of approximately 0.0000023 and maximum coordinate difference of 0.0124 input pixels. This supports numerical agreement for the checked input; it is not a full browser accuracy benchmark.

Browser preprocessing preserves aspect ratio with gray letterboxing, converts pixels to RGB float32 values divided by 255, and uses channel-first input. Postprocessing decodes boxes and classes and applies class-aware non-maximum suppression at IoU 0.7, capped at 1,000 detections. NMS removes redundant overlapping predictions of the same class.

## 8. Aerial Lens image website

The published site has separate Images and Videos workspaces. Image mode supports uploads, drag-and-drop, sample images, labeled boxes, confidence filtering, class counts and annotated PNG/JSON downloads.

The host serves static HTML, CSS, JavaScript, model files and runtime assets. ONNX Runtime Web performs inference inside the visitor's browser using WebGPU when available, with a WASM CPU fallback. Each visitor uses their own device's resources. The original owner's computer does not need to stay on for the published site.

Uploaded media is processed locally in the browser rather than sent to an inference server. The model itself is downloaded to the visitor, so the published model files can be retrieved by visitors. Initial download, memory use and inference speed vary across phones and computers.

Actual file upload, model inference, confidence changes, label toggling, empty results and exports were checked. A narrow mobile layout was inspected. The implemented CPU fallback was not separately exercised in the browser QA, so compatibility across all devices is not claimed.

## 9. Stage 2 browser video detection

The website accepts supported video formats up to 60 seconds and 150 MB, with an approximately 4K resolution cap. Users select analysis rates of 2, 5 or 10 sampled frames per second. Each sampled frame uses the same trained detector. Playback and scrubbing show boxes from the nearest analyzed frame, which can drift between samples.

Users can stop analysis and retain partial results, change confidence, inspect per-frame counts, export timestamped JSON and export a silent annotated WebM containing sampled frames. The exported clip does not retain original audio, and browser recording/seek timing can slightly affect duration.

A six-second sample produced 30 analyzed frames at 5 fps. Real disk upload, partial cancellation, JSON export and a decoded 30-frame WebM export were verified. These functional checks do not measure video detection accuracy. Browser counts are repeated per-frame detections, not unique people or vehicles. Browser tracking IDs are not implemented.

## 10. Separate local ByteTrack demonstration

The local pipeline runs YOLOv8s and ByteTrack on every frame. ByteTrack associates detections over time using motion prediction and matching, including lower-confidence detections that can help continue tracks. Assigned IDs can still switch or fragment after occlusion or missed detections.

We processed 180 frames from a VisDrone MOT training sequence. The demonstration produced 11,483 displayed detection instances across frames and 318 distinct assigned track IDs. Neither number is a ground-truth count of unique objects.

End-to-end processing took approximately 17.78 seconds, or 10.12 fps including tracking and encoding. The six-second output plays at 30 fps; that does not establish 30 fps real-time processing or original capture timing. Outputs include an annotated MP4 and JSON/CSV logs.

This is a functional training-sequence demonstration. IDF1, HOTA and held-out video detection AP were not measured. Stage 1 image metrics must not be presented as video tracking metrics.

## 11. Stage 3 and the earlier control components

The earlier controller computes the selected target's center offset from the image center, normalizes it, applies a deadband and produces bounded proportional yaw and vertical commands. Forward velocity is zero because box-center position alone does not determine target distance. Target loss produces zero commands. A target-lock component avoids deliberately switching to another ID when the selected target disappears, although tracker identity errors remain possible.

A toy camera simulation exercised controller signs, target motion and temporary occlusion using analytic boxes. It did not test the learned detector. An optional SITL adapter can replay recorded commands, but replay is not closed-loop camera feedback. These components were not validated as autonomous flight.

Further Stage 3 work was stopped at the owner's request. No completed full drone-flight simulation, real drone trial or new Stage 3 website section is claimed.

## 12. Deliverables and remaining work

We produced Python source, Colab notebooks, downloader and preparation tools, split manifests, trained weights, ONNX exports, training logs, evaluation plots, a hosted image/video website, tracking demonstration videos and per-frame logs. Earlier full and lightweight ZIP handoffs were prepared for teammate review. This repository adds a central explanation and source history; separately versioned runtime assets restore the binary dependencies without committing datasets.

Remaining work includes collecting independent evaluation scenes, analyzing weak classes, selecting new accuracy experiments on validation, evaluating tracking identity quality, adding browser tracking if needed, and implementing and validating closed-loop simulation when Stage 3 is resumed. No additional training or Stage 3 implementation was performed while preparing this explanation and repository.
