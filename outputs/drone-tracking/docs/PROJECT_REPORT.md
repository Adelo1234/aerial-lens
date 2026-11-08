# Autonomous drone target tracking using computer vision

AI3101 project implementation notes. Add your group details and real experimental results before submitting.

## Objective

Detect people and vehicles in aerial imagery, associate detections across video frames, and derive simulated camera-motion commands from the selected target's image position. The pipeline combines YOLOv8, ByteTrack, and proportional image-based visual servoing.

## Dataset preparation

The project obtains the official VisDrone-DET training archive from the download link published in the VisDrone dataset repository. Raw annotations express top-left coordinates and box width/height in pixels. Preparation filters ignored/other categories and score-zero records, clips out-of-frame boxes, normalizes center coordinates and dimensions, and maps ten classes to zero-based IDs. It validates label geometry and keeps exact duplicate images in the same split using SHA-256 grouping. A manifest records the source image hash, group, image dimensions, split, and retained object count. Download provenance includes the repository/archive URL, archive size, and a locally calculated SHA-256 digest.

The assignment's approximate 70/20/10 split uses seed 42: 4,529 training, 1,295 validation, and 647 test images. In the completed run, grouping covers exact duplicates, not every related sequence frame. Related scenes may cross splits. This is a custom project split, not the official VisDrone benchmark protocol.

## Detector training

The baseline is pretrained YOLOv8n, fine-tuned on the prepared data. Initial configuration: 50 epochs, image size 960, batch size 8, AdamW, initial learning rate 0.001, weight decay 0.0005, early-stopping patience 15. Augmentation includes mosaic, scaling, horizontal flips, and HSV changes. Mosaic is closed during the last ten epochs. Values are initial experiment settings, not evidence that they are optimal.

Ultralytics saves the best and last checkpoints, training curves, configuration, and validation artifacts. The project records library/CUDA versions and actual training arguments. Validation guides model selection; the test set is reserved for final evaluation.

The completed candidate fine-tuned pretrained YOLOv8s for 100 epochs at 1280 pixels, batch 4, AdamW, cosine learning-rate decay, and patience 20. Mosaic was disabled during the final ten epochs. Final evaluation used batch 2, confidence floor .001, NMS IoU .7, and max_det 1000. The Colab baseline was not recovered for a controlled comparison.

## Temporal association

ByteTrack associates detections across frames and assigns track IDs. The application locks to the largest eligible tracked box on acquisition or to an explicitly selected ID. If that ID disappears, motion commands become zero. It resumes if the same ID reappears; it does not automatically choose another object. Internal tracker identity switches remain possible and require annotated video evaluation to quantify.

## Visual control

The target-center error is normalized relative to half the frame width and height. A proportional controller converts horizontal error to bounded yaw rate and vertical error to bounded upward/downward movement in a toy camera model. A small deadband reduces tiny corrections. Forward velocity stays zero because this implementation does not estimate depth. This is a simplified IBVS approximation, not a calibrated interaction-matrix or full aircraft controller.

## Simulation

The supplied synthetic camera simulator uses analytic target boxes and temporary occlusion. It validates controller signs, convergence, and zero commands on loss. It does not test learned detection or ByteTrack. ArduPilot SITL replay is provided as a separate integration path; a synchronized simulator camera is still needed for a complete closed-loop 3D vision test.

## Results status

Part 1 detector training and final custom held-out test evaluation are complete. The frozen YOLOv8s checkpoint achieved test precision 64.8%, recall 54.0%, mAP50 57.8%, and mAP50-95 35.5% on 647 images. These precision and recall values follow the Ultralytics class-mean reporting convention. Validation at the same settings achieved precision 66.9%, recall 54.3%, mAP50 58.4%, and mAP50-95 36.5%.

Validation selected a confidence threshold of approximately .278. At that fixed threshold, interpolated test curves indicate approximate class-mean precision 65.2% and recall 53.6%. The metric confidence floor is not the deployment threshold.

See [the final Part 1 report](../../part1-results/stage1-final/PART1_RESULTS.md) for the full protocol, per-class metrics, limitations, and evidence. [The results bundle](../../part1-results/) includes the trained checkpoint, training curves, confusion matrices, precision-recall curves, and labeled/predicted samples.

The split groups exact duplicates but not all related scene frames. No statistically controlled improvement over the Colab baseline or official benchmark result is claimed. Tracking IDF1/HOTA and real closed-loop SITL flight metrics have not been evaluated. The toy simulation is separate from learned detector performance.

## References

1. VisDrone: https://github.com/VisDrone/VisDrone-Dataset
2. Ultralytics YOLOv8: https://docs.ultralytics.com/models/yolov8/
3. ByteTrack: https://github.com/FoundationVision/ByteTrack
4. ArduPilot SITL: https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html

