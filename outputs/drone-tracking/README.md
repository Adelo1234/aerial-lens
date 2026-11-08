# Autonomous drone target tracking using computer vision

A runnable student project based on the supplied AI3101 assignment: VisDrone data preparation, YOLOv8 detection, ByteTrack tracking, and simulated image-based visual control.

## Start in Google Colab

1. Download `drone-tracking.zip` and `Colab_Training.ipynb` from this chat.
2. Upload the ZIP to Google Drive as `MyDrive/drone-tracking.zip`.
3. Open the notebook through Colab's **File > Upload notebook**.
4. Select **Runtime > Change runtime type > T4 GPU** (or another available GPU).
5. Run the dataset cell. It downloads all 20 official dataset files (~98.45 GB compressed) across Tasks 1–5 and extracts DET training for the YOLO workflow. It checks free disk and reports failures.
6. Run the notebook cells in order. Training, checkpoints, plots, and metrics are saved to Drive.

A trained YOLOv8s checkpoint and completed Part 1 evaluation are available in [the local results bundle](../part1-results/stage1-final/PART1_RESULTS.md). `demo/simulation` is a completed synthetic control demonstration, not a learned detector evaluation. No mAP, IDF1, or drone flight performance is invented.

## Data formats

Dataset source: https://github.com/VisDrone/VisDrone-Dataset

Dataset download scope is **all five tasks**: DET, VID, SOT, MOT, and DroneCrowd. Every dataset Google Drive link in the official README is included (20 files, including all splits, both SOT parts, and challenge initialization). The legacy `drone_project.download` command still downloads DET training alone; use `python -m drone_project.download_all --output /content/visdrone-all` for all tasks.

For downloads only, open `Colab_Download_All_VisDrone.ipynb` in Colab and run all cells on a CPU runtime. It is standalone: no ZIP upload or GPU needed. All archives are retained under `/content/visdrone-all/<task>/<split>.zip`. Their published compressed sizes total ~98.45 GB. Disk is finite and runtime-local files are temporary. Optional extraction requires additional disk, checked against exact ZIP uncompressed sizes. Completed archives have CRC checks and SHA-256 records; reruns reuse them and retry failures. Locally calculated hashes record provenance, not upstream authenticity. The downloader raises an error if any file is incomplete.

The training notebook downloads the same complete collection, but only **DET training** enters the assignment's custom 70/20/10 split. Other tasks and official validation/test datasets stay separate. Downloading SOT/MOT/DroneCrowd does not implement their separate benchmark training/evaluation methods. Challenge ground truth is unavailable; published initialization data is included.

The default input is an **extracted, labeled image pool**:

```text
VisDrone2019-DET-train/
  images/0000001_00001_d_0000001.jpg
  annotations/0000001_00001_d_0000001.txt
```

Use `--format visdrone` for raw comma-separated annotations. Conversion uses
`cx=(x+w/2)/W`, `cy=(y+h/2)/H`, `bw=w/W`, `bh=h/H` after clipping to the image.
Category 0 (ignored regions), category 11 (others), and score-zero entries are omitted.
VisDrone's score field is an annotation flag, not a model confidence prediction.
Categories 1..10 are mapped to 0..9. Truncation and occlusion fields are retained only in the source labels; they do not change the training classes.

For existing YOLO labels use `--format yolo`; they must ALREADY use the ten-class mapping below. Conversion does not guess a different class mapping. Missing label files are errors; empty files explicitly represent background images. Malformed rows stop preparation. Exact duplicate image content is rejected before copying. Conversion statistics are recorded.

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

`prepare` produces a seeded 70/20/10 train/validation/test split as requested in the assignment. This is a custom project split, not the official VisDrone benchmark split. Use only a labeled development pool; keep any official benchmark test set separate. Test images never participate in checkpoint selection.

For video-derived images, supply a CSV:

```csv
image_stem,group_id
0000001_00001_d_0000001,sequence-01
0000001_00002_d_0000001,sequence-01
```

All images from a group stay together. Ratios then apply to the number of groups and may differ from 70/20/10 by image count. Without that CSV, the split is image-level and nearby frames may leak scene information. Review `summary.json` for missing classes and imbalance before training. Exact duplicate checks cannot detect every near-duplicate.

## Local commands

Python 3.10-3.12 is recommended. Run commands from this project's root.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

python -m drone_project.download --output data/raw
python -m drone_project.cli prepare --images /path/to/images --labels /path/to/annotations --format visdrone --output data/visdrone
python -m drone_project.cli train --data data/visdrone/data.yaml --device 0 --epochs 50 --imgsz 960 --batch 8
python -m drone_project.cli evaluate --data data/visdrone/data.yaml --weights runs/yolov8n-visdrone/weights/best.pt --split test --device 0
python -m drone_project.cli track --weights runs/yolov8n-visdrone/weights/best.pt --source /path/to/aerial-video.mp4 --classes 0,1 --device 0
python -m drone_project.cli simulate --output runs/simulation
python -m unittest discover -s tests -v
```

Replace `--device 0` with `--device cpu` for CPU inference/debugging. If GPU memory is exhausted, lower batch size before lowering image size. A larger image size preserves more small-object detail, but costs memory and speed. To resume an interrupted run:

```bash
python -m drone_project.cli train --data data/visdrone/data.yaml --resume runs/yolov8n-visdrone/weights/last.pt
```

Ultralytics restores the checkpoint's original configuration on resume. `prepare`, tracking, and simulation refuse to overwrite existing output directories. Use a new path for another run.

## Detection and tracking behavior

YOLOv8n is the initial speed-oriented baseline. Train YOLOv8s in a separate run if you need an accuracy/speed comparison. Training uses AdamW, horizontal flips, scaling, mosaic, and HSV augmentation. The best checkpoint is chosen using validation metrics; evaluate the test set after final model selection.

Tracking uses Ultralytics' `bytetrack.yaml` and persistent state across frames. At first acquisition, the largest eligible tracked box is selected. `--target-id` can select a known ID explicitly. Once acquired, the project keeps that ID and does not silently switch to another object when it disappears. ByteTrack can still suffer identity switches internally and can give a reappearing object a new ID; no long-term re-identification claim is made. Restart tracking or select the new ID explicitly in that case.

`--classes 0,1` selects VisDrone pedestrian/people; `--classes 3,4,5,8,9` selects common vehicles. These IDs are specific to a VisDrone-trained model. COCO checkpoints use a different mapping.

Outputs include `tracked.mp4`, `commands.csv`, `run.json`, and `performance.json`. The CSV contains normalized x/y errors, bounded yaw and vertical commands, and a lost/tracking state. Recorded-video tracking logs commands but does not alter the video viewpoint. The GIF demonstration separately exercises a closed-loop toy camera plant with analytic measurements. Neither demonstration is a measured ArduPilot flight trial.

## Control and simulator scope

The controller is a proportional image-based visual-servo approximation. It normalizes the target-center offset by half the frame dimensions, applies a deadband, and limits yaw/vertical command magnitudes. Forward velocity is zero because bounding-box center alone does not determine distance. On target loss, all commands are immediately zero.

The built-in simulator models camera pan/vertical movement, a drifting target, and temporary occlusion. It writes a GIF, CSV, and measured error summary. It requires only Pillow. It is useful for checking signs and target-loss behavior, but it is not a Tello EDU or ArduPilot physics simulator. `docs/SITL.md` and `scripts/replay_sitl.py` provide a separate ArduPilot SITL command replay path. A feedback camera renderer would be required to turn that path into a full 3D closed-loop vision simulation.

## Deliverables and results

- `Colab_Training.ipynb`: Drive setup, validation, data preparation, training, test evaluation, video tracking, and checkpoint download.
- `drone_project/`: runnable implementation.
- `tests/`: label, split, identity-lock, and controller tests.
- `demo/simulation/`: generated control demo and actual synthetic measurements.
- `docs/PROJECT_REPORT.md`: assignment-aligned methods and result template.
- `docs/EXPERIMENTS.md`: evaluation protocol and experiment log.
- `docs/SITL.md`: optional ArduPilot setup and command replay.

The requested vision-training skill informed dataset validation and metrics. The Lightning skill informed reproducibility, checkpointing, and logging; YOLO keeps its native Ultralytics trainer. The LLM-training skill was reviewed, but this assignment has no language-model component, so TRL/Unsloth are not dependencies.

## Sources and licenses

- VisDrone original data and terms: https://github.com/VisDrone/VisDrone-Dataset
- Dataset class definitions: https://docs.ultralytics.com/datasets/detect/visdrone/
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- Training API: https://docs.ultralytics.com/modes/train/
- Tracking API: https://docs.ultralytics.com/modes/track/
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- ArduPilot SITL: https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html

Use data under its source terms and record any extra image source/license in your report. Ultralytics is distributed under AGPL-3.0 or its commercial license. The requirements pin the Ultralytics release; `environment.json` records actual torch/CUDA versions and training arguments for each completed run.

