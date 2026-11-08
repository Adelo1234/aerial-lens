# Aerial Lens: aerial image and video object detection

A bachelor's computer-vision project using a fine-tuned YOLOv8s model on VisDrone, a browser-based image/video demo, and a separate local ByteTrack tracking demonstration.

**Current status:** Stage 1 training and evaluation are complete. Stage 2 browser video detection and a local tracking demonstration are complete. Stage 3 closed-loop drone simulation is incomplete and work on it was stopped.

- [Full explanation of the project](docs/PROJECT_EXPLANATION.md)
- [Final image evaluation](outputs/part1-results/stage1-final/PART1_RESULTS.md)
- [Video and tracking results](outputs/stage2-results/STAGE2_RESULTS.md)
- [Live image demo](https://aerial-lens-adel.astralbrook256.chatgpt.site)
- [Live video demo](https://aerial-lens-adel.astralbrook256.chatgpt.site/video)

## Measured image results

| Split | Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validation | 1,295 | 66.9% | 54.3% | 58.4% | 36.5% |
| Test | 647 | 64.8% | 54.0% | 57.8% | 35.5% |

These are class means on a custom split of the VisDrone DET training pool, not official benchmark scores. Related scene frames were not all grouped, and the test results have already been inspected. See the full explanation for evaluation limitations.

## Run the website locally

The source checkout needs the separately packaged model and WASM files. Download `Aerial_Lens_Runtime_Assets.zip` from this repository's `v1.0.0` release. Put the ZIP outside the checkout, then restore its verified files:

```powershell
python tools/restore_runtime_assets.py ../Aerial_Lens_Runtime_Assets.zip
cd sites/aerial-lens
node serve.mjs
```

Open http://127.0.0.1:8766. Node.js only serves static files. Inference runs on the visitor's device using ONNX Runtime Web, with WebGPU when supported and WASM CPU fallback. There is no Python inference server. Uploaded images and videos stay in the browser. Device performance and codec support vary.

Image mode supports JPG/PNG/WEBP, confidence filtering, class counts, labels, annotated PNG and JSON export. Video mode supports clips up to 60 seconds / 150 MB, analyzes at 2/5/10 sampled frames per second, and exports per-frame JSON or a silent annotated sampled-frame WebM. Browser video mode does not assign tracking IDs.

## Python implementation

Python 3.12 was used. Create a fresh environment; install an appropriate PyTorch build for your platform, then the project requirements:

```powershell
python -m venv .venv-training
.venv-training/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0
.venv-training/Scripts/python.exe -m pip install -r outputs/drone-tracking/requirements.txt
.venv-training/Scripts/python.exe -m pip install onnx==1.19.1 onnxruntime==1.23.2 imageio-ffmpeg==0.6.0
```

Choose the appropriate CUDA wheel index from PyTorch's installation instructions if using accelerated training/inference. The complete historical environment is recorded in `requirements-environment-current.txt`.

After restoring runtime assets, example image inference:

```powershell
.venv-training/Scripts/yolo.exe predict model=local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt source=your-image.jpg imgsz=1280
```

Source entry points:

| File | Purpose |
|---|---|
| `outputs/drone-tracking/drone_project/data.py` | Label conversion, validation, clipping and splitting |
| `outputs/drone-tracking/drone_project/download_all.py` | Catalog and downloader for the broader VisDrone collection |
| `outputs/drone-tracking/scripts/train_local_accuracy.py` | Completed YOLOv8s training configuration |
| `outputs/drone-tracking/scripts/evaluate_part1.py` | Frozen-model validation/test evaluation |
| `outputs/drone-tracking/scripts/stage2_video_demo.py` | Full-frame local YOLOv8s + ByteTrack demo |
| `sites/aerial-lens/dist/` | Image/video UI, inference worker and postprocessing |
| `outputs/drone-tracking/drone_project/control.py` | Earlier proportional controller; not validated flight control |
| `tools/export_browser_model.py` | Reproduce ONNX export without retraining |

Training requires restoring the external DET archive to `local-training/raw/DET/train.zip`. The local tracking demo currently references a MOT archive under the original author's Downloads directory: change that path before rerunning it on another computer. Dataset and evaluation metadata preserve historical paths; regenerate local dataset paths for a new training run. Completed, stripped `last.pt` cannot restore an interrupted optimizer state.

Existing checks, from the repository root:

```powershell
cd outputs/drone-tracking
python -m unittest discover -s tests -v
cd ../..
node sites/aerial-lens/check-detection.mjs
node sites/aerial-lens/check-video.mjs
```

## Source, assets and historical files

Source code, notebooks, dataset manifests, training logs, evaluation metrics and tracking logs are in Git. Binary models, browser WASM runtime, sample images, plots and demo videos are in the versioned runtime ZIP. Its SHA-256 is recorded in `runtime-assets.json`; the restore helper checks both the archive and individual files.

Raw datasets, virtual environments, credentials, private publishing state and assignment PDFs are not part of this repository. Downloading the broader VisDrone collection did not mean training on all of it. The completed detector used 6,471 images from DET train.

`work/`, the older READMEs, status reports and notebooks preserve historical workflows. They can describe earlier milestones and contain original paths. Read this README and the final stage reports first; do not run historical artifact generators blindly. `FILE_MANIFEST.csv` from the earlier ZIP handoff is omitted because this repository adds documentation and uses its own Git history.

## Licensing and attribution

The included application license is AGPL-3.0, matching the existing website distribution. Ultralytics code/models remain subject to their applicable AGPL-3.0 or commercial terms. ONNX Runtime is MIT licensed; bundled notices accompany its source/runtime. VisDrone data and samples remain subject to the dataset's terms; this repository does not relicense the dataset. See [third-party notes](THIRD_PARTY_NOTICES.md).
