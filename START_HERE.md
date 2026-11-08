# START HERE — project team handoff

This is a snapshot of the actual code and measured outputs, prepared for a project teammate to review. No credentials, browser sessions, or Python installation are included. Stage 3 was stopped at the user's request before new Stage 3 implementation changes were made.

## What is finished

- Stage 1: VisDrone annotation preparation, trained YOLOv8s detector, and held-out custom test evaluation. Test precision 64.8%, recall 54.0%, mAP50 57.8%, mAP50-95 35.5% on 647 images. These are class means on a custom split, not official VisDrone benchmark results.
- Website: separate prominent Images and Videos sections. Actual trained model inference runs inside the visitor's browser using WebGPU or WASM CPU fallback. Your PC is not the inference server. Uploaded files stay in the browser.
- Stage 2 browser video detection: up to 60 seconds / 150 MB, sampling at 2/5/10 fps, confidence filtering, timeline/playback, partial results on stop, JSON export and silent sampled-frame WebM export. Website video does not assign tracking IDs.
- Local Stage 2 tracking demonstration: real YOLOv8s + ByteTrack on 180 frames, with annotated MP4 and JSON/CSV logs. Tracking accuracy metrics IDF1/HOTA are not measured; assigned IDs can fragment or switch.

## What is not finished

Stage 3 closed-loop flight simulation is not complete. Existing `control.py`, synthetic camera demo, and SITL replay adapter were prepared earlier and are included, but they are not a validated ArduPilot flight system. The synthetic demonstration uses analytic boxes. SITL replay transports recorded commands without camera feedback. WSL/ArduPilot was not installed at the Stage 3 check. No new Stage 3 work was completed after that check. No further training was started.

## Start with these files

1. `outputs/drone-tracking/drone_project/data.py` — actual annotation filtering, clipping, normalization, validation, and split construction.
2. `outputs/drone-tracking/scripts/train_local_accuracy.py` — the completed local YOLOv8s training workflow.
3. `outputs/drone-tracking/scripts/evaluate_part1.py` — frozen-checkpoint validation/test evaluation and reporting.
4. `local-training/runs/yolov8s-1280-20261008-142900/stage1-final/PART1_RESULTS.md` — final image results and limitations.
5. `sites/aerial-lens/dist/index.html`, `app.js`, `video.html`, `video.js`, `inference-worker.js`, `detection.js` — website interface, video sampling, model loading, inference and class-aware NMS.
6. `outputs/drone-tracking/scripts/stage2_video_demo.py` — local full-frame detector/ByteTrack demonstration.
7. `outputs/stage2-results/STAGE2_RESULTS.md` — video results, verification and limitations.
8. `outputs/drone-tracking/drone_project/control.py` and `docs/SITL.md` — earlier controller and unvalidated SITL integration notes.

## Run the website locally — easiest way to try the model

Use the FULL HANDOFF ZIP, extract it, and install Node.js if necessary. From the extracted `Aerial_Project` folder:

```powershell
cd sites/aerial-lens
node serve.mjs
```

Open http://127.0.0.1:8766. Select Images or Videos. This website needs no Python inference server and no npm dependency installation because the browser runtime/model are bundled. Keep the local Node process running while using that local address. The hosted site works independently of the original PC:

https://aerial-lens-adel.astralbrook256.chatgpt.site

`sites/aerial-lens/.openai/hosting.json` identifies the existing owner's Site. It is not a credential and does not grant deployment rights. Do not replace the owner's hosted site when reviewing locally.

## Set up the Python project

Python 3.12 was used. From the extracted project root, create a fresh environment; the original .venv is deliberately not bundled:

```powershell
python -m venv .venv-training
.venv-training/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-training/Scripts/python.exe -m pip install -r outputs/drone-tracking/requirements.txt
.venv-training/Scripts/python.exe -m pip install onnx==1.19.1 onnxruntime==1.23.2 imageio-ffmpeg==0.6.0
```

The CUDA install is for a compatible NVIDIA setup; CPU-only installs need the appropriate PyTorch CPU distribution. The bundled environment snapshot records exact versions for reference. Training requires dataset restoration first. Existing evaluation output directories are protected against accidental overwrite.

Example image inference after setup (replace the image path):

```powershell
.venv-training/Scripts/yolo.exe predict model=local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt source=your-image.jpg imgsz=1280 device=0
```

Completed `last.pt` is stripped for inference and cannot be resumed as an interrupted optimizer checkpoint. Start a new fine-tuning run from weights if continuing after a completed run. Do not rerun training merely to inspect the project.

## Data and reproducibility

The completed detector used the official DET training archive: 6,471 images and 343,204 retained annotations. The custom seed-42 split is 4,529 train / 1,295 validation / 647 test. Exact duplicates are grouped, but all related scene frames are not grouped. The raw archive is 1,549,875,511 bytes. Downloading the broader VisDrone collection did not mean training on all of it.

Raw images and multi-GB dataset archives are not duplicated into this handoff. `local-training/dataset/manifest.json`, `summary.json`, the group CSV and all download/preparation code ARE included. See NOT_INCLUDED.md for the Drive and official source links. The historical official Colab YOLOv8n checkpoint was not recovered locally, so a controlled identical-setting baseline comparison is not claimed.

The website ONNX export was checked against PyTorch with maximum class-score difference about 0.0000023 and maximum coordinate difference about 0.0124 input pixels on the validation sample. Browser canvas resizing can cause small prediction differences. `tools/export_browser_model.py` reproduces the export command used; it does not retrain the model.

## Code review ZIP versus full ZIP

The CODE REVIEW ZIP contains readable source, notebooks, reports, configuration, logs and data manifests. Binary weights, runtime WASM files, pictures, videos and PDFs are omitted from that small copy. Use the FULL HANDOFF ZIP to run the prebuilt website or load the saved trained model. FILE_MANIFEST.csv in each ZIP lists exactly its included files and SHA-256 hashes.

`work/` contains historical notebook generators, checks and packaging helpers. They show how artifacts were produced; do not run them blindly, because some regenerate earlier output files. Reference assignment PDFs and their original group details are included in the full package. Historical README/status entries may describe earlier incomplete milestones; the stage-specific final reports and this START_HERE file describe the current snapshot.
