"""Create complete runnable and lightweight code-review snapshots for the team."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
META=OUT/'team-handoff'
META.mkdir(exist_ok=True)
files={}


def add(path,relative=None):
    path=Path(path)
    if path.is_file():files[str(relative or path.relative_to(ROOT)).replace('\\','/')]=path


def tree(folder):
    for path in (ROOT/folder).rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in {'.git','node_modules','__pycache__','.venv','.venv-training'} for part in relative.parts):continue
        if path.name.startswith('.env') or path.suffix in {'.pyc','.tgz'}:continue
        if '.openai' in relative.parts and path.name!='hosting.json':continue
        add(path)


for folder in ['outputs/drone-tracking','sites/aerial-lens','outputs/part1-results','outputs/stage2-results','local-training/stage2']:
    tree(folder)
run=ROOT/'local-training/runs/yolov8s-1280-20261008-142900'
excluded_checkpoints=[]
for path in run.rglob('*'):
    if not path.is_file():continue
    if path.name.startswith('epoch') and path.suffix=='.pt':
        excluded_checkpoints.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size));continue
    add(path)
for path in (ROOT/'work').glob('*.py'):add(path)
for path in OUT.glob('*.ipynb'):add(path)
for name in ['PART1_STATUS.md','DOWNLOAD_STATUS.md','AerialLens_Deployment.json','AerialLens_Stage2_Deployment.json']:
    add(OUT/name)
for path in OUT.glob('AerialLens*.png'):add(path)
add(OUT/'Stage2-video-live.png')
for name in ['README.md','environment-lock.txt','training.stdout.log','training.stderr.log','stage1-evaluation.log','stage2-demo.log','active-run.json','det-image-groups.csv']:
    add(ROOT/'local-training'/name)
for name in ['data.yaml','manifest.json','summary.json']:
    add(ROOT/'local-training/dataset'/name)
sys.path.insert(0,str(ROOT/'outputs/drone-tracking'))
from drone_project.data import raw_to_yolo
examples=json.loads((ROOT/'local-training/dataset/manifest.json').read_text())[:3]
example_notes=[]
for record in examples:
    stem=Path(record['image']).stem
    raw=ROOT/'local-training/extracted/DET/train/VisDrone2019-DET-train/annotations'/f'{stem}.txt'
    converted=ROOT/'local-training/dataset/labels'/record['split']/f'{stem}.txt'
    rows,_=raw_to_yolo(raw.read_text(encoding='utf-8-sig'),record['width'],record['height'])
    assert '\n'.join(rows).strip()==converted.read_text().strip(),stem
    add(raw,f'data-examples/visdrone/{stem}.txt')
    add(converted,f'data-examples/yolo/{stem}.txt')
    add(ROOT/'local-training/dataset/images'/record['split']/record['image'],f'data-examples/images/{record["image"]}')
    example_notes.append(dict(image=record['image'],width=record['width'],height=record['height'],split=record['split'],objects=record['objects']))
(META/'data-examples.json').write_text(json.dumps(example_notes,indent=2))
add(META/'data-examples.json','data-examples/examples.json')
for path in Path('C:/Users/A/Downloads').glob('Assignment*.pdf'):
    if path.name=='Assignment3.pdf' or path.name.startswith('Assignment2 '):add(path,'reference-documents/'+path.name)

start='''# START HERE — project team handoff

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
'''
(META/'START_HERE.md').write_text(start,encoding='utf-8')
not_included=f'''# Deliberately separate from this handoff

- Raw VisDrone image folders and downloaded dataset ZIPs: use the existing Drive folders or the official downloader. They are multi-GB and are external data dependencies, not source code.
- Installed Python/Node packages, virtual environments, caches and bytecode: recreate from requirements and package/version records.
- Git history, short-lived publishing credentials, .env files, private Sites helper state, and Google sign-in screenshots: not required for code review and not shared.
- {len(excluded_checkpoints)} periodic optimizer checkpoints (`epoch*.pt`), totaling {sum(p['bytes'] for p in excluded_checkpoints)/1e9:.2f} GB. Best/last trained weights, ONNX export, complete 100-epoch CSV history, plots, evaluation and training logs ARE included in the full package.
- Older generated ZIP/tar copies of the same source: files are included directly instead of nesting redundant archives.

Official dataset repository: https://github.com/VisDrone/VisDrone-Dataset
Existing VisDrone-all Drive folder: https://drive.google.com/drive/folders/1BVsiZ3oe8pUm629wVJCWcDSaudz1cq4O
Original assignment data folder: https://drive.google.com/drive/folders/1jZFmpBKzvGpA6oFmQ0FcX3duB6rcY9gz
Drive permissions remain unchanged; a teammate may need the owner to grant access.

Raw DET download helper: work/download_local_det.py (requires gdown in the active Python environment). All-task catalog/downloader: outputs/drone-tracking/drone_project/download_all.py.
The local Stage 2 demo script references the original MOT ZIP under C:/Users/A/Downloads; change that source path on a teammate's computer. It does not need all 98 GB of the broader collection for the six-second demo.
'''
(META/'NOT_INCLUDED.md').write_text(not_included,encoding='utf-8')
(META/'EXCLUDED_CHECKPOINTS.json').write_text(json.dumps(excluded_checkpoints,indent=2))
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(META/'requirements-environment-current.txt').write_text(freeze,encoding='utf-8')
export='''"""Reproduce the browser-model export; no training is performed."""
import functools
import os
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(root/'work/yolo-settings')
os.environ['YOLO_AUTOINSTALL']='false'
from ultralytics import YOLO
torch.onnx.export=functools.partial(torch.onnx.export,dynamo=False)
weights=root/'local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt'
YOLO(str(weights)).export(format='onnx',imgsz=1280,batch=1,opset=17,simplify=False,dynamic=False,half=False,device='cpu')
print('Next: run sites/aerial-lens/prepare_assets.py to rebuild the existing browser chunks.')
'''
(META/'export_browser_model.py').write_text(export,encoding='utf-8')
compile(export,'export_browser_model.py','exec')
for name in ['START_HERE.md','NOT_INCLUDED.md','EXCLUDED_CHECKPOINTS.json','requirements-environment-current.txt']:
    add(META/name,name)
add(META/'export_browser_model.py','tools/export_browser_model.py')

text_suffixes={'.py','.js','.mjs','.html','.css','.md','.txt','.json','.yaml','.yml','.csv','.ipynb','.cmd','.log'}
full=dict(sorted(files.items()))
review={name:path for name,path in full.items() if path.suffix.lower() in text_suffixes or path.name in {'.gitignore','LICENSE','NOTICE'}}
for label,items in [('Full_Handoff',full),('Code_Review',review)]:
    inventory=io.StringIO();writer=csv.writer(inventory);writer.writerow(['path','bytes','sha256'])
    digests={}
    for name,path in items.items():
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        digests[name]=sha;writer.writerow([name,path.stat().st_size,sha])
    target=OUT/f'Aerial_Project_{label}.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in items.items():archive.write(path,'Aerial_Project/'+name)
        archive.writestr('Aerial_Project/FILE_MANIFEST.csv',inventory.getvalue())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,sha in digests.items():
            with archive.open('Aerial_Project/'+name) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            assert actual==sha,name
    print(json.dumps(dict(archive=str(target),files=len(items)+1,bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest())),flush=True)
