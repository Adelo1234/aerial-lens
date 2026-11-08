"""Bundle the trained ONNX model and pinned browser inference runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
assets = SITE/'dist/assets'
vendor = SITE/'dist/vendor'
assets.mkdir(exist_ok=True)
vendor.mkdir(exist_ok=True)
model = ROOT/'local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.onnx'
data = model.read_bytes()
parts = []
for start in range(0, len(data), 12*1024*1024):
    name = f'model-{len(parts)+1}.bin'
    (assets/name).write_bytes(data[start:start+12*1024*1024])
    parts.append('assets/'+name)
metadata = dict(name='VisDrone YOLOv8s', input=[1,3,1280,1280], output=[1,14,33600],
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), parts=parts,
                source_checkpoint_sha256=hashlib.sha256(model.with_suffix('.pt').read_bytes()).hexdigest())
(assets/'model.json').write_text(json.dumps(metadata, indent=2))
runtime_files = ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
                 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
                 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']
with tarfile.open(SITE/'onnxruntime-web-1.23.2.tgz') as package:
    for name in runtime_files:
        (vendor/name).write_bytes(package.extractfile('package/dist/'+name).read())
    for name in ['LICENSE','ThirdPartyNotices.txt']:
        matches=[member for member in package.getmembers() if member.name == 'package/'+name]
        if matches:
            (vendor/name).write_bytes(package.extractfile(matches[0]).read())
images=sorted((ROOT/'local-training/dataset/images/val').glob('*.jpg'))
samples=[]
for label, index in [('city',0),('traffic',50)]:
    source=images[index]
    shutil.copy2(source,assets/f'sample-{label}.jpg')
    samples.append(dict(name=label, source=source.name, split='validation', repository='https://github.com/VisDrone/VisDrone-Dataset'))
(assets/'samples.json').write_text(json.dumps(samples,indent=2))
print(json.dumps(dict(model_bytes=len(data), parts=len(parts), runtime_assets={p.name:p.stat().st_size for p in vendor.iterdir()}, samples=samples),indent=2))
