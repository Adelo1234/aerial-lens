import base64
import io
import json
import zipfile
from pathlib import Path

root = Path('outputs/drone-tracking')
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in list((root/'drone_project').glob('*.py')) + list((root/'tests').glob('*.py')) + [root/'requirements.txt']:
        z.write(path, path.relative_to(root).as_posix())
payload = base64.b64encode(buf.getvalue()).decode()
nb = json.loads((root/'Colab_Training.ipynb').read_text())
nb['cells'][0]['source'] = ['# Part 1: VisDrone detection training\n', '\n', 'Uses the existing DET training archive in your Drive and the assignment custom 70/20/10 split. YOLOv8n, 50 epochs, 960 pixel images, batch 8. Checkpoints and measured test results persist in MyDrive/aerial-tracking-project. Select a free T4 GPU runtime and run all. Project code is bundled below; no upload is required.\n']
nb['cells'][0]['source'][2] = nb['cells'][0]['source'][2].replace('custom 70/20/10 split', 'custom approximately 70/20/10 split, keeping identical images together')
setup = f'''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys, subprocess, shutil, json, base64, io, zipfile
PROJECT = Path('/content/drone-tracking')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload}'))) as bundle:
    bundle.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
print('GPU:', torch.cuda.get_device_name(0))
'''
nb['cells'][1]['source'] = setup.splitlines(True)
nb['cells'][2]['source'] = ['## Prepare Part 1\n', 'Extracts only DET training to runtime disk. Converts valid VisDrone labels to normalized YOLO coordinates and prepares the assignment custom 70/20/10 split with seed 42. Official DET validation and test archives stay separate. These are custom project scores, not official benchmark scores.\n']
nb['cells'][2]['source'][1] += 'Exact duplicate images share a split; related sequence frames are not automatically grouped. Invalid zero-size boxes and ignored annotations are counted and filtered.\n'
data = ''.join(nb['cells'][3]['source'])
data = data.replace("# Update from the account's Drive storage page before a later fresh download.\nALL_DOWNLOADS = download_all(RAW, extract=False, drive_free_gb=188.85)\n", "assert (RAW/'DET/train.zip').is_file(), 'DET training archive must exist in Drive.'\n")
data = data.replace('All five tasks downloaded; DET training only used in YOLO workflow.', 'Part 1 only: DET training archive used in YOLO workflow.')
data = data.replace('GROUPS_CSV = None', '''import csv, hashlib
GROUPS_CSV = Path('/content/det-image-groups.csv')
with GROUPS_CSV.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['image_stem', 'group_id'])
    for image in sorted(IMAGES.glob('*.jpg')):
        writer.writerow([image.stem, hashlib.sha256(image.read_bytes()).hexdigest()])
print('Exact duplicate images will stay in the same split.')''')
data = data.replace("subprocess.run([sys.executable, '-m', 'drone_project.cli', *map(str, args)], check=True)", """with subprocess.Popen([sys.executable, '-u', '-m', 'drone_project.cli', *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'Project command failed: {args[0]}')""")
data = data.replace('Custom 70/20/10 from official DET training only', 'Custom approximately 70/20/10 from DET training; identical images grouped')
nb['cells'][3]['source'] = data.splitlines(True)
checks = ''.join(nb['cells'][5]['source'])
checks = checks.replace("subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)", "checks = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], capture_output=True, text=True)\nprint(checks.stdout + checks.stderr)\nchecks.check_returncode()")
nb['cells'][5]['source'] = checks.splitlines(True)
# Limit this notebook to detection preparation, training, resumption, and evaluation.
nb['cells'] = nb['cells'][:14]
nb['metadata']['colab']['name'] = 'Colab_Part1_Training.ipynb'
nb['metadata']['accelerator'] = 'GPU'
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell {i}', 'exec')
target = Path('outputs/Colab_Part1_Training.ipynb')
target.write_text(json.dumps(nb, indent=2), encoding='utf-8')
print(target, target.stat().st_size, 'bytes; code syntax verified')

