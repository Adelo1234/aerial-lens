import os
from pathlib import Path
import sys
import json
import csv

root = Path(__file__).resolve().parent.parent
(root/'work/yolo-settings').mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(root/'work/yolo-settings')
sys.path.insert(0,str(root/'work/ml-deps'))
project = root/'outputs/drone-tracking'
sys.path.insert(0,str(project))
from PIL import Image
import numpy as np
import torch
import cv2
from drone_project.cli import main

torch.set_num_threads(2)
folder = root/'work/ml-smoke-complete'
folder.mkdir(exist_ok=True)
images,labels = folder/'images',folder/'annotations'
images.mkdir(exist_ok=True); labels.mkdir(exist_ok=True)
for i in range(20):
    rng = np.random.default_rng(i)
    pixels = rng.integers(0,180,(64,64,3),dtype=np.uint8)
    pixels[20:40,20:40] = [240,80,40]
    Image.fromarray(pixels).save(images/f'{i}.png')
    (labels/f'{i}.txt').write_text('20,20,20,20,1,4,0,0')

def run(*args):
    sys.argv = ['drone_project',*map(str,args)]
    main()

run('prepare','--images',images,'--labels',labels,'--output',folder/'data')
run('train','--data',folder/'data/data.yaml','--model','yolov8n.yaml',
    '--epochs',1,'--imgsz',64,'--batch',4,'--device','cpu','--workers',0,
    '--output',folder/'runs','--name','synthetic-smoke')
best = folder/'runs/synthetic-smoke/weights/best.pt'
assert best.is_file()
run('evaluate','--data',folder/'data/data.yaml','--weights',best,
    '--device','cpu','--imgsz',64,'--batch',2,'--output',folder/'runs')
video = folder/'synthetic.mp4'
writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
assert writer.isOpened()
for i in range(12):
    frame = np.zeros((64,64,3),dtype=np.uint8)
    cv2.rectangle(frame,(10+i,20),(30+i,40),(40,80,240),-1)
    writer.write(frame)
writer.release()
run('track','--weights',best,'--source',video,'--classes','3',
    '--device','cpu','--imgsz',64,'--output',folder/'tracking')
with (folder/'tracking/commands.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows)==12
assert (folder/'tracking/tracked.mp4').stat().st_size>0
summary = dict(scope='CPU integration smoke test on synthetic images, not VisDrone accuracy.',
               trained_epochs=1,train_images=14,val_images=4,test_images=2,
               training_checkpoint_created=True,evaluation_json_created=True,
               tracking_frames=len(rows),note='Randomly initialized YOLOv8n, 64px; no useful accuracy claim.')
(project/'demo/ml-smoke-verification.json').write_text(json.dumps(summary,indent=2))
print('SMOKE_VERIFIED',json.dumps(summary))
