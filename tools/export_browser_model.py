"""Reproduce the browser-model export; no training is performed."""
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
