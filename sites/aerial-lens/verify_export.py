"""Check exported model predictions against the trained PyTorch network."""
import json
import os
from pathlib import Path

SITE=Path(__file__).resolve().parent
ROOT=SITE.parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'work/yolo-settings')
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

weights=ROOT/'local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt'
image=cv2.imread(str(SITE/'dist/assets/sample-city.jpg'))
h,w=image.shape[:2]
gain=min(1280/w,1280/h)
nw,nh=round(w*gain),round(h*gain)
left,top=round((1280-nw)/2-.1),round((1280-nh)/2-.1)
padded=np.full((1280,1280,3),114,dtype=np.uint8)
padded[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
input=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
model=YOLO(str(weights)).model.eval()
with torch.no_grad():
    native=model(torch.from_numpy(input))[0].numpy()
options=ort.SessionOptions();options.intra_op_num_threads=4
session=ort.InferenceSession(str(weights.with_suffix('.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
exported=session.run(None,{session.get_inputs()[0].name:input})[0]
assert native.shape==exported.shape==(1,14,33600)
score_error=float(np.max(np.abs(native[:,4:]-exported[:,4:])))
assert score_error<.001,score_error
np.testing.assert_allclose(native[:,:4],exported[:,:4],atol=.1,rtol=.001)
report=dict(shape=list(exported.shape),maximum_class_score_error=score_error,
            maximum_coordinate_error_pixels=float(np.max(np.abs(native[:,:4]-exported[:,:4]))),
            passed=True,note='Same RGB float32 input in PyTorch and ONNX CPU. Browser canvas resampling may differ slightly.')
(SITE/'qa-export.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
