import ast
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path('work/ml-deps').resolve()))
import numpy as np
from ultralytics.cfg import get_cfg
nb=json.loads(Path('outputs/Colab_Part1_Improve_Accuracy.ipynb').read_text())
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code': compile(''.join(c['source']),f'cell{i}','exec')
worker=ast.parse(''.join(nb['cells'][8]['source']))
script=ast.literal_eval(worker.body[2].value) if False else ast.literal_eval(worker.body[0].value)
# Execute only argument/config parsing from the real worker, with no model/GPU operations.
tree=ast.parse(script)
statements=[node for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='config']
env=dict(folder='/tmp/experiment',dataset='data.yaml',size='1280',batch='2',epochs='100',patience='20')
exec(compile(ast.Module(body=statements,type_ignores=[]),'config','exec'),env)
cfg=get_cfg(overrides=env['config'])
assert cfg.imgsz==1280 and cfg.batch==2 and cfg.max_det==1000 and cfg.cos_lr
assert cfg.model is None # YOLO model instantiated separately from yolov8s.pt.
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    fake_box=SimpleNamespace(mp=.6,mr=.5,map50=.5,map=.3,p=np.array([.6]),r=np.array([.5]),
       ap50=np.array([.5]),ap=np.array([.3]),ap_class_index=np.array([0]),
       f1_curve=np.array([[.2,.6,.4]]),px=np.array([.1,.3,.8]),
       p_curve=np.array([[.4,.6,.9]]),r_curve=np.array([[.8,.6,.1]]))
    fake_result=SimpleNamespace(box=fake_box,speed={'inference':1.0},save_dir=root)
    model=SimpleNamespace(names={0:'pedestrian'},val=Mock(return_value=fake_result))
    env=dict(YOLO=lambda _:model,DATASET=root,EXPERIMENT=root,BATCH=2,MAX_DET=1000,
             Path=Path,json=json,np=np,torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    helper=ast.parse(''.join(nb['cells'][6]['source'])).body[0]
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'measure','exec'),env)
    with contextlib.redirect_stdout(io.StringIO()):
        value=env['measure']('saved.pt',1280,'validation')
        test=env['measure']('saved.pt',1280,'test',split='test')
    assert value['deployment_confidence']==.3
    assert model.val.call_args.kwargs['max_det']==1000
    assert model.val.call_args.kwargs['split']=='test'
    assert 'deployment_confidence' not in test
    assert (root/'metrics.json').is_file()
# Use the actual selection expression and verify it preserves baseline when candidate is worse.
tree=ast.parse(''.join(nb['cells'][10]['source']))
assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='selected')
for candidate_score,winner in [(0.2,'baseline'),(.4,'candidate'),(.3,'baseline')]:
    env={'comparisons':[{'label':'baseline','map50_95':.3},{'label':'candidate','map50_95':candidate_score}]}
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'selection','exec'),env)
    assert env['selected']['label']==winner
assert 'identity(original_manifest) == identity(new_manifest)' in ''.join(nb['cells'][3]['source'])
print('PASS: all cells compile; pinned Ultralytics accepts configuration; validation threshold and test separation work; worse/tied candidates preserve baseline; split verification present.')
