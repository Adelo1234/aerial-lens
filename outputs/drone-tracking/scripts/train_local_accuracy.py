"""Run the fixed Part 1 accuracy candidate on the local NVIDIA GPU."""
import csv
import hashlib
import json
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO
    from drone_project.data import prepare
    from drone_project.download_all import extract_one

    assert torch.cuda.is_available(), 'CUDA GPU is required.'
    # Exercise the actual CUDA kernels before committing to a long run.
    probe = torch.randn(64, 64, device='cuda')
    (probe @ probe).sum().item()
    print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__, flush=True)
    local = ROOT/'local-training'
    raw = extract_one(local/'raw/DET/train.zip', local/'extracted/DET/train')
    source = raw/'VisDrone2019-DET-train'
    images, labels = source/'images', source/'annotations'
    assert images.is_dir() and labels.is_dir(), 'Unexpected DET archive layout.'
    groups = local/'det-image-groups.csv'
    with groups.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_stem', 'group_id'])
        for path in sorted(images.glob('*.jpg')):
            writer.writerow([path.stem, hashlib.sha256(path.read_bytes()).hexdigest()])
    dataset = local/'dataset'
    if not dataset.exists():
        summary = prepare(images, labels, dataset, label_format='visdrone', seed=42, groups_csv=groups)
    else:
        summary = json.loads((dataset/'summary.json').read_text())
    assert summary['images_by_split'] == {'train': 4529, 'val': 1295, 'test': 647}, summary
    print('Prepared dataset:', json.dumps(summary), flush=True)
    original = local/'baseline/manifest.json'
    if original.is_file():
        identity = lambda rows: sorted((r['image'], r['split'], r['sha256']) for r in rows)
        assert identity(json.loads(original.read_text())) == identity(json.loads((dataset/'manifest.json').read_text())), 'Original split differs.'
    run = local/'runs'/datetime.now(timezone.utc).strftime('yolov8s-1280-%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    config = dict(data=str(dataset/'data.yaml'), epochs=100, patience=20, imgsz=1280,
                  batch=4, device=0, workers=2, seed=42, deterministic=True,
                  optimizer='AdamW', lr0=.001, lrf=.01, cos_lr=True, weight_decay=.0005,
                  warmup_epochs=3, mosaic=1., close_mosaic=10, scale=.5, fliplr=.5,
                  flipud=0., hsv_h=.015, hsv_s=.7, hsv_v=.4, degrees=0., max_det=1000,
                  amp=True, cache=False, save=True, save_period=5, plots=True,
                  project=str(run), name='candidate', exist_ok=False)
    metadata = dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                    gpu=torch.cuda.get_device_name(0), config=config,
                    original_manifest_verified=original.is_file(),
                    note='Original Colab checkpoint not available locally. No measured improvement claim; compare on validation when recovered.')
    (run/'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (local/'active-run.json').write_text(json.dumps(dict(pid=os.getpid(), run_dir=str(run), stage='training'), indent=2))
    os.chdir(PROJECT)
    model = YOLO('yolov8s.pt')
    model.train(**config)
    best = Path(model.trainer.save_dir)/'weights/best.pt'
    result = YOLO(str(best)).val(data=str(dataset/'data.yaml'), split='val', imgsz=1280,
                               batch=4, device=0, workers=2, conf=.001, iou=.7,
                               max_det=1000, project=str(run), name='candidate-val', plots=True)
    metrics = dict(weights=str(best), split='val', precision=float(result.box.mp),
                   recall=float(result.box.mr), map50=float(result.box.map50), map50_95=float(result.box.map))
    (run/'validation-metrics.json').write_text(json.dumps(metrics, indent=2))
    (local/'active-run.json').write_text(json.dumps(dict(pid=os.getpid(), run_dir=str(run), stage='complete', metrics=metrics), indent=2))
    print('Training complete:', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
