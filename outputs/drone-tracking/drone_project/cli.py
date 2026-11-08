import argparse
import csv
import json
import math
import time
from pathlib import Path

from .control import Detection, IBVSController, TargetLock
from .data import NAMES, prepare


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def train(args):
    from ultralytics import YOLO
    import torch
    import ultralytics
    data = Path(args.data).resolve()
    if not data.is_file():
        raise ValueError(f'Dataset configuration not found: {data}')
    config = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, workers=args.workers, seed=args.seed,
                  deterministic=True, project=str(Path(args.output).resolve()), name=args.name,
                  patience=args.patience, optimizer='AdamW', lr0=.001, weight_decay=.0005,
                  mosaic=1., close_mosaic=10, fliplr=.5, flipud=0., scale=.5,
                  hsv_h=.015, hsv_s=.7, hsv_v=.4, degrees=0., save=True, plots=True,
                  pretrained=True, exist_ok=False)
    model = YOLO(args.resume or args.model)
    if args.resume:
        result = model.train(resume=True)
    else:
        result = model.train(**config)
    folder = Path(model.trainer.save_dir)
    save_json(folder/'environment.json', dict(torch=torch.__version__,
              ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
              config=config, actual_args=vars(model.trainer.args),
              cuda_available=torch.cuda.is_available()))
    print(f'Training saved to {folder}; use weights/best.pt for evaluation and tracking.')


def evaluate(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    result = model.val(data=str(Path(args.data).resolve()), split=args.split,
                       imgsz=args.imgsz, batch=args.batch, device=args.device,
                       project=str(Path(args.output).resolve()), name=f'eval-{args.split}', plots=True)
    scores = dict(result.results_dict)
    scores.update(split=args.split, weights=args.weights, speed_ms=result.speed,
                  per_class=[dict(class_id=int(c), name=model.names[int(c)],
                                  precision=float(result.box.p[i]), recall=float(result.box.r[i]),
                                  map50=float(result.box.ap50[i]), map50_95=float(result.box.ap[i]))
                             for i, c in enumerate(result.box.ap_class_index)])
    save_json(Path(result.save_dir)/'metrics.json', scores)
    print(json.dumps(scores, indent=2))


def track(args):
    import cv2
    from ultralytics import YOLO
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video source: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if math.isfinite(fps) and fps > 0 else 30.
    model, controller = YOLO(args.weights), IBVSController()
    classes = tuple(int(x) for x in args.classes.split(','))
    unknown = set(classes)-set(model.names)
    if unknown:
        raise ValueError(f'Class IDs are absent from this checkpoint: {unknown}')
    lock, writer = TargetLock(classes, args.target_id), None
    config = dict(weights=args.weights, source=args.source, classes=classes,
                  tracker='bytetrack.yaml', fps=fps, note='Commands are logged; no aircraft connection.')
    save_json(output/'run.json', config)
    count, started = 0, time.perf_counter()
    try:
        with (output/'commands.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = ['frame', 'time_s', 'target_id', 'state', 'ex', 'ey', 'yaw', 'vertical', 'forward']
            log = csv.DictWriter(stream, fieldnames=fields)
            log.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(output/'tracked.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
                    if not writer.isOpened():
                        raise RuntimeError('Video encoder could not open output.')
                result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                                     conf=.1, iou=.5, imgsz=args.imgsz, device=args.device,
                                     classes=list(classes), verbose=False)[0]
                detections = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for box, tid, category, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(),
                                                              boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                        detections.append(Detection(int(tid), int(category), float(confidence), tuple(box)))
                target = lock.select(detections)
                command = controller.command(target.xyxy if target else None, width, height)
                log.writerow(dict(frame=count, time_s=count/fps, target_id=lock.target_id, **command))
                annotated = result.plot()
                cv2.drawMarker(annotated, (width//2,height//2), (255,255,255), cv2.MARKER_CROSS, 25, 2)
                if target:
                    x1,y1,x2,y2 = map(int, target.xyxy)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(annotated, f'ID {lock.target_id} | {command["state"]} | yaw {command["yaw"]:.2f}',
                            (15,30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
                writer.write(annotated)
                count += 1
                if args.show:
                    cv2.imshow('Aerial tracking - Q to quit', annotated)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter()-started
    save_json(output/'performance.json', dict(frames=count, total_seconds=elapsed,
              pipeline_fps=count/elapsed if elapsed else 0,
              note='End-to-end throughput including encoding; not detector-only inference speed.'))
    print(f'Video and command log saved to {output}')


def simulate(args):
    """Closed-loop camera plant with analytic target measurements (no learned detector)."""
    from PIL import Image, ImageDraw
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    controller = IBVSController(gain=args.gain)
    pan, tilt, rows, frames = 0., 0., [], []
    width, height, dt = 640, 360, 1/20
    for i in range(args.steps):
        t = i*dt
        # World target drifts horizontally. Camera pan right reduces screen x;
        # camera movement up increases screen y in this simplified plant.
        world_x, world_y = .65+.12*math.sin(t*.6), -.35+.08*math.sin(t*.4)
        ex, ey = world_x-pan, world_y+tilt
        cx, cy = width*(1+ex)/2, height*(1+ey)/2
        box = (cx-18, cy-24, cx+18, cy+24)
        lost = args.occlusion_start <= i < args.occlusion_end
        command = controller.command(None if lost else box, width, height)
        pan += command['yaw']*dt
        tilt += command['vertical']*dt
        rows.append(dict(frame=i, time_s=t, **command, actual_ex=ex, actual_ey=ey))
        im = Image.new('RGB', (width,height), '#0e1b29')
        draw = ImageDraw.Draw(im)
        for x in range(0,width,40):
            draw.line((x,0,x,height), fill='#213246')
        for y in range(0,height,40):
            draw.line((0,y,width,y), fill='#213246')
        draw.line((width/2-14,height/2,width/2+14,height/2), fill='white', width=2)
        draw.line((width/2,height/2-14,width/2,height/2+14), fill='white', width=2)
        if not lost:
            draw.rectangle(box, fill='#2cbda0', outline='white', width=2)
        draw.text((12,12), f'CAMERA SIMULATION | t={t:.2f}s | {command["state"]}', fill='white')
        draw.text((12,30), f'error x={ex:.3f} y={ey:.3f} | yaw={command["yaw"]:.3f} vz={command["vertical"]:.3f}', fill='white')
        frames.append(im)
    with (out/'simulation.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frames[0].save(out/'simulation.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    error = lambda row: math.hypot(row['actual_ex'],row['actual_ey'])
    summary = dict(measurement_source='Analytic synthetic ground-truth boxes; no YOLO/ByteTrack in this demo.',
                   initial_error=error(rows[0]), final_error=error(rows[-1]),
                   mean_error=sum(error(r) for r in rows)/len(rows),
                   zero_commands_while_lost=all(r['yaw']==r['vertical']==0 for r in rows if r['state']=='lost'),
                   steps=args.steps, dt=dt, gain=args.gain)
    save_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description='Aerial detection, tracking, and simulated image-based control')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('prepare', help='Validate labels and create a reproducible 70/20/10 split')
    for name in ('images','labels','output'):
        d.add_argument(f'--{name}', required=True)
    d.add_argument('--format', choices=['visdrone','yolo'], default='visdrone')
    d.add_argument('--seed', type=int, default=42)
    d.add_argument('--groups-csv')
    d.set_defaults(func=lambda a: print(json.dumps(prepare(a.images,a.labels,a.output,a.format,a.seed,a.groups_csv),indent=2)))
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--model', default='yolov8n.pt')
    t.add_argument('--resume', help='Path to last.pt; restores original run arguments')
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--workers', type=int, default=2)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--patience', type=int, default=15)
    t.add_argument('--name', default='yolov8n-visdrone')
    t.set_defaults(func=train)
    e = sub.add_parser('evaluate')
    e.add_argument('--data', required=True)
    e.add_argument('--weights', required=True)
    e.add_argument('--split', choices=['val','test'], default='test')
    e.set_defaults(func=evaluate)
    for parser in (t,e):
        parser.add_argument('--imgsz', type=int, default=960)
        parser.add_argument('--batch', type=int, default=8)
        parser.add_argument('--device', default='0')
        parser.add_argument('--output', default='runs')
    v = sub.add_parser('track')
    v.add_argument('--weights', required=True)
    v.add_argument('--source', required=True, help='Video file or webcam index')
    v.add_argument('--classes', default='0,1', help='VisDrone zero-based class IDs; 0,1 selects people')
    v.add_argument('--target-id', type=int)
    v.add_argument('--output', default='runs/tracking')
    v.add_argument('--show', action='store_true')
    v.add_argument('--device', default='0')
    v.add_argument('--imgsz', type=int, default=960)
    v.set_defaults(func=track)
    s = sub.add_parser('simulate')
    s.add_argument('--output', default='runs/simulation')
    s.add_argument('--steps', type=int, default=240)
    s.add_argument('--gain', type=float, default=1.)
    s.add_argument('--occlusion-start', type=int, default=100)
    s.add_argument('--occlusion-end', type=int, default=120)
    s.set_defaults(func=simulate)
    args = p.parse_args()
    if hasattr(args,'steps') and args.steps < 1:
        p.error('--steps must be positive')
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

