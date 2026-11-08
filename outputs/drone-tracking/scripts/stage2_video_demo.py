"""Build a real aerial clip and test the trained detector with ByteTrack."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[3]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'work/yolo-settings')


def encoder(path,width,height,fps):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','veryfast',
        '-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO
    output=ROOT/'local-training/stage2'
    output.mkdir(exist_ok=True)
    clip=ROOT/'sites/aerial-lens/dist/assets/sample-aerial.mp4'
    sequence='VisDrone2019-MOT-train/sequences/uav0000013_00000_v/'
    fps=30
    with ZipFile('C:/Users/A/Downloads/VisDrone2019-MOT-train.zip') as archive:
        names=sorted(n for n in archive.namelist() if n.startswith(sequence) and n.endswith('.jpg'))[:180]
        first=cv2.imdecode(np.frombuffer(archive.read(names[0]),np.uint8),cv2.IMREAD_COLOR)
        height,width=first.shape[:2]
        writer=encoder(clip,width,height,fps)
        try:
            for name in names:
                frame=cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_COLOR)
                writer.stdin.write(frame.tobytes())
        finally:
            writer.stdin.close()
        assert writer.wait()==0,'Sample encoding failed.'
    print('Sample encoded:',clip,flush=True)
    weights=ROOT/'local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt'
    model=YOLO(str(weights))
    capture=cv2.VideoCapture(str(clip))
    assert capture.isOpened()
    out_width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));out_height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=encoder(output/'tracked-demo.mp4',out_width,out_height,fps)
    frames=[];unique=set();total=0;started=time.perf_counter()
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            result=model.track(frame,persist=True,tracker='bytetrack.yaml',conf=.1,iou=.7,
                               max_det=1000,imgsz=1280,device=0,verbose=False)[0]
            detections=[]
            for box in result.boxes:
                score=float(box.conf[0])
                if score<.28:continue
                cls=int(box.cls[0]);track=int(box.id[0]) if box.id is not None else None
                xyxy=box.xyxy[0].cpu().tolist()
                detections.append(dict(class_id=cls,class_name=model.names[cls],confidence=score,
                                       track_id=track,xyxy=xyxy))
                if track is not None:unique.add(track)
                x1,y1,x2,y2=map(int,xyxy)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(160,235,190),2)
                cv2.putText(frame,f'{model.names[cls]} #{track if track is not None else "?"} {score:.0%}',
                            (x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(160,235,190),1,cv2.LINE_AA)
            index=len(frames)
            cv2.putText(frame,f'YOLOv8s + ByteTrack | frame {index+1} | {len(detections)} detections',
                        (12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            writer.stdin.write(frame.tobytes())
            frames.append(dict(frame=index,time_s=index/fps,detections=detections));total+=len(detections)
            if len(frames)%30==0:print('Processed frames:',len(frames),flush=True)
    finally:
        capture.release();writer.stdin.close()
    assert writer.wait()==0,'Annotated video encoding failed.'
    seconds=time.perf_counter()-started
    summary=dict(frames=len(frames),playback_fps=fps,duration_s=len(frames)/fps,
                 detections_across_frames=total,distinct_track_ids=len(unique),
                 processing_seconds=seconds,pipeline_fps=len(frames)/seconds,
                 display_confidence=.28,tracking_confidence_floor=.1,
                 source_sequence=sequence,source_repository='https://github.com/VisDrone/VisDrone-Dataset',
                 note='MOT training-sequence excerpt for a functional demo, not a held-out tracking benchmark. Playback set to 30 fps for this demo. Track IDs may switch; distinct IDs do not equal verified unique objects. No drone control is run.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'detections.json').write_text(json.dumps(dict(summary=summary,frames=frames)))
    with (output/'detections.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','time_s','class_id','class_name','confidence','track_id','x1','y1','x2','y2'])
        for frame in frames:
            for box in frame['detections']:
                writer.writerow([frame['frame'],frame['time_s'],box['class_id'],box['class_name'],box['confidence'],box['track_id'],*box['xyxy']])
    (ROOT/'sites/aerial-lens/dist/assets/video-sample.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
