"""Collect the measured video demo artifacts and document its scope."""
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
site=ROOT/'sites/aerial-lens'
summary=json.loads((ROOT/'local-training/stage2/summary.json').read_text())
output=ROOT/'outputs/stage2-results'
output.mkdir(exist_ok=True)
for path in (ROOT/'local-training/stage2').iterdir():
    if path.is_file():shutil.copy2(path,output/path.name)
shutil.copy2(site/'dist/assets/sample-aerial.mp4',output/'source-sample.mp4')
shutil.copy2(Path('C:/Users/A/Downloads/visdrone-sample-video-detections.json'),output/'browser-detections.json')
shutil.copy2(Path('C:/Users/A/Downloads/visdrone-sample-annotated.webm'),output/'browser-annotated.webm')
report=f'''# Stage 2: video detection

## Website

The existing Aerial Lens site now includes a Video workspace at `/video.html`. Upload an MP4, WebM, or browser-decodable MOV, or try the bundled six-second aerial sample. Limits are 60 seconds, 150 MB, and up to approximately 4K resolution. Videos stay on the visitor's device.

The same trained YOLOv8s ONNX model analyzes sampled frames at 2, 5, or 10 fps. The timeline uses the nearest analyzed frame for boxes; boxes can drift between samples. Users can play/pause, scrub, change confidence, toggle labels, stop analysis and preserve partial results, export per-frame JSON, or export a silent annotated WebM containing analyzed frames. The exported video's duration can vary slightly due to browser recording/seek timing. Detection counts are not unique object counts. Browser video mode does not yet assign tracking IDs or run flight control.

## Functional validation

- Real browser sample: 30 analyzed frames from six seconds at 5 fps, completed in 4.3 seconds in the local QA session. This includes one-time setup and is not a general speed guarantee.
- Actual disk-file upload decoded successfully.
- Cancellation preserved 37 of 60 frames in a separate 10 fps run and displayed PARTIAL.
- JSON export contained 30 timestamped frames and respected confidence .28.
- Annotated WebM export decoded successfully and contained all 30 sampled frames.
- Unit checks covered duration/rate limits, timestamp scheduling, nearest-frame alignment, and gaps after partial analysis.

## Local ByteTrack demonstration

The full-frame local pipeline additionally tested the trained PyTorch model with the real ByteTrack tracker. It processed {summary['frames']} frames, with {summary['detections_across_frames']:,} displayed detection instances across frames and {summary['distinct_track_ids']} distinct assigned track IDs. Repeated observations count again in the detection total. Identity changes and fragmentation mean the ID count is not a ground-truth object count.

Measured end-to-end processing took {summary['processing_seconds']:.2f} seconds ({summary['pipeline_fps']:.2f} fps), including tracking and video encoding. Demo playback is set to 30 fps and lasts six seconds; this does not establish original capture timing or real-time 30 fps processing. Tracking confidence floor .1, displayed confidence .28, NMS IoU .7, max_det 1000, inference input 1280 pixels.

Source: the first 180 frames from `uav0000013_00000_v` in the existing VisDrone2019-MOT-train archive. This is a functional demonstration on a training-sequence excerpt, not a held-out tracking benchmark. IDF1, HOTA, and video detection AP have not been measured. The image-test metrics from Stage 1 must not be presented as video tracking metrics. No new model training was performed and no drone-control commands were issued.

## Included files

- `source-sample.mp4`: real aerial sample without annotations.
- `tracked-demo.mp4`: full-frame local YOLOv8s + ByteTrack annotated output.
- `detections.json`, `detections.csv`, `summary.json`: local tracking logs.
- `browser-annotated.webm`, `browser-detections.json`: tested website exports.

Source repository: https://github.com/VisDrone/VisDrone-Dataset

## Remaining stages

If persistent IDs are needed in the public browser interface, integrate and evaluate a real tracking implementation there. The local ByteTrack demonstration does not mean browser IDs have been implemented. Stage 3 still requires target selection, controller integration, and a closed-loop simulation with camera feedback.
'''
(output/'STAGE2_RESULTS.md').write_text(report,encoding='utf-8')
readme=site/'README.md'
text=readme.read_text(encoding='utf-8').replace('An image-only demo of the trained VisDrone YOLOv8s detector.','An image and sampled-video demo of the trained VisDrone YOLOv8s detector.').replace('There is no video tracking or remote image storage.','Video detection is available at `/video.html`; videos and images are not uploaded to a server. Browser tracking IDs are not implemented.').replace('There is no video tracking or remote image storage.','There is no remote image storage.')
text+='\n## Stage 2 video workspace\n\nVideo upload supports up to 60 seconds / 150 MB / approximately 4K. Analysis samples 2, 5, or 10 fps using the existing model. Confidence filtering, playback timeline, frame detection counts, stop/partial results, per-frame JSON, and silent sampled-frame WebM export are supported. Browser codec support varies; H.264 MP4 or WebM is recommended. The browser interface is detection-only; local ByteTrack validation artifacts are outside the Site in outputs/stage2-results. Run `node check-video.mjs` for video scheduling/alignment checks. Real upload, 30-frame sample analysis, stopping, JSON and 30-frame WebM export passed browser QA.\n'
readme.write_text(text,encoding='utf-8')
with zipfile.ZipFile(ROOT/'outputs/Stage2_Results.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in output.iterdir():archive.write(path,Path('Stage2_Results')/path.name)
with zipfile.ZipFile(ROOT/'outputs/Stage2_Results.zip') as archive:assert archive.testzip() is None
print('Stage 2 artifacts packaged.')
