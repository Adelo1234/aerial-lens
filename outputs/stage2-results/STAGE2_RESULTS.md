# Stage 2: video detection

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

The full-frame local pipeline additionally tested the trained PyTorch model with the real ByteTrack tracker. It processed 180 frames, with 11,483 displayed detection instances across frames and 318 distinct assigned track IDs. Repeated observations count again in the detection total. Identity changes and fragmentation mean the ID count is not a ground-truth object count.

Measured end-to-end processing took 17.78 seconds (10.12 fps), including tracking and video encoding. Demo playback is set to 30 fps and lasts six seconds; this does not establish original capture timing or real-time 30 fps processing. Tracking confidence floor .1, displayed confidence .28, NMS IoU .7, max_det 1000, inference input 1280 pixels.

Source: the first 180 frames from `uav0000013_00000_v` in the existing VisDrone2019-MOT-train archive. This is a functional demonstration on a training-sequence excerpt, not a held-out tracking benchmark. IDF1, HOTA, and video detection AP have not been measured. The image-test metrics from Stage 1 must not be presented as video tracking metrics. No new model training was performed and no drone-control commands were issued.

## Included files

- `source-sample.mp4`: real aerial sample without annotations.
- `tracked-demo.mp4`: full-frame local YOLOv8s + ByteTrack annotated output.
- `detections.json`, `detections.csv`, `summary.json`: local tracking logs.
- `browser-annotated.webm`, `browser-detections.json`: tested website exports.

Source repository: https://github.com/VisDrone/VisDrone-Dataset

## Remaining stages

If persistent IDs are needed in the public browser interface, integrate and evaluate a real tracking implementation there. The local ByteTrack demonstration does not mean browser IDs have been implemented. Stage 3 still requires target selection, controller integration, and a closed-loop simulation with camera feedback.
