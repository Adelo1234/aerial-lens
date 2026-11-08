# Aerial Lens

An image and sampled-video demo of the trained VisDrone YOLOv8s detector. Images stay in the browser. The site downloads the converted model and runs it in a module Worker using ONNX Runtime Web 1.23.2, with WebGPU when supported and WASM fallback. Inference uses 1280-square letterboxing, RGB float32 input, class-aware NMS at .7, and at most 1000 detections. The confidence slider filters cached detections above a .05 floor; its initial value is .28, rounded from the validation-selected threshold.

Features: JPG/PNG/WEBP upload and drag-and-drop, validation samples, bounding boxes, confidence filtering, labels toggle, class/object counts, annotated PNG and JSON downloads, and model information. Video detection is available at `/video.html`; videos and images are not uploaded to a server. Browser tracking IDs are not implemented.

Serve `dist` over HTTP; opening index.html as a file will not support module workers. From this folder, run `npm run dev` or double-click `Start Aerial Lens.cmd`, then open http://127.0.0.1:8766. Node.js is the only server dependency; no Python inference server is needed. First use downloads approximately 45 MB of model weights plus the runtime; performance depends on the browser/device. Local assets are bundled; Google Fonts are optional presentation resources.

The original checkpoint is `../../local-training/runs/yolov8s-1280-20261008-142900/candidate/weights/best.pt`. Export is ONNX opset 17, float32, fixed batch 1 and 1280 resolution. Model chunk checksums are verified before initialization. The inference runtime is vendored with its notices. Run `node check-detection.mjs` for decoding checks.

VisDrone samples and model training source: https://github.com/VisDrone/VisDrone-Dataset. Ultralytics code and model terms: https://github.com/ultralytics/ultralytics (AGPL-3.0 or applicable license). ONNX Runtime: https://github.com/microsoft/onnxruntime (MIT). Reported 64.8% precision and 57.8% mAP50 are the original PyTorch custom-test metrics, not new browser benchmarks.

Verification: numerical export agreement passed (maximum class-score difference 0.0000023, coordinate difference 0.0124 input pixels). Browser tests passed for sample selection, a real disk upload, WebGPU inference, live confidence filtering, zero-detection results, label toggle, JSON and annotated PNG downloads. Mobile width 390px showed no horizontal overflow. A sample produced 41 objects across four classes; another produced 70 across nine classes at .28 confidence. These counts are predictions, not accuracy measurements. CPU fallback is implemented but was not separately exercised in browser QA.

Publication: the user authorized publishing the website and trained model assets for anyone with the link to try. Site identity is in .openai/hosting.json. Uploaded images are processed in each visitor's browser and are not sent to the server.

## Stage 2 video workspace

Video upload supports up to 60 seconds / 150 MB / approximately 4K. Analysis samples 2, 5, or 10 fps using the existing model. Confidence filtering, playback timeline, frame detection counts, stop/partial results, per-frame JSON, and silent sampled-frame WebM export are supported. Browser codec support varies; H.264 MP4 or WebM is recommended. The browser interface is detection-only; local ByteTrack validation artifacts are outside the Site in outputs/stage2-results. Run `node check-video.mjs` for video scheduling/alignment checks. Real upload, 30-frame sample analysis, stopping, JSON and 30-frame WebM export passed browser QA.
