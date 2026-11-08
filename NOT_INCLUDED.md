# Deliberately separate from this handoff

- Raw VisDrone image folders and downloaded dataset ZIPs: use the existing Drive folders or the official downloader. They are multi-GB and are external data dependencies, not source code.
- Installed Python/Node packages, virtual environments, caches and bytecode: recreate from requirements and package/version records.
- Git history, short-lived publishing credentials, .env files, private Sites helper state, and Google sign-in screenshots: not required for code review and not shared.
- 20 periodic optimizer checkpoints (`epoch*.pt`), totaling 1.35 GB. Best/last trained weights, ONNX export, complete 100-epoch CSV history, plots, evaluation and training logs ARE included in the full package.
- Older generated ZIP/tar copies of the same source: files are included directly instead of nesting redundant archives.

Official dataset repository: https://github.com/VisDrone/VisDrone-Dataset
Existing VisDrone-all Drive folder: https://drive.google.com/drive/folders/1BVsiZ3oe8pUm629wVJCWcDSaudz1cq4O
Original assignment data folder: https://drive.google.com/drive/folders/1jZFmpBKzvGpA6oFmQ0FcX3duB6rcY9gz
Drive permissions remain unchanged; a teammate may need the owner to grant access.

Raw DET download helper: work/download_local_det.py (requires gdown in the active Python environment). All-task catalog/downloader: outputs/drone-tracking/drone_project/download_all.py.
The local Stage 2 demo script references the original MOT ZIP under C:/Users/A/Downloads; change that source path on a teammate's computer. It does not need all 98 GB of the broader collection for the six-second demo.
