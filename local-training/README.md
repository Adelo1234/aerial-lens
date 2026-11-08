# Local Part 1 training

Hardware verified: NVIDIA GeForce RTX 5090, 32 GB VRAM. Environment is isolated in `../.venv-training` using Python 3.12, official PyTorch 2.11.0 CUDA 12.8, and Ultralytics 8.3.203.

The DET train archive is downloaded from the Google Drive link in the official VisDrone repository and checked for ZIP CRC errors. Other dataset tasks are not needed for this detection run.

The candidate uses pretrained YOLOv8s, 1280 pixels, batch 4, up to 100 epochs, and patience 20. A seed-42 custom 70/20/10 split groups exact image duplicates, as in the earlier Colab preparation. Related sequence frames may still share splits. The original Drive split manifest and baseline checkpoint are not available locally yet, so identity with the original split is not verified and no accuracy improvement is claimed.

`training.stdout.log` and `training.stderr.log` contain progress. `active-run.json` identifies the current output directory and process. Each epoch saves last/best weights, with an additional checkpoint every five epochs. Completed training is followed by candidate validation; test evaluation is deferred until baseline comparison and final model selection.

Keep the PC awake while training. Logs and checkpoints are on local disk, not automatically synced to Drive.

Run from the workspace:

```powershell
& .venv-training/Scripts/python.exe -u outputs/drone-tracking/scripts/train_local_accuracy.py
```

After interruption, restore the same local dataset and resume an unstripped `last.pt` using the Ultralytics CLI in this environment with `yolo train resume model=<absolute-last.pt>`. Do not launch a second job while the PID in `active-run.json` is still running.
