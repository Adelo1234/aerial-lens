# Accuracy experiment

Colab_Part1_Improve_Accuracy.ipynb is a standalone notebook with embedded project code. It restores the Drive dataset and checks the original manifest before training. The saved YOLOv8n baseline remains available. Results, logs and checkpoints are stored under MyDrive/aerial-tracking-project/accuracy-experiments/<UTC timestamp>.

The experiment compares baseline inference at 960 and 1280 pixels against a pretrained YOLOv8s candidate trained at 1280 pixels. Training uses at most 100 epochs, early stopping patience 20, batch 2, AdamW and cosine decay. A larger model and more pixels may help small objects; this must be verified rather than assumed. The 1000-detection limit avoids the default 300-detection cap in dense images and is shared across comparisons. Higher resolution and more detections increase runtime and may affect tracking throughput.

Validation mAP50-95 selects the final model and input size. Per-class precision, recall and AP are also reported. A deployment confidence is chosen from validation F1 curves; increasing confidence alone trades recall for precision and does not establish improved AP. Final test AP uses conf=0.001 for the full precision-recall curve. It does not report deployment-threshold precision as test AP.

Original metrics used different evaluation settings; compare the new measurements within this notebook. This custom split groups exact duplicates, but it does not prevent all related sequence frames from sharing splits. Report that limitation alongside results. A single-seed gain is not a statistical significance result.

The notebook has been syntax and configuration checked locally; it has not been run on the Colab GPU. Training may require multiple Colab sessions. For interruption recovery, set RESUME_CHECKPOINT to the unstripped candidate last.pt in the configuration cell. Completed stripped checkpoints are for evaluation; do not resume them. Keep the prior experiment folder for its logs and provenance.
