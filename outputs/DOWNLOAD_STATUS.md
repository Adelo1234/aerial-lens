# VisDrone download status

Account: adelalyamani2006@gmail.com

Checked on 2026-10-06, approximately 18:41 Asia/Riyadh.

**15 of 20 archives downloaded and verified, totaling 45.383 GB (decimal).** ZIP CRC checks and locally calculated SHA-256 values are recorded in the Drive download manifest. The full collection is incomplete.

- [Saved datasets in Google Drive](https://drive.google.com/drive/u/1/folders/1BVsiZ3oe8pUm629wVJCWcDSaudz1cq4O)
- [Running notebook and retry cell](https://colab.research.google.com/drive/1l5Q_DEkQaOpotxnzlZBCdk6u-47EIi1G?authuser=1)

Completed: all four DET splits, all four VID splits, both SOT training parts, SOT validation, SOT challenge initialization, MOT validation, MOT test-dev, and DroneCrowd.

Pending: SOT test-dev, SOT test-challenge part 1, SOT test-challenge part 2, MOT training, and MOT test-challenge. All five returned Google's message that too many users recently viewed or downloaded the source file, including on a second attempt. Google says availability may take up to 24 hours to recover. These are source download quotas; the destination Drive account has sufficient storage.

The notebook's dataset cell is configured to retry only pending files. When downloads become available, run setup/definitions if the runtime has reset, then run the retry cell and summary cell. The five failures remain explicitly recorded in `MyDrive/VisDrone-all/download-all-manifest.json`. No automatic retry or scheduled follow-up has been created.

Archives were retained without extraction. Full YOLO training was not started during this download task.

Retry at 18:45 Arabia Standard Time on 2026-10-06: all five pending archives were attempted again in the connected Colab runtime. Each returned Google's source download quota error. Total remains 15/20 verified archives (45.383 GB); no new archives downloaded. Refreshed notebook summary and saved visdrone-retry-result.png.
