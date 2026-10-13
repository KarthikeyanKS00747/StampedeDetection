# Data workflow

## Current benchmark

`data/raw/umn/Crowd-Activity-All.avi` is the official UMN crowd-activity benchmark downloaded by `scripts/download_datasets.py`. It contains 7,739 frames, 30 FPS, at 320 x 240, concatenating 11 short sequences over three scenes. Each sequence transitions from ordinary walking to collective escape motion.

It is a useful first test of the project's direction-variance and velocity-change features. It does **not** contain verified stampedes, calibrated real-world density, or four risk classes. Never train the final four-level model by automatically treating "escape" as Critical.

## Review gate

1. Inspect `data/review/umn_contact_sheet.jpg`.
2. Set `keep=yes/no` and a reason in `data/review/clip_review.csv`.
3. If it is kept, run feature extraction using a measured or clearly documented provisional `--area-m2`.
4. Create a label template, review each window, and only then train the risk model.

```powershell
python scripts/extract_dataset_features.py --input data/raw/umn/Crowd-Activity-All.avi --output data/features/umn_features.csv --area-m2 100 --model yolo11n.pt
python scripts/make_label_template.py --features data/features/umn_features.csv --output data/review/umn_labels.csv
python -m training.train --input data/review/umn_labels.csv --output models/risk_classifier.joblib
```

`100` is only a placeholder because UMN does not provide an on-ground calibration in this project. The resulting density is therefore a relative measure and cannot be used as a field threshold.

## GPU gate

The installed Torch build is CPU-only even though the RTX 4070 is visible to Windows. GPU preprocessing should begin only after `torch.cuda.is_available()` reports `True`. Use the official CUDA wheel selected for Windows, then verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The large CUDA package download was started but not completed within this time-box, so the existing environment remains CPU-only.
