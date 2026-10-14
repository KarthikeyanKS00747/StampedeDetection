# Crowd Anomaly Detection Pipeline

Experimental sequence-level normal/abnormal classifier for the *Abnormal High-density Crowds* dataset.

> **This is an experimental model. It is NOT a validated stampede predictor. The existing YOLO + rule-based baseline remains primary.**

---

## Dataset

4 real-world crowd incidents (17,468 extracted PNG frames):

| Incident | Normal frames | Abnormal frames | YOLO bbox labels |
|---|---|---|---|
| 1_Times_Square (3 views) | 713 | 3,350 | None |
| 2_Las_Vegas | 4,347 | 7,243 | None |
| 3_Love Parade | 250 | 361 | Yes — class `fight_onset` |
| 4_Italy (2 views) | 0 | 1,404 | None |

The `fight_onset` label was applied by the dataset author (OpenLabeling, class name `Fight`) to frames showing panic onset in Love Parade footage. It localises one region in one viewpoint of one incident. It is preserved as `fight_onset` and has not been validated as a general stampede marker.

---

## Known Limitations

1. **4 incidents total.** ~17k frames have high temporal autocorrelation; effective independent samples ≈ 4.
2. **Single evaluation event.** Italy (held-out) is one event — metric variance is unknown.
3. **Extracted FPS may differ from source.** Always pass `--fps` explicitly (see README values per scene below).
4. **View-dependent features.** Count and motion features are camera-specific; cross-scene comparison is not valid.
5. **`fight_onset` covers one region in one viewpoint.** Not validated as a general panic indicator.
6. **Experimental.** Do not replace the baseline until integration is reviewed.

Source FPS per scene (from dataset README):
- Times Square: 29.97
- Las Vegas: 15.17
- Love Parade: 25
- Italy: 25

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

### Step 1 — Extract features

```bash
# Normal sequence example (Las Vegas train)
python extract_features.py \
    --image_dir "Abnormal High-density Crowds/2_Las_Vegas/Train" \
    --fps 15.17 \
    --output_csv features/lv_train.csv

# Abnormal sequence example (Las Vegas Test_1)
python extract_features.py \
    --image_dir "Abnormal High-density Crowds/2_Las_Vegas/Test_1" \
    --fps 15.17 \
    --output_csv features/lv_test1.csv
```

Feature vector (7 columns, frozen order in `model/feature_order.txt`):

| Feature | Description |
|---|---|
| `N_mean` | Mean person count across window |
| `N_std` | Std of person count |
| `overlap_mean` | Mean pairwise IoU of person boxes |
| `overlap_std` | Std of pairwise IoU |
| `motion_mean` | Mean frame-difference magnitude |
| `motion_std` | Std of frame-difference magnitude |
| `delta_N` | Change in person count across window |

### Step 2 — Train classifier

```bash
python train_classifier.py \
    --normal   features/lv_train.csv features/ts_v1_train.csv features/lp_train1.csv features/lp_train2.csv \
    --abnormal features/lv_test1.csv features/ts_v1_test.csv \
    --model_dir model
```

**Do NOT pass Italy CSVs here.** Italy is the held-out evaluation set.

Training incidents: Las Vegas, Times Square, Love Parade  
Holdout incident: Italy (4_Italy) — zero training contact

Saved artefacts:
- `model/classifier.pkl` — trained Random Forest
- `model/feature_order.txt` — 7 feature names (frozen)
- `model/metadata.json` — training provenance, sklearn version, FPS used

### Step 3 — Inference

```bash
python infer.py \
    --image_dir "Abnormal High-density Crowds/4_Italy/View_1/Test" \
    --fps 25 \
    --model_dir model \
    --ground_truth 1 \
    --output_csv italy_v1_results.csv
```

Output columns: `window_start_frame`, `window_end_frame`, `window_start_ts`, `predicted_label`, `confidence`, + 7 features.

Output label is `"normal"` or `"abnormal"`. **There is no automatic mapping to any risk tier.** Integration requires a separate, documented review.

### Step 4 — Evaluation metrics

When `--ground_truth` is passed, `infer.py` reports:

| Metric | Description |
|---|---|
| **Recall** | Fraction of abnormal sequences detected (primary) |
| **FAR** | False alarms per normal sequence (primary) |
| **Detection Latency** | Frames from sequence start to first alarm |
| **Failure Cases** | Printed when recall = 0 |

Training loss and mAP are not primary outcomes.

---

## Optional: YOLO fight_onset fine-tune (experimental)

```bash
python train_yolo.py \
    --data_dir  "Abnormal High-density Crowds/3_Love Parade/Test" \
    --label_dir "Abnormal High-density Crowds/3_Love Parade/Test/Label/YOLO_darknet" \
    --output_dir runs/fight_onset_yolo \
    --epochs 50
```

This fine-tunes YOLOv8n on 361 frames from one incident. Scope: localisation model only. Do not integrate into the primary pipeline.

---

## Baseline Preservation

No existing workspace files are modified. All scripts are additive. The anomaly classifier output is independent of the current alert system. Promotion to primary requires a separate integration review.
