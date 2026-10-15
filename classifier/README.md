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
    --normal   features/lp_train1.csv features/lp_train2.csv features/ts_v1_train.csv features/ts_v2_train.csv features/ts_v3_train.csv features/lv_train.csv \
    --abnormal features/ts_v1_test.csv features/ts_v2_test.csv features/ts_v3_test.csv features/lv_test1.csv features/lv_test2.csv features/lv_test3.csv features/lv_test4.csv \
    --model_dir model
```

**Do NOT pass Italy CSVs here.** Italy is the held-out evaluation set.

Training incidents: Las Vegas, Times Square, Love Parade  
Holdout incident: Italy (4_Italy) — zero training contact

Saved artefacts:
- `model/classifier.pkl` — trained Random Forest
- `model/feature_order.txt` — 7 feature names (frozen)
- `model/metadata.json` — training CSV provenance and sklearn version; source FPS is documented above

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

When `--ground_truth` is passed, `infer.py` reports an offline summary:

| Metric | Description |
|---|---|
| **Any-window detection** | Whether any window in this abnormal sequence was positive |
| **FP window fraction** | Positive windows / total windows on the supplied normal footage |
| **First positive window** | Window frame span and **start** timestamp; not live delay |
| **Failure case** | Printed when no abnormal window is detected |

The six saved normal-training replays contain **0/345 in-sample false-positive
windows**. This does not establish an independent false-positive rate or
false-alert episodes/hour. Two windows beginning at 5.4s and 6.0s have start
timestamps, not detection timestamps. Their combined frame span is 135–179;
at 25 FPS that is approximately 5.40–7.16s in frame timestamps (through about
7.20s of captured footage). If 0.174s/frame is compute time, processing 30
frames takes about 5.22s; live detection delay needs a separate timing test.
For those two Italy View 2 windows, `N_mean`, `N_std`, both overlap features,
and `delta_N` are zero, while `motion_mean` is 3.97007 and 4.19821. Thus
`N_mean=0` is only a suspected failure mechanism; inspect detector output,
the remaining feature distributions, and extraction behavior before assigning
causality.

Next: evaluate independent normal footage, review the proposed
`TRIGGER_POLICY.md`, and use the exact-matched `deploy/` package for
experimental integration. Optional YOLO fine-tuning is deferred.

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
