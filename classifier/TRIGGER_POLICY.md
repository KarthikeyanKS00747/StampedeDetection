# Trigger and Clearance Policy — Crowd Anomaly Classifier

**Version:** 1.0-draft  
**Status:** Pre-integration — requires operational review before activation  
**Date:** 2026-10-15

---

## Context

The classifier outputs per-window observations (`normal` / `abnormal`) at a stride
of 15 frames (~0.5–1.0 s depending on camera FPS). A single window should not
trigger an alert — temporal smoothing is required to suppress noise.

---

## Definitions

| Term | Definition |
|---|---|
| **Window** | 30-frame sliding inference unit. One result every 15 frames. |
| **Alert episode** | A contiguous run of consecutive abnormal windows meeting the trigger threshold |
| **Clearance** | Return to normal state after a minimum quiet period |

---

## Trigger Policy (draft — not yet validated)

> [!IMPORTANT]
> These thresholds are **initial proposals** based on the window length and stride.
> They have not been tuned against any data. Adjust after independent normal footage tests.

| Parameter | Proposed value | Rationale |
|---|---|---|
| **Trigger** | ≥ 3 consecutive `abnormal` windows | ~1.5 s at 30 FPS, ~3 s at 15 FPS. Filters single-window noise. |
| **Alert action** | Log observation to separate channel. Notify human operator. | Do NOT auto-trigger buzzer. |
| **Clearance** | ≥ 5 consecutive `normal` windows | Prevents rapid flicker at boundary. |
| **Minimum episode duration** | — | Not specifiable until independent FAR is measured. |

---

## What Does NOT Happen Automatically

- `"abnormal"` does **not** map to any risk tier (High / Critical).
- Buzzer does **not** activate.
- Existing YOLO + rule-based baseline is **not** overridden.
- The anomaly classifier observation is logged in a **separate channel** alongside the baseline.

---

## Operating Assumptions

1. Camera FPS is known and passed correctly to the extractor (`--fps`).
2. Person detector is `yolov8n.pt` — not YOLO11s or any other variant.
3. Window length = 30 frames, stride = 15 frames.
4. Processing duration per window ≈ 5.2 s (offline test; live timing not yet measured).
5. First decision is available after the first complete window, not at stream start.

---

## Required Before Activation

- [ ] Independent normal footage FAR measured (routine walking, stationary, empty scene,  
      lighting changes, camera movement). Report FP window fraction + episodes/hour.
- [ ] Trigger threshold tuned from FAR results (adjust the ≥3 window proposal).
- [ ] Live processing latency measured (not offline estimate).
- [ ] Detector mismatch resolved (YOLOv8n vs YOLO11s).
- [ ] Human operator review of at least one full live session before unattended operation.

---

## Independent Normal Footage Checklist

Collect and run `infer.py --ground_truth 0` on each of the following:

| Scenario | Min duration | Why |
|---|---|---|
| Routine crowd walking (similar density to training) | 5 min | Baseline normal |
| Stationary crowd (low motion) | 2 min | Tests motion feature lower bound |
| Empty or near-empty scene | 2 min | Tests N_mean=0 path (same as missed windows) |
| Lighting change (sun/cloud, day/night transition) | 1 min | Tests motion feature stability |
| Camera movement / pan | 1 min | Tests motion_mean spike under camera shake |

For each: report false-positive window count, total windows, FP rate (%), and distinct
alert episodes (≥3 consecutive abnormal windows) per hour.
