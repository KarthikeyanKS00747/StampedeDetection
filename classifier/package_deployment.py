"""
package_deployment.py
---------------------
Assembles the exact-matched deployment package for experimental integration.

Creates deploy/ containing:
  deploy/
    infer.py                  <- inference entry point
    extract_features.py       <- feature extractor (must match training exactly)
    model/
      classifier.pkl
      feature_order.txt
      metadata.json
    requirements.txt
    INTEGRATION_NOTES.md      <- constraints and warnings

Usage
-----
  python package_deployment.py
  # Output: deploy/ folder ready to copy to integration environment
"""

import shutil
import json
from pathlib import Path

BASE = Path(__file__).parent
DEPLOY = BASE / "deploy"

FILES = [
    ("infer.py",            "infer.py"),
    ("extract_features.py", "extract_features.py"),
    ("requirements.txt",    "requirements.txt"),
]
MODEL_FILES = [
    "classifier.pkl",
    "feature_order.txt",
    "metadata.json",
]

INTEGRATION_NOTES = """\
# Integration Notes — Crowd Anomaly Classifier

## What this package contains
- `infer.py`            — inference entry point
- `extract_features.py` — feature extractor (must not be modified)
- `model/`              — trained Random Forest + feature order + metadata
- `requirements.txt`    — pinned dependencies

## Critical constraints before going live

### 1. Detector must match training exactly
This model was trained on **YOLOv8n** person detections at **source video FPS**.
Your live system uses **YOLO11s at 5 FPS**.
These are NOT interchangeable. Features (N_mean, overlap_mean, motion_mean) will
differ systematically. Before live inference:
  Option A: use yolov8n.pt in the live pipeline (same as training)
  Option B: re-extract all training features using YOLO11s/5-FPS, retrain

### 2. FPS must be passed explicitly
Pass --fps matching the camera's actual frame rate. Do not assume.

### 3. Output is observation only
infer.py outputs "normal" or "abnormal" per window.
Do NOT automatically map "abnormal" to High/Critical risk tier.
Do NOT trigger the buzzer automatically.
Log alongside the existing YOLO + rule-based baseline.

### 4. Trigger and clearance policy
See TRIGGER_POLICY.md for the agreed alert episode definition.

### 5. Independent FAR not yet measured
False-positive rate on independent normal footage is unknown.
Do not use in production until tested on: routine walking, stationary
crowds, empty scene, lighting changes, camera movement.

## Runnable inference command
  python infer.py \\
      --image_dir path/to/frames \\
      --fps 25 \\
      --model_dir model \\
      --output_csv results/output.csv

## Window parameters (must match training)
  Window length : 30 frames
  Stride        : 15 frames (50% overlap)
  Threshold     : 0.5 (RF majority vote)
"""


def main():
    if DEPLOY.exists():
        shutil.rmtree(DEPLOY)
    DEPLOY.mkdir()

    # Copy scripts
    for src_name, dst_name in FILES:
        src = BASE / src_name
        if not src.exists():
            print(f"MISSING: {src_name}")
            continue
        shutil.copy2(src, DEPLOY / dst_name)
        print(f"Copied {src_name}")

    # Copy model directory
    model_src = BASE / "model"
    model_dst = DEPLOY / "model"
    model_dst.mkdir()
    for f in MODEL_FILES:
        src = model_src / f
        if not src.exists():
            print(f"MISSING: model/{f}")
            continue
        shutil.copy2(src, model_dst / f)
        print(f"Copied model/{f}")

    # Write integration notes
    (DEPLOY / "INTEGRATION_NOTES.md").write_text(INTEGRATION_NOTES)
    print("Wrote INTEGRATION_NOTES.md")

    # Verify feature_order matches extractor
    feat_order = (model_dst / "feature_order.txt").read_text().strip().splitlines()
    extractor_text = (DEPLOY / "extract_features.py").read_text()
    mismatches = [f for f in feat_order if f not in extractor_text]
    if mismatches:
        print(f"WARNING: feature_order.txt entries not found in extractor: {mismatches}")
    else:
        print("Verified: all feature_order.txt entries present in extract_features.py")

    # Print metadata summary
    meta = json.loads((model_dst / "metadata.json").read_text())
    print(f"\nModel metadata:")
    print(f"  Trained at      : {meta.get('trained_at')}")
    print(f"  sklearn version : {meta.get('sklearn_version')}")
    print(f"  Training scenes : {meta.get('training_incidents')}")
    print(f"  Holdout         : {meta.get('holdout_incident')}")

    print(f"\nDeployment package ready -> {DEPLOY}/")
    print("Next: resolve detector mismatch (YOLOv8n vs YOLO11s) before live inference.")


if __name__ == "__main__":
    main()
