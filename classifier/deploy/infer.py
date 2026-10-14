"""
infer.py
--------
Runs the trained anomaly classifier on a directory of PNG frames.

Output: per-window label ("normal" or "abnormal") and confidence score,
written to a CSV and printed to stdout.

When --ground_truth is provided (0=normal, 1=abnormal), the script also
computes Recall, False Alarm Rate (FAR), and Detection Latency.

IMPORTANT: The output label "abnormal" is NOT automatically mapped to any
risk tier. Integration with the alert system requires a separate, documented
step and human sign-off.

Usage (Italy holdout evaluation)
---------------------------------
  python infer.py \
      --image_dir "Abnormal High-density Crowds/4_Italy/View_1/Test" \
      --fps 25 \
      --model_dir model \
      --ground_truth 1 \
      --output_csv italy_v1_results.csv

Runnable command (no ground truth):
  python infer.py --image_dir path/to/frames --fps 25 --output_csv results.csv
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Must match extract_features.py and train_classifier.py exactly
FEATURE_NAMES = [
    "N_mean", "N_std",
    "overlap_mean", "overlap_std",
    "motion_mean", "motion_std",
    "delta_N",
]


def load_model(model_dir: Path):
    pkl = model_dir / "classifier.pkl"
    feat_file = model_dir / "feature_order.txt"
    meta_file = model_dir / "metadata.json"

    if not pkl.exists():
        sys.exit(f"ERROR: model not found at {pkl}. Run train_classifier.py first.")

    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    saved_features = feat_file.read_text().strip().splitlines()
    if saved_features != FEATURE_NAMES:
        sys.exit(
            f"ERROR: Feature order mismatch.\n"
            f"  model/feature_order.txt : {saved_features}\n"
            f"  This script expects     : {FEATURE_NAMES}\n"
            f"Retrain or align feature sets."
        )

    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return clf, meta


def metrics_report(windows, ground_truth_label: int):
    """
    Compute Recall, FAR, and Detection Latency.

    ground_truth_label:
      1 => this sequence is abnormal  (recall + latency apply)
      0 => this sequence is normal    (FAR applies)
    """
    labels = [w["predicted_label"] for w in windows]

    if ground_truth_label == 1:
        # Recall: was any window flagged abnormal?
        detected = any(l == "abnormal" for l in labels)
        recall = 1.0 if detected else 0.0
        # Detection latency: index of first abnormal window
        latency_windows = next(
            (i for i, l in enumerate(labels) if l == "abnormal"), None
        )
        latency_frames = (
            windows[latency_windows]["window_start_frame"]
            if latency_windows is not None else None
        )
        print(f"\n--- Evaluation (ground truth: ABNORMAL) ---")
        print(f"  Recall          : {recall:.2f}")
        print(f"  First alarm at  : window {latency_windows}, "
              f"frame {latency_frames}")
        if recall == 0.0:
            print("  FAILURE CASE: model did not detect any abnormal window.")

    else:
        n_alarms = sum(1 for l in labels if l == "abnormal")
        n_windows = len(labels)
        far = n_alarms / max(n_windows, 1)
        print(f"\n--- Evaluation (ground truth: NORMAL) ---")
        print(f"  Total windows   : {n_windows}")
        print(f"  False alarms    : {n_alarms}")
        print(f"  FAR             : {far:.4f} ({far*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly classifier on a frame directory."
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory of scene-NNNNN.png frames."
    )
    parser.add_argument(
        "--fps", required=True, type=float,
        help="Source video FPS (required — not auto-detected)."
    )
    parser.add_argument(
        "--model_dir", default="model",
        help="Directory with classifier.pkl and feature_order.txt (default: model/)."
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help="Sliding window size in frames, must match training (default: 30)."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Path for the output results CSV."
    )
    parser.add_argument(
        "--ground_truth", type=int, choices=[0, 1], default=None,
        help="0=normal sequence, 1=abnormal sequence. "
             "Enables Recall/FAR/Latency reporting."
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    model_dir = Path(args.model_dir)

    clf, meta = load_model(model_dir)

    if meta:
        print(f"Model trained at  : {meta.get('trained_at', 'unknown')}")
        print(f"Training incidents: {meta.get('training_incidents', [])}")
        print(f"Holdout           : {meta.get('holdout_incident', '')}")
        print(f"Note              : {meta.get('note', '')}\n")

    # Reuse extract_features logic inline to avoid subprocess
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "extract_features",
            Path(__file__).parent / "extract_features.py"
        )
        ef = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ef)
    except Exception as e:
        sys.exit(f"ERROR: Could not import extract_features.py: {e}")

    print(f"Extracting features from: {image_dir}")
    per_frame = list(ef.extract_per_frame(image_dir, args.fps))
    if len(per_frame) < args.window:
        sys.exit(
            f"ERROR: only {len(per_frame)} frames, need >= {args.window}"
        )

    windows_feat = list(ef.window_aggregate(per_frame, args.window))

    results = []
    for wf in windows_feat:
        x = np.array([[wf[k] for k in FEATURE_NAMES]], dtype=np.float32)
        pred_int = clf.predict(x)[0]
        prob = clf.predict_proba(x)[0]
        label = "normal" if pred_int == 0 else "abnormal"
        confidence = float(prob[pred_int])
        results.append({
            "window_start_frame": wf["window_start_frame"],
            "window_end_frame": wf["window_end_frame"],
            "window_start_ts": wf["window_start_ts"],
            "predicted_label": label,
            "confidence": round(confidence, 4),
            **{k: round(wf[k], 5) for k in FEATURE_NAMES},
        })

    # Print summary
    n_abn = sum(1 for r in results if r["predicted_label"] == "abnormal")
    print(f"\nResults: {len(results)} windows | "
          f"normal={len(results)-n_abn} | abnormal={n_abn}")

    # Optional metrics
    if args.ground_truth is not None:
        metrics_report(results, args.ground_truth)

    # Write CSV
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_start_frame", "window_end_frame", "window_start_ts",
        "predicted_label", "confidence",
    ] + FEATURE_NAMES
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\nWrote {len(results)} rows -> {out}")
    print("\nREMINDER: 'abnormal' label must NOT be mapped to any risk tier "
          "without a separate integration review.")


if __name__ == "__main__":
    main()
