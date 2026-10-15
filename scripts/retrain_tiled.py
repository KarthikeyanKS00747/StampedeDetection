"""
retrain_tiled.py
----------------
Full pipeline: re-extract features with tiled detection, retrain the
Random Forest classifier, and evaluate on Italy holdout.

Run:
    python scripts/retrain_tiled.py

This replaces the old feature CSVs with tiled-detector versions, trains a new
model at classifier/model_tiled/, and prints Italy holdout results.

Tiled detection settings (see shared/tiled_detector.py):
  conf=0.05, tile=640, overlap=0.25, max_det=1000
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make project importable

# ── scene definitions ────────────────────────────────────────────────────────
# label=0 normal, label=1 abnormal
SCENES = [
    # (image_dir,                                               fps,   label, out_csv_stem)
    ("Abnormal High-density Crowds/2_Las_Vegas/Train",         15.17, 0, "lv_train"),
    ("Abnormal High-density Crowds/1_Times_Square/View_1/Train", 29.97, 0, "ts_v1_train"),
    ("Abnormal High-density Crowds/1_Times_Square/View_2/Train", 29.97, 0, "ts_v2_train"),
    ("Abnormal High-density Crowds/1_Times_Square/View_3/Train", 29.97, 0, "ts_v3_train"),
    ("Abnormal High-density Crowds/3_Love Parade/Train_1",     25.0,  0, "lp_train1"),
    ("Abnormal High-density Crowds/3_Love Parade/Train_2",     25.0,  0, "lp_train2"),
    # abnormal (in-sample — Las Vegas and Times Square test sets used in original)
    ("Abnormal High-density Crowds/2_Las_Vegas/Test_1",        15.17, 1, "lv_test1"),
    ("Abnormal High-density Crowds/1_Times_Square/View_1/Test", 29.97, 1, "ts_v1_test"),
]

HOLDOUT_SCENES = [
    ("classifier/results/italy_v1.csv", "italy_v1"),
    ("classifier/results/italy_v2.csv", "italy_v2"),
]

FEATURE_NAMES = [
    "N_mean", "N_std",
    "overlap_mean", "overlap_std",
    "motion_mean", "motion_std",
    "delta_N",
]

OUT_FEATURES = ROOT / "classifier/features_tiled"
MODEL_DIR    = ROOT / "classifier/model_tiled"


def extract_all() -> None:
    """Re-extract features for all scenes using tiled detection."""
    from classifier.extract_features import extract_per_frame, window_aggregate

    OUT_FEATURES.mkdir(parents=True, exist_ok=True)

    for rel_dir, fps, _label, stem in SCENES:
        image_dir = ROOT / rel_dir
        if not image_dir.is_dir():
            print(f"  [SKIP] directory not found: {image_dir}")
            continue
        out_csv = OUT_FEATURES / f"{stem}.csv"
        print(f"\n  Extracting: {rel_dir}  (fps={fps})")
        t0 = time.monotonic()
        per_frame = list(extract_per_frame(image_dir, fps, tiled=True, conf=0.05))
        windows   = list(window_aggregate(per_frame, 30))
        fieldnames = (["window_start_frame", "window_end_frame", "window_start_ts"]
                      + FEATURE_NAMES)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows({k: row[k] for k in fieldnames} for row in windows)
        elapsed = time.monotonic() - t0
        print(f"    -> {len(windows)} windows  ({elapsed:.0f}s)  saved: {out_csv.name}")


def load_csvs(paths: list[Path], label: int):
    X, y = [], []
    for p in paths:
        if not p.exists():
            print(f"  [WARN] missing CSV: {p}")
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                try:
                    X.append([float(row[k]) for k in FEATURE_NAMES])
                    y.append(label)
                except (KeyError, ValueError) as e:
                    sys.exit(f"Bad row in {p}: {e}")
    return X, y


def train() -> None:
    """Retrain RF on tiled features."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        import sklearn
    except ImportError:
        sys.exit("scikit-learn not installed")

    normal_csvs  = [OUT_FEATURES / f"{s}.csv" for _, _, l, s in SCENES if l == 0]
    abnormal_csvs = [OUT_FEATURES / f"{s}.csv" for _, _, l, s in SCENES if l == 1]

    print("\n  Loading normal windows...")
    Xn, yn = load_csvs(normal_csvs, 0)
    print(f"    {len(Xn)} windows from {len(normal_csvs)} CSVs")
    print("  Loading abnormal windows...")
    Xa, ya = load_csvs(abnormal_csvs, 1)
    print(f"    {len(Xa)} windows from {len(abnormal_csvs)} CSVs")

    X = np.array(Xn + Xa, dtype=np.float32)
    y = np.array(yn + ya, dtype=np.int32)
    print(f"\n  Total: {len(y)} windows | normal={sum(y==0)} abnormal={sum(y==1)}")

    # Report feature means so we can see how much tiling improved counts
    print("\n  Feature means (tiled vs old):")
    feat_arr = np.array(Xn, dtype=np.float32)
    old_means = {"N_mean": 0.286, "N_std": 0.068, "overlap_mean": 0.0001,
                 "motion_mean": 1.805, "delta_N": -0.003}
    for i, f in enumerate(FEATURE_NAMES):
        new_m = feat_arr[:, i].mean()
        old_m = old_means.get(f, "N/A")
        tag = f"  (was {old_m})" if f in old_means else ""
        print(f"    {f:<20} {new_m:.4f}{tag}")

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\n  Fitting Random Forest (n_estimators=300)...")
    clf.fit(X, y)

    preds = clf.predict(X)
    print("\n  In-sample confusion (rows=true, cols=pred):")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=["normal", "abnormal"]))

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("  Feature importances:")
    for name, imp in importances:
        print(f"    {name:<20} {imp:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "classifier.pkl").open("wb") as f:
        pickle.dump(clf, f)
    (MODEL_DIR / "feature_order.txt").write_text("\n".join(FEATURE_NAMES) + "\n")

    meta = {
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_estimators":   300,
        "seed":           42,
        "feature_names":  FEATURE_NAMES,
        "detection":      {"conf": 0.05, "tiled": True, "tile_size": 640,
                           "overlap": 0.25, "max_det": 1000},
        "training_incidents": ["Las Vegas", "Times Square", "Love Parade"],
        "holdout_incident":   "Italy -- NEVER used in training",
        "note": ("Retrained with tiled YOLO detection. "
                 "Features based on real head counts are now much higher."),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Model saved -> {MODEL_DIR}")
    return clf


def evaluate_holdout(clf) -> None:
    """Score on saved Italy holdout CSVs (cached from original extraction)."""
    print("\n" + "="*54)
    print("  ITALY HOLDOUT EVALUATION")
    print("="*54)
    print("  NOTE: Italy holdout CSVs were extracted with the OLD")
    print("  detector. This gives a lower-bound estimate. For a true")
    print("  evaluation, re-extract Italy frames with tiled detection.")
    print("="*54)

    baseline_total = 90
    baseline_tp    = 88

    by_clip = []
    for csv_rel, name in HOLDOUT_SCENES:
        path = ROOT / csv_rel
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
        preds = clf.predict(X)
        TP = int(np.sum(preds == 1))
        FN = int(np.sum(preds == 0))
        recall = TP / len(rows)
        print(f"\n  {name}: {len(rows)} windows | TP={TP} FN={FN} | recall={recall:.3f}")
        for i, (p, row) in enumerate(zip(preds, rows)):
            if p == 0:
                print(f"    [FN] w{i}: N_mean={row['N_mean']:.2f} "
                      f"overlap_mean={row['overlap_mean']:.5f} "
                      f"motion_mean={row['motion_mean']:.3f}")
        by_clip.append({"clip": name, "windows": len(rows), "TP": TP, "FN": FN})

    total_w  = sum(c["windows"] for c in by_clip)
    total_tp = sum(c["TP"] for c in by_clip)
    total_fn = sum(c["FN"] for c in by_clip)
    recall   = total_tp / total_w if total_w else 0.0

    print(f"\n  Total (old-detector features, lower bound):")
    print(f"    {total_w} windows | TP={total_tp} FN={total_fn}")
    print(f"    Recall: {total_tp}/{total_w} = {recall*100:.1f}%")
    print(f"    Baseline (model_retrained_2026_10_15): {baseline_tp}/{baseline_total} = "
          f"{baseline_tp/baseline_total*100:.1f}%")
    delta = (recall - baseline_tp / baseline_total) * 100
    print(f"    Delta vs baseline: {delta:+.1f} pp")

    out = {
        "model": "model_tiled",
        "holdout_note": "Italy CSVs extracted with OLD detector - lower bound",
        "total_windows": total_w,
        "true_positive_windows": total_tp,
        "false_negative_windows": total_fn,
        "recall": round(recall, 6),
        "baseline_recall": round(baseline_tp / baseline_total, 6),
        "delta_pp": round(delta, 2),
        "by_clip": by_clip,
    }
    out_path = ROOT / "classifier/results/tiled_italy_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved -> classifier/results/tiled_italy_holdout.json")
    return out


def main() -> None:
    print("=" * 54)
    print("  Step 1/3: Re-extract features (tiled detection)")
    print("=" * 54)
    extract_all()

    print("\n" + "=" * 54)
    print("  Step 2/3: Retrain classifier")
    print("=" * 54)
    clf = train()

    print("\n" + "=" * 54)
    print("  Step 3/3: Evaluate on Italy holdout")
    print("=" * 54)
    evaluate_holdout(clf)

    print("\n  DONE. See classifier/model_tiled/ for the new model.")


if __name__ == "__main__":
    main()
