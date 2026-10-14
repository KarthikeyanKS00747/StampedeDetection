"""
train_classifier.py
-------------------
Trains a sequence-level binary anomaly classifier (Normal / Abnormal) using
pre-extracted feature CSVs from extract_features.py.

Incident-based split (FIXED — do not change without updating metadata.json):
  TRAIN incidents : Las Vegas, Times Square, Love Parade
  HOLDOUT         : Italy  (NEVER used here — reserved for infer.py evaluation)

Each CSV must be produced by extract_features.py with a matching --fps value
and labelled as 0 (normal) or 1 (abnormal) via the --label flag below.

Usage
-----
  # First extract features for each split, e.g.:
  python extract_features.py --image_dir "2_Las_Vegas/Train"      --fps 15.17 --output_csv features/lv_train.csv
  python extract_features.py --image_dir "2_Las_Vegas/Test_1"     --fps 15.17 --output_csv features/lv_test1.csv
  python extract_features.py --image_dir "1_Times_Square/View_1/Train" --fps 29.97 --output_csv features/ts_v1_train.csv
  python extract_features.py --image_dir "1_Times_Square/View_1/Test"  --fps 29.97 --output_csv features/ts_v1_test.csv
  python extract_features.py --image_dir "3_Love Parade/Train_1"  --fps 25    --output_csv features/lp_train1.csv
  python extract_features.py --image_dir "3_Love Parade/Train_2"  --fps 25    --output_csv features/lp_train2.csv
  # (Love Parade Test is abnormal; include only if explicitly documenting this)

  # Then train:
  python train_classifier.py \
      --normal   features/lv_train.csv features/ts_v1_train.csv features/lp_train1.csv features/lp_train2.csv \
      --abnormal features/lv_test1.csv features/ts_v1_test.csv \
      --model_dir model

Notes
-----
- Italy is the holdout. Do NOT pass any Italy CSV here.
- All viewpoints of the same incident travel together (no cross-incident mixing).
- Classifier is Random Forest for interpretability and CPU inference.
- Outputs: model/classifier.pkl, model/feature_order.txt, model/metadata.json
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "N_mean", "N_std",
    "overlap_mean", "overlap_std",
    "motion_mean", "motion_std",
    "delta_N",
]


def load_csvs(paths, label: int):
    import csv
    X, y = [], []
    for p in paths:
        path = Path(p)
        if not path.exists():
            sys.exit(f"ERROR: CSV not found: {p}")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feat = [float(row[k]) for k in FEATURE_NAMES]
                except (KeyError, ValueError) as e:
                    sys.exit(
                        f"ERROR in {p}: missing or bad feature column. "
                        f"Was this file produced by extract_features.py? ({e})"
                    )
                X.append(feat)
                y.append(label)
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train binary anomaly classifier from feature CSVs."
    )
    parser.add_argument(
        "--normal", nargs="+", required=True,
        help="Feature CSVs labelled as normal (0)."
    )
    parser.add_argument(
        "--abnormal", nargs="+", required=True,
        help="Feature CSVs labelled as abnormal (1)."
    )
    parser.add_argument(
        "--model_dir", default="model",
        help="Directory to save model artefacts (default: model/)."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200,
        help="Number of Random Forest trees (default: 200)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            classification_report, confusion_matrix, recall_score
        )
        import sklearn
    except ImportError:
        sys.exit("ERROR: scikit-learn not installed. Run: pip install scikit-learn")

    print("Loading normal sequences...")
    X_norm, y_norm = load_csvs(args.normal, label=0)
    print(f"  {len(X_norm)} windows from {len(args.normal)} CSV(s)")

    print("Loading abnormal sequences...")
    X_abn, y_abn = load_csvs(args.abnormal, label=1)
    print(f"  {len(X_abn)} windows from {len(args.abnormal)} CSV(s)")

    X = np.array(X_norm + X_abn, dtype=np.float32)
    y = np.array(y_norm + y_abn, dtype=np.int32)

    print(f"\nTotal: {len(y)} windows  |  normal={sum(y==0)}  abnormal={sum(y==1)}")
    if sum(y == 0) == 0 or sum(y == 1) == 0:
        sys.exit("ERROR: Need at least one window in each class.")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    print("\nFitting Random Forest...")
    clf.fit(X, y)

    # In-sample metrics (for sanity only — real evaluation uses Italy holdout)
    preds = clf.predict(X)
    print("\nIn-sample confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, preds))
    print("\nIn-sample classification report:")
    print(classification_report(y, preds, target_names=["normal", "abnormal"]))
    print(
        "NOTE: These are in-sample metrics. True evaluation must use the "
        "Italy holdout via infer.py."
    )

    # Feature importances (useful for debugging)
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:20s} {imp:.4f}")

    # Save artefacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = model_dir / "classifier.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved classifier -> {pkl_path}")

    feat_order_path = model_dir / "feature_order.txt"
    feat_order_path.write_text("\n".join(FEATURE_NAMES) + "\n")
    print(f"Saved feature order -> {feat_order_path}")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "training_normal_csvs": args.normal,
        "training_abnormal_csvs": args.abnormal,
        "training_incidents": ["Las Vegas", "Times Square", "Love Parade"],
        "holdout_incident": "Italy (4_Italy) — NEVER used in training",
        "class_mapping": {"0": "normal", "1": "abnormal"},
        "note": (
            "This is an experimental model. It is NOT a validated stampede "
            "predictor. The YOLO + rule-based baseline remains primary."
        ),
    }
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata     -> {meta_path}")
    print("\nDone. Run inference with: python infer.py --help")


if __name__ == "__main__":
    main()
