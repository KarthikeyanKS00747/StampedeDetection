"""
train_reweighted.py
-------------------
Retrains the binary anomaly classifier with feature reweighting:
  - DOWN-weights: N_mean, N_std, delta_N  (YOLO-count-derived, unreliable in
                                            dense crowds per head annotation audit)
  - UP-weights:   overlap_mean, motion_mean  (geometry + motion, YOLO-count-independent)
  - NEUTRAL:      overlap_std, motion_std

Reweighting is implemented as feature scaling applied to X before fitting the
Random Forest.  The scaler is saved alongside the model and applied at inference.

Outputs -> classifier/model_reweighted/
  classifier.pkl      - trained RF
  scaler.pkl          - FeatureScaler object (apply before predict)
  feature_order.txt   - same 7 features, same order
  metadata.json       - provenance + weight config

Baseline to beat:
  model_retrained_2026_10_15  ->  88/90 Italy windows correct (97.8% recall)
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

FEATURE_NAMES = [
    "N_mean", "N_std",
    "overlap_mean", "overlap_std",
    "motion_mean", "motion_std",
    "delta_N",
]

# Multipliers applied to each feature before fitting.
# Values > 1 make the RF splits favour that feature.
# Values < 1 suppress the feature's influence.
FEATURE_WEIGHTS = {
    "N_mean":       0.2,   # YOLO count - severely unreliable in dense crowds
    "N_std":        0.2,   # derived from YOLO count, same problem
    "delta_N":      0.2,   # also count-derived
    "overlap_mean": 3.0,   # box geometry - crowd packing signal, count-independent
    "overlap_std":  1.0,   # neutral
    "motion_mean":  3.0,   # frame-to-frame pixel motion - stampede signal, count-independent
    "motion_std":   1.0,   # neutral
}

WEIGHT_VECTOR = np.array([FEATURE_WEIGHTS[f] for f in FEATURE_NAMES], dtype=np.float32)


class FeatureScaler:
    """Multiply features by a fixed weight vector. Applied before clf.predict."""
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.weights


def load_csvs(paths: list[str], label: int):
    X, y = [], []
    for p in paths:
        path = Path(p)
        if not path.exists():
            sys.exit(f"ERROR: CSV not found: {p}")
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    feat = [float(row[k]) for k in FEATURE_NAMES]
                except (KeyError, ValueError) as e:
                    sys.exit(f"ERROR in {p}: {e}")
                X.append(feat)
                y.append(label)
    return X, y


def main() -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        import sklearn
    except ImportError:
        sys.exit("scikit-learn not installed")

    # ── same training data as original ─────────────────────────────────────
    normal_csvs = [
        ROOT / "features/lv_train.csv",
        ROOT / "features/ts_v1_train.csv",
        ROOT / "features/ts_v2_train.csv",
        ROOT / "features/ts_v3_train.csv",
        ROOT / "features/lp_train1.csv",
        ROOT / "features/lp_train2.csv",
    ]
    abnormal_csvs = [
        ROOT / "features/lv_test1.csv",
        ROOT / "features/ts_v1_test.csv",
    ]

    print("Loading normal windows...")
    X_norm, y_norm = load_csvs([str(p) for p in normal_csvs], 0)
    print(f"  {len(X_norm)} windows from {len(normal_csvs)} CSVs")

    print("Loading abnormal windows...")
    X_abn, y_abn = load_csvs([str(p) for p in abnormal_csvs], 1)
    print(f"  {len(X_abn)} windows from {len(abnormal_csvs)} CSVs")

    X_raw = np.array(X_norm + X_abn, dtype=np.float32)
    y     = np.array(y_norm + y_abn, dtype=np.int32)
    print(f"\nTotal: {len(y)} windows | normal={sum(y==0)} abnormal={sum(y==1)}")

    # ── apply feature weights ────────────────────────────────────────────────
    scaler = FeatureScaler(WEIGHT_VECTOR)
    X = scaler.transform(X_raw)

    print("\nFeature weight config:")
    for name, w in FEATURE_WEIGHTS.items():
        tag = ("DOWN" if w < 1 else ("UP  " if w > 1 else "    "))
        print(f"  {tag}  {name:20s}  x{w}")

    # ── train ────────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nFitting Random Forest (n_estimators=300)...")
    clf.fit(X, y)

    # In-sample report
    preds = clf.predict(X)
    print("\nIn-sample confusion (rows=true, cols=pred):")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=["normal", "abnormal"]))

    # Feature importances on the SCALED space
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("Feature importances (on scaled features):")
    for name, imp in importances:
        print(f"  {name:20s}  {imp:.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    out_dir = ROOT / "model_reweighted"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (out_dir / "feature_order.txt").write_text("\n".join(FEATURE_NAMES) + "\n")

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "sklearn_version":   sklearn.__version__,
        "n_estimators":      300,
        "seed":              42,
        "feature_names":     FEATURE_NAMES,
        "feature_weights":   FEATURE_WEIGHTS,
        "weight_rationale":  (
            "N_mean/N_std/delta_N down-weighted 5x because head-annotation audit "
            "showed YOLO undercounts by 5-10x in dense crowd footage. "
            "overlap_mean and motion_mean up-weighted 3x as count-independent signals."
        ),
        "training_normal_csvs":  [str(p) for p in normal_csvs],
        "training_abnormal_csvs": [str(p) for p in abnormal_csvs],
        "baseline_italy_recall": "97.8% (88/90 windows, model_retrained_2026_10_15)",
        "holdout_incident":      "Italy (4_Italy) -- NEVER used in training",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model -> {out_dir}")
    print("Run holdout evaluation next.")


if __name__ == "__main__":
    main()
