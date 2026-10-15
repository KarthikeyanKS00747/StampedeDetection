"""Evaluate reweighted classifier on saved Italy holdout feature CSVs."""
import csv, json, pickle
import numpy as np
from pathlib import Path


# Must match the class defined in train_reweighted.py (needed for pickle)
class FeatureScaler:
    def __init__(self, weights):
        self.weights = weights

    def transform(self, X):
        return X * self.weights

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["N_mean","N_std","overlap_mean","overlap_std","motion_mean","motion_std","delta_N"]

with open(ROOT / "classifier/model_reweighted/classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open(ROOT / "classifier/model_reweighted/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

clips = {
    "italy_v1": ROOT / "classifier/results/italy_v1.csv",
    "italy_v2": ROOT / "classifier/results/italy_v2.csv",
}

by_clip = []
for name, path in clips.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)
    TP = int(np.sum(preds == 1))
    FN = int(np.sum(preds == 0))
    print(f"{name}: {len(rows)} windows | TP={TP} FN={FN} | recall={TP/len(rows):.3f}")
    for i, (p, row) in enumerate(zip(preds, rows)):
        if p == 0:
            print(f"  [FN] window {i}: N_mean={row['N_mean']} "
                  f"overlap_mean={row['overlap_mean']} motion_mean={row['motion_mean']}")
    by_clip.append({"clip": name, "windows": len(rows), "TP": TP, "FN": FN})

total_w  = sum(c["windows"] for c in by_clip)
total_tp = sum(c["TP"] for c in by_clip)
total_fn = sum(c["FN"] for c in by_clip)
recall   = total_tp / total_w

print(f"\nTotal: {total_w} windows | TP={total_tp} FN={total_fn} | recall={recall:.4f} ({recall*100:.1f}%)")
print(f"Baseline (model_retrained_2026_10_15): 88/90 = 97.8%")
delta = recall - 0.9778
print(f"Delta vs baseline: {delta:+.4f} ({delta*100:+.1f} pp)")

out = {
    "model": "model_reweighted",
    "total_windows": total_w,
    "true_positive_windows": total_tp,
    "false_negative_windows": total_fn,
    "recall": round(recall, 6),
    "baseline_recall": 0.9778,
    "delta_pp": round((recall - 0.9778) * 100, 2),
    "by_clip": by_clip,
}
out_path = ROOT / "classifier/results/reweighted_italy_holdout.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Saved -> {out_path.relative_to(ROOT)}")
