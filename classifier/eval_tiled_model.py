"""Quick eval of model_tiled on Italy holdout CSVs (old-detector features)."""
import csv, json, pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["N_mean","N_std","overlap_mean","overlap_std","motion_mean","motion_std","delta_N"]

with open(ROOT / "classifier/model_tiled/classifier.pkl", "rb") as f:
    clf = pickle.load(f)

clips = {
    "italy_v1": ROOT / "classifier/results/italy_v1.csv",
    "italy_v2": ROOT / "classifier/results/italy_v2.csv",
}

by_clip = []
for name, path in clips.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    preds = clf.predict(X)
    TP = int(np.sum(preds == 1))
    FN = int(np.sum(preds == 0))
    print(f"{name}: {len(rows)} windows | TP={TP} FN={FN} | recall={TP/len(rows):.3f}")
    for i, (p, row) in enumerate(zip(preds, rows)):
        if p == 0:
            print(f"  [FN] w{i}: N_mean={float(row['N_mean']):.2f} "
                  f"overlap={float(row['overlap_mean']):.5f} "
                  f"motion={float(row['motion_mean']):.3f}")
    by_clip.append({"clip": name, "windows": len(rows), "TP": TP, "FN": FN})

total_w  = sum(c["windows"] for c in by_clip)
total_tp = sum(c["TP"] for c in by_clip)
total_fn = sum(c["FN"] for c in by_clip)
recall   = total_tp / total_w

print(f"\nTotal: {total_w} windows | TP={total_tp} FN={total_fn}")
print(f"Recall: {total_tp}/{total_w} = {recall*100:.1f}%")
print(f"Baseline (model_retrained_2026_10_15): 88/90 = 97.8%")
delta = (recall - 88/90) * 100
print(f"Delta vs baseline: {delta:+.1f} pp")

out = {
    "model": "model_tiled",
    "holdout_note": "Italy CSVs extracted with OLD detector - lower bound recall",
    "total_windows": total_w, "TP": total_tp, "FN": total_fn,
    "recall": round(recall, 4), "baseline_recall": round(88/90, 4),
    "delta_pp": round(delta, 2), "by_clip": by_clip,
}
p = ROOT / "classifier/results/tiled_italy_holdout.json"
p.write_text(json.dumps(out, indent=2))
print(f"Saved -> classifier/results/tiled_italy_holdout.json")
