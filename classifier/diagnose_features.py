"""Quick diagnostic: compare feature distributions between normal training and Italy holdout."""
import csv, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["N_mean","N_std","overlap_mean","overlap_std","motion_mean","motion_std","delta_N"]

def load(path):
    rows = []
    with open(ROOT / path) as f:
        for r in csv.DictReader(f):
            rows.append([float(r[k]) for k in FEATURES])
    return np.array(rows)

normal_files = [
    "classifier/features/lv_train.csv",
    "classifier/features/ts_v1_train.csv",
    "classifier/features/ts_v2_train.csv",
    "classifier/features/ts_v3_train.csv",
    "classifier/features/lp_train1.csv",
    "classifier/features/lp_train2.csv",
]
Xn = np.vstack([load(p) for p in normal_files])
Xi = np.vstack([load("classifier/results/italy_v1.csv"),
                load("classifier/results/italy_v2.csv")])

print("Feature means: NORMAL vs ITALY-ABNORMAL")
print(f"  {'Feature':<20} {'Normal_mean':>14} {'Italy_mean':>14} {'Ratio':>10}")
for i, f in enumerate(FEATURES):
    nm = Xn[:,i].mean()
    im = Xi[:,i].mean()
    ratio = im/nm if nm > 1e-9 else 9999.0
    print(f"  {f:<20} {nm:>14.4f} {im:>14.4f} {ratio:>10.2f}x")

print()
print("Feature 95th percentile: NORMAL vs ITALY")
for i, f in enumerate(FEATURES):
    np95 = np.percentile(Xn[:,i], 95)
    ip05 = np.percentile(Xi[:,i], 5)
    separable = "SEPARATE" if ip05 > np95 else "overlap"
    print(f"  {f:<20} normal_p95={np95:.4f}  italy_p05={ip05:.4f}  -> {separable}")
