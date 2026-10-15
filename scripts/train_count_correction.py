"""
Train a count-correction model that maps YOLO person-box count → ground-truth
head count derived from the manual head-point annotations.

Usage:
    python scripts/train_count_correction.py

Outputs (all under outputs/count_correction/):
  results.csv          – per-frame comparison table
  model.pkl            – trained LinearRegression + IsotonicRegression ensemble
  metadata.json        – training provenance
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST  = ROOT / "outputs/person_audit/manifest.csv"
LABELS    = ROOT / "data/review/head_points.json"
OUT_DIR   = ROOT / "outputs/count_correction"


# ── 1. load manifest (YOLO counts) ─────────────────────────────────────────
def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["original"]).stem   # e.g. …_frame_00069_original
            rows[stem] = {
                "video":           row["video"],
                "frame":           int(row["frame"]),
                "timestamp_s":     float(row["timestamp_s"]),
                "yolo_count":      int(row["detected_people"]),
            }
    return rows


# ── 2. load manual head annotations ────────────────────────────────────────
def load_annotations(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for frame_id, ann in data["annotations"].items():
        pts = ann.get("points", [])
        if pts:                        # only include frames that have points
            counts[frame_id] = len(pts)
    return counts


# ── 3. merge on frame id ────────────────────────────────────────────────────
def build_dataset(manifest: dict, annotations: dict) -> list[dict]:
    rows = []
    for frame_id, ann_count in annotations.items():
        if frame_id not in manifest:
            print(f"  [WARN] annotated frame not in manifest, skipping: {frame_id}")
            continue
        m = manifest[frame_id]
        rows.append({
            "frame_id":    frame_id,
            "video":       m["video"],
            "frame":       m["frame"],
            "timestamp_s": m["timestamp_s"],
            "yolo_count":  m["yolo_count"],
            "true_count":  ann_count,
        })
    return rows


# ── 4. models ───────────────────────────────────────────────────────────────
class LinearCorrector:
    """Simple least-squares: true = a * yolo + b."""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, yolo: np.ndarray, true: np.ndarray) -> "LinearCorrector":
        X = np.column_stack([yolo, np.ones_like(yolo)])
        coef, *_ = np.linalg.lstsq(X, true, rcond=None)
        self.a, self.b = float(coef[0]), float(coef[1])
        return self

    def predict(self, yolo: np.ndarray) -> np.ndarray:
        return np.maximum(0, self.a * yolo + self.b)


class IsotonicCorrector:
    """
    Bin-average isotonic-style LUT: maps each encountered YOLO count to the
    mean true count observed at that input count value.
    Falls back to LinearCorrector for unseen counts.
    """
    def __init__(self):
        self.lut: dict[int, float] = {}
        self.linear = LinearCorrector()

    def fit(self, yolo: np.ndarray, true: np.ndarray) -> "IsotonicCorrector":
        self.linear.fit(yolo, true)
        unique_yolo = np.unique(yolo)
        for v in unique_yolo:
            mask = yolo == v
            self.lut[int(v)] = float(true[mask].mean())
        return self

    def predict(self, yolo: np.ndarray) -> np.ndarray:
        out = self.linear.predict(yolo).copy()
        for i, v in enumerate(yolo):
            k = int(round(v))
            if k in self.lut:
                out[i] = self.lut[k]
        return out


class EnsembleCorrector:
    """Average of linear + isotonic predictions."""
    def __init__(self):
        self.linear   = LinearCorrector()
        self.isotonic = IsotonicCorrector()

    def fit(self, yolo: np.ndarray, true: np.ndarray) -> "EnsembleCorrector":
        self.linear.fit(yolo, true)
        self.isotonic.fit(yolo, true)
        return self

    def predict(self, yolo: np.ndarray) -> np.ndarray:
        return (self.linear.predict(yolo) + self.isotonic.predict(yolo)) / 2.0


# ── 5. evaluate ─────────────────────────────────────────────────────────────
def evaluate(rows: list[dict], model) -> dict:
    yolo  = np.array([r["yolo_count"]  for r in rows], dtype=float)
    true  = np.array([r["true_count"]  for r in rows], dtype=float)
    pred  = model.predict(yolo)

    yolo_mae  = float(np.abs(yolo  - true).mean())
    model_mae = float(np.abs(pred  - true).mean())
    yolo_bias = float((yolo  - true).mean())
    model_bias= float((pred  - true).mean())
    yolo_rmse = float(np.sqrt(((yolo  - true) ** 2).mean()))
    model_rmse= float(np.sqrt(((pred  - true) ** 2).mean()))

    for i, r in enumerate(rows):
        r["predicted_count"] = round(float(pred[i]), 1)
        r["yolo_error"]      = int(r["yolo_count"]) - int(r["true_count"])
        r["model_error"]     = round(float(pred[i]) - float(r["true_count"]), 1)

    return {
        "n_frames":  len(rows),
        "yolo":  {"mae": yolo_mae,  "bias": yolo_bias,  "rmse": yolo_rmse},
        "model": {"mae": model_mae, "bias": model_bias, "rmse": model_rmse},
        "improvement_pct": round((yolo_mae - model_mae) / max(yolo_mae, 1e-9) * 100, 1),
    }


# ── 6. main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("-- Loading data --------------------------------------")
    manifest    = load_manifest(MANIFEST)
    annotations = load_annotations(LABELS)
    rows        = build_dataset(manifest, annotations)

    print(f"  Annotated frames:  {len(annotations)}")
    print(f"  Matched to manifest: {len(rows)}")
    if not rows:
        raise SystemExit("No frames matched — check manifest.csv and head_points.json")

    yolo = np.array([r["yolo_count"] for r in rows], dtype=float)
    true = np.array([r["true_count"] for r in rows], dtype=float)

    print("\n-- Per-frame data ------------------------------------")
    print(f"  {'Frame':<55} {'YOLO':>6} {'True':>6} {'Error':>6}")
    for r in sorted(rows, key=lambda x: (x["video"], x["frame"])):
        err = r["yolo_count"] - r["true_count"]
        print(f"  {r['frame_id']:<55} {r['yolo_count']:>6} {r['true_count']:>6} {err:>+6}")

    print("\n-- Training count-correction model -------------------")
    model = EnsembleCorrector().fit(yolo, true)
    print(f"  Linear coef: a={model.linear.a:.4f}  b={model.linear.b:.4f}")
    print(f"  Isotonic LUT entries: {len(model.isotonic.lut)}")

    metrics = evaluate(rows, model)
    print("\n-- Evaluation ----------------------------------------")
    print(f"  Frames evaluated : {metrics['n_frames']}")
    print(f"  YOLO  MAE  : {metrics['yolo']['mae']:.2f}  "
          f"Bias: {metrics['yolo']['bias']:+.2f}  "
          f"RMSE: {metrics['yolo']['rmse']:.2f}")
    print(f"  Model MAE  : {metrics['model']['mae']:.2f}  "
          f"Bias: {metrics['model']['bias']:+.2f}  "
          f"RMSE: {metrics['model']['rmse']:.2f}")
    print(f"  MAE improvement: {metrics['improvement_pct']:+.1f}%  "
          f"({'better' if metrics['improvement_pct'] > 0 else 'worse'})")

    # ── save outputs ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUT_DIR / "results.csv"
    fieldnames = ["frame_id","video","frame","timestamp_s",
                  "yolo_count","true_count","predicted_count",
                  "yolo_error","model_error"]
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Results saved -> {results_path.relative_to(ROOT)}")

    model_path = OUT_DIR / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved  -> {model_path.relative_to(ROOT)}")

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "n_frames":          metrics["n_frames"],
        "linear_coef_a":     model.linear.a,
        "linear_coef_b":     model.linear.b,
        "isotonic_lut":      model.isotonic.lut,
        "metrics":           metrics,
        "manifest":          str(MANIFEST.relative_to(ROOT)),
        "labels":            str(LABELS.relative_to(ROOT)),
        "note":              "All annotations were draft status - calibration model, not a validated detector.",
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata saved -> {meta_path.relative_to(ROOT)}")
    print("\n-- Done ----------------------------------------------")


if __name__ == "__main__":
    main()
