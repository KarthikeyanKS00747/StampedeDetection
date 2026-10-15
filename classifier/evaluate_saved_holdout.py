"""Score a fitted classifier on saved abnormal holdout feature rows only.

This avoids rerunning the detector or evaluating on training footage. Saved
result CSVs round features to five decimals, so predictions near a tree split
may differ from a full-precision re-extraction.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.infer import FEATURE_NAMES, load_model


def evaluate(model_dir: Path, paths: list[Path]) -> dict:
    model, _ = load_model(model_dir)
    by_clip = []
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"No feature rows in {path}")
        features = np.array(
            [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
            dtype=np.float32,
        )
        predictions = model.predict(features)
        if not set(map(int, predictions)) <= {0, 1}:
            raise ValueError("Model returned labels outside the binary contract")
        false_negative = int(np.sum(predictions == 0))
        changed = sum(
            ("abnormal" if int(pred) else "normal") != row["predicted_label"]
            for pred, row in zip(predictions, rows)
        )
        by_clip.append({
            "source": str(path),
            "positive_windows": len(rows),
            "true_positive_windows": len(rows) - false_negative,
            "false_negative_windows": false_negative,
            "changed_from_saved_prediction": changed,
        })
    total = sum(item["positive_windows"] for item in by_clip)
    fn = sum(item["false_negative_windows"] for item in by_clip)
    return {
        "model_dir": str(model_dir),
        "scope": "Italy abnormal holdout, cached rounded features; window-level",
        "positive_windows": total,
        "true_positive_windows": total - fn,
        "false_negative_windows": fn,
        "false_negative_window_fraction": fn / total,
        "false_positive_windows_independent": None,
        "precision_independent": None,
        "accuracy_independent": None,
        "combined_fusion_metrics": None,
        "reason_unavailable": (
            "No independent normal footage and no aligned baseline outputs "
            "for this holdout are available."
        ),
        "by_clip": by_clip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--abnormal", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.model_dir, args.abnormal)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
