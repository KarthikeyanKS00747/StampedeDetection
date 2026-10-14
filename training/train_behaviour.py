"""Train observable behaviour, never four-tier risk, from the review export."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from fog.anomaly import NAMES

LABELS = {"stationary", "walking", "running", "mixed_motion"}


def prepare(data):
    if data.get("schema") != "reviewed_behaviour_features_v1" or data.get("feature_names") != NAMES:
        raise ValueError("Wrong export schema or feature order")
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("No reviewed exportable windows; review data first")
    seen = set()
    for row in rows:
        if row.get("rights_approved") is not True or row.get("quality_eligible") is not True:
            raise ValueError("Every row requires permissions and quality eligibility")
        for key in ("window_id", "event_id", "reviewer", "review_note", "permission_evidence"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
        if row["window_id"] in seen:
            raise ValueError("Duplicate window")
        seen.add(row["window_id"])
        if row.get("behaviour_label") not in LABELS:
            raise ValueError("Only definite reviewed behaviours can be trained")
        if len(row["features"]) != len(NAMES) or not np.isfinite(row["features"]).all():
            raise ValueError("Invalid feature vector")
    x = np.asarray([r["features"] for r in rows], dtype=float)
    y = np.asarray([r["behaviour_label"] for r in rows])
    groups = np.asarray([r["event_id"].strip() for r in rows])
    labels = set(y)
    if len(labels) < 2:
        raise ValueError("Need at least two observed behaviour classes")
    for label in labels:
        if len(set(groups[y == label])) < 2:
            raise ValueError("Each behaviour needs at least two independent event groups")
    for train, test in GroupShuffleSplit(n_splits=300, test_size=.3, random_state=42).split(x, y, groups):
        if set(y[train]) == labels == set(y[test]):
            return x, y, groups, train, test
    raise ValueError("No event-separated holdout covering every class; add independent events")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    x, y, groups, train, test = prepare(data)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(pd.DataFrame(x[train], columns=NAMES), y[train])
    predicted = model.predict(pd.DataFrame(x[test], columns=NAMES))
    labels = sorted(set(y))
    bundle = {"artifact_kind": "reviewed_behaviour_classifier_v1", "pipeline": model,
              "feature_names": NAMES, "observed_classes": labels,
              "risk_classes": None, "deployable": False, "hardware_alarm_enabled": False}
    joblib.dump(bundle, output / "behaviour.joblib")
    report = {"train_rows": len(train), "test_rows": len(test),
              "train_events": sorted(set(groups[train])), "test_events": sorted(set(groups[test])),
              "classification_report": classification_report(y[test], predicted, output_dict=True, zero_division=0),
              "confusion_matrix_labels": labels, "confusion_matrix": confusion_matrix(y[test], predicted, labels=labels).tolist(),
              "warning": "Behaviour recognition, not stampede detection. Development holdout is not field safety validation."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
