from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

from shared.schemas import FEATURE_COLUMNS, RiskLevel


def prepare_split(frame: pd.DataFrame, test_size: float = 0.2):
    """Reject unreviewed data and keep every crop/window of an event on one side."""
    required = FEATURE_COLUMNS + ["risk", "event_id", "reviewer", "rights_approved", "permission_evidence", "motion_units"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    for field in ("risk", "event_id", "reviewer", "permission_evidence", "motion_units"):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"All rows need a nonempty {field}")
    approved = frame.rights_approved.astype(str).str.strip().str.lower().eq("true")
    if not approved.all():
        raise ValueError("Training requires rights_approved=true and permission evidence for every row")
    if not frame.motion_units.eq("pixels_per_second").all():
        raise ValueError("Use consistent pixels_per_second motion features; calibrated metres require a separate contract")
    labels = set(frame.risk.unique())
    allowed = {level.value for level in RiskLevel}
    if not labels <= allowed:
        raise ValueError(f"Invalid risk values: {sorted(labels - allowed)}")
    if len(labels) < 2:
        raise ValueError("At least two reviewed risk classes are required")
    values = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite; no blanks, NaN or infinity")
    nonnegative = [col for col in FEATURE_COLUMNS if col != "density_growth_rate"]
    if (values[nonnegative] < 0).any().any():
        raise ValueError("Only density_growth_rate may be negative")
    if ((values[["direction_variance", "congestion_index"]] > 1).any().any()
            or (values.crowd_count % 1 != 0).any()):
        raise ValueError("Invalid bounded feature or noninteger crowd_count")
    groups = frame.event_id.astype(str).str.strip()
    # Different files from one scene/event must share an event_id. This cannot
    # be inferred safely from a filename or a crop, so human provenance matters.
    coverage = frame.assign(_event=groups).groupby("risk")._event.nunique()
    if (coverage < 2).any():
        raise ValueError("Each class needs at least two independent event groups for held-out evaluation")
    splitter = GroupShuffleSplit(n_splits=200, test_size=test_size, random_state=42)
    for train_idx, test_idx in splitter.split(values, frame.risk, groups):
        if set(frame.risk.iloc[train_idx]) == labels and set(frame.risk.iloc[test_idx]) == labels:
            return values, train_idx, test_idx, groups
    raise ValueError("Cannot make an event-separated split covering all observed classes; add independent events or adjust test_size")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fog risk classifier from labelled feature windows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    values, train_idx, test_idx, groups = prepare_split(frame, args.test_size)
    x_train, x_test = values.iloc[train_idx], values.iloc[test_idx]
    y_train, y_test = frame.risk.iloc[train_idx], frame.risk.iloc[test_idx]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "motion_units": "pixels_per_second",
        "train_events": sorted(set(groups.iloc[train_idx])),
        "test_events": sorted(set(groups.iloc[test_idx])),
        "observed_classes": sorted(set(frame.risk)),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "warning": "Event-separated development holdout, not an independent final safety validation. Unobserved classes are not validated.",
    }
    Path(args.output + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
