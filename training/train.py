from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from shared.schemas import FEATURE_COLUMNS, RiskLevel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fog risk classifier from labelled feature windows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    missing = set(FEATURE_COLUMNS + ["risk"]) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    labels = set(frame.risk.unique())
    allowed = {level.value for level in RiskLevel}
    if not labels <= allowed:
        raise ValueError(f"Invalid risk values: {sorted(labels - allowed)}")
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURE_COLUMNS], frame.risk, test_size=0.2, stratify=frame.risk, random_state=42)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)


if __name__ == "__main__":
    main()
