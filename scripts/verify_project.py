"""Write reproducible model-to-fog integration evidence; not an accuracy test."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
from fog.anomaly import AnomalyClassifier
from fog.app import app, state
from scripts.replay_ml import windows


def verify(model_path, features_path):
    previous, previous_latest = state.anomaly_classifier, state.anomaly_latest
    state.anomaly_classifier = AnomalyClassifier(model_path)
    state.anomaly_latest = {}
    try:
        results = []
        with TestClient(app) as client:
            for row in windows(features_path):
                response = client.post("/api/ml/windows", json=row.model_dump())
                response.raise_for_status()
                results.append(response.json())
        if not results:
            raise ValueError("No windows tested")
        return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(Path(model_path).resolve()), "features_path": str(Path(features_path).resolve()),
                "windows_processed": len(results), "observations": dict(Counter(r["observation"] for r in results)),
                "all_hardware_disabled": all(r["hardware_alarm_enabled"] is False for r in results),
                "risk_labels_created": False, "detection_accuracy_available": False,
                "integration_passed": True, "results": results,
                "warning": "Checks inference/API wiring only. No independent stampede ground truth or field validation."}
    finally:
        state.anomaly_classifier, state.anomaly_latest = previous, previous_latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/training/quality_motion_2026_10_14_run04/quality_joint.joblib")
    parser.add_argument("--features", default="outputs/training/quality_motion_2026_10_14_run04/features_quality.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a fresh report path")
    report = verify(args.model, args.features)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
