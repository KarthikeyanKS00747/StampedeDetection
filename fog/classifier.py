from __future__ import annotations

from shared.schemas import FEATURE_COLUMNS, FeatureVector, RiskLevel


class RiskClassifier:
    """Uses a trained model when supplied; otherwise a conservative explainable baseline."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        if model_path:
            try:
                import joblib
            except ImportError as error:
                raise RuntimeError("Install joblib and scikit-learn to load a trained classifier") from error
            self.model = joblib.load(model_path)

    def assess(self, metrics: FeatureVector) -> tuple[RiskLevel, dict[RiskLevel, float], list[str]]:
        if self.model is not None:
            row = [[getattr(metrics, column) for column in FEATURE_COLUMNS]]
            risk = RiskLevel(str(self.model.predict(row)[0]))
            raw = self.model.predict_proba(row)[0]
            probabilities = {level: 0.0 for level in RiskLevel}
            for label, value in zip(self.model.classes_, raw):
                probabilities[RiskLevel(str(label))] = float(value)
            return risk, probabilities, ["Trained classifier result"]
        return self._heuristic(metrics)

    @staticmethod
    def _heuristic(m: FeatureVector) -> tuple[RiskLevel, dict[RiskLevel, float], list[str]]:
        score = 0
        reasons: list[str] = []
        for condition, points, message in (
            (m.density >= 4.0, 3, "Very high crowd density"),
            (m.density >= 2.5, 2, "High crowd density"),
            (m.density_growth_rate >= 0.45, 2, "Rapid density growth"),
            (m.direction_variance >= 0.55, 2, "Erratic crowd direction"),
            (m.velocity_variance >= 250, 1, "High velocity variation"),
            (m.congestion_index >= 0.75, 2, "High congestion index"),
        ):
            if condition:
                score += points
                reasons.append(message)
        risk = RiskLevel.CRITICAL if score >= 6 else RiskLevel.HIGH if score >= 4 else RiskLevel.MODERATE if score >= 2 else RiskLevel.NORMAL
        # UI confidence only. These are not calibrated probabilities until a model is trained.
        values = {RiskLevel.NORMAL: 0.05, RiskLevel.MODERATE: 0.05, RiskLevel.HIGH: 0.05, RiskLevel.CRITICAL: 0.05}
        values[risk] = 0.85
        return risk, values, reasons or ["Metrics are within configured baseline limits"]
