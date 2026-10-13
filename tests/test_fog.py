from fog.alerts import AlertEngine
from fog.classifier import RiskClassifier
from shared.schemas import FeatureVector, RiskLevel


def metrics(**overrides):
    values = dict(crowd_count=20, density=1, density_growth_rate=0, average_velocity=2, velocity_variance=1, direction_variance=0, congestion_index=0.1)
    values.update(overrides)
    return FeatureVector(**values)


def test_baseline_classifier_flags_critical():
    risk, _, reasons, _mode = RiskClassifier().assess(metrics(density=5, density_growth_rate=.7, direction_variance=.8, congestion_index=.9))
    assert risk is RiskLevel.CRITICAL
    assert reasons


def test_sustained_moderate_raises_alert():
    engine = AlertEngine()
    for _ in range(2):
        assert engine.evaluate("a", RiskLevel.MODERATE, ["test"]) is None
    event = engine.evaluate("a", RiskLevel.MODERATE, ["test"])
    assert event and event.active
