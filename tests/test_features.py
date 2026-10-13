from edge.features import FeatureExtractor, circular_variance


def test_circular_variance_opposed_vectors_is_high():
    assert circular_variance([(1, 0), (-1, 0)]) > 0.99


def test_density_growth_is_positive():
    subject = FeatureExtractor(area_m2=10)
    subject.extract("z", 0, 10, [], [])
    result = subject.extract("z", 2, 30, [], [])
    assert result.density == 3
    assert result.density_growth_rate == 1
