
import pytest

from evaluation import calculate_mae, calculate_rmse


def test_mae_assignment_example():
    assert calculate_mae(
        [10, 20, 30], [12, 17, 31]
    ) == pytest.approx(2)


def test_rmse_assignment_example():
    assert calculate_rmse(
        [10, 20, 30], [12, 17, 31]
    ) == pytest.approx((14 / 3) ** 0.5)


@pytest.mark.parametrize("metric", [calculate_mae, calculate_rmse])
def test_perfect_prediction(metric):
    assert metric([0, 10, 20], [0, 10, 20]) == 0


@pytest.mark.parametrize("metric", [calculate_mae, calculate_rmse])
def test_fractional_predictions(metric):
    assert metric(
        [1, 2], [1.5, 2.5]
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [calculate_mae, calculate_rmse])
@pytest.mark.parametrize("actual,predicted", [
    ([], []),
    ([1, 2], [1]),
    ([[1]], [[1]]),
    ([float("nan")], [1]),
    ([1], [float("inf")]),
])
def test_invalid_inputs(metric, actual, predicted):
    with pytest.raises(ValueError):
        metric(actual, predicted)
