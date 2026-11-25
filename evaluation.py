
import numpy as np


def _validate_inputs(actual, predicted):
    """Validate paired, finite, one-dimensional scoring inputs."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Inputs must have equal nonzero length")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Inputs must be finite")

    return actual, predicted


def calculate_mae(actual, predicted):
    """Return mean absolute error."""
    actual, predicted = _validate_inputs(actual, predicted)

    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual, predicted):
    """Return root mean squared error."""
    actual, predicted = _validate_inputs(actual, predicted)

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
