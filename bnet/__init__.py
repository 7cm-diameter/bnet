__version__ = '0.1.0'

import numpy as np
from numpy.typing import NDArray


def _softmax(x: NDArray[np.float_], beta: float) -> NDArray[np.float_]:
    xmax = np.max(x)
    x_ = np.exp((x - xmax) * beta)
    return x_ / np.sum(x_)


def _propotional_allocation(x: NDArray[np.float_]) -> NDArray[np.float_]:
    probs = x / np.sum(x)
    return probs / np.sum(probs)
