__version__ = '0.1.0'

import numpy as np
import scipy.stats as st
from numpy.typing import NDArray


def softmax(x: NDArray[np.float_], beta: float) -> NDArray[np.float_]:
    xmax = np.max(x)
    x_ = np.exp((x - xmax) * beta)
    return x_ / np.sum(x_)


def propotional_allocation(x: NDArray[np.float_]) -> NDArray[np.float_]:
    probs = x / np.sum(x)
    return probs / np.sum(probs)


def exp_rng(mean: float, n: int, _min: float) -> NDArray[np.float_]:
    rn = st.expon.ppf(np.linspace(0.001, 0.999, n), scale=mean, loc=_min)
    return np.random.choice(rn, size=n, replace=False)


def geom_rng(mean: float, n: int, _min: float) -> NDArray[np.int_]:
    rn = st.geom.ppf(np.linspace(0.001, 0.999, n), p=1 / mean, loc=_min)
    return np.random.choice(rn, size=n, replace=False)
