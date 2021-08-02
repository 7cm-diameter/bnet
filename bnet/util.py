from pathlib import Path as _Path
from typing import Any

import numpy as np
import scipy.stats as st
from nptyping import NDArray


def exp_rng(mean: float, n: int, _min: float) -> NDArray[1, float]:
    return st.expon.ppf(np.linspace(0.001, 0.999, n),
                        scale=(mean - _min),
                        loc=_min)


def geom_rng(mean: float, n: int, _min: float) -> NDArray[1, int]:
    return st.geom.ppf(np.linspace(0.001, 0.999, n),
                       p=(1 / (mean - _min)),
                       loc=_min)


def randomize(v: NDArray[1, Any]) -> NDArray[1, Any]:
    return np.random.choice(v, size=len(v), replace=False)


def uniform(x: NDArray[1, float]) -> NDArray[1, float]:
    n = len(x)
    probs = np.array([1 / n for _ in range(n)])
    return probs / np.sum(probs)


def propotional_allocation(x: NDArray[1, float]) -> NDArray[1, float]:
    probs = x / np.sum(x)
    return probs / np.sum(probs)


def softmax(x: NDArray[1, float], beta: float = 1.) -> NDArray[1, float]:
    x_ = x - np.max(x)  # countermeasure aginst oveflow
    y = np.exp(x_ * beta)
    probs = y / np.sum(y)
    return probs / np.sum(probs)


def get_current_dir(relpath: str) -> _Path:
    return _Path(relpath).absolute()


def create_data_dir(relpath: str, parent: str):
    cur_dir = get_current_dir(relpath)
    target_dir = cur_dir
    while not target_dir.stem == parent:
        target_dir = target_dir.parent
    data_dir = target_dir.joinpath("data")
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir
