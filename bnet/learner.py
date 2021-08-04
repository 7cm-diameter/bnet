import numpy as np
from numpy.typing import NDArray

import bnet.typing as tp


class QLearner(tp.Learner):
    def __init__(self, n: int, alpha: float):
        self._alpha = alpha
        self._q_values = np.full((n, n), 1e-6)

    def update(self, s: tp.Node, t: tp.Node, reward: tp.Reward):
        td_err = reward - self._q_values[s][t]
        self._q_values[s][t] += self._alpha * td_err

    @property
    def q_values(self) -> NDArray[tp.QValue]:
        return self._q_values
