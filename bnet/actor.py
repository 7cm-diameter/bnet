import numpy as np
from numpy.typing import NDArray

import bnet.typing as tp
from bnet import _propotional_allocation


class PropotionalActor(tp.Actor):
    def __init__(self, response_times: NDArray[tp.ResponseTime]):
        self._rt = response_times

    def choose_action(self, rewards: NDArray[tp.Reward]) -> tp.Node:
        n = len(rewards)
        probs = _propotional_allocation(rewards)
        return np.random.choice(n, p=probs)
