import numpy as np

import bnet.utype as ut


class TDLearner(ut.Learner):
    def __init__(self, n: int, alpha: float):
        self.__alpha = alpha
        self.__weights = np.full((n, n), 1e-6)

    def update(self, i: ut.Node, j: ut.Node, reward: ut.RewardValue):
        old_weight = self.__weights[i][j]
        td_err = reward - old_weight
        new_weight = old_weight + self.__alpha * td_err
        self.__weights[i][j] = new_weight

    @property
    def weights(self):
        return self.__weights


class SarsaLearner(ut.Learner):
    def __init__(self, n: int, alpha: float, gamma: float):
        self.__alpha = alpha
        self.__gamma = gamma
        self.__weights = np.full((n, n), 1e-6)

    def update(self, i: ut.Node, j: ut.Node, reward: ut.RewardValue,
               rt: ut.ResponseTime, k: ut.Node):
        old_weight = self.__weights[i][j]
        next_weight = self.__weights[j][k]
        td_err = (reward + self.__gamma * next_weight) - old_weight
        new_weight = old_weight + self.__alpha * td_err
        self.__weights[i][j] = new_weight

    @property
    def weights(self):
        return self.__weights
