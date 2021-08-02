from typing import List

import bnet.utype as ut
import numpy as np
from bnet.actor import PropotionalActor
from bnet.learner import SarsaLearner, TDLearner
from bnet.network import WeightAgnosticNetwork
from bnet.util import exp_rng
from nptyping import NDArray


class HQAgent(ut.Agent):
    def __init__(self, q_opeant: float, q_other: float, n: int):
        q_ = np.full(n, q_other)
        q_[0] = q_opeant
        self.__weights: NDArray[2, float] = np.outer(q_, q_)
        self.__actor = PropotionalActor(np.zeros(n))
        self.__net = WeightAgnosticNetwork()

    def update(self):
        pass

    @property
    def weights(self) -> ut.WeightMatrix:
        return self.__weights

    def construct_network(self, mindeg: int):
        self.__net.construct_network(mindeg, self.weights,
                                     ut.ChoiceMethod.Propotional)

    def find_path(self, start: ut.Node, goal: ut.Node) -> List[ut.Path]:
        return self.__net.find_path(start, goal)

    @property
    def network(self) -> ut.Network:
        return self.__net

    def choose_action(self, weights: ut.RewardValues) -> ut.Node:
        return self.__actor.choose_action(weights)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__actor.take_time(i)


class TDAgent(ut.Agent):
    def __init__(self, n: int, alpha: float, rt: float):
        self.__learner = TDLearner(n, alpha)
        rts = exp_rng(rt, n, 0.1)
        self.__actor = PropotionalActor(rts)
        self.__net = WeightAgnosticNetwork()

    def update(self, i: ut.Node, j: ut.Node, reward: ut.RewardValue):
        self.__learner.update(i, j, reward)

    @property
    def weights(self) -> ut.WeightMatrix:
        return self.__learner.weights

    def construct_network(self, mindeg: int):
        self.__net.construct_network(mindeg, self.weights,
                                     ut.ChoiceMethod.Propotional)

    def find_path(self, start: ut.Node, goal: ut.Node) -> List[ut.Path]:
        return self.__net.find_path(start, goal)

    @property
    def network(self) -> ut.Network:
        return self.__net

    def choose_action(self, weights: ut.RewardValues) -> ut.Node:
        return self.__actor.choose_action(weights)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__actor.take_time(i)


class SarsaAgent(ut.Agent):
    def __init__(self, n: int, alpha: float, rt: float):
        self.__learner = SarsaLearner(n, alpha, 0.9)
        rts = exp_rng(rt, n, 0.1)
        self.__actor = PropotionalActor(rts)
        self.__net = WeightAgnosticNetwork()

    def update(self, i: ut.Node, j: ut.Node, reward: ut.RewardValue,
               k: ut.Node):
        rt = self.take_time(k)
        self.__learner.update(i, j, reward, rt, k)

    @property
    def weights(self) -> ut.WeightMatrix:
        return self.__learner.weights

    def construct_network(self, mindeg: int):
        self.__net.construct_network(mindeg, self.weights,
                                     ut.ChoiceMethod.Propotional)

    def find_path(self, start: ut.Node, goal: ut.Node) -> List[ut.Path]:
        return self.__net.find_path(start, goal)

    @property
    def network(self) -> ut.Network:
        return self.__net

    def choose_action(self, weights: ut.RewardValues) -> ut.Node:
        return self.__actor.choose_action(weights)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__actor.take_time(i)
