import numpy as np
from nptyping import NDArray

import bnet.utype as ut


class UniformActor(ut.Actor):
    def __init__(self, response_times: ut.ResponseTimes):
        self.__rt = response_times

    def choose_action(self, weights: NDArray[1, float], *args,
                      **kwargs) -> ut.Node:
        probs = ut.ChoiceMethod.Uniform(weights)
        n = len(weights)
        return np.random.choice(n, p=probs)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__rt[i]


class SoftmaxActor(ut.Actor):
    def __init__(self, response_times: ut.ResponseTimes):
        self.__rt = response_times

    def choose_action(self, weights: NDArray[1, float], *args,
                      **kwargs) -> ut.Node:
        beta = kwargs.get("beta", 1.)
        probs = ut.ChoiceMethod.Softmax(weights, beta)
        n = len(weights)
        return np.random.choice(n, p=probs)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__rt[i]


class PropotionalActor(ut.Actor):
    def __init__(self, response_times: ut.ResponseTimes):
        self.__rt = response_times

    def choose_action(self, weights: NDArray[1, float], *args,
                      **kwargs) -> ut.Node:
        probs = ut.ChoiceMethod.Propotional(weights)
        n = len(weights)
        return np.random.choice(n, p=probs)

    def take_time(self, i: ut.Node) -> ut.ResponseTime:
        return self.__rt[i]
