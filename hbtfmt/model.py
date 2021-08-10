import bnet.typing as tp
import numpy as np
import numpy.typing as npt
from bnet import propotional_allocation
from bnet.network import SimpleBehavioralNetwork


class HQAgent(tp.Agent):
    def __init__(self, n: int, q_operant: float, q_others: float):
        self._n = n
        q = np.full((n, n), q_others)
        q[0] = q_operant
        self._q_values = q
        self._net = SimpleBehavioralNetwork()

    def construct_network(self, min_: int):
        self._net.construct_network(self._q_values, propotional_allocation,
                                    min_)

    def generate_action_sequence(self, s: tp.Node, t: tp.Node) -> tp.Path:
        paths = self._net.find_path(s, t)
        i = np.random.choice(len(paths))
        return paths[i]

    def update(self, s: tp.Node, t: tp.Node, reward: tp.Reward):
        pass

    def choose_action(self, rewards: npt.NDArray[tp.Reward]) -> tp.Node:
        probs = propotional_allocation(rewards)
        return np.random.choice(self._n, p=probs)

    def engage_response(self, response: tp.Node) -> tp.ResponseTime:
        return np.float_(0.)

    @property
    def n(self) -> int:
        return self._n

    @property
    def network(self) -> SimpleBehavioralNetwork:
        return self._net


class QAgent(tp.Agent):
    def __init__(self, n: int, alpha: float,
                 response_times: npt.NDArray[tp.ResponseTime]):
        self._n = n
        self._alpha = alpha
        self._q_values = np.full((n, n), 1e-6)
        self._response_times = response_times
        self._net = SimpleBehavioralNetwork()

    def construct_network(self, min_: int):
        self._net.construct_network(self._q_values, propotional_allocation,
                                    min_)

    def generate_action_sequence(self, s: tp.Node, t: tp.Node) -> tp.Path:
        paths = self._net.find_path(s, t)
        i = np.random.choice(len(paths))
        return paths[i]

    def update(self, s: tp.Node, t: tp.Node, reward: tp.Reward):
        self._q_values[s][t] += self._alpha * (reward - self._q_values[s][t])

    def choose_action(self, rewards: npt.NDArray[tp.Reward]) -> tp.Node:
        probs = propotional_allocation(rewards)
        return np.random.choice(self._n, p=probs)

    def engage_response(self, response: tp.Node) -> tp.ResponseTime:
        return self._response_times[response]

    @property
    def n(self) -> int:
        return self._n

    @property
    def network(self) -> SimpleBehavioralNetwork:
        return self._net
