from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy import typing as npt

Node = int
Edge = Tuple[Node, Node]
Path = List[Node]
QValue = np.float_
Reward = np.float_
ResponseTime = np.float_


class BehavioralNetwork(metaclass=ABCMeta):
    @abstractmethod
    def construct_network(self, q_values: npt.NDArray[QValue], *args,
                          **kwargs):
        pass

    @abstractmethod
    def find_path(self, s: Node, t: Node) -> List[Path]:
        pass


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def construct_network(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_action_sequence(self, s: Node, t: Node) -> Path:
        pass

    @abstractmethod
    def update(self, s: Node, t: Node, reward: Reward):
        pass

    @abstractmethod
    def choose_action(self, rewards: npt.NDArray[Reward]) -> Node:
        pass

    @abstractmethod
    def engage_response(self, response: Node) -> ResponseTime:
        pass


class Schedule(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action: int, time: ResponseTime) -> Reward:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass
