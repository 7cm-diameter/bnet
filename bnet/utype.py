from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple

from nptyping import NDArray

from bnet.util import propotional_allocation, softmax, uniform

Node = int
Edge = Tuple[Node, Node]
Path = List[Node]
Weight = float
WeightVector = NDArray[1, Weight]
WeightMatrix = NDArray[2, Weight]
RewardValue = float
RewardValues = NDArray[1, RewardValue]
ResponseTime = float
ResponseTimes = NDArray[1, ResponseTime]
Action = int
Parameters = Dict[str, Any]


class ChoiceMethod(Enum):
    Uniform = uniform
    Propotional = propotional_allocation
    Softmax = softmax


class Network(metaclass=ABCMeta):
    @abstractmethod
    def construct_network(self, weights: WeightMatrix, method: ChoiceMethod,
                          *args, **kwargs):
        pass

    @abstractmethod
    def find_path(self, start: Node, goal: Node) -> List[Path]:
        pass

    @property
    @abstractmethod
    def network(self) -> "Network":
        pass


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def update(self, i: Node, j: Node, reward: RewardValue, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def weights(self) -> WeightMatrix:
        pass


class Actor(metaclass=ABCMeta):
    @abstractmethod
    def choose_action(self, weights: NDArray[1, float], *args,
                      **kwargs) -> Node:
        pass

    @abstractmethod
    def take_time(self, i: Node) -> ResponseTime:
        pass


class Agent(Network, Learner, Actor):
    pass
