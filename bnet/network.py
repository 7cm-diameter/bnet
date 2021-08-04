from typing import Callable, List

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import bnet.typing as tp


class SimpleBehavioralNetwork(tp.BehavioralNetwork, nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def construct_network(self, q_values: NDArray[tp.QValue], q2p: Callable,
                          min_):
        n, _ = q_values.shape
        for f in range(n):
            probs = q2p(q_values[f])
            t = np.random.choice(n, size=min_, p=probs, replace=False)
            [self.add_edge(f, t_) for t_ in t]

    def find_path(self, s: tp.Node, t: tp.Node) -> List[tp.Path]:
        if not s == t:
            return list(nx.all_shortest_paths(self, s, t))

        if self.has_edge(s, s):
            return [[s, t]]

        neighbors = list(nx.all_neighbors(self, s))
        s_ = np.random.choice(neighbors)
        return [[s, s_, t]]
