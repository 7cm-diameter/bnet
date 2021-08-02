from typing import List

import networkx as nx
import numpy as np

import bnet.utype as ut


class WeightAgnosticNetwork(ut.Network, nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def construct_network(self, mindeg: int, weights: ut.WeightMatrix,
                          method: ut.ChoiceMethod, *args, **kwargs):
        n, _ = weights.shape
        for start in range(n):
            if method is ut.ChoiceMethod.Softmax:
                beta = kwargs.get("beta", 1.)
                probs = method(weights[start], beta)
            else:
                probs = method(weights[start])
            destinations = np.random.choice(n,
                                            size=mindeg,
                                            p=probs,
                                            replace=False)
            [self.add_edge(start, dest) for dest in destinations]

    def find_path(self, start: ut.Node, goal: ut.Node) -> List[ut.Path]:
        if not start == goal:
            return list(nx.all_shortest_paths(self, start, goal))

        has_self_loop = self.has_edge(start, start)
        if not has_self_loop:
            neignbors = list(nx.all_neighbors(self, start))
            start_ = np.random.choice(neignbors)
            paths = self.find_path(start_, goal)
            [path.insert(0, start) for path in paths]
            return paths
        return [[start, goal]]

    @property
    def network(self) -> "WeightAgnosticNetwork":
        return self
