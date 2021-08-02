import argparse as ap
from itertools import product
from typing import List, Tuple

import bnet.utype as ut
import numpy as np
import yaml
from nptyping import NDArray
from pandas.core.frame import DataFrame


class Config(dict):
    def __init__(self, path: str):
        f = open(path, "r")
        self.__path = path
        d: dict = yaml.safe_load(f)
        [self.__setitem__(item[0], item[1]) for item in d.items()]
        f.close()

    @property
    def n(self) -> List[int]:
        return self["n"]

    @property
    def ro(self) -> List[float]:
        return self["ro"]

    @property
    def q_operant(self) -> Tuple[float, float, float]:
        s = str(self["q-operant"]).lstrip("(").rstrip(")").split(",")
        return tuple(map(lambda c: float(c), s))

    @property
    def q_other(self) -> Tuple[float, float, int]:
        s = str(self["q-other"]).lstrip("(").rstrip(")").split(",")
        return tuple(map(lambda c: float(c), s))

    @property
    def amount_training(self) -> List[int]:
        return self["amount-training"]

    @property
    def schedule_value(self) -> List[float]:
        return self["schedule-value"]

    @property
    def loop_per_condition(self) -> int:
        return self["loop-per-condition"]

    @property
    def loop_per_simulation(self) -> int:
        return self["loop-per-simulation"]

    @property
    def filename(self) -> str:
        return self["filename"]


class ConfigClap(object):
    def __init__(self):
        self._parser = ap.ArgumentParser()
        self._parser.add_argument("--yaml",
                                  "-y",
                                  help="path to configuration file (`yaml`)",
                                  type=str)
        self._args = self._parser.parse_args()

    def config(self) -> Config:
        yml = self._args.yaml
        return Config(yml)


def generate_rewards(operant: ut.RewardValue, others: ut.RewardValue,
                     n: int) -> ut.RewardValues:
    rewards: ut.RewardValues = np.full(n, others)
    rewards[0] = operant
    return rewards


def free_access(agent: ut.Agent, rewards: ut.RewardValues, n: int, loop: int):
    current_action: ut.Node = np.random.choice(n)
    number_of_operant = 0
    number_of_all_response = 0
    for i in range(loop):
        next_action = agent.choose_action(rewards)
        if next_action == current_action:
            continue
        paths = agent.find_path(current_action, next_action)
        path = paths[np.random.choice(len(paths))][1:]
        number_of_operant += int(0 in path)
        number_of_all_response += len(path)
        current_action = next_action
    return number_of_operant, number_of_all_response


def run_baseline_test(agent: ut.Agent, rewards_baseline: ut.RewardValues,
                      rewards_test: ut.RewardValues, n: int,
                      loop: int) -> Tuple[int, int, float, float]:
    operant_baseline, total_baseline = free_access(agent, rewards_baseline, n,
                                                   loop)
    operant_test, total_test = free_access(agent, rewards_test, n, loop)

    operant_deg = int(agent.network.degree[0])
    median_deg = int(np.median(agent.network.degree))
    prop_baseline = operant_baseline / total_baseline
    prop_test = operant_test / total_test
    return operant_deg, median_deg, prop_baseline, prop_test


def format_data1(result: List[Tuple[int, float, float, float, int, int, float,
                                    float]]):
    columns = [
        "n", "ro", "q-operant", "q-other", "operant-degree", "other-degree",
        "baseline", "test"
    ]
    data = DataFrame(result, columns=columns)
    return data


def format_data2(result: List[Tuple[float, int, float, int, int, float,
                                    float]]):
    columns = [
        "schedule-value", "n", "other-reward", "amount-training",
        "operant-degree", "other-degree", "baseline", "test"
    ]
    data = DataFrame(result, columns=columns)
    return data


def parameter_product1(
        config: Config) -> List[Tuple[int, float, float, float]]:
    n = config.n
    ro = config.ro
    ops, ops2, opn = config.q_operant
    ots, ots2, otn = config.q_other
    qop: List[float] = list(np.linspace(ops, ops2, int(opn)))
    qot: List[float] = list(np.linspace(ots, ots2, int(otn)))
    return list(product(n, ro, qop, qot))


def parameter_product2(config: Config):
    sv = config.schedule_value
    n = config.n
    ro = config.ro
    at = config.amount_training
    return list(product(sv, n, ro, at))


def to_onehot(i: int, n: int) -> NDArray[1, int]:
    return np.identity(n)[i]
