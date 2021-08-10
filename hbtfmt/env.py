from typing import List

import bnet.typing as tp
import numpy as np
from bnet import exp_rng, geom_rng
from numpy.typing import NDArray


class FixedRatio(tp.Schedule, tp.Repeatable):
    def __init__(self,
                 x: int,
                 n: int,
                 reward: tp.Reward,
                 repeat: bool = False):
        self._reward = reward
        self._n = n
        self._required_response = x
        self._cumulative_rewards = 0
        self._required_response_in_trial = x
        self._repeat = repeat

    def reset(self):
        self._cumulative_rewards = 0
        self._required_response_in_trial = self._required_response

    def finished(self) -> bool:
        if self._repeat:
            return True
        return self._cumulative_rewards >= self._n

    def step(self, action: int, time: tp.ResponseTime) -> tp.Reward:
        if self.finished():
            return np.float_(0.)
        self._required_response_in_trial -= action
        if self._required_response_in_trial <= 0:
            self._cumulative_rewards += 1
            if not self.finished():
                self._required_response_in_trial = self._required_response
            if self._repeat:
                self.reset()
            return self._reward
        return np.float_(0.)


class VariableRatio(tp.Schedule, tp.Repeatable):
    def __init__(self,
                 x: int,
                 n: int,
                 reward: tp.Reward,
                 repeat: bool = False):
        self._reward = reward
        self._n = n
        self._x = x
        self._required_response = geom_rng(x, n, 1)
        self._cumulative_rewards = 0
        self._required_response_in_trial = self._required_response[0]
        self._repeat = repeat

    def reset(self):
        self._required_response = geom_rng(self._x, self._n, 1)
        self._cumulative_rewards = 0
        self._required_response_in_trial = self._required_response[0]

    def finished(self) -> bool:
        if self._repeat:
            return True
        return self._cumulative_rewards >= self._n

    def step(self, action: int, time: tp.ResponseTime) -> tp.Reward:
        if self.finished():
            return np.float_(0.)
        self._required_response_in_trial -= action
        if self._required_response_in_trial <= 0:
            self._cumulative_rewards += 1
            if not self.finished():
                self._required_response_in_trial = \
                    self._required_response[self._cumulative_rewards]
            if self._repeat:
                self.reset()
            return self._reward
        return np.float_(0.)


class VariableInterval(tp.Schedule, tp.Repeatable):
    def __init__(self,
                 x: int,
                 n: int,
                 reward: tp.Reward,
                 repeat: bool = False):
        self._reward = reward
        self._n = n
        self._x = x
        self._intervals = exp_rng(x, n, 1)
        self._cumulative_rewards = 0
        self._interval = self._intervals[0]
        self._repeat = repeat

    def reset(self):
        self._intervals = exp_rng(self._x, self._n, 1)
        self._cumulative_rewards = 0
        self._interval = self._intervals[0]

    def finished(self) -> bool:
        if self._repeat:
            return True
        return self._cumulative_rewards >= self._n

    def step(self, action: int, time: tp.ResponseTime) -> tp.Reward:
        if self.finished():
            return np.float_(0.)
        self._interval -= time
        if self._interval <= 0. and action == 1:
            self._cumulative_rewards += 1
            if not self.finished():
                self._interval = \
                    self._intervals[self._cumulative_rewards]
            if self._repeat:
                self.reset()
            return self._reward
        return np.float_(0.)


class ConcurrentSchedule(object):
    def __init__(self, schedules: List[tp.Schedule]):
        self._schedules = schedules

    def finished(self) -> List[bool]:
        return [s.finished() for s in self._schedules]

    def step(self, actions: NDArray[np.int_],
             time: tp.ResponseTime) -> tp.Reward:
        reward = 0.
        for s, a in zip(self._schedules, actions):
            reward += s.step(a, time)
        return np.float_(reward)
