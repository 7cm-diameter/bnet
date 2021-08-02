from abc import ABCMeta, abstractmethod
from typing import Any, List, Sequence, Union

import numpy as np
from nptyping import NDArray

import bnet.utype as ut
from bnet.util import exp_rng, geom_rng, randomize


class Schedule(metaclass=ABCMeta):
    @abstractmethod
    def step(
        self,
        stepsize: Union[Any, Sequence[Any]],
        action: ut.Action,
    ) -> Union[ut.RewardValue, Sequence[ut.RewardValue]]:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def forever(self):
        pass

    @abstractmethod
    def once(self):
        pass


class VariableInterval(Schedule):
    def __init__(self, val: float, n: int, _min: float,
                 reward: ut.RewardValue):
        self.__count = 0
        self.__reward = reward
        self.__n = n
        self.__intervals = randomize(exp_rng(val, n, _min))
        self.__interval = self.__intervals[self.__count]
        self.__repeat = False

    def step(self, stepsize: float, action: ut.Action) -> ut.RewardValue:
        if self.finished():
            return 0.
        self.__interval -= stepsize
        if self.__interval <= 0. and action == 1:
            self.__count += 1
            if not self.finished():
                self.__interval = self.__intervals[self.__count]
            elif self.__repeat:
                self.reset()
            return self.__reward
        return 0.

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__intervals = randomize(self.__intervals)
        self.__interval = self.__intervals[self.__count]

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False


class VariableRatio(Schedule):
    def __init__(self, val: float, n: int, _min: float,
                 reward: ut.RewardValue):
        self.__count = 0
        self.__reward = reward
        self.__n = n
        self.__required_responses = randomize(geom_rng(val, n, _min))
        self.__required_response = self.__required_responses[self.__count]
        self.__repeat = False

    def step(self, stepsize: float, action: ut.Action) -> ut.RewardValue:
        if self.finished():
            return 0.
        self.__required_response -= stepsize
        if self.__required_response <= 0. and action == 1:
            self.__count += 1
            if not self.finished():
                self.__required_response = self.__required_responses[
                    self.__count]
            elif self.__repeat:
                self.reset()
            return self.__reward
        return 0.

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__required_responses = randomize(self.__required_responses)
        self.__required_response = self.__required_responses[self.__count]

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False


class FixedRatio(Schedule):
    def __init__(self, val: int, n: int, reward: ut.RewardValue):
        self.__count = 0
        self.__reward = reward
        self.__n = n
        self.__required_response = val
        self.__required_response_ = val
        self.__repeat = False

    def step(self, stepsize: int, action: ut.Action) -> ut.RewardValue:
        if self.finished():
            return 0.
        self.__required_response_ -= stepsize
        if self.__required_response_ <= 0 and action == 1:
            self.__count += 1
            if not self.finished():
                self.__required_response_ = self.__required_response
            elif self.__repeat:
                self.reset()
            return self.__reward
        return 0.

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__required_response_ = self.__required_response

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False


class ConcurrentSchedule(Schedule):
    def __init__(self, schedules: List[Schedule]):
        self.__schedules = schedules
        self.__repeat = False

    def step(self, stepsize: Sequence[Any], action: Sequence[ut.Action]):
        if self.finished():
            return np.zeros(len(self.__schedules))
        rewards: NDArray[1, ut.RewardValue] = np.zeros(len(self.__schedules))
        for i in range(len(self.__schedules)):
            rew = self.__schedules[i].step(stepsize[i], action[i])
            rewards[i] = rew
        return rewards

    def finished(self) -> bool:
        return sum(s.finished() for s in self.__schedules) > 0

    def reset(self):
        for i in range(len(self.__schedules)):
            self.__schedules[i].reset()

    def forever(self):
        self.__repeat = True
        for i in range(len(self.__schedules)):
            self.__schedules[i].forever()

    def once(self):
        self.__repeat = False
        for i in range(len(self.__schedules)):
            self.__schedules[i].once()


class FreeAccess(Schedule):
    def __init__(self, n: int):
        self.__n = n
        self.__count = 0

    def step(self, stepsize: float, action: ut.Action) -> ut.RewardValue:
        self.__count += 1
        return 0.

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False
