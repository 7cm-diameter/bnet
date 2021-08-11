import bnet.typing as tp
import numpy as np
from numpy.typing import NDArray

from hbtfmt.env import ConcurrentSchedule


def as_onehot(x: int, n: int) -> NDArray[np.int_]:
    return np.identity(n)[x]


def free_access(agent: tp.Agent, rewards: NDArray[np.float_],
                travel: int) -> float:
    number_of_operant = 0
    number_of_overall = 0
    current_response = np.random.choice(agent.n)
    for _ in range(travel):
        next_response = agent.choose_action(rewards)
        # To avoid choosing the same response over and over again
        while next_response == current_response:
            next_response = agent.choose_action(rewards)
        response_sequence = agent.generate_action_sequence(
            current_response, next_response)
        number_of_operant += int(
            0 in response_sequence)  # 0 denotes operant response
        number_of_overall += len(response_sequence)
        current_response = next_response
    return number_of_operant / number_of_overall


# We assumed that an agent's behavior as a choice
# between the operant response and other responses
# so all enviroments used in simulations are conccurent schedules
def train(agent: tp.Agent, env: ConcurrentSchedule):
    n = agent.n
    current_response = np.random.choice(n)
    main_schedule_continue = not env.finished()[0]

    while main_schedule_continue:
        next_response = np.random.choice(n)
        t = agent.engage_response(next_response)
        onehot_response = as_onehot(next_response, n)
        reward = env.step(onehot_response, t)
        agent.update(current_response, next_response, reward)
        current_response = next_response
        main_schedule_continue = not env.finished()[0]
