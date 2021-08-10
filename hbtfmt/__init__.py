import bnet.typing as tp
import numpy as np
from numpy.typing import NDArray


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
