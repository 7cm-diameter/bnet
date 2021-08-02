import numpy as np
from bnet.env import ConcurrentSchedule
from hbtfmt.model import SarsaAgent
from hbtfmt.util import to_onehot


def train_under_varibale_ratio(agent: SarsaAgent, env: ConcurrentSchedule):
    n, _ = agent.weights.shape
    pseudo_rewards = np.ones(n)
    prev_response = agent.choose_action(pseudo_rewards)
    curr_response = agent.choose_action(pseudo_rewards)

    while not env.finished():
        onehot_action = to_onehot(curr_response, n)
        stepsize = onehot_action.copy()
        rewards = env.step(stepsize, onehot_action)
        reward = rewards[curr_response]
        next_action = agent.choose_action(pseudo_rewards)
        agent.update(prev_response, curr_response, reward, next_action)
        prev_response = curr_response
        curr_response = next_action

    return None


if __name__ == '__main__':
    from typing import List, Tuple

    from bnet.env import FixedRatio, VariableRatio
    from bnet.util import create_data_dir
    from hbtfmt.util import (ConfigClap, format_data2, free_access,
                             generate_rewards, parameter_product2)

    clap = ConfigClap()
    config = clap.config()
    params = parameter_product2(config)

    data_dir = create_data_dir(__file__, "bnet")
    filepath = data_dir.joinpath(config.filename)

    results: List[Tuple[float, int, float, int, int, float, float]] = []

    for param in params:
        ratio, n, ro, training = param
        rewards_baseline = generate_rewards(1., ro, n)
        rewards_test = generate_rewards(0., ro, n)

        print(f"ratio = {ratio}, n = {n}, ro = {ro}, training = {training}")

        for i in range(config.loop_per_condition):
            agent = SarsaAgent(n, 0.1, 2.5)
            variable_ratio = VariableRatio(ratio, training, .1, 1.)
            fixed_ratio = [FixedRatio(1, 10000, ro) for _ in range(n - 1)]
            concurrent_schedule = ConcurrentSchedule([variable_ratio] +
                                                     fixed_ratio)

            train_under_varibale_ratio(agent, concurrent_schedule)
            agent.construct_network(2)

            operant_baseline, total_baseline = \
                free_access(agent, rewards_baseline,
                            n, config.loop_per_simulation)
            operant_test, total_test = \
                free_access(agent, rewards_test,
                            n, config.loop_per_simulation)

            propotion_baseline = operant_baseline / total_baseline
            propotion_test = operant_test / total_test
            operant_degree = int(agent.network.degree[0])
            median_degree = int(np.median(agent.network.degree))
            result = param + (operant_degree, median_degree,
                              propotion_baseline, propotion_test)
            results.append(result)

    output_data = format_data2(results)
    output_data.to_csv(filepath, index=False)
