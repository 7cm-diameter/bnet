import numpy as np
from bnet.env import ConcurrentSchedule
from hbtfmt.model import TDAgent
from hbtfmt.util import to_onehot


def train_under_concurrent_vi_vi(agent: TDAgent, env: ConcurrentSchedule):
    n, _ = agent.weights.shape
    pseudo_rewards = np.ones(n)
    prev_response = agent.choose_action(pseudo_rewards)
    while not env.finished():
        curr_response = agent.choose_action(pseudo_rewards)
        onehot_action = to_onehot(curr_response, n)
        t = agent.take_time(curr_response)
        stepsize = onehot_action.copy()
        stepsize[0] = t
        stepsize[1] = t
        rewards = env.step(stepsize, onehot_action)
        reward = rewards[curr_response]
        agent.update(prev_response, curr_response, reward)
        prev_response = curr_response


if __name__ == '__main__':
    from typing import List, Tuple

    from bnet.env import FixedRatio, VariableInterval
    from bnet.util import create_data_dir
    from hbtfmt.util import (ConfigClap, format_data2, generate_rewards,
                             parameter_product2, run_baseline_test)
    from networkx import to_numpy_matrix

    clap = ConfigClap()
    config = clap.config()
    params = parameter_product2(config)

    data_dir = create_data_dir(__file__, "bnet")
    filepath = data_dir.joinpath(config.filename)

    results: List[Tuple[float, int, float, int, int, float, float]] = []

    for param in params:
        interval, n, ro, training = param
        rewards_baseline = generate_rewards(1., ro, n)
        # since there are two operant responses set the second reward value 1.
        rewards_baseline[1] = 1.
        # only the reward obtained by one operant response is devalued.
        rewards_test = generate_rewards(0., ro, n)
        rewards_test[1] = 1.

        print(
            f"interval = {interval}, n = {n}, ro = {ro}, training = {training}"
        )

        for i in range(config.loop_per_condition):
            agent = TDAgent(n, 0.01, 2.5)
            variable_interval_0 = VariableInterval(interval, training, .1, 1.)
            variable_interval_1 = VariableInterval(interval, training, .1, 1.)
            fixed_ratio = [FixedRatio(1, 10000, ro) for _ in range(n - 2)]
            concurrent_schedule = ConcurrentSchedule(
                [variable_interval_0, variable_interval_1] + fixed_ratio)

            train_under_concurrent_vi_vi(agent, concurrent_schedule)
            agent.construct_network(2)

            result = run_baseline_test(agent, rewards_baseline, rewards_test,
                                       n, config.loop_per_simulation)
            result_with_params = param + result
            results.append(result_with_params)

        mat = to_numpy_matrix(agent.network).astype(np.uint8)
        matpath = data_dir.joinpath(filepath.stem + "-mat-choice.csv")
        np.savetxt(matpath, mat, delimiter=",", fmt="%d")

    output_data = format_data2(results)
    output_data.to_csv(filepath, index=False)
