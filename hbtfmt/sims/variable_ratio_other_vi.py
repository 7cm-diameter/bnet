import numpy as np
from bnet.env import ConcurrentSchedule
from hbtfmt.model import TDAgent
from hbtfmt.util import to_onehot


def train_under_varibale_ratio(agent: TDAgent, env: ConcurrentSchedule):
    n, _ = agent.weights.shape
    pseudo_rewards = np.ones(n)
    prev_response = agent.choose_action(pseudo_rewards)

    while not env.finished():
        curr_response = agent.choose_action(pseudo_rewards)
        onehot_action = to_onehot(curr_response, n)
        rt = agent.take_time(curr_response)
        stepsize = np.full(n, rt)
        stepsize[0] = onehot_action[0]
        rewards = env.step(stepsize, onehot_action)
        reward = rewards[curr_response]
        agent.update(prev_response, curr_response, reward)
        prev_response = curr_response

    return None


if __name__ == '__main__':
    from typing import List, Tuple

    from bnet.env import VariableInterval, VariableRatio
    from bnet.util import create_data_dir
    from hbtfmt.util import (ConfigClap, format_data2, generate_rewards,
                             parameter_product2, run_baseline_test)

    clap = ConfigClap()
    config = clap.config()
    params = parameter_product2(config)

    data_dir = create_data_dir(__file__, "bnet")
    filepath = data_dir.joinpath(config.filename)

    results: List[Tuple[float, int, float, int, int, float, float]] = []

    for param in params:
        ratio, n, ro, training = param
        rewards_baseline = generate_rewards(1., ro / n, n)
        rewards_test = generate_rewards(0., ro / n, n)

        print(f"ratio = {ratio}, n = {n}, ro = {ro}, training = {training}")

        for i in range(config.loop_per_condition):
            agent = TDAgent(n, 0.1, 2.5)
            variable_ratio = VariableRatio(ratio, training, .1, 1.)
            other_vi = [
                VariableInterval(360, 10000, .1, ro / n) for _ in range(n - 1)
            ]
            concurrent_schedule = ConcurrentSchedule([variable_ratio] +
                                                     other_vi)

            train_under_varibale_ratio(agent, concurrent_schedule)
            agent.construct_network(2)

            result = run_baseline_test(agent, rewards_baseline, rewards_test,
                                       n, config.loop_per_simulation)
            result_with_params = param + result
            results.append(result_with_params)

    output_data = format_data2(results)
    output_data.to_csv(filepath, index=False)
