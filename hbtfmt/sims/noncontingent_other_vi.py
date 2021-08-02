import numpy as np
from bnet.env import ConcurrentSchedule
from hbtfmt.model import TDAgent
from hbtfmt.util import to_onehot


def train_with_vi_with_noncontingent(agent: TDAgent, env: ConcurrentSchedule):
    n, _ = agent.weights.shape
    pseudo_rewards = np.ones(n)
    prev_response = agent.choose_action(pseudo_rewards)
    while not env.finished():
        curr_response = agent.choose_action(pseudo_rewards)
        onehot_action = to_onehot(curr_response, n)
        t = agent.take_time(curr_response)
        # since the reward of the noncontingent schedule is presented
        # independently of the response, its count is always counted down
        stepsize = np.full(n + 1, t)
        # since the reward of the noncontingent schedule is presented
        # independently of the response, set action as 1.
        onehot_action = np.append(onehot_action, 1)
        rewards = env.step(stepsize, onehot_action)
        reward = rewards[curr_response]
        # noncontingent reward is summed to the reward obtained from any response
        reward += rewards[-1]
        agent.update(prev_response, curr_response, reward)
        prev_response = curr_response


if __name__ == '__main__':
    from typing import List, Tuple

    from bnet.env import FixedRatio, VariableInterval
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
        interval, n, ro, training = param
        rewards_baseline = generate_rewards(1., ro / n, n)
        # distribute the value of noncontigent reward to all response
        rewards_baseline += 1. / n
        rewards_test = generate_rewards(0., ro / n, n)
        rewards_test += 1. / n

        print(
            f"interval = {interval}, n = {n}, ro = {ro}, training = {training}"
        )

        for i in range(config.loop_per_condition):
            agent = TDAgent(n, 0.01, 2.5)
            variable_interval = VariableInterval(interval, training, .1, 1.)
            noncontingent_schedule = VariableInterval(interval, training, .1,
                                                      1.)
            noncontingent_schedule.forever()
            other_vi = [
                VariableInterval(360, 10000, .1, ro / n) for _ in range(n - 2)
            ]

            concurrent_schedule = ConcurrentSchedule([variable_interval] +
                                                     other_vi +
                                                     [noncontingent_schedule])

            train_with_vi_with_noncontingent(agent, concurrent_schedule)
            agent.construct_network(2)

            result = run_baseline_test(agent, rewards_baseline, rewards_test,
                                       n, config.loop_per_simulation)
            result_with_params = param + result
            results.append(result_with_params)

    output_data = format_data2(results)
    output_data.to_csv(filepath, index=False)
