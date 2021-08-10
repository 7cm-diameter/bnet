if __name__ == '__main__':
    import numpy as np
    from hbtfmt import free_access
    from hbtfmt.model import HQAgent

    N = 50
    q_operant = 1.
    q_other = 0.01
    reward_operant = 1.
    reward_others = 0.1

    agent = HQAgent(N, 1, 0.001)
    agent.construct_network(2)
    rewards = np.full(N, reward_others)
    rewards[0] = reward_operant  # 0 denotes the operant response

    pre_devaluation_propotion = free_access(agent, rewards, 500)
    # reduce the value of reward obtained by the operant response
    # by setting the value of `rewards[0]` to 0
    rewards[0] = 0.
    post_devaluation_propotion = free_access(agent, rewards, 500)
