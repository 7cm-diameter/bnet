if __name__ == '__main__':
    from typing import List, Tuple

    import numpy as np
    from bnet.util import create_data_dir
    from hbtfmt.model import HQAgent
    from hbtfmt.util import (ConfigClap, format_data1, generate_rewards,
                             parameter_product1, run_baseline_test)
    from networkx import to_numpy_matrix

    clap = ConfigClap()
    config = clap.config()
    params = parameter_product1(config)

    data_dir = create_data_dir(__file__, "bnet")
    filepath = data_dir.joinpath(config.filename)

    results: List[Tuple[int, float, float, float, int, int, float, float]] = []

    for param in params:
        n, ro, qop, qot = param
        rewards_baseline = generate_rewards(1., ro, n)
        rewards_test = generate_rewards(0., ro, n)

        print(f"n = {n}, ro = {ro}, qop = {qop}, qot  = {qot}")

        for i in range(config.loop_per_condition):
            agent = HQAgent(qop, qot, n)
            agent.construct_network(2)

            result = run_baseline_test(agent, rewards_baseline, rewards_test,
                                       n, config.loop_per_simulation)
            result_with_params = param + result
            results.append(result_with_params)

        mat = to_numpy_matrix(agent.network).astype(np.uint8)
        matpath = data_dir.joinpath(filepath.stem + f"-mat{-qop}.csv")
        np.savetxt(matpath, mat, delimiter=",", fmt="%d")

    output_data = format_data1(results)
    output_data.to_csv(filepath, index=False)
