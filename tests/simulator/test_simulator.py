import os
import simpy

import numpy as np

from sandbox.simulator.simulate import run_simulation, build_network
from sandbox.simulator.utils import proportion_below_zero

from meio.gsm.utils import read_supply_chain_from_txt
from meio.experiment.numerical_simulator import simulate


dirname = os.path.dirname(__file__)


def test_network_parsing():
    network_name = "bulldozer"
    env = simpy.Environment()
    network = build_network(env, None, network_name, dirname)


def test_against_numerical_simulator():
    np.random.seed(seed=8675309)
    nodes = run_simulation("basic_serial_network", dirname, logging_disabled=True)

    demand_failure = proportion_below_zero("inventory_position", nodes["Demand"])
    dist_failure = proportion_below_zero("inventory_position", nodes["Dist"])
    data_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(data_path, "../../meio/experiment/basic_serial_network_config.txt")
    stages = read_supply_chain_from_txt(path)
    # stages = read_supply_chain_from_txt(os.path.join(dirname, "basic_serial_network_config.txt"))
    policy = {"Demand": {"s": 0, "si": 3}, "Dist": {"s": 3, "si": 0}}

    n = 1000
    lam = 10
    np.random.seed(seed=8675309)
    demand_history = np.random.poisson(size=n, lam=lam)

    base_stocks = {"Demand": 51, "Dist": 393}
    casc_inv_histories = simulate(stages, policy, base_stocks, {}, demand_history,
                                  stockout_stages=None)

    np.testing.assert_almost_equal(demand_failure, np.mean(casc_inv_histories["Demand"] < 0),
                                   decimal=1)
    np.testing.assert_almost_equal(dist_failure, np.mean(casc_inv_histories["Dist"] < 0), decimal=1)

