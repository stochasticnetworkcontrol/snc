import numpy as np

from matplotlib import pyplot as plt

import snc.simulation.plot.plotting_utils
from snc.environments import examples
from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
import snc.simulation.snc_simulator as ps
import snc.simulation.plot.plotting_handlers as hand
import snc.simulation.store_data.reporter as rep
import snc.simulation.utils.load_agents as load_agents


def build_default_simple_reentrant_line_simulator(seed):
    """
    Helper function that returns a simulator to be used by the tests below.
    """

    env = examples.simple_reentrant_line_model(job_gen_seed=seed)
    overrides = {}
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = load_agents.get_hedgehog_hyperparams(**overrides)

    # Create Policy Simulator
    discount_factor = 0.95
    agent = BigStepHedgehogAgent(env, discount_factor, wk_params, hh_params, ac_params, si_params,
                                 po_params, si_class, dp_params, name)
    return ps.SncSimulator(env, agent, discount_factor=discount_factor)


def test_live_plotting_to_existing_axes():
    """Test live plotting works on existing axes."""

    seed = 42
    np.random.seed(seed)
    num_simulation_steps = 30
    time_interval = 1
    plot_frequency = 10

    hedgehog_simulator = build_default_simple_reentrant_line_simulator(seed)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    handlers = [hand.StateCostPlotter(num_simulation_steps, time_interval, plot_frequency,
                                      ax=ax1, testing_mode=True)]
    reporter = rep.Reporter(handlers=handlers)

    # Run Simulation
    with snc.simulation.plot.plotting_utils.non_gui_matplotlib():
        hedgehog_simulator.run(num_simulation_steps, reporter=reporter)

    for h in handlers:
        assert h.data_cache.shape[1] == num_simulation_steps
        assert not np.all(h.data_cache == np.zeros_like(h.data_cache))


def test_multiple_handlers():
    """Test live plotting works with multiple handlers """

    seed = 42
    np.random.seed(seed)
    num_simulation_steps = 30
    time_interval = 1
    plot_frequency = 10

    hedgehog_simulator = build_default_simple_reentrant_line_simulator(seed)

    handlers = [
        hand.StateCostPlotter(num_simulation_steps, time_interval, plot_frequency,
                              testing_mode=True),
        hand.WorkloadPlotter(num_simulation_steps, time_interval, plot_frequency, testing_mode=True)
    ]

    reporter = rep.Reporter(handlers=handlers)

    # Run Simulation
    with snc.simulation.plot.plotting_utils.non_gui_matplotlib():
        hedgehog_simulator.run(num_simulation_steps, reporter=reporter)

    for h in handlers:
        assert h.data_cache.shape[1] == num_simulation_steps
        assert not np.all(h.data_cache == np.zeros_like(h.data_cache))
