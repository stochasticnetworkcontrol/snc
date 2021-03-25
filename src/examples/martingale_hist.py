import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
from src import snc as snc_types, snc as ps
import src.snc.simulation.utils.validation_utils as validation_utils

from src.snc.environments import examples
from src.snc.agents.hedgehog.workload import workload
import src.snc.simulation.utils.load_agents as load_agents
from src.snc.simulation.plot import Handler, ProgressBarHandler
from src.snc import BigStepHedgehogAgent
import src.snc.simulation.store_data.reporter as rep
import src.snc.environments.controlled_random_walk as crw


def run_simulations(
        num_sim: int, num_sim_steps: int, env: crw.ControlledRandomWalk,
        discount_factor: float) -> Tuple[List[snc_types.ActionSpace], List[snc_types.StateSpace]]:
    """ Run multiple simulations on a given model and return all the actions and states.

    :param num_sim: The number of simulations to run.
    :param num_sim_steps: The number of simulation steps to run for each simulation.
    :param env: the environment to stepped through.
    :param discount_factor: discount factor used to compute the long term cost function.
    """
    data_actions = []  # type: List[snc_types.ActionSpace]
    data_states = []  # type: List[snc_types.StateSpace]

    num_steps = num_sim * num_sim_steps
    # Set Up Handlers
    handlers = [
        ProgressBarHandler(num_simulation_steps=num_steps, trigger_frequency=1)
    ]  # type: List[Handler]

    # Create Reporter
    reporter = rep.Reporter(handlers=handlers)  # fill with handlers

    # Agent parameters
    overrides: Dict[str, Dict[str, Union[str, float]]] = {}
    ac_params, wk_params, si_params, po_params, hh_params, name, include_hedging \
        = load_agents.get_hedgehog_hyperparams(**overrides)

    for i in np.arange(num_sim):
        job_gen_seed = int(42 + i)
        np.random.seed(job_gen_seed)

        # Create Policy Simulator
        agent = BigStepHedgehogAgent(env, discount_factor, wk_params, hh_params, ac_params,
                                     si_params, po_params, include_hedging, name)
        simulator = ps.SncSimulator(env, agent, discount_factor=discount_factor)

        # Run Simulation
        data = simulator.run(num_simulation_steps=num_sim_steps, reporter=reporter,
                             job_gen_seed=job_gen_seed)

        data_actions.extend(data["action"])
        data_states.extend(data["state"])
    return data_actions, data_states


def compute_martingale(env: crw.ControlledRandomWalk, data_actions: List[snc_types.ActionSpace],
                       data_states: List[snc_types.StateSpace]) -> snc_types.Matrix:
    """ Compute the difference between the workload at time t+1 and the workload at time t minus
    the drift and plus the idling process (w(t+1) - (w(t) - delta + I(t))). The result should be a
    martingale with zero mean.

    :param env: the environment to stepped through.
    :param data_actions: List of actions.
    :param data_states: List of states.
    """
    # TODO: create a unit test for this function with a specific environment where we are sure that
    #  the martingale has zero mean.
    workload_tuple = workload.compute_load_workload_matrix(
        env=env, num_wl_vec=None, load_threshold=None, feasible_tol=1e-10)
    workload_mat = workload_tuple.workload_mat
    nu = workload_tuple.nu

    w_process = workload_mat @ np.array(data_states).T

    sort_index = np.squeeze(validation_utils.workload_to_physical_resources_index(nu=nu))
    constituency_matrix = env.constituency_matrix[sort_index, :]
    actions = np.array(data_actions).T

    idling = np.ones((constituency_matrix.shape[0], actions.shape[1] - 1)
                     ) - constituency_matrix @ actions[:, :-1]
    rho = workload_mat @ env.job_generator.demand_rate
    delta_mat = np.tile(1. - rho, (1, w_process.shape[1] - 1))

    w_tplusone = w_process[:, 1:]
    w_t = w_process[:, :-1]

    martingale = w_tplusone - w_t + delta_mat - idling
    martingale_mean = np.mean(martingale, axis=1)
    print("Martingale mean: ", martingale_mean)
    return martingale


def plot_martingale_hist(martingale: snc_types.Matrix, plot_hist2d: Optional[bool] = False) \
        -> None:
    """ Plot a histogram for each workload of the martingale given as input.

    :param martingale: Result from compute_martingale().
    :param plot_hist2d: Whether to plot the histogram in 2D or not.
    """
    fig = plt.figure(figsize=(12, 6))
    num_workload_vectors = martingale.shape[0]
    for w in np.arange(num_workload_vectors):
        ax = fig.add_subplot(1, num_workload_vectors, w + 1)
        ax.hist(martingale[w, :], bins=20, density=True)

    if plot_hist2d:
        assert num_workload_vectors == 2
        fig2 = plt.figure(figsize=(6, 6))
        ax = fig2.add_subplot(1, 1, 1)
        ax.hist2d(martingale[0, :], martingale[1, :], bins=50, normed=True)
        plt.show()


def martingale_hist_double_reentrant_line_only_shared_resources_model(
        num_sim: Optional[int] = 10, num_sim_steps: Optional[int] = 200,
        job_gen_seed: Optional[int] = None) -> None:
    """ Run multiple simulations for the double re-entrant line with only shared resources model.
    Then, using all the actions and states simulated, compute the difference between the workload at
    time t+1 and the workload at time t minus the drift and plus the idling process
    (w(t+1) - (w(t) - delta + I(t))). Finally, plot a histogram of the result which should be a
    martingale with zero mean.

    :param num_sim: The number of simulations to run.
    :param num_sim_steps: The number of simulation steps to run for each simulation.
    :param job_gen_seed: Job generator random seed.
    """
    # Create Environment
    init_state = np.array([100, 100, 100, 100])[:, None]
    env = examples.double_reentrant_line_only_shared_resources_model(
        alpha=1, mu1=6, mu2=1.5, mu3=6, mu4=1.5, cost_per_buffer=np.array([1, 1, 1, 1])[:, None],
        initial_state=init_state, capacity=np.ones((4, 1)) * np.inf, job_conservation_flag=False,
        job_gen_seed=job_gen_seed)

    # Run Simulations
    data_actions, data_states = run_simulations(
        num_sim=num_sim, num_sim_steps=num_sim_steps, env=env, discount_factor=0.95)

    # Compute and Plot the Martingale Histogram
    martingale = compute_martingale(env=env, data_actions=data_actions, data_states=data_states)
    plot_martingale_hist(martingale=martingale, plot_hist2d=True)


if __name__ == '__main__':
    martingale_hist_double_reentrant_line_only_shared_resources_model()
