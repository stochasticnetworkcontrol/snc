from matplotlib import pyplot as plt
import numpy as np
from typing import List

from src.snc import AgentInterface
from src.snc import HedgehogAgentInterface
from src.snc import MaxWeightAgent
from src.snc.simulation.plot import plotting_handlers as hand
from src.snc.simulation.plot import (
    Handler,
    PrintActualtoFluidRatio,
    PrintInverseLoadings,
    PrintStateHedging
)
from src.snc.utils import snc_types as types


def print_workload_to_physical_resources_indexes(nu: types.NuMatrix) -> None:
    """
    Print the mapping of workload vectors to physical resources

    :param nu: the nu matrix from the workload tuple
    :return: None
    """

    workload_to_station_indexes = workload_to_physical_resources_index(nu)
    for workload_index, station_indexes in enumerate(workload_to_station_indexes):
        s = [str(i + 1) for i in station_indexes]
        station_list = (", ".join(s))
        if len(s) > 1:
            corresponds_str = "corresponds to Stations "
        else:
            corresponds_str = "correspond to Station "
        print("W_{}".format(workload_index + 1), corresponds_str, station_list)


def get_handlers(server_mode: bool, num_sim_steps: int, plot_freq: int,
                 time_interval: int, is_hedgehog: bool, is_routing: bool,
                 reproduce_mode: bool = False) -> List[Handler]:
    """
    Get the handlers used to validate the examples.

    :param server_mode: when experiment runs locally, this flag controls whether to show live plots
        and wait for input before closing them at the end of the simulation.
    :param num_sim_steps: the number of steps the simulation runs.
    :param plot_freq: number of timesteps after which the plots are updated.
    :param time_interval: the time interval of the environment.
    :param is_hedgehog: whether the use hedgehog specific handlers (eg for workload).
    :param is_routing: whether the environment is a routing model.
    :param reproduce_mode: flag indicating whether we are live plotting or reproducing from already
        stored data.

    :return: handlers: List of reporting handlers.
    """
    if server_mode:
        return []

    handlers = []  # type: List[Handler]
    compact_panel = False
    if compact_panel:
        rows = 3 if is_hedgehog else 2
        plots = 6 if is_hedgehog else 4

        fig = plt.figure(figsize=(12, 4 * rows))
        axs = [fig.add_subplot('{}2{}'.format(rows, i + 1)) for i in range(plots)]

        handlers.extend([
            hand.StateCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[0], do_plot_cost=False,
                reproduce_mode=reproduce_mode),
            hand.StateCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[1], do_plot_state=False,
                reproduce_mode=reproduce_mode),
            hand.CumulativeCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[3], discounted=False,
                reproduce_mode=reproduce_mode),
        ])
        if is_hedgehog:
            handlers.extend([
                hand.HedgingThresholdPlotter(
                    num_sim_steps, time_interval, plot_freq, ax=axs[2],
                    reproduce_mode=reproduce_mode),
                hand.EffectiveCostErrorPlotter(
                    num_sim_steps, time_interval, plot_freq, ax=axs[5],
                    eff_cost_err_method='absolute',
                    reproduce_mode=reproduce_mode),
                hand.IdlingPlotter(num_sim_steps, time_interval, plot_freq, ax=axs[4],
                                   reproduce_mode=reproduce_mode),
            ])
    else:
        rows = 3 if is_hedgehog else 2
        plots = 9 if is_hedgehog else 4
        plot_per_row = 3 if is_hedgehog else 2
        width = 17 if is_hedgehog else 12

        fig = plt.figure(figsize=(width, 3 * rows))
        axs = [fig.add_subplot(f'{rows}{plot_per_row}{i + 1}') for i in range(plots)]

        handlers.extend([
            hand.StateCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[0], do_plot_cost=False,
                reproduce_mode=reproduce_mode),
            hand.StateCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[1], do_plot_state=False,
                reproduce_mode=reproduce_mode),
            hand.CumulativeCostPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[2], discounted=False,
                reproduce_mode=reproduce_mode),
            hand.ArrivalsPlotter(
                num_sim_steps, time_interval, plot_freq, ax=axs[3], reproduce_mode=reproduce_mode)
        ])
        if is_hedgehog:
            handlers.extend([
                hand.WorkloadPlotter(
                    num_sim_steps, time_interval, plot_freq, ax=axs[4], plot_fluid_model=False,
                    plot_hedging=False, reproduce_mode=reproduce_mode),
                hand.HedgingThresholdPlotter(
                    num_sim_steps, time_interval, plot_freq, ax=axs[5],
                    reproduce_mode=reproduce_mode),
                hand.EffectiveCostErrorPlotter(
                    num_sim_steps, time_interval, plot_freq, ax=axs[6],
                    eff_cost_err_method='absolute', reproduce_mode=reproduce_mode),
                hand.IdlingPlotter(num_sim_steps, time_interval, plot_freq, ax=axs[7],
                                   reproduce_mode=reproduce_mode),
                hand.ActionsToFluidPlotter(num_sim_steps, time_interval, plot_freq, ax=axs[8])
            ])
            if not reproduce_mode:
                handlers.append(PrintStateHedging(plot_freq))
                if not is_routing:
                    handlers.append(PrintInverseLoadings(plot_freq))
                handlers.append(PrintActualtoFluidRatio(plot_freq))

    return handlers


def workload_to_physical_resources_index(nu: types.NuMatrix, eps: float = 1e-4) -> List[List[int]]:
    """
    Returns the index of the physical resources that correspond with each workload vector in the
    workload matrix.

    :param nu: matrix whose rows are transposed dual variables of the minimal draining time LP.
    :param eps: tolerance to decide whether the resource is pooled for any workload vector.
    :return: index of physical resources
    """
    workload_to_physical_resources_indexes = []
    for nu_s in nu:
        index_s = np.argwhere(nu_s > eps)
        index_s_list = []
        for i in index_s:  # There can be many physical resources associated to a workload vector.
            index_s_list.append(i.item())
        workload_to_physical_resources_indexes.append(index_s_list)
    return workload_to_physical_resources_indexes


def compute_idling_process_from_constituency(
        nu: types.NuMatrix, constituency_matrix: types.ConstituencyMatrix,
        actions: types.ActionProcess) -> types.WorkloadProcess:
    """
    Returns a binary matrix that represents the idling process for each physical resource, where
    number of rows equals number of workload vectors, and number of columns equals number of data
    points. A one in the (s, t) position of this matrix means that the s-th resource was idling
    (i.e. non of its activities were on) at time t. Such idling process is computed as:
        idling[s, t] = 1 - C[s, :] @ actions[:, t]

    :param nu: matrix whose rows are transposed dual variables of the minimal draining time LP.
    :param constituency_matrix: from the environment.
    :param actions: actions process to compute idling.
    :return: idling_process: binary idling process matrix.
    """
    num_workload_vec = nu.shape[0]
    num_data_points = actions.shape[1]

    sort_index = workload_to_physical_resources_index(nu)
    constituency_matrix = constituency_matrix[sort_index, :]
    idling_process = np.ones((num_workload_vec,
                              num_data_points - 1)) - constituency_matrix @ actions[:, :-1]
    return idling_process


def compute_idling_process_from_workload(workload_mat: types.WorkloadMatrix,
                                         buffer_processing_matrix: types.BufferMatrix,
                                         actions: types.ActionProcess) -> types.WorkloadProcess:
    """
    Returns a binary matrix that represents the idling process for each physical resource, where
    number of rows equals number of workload vectors, and number of columns equals number of data
    points. A one in the (s, t) position of this matrix means that the s-th resource was idling
    (i.e. non of its activities were on) at time t. Such idling process is computed as:
        idling[:, t] = 1 + Xi @ B @ actions[:, t]

    :param nu: matrix whose rows are transposed dual variables of the minimal draining time LP.
    :param workload_mat: workload matrix.
    :param buffer_processing_matrix: from environment.
    :param actions: actions process to compute idling.
    :return: idling_process: binary idling process matrix.
    """
    num_workload_vec = workload_mat.shape[0]
    num_data_points = actions.shape[1]
    idling_process = np.ones((num_workload_vec, num_data_points - 1)) \
                     + workload_mat @ buffer_processing_matrix @ actions[:, :-1]
    return idling_process


def print_hyperparams(**kwargs) -> None:
    for i, v in kwargs.items():
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
        print(f"\t{i}: {v}")


def print_hedgehog_params(agent: HedgehogAgentInterface) -> None:
    print("Hedgehog parameters:")
    print_hyperparams(ac_params=agent.asymptotic_covariance_params,
                      wk_params=agent.workload_relaxation_params,
                      si_params=agent.strategic_idling_params,
                      po_params=agent.policy_params,
                      hh_params=agent.hedgehog_hyperparams)


def print_maxweight_params(agent: MaxWeightAgent) -> None:
    print("MaxWeight parameters:")
    print_hyperparams(weight_per_buffer=agent.weight_per_buffer)


def print_agent_params(agent: AgentInterface) -> None:
    if isinstance(agent, HedgehogAgentInterface):
        print_hedgehog_params(agent)
    elif isinstance(agent, MaxWeightAgent):
        print_maxweight_params(agent)
