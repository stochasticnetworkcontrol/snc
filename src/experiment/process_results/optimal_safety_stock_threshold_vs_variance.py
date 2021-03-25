import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
import experiment.process_results.result_handling_utils as result_handling


def values_to_array(results, statistic, agent, env_key, env_val, agent_keys):
    """
    Return list of sorted input (x-axis) and output (y-axis) values ready to be plot. It expects
    results already processed in the form of a nested dictionary with the following hierarchy:
        [agent][env_key][env_val][agent_keys][statistic]

    :param results: Dictionary with statistics from experiment results.
    :param statistic: Key for values we want to convert: 'mean', 'std' or 'std_err'.
    :param agent: Agent name as specified in the directories that store the experiments' results.
    :param env_key: Environment parameter name we want to process.
    :param env_val: Specific environment parameter value we want to process.
    :param agent_keys: Agent parameter name from which we want to obtain list of values.
    :return: x and y values ready to be plot.
    """
    val_lst = [[a[0], a[1]] for a in results[agent][env_key][env_val][agent_keys][statistic]]
    ind = sorted(range(len(val_lst)), key=lambda k: val_lst[k][0])
    val_x_y = np.array([val_lst[m] for m in ind])
    x = val_x_y[:, 0]
    y = val_x_y[:, 1]
    return x, y


def plot_total_cost_vs_optimal_safety_stock_cost(path: str, uncertainty_mode: str,
                                                 save_fig: Optional[str] = None):
    """
    Plots the average cumulative cost with confidence intervals for each value of Hedgehog safety
    stock threshold parameter `hh_theta_0`. It returns a subplot for each value of the
    `demand_variance` environment parameter.

    :param path: Path to the directory with all runs.
    :param uncertainty_mode: It can be 'std_err' for standard error or 'variance' for the population
        variance.
    :param save_fig: pathname of the image file if to be saved on disk instead of displaying.
    """
    agent = 'hedgehog'
    data_keys = 'cost'
    agent_keys = 'hh_theta_0'
    env_keys = 'demand_variance'

    e_runs, _ = result_handling.read_experiment_runs(path, data_keys, agent_keys, env_keys)
    e_runs, _ = result_handling.stack_exp_runs(e_runs, data_keys, agent_keys, env_keys)
    e_runs = result_handling.get_total_sum_statistics(e_runs, data_keys, agent_keys, env_keys)
    e_runs = result_handling.aggregate_statistics(e_runs, data_keys, agent_keys, env_keys)
    results = result_handling.get_param_value_and_results_as_list(e_runs, data_keys, agent_keys,
                                                                  env_keys)
    env_param_values = results[agent][env_keys].keys()
    f, ax = plt.subplots(len(env_param_values), 1, figsize=(24, 24))
    for i, e_val in enumerate(env_param_values):
        if len(env_param_values) > 1:
            curr_ax = ax[i]
        else:
            curr_ax = ax
        x_axis, mean = values_to_array(results, 'mean', agent, env_keys, e_val, agent_keys)
        x_axis_b, uncertainty = values_to_array(results, uncertainty_mode, agent, env_keys, e_val,
                                                agent_keys)
        assert np.all(x_axis == x_axis_b)

        curr_ax.plot(x_axis, mean, label=f"{agent} demand var/mean ratio={e_val}")
        curr_ax.fill_between(x_axis, mean - uncertainty, mean + uncertainty, alpha=0.2)
        curr_ax.set_ylabel(data_keys)
        curr_ax.set_xlabel(agent_keys)
        curr_ax.legend()
    if save_fig is not None:
        f.savefig(save_fig)
    else:
        plt.show()


if __name__ == "__main__":
    PATH = "path_to_data_folder"
    UNCERTAINTY_MODE = 'std_err'
    SAVE_FIG = 'figure_name.pdf'
    plot_total_cost_vs_optimal_safety_stock_cost(PATH, UNCERTAINTY_MODE, SAVE_FIG)
