import src.experiment.process_results.result_handling_utils as result_handling


def get_num_actions(e_runs):
    """
    Takes all the data and extracts the number of actions. This is useful for defining the legend.
    """
    agent_name = list(e_runs.keys())[0]
    num_actions = e_runs[agent_name]['action_to_rate_ratio']['mean'].shape[1]
    return num_actions, agent_name


def plot_action_to_activity_rate_cumulative_ratio():
    """
    Generate a figures with confidence intervals of the average actual actions to activity rates
    cumulative ratios.
    It accepts three parameters by command line:
    --path: Mandatory string with path to the directory with all runs.
    --uncertainty_mode: Optional string, with possible values 'std_err', for standard error, or
        'variance' for the population variance. If nothing is passed, 'std_err' is used by default.
    --save_fig: Optional. If passed, it will store the figures in the data path. Otherwise, the
        figures won't be saved but shown in the display.
    Example:
        python experiments/process_results/average_action_to_activity_cum_ratio.py
        --save_fig --path path_to_data_folder
    """
    parsed_args = result_handling.parse_args()
    path = parsed_args.path
    if parsed_args.save_fig:
        save_fig = path + '/action_to_rate_cum_ratio.pdf'
    else:
        save_fig = None
    uncertainty_mode = parsed_args.uncertainty_mode

    data_keys = ['action', 'zeta_star']
    data_keys_to_remove = ['cost', 'action', 'added', 'arrivals', 'drained', 'processing', 'state']

    # Read data. Process data while reading to compute 'action_to_rate_ratio' (i.e. the cumulative
    # ratio of actual actions to activity rates) with function 'compute_action_to_rate_cum_ratio'.
    e_runs, _ = result_handling.read_experiment_runs(
        path, data_keys,
        function_pre_process=result_handling.compute_action_to_rate_cum_ratio,
        function_pre_process_param=data_keys_to_remove)

    # Stack 'compute_action_to_rate_cum_ratio' from different experiments as rows of a Numpy array.
    e_runs, _ = result_handling.stack_exp_runs(e_runs, ['action_to_rate_ratio'])

    # Compute statistics for the different runs.
    e_runs = result_handling.aggregate_statistics(e_runs, ['action_to_rate_ratio'])

    # Plot three figures.
    num_actions, agent_name = get_num_actions(e_runs)
    new_legend = [f"{agent_name} action {i}" for i in range(num_actions)]
    result_handling.plot_aggregated_results_vs_time(
        e_runs, 'Action to rate cumulative ratio', uncertainty_mode, save_fig,
        'action_to_rate_ratio', new_legend=new_legend)


if __name__ == "__main__":
    plot_action_to_activity_rate_cumulative_ratio()
