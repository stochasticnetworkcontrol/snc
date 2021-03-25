import experiment.process_results.result_handling_utils as result_handling


def plot_cost_and_eff_cost_err():
    """
    Generate three figures with confidence intervals:
        - average instantaneous cost,
        - average distance to lower bound (i.e., eff_cost_error),
        - average instantaneous cost and average distance to lower bound together.
    It accepts three parameters by command line:
    --path: Mandatory string with path to the directory with all runs.
    --uncertainty_mode: Optional string, with possible values 'std_err', for standard error, or
        'variance' for the population variance. If nothing is passed, 'std_err' is used by default.
    --save_fig: Optional. If passed, it will store the figures in the data path. Otherwise, the
        figures won't be saved but shown in the display.
    Example:
       python snc/process_results/lower_bound_and_inst_cost.py --save_fig --path path_to_data_folder
    """
    parsed_args = result_handling.parse_args()
    path = parsed_args.path
    if parsed_args.save_fig:
        save_fig_1 = path + '/cost.pdf'
        save_fig_2 = path + '/eff_cost_error.pdf'
        save_fig_3 = path + '/cost_and_eff_cost_error.pdf'
    else:
        save_fig_1 = None
        save_fig_2 = None
        save_fig_3 = None
    uncertainty_mode = parsed_args.uncertainty_mode

    data_keys = ['cost', 'c_bar', 'w', 'num_mpc_steps']
    data_keys_to_remove = ['c_bar', 'w', 'num_mpc_steps', 'action', 'added', 'arrivals', 'drained',
                           'processing', 'state', 'beta_star', 'beta_star_cone_list', 'c_plus',
                           'delta_h', 'height_process', 'k_idling_set', 'lambda_star', 'psi_plus',
                           'psi_plus_cone_list', 'sigma_2_h', 'theta_roots', 'w_star']

    # Read data. Process data while reading to compute 'eff_cost_error' (i.e. distance to lower
    # bound) with function 'compute_eff_state_error'.
    e_runs, _ = result_handling.read_experiment_runs(
        path, data_keys, function_pre_process=result_handling.compute_eff_state_error,
        function_pre_process_param=data_keys_to_remove)

    # Stack 'cost' and 'eff_cost_error' from different experiments as rows of a Numpy array.
    e_runs, _ = result_handling.stack_exp_runs(e_runs, ['cost', 'eff_cost_error'])

    # Compute statistics for the different runs.
    e_runs = result_handling.aggregate_statistics(e_runs, ['cost', 'eff_cost_error'])

    # Plot three figures.
    result_handling.plot_aggregated_results_vs_time(e_runs, 'Cost', uncertainty_mode, save_fig_1,
                                                    'cost', new_legend=['Cost'])
    result_handling.plot_aggregated_results_vs_time(e_runs, 'Eff cost error', uncertainty_mode,
                                                    save_fig_2, 'eff_cost_error',
                                                    new_legend=['Distance to lower bound'])
    result_handling.plot_aggregated_results_vs_time(
        e_runs, 'Cost', uncertainty_mode, save_fig_3, ['cost', 'eff_cost_error'],
        new_legend=['Cost', 'Distance to lower bound'])


if __name__ == "__main__":
    plot_cost_and_eff_cost_err()
