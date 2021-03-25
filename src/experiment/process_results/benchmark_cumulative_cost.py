import src.experiment.process_results.result_handling_utils as result_handling


def plot_cum_cost():
    """
    Function plots the average cumulative cost with confidence intervals. It accepts three
    parameters by command line:
    --path: Mandatory string with path to the directory with all runs.
    --uncertainty_mode: Optional string, with possible values 'std_err', for standard error, or
        'variance' for the population variance. If nothing is passed, 'std_err' is used by default.
    --save_fig: Optional. If passed, it will store the figure in the data path. Otherwise, the
        figure won't be saved but shown in the display.
    Example:
        python experiment/process_results/benchmark_cumulative_cost.py --save_fig
            --path path_to_data_folder
    """
    parsed_args = result_handling.parse_args()
    path = parsed_args.path
    if parsed_args.save_fig:
        save_fig = path + '/cumulative_cost.pdf'
    else:
        save_fig = None
    uncertainty_mode = parsed_args.uncertainty_mode

    e_runs, e_params = result_handling.read_experiment_runs(path, 'cost')
    e_runs, _ = result_handling.stack_exp_runs(e_runs, 'cost')
    discount_factor = e_params['discount_factor']
    e_runs = result_handling.get_cum_sum_statistics(e_runs, discount_factor, 'cost')
    e_runs = result_handling.aggregate_statistics(e_runs, 'cost')
    result_handling.plot_aggregated_results_vs_time(e_runs, 'Cumulative cost', uncertainty_mode,
                                                    save_fig, 'cost')


if __name__ == "__main__":
    plot_cum_cost()
