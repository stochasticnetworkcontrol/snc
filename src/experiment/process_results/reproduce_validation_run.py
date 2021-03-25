import json
import sys
import numpy as np

import matplotlib.pyplot as plt

import src.snc.simulation.store_data.reporter as rep
import src.snc.simulation.utils.validation_utils as validation_utils
import src.experiment.process_results.result_handling_utils as handle_results


def main():
    """
    Reproduce a single validation run, i.e., it plots all data in the live plot panel from data.
    It accepts two parameters by command line:
    --path: Mandatory string with path to the directory with all runs.
    --is_hedgehog: Optional. If passed it understands that the experiment in the path corresponds
        to Hedgehog, so it will reproduce variables from the workload space. Otherwise, it assumes
        it is not Hedgehog and plots a reduced panel.
    Example:
        python experiment/process_results/reproduce_validation_run.py --is_hedgehog
        --path path_to_data_folder
    """
    parsed_args = handle_results.parse_args()
    data_path = parsed_args.path
    is_hedgehog = parsed_args.is_hedgehog
    is_routing = parsed_args.is_routing

    data_dict_read_ok = False
    try:
        with open('{}/datadict.json'.format(data_path), 'r') as f:
            data_dict = json.load(f)
        data_dict_read_ok = True
    except IOError:
        print(f'Cannot open {data_path}')

    params_read_ok = False
    try:
        with open('{}/parameters.json'.format(data_path), 'r') as f:
            params = json.load(f)
        params_read_ok = True
    except IOError:
        print(f'Cannot open {data_path}')

    if data_dict_read_ok and params_read_ok:
        n = len(data_dict["state"])
        time_interval = params["env"]["_job_generator"]["sim_time_interval"]
    else:
        print('Not possible to continue.')
        sys.exit()  # Exit script.

    handlers = validation_utils.get_handlers(False, n, n - 1, time_interval, is_hedgehog,
                                             is_routing, reproduce_mode=True)
    reporter = rep.Reporter(handlers=handlers)
    reporter._cache = params

    if is_hedgehog:
        strategic_idling_tuple_read_ok = False
        try:
            with open('{}/reporter/strategic_idling_tuple.json'.format(data_path), "r") as f:
                sub = json.load(f)
            strategic_idling_tuple_read_ok = True
        except IOError:
            print(f'Cannot open {data_path}')
        if strategic_idling_tuple_read_ok:
            for k, v in sub.items():
                sub[k] = np.array(v)
            reporter.cache["strategic_idling_tuple"] = sub

        try:
            with open('{}/reporter/w.json'.format(data_path), "r") as f:
                reporter.cache.update(json.load(f))
        except IOError:
            print(f'Cannot open {data_path}')

        try:
            with open('{}/reporter/num_mpc_steps.json'.format(data_path), "r") as f:
                reporter.cache.update(json.load(f))
        except IOError:
            print(f'Cannot open {data_path}')

        try:
            with open('{}/reporter/horizon.json'.format(data_path), "r") as f:
                reporter.cache.update(json.load(f))
        except IOError:
            print(f'Cannot open {data_path}')

    reporter.report(data_dict, 0)
    reporter.report(data_dict, n-1)

    # Keep the window open
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
