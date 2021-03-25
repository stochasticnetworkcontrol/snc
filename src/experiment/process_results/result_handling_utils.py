import argparse
from collections import defaultdict
import json
from matplotlib import pyplot as plt
import numpy as np

import os
import sys
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, Union

# Parameters that can be processed. Per parameter method must be implemented in
# `get_value_from_parameters` function.
VALID_PARAM_KEYS: List[str] = ['hh_theta_0', 'demand_variance']

# Valid modes when plotting uncertainty bars around the mean.
VALID_UNCERTAINTY_MODE = ['std_err', 'std']

# Keys stored in datadict.json.
KEYS_DATADICT = set(['state', 'action', 'z_star', 'arrivals', 'processing'])

# Keys stored in strategic_idling_tuple.json.
KEYS_STRATEGIC_IDLING = set(['beta_star', 'c_bar', 'c_plus', 'delta_h', 'height_process',
                             'k_idling_set', 'lambda_star', 'psi_plus', 'sigma_2_h',
                             'theta_roots', 'w_star'])


def parse_args():
    parser = argparse.ArgumentParser(description='Path.')
    parser.add_argument('--path', type=str, help='Path with logs to be read')
    parser.add_argument('--save_fig', action="store_true", help='If this argument is passed, the'
                                                                'figure will be stored in the data'
                                                                'path.')
    parser.add_argument("--uncertainty_mode", type=str, default='std_err',
                        help="It can be 'std_err' for standard error or 'variance' for the "
                             "population variance.")
    parser.add_argument('--is_hedgehog', action="store_true",
                        help='Indicates if run corresponds with Hedgehog algorithm.')
    parser.add_argument('--is_routing', action="store_true",
                        help='Indicates if run corresponds with a routing model.')

    return parser.parse_args()


def get_list_of_uuids(
        lab_name: str,
        results_directory: str,
        shared_dir_name: str = 'shared') -> Tuple[List, str]:
    """
    For a given lab_name, which corresponds with a directory (containing multiple results files), it
    returns the list of UUIDs that correspond with the subdirectories.
    :param lab_name: Name of the directory with the runs for which we want to rename the directories
        with the agents' data log. This is typically the `lab_name` used to group experiment runs.
        Example: lab_name = 'naive-maxweight_sirl_steady_state'
    :param results_directory: The parent directory contraining all results (all labs)
    :param shared_dir_name: Name of the shared directory created for generic logging across all
        experiments.
    :return: (uuid_list, lab_path):
        - uuid_list: List of UUIDs as strings.
        - lab_path: String with the path to the lab_name directory.
    """

    assert os.path.isdir(results_directory), f"results directory: {results_directory} is not valid"

    # Build lab path.
    lab_path = results_directory + '/' + lab_name
    assert os.path.isdir(lab_path), f"{lab_path} is not a valid directory."
    print(f"lab_path: {lab_path}")

    # Obtain list of UUIDs and remove shared folder.
    uuid_list = os.listdir(lab_path)
    try:
        uuid_list.remove(shared_dir_name)
    except ValueError:
        Warning(f"No directory named: {shared_dir_name}\n.")

    return uuid_list, lab_path


def create_results_container(agent_keys: Optional[Union[str, List[str]]],
                             env_keys: Optional[Union[str, List[str]]],
                             contained_type):
    """
    Create a nested-dictionary container for results to be read from multiple experiments. The 1st
    level of the dictionary is always the agent. If the lab folder contains results from multiple
    agents, then their results will be separated at this first level. Otherwise, this level is still
    included, but there will have only one key.
    The rest of levels are controlled by whether we pass `agent_keys` and `env_keys`, which are
    lists with the agent and environment parameter keys, respectively, that we want to use to
    organise the results.
    If `env_keys` is not None, then the 2nd and 3rd levels of the dictionary are given by the
    list of environment parameters and their values (these values don't have to be passed but are
    those found in the actual log files).
    If `agent_keys` is not None, then the next two levels (i.e. 4th and 5th if `env_keys` is not
    None or 2nd and 3rd if `env_keys` is None) are given by list of agent parameters and their
    values (again, the values are not passed explicitly but rather they are obtained from the actual
    log files).
    The last two levels of the dictionary correspond with the data variable (e.g. 'cost', 'state',
    'w', 'beta_star'), and the actual values found in the log files.
    In summary:
        - if env_keys and agent_keys, then the nested dictionary is organised as follows:
            Dict [Agent name][env key][env val][Param key][agent val][Var key][Var val].
        - if env_keys and not agent_keys:
            Dict [Agent name][env key][env val][Var key][Var val].
        - if not env_keys, and agent_keys:
            Dict [Agent name][agent key][agent val][Var key][Var val].
        - if not env_keys and not agent_keys:
            Dict [Agent name][Var key][Var val].

    :param agent_keys: String or list of strings. In the case of a string, it contains the name of
        the agent parameter we want to use to separate results.
    :param env_keys: String or list of strings. In the case of a string, it contains the name of
        the environment parameter we want to use to separate results.
    :param contained_type: Type of data container: dict or list.
    :return: Nested dictionary to store and organise the data.
    """
    if env_keys:
        if agent_keys:
            # Dict [Agent name][env key][env val][agent key][agent val][Var key][Var val].
            return defaultdict(  # [Agent name]
                lambda: defaultdict(  # [env key]
                    lambda: defaultdict(  # [env val]
                        lambda: defaultdict(  # [agent key]
                            lambda: defaultdict(  # [agent val]
                                lambda: defaultdict(  # [Var key]
                                    contained_type))))))  # [Var val]
        else:
            # Dict [Agent name][env key][env val][Var key][Var val].
            return defaultdict(  # [Agent name]
                lambda: defaultdict(  # [env key]
                    lambda: defaultdict(  # [env val]
                        lambda: defaultdict(  # [Var key]
                            contained_type))))  # [Var val]
    else:
        if agent_keys:
            # Dict [Agent name][agent key][agent val][Var key][Var val].
            return defaultdict(  # [Agent name]
                lambda: defaultdict(  # [agent key]
                    lambda: defaultdict(  # [agent val]
                        lambda: defaultdict(  # [Var key]
                            contained_type))))  # [Var val]
        else:
            # Dict [Agent name][Var key][Var val].
            return defaultdict(  # [Agent name]
                lambda: defaultdict(  # [Var key]
                    contained_type))  # [Var val]


def validate_input_params(data_keys: Optional[Union[str, List[str]]],
                          agent_keys: Optional[Union[str, List[str]]],
                          env_keys: Optional[Union[str, List[str]]]) \
        -> Tuple[Union[str, List[str]], Union[str, List[str]], Union[str, List[str]]]:
    """
    Validate that if any input data and parameter keys are passed, they are strings or list of
    strings. In the case either is a string, wrap it with a list so it can be iterated.

    :param data_keys: Optional string or list of strings with the keys of the data variables we
        want to store from the data json file.
    :param agent_keys: String or list of strings. In the case of a string, it contains the name of
        the agent parameter we want to use to separate results.
    :param env_keys: String or list of strings. In the case of a string, it contains the name of
        the environment parameter we want to use to separate results.
    :return: (data_keys, agent_keys, env_keys)
        - data_keys: Return None if that was the original value, or a list if it was a string.
        - agent_keys: Return None if that was the original value, or a list if it was a string.
        - env_keys: Return None if that was the original value, or a list if it was a string.
    """
    if data_keys:
        assert isinstance(data_keys, str) or isinstance(data_keys, list), \
            f"data_keys must be string or list of strings, but provided: {data_keys}."
        if isinstance(data_keys, str):  # Convert to list to be iterated later.
            data_keys = [data_keys]
    if agent_keys:
        assert isinstance(agent_keys, str) or isinstance(agent_keys, list), \
            f"data_keys must be string or list of strings, but provided: {agent_keys}."
        if isinstance(agent_keys, str):  # Convert to list to be iterated later.
            agent_keys = [agent_keys]
    if env_keys:
        assert isinstance(env_keys, str) or isinstance(env_keys, list), \
            f"env_keys must be string or list of strings, but provided: {env_keys}."
        if isinstance(env_keys, str):  # Convert to list to be iterated later.
            env_keys = [env_keys]
    return data_keys, agent_keys, env_keys


def read_json_file(filename):
    """
    Read json file and return data dictionary and boolean flag indicating whether there has been an
    exception.

    :param filename: File name.
    :return: (data, exception)
        - data: Data dictionary.
        - exception: Boolean flag indicating whether there has been an exception while reading data.
    """
    data = dict()
    exception = False
    try:
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"failed to read: {filename}")
                exception = True
    except FileNotFoundError:
        print(f"No such file or directory: {filename}.")
        exception = True
    return data, exception


def read_data(data_keys: Union[str, List[str]], data_path: str) -> Tuple[Dict[str, List], bool]:
    """
    Read data from files as specified in data_keys.

    :param data_keys: Data to be read.
    :param data_path: Path where data is stored.
    :return:
        - exp_data: Dictionary with data.
        - exception_data: Boolean flag indicating whether there was an exception while reading.
    """
    exp_data = dict()  # Initialise data container.
    exception_cost = False
    exception_datadict = False
    exception_idling = False
    exception_num_mpc_steps = False
    exception_w = False

    # Read instantaneous cost.
    if 'cost' in data_keys:  # Try reading from new storing method.
        cost_filename = os.path.join(data_path, "cost.json")
        exp_cost, exception_cost = read_json_file(cost_filename)
        if not exception_cost:
            exp_data['cost'] = exp_cost

    # Read state, action, arrivals, and processing data. Read also cost if previous method
    # failed. In the future, this can be modified such that each element of data_key is read
    # from a single file name, so the check on `exception_cost` won't apply.
    if bool(KEYS_DATADICT.intersection(set(data_keys))) \
            or exception_cost:
        datadict_filename = os.path.join(data_path, "datadict.json")
        exp_datadict, exception_datadict = read_json_file(datadict_filename)
        if not exception_datadict:
            exp_data.update(exp_datadict)
            exception_cost = False  # Needed for legacy as datadict also includes cost for now.

    # Read strategic idling data.
    if bool(KEYS_STRATEGIC_IDLING.intersection(set(data_keys))):
        idling_filename = os.path.join(data_path, 'reporter', "strategic_idling_tuple.json")
        exp_idling_data, exception_idling = read_json_file(idling_filename)
        if not exception_idling:
            exp_data.update(exp_idling_data)  # Merge exp_idling_data in exp_data.

    if 'num_mpc_steps' in data_keys:
        num_mpc_steps_filename = os.path.join(data_path, 'reporter', "num_mpc_steps.json")
        exp_num_mpc_steps, exception_num_mpc_steps = read_json_file(num_mpc_steps_filename)
        if not exception_num_mpc_steps:
            exp_data.update(exp_num_mpc_steps)  # Merge exp_num_mpc_steps in exp_data.

    if 'w' in data_keys:
        w_filename = os.path.join(data_path, 'reporter', "w.json")
        exp_w, exception_w = read_json_file(w_filename)
        if not exception_w:
            exp_data.update(exp_w)  # Merge exp_w in exp_data.

    exception_data = (exception_cost or exception_datadict or exception_idling
                      or exception_num_mpc_steps or exception_w)

    return exp_data, exception_data


def get_value_from_parameters(param_key: str, exp_param):
    """
    When running experiments, we could change multiple parameters. These parameters are stored in
    different places (e.g., Hedgehog safety stock threshold is in `hedgehog_hyperparams` inside
    `agent`; while demand variance parameter is in the job generator in the environment). This
    function specifies how to obtain each parameter. In order to do so, the parameter key should be
    defined in the list VALID_PARAM_KEYS.

    :param param_key: Parameter name.
    :param exp_param: Dictionary with all the experiment parameters.
    :return: Value.
    """
    assert param_key in VALID_PARAM_KEYS, f"{param_key} is not a valid parameter key. Check that" \
                                          f" it belongs to the VALID_PARAM_KEYS list."
    if param_key == 'hh_theta_0':
        val = exp_param['agent']['hedgehog_hyperparams'][2]
        return round(val, 10)  # Remove numerical noise for hash.
    elif param_key == 'demand_variance':
        demand_rate = np.array(exp_param['env']['_job_generator']['_demand_rate'])
        demand_variance = np.array(exp_param['env']['_job_generator']['demand_variance'])
        ind = demand_variance > 0
        val = max(demand_variance[ind] / demand_rate[ind])
        if isinstance(val, np.ndarray):
            val = val[0]
        return round(val, 10)  # Remove numerical noise for hashing.
    else:
        raise ValueError(f"not valid Parameter key {param_key} is .")


def del_keys(exp_data, data_keys_to_remove):
    for key in data_keys_to_remove:
        if exp_data.get(key) is not None:
            del exp_data[key]
    new_data_keys = [key for key in exp_data.keys()]
    return new_data_keys


def compute_eff_state_error(exp_data, data_keys_to_remove):
    """
    Compute effective cost error, i.e. instantaneous cost vs lower bound cost at effective state,
    and return it as another data variable in the results dictionary.

    :param exp_data: Dictionary with data.
    :param data_keys_to_remove: data_keys to be removed after computing the effective cost error.
    :return:
        - exp_data: Updated data dictionary with new eff_cost_error data key, and without the keys
            that have been removed.
        - new_data_keys: Updated list of data keys in exp_data.
    """
    assert isinstance(exp_data, dict), "exp_data should be a dictionary."
    assert 'cost' in exp_data, "exp_data should contain 'cost' key."
    assert 'c_bar' in exp_data, "exp_data should contain 'c_bar' key."
    assert 'w' in exp_data, "exp_data should contain 'w' key."
    num_mpc_steps = np.array(exp_data.get('num_mpc_steps'))

    cost = np.array(exp_data.get('cost'))
    c_bars = exp_data.get('c_bar')
    w_process = exp_data.get('w')

    effective_cost = np.nan * np.ones_like(cost)
    for i, (c_bar, w) in enumerate(zip(c_bars, w_process)):
        if c_bar is not None and w is not None:
            effective_cost[i] = np.squeeze(np.array(c_bar).T @ np.array(w))

    if np.all(num_mpc_steps == np.ones_like(num_mpc_steps)):
        assert len(cost) == len(c_bars) == len(w_process)
        exp_data['eff_cost_error'] = cost - effective_cost
    else:
        eff_cost_error = np.nan * np.ones_like(cost)
        cum_num_mpc_steps = np.cumsum(num_mpc_steps)
        j = 0
        for i in range(int(sum(num_mpc_steps))):
            if i >= cum_num_mpc_steps[j]:
                j += 1
                if not i < cum_num_mpc_steps[j]:
                    print(i, j, cum_num_mpc_steps[j])
                assert i < cum_num_mpc_steps[j]
            eff_cost_error[i] = cost[i] - effective_cost[j]
        exp_data['eff_cost_error'] = eff_cost_error

    for key in data_keys_to_remove:
        if exp_data.get(key) is not None:
            del exp_data[key]
    new_data_keys = [key for key in exp_data.keys()]

    return exp_data, new_data_keys


def compute_action_to_rate_cum_ratio(exp_data, data_keys_to_remove):
    """
    Compute cumulative ratio of actual taken actions to policy activity rates, and return it as
    another data variable in the results dictionary.
    :param exp_data: Dictionary with data.
    :param data_keys_to_remove: data_keys to be removed after computing the relevant entity.
    :return:
        - exp_data: Updated data dictionary with new action_to_rate_ratio data key, and without the
            keys that have been removed.
        - new_data_keys: Updated list of data keys in exp_data.
    """
    assert isinstance(exp_data, dict), "exp_data should be a dictionary."
    assert 'action' in exp_data, "exp_data should contain 'action' key."
    assert 'zeta_star' in exp_data, "exp_data should contain 'zeta_star' key."
    action_process = np.array(exp_data.get('action'))
    zeta_star_process = np.array(exp_data.get('zeta_star'))

    cum_actions = np.cumsum(action_process, axis=0)
    cum_fluid_policy = np.cumsum(zeta_star_process, axis=0)
    exp_data['action_to_rate_ratio'] = cum_actions/cum_fluid_policy
    for key in data_keys_to_remove:
        if exp_data.get(key) is not None:
            del exp_data[key]
    new_data_keys = [key for key in exp_data.keys()]

    return exp_data, new_data_keys


def remove_files_from_list(base_path, listdir):
    """
    Remove files from a directory list, and returns the list only containing directories.

    :param base_path: Base directory path.
    :param listdir: List of directories and files.
    :return: listdir: List of directories without files.
    """
    for file_name in listdir:
        if os.path.isfile(os.path.join(base_path, file_name)):
            listdir.remove(file_name)
    return listdir


def read_experiment_runs(experiment_path: str,
                         data_keys: Optional[Union[str, List[str]]] = None,
                         agent_keys: Optional[Union[str, List[str]]] = None,
                         env_keys: Optional[Union[str, List[str]]] = None,
                         function_pre_process: Optional[Callable] = None,
                         function_pre_process_param: Optional[Any] = None):
    """
    Loops through all runs in a given folder and collects the logs. It can separate the results per
    agent, per agent parameter and value, and per environment parameter and value.

    If no data_keys is passed, it returns all data variables available in the log files. If env_keys
    are passed, it separates results per environment parameter and value. Similarly, if agent_keys
    are passed, it separates per agent parameter and value.

    While reading the data, it also allows to preprocess the data (e.g. to create intermediate
    variables). This is done by specifying the callable function_pre_process and its function
    parameters.

    :param experiment_path: name of the directory where the different seeds for a single
        scenario experiments are stored
    :param data_keys: Optional string or list of strings with the keys of the data variable we
        want to store from the data json file.
    :param agent_keys: Optional string or list of strings. In the case of a string, it
        contains the key of the parameter we want to store from the parameter json file.
    :param env_keys: List of strings with the name of the environment parameter(s) we want to use to
        separate results.
    :param function_pre_process: Function to pre-process data before traversing per data variable,
        agent and env keys. Typically used to compute an intermediate data variables from other
        variables of a single agent and seed (e.g. computing the effective cost error from cost,
        c_bar and w). This is complementary to processing the data while traversing parameters,
        since the latter assumes that only one data variable is processed at a time.
    :param function_pre_process_param: Parameters passed to the function that pre-process data
        before traversing.
    :return (exp_runs, exp_param)
        - exp_runs: Nested dictionary with all data logs separated by agent, data variable and maybe
            by environment and agent parameters and values.
        - exp_param: Dictionary with all agent and environment parameters.
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)

    if function_pre_process is not None:
        assert callable(function_pre_process)

    # Create container for results.
    exp_runs = create_results_container(agent_keys, env_keys, list)

    # Get list of directories that correspond to different seeds.
    listdir_exp = os.listdir(experiment_path)
    seed_instances = remove_files_from_list(experiment_path, listdir_exp)

    for i, seed_instance in enumerate(seed_instances):
        print(f"Reading {i}: {experiment_path}/{seed_instance}")

        # Get list of directories that correspond to different agents.
        seed_instance_path = os.path.join(experiment_path, seed_instance)
        listdir_seed = os.listdir(seed_instance_path)
        agent_instances = remove_files_from_list(seed_instance_path, listdir_seed)

        for agent in agent_instances:
            # Read parameters.
            param_filename = os.path.join(experiment_path, seed_instance, agent, "parameters.json")
            exp_param, exception_param = read_json_file(param_filename)

            # Read data.
            data_path = os.path.join(experiment_path, seed_instance, agent)
            exp_data, exception_data = read_data(data_keys, data_path)

            if exception_param or exception_data:
                print('Invalid.')
                continue

            # Speed up data handling. This won't be needed if we store variables in different files.
            if data_keys:  # Store only passed keys.
                data_keys = data_keys
            else:  # If no key is passed, store all data variables.
                data_keys = [key for key in exp_data.keys()]

            # Process data if needed.
            if function_pre_process is not None:
                exp_data, new_data_keys = function_pre_process(exp_data, function_pre_process_param)
            else:
                new_data_keys = data_keys  # No data variables were generated or deleted.

            for k in new_data_keys:
                if isinstance(exp_data, dict):  # When loaded from datadict.json.
                    data = np.array(exp_data[k])
                else:  # When loaded from specific files, like cost.json.
                    data = np.array(exp_data)

                # Flatten 1-D data.
                if data.shape[1] == 1:
                    data = data.ravel()

                if env_keys:  # Per environment parameter disaggregation.
                    for e in env_keys:
                        e_val = get_value_from_parameters(e, exp_param)

                        if agent_keys:  # Per agent parameter disaggregation.
                            for p in agent_keys:
                                p_val = get_value_from_parameters(p, exp_param)

                                # Store data
                                exp_runs[agent][e][e_val][p][p_val][k].append(data)

                        else:  # No disaggregation per agent parameter.
                            exp_runs[agent][e][e_val][k].append(data)

                else:  # No disaggregation per environment parameter.

                    if agent_keys:  # Per agent parameter disaggregation.
                        for p in agent_keys:
                            p_val = get_value_from_parameters(p, exp_param)

                            # Store data
                            exp_runs[agent][p][p_val][k].append(data)

                    else:  # No disaggregation per agent parameter.
                        exp_runs[agent][k].append(data)

    return exp_runs, exp_param


# Callable functions to be used when traversing experiments.
def min_duration_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Returns the minimum between the number of steps for the data log of a set of experiments and the
    previous minimum value. It is used while traversing experiments to obtain the minimum length
    for all agents, environment parameters and agent parameter values, depending on how data are
    separated.

    :param exp_run: Dictionary with data for a set of experiments.
    :param kwargs: Parameter dictionary with key min_dur representing the previous minimum length.
    :return: Scalar value with minimum length across all experiments.
    """
    min_dur = kwargs.get('min_dur')
    return min(min_dur, min(len(a) for a in exp_run))


def stack_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Stacks data from a set of experiments (truncated to the minimum length) into a numpy array,
    where each row corresponds with a single run, and each column corresponds to a time-step.
    The Numpy array typically corresponds with a single agent and a single data variable (e.g.
    cost), and can either separate or aggregate results from different environment and agent
    parameter values, depending on how the experiments are being traversed.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
    variable, and maybe environment and agent parameters and values.
    :param kwargs: Parameter dictionary with key min_dur representing the minimum length across all
        experiments.
    :return: Numpy array where each row corresponds with a single run.
    """
    min_dur = kwargs.get('min_dur')
    if len(exp_run[0].shape) == 1:  # If data is flattened, return matrix.
        return np.vstack([a[:min_dur] for a in exp_run])

    assert len(exp_run[0].shape) == 2, "We only stack flattened vectors or matrices."
    # If data is a matrix, create 3-dim array.
    num_runs = len(exp_run)
    num_steps = exp_run[0].shape[0]
    num_vars = exp_run[0].shape[1]
    data_container = np.nan * np.ones((num_runs, num_steps, num_vars))
    for i in range(num_vars):
        data_container[:, :, i] = np.vstack([a[:min_dur, i] for a in exp_run])
    return data_container


def discounted_cum_sum_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Computes the discounted sum along each row of exp_run. Rows are processed independently and
    typically correspond with the cost for a single agent, and can either separate or aggregate
    results from different environment and agent parameter values, depending on how the experiments
    are being traversed.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
    variable, and maybe environment and agent parameters and values.
    :param kwargs: Parameter dictionary with key discount_factor representing the discount_factor,
        which is assumed the same for all experiments.
    :return: List with discounted cumulative sum.
    """
    discount_factor = kwargs.get('discount_factor')
    min_dur = exp_run.shape[1]
    log_discount_multipliers = np.arange(min_dur) * np.log(discount_factor)
    exp_run *= np.exp(log_discount_multipliers)
    exp_run = np.cumsum(exp_run, axis=1)
    return exp_run


def discounted_sum_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Computes the discounted sum along each row of exp_run. Rows are processed independently and
    typically correspond with the cost for a single agent, and can either separate or aggregate
    results from different environment and agent parameter values, depending on how the experiments
    are being traversed.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
    variable, and maybe environment and agent parameters and values.
    :param kwargs: Parameter dictionary with key discount_factor representing the discount_factor,
        which is assumed the same for all experiments.
    :return: Scalar value with the cost discounted sum per run.
    """
    discount_factor = kwargs.get('discount_factor')
    min_dur = exp_run.shape[1]
    log_discount_multipliers = np.arange(min_dur) * np.log(discount_factor)
    exp_run *= np.exp(log_discount_multipliers)
    exp_run = np.sum(exp_run, axis=1)
    return exp_run


def total_sum_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Computes the total sum along each row of exp_run. Rows are processed independently and
    typically correspond with the cost for a single agent, and can either separate or aggregate
    results from different environment and agent parameter values, depending on how the experiments
    are being traversed.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
        variable, and maybe environment and agent parameters and values.
    :param kwargs: None expected.
    :return: Scalar value with the total cost per run.
    """
    return np.sum(exp_run, axis=1)


def statistics_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Computes mean, standard deviation, and standard error per time-step (i.e. columns of exp_run)
    across the set of experiments (i.e. rows of exp_run). These statistics can be computed for any
    data variable (e.g. cost, state, action, w, beta_star, etc.) for a single agent, and can either
    separate or aggregate results from different environment and agent parameter values, depending
    on how the experiments are being traversed.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
        variable, and maybe environment and agent parameters and values.
    :param kwargs: None expected.
    :return: Dictionary with values the list of each statistic with as many items as time-steps.
    """
    mean = np.mean(exp_run, axis=0)
    std = np.std(exp_run, axis=0, ddof=1)
    num_run = exp_run.shape[0]
    std_err = std / np.sqrt(num_run)
    return {'mean': mean, 'std': std, 'std_err': std_err}


def plot_mean_call(exp_run, axis, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Plots the mean (typically obtained by previously traversing the experiments with callable
    `statistics_call`) for each set or subset of experiments, depending on how the data has been
    separated. This function can be naturally applied in any data variable (e.g. cost, state,
    action, w, beta_star, etc.)

    :param exp_run: Dictionary with the mean of a set of experiments, separated by agent, data
        variable, and maybe environment and agent parameters and values.
    :param axis: Structure where the new axis is going to be appended.
    :param kwargs: Dictionary with the agent name.
    :return: axis with the new plot appended.
    """
    agent = kwargs.get('agent')
    axis.plot(exp_run['mean'], label=agent)
    return axis


def plot_uncertainty_call(exp_run, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Plots the uncertainty (typically obtained by previously traversing the experiments with callable
    `statistics_call`) around the mean, either standard deviation or standard error, for each set or
    subset of experiments (depending on how the data has been separated). The uncertainty mode is
    passed through kwargs.
    This function can be naturally applied in any data variable (e.g. cost, state, action, w,
    beta_star, etc.)

    :param exp_run: Dictionary with the mean, and uncertainty measure (standard or standard error)
        of a set of experiments, separated by agent, data variable, and maybe environment and agent
        parameters and values.
    :param kwargs: Dictionary with at least two fields: i) axis with the mean to which we are going
        to add the uncertainty plot; and 2) the uncertainty mode.
    :return: axis with the new fill_between plot.
    """
    axis = kwargs.get('axis')
    uncertainty_mode = kwargs.get('uncertainty_mode')
    num_time_steps = len(exp_run['mean'])

    if len(exp_run['mean'].shape) == 1:
        me = exp_run['mean']
        un = exp_run[uncertainty_mode]
        axis.fill_between(np.arange(num_time_steps), me - un, me + un, alpha=0.2)
    else:
        for me, un in zip(exp_run['mean'].T, exp_run[uncertainty_mode].T):
            axis.fill_between(np.arange(num_time_steps), me - un, me + un, alpha=0.2)
    return axis


def get_param_value_and_result_list_call(exp_run, val_results_list, **kwargs):
    """
    Callable functions to be used when traversing experiments.
    Returns dictionary with data organised by environment and agent parameters and their available
    values in any of the experiments.

    :param exp_run: Dictionary with data from a set of experiments, separated by agent, data
        variable, and environment and agent parameters and values.
    :param val_results_list: Container that grows at each iteration when adding data corresponding
        to each environment and agent parameter value.
    :param kwargs: Dictionary with the current keys that will be populated in this call (one
        particular combination of agent, environment and agent parameter name, and environment and
        agent parameter value at a time).
    :return: Dictionary with organised data.
    """
    agent = kwargs.get('agent')
    env_key = kwargs.get('env_param')
    env_val = kwargs.get('env_val')
    agent_key = kwargs.get('agent_param')
    agent_value = kwargs.get('agent_val')
    for v in exp_run:
        val_results_list[agent][env_key][env_val][agent_key][v].append((agent_value, exp_run[v]))
    return val_results_list


def traverse_experiments(exp_runs,
                         function_on_data: Callable,
                         func_param: Any,
                         data_keys: Union[str, List[str]],
                         agent_keys: Optional[Union[str, List[str]]] = None,
                         env_keys: Optional[Union[str, List[str]]] = None,
                         mode: str = 'disaggregate'):
    """
    Process data stored in exp_runs. Data is traversed taking into account agent, and possibly
    environment and agent parameters and their values.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param function_on_data: Function to process the data.
    :param func_param: Extra parameters to be passed to the function that will process the data.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :param mode: Specify how the results should be returned:
        - aggregate: Return one single result per data variable for all experiments (i.e. agents,
            and possibly environment and agent parameters and their values).
        - disaggregate: Return dictionary with processed data separated by agent and per variable,
            and possibly per agent and environment parameters and their values.
    :return result: Result from the function after traversed the data.
    """
    new_exp_runs = create_results_container(agent_keys, env_keys, dict)

    for agent in exp_runs.keys():  # Traverse agents.
        func_param['agent'] = agent

        for k in data_keys:  # Traverse data variables.
            func_param['variable'] = k

            if env_keys:  # Per environment parameter disaggregation.
                for e in env_keys:  # Traverse env parameters.
                    func_param['env_param'] = e
                    for e_val in exp_runs[agent][e].keys():  # Traverse values per env parameter.
                        func_param['env_val'] = e_val

                        if agent_keys:  # Per agent parameter disaggregation.
                            for p in agent_keys:  # Traverse agent parameters.
                                func_param['agent_param'] = p
                                # Traverse values per agent parameter.
                                for p_val in exp_runs[agent][e][e_val][p].keys():
                                    func_param['agent_val'] = p_val

                                    result = function_on_data(
                                        exp_runs[agent][e][e_val][p][p_val][k], **func_param)

                                    if mode == 'disaggregate':  # Store disaggregated.
                                        new_exp_runs[agent][e][e_val][p][p_val][k] = result

                        else:  # No agent parameters.
                            result = function_on_data(exp_runs[agent][e][e_val][k], **func_param)
                            if mode == 'disaggregate':  # Store disaggregated.
                                new_exp_runs[agent][k] = result
            else:
                if agent_keys:
                    for p in agent_keys:  # Traverse parameters.
                        for p_val in exp_runs[agent][p].keys():
                            func_param['agent_param'] = p
                            func_param['agent_val'] = p_val
                            result = function_on_data(exp_runs[agent][p][p_val][k], **func_param)
                            if mode == 'disaggregate':  # Store disaggregated.
                                new_exp_runs[agent][p][p_val][k] = result
                else:
                    result = function_on_data(exp_runs[agent][k], **func_param)
                    if mode == 'disaggregate':  # Store disaggregated.
                        new_exp_runs[agent][k] = result

    if mode == 'disaggregate':  # Return disaggregated results.
        return new_exp_runs
    elif mode == 'aggregate':  # Return aggregated results.
        return result
    else:
        raise ValueError(f"Invalid mode: {mode} not in [disaggregate, aggregate]")


def find_min_duration(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                      data_keys: Union[str, List[str]],
                      agent_keys: Optional[Union[str, List[str]]] = None,
                      env_keys: Optional[Union[str, List[str]]] = None) -> int:
    """
    Returns minimum duration across all variables, all parameters (specified in data_keys
    and agent_keys, respectively), and all experiments stored in exp_runs.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return min_dur: Scalar value with minimum length across all experiments.
    """
    min_dur = sys.maxsize  # Initialise to maximum possible value.
    function_on_data = min_duration_call
    func_param = {'min_dur': min_dur}
    min_dur = traverse_experiments(exp_runs, function_on_data, func_param, data_keys,
                                   agent_keys, env_keys, mode='aggregate')
    return min_dur


def stack_exp_runs(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                   data_keys: Union[str, List[str]],
                   agent_keys: Optional[Union[str, List[str]]] = None,
                   env_keys: Optional[Union[str, List[str]]] = None):
    """
    Converts data to a 2-D numpy array with one row per run, and one column per time step. The
    number of columns is truncated to the minimum number of time-steps across all available
    experiments.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
     :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return (agent_exp_runs, min_dur)
        - new_exp_runs: Nested dictionary with data stored as a single Numpy array, with one row per
            experiment, and one column per time step. Number of columns is truncated to the minimum
            length across all agents, all parameters and all runs.
        - min_dur: Minimum length across all experiments.
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)

    min_dur = find_min_duration(exp_runs, data_keys, agent_keys, env_keys)

    function_on_data = stack_call
    func_param = {'min_dur': min_dur}
    new_exp_runs = traverse_experiments(exp_runs, function_on_data, func_param,
                                        data_keys, agent_keys, env_keys,
                                        mode='disaggregate')
    return new_exp_runs, min_dur


def aggregate_statistics(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                         data_keys: Union[str, List[str]],
                         agent_keys: Optional[Union[str, List[str]]] = None,
                         env_keys: Optional[Union[str, List[str]]] = None):
    """
    Computes mean, standard deviation, and standard error of some data variables across a set of
    experiments.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return: new_exp_runs: New nested dictionary with same fields as exp_runs, but replacing the
        Numpy array with all runs with another dictionary that contains the per time-step mean and
        standard deviation across runs.
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)
    function_on_data = statistics_call
    func_param: Dict['str', Any] = {}
    new_exp_runs = traverse_experiments(exp_runs, function_on_data, func_param,
                                        data_keys, agent_keys, env_keys,
                                        mode='disaggregate')
    return new_exp_runs


def get_cum_sum_statistics(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                           discount_factor: float,
                           data_keys: Union[str, List[str]],
                           agent_keys: Optional[Union[str, List[str]]] = None,
                           env_keys: Optional[Union[str, List[str]]] = None):
    """
    Computes discounted cumulative sum of some data variables (typically the instantaneous cost) for
    each experiment (row) in `exp_runs`.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param discount_factor: Discount factor.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return: new_exp_runs: New dictionary with same fields as exp_runs, but replacing each row with
        its discounted cumulative sum.
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)
    function_on_data = discounted_cum_sum_call
    func_param = {"discount_factor": discount_factor}
    new_exp_runs = traverse_experiments(exp_runs, function_on_data, func_param,
                                        data_keys, agent_keys, env_keys,
                                        mode='disaggregate')
    return new_exp_runs


def get_total_sum_statistics(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                             data_keys: Union[str, List[str]],
                             agent_keys: Optional[Union[str, List[str]]] = None,
                             env_keys: Optional[Union[str, List[str]]] = None):
    """
    Computes the sum of some data variables (typically the instantaneous cost) for each experiment
    (row) in `exp_runs`.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return: new_exp_runs: New dictionary with same fields as exp_runs, but replacing each row with
        its sum (i.e. a single scalar).
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)
    function_on_data = total_sum_call
    func_param: Dict[str, Union[str, float]] = dict()
    new_exp_runs = traverse_experiments(exp_runs, function_on_data, func_param,
                                        data_keys, agent_keys, env_keys,
                                        mode='disaggregate')
    return new_exp_runs


def get_param_value_and_results_as_list(exp_runs: Dict[str, Dict[str, List[np.ndarray]]],
                                        data_keys: Union[str, List[str]],
                                        agent_keys: Optional[Union[str, List[str]]] = None,
                                        env_keys: Optional[Union[str, List[str]]] = None):
    """
    Get parameter values as a list. Useful for plotting.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :return:
    """
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)
    function_on_data = get_param_value_and_result_list_call
    if env_keys:
        #  [env key][env val][agent key][agent val][Var key][Var val].
        val_results_list: DefaultDict[str, Any] = \
            defaultdict(lambda:  # [env key].
                        defaultdict(lambda:  # [env val].
                                    defaultdict(lambda:  # [agent key].
                                                defaultdict(lambda:  # [agent val]
                                                            defaultdict(  # [Var key]
                                                                list)))))  # [Var val].
    else:
        # [agent key][agent val][Var key][Var val].
        val_results_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    func_param = {'val_results_list': val_results_list}
    new_exp_runs = traverse_experiments(exp_runs, function_on_data, func_param,
                                        data_keys, agent_keys, env_keys, mode='aggregate')
    return new_exp_runs


def plot_aggregated_results_vs_time(exp_runs,
                                    y_label: str,
                                    uncertainty_mode: str,
                                    save_fig: Optional[str],
                                    data_keys: Union[str, List[str]],
                                    agent_keys: Optional[Union[str, List[str]]] = None,
                                    env_keys: Optional[Union[str, List[str]]] = None,
                                    new_legend: Optional[List[str]] = None):
    """
    Plots results.

    :param exp_runs: Nested dictionary with all data logs separated by agent, data variable and
        maybe by environment and agent parameters and their values.
    :param y_label: Y axis label.
    :param uncertainty_mode: It can be 'std_err' for standard error or 'std' for the population
        standard deviation.
    :param save_fig: pathname of the image file if to be saved on disk instead of
        displaying.
    :param data_keys: String or list of strings with the keys of the data variables to be
        processed.
    :param agent_keys: Optional string or list of strings with the agent parameter keys according to
        which data should be separated when being processed.
    :param env_keys: Optional string or list of strings with the environment parameter keys
        according to which data should be separated when being processed.
    :param new_legend: Optional list of strings to overwrite the legend of the figure.
    :return : None

    """
    assert uncertainty_mode in VALID_UNCERTAINTY_MODE, \
        f"Invalid uncertainty mode: {uncertainty_mode} not in {VALID_UNCERTAINTY_MODE}."
    data_keys, agent_keys, env_keys = validate_input_params(data_keys, agent_keys, env_keys)
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_ylabel(y_label)
    ax.set_xlabel("Time step")

    # Plot mean.
    ax = traverse_experiments(exp_runs, plot_mean_call, {'axis': ax},
                              data_keys, agent_keys, env_keys, mode='aggregate')

    # Plot uncertainty around the mean.
    func_param = {'axis': ax, 'uncertainty_mode': uncertainty_mode}
    ax = traverse_experiments(exp_runs, plot_uncertainty_call, func_param,
                              data_keys, agent_keys, env_keys, mode='aggregate')

    # Save figure or show plot.
    ax.legend(labels=new_legend)
    if save_fig is not None:
        f.savefig(save_fig)
    else:
        plt.show()
