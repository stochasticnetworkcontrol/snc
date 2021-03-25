"""
Constructs commands to run experiments in docker images.
Originally written to facilitate large scale experimentation using cloud-based compute clusters.

Warning: This code has been significantly refactored to make it cloud platform agnostic and has not
been tested. These are the building blocks which will likely need to be adjusted to your chosen
cloud.
"""

import os
import json
import itertools
import subprocess
import numpy as np

from datetime import datetime
from copy import deepcopy
from typing import Optional, Iterable, Any, Dict, List, Tuple
from src.snc.environments import scenarios
from src.experiment.experiment_utils import flatten_dict, unflatten_dict


def prep_for_json(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a serialisable dictionary suitable for cross-language JSON encoding.

    :param data_dict: The data dictionary to be updated.
    :return: A copy of the dictionary with troublesome values recoded as acceptable values,
    """
    out_dict = deepcopy(data_dict)
    for k, v in data_dict.items():
        # Handle numpy arrays and similar.
        if hasattr(v, 'tolist'):
            out_dict[k] = v.tolist()
    str_dict = json.dumps(out_dict)
    str_dict = str_dict.replace("Infinity", '"Infinity"')

    out_dict = json.loads(str_dict)
    return out_dict


def frange(start, stop, step):
    """
    Formulates a range for floating point values similar to Python's built-in range class.
    Note that this may be subject to floating point inaccuracies.

    :param start: The initial value.
    :param stop:  The maximal value (not included in the range),
    :param step:  The difference between adjacent values.
    :returns: A generator similar to that of Python's built-in range functionality.
    """
    assert (start < stop and step > 0) or (start > stop and step < 0)
    increasing = start < stop
    while (increasing and start < stop) or (not increasing and stop < start):
        yield float(start)
        start += step


def build_command(
        lab_name: str,
        docker_image: str,
        environment: str,
        param_names: List[str],
        param_values: List[Any],
        entry_point: Optional[str] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Builds a command compatible with the subprocess module (specifically subprocess.run) using the
    parameter values provided.

    :param lab_name: The name of the 'lab' under which all experiments are to be collected.
    :param docker_image: URL to the docker image to run.
    :param environment: The name of the SNC environment to run. Must be implemented in
        `scenarios.py`.
    :param param_names: List of the parameter names to pass in the CLI command.
    :param param_values: The values corresponding to the parameter names in the param_names list.
    :param entry_point: Optional entry point override for the docker image.
    :return: A list which can be passed to subprocess.run to kick off an experiment and
        a dictionary summarising the parameters of each experiment.
    """
    cmd = ['docker', 'run', docker_image]
    cmd_dict: Dict[str, Any] = {'docker_image': docker_image}
    assert entry_point is not None
    assert isinstance(entry_point, str) and entry_point[-3:] == '.py', \
        "entry_point script is not a Python script. Only python scripts currently supported."
    cmd += ['--command', f'python {entry_point}', '--lab_name', lab_name]
    cmd_dict['command'] = f'python {entry_point}'
    script_args = f'--env_name {environment} --lab_name {lab_name}'
    script_args_dict: Dict[str, Any] = {'environment': environment, 'lab_name': lab_name}
    for name, value in zip(param_names, param_values):
        script_args += f' --{name} {value}'
        if name == 'env_param_overrides':
            # Transform the string back to a dict so that it can be saved nicely in JSON.
            # Call json.loads twice in order to undo the double encoding required for passing the
            # dictionary to subprocess.run as a string.
            env_params = prep_for_json(scenarios.get_scenario_default_params(environment))
            env_params.update(json.loads(json.loads(value)))
            script_args_dict[name] = env_params
        elif name in ['rl_agent_params', 'hedgehog_param_overrides']:
            # Transform the string back to a dict so that it can be saved nicely in JSON.
            # Call json.loads twice in order to undo the double encoding required for passing the
            # dictionary to subprocess.run as a string.
            agent_params = json.loads(json.loads(value))
            script_args_dict[name] = agent_params
        else:
            script_args_dict[name] = value
    cmd += ['--args', script_args]
    cmd_dict['script_args'] = script_args_dict
    return cmd, cmd_dict


def get_iterator_from_param_name(environment_config: Dict[str, List], param_name: str) -> Iterable:
    """
    Builds an iterator for the parameter of the SNC environment provided given the specification
    dictionary read in from the JSON config file.

    :param environment_config: Dictionary of parameters and values for SNC environment set up.
    :param param_name: Name of the parameter for which an iterator is to be created.
    :return: The iterable containing all specified parameter values.
    """
    # Attain and validate the list specifying which values to sweep over.
    spec_list = environment_config[param_name]
    assert isinstance(spec_list, list) and len(spec_list) == 2
    # Determine the type and values over which to sweep. Only ranges and sets are supported.
    sweep_type, sweep_values = spec_list
    assert isinstance(sweep_type, str)
    if sweep_type == "frange":
        # For a range build a generator which acts in effectively the same way as Python's built-in
        # range object but can handle non-integer values. (frange for float range).
        assert len(sweep_values) == 3, \
            'Each range specification needs a start, stop and step value.'
        assert all([isinstance(x, (float, int)) for x in sweep_values])
        param_iterator = frange(*sweep_values)
    elif sweep_type == "irange":
        # For an integer range use as Python's built-in range object.
        assert len(sweep_values) == 3, \
            'Each range specification needs a start, stop and step value.'
        assert all([isinstance(x, int) for x in sweep_values])
        param_iterator = range(*sweep_values)
    elif sweep_type == "set":
        # If sweeping over a set of discrete values simply loop over the values provided.
        # We ensure that the values are all of the same type and are of a supported type first.
        # The iterable is then simply the original list provided.
        assert all([isinstance(x, type(sweep_values[0])) for x in sweep_values])
        assert isinstance(sweep_values[0], (int, float, bool, list, str))
        if isinstance(sweep_values[0], list):
            assert all([isinstance(a, type(sweep_values[0])) for a in sweep_values])
            assert isinstance(sweep_values[0][0], (int, bool, float, list))
        param_iterator = sweep_values
    else:
        raise ValueError('JSON config file contains values that are not supported')
    return param_iterator


def build_arg_strings_from_dict(
        param_dict: Dict[str, List],
        param_spaces: bool,
        flattened: bool = False
    ) -> List[str]:
    """
    Create JSON strings for each permutation of the environment override parameters to search over.

    :param param_dict: Dictionary where keys are parameters and values are lists
        denoting the search space type and the search space components for the space of parameter
        values to consider.
    :param param_spaces: Boolean denoting whether the config is describing spaces of parameters or
        simply parameter values themselves.
    :param flattened: Boolean determining whether the incoming param_dictionary has been flattened
        and should therefore be unflattened (nested) once processed.
    :return: A list of JSON strings which when parsed defines a dictionary of environment parameter
        overrides.
    """
    # Initialise an empty list to collect the argument strings.
    strings = []
    # The order of the parameters will be fixed so we can simply form a list of them from the set
    # of dictionary keys.
    params = list(param_dict.keys())
    # Initialise a list for the iterators corresponding to each parameter.
    iterators = [*range(len(params))]  # type: List[Any]
    # Iterate over parameters and attain the corresponding value iterators.
    for i, param in enumerate(params):
        assert isinstance(param, str)
        if param_spaces:
            # Attain an iterator of parameter values to sweep through.
            iterators[i] = get_iterator_from_param_name(param_dict, param)
        else:
            # The config specifies a single value. To be compatible with itertools.product below
            # we place the single parameter in a list on its own.
            iterators[i] = [param_dict[param]]
    # Take each parameter combination in the Cartesian product of all off the search spaces and form
    # a valid JSON string for the values which can later be passed to the snc script.
    for combination in itertools.product(*iterators):
        param_dict = dict(zip(params, combination))
        if flattened:
            param_dict = unflatten_dict(param_dict)
        strings.append(json.dumps(json.dumps(param_dict)))
    return strings


def validate_env_config(environment_name: str, env_config: Dict[str, List],
                        sweep_space: bool = True) -> None:
    """
    Validates an environment config dictionary which results from parsing a JSON file.
    This ensures that the iterators generated match the valid parameter types for the environment
    used in experiments.

    :param environment_name: The name of the environment to be parameterised.
    :param env_config: Dictionary of parameter: iterator specification pairs.
    :param sweep_space: Boolean denoting whether the config is describing spaces of parameters or
        simply parameter values themselves.
    """
    default_params = scenarios.get_scenario_default_params(environment_name, 1)
    for param, value in env_config.items():
        assert param in default_params, f"Passing a parameter not supported for {environment_name}."

        if isinstance(default_params[param], (np.ndarray, list, tuple)):
            if sweep_space:
                assert env_config[param][0] == 'set', 'Trying to form non set iterator for list ' \
                                                      'parameter'
                assert isinstance(value[1], list)
                for element in value[1]:
                    assert np.array(element).shape == np.array(default_params[param]).shape, \
                        f"Passing an array of an incompatible shape for parameter {param} of " \
                        f"{environment_name}."
            else:
                assert np.array(value).shape == np.array(default_params[param]).shape, \
                    f"Passing an array of an incompatible shape for parameter {param} of " \
                    f"{environment_name}."

        elif isinstance(default_params[param], float):
            if sweep_space:
                assert value[0] in ["frange", "set", "irange"], \
                    f"Trying an unsupported iterator type {value[0]} for float parameter" \
                    f"{param} of {environment_name}."
                if value[0] == "frange":
                    assert len(value[1]) == 3
                    assert all([isinstance(x, (int, float)) for x in value[1]])
                if value[0] == "irange":
                    assert len(value[1]) == 3
                    assert all([isinstance(x, int) for x in value[1]])
            else:
                assert isinstance(value, (int, float)), "Non-float value passed for float" \
                                                        f"parameter {param} of {environment_name}."

        elif isinstance(default_params[param], int):
            if sweep_space:
                assert value[0] in ["irange", "set"], "Trying an iterator other than irange " \
                                                      f"or set for integer parameter {param}" \
                                                      f"of {environment_name}."
                if value[0] == "irange":
                    assert len(value[1]) == 3
                    assert all([isinstance(x, int) for x in value[1]])
            else:
                assert isinstance(value, int), f"Non-integer value passed for parameter {param}" \
                                               f"of {environment_name}."

        else:
            raise NotImplementedError(
                f'Default environment parameter type {type(default_params[param])} not currently '
                'supported. Verify that parameter type is correct and then extend '
                'command_builder.py to handle new parameter type.'
            )


def build_command_iterator_dict(config: Dict[str, Dict[str, Any]]) -> Dict[str, Iterable]:
    """
    Build a dictionary of parameter: iterable pairs from a dictionary which contains a range/set
    of values to iterate over for each parameter (in each environment).

    :param config: Dictionary where the keys are environments and the values are dictionaries of
        parameter name (str): parameter sweep range/set pairs.
    :return: A dictionary where the keys are the environment names and the values are tuples
        containing a list of parameter names, an iterator over parameter settings and a integer
        number of repeats for each experiment.
    """
    # Initialise the dictionary to be filled with iterators for each parameter.
    iterator_dict: Dict[str, Iterable] = dict()
    # The keys of the config dictionary should be the environment names.
    # The values are dictionaries of experiment set configurations OR a list of experiments
    # (i.e. one parameter set per experiment).
    for environment, environment_config in config.items():
        # Ensure that the environment name is supported.
        assert environment in scenarios.SCENARIO_CONSTRUCTORS, \
            'Environment provided not supported by snc.'
        # Make sure that we aren't repeating an environment.
        assert environment not in iterator_dict, 'Repeated environment configuration.'

        # Test to see if we are in the list of experiments case or the parameter space specification
        # case.
        if isinstance(environment_config, list):
            # Loop through the list of parameterisations and build the run commands one at a time.
            for i, param_set in enumerate(environment_config):
                param_names, param_values, repeats, lab_name = get_iterator_dict_entry(
                    param_set, environment, False
                )
                iterator_dict[environment + str(i)] = (environment, param_names, [param_values],
                                                       repeats, lab_name)
        else:
            # We need to build iterators for each variable and will ultimately take the Cartesian
            # product of these spaces and sweep through it.
            param_names, param_iterables, repeats, lab_name = get_iterator_dict_entry(
                environment_config, environment, True)
            # The overall parameter iterator iterates through the space formed by the Cartesian
            # product of the search spaces for each parameter.
            iterator_dict[environment] = (environment, param_names,
                                          itertools.product(*param_iterables), repeats, lab_name)
    return iterator_dict


def get_iterator_dict_entry(environment_config: Dict[str, Any],
                            environment: str,
                            param_spaces: bool = False) -> Tuple[List[str], List[Any], int, str]:
    """
    Attains the entry for a parameter in the dictionary used to build up commands to run
    experiments. This handles both cases of listed experiments and parameter spaces to iterate
    through.

    :param environment_config: The dictionary of parameter specifications (or explicit values) for a
        (set of) experiment(s).
    :param environment: The name of the environment for which experiment command strings are to be
        generated.
    :param param_spaces: Boolean flagging whether the config specifies spaces to sweep over or
        specific parameter values.
    :return: A list of parameter names, a list of parameter values or search spaces, an integer
        number of experiment repeats to be performed (in parallel) and the lab in which to run
        experiments.
    """
    # Initialise lists for parameter names and the iterators build_command_iterator_dict over which
    # we will loop.
    param_names = []
    param_values: List[Iterable] = []
    repeats = 1  # Default to one run per parameter set overridden if required.
    lab_name = 'default_lab'  # Allow a default lab in case none is provided.
    # For each possible parameter to pass to the snc script call build an iterator/attain the value.
    for param, value in environment_config.items():
        assert isinstance(param, str)
        # Handle repeat experiments.
        if param == 'repeats':
            assert isinstance(value, int), 'Number of experiment repeats is not an integer.'
            repeats = value
        elif param == 'lab_name':
            assert isinstance(param, str), 'Lab name must be a string.'
            lab_name = value
        else:
            # Handle the special cases where we pass dictionaries of parameters.
            if param in ['rl_agent_params', 'hedgehog_param_overrides', 'env_param_overrides',
                         'maxweight_param_overrides']:
                flattened = False
                if param == 'env_param_overrides':
                    # Validate that the parameters provided are suitable for the environment to
                    # be run.
                    validate_env_config(environment, value, param_spaces)
                elif param == 'hedgehog_param_overrides':
                    value = flatten_dict(value)
                    flattened = True
                param_strings = build_arg_strings_from_dict(value, param_spaces, flattened)
                if param_spaces:
                    # We have a list of strings one for each agent set up.
                    param_values.append(param_strings)
                else:
                    # In this case there should only be one string for the agent parameters as
                    # the config lists experiment set ups one at a time.
                    assert len(param_strings) == 1
                    param_values.append(param_strings[0])
            else:
                if param_spaces:
                    # Attain a standard iterable for the parameter where it is not an
                    # environment override.
                    param_iterable = get_iterator_from_param_name(environment_config, param)
                    param_values.append(param_iterable)
                else:
                    # There is no iterator as there is only one parameter value so we take this
                    # value on its own.
                    param_values.append(value)
            param_names.append(param)
    return param_names, param_values, repeats, lab_name


def run_commands(config_file_path: str,
                 docker_image: str,
                 entry_point: Optional[str] = None
                 ) -> List[Dict[str, Any]]:
    """
    Formulates, runs and collects responses for commands for all parameter set ups in the
    space of parameterisations defined by the JSON config.

    :param config_file_path: Path to the JSON file defining the parameter values to sweep over in
        the experiments.
    :param docker_image: name/location of a docker image to run.
    :param entry_point: An optional override for the docker image entry point when running
        experiments. Note that this is applied to all runs.
    :return: A dictionary of uuid: list_of_experiment_arguments key-value pairs.
    """
    # Read in the experiment set configuration.
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)
    # Attain a dictionary of parameter: iterator pairs which can be used to form the space of
    # experiment parameterisations.
    iterator_dict = build_command_iterator_dict(config)
    # Initialise a dictionary to track experiment parameters alongside the parameters used to set
    # them up.
    completed_commands = []
    i = 0  # Run counter.
    # The keys of the dictionary of iterators are the environment names.
    for environment in iterator_dict:
        # The value of the dictionary is a tuple of parameter names, parameter value iterables and
        # an integer number of experiment repeats.
        environment_name, param_names, iterator, repeats, lab_name = iterator_dict[environment]
        for param_values in iterator:
            # Build up the command for the current experiment
            cmd, cmd_dict = build_command(
                lab_name, docker_image, environment_name, param_names, param_values, entry_point
            )
            # and run it as many times as requested.
            for _ in range(repeats):
                # Update counter and print counter and command.
                i += 1
                print(f"{i}: {cmd[-1]}")
                # Pass the command to the command line as a list and capture any response as a
                # string.
                response = subprocess.run(cmd, capture_output=True, text=True)

                if response.returncode == 0:
                    cmd_dict['user'] = os.environ['USER']
                    now = datetime.now()
                    cmd_dict['date_timestamp'] = datetime.strftime(now, "%Y_%m_%d__%H:%M:%S.%f")
                    cmd_dict['date_ordinal'] = now.date().toordinal()
                    completed_commands.append(cmd_dict)
                else:
                    raise RuntimeError('Experiment Run Failed')
    # Return the dictionary of uuid: experiment_parameter pairs.
    return completed_commands
