import argparse
import ast
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Optional, Dict, Union, List
from warnings import warn

from src.experiment.experiment_utils import get_args_string
from src import snc, snc as mdt, snc as ps
from src.snc import BigStepHedgehogAgent
from src.snc \
    import PureFeedbackMIPHedgehogAgent
from src.snc \
    import PureFeedbackStationaryHedgehogAgent
import src.snc.environments.scenarios as scenarios
from src.snc import set_up_json_logging
import src.snc.simulation.store_data.reporter as rep
from src.snc.simulation.utils import load_agents
import src.snc.simulation.utils.validation_utils as validation_utils
from src.snc import is_routing_network

set_up_json_logging()


def run_policy(policy_simulator: ps.SncSimulator, num_simulation_steps: int, server_mode: bool,
               is_hedgehog: bool, save_location: str, job_gen_seed: Optional[int] = None):
    """
    Run a policy simulator, with a reporter, saving the result in given location.

    :param policy_simulator: the policy simulator object.
    :param num_simulation_steps: the number of steps the simulation runs.
    :param server_mode: when experiment runs locally, this flag controls whether to show live plots
        and wait for input before closing them at the end of the simulation.
    :param is_hedgehog: whether the policy simulator object is hedgehog specifically.
    :param save_location: what to name the data when it is saved in logs directory.
    :param job_gen_seed: Job generator random seed.
    :return: Run results - a dictionary including data on cost, state and action processes.
    """
    env = policy_simulator.env

    initial_state = env.state
    time_interval = env.job_generator.sim_time_interval

    min_draining_time = mdt.compute_minimal_draining_time_from_env_cvxpy(initial_state, env)
    print(f"Minimal draining time: {min_draining_time}\n"
          f"Minimal draining simulation steps: {min_draining_time * time_interval}")

    plot_freq = 100
    is_routing = is_routing_network(env)
    handlers = validation_utils.get_handlers(server_mode, num_simulation_steps, plot_freq,
                                             time_interval, is_hedgehog, is_routing)
    reporter = rep.Reporter(handlers=handlers)

    return policy_simulator.run(num_simulation_steps, reporter, save_location, job_gen_seed)


def run_validation(arguments: argparse.Namespace) -> Dict[str, str]:
    """
    Run the validation on a particular scenario.

    :param arguments: Namespace of experiment parameters.
    """
    assert arguments.env_param_overrides['job_gen_seed'] is not None
    assert arguments.seed is not None
    # Note that if job_gen_seed was not in env_param_overrides, then at this point we will have:
    #   arguments.env_param_overrides['job_gen_seed'] == arguments.seed.
    job_gen_seed = arguments.env_param_overrides['job_gen_seed']
    global_seed = arguments.seed + 100
    agent_seed = arguments.seed + 200
    mpc_seed = arguments.seed + 300
    np.random.seed(global_seed)
    print(f"job_gen_seed {job_gen_seed}")
    print(f"global_seed {global_seed}")
    print(f"agent_seed {agent_seed}")
    print(f"mpc_seed {mpc_seed}")
    save_locations = dict()

    # Get Scenario
    _, env = scenarios.load_scenario(
        arguments.env_name, job_gen_seed, arguments.env_param_overrides)

    # Initialise an agent counter to ensure that the right checkpoint is loaded for each agent.
    rl_agent_count = 0
    for agent_name in arguments.agents:
        env.reset_with_random_state(job_gen_seed)
        agent_args = {}
        name_alias = agent_name  # Set name of folder storing results to agent_name by default.
        if agent_name in load_agents.HEDGEHOG_AGENTS:
            if arguments.hedgehog_param_overrides is None:
                arguments.hedgehog_param_overrides = dict()
            agent_args['hh_overrides'] = arguments.hedgehog_param_overrides
            agent_args['discount_factor'] = arguments.discount_factor
            agent_args['debug_info'] = arguments.debug_info
            agent_args['agent_seed'] = agent_seed
            agent_args['mpc_seed'] = mpc_seed
            # Replace directory name if name passed as an agent parameter.
            name_alias = arguments.hedgehog_param_overrides.get('name', agent_name)
        elif agent_name == 'distribution_with_rebalancing_heuristic':
            agent_args['safety_stocks'] = 20 * np.ones(env.state.shape)
        elif agent_name in ['reinforce', 'ppo']:
            agent_args['discount_factor'] = arguments.discount_factor
            if arguments.rl_agent_params:
                # Update agent_args accordingly.
                if rl_agent_count < len(arguments.rl_agent_params):
                    if 'discount_factor' in arguments.rl_agent_params[rl_agent_count]:
                        warn('WARNING: Overriding provided discount factor with agent specific '
                             'discount factor for {agent_name} agent')
                agent_args.update(arguments.rl_agent_params[rl_agent_count])
            else:
                if agent_name == "ppo":
                    raise ValueError("When running a PPO agent you must provide agent parameters.")
                else:
                    warn("REINFORCE agent being run default agent parameters.")
            agent_args['rl_checkpoint'] = arguments.rl_checkpoints[rl_agent_count]
            rl_agent_count += 1
        elif agent_name == 'maxweight' or agent_name == 'scheduling_maxweight':
            if arguments.maxweight_param_overrides is None:
                arguments.maxweight_param_overrides = dict()
            agent_args['overrides'] = arguments.maxweight_param_overrides
            agent_args['agent_seed'] = agent_seed
            agent_args['mpc_seed'] = mpc_seed
            # Replace directory name if name passed as an agent parameter.
            name_alias = arguments.maxweight_param_overrides.get('name', agent_name)
        else:
            agent_args['agent_seed'] = agent_seed

        agent = load_agents.get_agent(agent_name, env, **agent_args)
        sim = ps.SncSimulator(env, agent, **arguments.__dict__)

        print(f'\nSimulating {agent.name}...')
        validation_utils.print_agent_params(agent)

        is_hedgehog = isinstance(agent, (BigStepHedgehogAgent, PureFeedbackStationaryHedgehogAgent,
                                         PureFeedbackMIPHedgehogAgent))
        save_location = f'{arguments.logdir}/{name_alias}'
        run_policy(sim, arguments.num_steps, arguments.server_mode, is_hedgehog, save_location,
                   job_gen_seed)

        if is_hedgehog:
            assert isinstance(agent, (BigStepHedgehogAgent, PureFeedbackStationaryHedgehogAgent,
                                      PureFeedbackMIPHedgehogAgent))
            validation_utils.print_workload_to_physical_resources_indexes(agent.workload_tuple.nu)

        save_locations[agent.name] = save_location
        print(f'Data stored at: {save_location}.')
        print(f'Finished simulating {agent.name}.\n')

    print(f"job_gen_seed: {arguments.env_param_overrides.get('job_gen_seed')}")
    print("End of simulation!")
    if not arguments.server_mode:
        plt.ioff()
        plt.show()
    return save_locations


def parse_args() -> argparse.Namespace:
    """Processes command line arguments and collects them in the named tuple returned·"""
    params = argparse.ArgumentParser(description="Experiment Arguments.")
    params.add_argument("--env_name", type=str, default="simple_reentrant_line",
                        help="Name of environment to run on."
                             "Must be in the list of implemented scenarios.")
    params.add_argument("--agents", type=str, default='all',
                        help='whitespace separated list of agents to run.'
                             'Also accepts special case of "all" which runs all known agents.')
    params.add_argument("-ns", "--num_steps", type=int, default=2500,
                        help="Number of simulation steps.")
    params.add_argument("-ep", "--env_param_overrides", type=str, default='{}',
                        help="JSON formatted dictionary of environment parameter overrides. "
                             "May be a string or a path to a JSON file.")
    params.add_argument("--rl_agent_params", type=str, default=None,
                        help="JSON formatted list of dictionaries of agent parameters. "
                             "One dictionary per RL agent to be run. "
                             "If a single RL agent is run this may be a single dictionary or"
                             "JSON file.")
    params.add_argument("-hhp", "--hedgehog_param_overrides", type=str, default='{}',
                        help="JSON formatted dictionary of Hedgehog parameter overrides. "
                             "May be a string or a path to a JSON file.")
    params.add_argument("-mwp", "--maxweight_param_overrides", type=str, default='{}',
                        help="JSON formatted dictionary of MaxWeight parameter overrides. "
                             "May be a string or a path to a JSON file.")
    params.add_argument("-df", "--discount_factor", type=float, default=0.999999,
                        help="Discount factor applied to future rewards.")
    params.add_argument("--logdir", type=str, default=os.path.join(
        os.path.dirname(snc.__file__), 'logs'))
    params.add_argument("--rl_checkpoints", type=str, default=None,
                        help="A list of paths to a directories where the TensorFlow model weights "
                             "are saved. These should be ordered in a corresponding order to the "
                             "agents in the agents list. If using --agents all then REINFORCE "
                             "precedes PPO.")
    params.add_argument("--seed", type=int, default=42, help="Random seed.")
    params.add_argument("--server_mode", action="store_true",
                        help="Neither show live plots nor ask for input before closing them at the "
                             "end of the simulation. If simulation runs in art, this parameter is "
                             "set to True independent on this argument.")
    params.add_argument("--debug_info", action="store_true",
                        help="Print debug information while running.")

    # Collect the parameters in a namespace.
    parsed_params = params.parse_args()
    parsed_params = process_parsed_args(parsed_params)

    return parsed_params


def process_parsed_args(parsed_params: argparse.Namespace) -> argparse.Namespace:
    assert parsed_params.env_name in scenarios.SCENARIO_CONSTRUCTORS, \
        "Scenario passed does not exist."

    if parsed_params.rl_checkpoints:
        # Cast the string to a list.
        parsed_params.rl_checkpoints = ast.literal_eval(parsed_params.rl_checkpoints)

        for ckpt in parsed_params.rl_checkpoints:
            assert os.path.exists(ckpt), \
                f'RL parameter folder {ckpt} not found.'

    if parsed_params.agents.lower() == 'all':
        # Only include RL agents if model weights supplied.
        with_rl_agent = bool(parsed_params.rl_checkpoints)
        if with_rl_agent:
            assert len(parsed_params.rl_checkpoints) == load_agents.NUM_RL_AGENTS, \
                "The number of agent checkpoints provided does not match the number of RL agents."
        else:
            warn("No reinforcement learning agents will be run as no load checkpoints provided.")
        parsed_params.agents = load_agents.get_all_agent_names(parsed_params.env_name,
                                                               with_rl_agent)
    else:
        parsed_params.agents = [a.strip() for a in parsed_params.agents.split()]

    # Set the logging folder from the load automatically.
    now = datetime.now()
    time_stamp_for_logs = datetime.strftime(now, "%y_%m_%d_%H%M%S.%f")
    parsed_params.logdir = os.path.join(parsed_params.logdir, time_stamp_for_logs)

    # We want to make sure that the seed is not None
    if parsed_params.seed is None:
        parsed_params.seed = int(time.time())

    if parsed_params.env_param_overrides:
        parsed_params.env_param_overrides = process_environment_parameters(
            parsed_params.env_param_overrides, parsed_params.seed)

    if parsed_params.rl_agent_params:
        rl_agent_params = process_agent_parameters(parsed_params.rl_agent_params)
        parsed_params.rl_agent_params = post_process_rl_agent_params(rl_agent_params,
                                                                     parsed_params.agents)
    if parsed_params.hedgehog_param_overrides:
        hh_agent_params = process_agent_parameters(parsed_params.hedgehog_param_overrides)
        parsed_params.hedgehog_param_overrides = cast_overrides_to_numpy_arrays(hh_agent_params)

    if parsed_params.maxweight_param_overrides:
        mw_agent_params = process_agent_parameters(parsed_params.maxweight_param_overrides)
        parsed_params.maxweight_param_overrides = cast_overrides_to_numpy_arrays(mw_agent_params)

    if not os.path.isdir(parsed_params.logdir):
        os.makedirs(parsed_params.logdir)
    with open(os.path.join(parsed_params.logdir, 'validation_params.txt'), 'w') as param_file:
        param_file.write(get_args_string(parsed_params))

    return parsed_params


def process_environment_parameters(env_param_overrides: str, seed: int) \
        -> Dict[str, Union[float, List, np.ndarray]]:
    """
    Returns updated environment parameter from a JSON file or a string of a dictionary.

    :param env_param_overrides: The namespace to be updated.
    :param seed: General random number generator seed.
    :return: env_param_overrides: Dictionary containing updated environment np.ndarray.
    """
    # Support environment parameters passed as a path to a JSON file or as a string of a dictionary.
    if os.path.exists(env_param_overrides):
        with open(env_param_overrides, 'r') as json_file:
            env_param_overrides_json = json.load(json_file)
    else:
        env_param_overrides_json = json.loads(env_param_overrides)
    # We are always going to use the seed set here and not the one in environment parameters
    if 'job_gen_seed' in env_param_overrides_json:
        assert env_param_overrides_json["job_gen_seed"] is not None
        if env_param_overrides_json["job_gen_seed"] != seed:
            warn("Seed for environment job generator differs from general random seed ")
    else:
        env_param_overrides_json['job_gen_seed'] = seed
    env_param_overrides_updated = cast_overrides_to_numpy_arrays(env_param_overrides_json)
    return env_param_overrides_updated


def cast_overrides_to_numpy_arrays(param_overrides: Dict[str, Union[float, List]]) \
        -> Dict[str, Union[float, List, np.ndarray]]:
    """
    All list type objects will be cast to numpy arrays before passing. This is needed for the
    current implementation of ControlledRandomWalk environments.

    :param param_overrides: Dictionary of parameters.
    :return: new_param_overrides: Dictionary similar to param_overrides, but with list values
        replaced with numpy arrays.
    """
    new_param_overrides: Dict[str, Union[float, List, np.ndarray]] = dict()
    for p in param_overrides:
        if isinstance(param_overrides[p], list):
            new_param_overrides[p] = np.array(param_overrides[p])
        else:
            new_param_overrides[p] = param_overrides[p]
    return new_param_overrides


def process_agent_parameters(agent_params: str) -> Dict[str, Union[float, List]]:
    """
    Process agents params.
    Note that this function purposely mutates params causing side effects.

    :param agent_params: Agent parameters to be processed passed as a file path, or as a dictionary,
        or as a list of dictionaries.
    :return: agent_params:
    """
    # Load agent parameters considering several cases.

    # If simply a file path is provided.
    if os.path.exists(agent_params):
        with open(agent_params, 'r') as json_file:
            agent_params_data = json.load(json_file)
    else:
        # If a dictionary or list of dictionaries and/or file paths is provided first attain the
        # list.
        try:
            agent_params_data = ast.literal_eval(agent_params)
        except SyntaxError:
            raise ValueError('Argument is invalid. Possibly file doesnt exist. Value passed was '
                             f'{agent_params}')
        # Handle the case of a list for multiple (or even a single) agents.
        if isinstance(agent_params_data, list):
            for i, entry in enumerate(agent_params_data):
                if not isinstance(entry, dict):
                    assert isinstance(entry, str) and os.path.exists(entry), \
                        "JSON file for agent parameters not found."
                    with open(entry, 'r') as json_file:
                        agent_params_data[i] = json.load(json_file)
                else:
                    assert isinstance(entry, dict), "If not passing a JSON file of agent " \
                                                    "parameters you must pass a dictionary " \
                                                    "(JSON string)."
        else:
            # Users may pass a single dictionary of agent parameters when running a single RL agent.
            assert isinstance(agent_params_data, dict), "If not passing a JSON file of agent " \
                                                    "parameters you must pass a dictionary " \
                                                    "(JSON string)."
    return agent_params_data


def post_process_rl_agent_params(agent_params: Union[Dict, List[Dict]], agent_names: List[str]) \
        -> List[Dict[str, Union[float, List]]]:
    """
    If we have parsed parameters for multiple RL agents, this function checks that the list of
    parameters has as many items as the list of RL agents (currently only checks for 'ppo' and
    'reinforce' names). Otherwise, if we have parsed parameters for a single RL agent, it wraps the
    parameters with a list.

    :param agent_params: Dictionary or list of dictionaries with the parsed parameters for the RL
        agents.
    :param agent_names: List of parsed names of agents.
    :return: List of dictionaries with the parsed parameters for all the RL agents, even if there
        is only one.
    """
    if isinstance(agent_params, list):
        assert len(agent_params) == agent_names.count('ppo') \
               + agent_names.count('reinforce'), "The number of agent parameter sets provided " \
                                                 "for RL agents does not match the number of RL " \
                                                 "agents to run."
    else:
        assert agent_names.count('ppo') + agent_names.count('reinforce') == 1, \
            "Users may only pass a dictionary of RL agent parameters when there a single RL " \
            "agent is being validated."
        agent_params = [agent_params]
    return agent_params


if __name__ == '__main__':
    # Simply get the arguments and run main.
    args = parse_args()
    run_validation(args)
