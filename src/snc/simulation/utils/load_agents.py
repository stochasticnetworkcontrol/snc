from typing import Any, cast, Callable, Dict, List, Union, Tuple, Type, Optional

from snc.agents.agent_interface import AgentInterface
import snc.agents.general_heuristics.custom_parameters_priority_agent as priority
from snc.agents.general_heuristics.random_nonidling_agent import RandomNonIdlingAgent
import snc.agents.hedgehog.hh_agents.hedgehog_agent_interface as hh_int
from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
from snc.agents.hedgehog.hh_agents.pure_feedback_mip_hedgehog_agent \
    import PureFeedbackMIPHedgehogAgent
from snc.agents.hedgehog.hh_agents.pure_feedback_stationary_hedgehog_agent \
    import PureFeedbackStationaryHedgehogAgent
from snc.agents.hedgehog.params import AsymptoticCovarianceParams, \
    BigStepLayeredPolicyParams, \
    BigStepPenaltyPolicyParams, \
    DemandPlanningParams, \
    HedgehogHyperParams, \
    StrategicIdlingParams, \
    WorkloadRelaxationParams
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_foresight import StrategicIdlingForesight
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedgehog_gto import \
    StrategicIdlingHedgehogGTO, StrategicIdlingHedgehogGTO2, StrategicIdlingHedgehogNaiveGTO
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.agents.maxweight_variants.maxweight_agent import MaxWeightAgent
from snc.agents.maxweight_variants.scheduling_maxweight_agent import SchedulingMaxWeightAgent
from snc.agents.problem_specific_heuristics.distribution_with_rebalancing_local_priority_agent \
    import DistributionWithRebalancingLocalPriorityAgent
from snc.agents.rl.snc_agent_interface_for_rl import RLSimulationAgent
from snc.agents.rl.agents import create_reinforce_agent, create_ppo_agent
from snc.environments.controlled_random_walk import ControlledRandomWalk
import snc.environments.rl_environment_wrapper as rl_env
import snc.utils.snc_types as types


def get_strategic_idling_class(si_class_name: str) -> Type[StrategicIdlingCore]:
    """
    TODO: Move into HedgehogAgentInterface or BigStepHedgehogAgent.
    Return strategic idling class from class name.

    :param si_class_name: String with strategic class name.
    :return: Strategic idling class.
    """
    classes = [
        StrategicIdlingCore,
        StrategicIdlingForesight,
        StrategicIdlingHedgehogGTO,
        StrategicIdlingHedgehogGTO2,
        StrategicIdlingHedgehogNaiveGTO,
        StrategicIdlingHedging
    ]
    for s in classes:
        if si_class_name == s.__name__:
            return s
    raise Exception(f"Strategic idling class name is not valid: {si_class_name}.")


def get_hedgehog_params_as_named_tuple(
        default_params: Union[
            AsymptoticCovarianceParams,
            BigStepLayeredPolicyParams,
            BigStepPenaltyPolicyParams,
            DemandPlanningParams,
            HedgehogHyperParams,
            StrategicIdlingParams,
            WorkloadRelaxationParams
        ],
        **overrides: Optional[Dict[str, Dict[str, Union[str, float]]]]
) -> Union[
    AsymptoticCovarianceParams,
    BigStepLayeredPolicyParams,
    BigStepPenaltyPolicyParams,
    DemandPlanningParams,
    HedgehogHyperParams,
    StrategicIdlingParams,
    WorkloadRelaxationParams
]:
    """
    Returns parameters as named tuples with updated values, or with default ones if no overrides
    where given.

    :param default_params: Default parameters for one of the valid named tuples, namely
        AsymptoticCovarianceParams, BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams,
        DemandPlanningParams, HedgehogHyperParams, StrategicIdlingParams and
        WorkloadRelaxationParams.
    :param overrides: Dictionary with the parameters to replace the default values. The keys should
        be named after the named tuple they want to update.
    :return: Updated parameters, or default parameters if no overrides where given.
    """
    name_of_named_tuple = type(default_params)
    params_dict = overrides.get(name_of_named_tuple.__name__)
    if params_dict is None:
        return default_params

    # Convert boolean values passed as strings when using JSON and ART.
    for key, val in params_dict.items():
        if isinstance(val, str) and val in ["True", "False"]:
            params_dict[key] = eval(val)

    new_params_dict = {**default_params._asdict(), **params_dict}
    return name_of_named_tuple(**new_params_dict)


def get_hedgehog_hyperparams(**overrides) -> Tuple:
    """
    Get parameters for Hedgehog agent. If some parameter is not specified in overrides, then it
    takes a default value.

    :param overrides: Parameters coming from JSON file.
    :return: (ac_params, wk_params, si_params, po_params, hh_params, si_class, name):
        - ac_params: Hyperparameters to estimate the workload asymptotic covariance matrix.
        - wk_params: Hyperparameters to compute the workload matrix.
        - si_params: Hyperparameters for computing the strategic possible idling directions.
        - po_params: Hyperparameters for computing the activity rates policy.
        - hh_params: Hyperparameters for the main loop of the Hedgehog algorithm.
        - si_class: Strategic idling class.
        - dp_class: Demand planning class.
        - name: Agent name.
    """
    ac_params = get_hedgehog_params_as_named_tuple(AsymptoticCovarianceParams(), **overrides)
    wk_params = get_hedgehog_params_as_named_tuple(WorkloadRelaxationParams(), **overrides)
    si_params = get_hedgehog_params_as_named_tuple(StrategicIdlingParams(), **overrides)
    # TODO: Get si_class from class name inside HedgehogAgentInterface.
    si_class = get_strategic_idling_class(
        cast(StrategicIdlingParams, si_params).strategic_idling_class
    )
    hh_params = get_hedgehog_params_as_named_tuple(HedgehogHyperParams(), **overrides)

    ar_policy_class_name = cast(HedgehogHyperParams, hh_params).activity_rates_policy_class_name
    if ar_policy_class_name == 'BigStepLayeredPolicy':
        po_params = get_hedgehog_params_as_named_tuple(BigStepLayeredPolicyParams(), **overrides)
    else:
        po_params = get_hedgehog_params_as_named_tuple(BigStepPenaltyPolicyParams(), **overrides)

    dp_params = get_hedgehog_params_as_named_tuple(DemandPlanningParams(), **overrides)
    name = overrides.get('name', "BigStepHedgehogAgent")
    return ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name


def build_bs_hedgehog_agent(env: ControlledRandomWalk,
                            discount_factor: float,
                            hh_overrides: Dict[str, Any],
                            debug_info: bool = False, agent_seed: Optional[int] = None,
                            mpc_seed: Optional[int] = None) -> hh_int.HedgehogAgentInterface:
    """
    Sets up an instantiation of a Big Step Hedgehog agent.

    :param env: Environment the hedgehog agent will run in.
    :param discount_factor: Discount factor to future rewards/costs.
    :param hh_overrides: Dictionary of hedgehog parameter overrides.
    :param debug_info: Boolean flag that indicates whether printing useful debug info.
    :param agent_seed: Agent random seed.
    :param mpc_seed: MPC random seed.
    :return: A BigStepHedgehogAgent agent.
    """
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_hyperparams(**hh_overrides)
    return BigStepHedgehogAgent(
        env,
        discount_factor,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        po_params,
        si_class,
        dp_params,
        name,
        debug_info,
        agent_seed,
        mpc_seed
    )


def build_pf_stationary_hedgehog(
        env: ControlledRandomWalk,
        discount_factor: float,
        hh_overrides: Dict[str, Any],
        debug_info: bool = False, agent_seed: Optional[int] = None,
        mpc_seed: Optional[int] = None) -> hh_int.HedgehogAgentInterface:
    """
    Sets up an instantiation of a Pure Feedback with Stationary MPC Hedgehog agent.

    :param env: Environment the hedgehog agent will run in.
    :param discount_factor: Discount factor to future rewards/costs.
    :param hh_overrides: Dictionary of hedgehog parameter overrides.
    :param debug_info: Boolean flag that indicates whether printing useful debug info.
    :param agent_seed: Agent random seed.
    :param mpc_seed: MPC random seed.
    :return: A PureFeedbackStationaryHedgehogAgent agent.
    """
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_hyperparams(**hh_overrides)
    return PureFeedbackStationaryHedgehogAgent(
        env,
        discount_factor,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        po_params,
        si_class,
        dp_params,
        name,
        debug_info,
        agent_seed,
        mpc_seed
    )


def build_pf_mip_hedgehog(env: ControlledRandomWalk,
                          discount_factor: float,
                          hh_overrides: Dict[str, Any],
                          debug_info: bool = False, agent_seed: Optional[int] = None,
                          mpc_seed: Optional[int] = None) -> hh_int.HedgehogAgentInterface:
    """
    Sets up an instantiation of a Pure Feedback MIP Hedgehog agent.

    :param env: Environment the hedgehog agent will run in.
    :param discount_factor: Discount factor to future rewards/costs.
    :param hh_overrides: Dictionary of hedgehog parameter overrides.
    :param debug_info: Boolean flag that indicates whether printing useful debug info.
    :param agent_seed: Agent random seed.
    :param mpc_seed: MPC random seed.
    :return: A PureFeedbackMIPHedgehogAgent agent.
    """
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_hyperparams(**hh_overrides)
    return PureFeedbackMIPHedgehogAgent(
        env,
        discount_factor,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        po_params,
        si_class,
        dp_params,
        name,
        debug_info,
        agent_seed,
        mpc_seed
    )


def get_maxweight_params(**overrides):
    """
    Get parameters for MaxWeight agent. If some parameter is not specified in overrides, then it
    takes a default value.

    :param overrides: Parameters coming from JSON file.
    :return: (weight_per_buffer, name)
        - weight_per_buffer: List whose entries weight the difference pressure at each buffer
            that MaxWeight aims to balance. For the case where this is given by the cost per buffer
            given by the environment, this is passed as a string: 'cost_per_buffer'.
        - name: Agent name used also as name for the folder where results are stored.
    """
    weight_per_buffer = overrides.get('weight_per_buffer', None)
    name = overrides.get('name', "MaxWeightAgent")
    return weight_per_buffer, name


def build_maxweight_agent(env: ControlledRandomWalk,
                          mw_overrides: Dict[str, types.StateSpace],
                          agent_seed: Optional[int] = None,
                          mpc_seed: Optional[int] = None) -> MaxWeightAgent:
    """
    Sets up an instantiation of a MaxWeight agent with elements of diagonal matrix given by
    'weight_per_buffer' if provided in 'overrides' dictionary of parameter overrides.

    :param env: Environment the hedgehog agent will run in.
    :param mw_overrides: Dictionary of MaxWeight parameter overrides.
    :param agent_seed: Agent random seed.
    :param mpc_seed: MPC random seed.
    :return: A MaxWeight agent.
    """
    weight_per_buffer, name = get_maxweight_params(**mw_overrides)
    return MaxWeightAgent(env, weight_per_buffer, name, agent_seed, mpc_seed)


def load_rl_agent(
        env: ControlledRandomWalk,
        rl_algorithm: str,
        load_path: str,
        discount_factor: float = 0.99,
        agent_params: Optional[Dict[str, Any]] = None) -> RLSimulationAgent:
    """
    Instantiates an RL agent in the RLSimulationAgent interface for compatibility and loads the
    weights from training into it.

    :param env: The controlled random walk environment for which the agent is required.
    :param rl_algorithm: The name of the RL algorithm used to train the agent.
    :param load_path: Path to a directory where TensorFlow checkpoints have been saved (i.e. where
        the model's weights are saved).
    :param discount_factor: A scalar discount factor to pass to the agent.
    :param agent_params: A dictionary of possible overrides for the default TF-Agents agent set up.
    :return: An RL agent initialised with saved weights ready for evaluation.
    """
    # Lazy import of TensorFlow as if no RL agent is run then it isn't needed.
    import tensorflow as tf

    # Attain a TensorFlow compatible version of the environment.
    # We need a TensorFlow environment to initialise the agent correctly.
    # First determine whether or not to normalise observations, PPO has its own normalisation so we
    # only normalise for reinforce agents or PPO agents where normalisation is turned off.
    normalise_obs = rl_algorithm == 'reinforce' or \
                    (rl_algorithm == 'ppo' and not agent_params.get('normalize_observations', True))
    tf_env, _ = rl_env.rl_env_from_snc_env(
        env,
        discount_factor,
        normalise_observations=normalise_obs
    )

    # Set up an enumeration of functions which build agents to allow for extending to new agents.
    # Pick out the correct RL agent from those we have implemented.
    if rl_algorithm.lower() == 'reinforce':
        agent = create_reinforce_agent(tf_env, gamma=discount_factor, agent_params=agent_params)
    elif rl_algorithm.lower() == 'ppo':
        agent = create_ppo_agent(tf_env, gamma=discount_factor, agent_params=agent_params)
    else:
        raise NotImplementedError(
            "An agent using the RL algorithm requested is not yet implemented")

    # Initialise the agent and load in parameters from the most recent save.
    # Note that this can be adjusted to load in weights from any point in training (so long as they
    # have been saved).
    agent.initialize()
    restorer = tf.train.Checkpoint(agent=agent)
    restore_manager = tf.train.CheckpointManager(restorer, directory=load_path, max_to_keep=20)
    restorer.listed = agent.trainable_variables
    restoration = restorer.restore(restore_manager.latest_checkpoint)
    restoration.run_restore_ops()
    # Check that the weights have been loaded and that the model from which the weights were saved
    # matches the model which they are being loaded into.
    restoration.assert_nontrivial_match()
    restoration.assert_existing_objects_matched()

    # We name the agent in line with the checkpoint used to restore the weights. This aids in
    # identifying which experiment run is being looked at from log files.
    agent_name = f"RLSimulationAgent - {restore_manager.latest_checkpoint}"

    # Finally wrap the agent for compatibility with the SNC simulator.
    simulation_agent = RLSimulationAgent(env,
                                         agent,
                                         normalise_obs,
                                         name=agent_name)
    return simulation_agent


def get_agent(agent_name: str, env: ControlledRandomWalk, **kwargs: Any) \
        -> Union[hh_int.HedgehogAgentInterface, AgentInterface]:
    """
    Gets an agent to run from its name, an environment and some agent parameters.

    :param agent_name: The name of the agent to attain.
    :param env: The environment in which the agent will run.
    :param kwargs: Any additional arguments the agent constructor will require
    :return: An instantiated agent.
    """
    if agent_name not in AGENT_CONSTRUCTORS:
        raise NotImplementedError(f'Requested agent "{agent_name}" not implemented.')
    if agent_name in ['bs_hedgehog', 'pf_stationary_hedgehog', 'pf_mip_hedgehog']:
        return AGENT_CONSTRUCTORS[agent_name](env,
                                              discount_factor=kwargs['discount_factor'],
                                              hh_overrides=kwargs['hh_overrides'],
                                              debug_info=kwargs['debug_info'],
                                              agent_seed=kwargs['agent_seed'],
                                              mpc_seed=kwargs['mpc_seed'])
    elif agent_name == 'distribution_with_rebalancing_heuristic':
        assert env.name == 'three_warehouses_simplified'
        assert 'safety_stocks' in kwargs, f'safety_stocks parameter of {agent_name} not provided.'
        return AGENT_CONSTRUCTORS[agent_name](env, safety_stocks=kwargs['safety_stocks'])
    elif agent_name in ['reinforce', 'ppo']:
        assert 'rl_checkpoint' in kwargs
        return AGENT_CONSTRUCTORS[agent_name](env, agent_name, kwargs['rl_checkpoint'],
                                              kwargs['discount_factor'], kwargs)
    elif agent_name == 'maxweight' or agent_name == 'scheduling_maxweight':
        return build_maxweight_agent(env,
                                     kwargs['overrides'],
                                     kwargs['agent_seed'],
                                     kwargs['mpc_seed'])
    else:
        return AGENT_CONSTRUCTORS[agent_name](env, kwargs['agent_seed'])


def get_all_agent_names(env_name: str, with_rl_agent: bool = False) -> List[str]:
    """
    Attains a list of all agents (their names) that can be run in an environment.
    :param env_name: The name of the environment to attain a list of possible agents for.
    :param with_rl_agent: Boolean determining whether to include RL agents or not.
    :return: A list of the names of agents that may be run in the environment.
    """
    # Attain the list of all agents.
    agent_list = list(AGENT_CONSTRUCTORS.keys())
    # Remove all unsuitable agents as appropriate.
    if env_name != 'three_warehouses_simplified':
        agent_list.remove('distribution_with_rebalancing_heuristic')
    if not with_rl_agent:
        agent_list.remove('reinforce')
        agent_list.remove('ppo')
    return agent_list


HEDGEHOG_AGENTS = ['bs_hedgehog', 'pf_stationary_hedgehog', 'pf_mip_hedgehog']

AGENT_CONSTRUCTORS: Dict[str, Callable] = {
    'bs_hedgehog': build_bs_hedgehog_agent,
    'pf_mip_hedgehog': build_pf_mip_hedgehog,
    'pf_stationary_hedgehog': build_pf_stationary_hedgehog,
    'distribution_with_rebalancing_heuristic': DistributionWithRebalancingLocalPriorityAgent,
    'scheduling_maxweight': SchedulingMaxWeightAgent,
    'maxweight': build_maxweight_agent,
    'priority_state': priority.PriorityState,
    'priority_cost': priority.PriorityCost,
    'priority_rate': priority.PriorityRate,
    'priority_state_cost': priority.PriorityStateCost,
    'priority_state_rate': priority.PriorityStateRate,
    'priority_cost_rate': priority.PriorityCostRate,
    'priority_state_cost_rate': priority.PriorityStateCostRate,
    'random_nonidling_agent': RandomNonIdlingAgent,
    'reinforce': load_rl_agent,
    'ppo': load_rl_agent
}

NUM_RL_AGENTS = 2
