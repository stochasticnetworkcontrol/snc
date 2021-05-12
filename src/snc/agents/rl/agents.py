from typing import Dict, Optional, Any


import tensorflow as tf
import tf_agents
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.agents.ppo.ppo_agent import PPOAgent

from snc.agents.rl.multi_headed_softmax_policy import MultiHeadedCategoricalActionNetwork


def create_reinforce_agent(
        env: TFPyEnvironment,
        gamma: float = 0.99,
        agent_name: str = 'reinforce_agent',
        debug: bool = False,
        training_step_counter: Optional[Any] = None,
        agent_params: Optional[Dict[str, Any]] = None) -> ReinforceAgent:
    """
    Function for creating a REINFORCE agent in line with the TensorFlow Agents implementation.
    This function builds an action network and uses this to instantiate the agent which is returned.

    :param env: TensorFlow Environment implementing the ControlledRandomWalk.
    :param gamma: Discount factor.
    :param agent_name: Name for the agent to aid in identifying TensorFlow variables etc. when
        debugging.
    :param debug: Flag which toggles debugging in the REINFORCE agent.
    :param training_step_counter: An optional counter to increment every time the train op of the
        agent is run. If None if provided it defaults to the global_step.
    :param agent_params: A dictionary of possible overrides for the default TF-Agents agent set up.
    :return: An instance of TensorFlow Agents REINFORCE agent.
    """
    # Process the action specification to attain the dimensions of the action subspaces to ensure
    # that in the case that there is only one resource set (and therefore only one action subspace)
    # the tuple of action specifications of length one is replaced by a single action specification.
    # This is to align with the fact that the actor network is implemented to return a tuple of
    # (OneHotCategorical) distributions (one for each resource set) where there are multiple action
    # subspaces and a single distribution (tfp.distributions.OneHotCategorical) otherwise.
    # First attain the action spec.
    action_spec = env.action_spec()

    # Extract the shape of the subspaces from the action specification tuple.
    # Action spaces are defined with shape (1, num_actions_for_resource_set) so take the -1th entry.
    action_subspace_dimensions = tuple(int(subspace.shape[-1]) for subspace in action_spec)

    # Then test if there is only one action subspace.
    if len(action_spec) == 1:
        # Pull out the only action spec.
        action_spec = action_spec[0]

    if agent_params is None:
        agent_params = dict()

    # Set up the action network. See `multi_headed_softmax_policy.py` for details.
    actor_network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec=env.observation_spec(),
        output_tensor_spec=action_spec,
        action_subspace_dimensions=action_subspace_dimensions,
        hidden_units=agent_params.get('hidden_units', (64,))
    )
    # Set up the REINFORCE agent in line with standard tf_agents.
    agent = ReinforceAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_network,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        value_network=agent_params.get('value_network', None),
        value_estimation_loss_coef=agent_params.get('value_estimation_loss_coef', 0.2),
        advantage_fn=agent_params.get('advantage_fn', None),
        use_advantage_loss=agent_params.get('use_advantage_loss', True),
        gamma=gamma,
        normalize_returns=agent_params.get('normalize_returns', True),
        gradient_clipping=agent_params.get('gradient_clipping', None),
        debug_summaries=debug,
        summarize_grads_and_vars=debug,
        entropy_regularization=agent_params.get('entropy_regularization', None),
        train_step_counter=training_step_counter,
        name=agent_name
    )

    return agent


def create_ppo_agent(
        env: TFPyEnvironment,
        num_epochs: int = 10,
        gamma: float = 0.99,
        agent_name: str = 'PPO_Agent',
        debug: bool = False,
        training_step_counter: Optional[Any] = None,
        agent_params: Optional[Dict[str, Any]] = None) -> PPOAgent:
    """
    Function for creating a Proximal Policy Optimisation agent in line with the TensorFlow Agents
    implementation.
    This function builds an action network and uses this to instantiate the agent which is returned.

    :param env: TensorFlow Environment implementing the ControlledRandomWalk.
    :param num_epochs: Number of epochs for computing policy updates.
    :param gamma: Discount factor.
    :param agent_name: Name for the agent to aid in identifying TensorFlow variables etc. when
        debugging.
    :param debug: Flag which toggles debugging in the PPO agent.
    :param training_step_counter: An optional counter to increment every time the train op of the
        agent is run. If None if provided it defaults to the global_step.
    :param agent_params: A dictionary of possible overrides for the default TF-Agents agent set up.
    :return: An instance of TensorFlow Agents PPO agent.
    """
    # Process the action specification to attain the dimensions of the action subspaces to ensure
    # that in the case that there is only one resource set (and therefore only one action subspace)
    # the tuple of action specifications of length one is replaced by a single action specification.
    # This is to align with the fact that the actor network is implemented to return a tuple of
    # (OneHotCategorical) distributions (one for each resource set) where there are multiple action
    # subspaces and a single distribution (tfp.distributions.OneHotCategorical) otherwise.
    # First attain the action spec.
    action_spec = env.action_spec()

    # Extract the shape of the subspaces from the action specification tuple.
    # Action spaces are defined with shape (1, num_actions_for_resource_set) so take the -1th entry.
    action_subspace_dimensions = tuple(int(subspace.shape[-1]) for subspace in action_spec)

    # Then test if there is only one action subspace.
    if len(action_spec) == 1:
        # Pull out the only action spec.
        action_spec = action_spec[0]

    if agent_params is None:
        agent_params = dict()

    # Set up the action network. See `multi_headed_softmax_policy.py` for details.
    actor_network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec=env.observation_spec(),
        output_tensor_spec=action_spec,
        action_subspace_dimensions=action_subspace_dimensions,
        hidden_units=agent_params.get('hidden_units', (64,))
    )

    # PPO Requires a value network, we set one up using the default tf_agents set up.
    value_network = tf_agents.networks.value_network.ValueNetwork(
        env.observation_spec(),
        fc_layer_params=agent_params.get('value_fc_layer_params', (128, 64)),
        activation_fn=agent_params.get('value_net_activation_fn', tf.nn.tanh)
    )

    # Set up the PPO agent in line with standard tf_agents.
    agent = PPOAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=action_spec,
        actor_net=actor_network,
        optimizer=tf.compat.v1.train.AdamOptimizer(agent_params.get('learning_rate', 0.001)),
        value_net=value_network,
        importance_ratio_clipping=agent_params.get('importance_ratio_clipping', 0.0),
        lambda_value=agent_params.get('lambda_value', 0.95),
        discount_factor=gamma,
        policy_l2_reg=agent_params.get('policy_l2_reg', 0.0),
        value_function_l2_reg=agent_params.get('value_function_l2_reg', 0.0),
        value_pred_loss_coef=agent_params.get('value_pred_loss_coef', 0.5),
        num_epochs=num_epochs,
        use_gae=agent_params.get('use_gae', False),
        use_td_lambda_return=agent_params.get('use_td_lambda_return', False),
        normalize_rewards=agent_params.get('normalise_rewards', True),
        reward_norm_clipping=agent_params.get('reward_norm_clipping', 10),
        kl_cutoff_factor=agent_params.get('kl_cutoff_factor', 2.0),
        kl_cutoff_coef=agent_params.get('kl_cutoff_coef', 1000),
        initial_adaptive_kl_beta=agent_params.get('initial_adaptive_kl_beta', 1.0),
        adaptive_kl_target=agent_params.get('adaptive_kl_target', 0.01),
        adaptive_kl_tolerance=agent_params.get('adaptive_kl_tolerance', 0.3),
        normalize_observations=agent_params.get('normalize_observations', True),
        gradient_clipping=agent_params.get('gradient_clipping', None),
        debug_summaries=debug,
        summarize_grads_and_vars=debug,
        check_numerics=agent_params.get('check_numerics', False),
        entropy_regularization=agent_params.get('entropy_regularization', 0.0),
        train_step_counter=training_step_counter,
        name=agent_name
    )

    return agent
