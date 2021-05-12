from mock import MagicMock

import json
import numpy as np

from tf_agents.agents.tf_agent import TFAgent
from tf_agents.policies import tf_policy
from tf_agents.trajectories.time_step import TimeStep, StepType

from snc.utils import snc_types

from snc.simulation.store_data.numpy_encoder import NumpyEncoder
from snc.environments.scenarios import load_scenario
from snc.agents.rl.agents import create_reinforce_agent, create_ppo_agent
from snc.agents.rl.snc_agent_interface_for_rl import RLSimulationAgent
from snc.environments.controlled_random_walk import ControlledRandomWalk
from snc.environments.rl_environment_wrapper import (
    RLControlledRandomWalk,
    rl_env_from_snc_env
)


def test_rl_simulation_agent_init():
    """
    Test the initialisation of an RL agent with an interface compatible with the SNC simulator.
    """
    # To instantiate an agent from tf_agents we need an RL environment which itself requires a
    # standard SNC environment. We therefore set up an SNC environment and then wrap it for the
    # TensorFlow agent. This TF environment is later deleted since it is no longer required and to
    # ensure that it is not used inadvertently.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    rl_agent = create_reinforce_agent(rl_env)
    rl_agent.initialize()
    del rl_env
    # Wrapping the agent for the SNC simulator using information from the environment and the agent.
    sim_agent = RLSimulationAgent(env, rl_agent, normalise_obs=True)

    # Test that the agent has all of the attributed we want and that they are of the right type.
    assert hasattr(sim_agent, "_rl_env") and isinstance(sim_agent._rl_env, RLControlledRandomWalk)
    assert hasattr(sim_agent, "_rl_agent") and isinstance(sim_agent._rl_agent, TFAgent)
    assert hasattr(sim_agent, "_policy") and isinstance(sim_agent._policy, tf_policy.TFPolicy)
    assert hasattr(sim_agent, "_is_eval_policy") and isinstance(sim_agent._is_eval_policy, bool)
    assert hasattr(sim_agent, "env") and isinstance(sim_agent.env, ControlledRandomWalk)
    assert hasattr(sim_agent, "buffer_processing_matrix") and isinstance(
        sim_agent.buffer_processing_matrix, snc_types.BufferMatrix)
    assert hasattr(sim_agent, "constituency_matrix") and isinstance(sim_agent.constituency_matrix,
                                                                    snc_types.ConstituencyMatrix)
    assert hasattr(sim_agent, "demand_rate") and isinstance(sim_agent.demand_rate, np.ndarray)
    assert hasattr(sim_agent, "list_boundary_constraint_matrices") and isinstance(
        sim_agent.list_boundary_constraint_matrices, list)
    assert hasattr(sim_agent, "name") and isinstance(sim_agent.name, str)


def test_rl_simulation_agent_action_mapping():
    """
    Tests that the RL Simulation Agent with the SNC interface is able to receive states and produce
    actions both of the expected type and form.
    """
    # Set up the agent as above
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    rl_agent = create_reinforce_agent(rl_env)
    rl_agent.initialize()
    del rl_env
    sim_agent = RLSimulationAgent(env, rl_agent, normalise_obs=True)

    # Attain a state and form an action.
    state = env.reset()
    action = sim_agent.map_state_to_actions(state)
    # Ensure that the action is as expected first with a formal assertion and then by passing it
    # to the environment.
    assert isinstance(action, snc_types.ActionProcess)
    env.step(action)


def test_rl_simulation_agent_serialisation():
    """
    Test the custom serialisation of the agent used when saving the state of the SNC simulator.
    The customised serialisation was required due to the inability to serialise TensorFlow objects.
    """
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    rl_agent = create_reinforce_agent(rl_env)
    rl_agent.initialize()
    del rl_env
    sim_agent = RLSimulationAgent(env, rl_agent, normalise_obs=True)

    # Attain the dictionary representation of the agent and test that all the attributes expected
    # are present.
    serialised_agent = sim_agent.to_serializable()
    assert all(attr in serialised_agent for attr in ["_rl_env", "_rl_agent", "_policy",
                                                     "_is_eval_policy", "env",
                                                     "buffer_processing_matrix",
                                                     "constituency_matrix", "demand_rate",
                                                     "list_boundary_constraint_matrices", "name"])
    # Ensure that the dictionary representation is compatible with the json module and the chosen
    # encoder.
    json_string = json.dumps(serialised_agent, cls=NumpyEncoder, indent=4, sort_keys=True)
    assert bool(json_string)


def test_rl_simulation_agent_string_representation():
    """
    Test miscellaneous additional features of the RLSimulationAgent.
    Currently tests:
        string representation (__str__)
    """
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    rl_agent = create_reinforce_agent(rl_env)
    rl_agent.initialize()
    del rl_env
    sim_agent = RLSimulationAgent(env, rl_agent, normalise_obs=True)
    # Ensure that the string representation of the agent contains the instance name at the end.
    assert str(sim_agent)[-len(sim_agent.name):] == sim_agent.name


def test_rl_simulation_agent_string_representation():
    """
    Tests that the string representation of the simulation agent is as expected.
    """
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    rl_agent = create_reinforce_agent(rl_env)
    rl_agent.initialize()
    del rl_env
    sim_agent = RLSimulationAgent(env, rl_agent, normalise_obs=True)
    # Ensure that the string representation of the agent contains the instance name at the end.
    assert str(sim_agent)[-len(sim_agent.name):] == sim_agent.name


def test_rl_simulation_agent_discount_factor_reinforce():
    """
    Tests that the discount factor is passed from a REINFORCE agent to an RLSimulationAgent
    correctly.
    """
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99)
    reinforce_agent = create_reinforce_agent(rl_env, gamma=0.97)
    reinforce_agent.initialize()
    del rl_env
    reinforce_sim_agent = RLSimulationAgent(env, reinforce_agent, normalise_obs=True)
    assert reinforce_sim_agent.discount_factor == 0.97


def test_rl_simulation_agent_discount_factor_ppo():
    """
    Tests that the discount factor is passed from a PPO agent to an RLSimulationAgent correctly.
    """
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99, normalise_observations=False)
    ppo_agent = create_ppo_agent(rl_env, gamma=0.90)
    ppo_agent.initialize()
    del rl_env
    ppo_sim_agent = RLSimulationAgent(env, ppo_agent, normalise_obs=False)
    assert ppo_sim_agent.discount_factor == 0.90


def test_rl_simulation_agent_normalise_obs_property():
    """Ensure that the _normalise_obs property of RLSimulationAgent is set correctly."""
    # Set up the agent as before.
    seed = 72
    env = load_scenario("single_server_queue", job_gen_seed=seed).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99, normalise_observations=False)
    ppo_agent = create_ppo_agent(rl_env, gamma=0.90)
    ppo_agent.initialize()
    del rl_env
    ppo_sim_agent = RLSimulationAgent(env, ppo_agent, normalise_obs=False)
    assert ppo_sim_agent._normalise_obs is False
    ppo_sim_agent = RLSimulationAgent(env, ppo_agent, normalise_obs=True)
    assert ppo_sim_agent._normalise_obs is True


def test_rl_simulation_agent_normalise_obs_usage_no_normalisation():
    """Ensure that the _normalise_obs property of RLSimulationAgent is used correctly."""
    # Set up the agent as before.
    seed = 72
    state = np.array([100, 100, 100, 100])
    env = load_scenario("klimov_model",
                        job_gen_seed=seed,
                        override_env_params={"initial_state": state}).env
    ppo_agent = MagicMock()
    ppo_agent.discount_factor = 0.99
    ppo_agent._gamma = 0.99
    policy = MagicMock()
    ppo_agent.collect_policy = policy

    ppo_sim_agent = RLSimulationAgent(env, ppo_agent, normalise_obs=False)
    ppo_sim_agent._rl_env.preprocess_action = MagicMock()
    ppo_sim_agent.map_state_to_actions(state)
    expected_timestep = TimeStep(
            step_type=StepType(0),
            reward=None,
            discount=0.99,
            observation=state.reshape(1, -1)
        )
    assert policy.action.call_count == 1
    call_timestep = policy.action.call_args[0][0]
    assert (call_timestep.observation == expected_timestep.observation).all()


def test_rl_simulation_agent_normalise_obs_usage_with_normalisation():
    """Ensure that the _normalise_obs property of RLSimulationAgent is used correctly."""
    # Set up the agent as before.
    seed = 72
    state = np.array([100, 100, 100, 100])
    env = load_scenario("klimov_model",
                        job_gen_seed=seed,
                        override_env_params={"initial_state": state}).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99, normalise_observations=True)
    ppo_agent = MagicMock()
    ppo_agent.discount_factor = 0.99
    ppo_agent._gamma = 0.99
    policy = MagicMock()
    ppo_agent.collect_policy = policy
    del rl_env
    ppo_sim_agent = RLSimulationAgent(env, ppo_agent, normalise_obs=True)
    ppo_sim_agent._rl_env.preprocess_action = MagicMock()
    ppo_sim_agent.map_state_to_actions(state)
    expected_timestep = TimeStep(
            step_type=StepType(0),
            reward=None,
            discount=0.99,
            observation=state.reshape(1, -1) / state.sum()
        )
    assert policy.action.call_count == 1
    call_timestep = policy.action.call_args[0][0]
    assert (call_timestep.observation == expected_timestep.observation).all()
