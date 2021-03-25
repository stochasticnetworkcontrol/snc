import pytest
import numpy as np
import tensorflow as tf
from copy import deepcopy

from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs.tensor_spec import BoundedTensorSpec

from src import snc as snc
from src.snc.environments import rl_env_from_snc_env
from src.snc import create_reinforce_agent
from src.snc.environments import load_scenario


@pytest.mark.parametrize(
    'env_name,expected_action_spec_shape', [('single_server_queue', tf.TensorShape((1, 2)))]
)
def test_reinforce_agent_init(env_name, expected_action_spec_shape):
    """
    Tests agent set up and initialisation.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(
        load_scenario(env_name, job_gen_seed=10, override_env_params={'max_episode_length': 25})[1],
        discount_factor=0.99
    )

    # Instantiate and initialise a REINFORCE agent for the environment.
    reinforce_agent = create_reinforce_agent(tf_env)
    reinforce_agent.initialize()
    # Validate initialisation by checking some properties of the initalised agent.
    assert isinstance(reinforce_agent.action_spec, BoundedTensorSpec)
    assert reinforce_agent.action_spec.shape == expected_action_spec_shape
    assert reinforce_agent.name == "reinforce_agent"
    assert reinforce_agent.time_step_spec == tf_env.time_step_spec()


def test_reinforce_agent_init_with_multiple_resource_sets():
    """
    Tests agent set up and initialisation with multiple action subspaces (multiple resource sets).
    """
    # Set the environment name for this case as the asserts are difficult to make as variables.
    env_name = 'double_reentrant_line_shared_res_homogeneous_cost'

    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(
        load_scenario(env_name, job_gen_seed=10, override_env_params={'max_episode_length': 25})[1],
        discount_factor=0.99
    )

    # Instantiate and initialise a REINFORCE agent for the environment.
    reinforce_agent = create_reinforce_agent(tf_env)
    reinforce_agent.initialize()
    # Validate initialisation by checking some properties of the initalised agent.
    assert isinstance(reinforce_agent.action_spec, tuple)
    assert len(reinforce_agent.action_spec) == 2
    assert isinstance(reinforce_agent.action_spec[0], BoundedTensorSpec)
    assert isinstance(reinforce_agent.action_spec[1], BoundedTensorSpec)
    assert reinforce_agent.action_spec[0].shape == tf.TensorShape((1, 3))
    assert reinforce_agent.action_spec[1].shape == tf.TensorShape((1, 3))
    assert reinforce_agent.name == "reinforce_agent"
    assert reinforce_agent.time_step_spec == tf_env.time_step_spec()


# Parameterise with environments which cover the cases of a single resource set and multiple
# resource sets.
@pytest.mark.parametrize(
    'env_name',
    ['single_server_queue', 'double_reentrant_line_shared_res_homogeneous_cost']
)
def test_reinforce_agent_play(env_name):
    """
    Extension of the agent set up and initialisation test to include playing episodes.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(
        load_scenario(env_name, job_gen_seed=10, override_env_params={'max_episode_length': 25})[1],
        discount_factor=0.99
    )

    # Instantiate and initialise a REINFORCE agent.
    reinforce_agent = create_reinforce_agent(tf_env)
    reinforce_agent.initialize()

    # Reset the environment
    tf_env.reset()
    # Play 5 time steps in the environment.
    for _ in range(5):
        # Since we do not have the state stored at this point we capture it from the environment
        # fresh each time step as a TimeStep object (a named tuple).
        time_step = tf_env.current_time_step()
        # Attain our agent's action.
        action_step = reinforce_agent.collect_policy.action(time_step)
        if isinstance(action_step.action, tuple):
            action = tf.concat(action_step.action, axis=-1)
        else:
            action = action_step.action

        # Ensure that the action is binary as expected.
        assert snc.is_binary(action)

        # Play the action out in the environment.
        tf_env.step(action_step.action)


# Parameterise with environments which cover the cases of a single resource set and multiple
# resource sets.
@pytest.mark.parametrize(
    'env_name',
    ['single_server_queue', 'double_reentrant_line_shared_res_homogeneous_cost']
)
def test_reinforce_agent_learning(env_name):
    """
    Extension of the test for an agent playing in the environment to include training.
    Note: This does not test that training improves the policy. It simply tests that the training
    loop runs effectively.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(
        load_scenario(env_name, job_gen_seed=10, override_env_params={'max_episode_length': 25})[1],
        discount_factor=0.99
    )

    # Set up a training step counter.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # Instantiate a REINFORCE agent
    reinforce_agent = create_reinforce_agent(tf_env, training_step_counter=global_step)

    # Instantiate a replay buffer.
    replay_buffer = TFUniformReplayBuffer(
        data_spec=reinforce_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000)

    # Initialise the action network weights etc.
    reinforce_agent.initialize()

    # Use a driver to handle data collection for the agent. This handles a lot of the backend
    # TensorFlow set up and solves previous errors with episodes of differing lengths.
    collect_driver = DynamicEpisodeDriver(
        tf_env,
        reinforce_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=2)

    # Get the initial states of the agent and environment before training.
    time_step = tf_env.reset()
    policy_state = reinforce_agent.collect_policy.get_initial_state(tf_env.batch_size)

    # Take a copy of the variables in order to ensure that training does lead to parameter changes.
    initial_vars = deepcopy(reinforce_agent.trainable_variables)
    assert len(initial_vars) > 0, "Agent has no trainable variables."

    # Set up a minimal training loop to simply test training mechanics work.
    for _ in range(5):
        # Collect experience.
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state
        )
        # Now the replay buffer should have data in it so we can collect the data and train the
        # agent.
        experience = replay_buffer.gather_all()
        reinforce_agent.train(experience)
        # Clear the replay buffer and return to play.
        replay_buffer.clear()

    # Check that training has had some effect
    for v1, v2 in zip(initial_vars, reinforce_agent.trainable_variables):
        assert not np.allclose(v1.numpy(), v2.numpy())