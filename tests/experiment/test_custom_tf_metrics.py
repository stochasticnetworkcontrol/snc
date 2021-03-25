import sys
import copy
import numpy as np
import tensorflow as tf
from collections import namedtuple

from tf_agents.trajectories.trajectory import _create_trajectory as create_trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.policy_step import PolicyStep

from experiment.rl.custom_tf_metrics import \
    EpisodicConditionalActivityTracker1D, EvalPerStepLogger, ActionProbabilityMetric

MockDistributionReturn = namedtuple("MockDistributionReturn", "action")


class MockDistribution:
    def __init__(self, action_probabilities):
        self.action_probs = action_probabilities

    def probs_parameter(self):
        return tf.convert_to_tensor(self.action_probs[None, None])


class MockActorPolicy:
    def __init__(self, action_probabilities: np.array):
        self._t = tf.Variable(0)
        self._action_probs = tf.convert_to_tensor(action_probabilities)

    def distribution(self, time_step: ts.TimeStep):
        rtn_value = MockDistributionReturn(MockDistribution(
            self._action_probs[self._t]
        ))
        self._t.assign_add(1)
        return rtn_value


def test_episodic_conditional_activity_tracker_single_ep():
    """
    Tests the conditional activity tracker for a single episode as used in the single server queue
    experiment.
    """
    # Set the random seed for reproducibility.
    np.random.seed(72)

    # Define some example conditions. Taken from the single server queue experiment.
    def activity_tracker_filter_condition(traj):
        return tf.reduce_any(traj.observation > 0, axis=-1)

    def activity_tracker_activity_condition(traj):
        return traj.action[..., 1] == 1

    # Instantiate the required metric.
    metric = EpisodicConditionalActivityTracker1D(
        filter_condition=activity_tracker_filter_condition,
        activity_condition=activity_tracker_activity_condition,
        name="active_when_non_empty", dtype=tf.float32, prefix="TEST")

    # Assert that the conditions are recorded correctly.
    assert metric.filter_condition == activity_tracker_filter_condition
    assert metric.activity_condition == activity_tracker_activity_condition

    # Set up an episode of trajectories as will be parsed during training and evaluation.
    # This set up is for an episode of 25 time steps and follows the specification from the Single
    # Server Queue experiment.
    episode_length = 25
    observations = [np.random.randint(10, size=(1, 1)) for _ in range(episode_length)]
    # Actions are random one-hot vectors of shape (1, 2).
    actions = [np.eye(2)[[np.random.randint(2)]] for _ in range(episode_length)]
    policy_info = ()
    # Assume -1 cost per item in the single buffer.
    rewards = [-1 * obs[0] for obs in observations]
    discount = tf.convert_to_tensor(np.array([0.99]))
    # Step types key: 0 is initial step, 1 is mid and 2 is terminal state.
    step_types = [ts.StepType(min(i, 1)) for i in range(episode_length - 1)]
    step_types.append(ts.StepType(2))
    # A scope for tensor creation as required by `create_trajectory`.
    scope = "test_trajectory"
    for i in range(episode_length):
        # Use a helper function from tf_agents to form a `Trajectory` object.
        trajectory = create_trajectory(
            observation=tf.convert_to_tensor(observations[i]),
            action=tf.convert_to_tensor(actions[i]),
            policy_info=policy_info,
            reward=tf.convert_to_tensor(rewards[i]),
            discount=discount,
            step_type=step_types[i],
            next_step_type=step_types[(i + 1) % episode_length],
            name_scope=scope
        )
        # Call the metric with the current data.
        metric(trajectory)
    # Calculate the expected metric values from the raw input data as the ground truth for the
    # TensorFlow metric calculations.
    num_qualifying_states = np.sum(np.array(observations) > 0)
    active_with_buffer_count = np.sum(
        (np.array(actions)[..., 1].flatten() * np.array(observations).flatten()) > 0
    )
    # Test the computation of the components of the ratio required as well as the overall ratio
    # calculation.
    assert metric._qualifying_timesteps_buffer.data[0].numpy() == num_qualifying_states
    assert metric._activity_buffer.data[0].numpy() == active_with_buffer_count
    assert metric.result() == active_with_buffer_count / num_qualifying_states


def test_episodic_conditional_activity_tracker_multiple_eps():
    """
    Tests the conditional activity tracker for two episodes following the set up of the single
    server queue experiment.
    """
    # Set the random seed for reproducibility.
    np.random.seed(72)

    # Define some example conditions. Taken from the single server queue experiment.
    def activity_tracker_filter_condition(traj):
        return tf.reduce_any(traj.observation > 0, axis=-1)

    def activity_tracker_activity_condition(traj):
        return traj.action[..., 1] == 1

    # Instantiate the required metric.
    metric = EpisodicConditionalActivityTracker1D(
        filter_condition=activity_tracker_filter_condition,
        activity_condition=activity_tracker_activity_condition,
        name="active_when_non_empty", dtype=tf.float32, prefix="TEST")

    # Assert that the conditions are recorded correctly.
    assert metric.filter_condition == activity_tracker_filter_condition
    assert metric.activity_condition == activity_tracker_activity_condition

    # Set up two episodes of trajectories as will be parsed during training and evaluation.
    # This set up is for two episodes of 25 time steps and follows the specification from the Single
    # Server Queue experiment.
    episode_length = 25
    num_episodes = 2
    num_steps = episode_length * num_episodes
    observations = [np.random.randint(10, size=(1, 1)) for _ in range(num_steps)]
    # Actions are random one-hot vectors of shape (1, 2).
    actions = [np.eye(2)[[np.random.randint(2)]] for _ in range(num_steps)]
    policy_info = ()
    # Assume -1 cost per item in the single buffer.
    rewards = [-1 * obs[0] for obs in observations]
    discount = tf.convert_to_tensor(np.array([0.99]))
    # Step types key: 0 is initial step, 1 is mid and 2 is terminal state.
    step_types = [ts.StepType(min(i, 1)) for i in range(episode_length - 1)]
    step_types.append(ts.StepType(2))
    # Extend the step_types list to cover two episodes.
    step_types = step_types * num_episodes
    # A scope for tensor creation as required by `create_trajectory`.
    scope = "test_trajectory"
    for i in range(num_steps):
        # Use a helper function from tf_agents to form a `Trajectory` object.
        trajectory = create_trajectory(
            observation=tf.convert_to_tensor(observations[i]),
            action=tf.convert_to_tensor(actions[i]),
            policy_info=policy_info,
            reward=tf.convert_to_tensor(rewards[i]),
            discount=discount,
            step_type=step_types[i],
            next_step_type=step_types[(i + 1) % episode_length],
            name_scope=scope
        )
        # Call the metric with the current data.
        metric(trajectory)
    # Calculate the expected metric values from the raw input data as the ground truth for the
    # TensorFlow metric calculations.
    # We reshape the arrays to attain metrics per episode which will later be averaged for the final
    # result.
    num_qualifying_states = np.sum(np.array(observations).reshape((num_episodes, -1)) > 0, axis=1)
    active_with_buffer_count = np.sum(
        ((np.array(actions)[..., 1].flatten() * np.array(observations).flatten()) > 0)
        .reshape((num_episodes, -1)),
        axis=1
    )
    # Test the computation of the components of the ratio required as well as the overall ratio
    # calculation.
    assert np.all(metric._qualifying_timesteps_buffer.data.numpy() == num_qualifying_states)
    assert np.all(metric._activity_buffer.data.numpy() == active_with_buffer_count)
    assert np.isclose(metric.result(), (active_with_buffer_count / num_qualifying_states).mean())


def test_eval_logger():
    """
    Tests the per step logging mediated through a custom TensorFlow metric.

    Note that due to the fact that TensorFlow places logging in a graph built through C++ which is
    only triggered when tensors are evaluated it is very difficult to capture the logging message
    even through using mocked output streams. Therefore, the test checks the attributes that can be
    tested and logs expected logging values for by-eye comparison. This is a fairly simple case
    since the logging code is simple but the test is in this sense in complete.
    """
    # Set up the logger using default parameters.
    logger = EvalPerStepLogger()
    # Test that the time step counter is initialised to zero.
    assert logger._t == 0

    # Build one time step's worth of data to be logged.
    observation = tf.convert_to_tensor(np.random.randint(10, size=(1, 1)), dtype=tf.float32)
    action = tf.convert_to_tensor(np.eye(2)[np.random.randint(2)])
    reward = -1 * observation
    discount = tf.convert_to_tensor(np.array([0.99]))

    # The logger takes in a tuple of (TimeStep, PolicyStep, TimeStep) the second time step
    # represents the next period and is not used so we simply pass a copy of the original time step.
    time_step = ts.TimeStep(ts.StepType(1), reward, discount, observation)
    policy_step = PolicyStep(action, state=(), info=())
    next_time_step = copy.deepcopy(time_step)
    # Collect the data in a tuple as required by the logger.
    time_step_data = (time_step, policy_step, next_time_step)

    # Print the expected logging term for comparison by eye.
    tf.print(
        "\nExpected Values\nStep: ", 0, "\t",
        "State: ", observation, "\t",
        "Action: ", action,
        end="\n",
        output_stream=sys.stdout
    )
    # Run the logging for a single time step.
    logger(time_step_data)
    # Check that the time step counter has incremented.
    assert logger._t == 1


def test_policy_probability_logging():
    np.random.seed(72)
    # Set up an episode of trajectories as will be parsed during training and evaluation.
    # This set up is for an episode of 25 time steps and follows the specification from the Single
    # Server Queue experiment.
    episode_length = 25
    num_actions = 2

    # Set up a mock policy to return deterministic values from a list.
    action_probabilities = np.random.dirichlet(alpha=(1,) * num_actions, size=(episode_length,))\
        .astype(np.float32)
    mock_policy = MockActorPolicy(action_probabilities)

    # Set an index for the metric to track and then set up the metric.
    action_index = 1
    metric = ActionProbabilityMetric(mock_policy, (action_index,))
    assert metric.action_indices == (action_index,)

    # Observations will be ignored but needed to complete the trajectory.
    observations = [np.random.randint(10, size=(1, 1)) for _ in range(episode_length)]

    # Actions are random one-hot vectors of shape (1, 2).
    actions = [np.eye(num_actions)[[np.random.randint(num_actions)]] for _ in range(episode_length)]

    # Assume -1 cost per item in the single buffer.
    rewards = [-1 * obs[0] for obs in observations]
    discount = tf.convert_to_tensor(np.array([0.99]))

    # Step types key: 0 is initial step, 1 is mid and 2 is terminal state.
    step_types = [ts.StepType(min(i, 1)) for i in range(episode_length - 1)]
    step_types.append(ts.StepType(2))

    # A scope for tensor creation as required by `create_trajectory`.
    scope = "test_trajectory"

    # Run the episode.
    for i in range(episode_length):
        # Use a helper function from tf_agents to form a `Trajectory` object.
        trajectory = create_trajectory(
            observation=tf.convert_to_tensor(observations[i]),
            action=tf.convert_to_tensor(actions[i]),
            policy_info=(),
            reward=tf.convert_to_tensor(rewards[i]),
            discount=discount,
            step_type=step_types[i],
            next_step_type=step_types[(i + 1) % episode_length],
            name_scope=scope
        )
        # Call the metric with the current data.
        metric(trajectory)

    # The average policy probability over an episode ignores the final time step because there is
    # no subsequent state so the action is never actually executed and should thus be ignored from
    # analysis.
    expected_average = action_probabilities[:-1, action_index].mean()
    # Test that there is an entry for each episode and that the average is calculated correctly.
    assert metric._buffer.length == 1
    assert np.isclose(expected_average, metric.result())


def test_policy_probability_logging_multiple_eps():
    np.random.seed(72)

    # Set up an episode of trajectories as will be parsed during training and evaluation.
    # This set up is for an episodes of 25 time steps.
    episode_length = 25
    num_episodes = 2
    num_actions = 3
    num_steps = episode_length * num_episodes

    # Set up a mock policy to return deterministic values from a list.
    action_probabilities = np.random.dirichlet(alpha=(1,) * num_actions, size=(num_steps,)) \
        .astype(np.float32)
    mock_policy = MockActorPolicy(action_probabilities)

    # Set an index for the metric to track and then set up the metric.
    action_index = 0
    metric = ActionProbabilityMetric(mock_policy, (action_index,))
    assert metric.action_indices == (action_index,)

    # Observations will be ignored but needed to complete the trajectory.
    observations = [np.random.randint(10, size=(1, 1)) for _ in range(num_steps)]

    # Actions are random one-hot vectors of shape (1, 2).
    actions = [np.eye(num_actions)[[np.random.randint(num_actions)]] for _ in range(num_steps)]

    # Assume -1 cost per item in the single buffer.
    rewards = [-1 * obs[0] for obs in observations]
    discount = tf.convert_to_tensor(np.array([0.99]))

    # Step types key: 0 is initial step, 1 is mid and 2 is terminal state.
    step_types = [ts.StepType(min(i, 1)) for i in range(episode_length - 1)]
    step_types.append(ts.StepType(2))

    # Extend step types list to cover all episodes.
    step_types *= num_episodes

    # A scope for tensor creation as required by `create_trajectory`.
    scope = "test_trajectory"

    # Run the episode.
    for i in range(num_steps):
        # Use a helper function from tf_agents to form a `Trajectory` object.
        trajectory = create_trajectory(
            observation=tf.convert_to_tensor(observations[i]),
            action=tf.convert_to_tensor(actions[i]),
            policy_info=(),
            reward=tf.convert_to_tensor(rewards[i]),
            discount=discount,
            step_type=step_types[i],
            next_step_type=step_types[(i + 1) % episode_length],
            name_scope=scope
        )
        # Call the metric with the current data.
        metric(trajectory)

    # The average policy probability over an episode ignores the final time step because there is
    # no subsequent state so the action is never actually executed and should thus be ignored from
    # analysis.
    expected_average = (action_probabilities[:, action_index].sum() -
                        action_probabilities[episode_length-1, action_index] -
                        action_probabilities[-1, action_index]) / (len(action_probabilities) - 2)
    # Test that there is an entry for each episode and that the average is calculated correctly.
    assert metric._buffer.length == num_episodes
    assert np.isclose(expected_average, metric.result())
