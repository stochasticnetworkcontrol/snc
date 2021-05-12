import sys

from typing import Callable, Tuple, IO, Union, Type

import tensorflow as tf
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.policies import tf_policy
from tf_agents.trajectories.time_step import TimeStep


class EpisodicConditionalActivityTracker1D(TFStepMetric):

    def __init__(self,
                 filter_condition: Callable,
                 activity_condition: Callable,
                 name: str,
                 prefix: str = "Metrics",
                 dtype: type = tf.int32,
                 batch_size: int = 1,
                 buffer_size: int = 10):
        """
        Custom TensorFlow metric which tracks activity in states which satisfy a certain condition.
        Used for example to see the behaviour of an agent when the buffers reach a certain level.

        :param filter_condition: The condition which determines which time steps to use for
            calculation. This defines the denominator.
        :param activity_condition: The condition which determines when the tracked activity occurs.
            This defined the numerator.
        :param name: Name of the metric as will be shown in TensorBoard.
        :param prefix: The name of the logging group to which this metric belongs.
        :param dtype: The data type of the metric and the components used for its calculation.
            Data type must be compatible with tf.reduce_sum (i.e. float or int).
        :param batch_size: The batch size of the environment.
        :param buffer_size: The length of the buffer used to store historical values of the metric.
        """
        # Initialise according to the parent class then add the inputs as attributes.
        super(EpisodicConditionalActivityTracker1D, self).__init__(name=name, prefix=prefix)
        self.filter_condition = filter_condition
        self.activity_condition = activity_condition
        self._dtype = dtype
        self._batch_size = batch_size
        # Build variables and storage to be used for storing and calculating the metrics.
        self._activity_buffer = TFDeque(buffer_size, dtype)
        self._qualifying_timesteps_buffer = TFDeque(buffer_size, dtype)
        # The activity accumulator becomes the numerator of the calculated rate/proportion.
        self._activity_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='ActivityAccumulator')
        # The number of valid time steps becomes the denominator of the calculated rate/proportion.
        self._num_valid_timesteps = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='EpLenAccumulator')

    @common.function(autograph=True)
    def call(self, trajectory: Trajectory) -> Trajectory:
        """
        Process the experience passed in to update the metric value (or the components required to
        calculate the final value).

        :param trajectory: Experience from the agent rolling out in the environment.
        :return: The unchanged input trajectory (as per the standard use of TensorFlow Metrics).
        """
        start_of_episode_indices = tf.squeeze(tf.where(trajectory.is_first()), axis=-1)
        mask = tf.ones(shape=(self._batch_size,), dtype=self._dtype)

        for idx in start_of_episode_indices:
            mask -= tf.eye(self._batch_size)[idx]

        # Reset the accumulators at the end of each episode.
        self._num_valid_timesteps.assign(self._num_valid_timesteps * mask)
        self._activity_accumulator.assign(self._activity_accumulator * mask)

        # Find the number of time steps satisfying the filter condition.
        # The reshape is to ensure compatibility with the variable below in the case of no batch
        # dimension.
        valid_timesteps = tf.reshape(
            tf.reduce_sum(
                tf.cast(self.filter_condition(trajectory), self._dtype),
                axis=-1),
            self._num_valid_timesteps.shape)

        # Track the number of time steps which meet the qualifying condition.
        self._num_valid_timesteps.assign_add(valid_timesteps, name="increment_valid_timesteps")

        # Update accumulator with activity counts where both the filtering and activity condition
        # are satisfied. Again the reshape is to ensure compatibility with the accumulator
        # variable in the case where there is no batch dimension.
        bool_values = tf.logical_and(self.filter_condition(trajectory),
                                     self.activity_condition(trajectory))
        to_add = tf.reshape(
            tf.reduce_sum(tf.cast(bool_values, self._dtype), axis=-1),
            self._activity_accumulator.shape)

        self._activity_accumulator.assign_add(to_add)

        # Add values to buffer at the end of the episode by first finding where the trajectories end
        # and then using the resulting indices to update the correct buffer locations.
        # At the same time build up a mask of values to use for resetting the accumulators.
        end_of_episode_indices = tf.squeeze(tf.where(trajectory.step_type == 2), axis=-1)

        for idx in end_of_episode_indices:
            self._activity_buffer.add(self._activity_accumulator[idx])
            self._qualifying_timesteps_buffer.add(self._num_valid_timesteps[idx])

        # Return the original trajectory data as is standard for TFStepMetrics.
        return trajectory

    def result(self) -> tf.Tensor:
        """
        Calculate the value of the metric from stored components.
        :return: The calculated metric value.
        """
        return tf.reduce_mean(self._activity_buffer.data / self._qualifying_timesteps_buffer.data)

    @common.function
    def reset(self) -> None:
        """Reset the metric calculation components."""
        pass
        self._activity_buffer.clear()
        self._qualifying_timesteps_buffer.clear()
        self._activity_accumulator.assign(tf.zeros_like(self._activity_accumulator))
        self._num_valid_timesteps.assign(tf.zeros_like(self._num_valid_timesteps))


class EvalPerStepLogger(TFStepMetric):

    def __init__(self, name: str = "EvalLogger", prefix: str = "Logging",
                 output_stream: Union[str, IO] = sys.stdout) -> None:
        """
        A logging tool designed to be run every time step (i.e. passed as a transition_observer).
        This will log the raw state and action at each time step through a trajectory.

        :param name: Name of the logger.
        :param prefix: The name of the group of metrics the logger belongs to.
        :param output_stream: A file out standard output stream to which logs will be written. If
            None is provided defaults to sys.stdout.
        """
        # Set up a time step counter as this information is not available from the environment
        # experience passed to call.
        self._t = common.create_variable(initial_value=0, dtype=tf.int32, shape=(), name='TimeStep')
        self._output_stream = output_stream if output_stream else sys.stdout
        # Initialise as per the parent class.
        super(EvalPerStepLogger, self).__init__(name, prefix)

    @common.function
    def call(self, time_step_data: Tuple[TimeStep, PolicyStep, TimeStep]) -> \
            Tuple[TimeStep, PolicyStep, TimeStep]:
        """
        Parses the time step of experience and logs relevant information.

        :param time_step_data: The experience of the agent in the environment.
        :return: The original experience data passed in.
        """
        # Log the values of the state and the action.
        tf.print(
            "Step: ", self._t, "\t",
            "State: ", time_step_data[0].observation, "\t",
            "Action: ", time_step_data[1].action,
            end="\n",
            output_stream=self._output_stream
        )
        # Increment the time step counter.
        self._t.assign_add(tf.constant(1))
        return time_step_data

    def reset(self) -> None:
        """Resets the values being tracked by the metric."""
        # Reset the time step counter to zero.
        self._t.assign(tf.constant(0))

    def result(self) -> None:
        """
        Computes and returns a final value for the metric.
        In this case there is no calculated metric value as this 'metric' simply performs logging.
        """
        pass


class ActionProbabilityMetric(TFStepMetric):
    """
    A metric that records the average action probabilities over a given period.
    Implementation similar to tf_agent.metrics.tf_metrics.AverageReturnMetric
    """

    def __init__(self,
                 policy: tf_policy.TFPolicy,
                 action_indices: Tuple[int, ...],
                 name: str = 'ActionProbability',
                 prefix: str = 'Metrics',
                 dtype: Type = tf.float32,
                 batch_size: int = 1,
                 buffer_size: int = 10):
        """
        :param policy: Policy of the agent used for reevaluation to attain action probabilities at
            each time step.
        :param action_indices: A tuple of indices of the action probability vector to track. This is
            a tuple to allow for the case where the action is a tuple of tensors.
        :param name: Name of the metric (as it will appear in tensorboard).
        :param prefix: Prefix to apply as part of the naming convention.
        :param dtype: Data type of the metric.
        :param batch_size: Batch size of the RL environment.
        :param buffer_size: The capacity of the buffer which will rewrite itself when full but is
            emptied at every logging point.
        """
        super().__init__(name=name, prefix=prefix)
        self._action_indices = action_indices
        self._dtype = dtype
        self._probability_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator'
        )
        self._policy = policy
        self._buffer = TFDeque(buffer_size, dtype)
        self._count_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='CountAccumulator'
        )

    @common.function(autograph=True)
    def call(self, trajectory: Trajectory) -> Trajectory:
        time_step = TimeStep(trajectory.step_type, trajectory.reward, trajectory.discount,
                             trajectory.observation)
        action_dist = self._policy.distribution(time_step).action

        # If the action distribution is in fact a tuple of distributions (one for each resource set)
        # then we need to index into them to attain the underlying distribution which can then be
        # used to attain probabilities. This is only the case where there are multiple resource
        # sets.
        for i in self._action_indices[:-1]:
            action_dist = action_dist[i]

        action_probs = action_dist.probs_parameter()
        # Zero out batch indices where a new episode is starting.
        self._probability_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._probability_accumulator),
                     self._probability_accumulator))
        self._count_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._count_accumulator),
                     self._count_accumulator))
        # Update accumulators with probability and count increments.
        self._probability_accumulator.assign_add(action_probs[..., 0, self._action_indices[-1]])
        self._count_accumulator.assign_add(tf.ones_like(self._count_accumulator))

        # Add final cumulants to buffer at the end of episodes.
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
        for idx in last_episode_indices:
            self._buffer.add(self._probability_accumulator[idx] / self._count_accumulator[idx])

        return trajectory

    def result(self) -> tf.Tensor:
        """Return the metric value."""
        return self._buffer.mean()

    @common.function
    def reset(self) -> None:
        """Clear the buffer and reset the accumulators."""
        self._buffer.clear()
        self._probability_accumulator.assign(tf.zeros_like(self._probability_accumulator))
        self._count_accumulator.assign(tf.zeros_like(self._count_accumulator))

    @property
    def action_indices(self) -> Tuple[int, ...]:
        return self._action_indices
