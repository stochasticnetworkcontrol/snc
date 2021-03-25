import os
import sys
import json
import time
import numpy as np
from datetime import datetime
import argparse
import tensorflow as tf
from warnings import warn

from typing import Dict, Tuple, List, Any, Optional, Sequence, Union

import tf_agents
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics import tf_metrics, tf_metric

from src.experiment.rl import custom_tf_metrics
from src.snc.environments import scenarios
from src.snc import compute_load_workload_matrix
from src.snc import compute_minimal_draining_time_from_env_cvxpy
from src.experiment.experiment_utils import get_args_string
from src.snc.environments import rl_env_from_snc_env
from src.snc import create_reinforce_agent, create_ppo_agent

from src.snc import set_up_json_logging

set_up_json_logging()


def get_environment(env_name: str,
                    agent_name: str,
                    episode_len_to_min_drain_time_ratio: float,
                    terminal_discount_factor: float = 0.7,
                    action_repetitions: int = 1,
                    parallel_environments: int = 8,
                    env_overload_params: Optional[Dict] = None,
                    agent_params: Optional[Dict] = None,
                    seed: Optional[int] = None) \
        -> Tuple[TFPyEnvironment, float, float, int, Tuple[int, ...]]:
    """
    Builds and initialises a TensorFlow environment implementation of the Single Server Queue.

    :param env_name: The name of the scenario to load. Must be in the list of implemented scenarios.
    :param agent_name: The name of the RL agent the environment is to be set up for.
    :param episode_len_to_min_drain_time_ratio: Maximum number of time steps per episode as a
        proportion of the minimal draining time.
    :param terminal_discount_factor: The discount applied to the final time step from which a
        per-step discount factor is calculated.
    :param action_repetitions: Number of time steps each selected action is repeated for.
    :param parallel_environments: Number of environments to run in parallel.
    :param env_overload_params: Dictionary of parameters to override the scenario defaults.
    :param agent_params: Optional dictionary of agent parameters the environment can be adapted for.
    :param seed: Random seed used to initialise the environment.
    :return: The environment wrapped and ready for TensorFlow Agents.
    """
    # Handle some default argument clean up.
    if env_overload_params is None:
        env_overload_params = {}

    env = scenarios.load_scenario(env_name, seed, env_overload_params).env

    if np.all(env.state_initialiser.initial_state == 0):
        env.max_episode_length = 450
    else:
        if env.state_initialiser.initial_state.ndim == 1:
            initial_state = env.state_initialiser.initial_state.reshape((-1, 1))
        else:
            initial_state = env.state_initialiser.initial_state
        minimal_draining_time = compute_minimal_draining_time_from_env_cvxpy(initial_state, env)
        env.max_episode_length = int(episode_len_to_min_drain_time_ratio * minimal_draining_time)
    discount_factor = np.exp(np.log(terminal_discount_factor) / env.max_episode_length)
    load = np.max(compute_load_workload_matrix(env).load)
    max_ep_len = env.max_episode_length

    # Allow toggling of observation normalisation in the environment.
    # The typical behaviour for PPO is that PPO normalises observations internally as necessary so
    # normalisation in the environment is not necessary.
    if agent_name == 'ppo' and agent_params.get('normalize_observations', True):
        normalise_obs_in_env = False
    else:
        normalise_obs_in_env = True

    # Wrap and parallelise environment for tf agents.
    tf_env, action_dims = rl_env_from_snc_env(env,
                                              discount_factor,
                                              action_repetitions,
                                              parallel_environments,
                                              normalise_observations=normalise_obs_in_env)
    return tf_env, discount_factor, load, max_ep_len, action_dims


def get_reinforce_agent(
        env: TFPyEnvironment,
        discount_factor: float,
        debug: bool = False,
        agent_params: Optional[Dict[str, Any]] = None
    ) -> ReinforceAgent:
    """
    Builds and initialises a REINFORCE learning agent for the environment.

    :param env: The TensorFlow environment used to set up the agent with correct action spaces etc.
    :param discount_factor: The discount applied to future rewards.
    :param debug: Flag which determines whether to include extra TensorBoard logs for debugging.
    :param agent_params: A dictionary of possible overrides for the default TF-Agents agent set up.
    :return: An initialised REINFORCE agent.
    """
    # Set up a training step counter.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = create_reinforce_agent(
        env,
        gamma=discount_factor,
        debug=debug,
        training_step_counter=global_step,
        agent_params=agent_params
    )
    agent.initialize()
    agent.train = tf.function(agent.train)
    return agent


def get_ppo_agent(
        env: TFPyEnvironment,
        num_epochs: int,
        discount_factor: float,
        debug: bool = False,
        agent_params: Optional[Dict[str, Any]] = None
    ) -> Union[ReinforceAgent, PPOAgent]:
    """
    Builds and initialises a reinforcement learning agent for the environment.

    :param env: The TensorFlow environment used to set up the agent with correct action spaces etc.
    :param num_epochs: The (maximal) number of internal PPO epochs to run.
    :param discount_factor: The discount applied to future rewards.
    :param debug: Flag which determines whether to include extra TensorBoard logs for debugging.
    :param agent_params: A dictionary of possible overrides for the default TF-Agents agent set up.
    :return: An initialised RL agent.
    """
    # Set up a training step counter.
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = create_ppo_agent(
        env,
        num_epochs,
        gamma=discount_factor,
        debug=debug,
        training_step_counter=global_step,
        agent_params=agent_params
    )
    agent.initialize()
    agent.train = tf.function(agent.train)
    return agent


def get_replay_buffer(
        env: TFPyEnvironment, agent: Union[ReinforceAgent, PPOAgent], max_length: int = 100000
    ) -> TFUniformReplayBuffer:
    """
    Sets up a replay buffer object for use in training the agent.

    :param env: TensorFlow environment which provides specifications for use in setting up a replay
        buffer.
    :param agent: The agent which provides specifications for use in setting up a replay buffer.
    :param max_length: The maximum length/capacity of the replay buffer.
    :return: A replay buffer (TFUniformReplayBuffer)
    """
    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=max_length
    )
    return replay_buffer


def set_up_summary_writers(log_dir: str,
                           action_space_dims: Tuple[int, ...],
                           flush_milliseconds: int = 500) \
        -> Tuple[tf.summary.SummaryWriter,
                 Sequence[tf.summary.SummaryWriter],
                 tf.summary.SummaryWriter]:
    """
    Sets up log directories and summary file writers to write log files which can be read by
    TensorBoard for in-play logging.

    :param log_dir: The base directory in which to store logs. If this directory does not exist it
        will be made along side 'train' and 'eval' subdirectories.
    :param action_space_dims: Action space dimensions, each entry is number of actions for a
        resource set.
    :param flush_milliseconds: The largest interval between flushes of the summary writers (in
        milliseconds)
    :return: A summary writer for training logging, a series of summary writers for action
        probability logging and another summary writer for logging at test/evaluation time.
    """
    # Build the two summary writers using TensorFlow's helper function.
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train"), flush_millis=flush_milliseconds, max_queue=3)
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "eval"), flush_millis=flush_milliseconds, max_queue=3)
    # Build a series of summary writers, one per action, to allow plotting on shared axes.

    action_probability_summary_writers = [
        tf.summary.create_file_writer(
            os.path.join(log_dir, f"train/action_{i}-{j}"),
            flush_millis=flush_milliseconds,
            max_queue=3)
        for i in range(len(action_space_dims))
        for j in range(action_space_dims[i])
    ]

    return train_summary_writer, action_probability_summary_writers, eval_summary_writer


def get_metrics(agent: Union[ReinforceAgent, PPOAgent],
                batch_size: int = 1,
                for_single_server_queue: bool = False,
                log_file_path: Optional[str] = None) -> \
        Tuple[List[tf_metric.TFStepMetric],
              List[tf_metric.TFStepMetric],
              List[tf_metric.TFStepMetric],
              List[tf_metric.TFStepMetric]]:
    """
    Sets up all of the metrics to track and log during training and evaluation.

    :param agent: The agent being trained and evaluated.
    :param batch_size: The batch size of the environment (i.e. how many environments are running in
        parallel).
    :param for_single_server_queue: Flag denoting whether the single server queue metrics can be
        utilised.
    :param log_file_path: Path to a file in which to save evaluation episode logs. If None defaults
        to sys.stdout.
    :return: Four lists of metrics, one for those to be tracked during training, one for action
        probability tracking and two for those to be tracked during periodic policy evaluation (the
        first at an episodic resolution and the second at a per time step level).
    """

    # We have a conditional tracker which requires the set up of conditions which evaluate which
    # time steps to include and which to discard when calculating the metric. The metric we wish to
    # calculate is the proportion of the number of time steps in which the buffer is non-empty that
    # the policy is not idling. This requires us to set up two conditions as below.
    # The first flags time steps where the buffer is not empty.
    def activity_tracker_filter_condition(trajectory):
        return tf.reduce_any(trajectory.observation > 0, axis=-1)

    # The second condition tests that the policy is not idling (True means the policy is acting to
    # drain the buffer).
    def activity_tracker_activity_condition(trajectory):
        return trajectory.action[..., 1] == 1

    # The metrics tracked during training are as follows:
    #   - Number of Episodes and Overall Number of Time Steps to allow for more informative
    #     TensorBoard plots.
    #   - The average return (over episodes) in the logging period.
    #   - [For SSQ] Our custom metric which tracks how the policy acts when the buffer is not empty.
    metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=batch_size)
    ]
    # Track the policy evolution through training by logging an (average) probability of each
    # action. This ignores the state but in many cases where the buffers are not empty (the usual
    # case for SNC) this logging is more valid. This is particularly useful where a steadystate is
    # reached.
    # We treat the case where action spec is a tuple (so that actions are a tuple of distributions)
    # separately. This occurs when there are multiple resource sets and hence there is a
    # distribution per resource set. See RLControlledRandomWalk for explicit treatment of resource
    # sets.
    if isinstance(agent.action_spec, tuple):
        policy_metrics = [custom_tf_metrics.ActionProbabilityMetric(agent.collect_policy, (i, j),
                                                                    batch_size=batch_size)
                          for i in range(len(agent.action_spec))
                          for j in range(agent.action_spec[i].shape[-1])]
    else:
        policy_metrics = [
            custom_tf_metrics.ActionProbabilityMetric(agent.collect_policy, (i,),
                                                      batch_size=batch_size)
            for i in range(agent.action_spec.shape[-1])
        ]
    if for_single_server_queue:
        metrics += [custom_tf_metrics.EpisodicConditionalActivityTracker1D(
            filter_condition=activity_tracker_filter_condition,
            activity_condition=activity_tracker_activity_condition,
            name="active_when_non_empty", dtype=tf.float32,
            batch_size=batch_size)]
    # The metrics tracked for policy evaluation at episode-level resolution are as follows:
    #   - The average return (over episodes) over the evaluation episodes.
    #   - [For SSQ] Our custom metric which tracks how the policy acts when the buffer is not empty.
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    if for_single_server_queue:
        eval_metrics += [custom_tf_metrics.EpisodicConditionalActivityTracker1D(
            filter_condition=activity_tracker_filter_condition,
            activity_condition=activity_tracker_activity_condition,
            name="active_when_non_empty", dtype=tf.float32, batch_size=batch_size)]
    # At every time step of each policy evaluation episode we log the state and the action. This is
    # performed by a metric which writes the related values to a log file or standard out.
    per_step_eval_metrics = [
        custom_tf_metrics.EvalPerStepLogger(output_stream=f'file://{log_file_path}')
    ]
    return metrics, policy_metrics, eval_metrics, per_step_eval_metrics


def get_collection_driver(
        env: TFPyEnvironment, agent: Union[ReinforceAgent, PPOAgent], observers: List[Any],
        policy_observers: Optional[List[tf_metric.TFStepMetric]], num_episodes: int
    ) -> DynamicEpisodeDriver:
    """
    Sets up a driver which will run data collection and in-training metric tracking.
    The driver is defined in tf_agents and handles agent play and monitoring as well as data
    storage for a fixed number of episodes at a time. This driver will be run to collect data once
    per training iteration.

    :param env: The TensorFlow environment object which will be run.
    :param agent: The agent to play in the environment.
    :param observers: A list of operations (including metrics to track) which will be executed in
        play to collect data and perform logging.
    :param policy_observers: A list of metrics to track which are executed in play throughout
        training.
    :param num_episodes: The number of episodes to play out in each driver run.
    :return: A driver to use for data collection (and in-play performance tracking)
    """
    collection_driver = DynamicEpisodeDriver(
        env,
        agent.collect_policy,
        observers=observers + policy_observers,
        num_episodes=num_episodes
    )
    # Wrap the run function for faster execution.
    collection_driver.run = tf.function(collection_driver.run)
    return collection_driver


def evaluate_policy(metrics: List[Any],
                    environment: TFPyEnvironment,
                    policy: tf_agents.policies.tf_policy.Base,
                    per_step_metrics: Optional[List[tf.Module]] = None,
                    num_episodes: int = 1,
                    train_step: Optional[Any] = None,
                    summary_writer: Optional[tf.summary.SummaryWriter] = None,
                    summary_prefix: str = "Eval",
                    logging: bool = False,
                    tf_log_stream_path: Optional[str] = None) -> None:
    """
    Track performance (via metrics) using policy in the environment provided.
    Prints a dictionary of results {metric_name: metric_value}.

    *NOTE*: Because placeholders are not compatible with Eager mode this is not compatible with
    python policies.

    This function is adapted from tf_agents.eval.metric_utils.eager_compute to allow for per time
    step logging.

    :param metrics: List of metrics to compute.
    :param environment: tf_environment instance.
    :param policy: tf_policy instance used to step the environment.
    :param per_step_metrics: List of metrics to be passed as observers to run every time step during
        evaluation.
    :param num_episodes: Number of episodes to compute the metrics over.
    :param train_step: An optional step to write summaries against.
    :param summary_writer: An optional writer for generating metric summaries.
    :param summary_prefix: An optional prefix scope for metric summaries.
    :param logging: Option to enable logging to the console of standard metrics.
    :param tf_log_stream_path: Path to a file which tf.print calls are set to write to. If none
        tf.print statements print to sys.stdout.
    """
    # Reset the state of all metrics (e.g. running totals for averages).
    for metric in metrics + per_step_metrics:
        metric.reset()

    # Attain the initial state of the environment and policy.
    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)

    # Set up a driver to run the evaluation episodes while logging the desired metrics.
    driver = DynamicEpisodeDriver(
        environment,
        policy,
        observers=metrics,
        transition_observers=per_step_metrics,
        num_episodes=num_episodes)

    # Run the driver which adds experience to the replay buffer.
    driver.run(time_step, policy_state)

    # If we have the required prerequisites then perform the TensorBoard logging as well as logging
    # results to the console.
    if train_step and summary_writer:
        # Utilise a (possibly) different summary writer to put the evaluation metrics to
        # TensorBoard.
        with summary_writer.as_default():
            for m in metrics:
                # Attain the full name of the metric to record.
                tag = "/".join([summary_prefix, m.name])
                # Simply calculating and forming the scalar summary in the current context with a
                # default summary writer does the logging to TensorBoard for us.
                tf.summary.scalar(name=tag, data=m.result(), step=train_step)
    # If requested to then log metrics to the console.
    if logging and train_step:
        for m in metrics:
            tf.print(f"Evaluation at step {train_step.numpy()}: {m.name}\t{m.result()}",
                     output_stream=f'file://{tf_log_stream_path}' if tf_log_stream_path else
                     sys.stdout)


def train_agent(
        env: TFPyEnvironment,
        agent: Union[ReinforceAgent, PPOAgent],
        data_collection_driver: DynamicEpisodeDriver,
        replay_buffer: TFUniformReplayBuffer,
        num_iters: int,
        global_step=None,
        metrics: Optional[Sequence[tf_metric.TFStepMetric]] = None,
        policy_metrics: Optional[Sequence[tf_metric.TFStepMetric]] = None,
        policy_summary_writers: Optional[Sequence[tf.summary.SummaryWriter]] = None,
        eval_env: Optional[TFPyEnvironment] = None,
        eval_summary_writer: Optional[tf.summary.SummaryWriter] = None,
        num_eval_episodes: int = 1,
        eval_metrics: Optional[List[tf_metric.TFStepMetric]] = None,
        per_step_eval_metrics: Optional[List[Any]] = None,
        eval_freq: int = 10,
        log_freq: int = 5,
        save_freq: int = 5,
        model_save_path: Optional[str] = None,
        tf_log_stream_path: Optional[str] = None) -> None:
    """
    Function for putting the pieces together to train and evaluate an agent.

    :param env: The environment for which the agent will be trained.
    :param agent: The agent to train.
    :param data_collection_driver: The driver used for data collection and metric tracking.
    :param replay_buffer: Replay buffer in which to store experience.
    :param num_iters: The number of training iterations to perform.
    :param global_step: A counter of the number of training iterations.
    :param metrics: A list of the metrics to track during training.
    :param policy_metrics: A list of metrics related to the policy distribution to track during
        training.
    :param policy_summary_writers: A list of summary writers to facilitate overlaying plots of
        policy metrics in TensorBoard.
    :param eval_env: The environment in which to play out evaluations of the policy.
    :param eval_summary_writer: The summary writer used for evaluation metrics.
    :param num_eval_episodes: The number of evaluation episodes to run at each evaluation point.
    :param eval_metrics: The metrics to track when evaluating the policy (with episodic resolution).
    :param per_step_eval_metrics: The metrics to track when evaluating the policy (with time step
        resolution).
    :param eval_freq: The number of training iterations between runs of policy evaluation logging.
    :param log_freq: The frequency with which to log values to TensorBoard.
    :param save_freq: The number of training iterations between model saves.
    :param model_save_path: Directory in which to save model checkpoints (weights etc). If None
        model will not be saved.
    :param tf_log_stream_path:
    """
    # Get the initial states of the agent and environment before training.
    time_step = env.reset()
    policy_state = agent.collect_policy.get_initial_state(env.batch_size)

    # Set up the model saving infrastructure if a path to save to is provided.
    save_model = bool(model_save_path)
    if save_model:
        # Ensure that we save all trackable values (i.e. variables) from the TensorFlow Agent.
        checkpoint = tf.train.Checkpoint(agent=agent)
        # The checkpoint manager enables us to save multiple versions of the check point at
        # different training steps. We save the 20 most recent saves to span a wide section of
        # training.
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_save_path, max_to_keep=20)
    else:
        # Warn the user that training will continue but models will not be saved.
        warn("No save directory provided. Model will not be saved.")

    if metrics is None:
        metrics = []
    if per_step_eval_metrics is None:
        per_step_eval_metrics = []
    # Set up a minimal training loop to simply test training mechanics work.
    for i in range(num_iters):
        with tf.summary.record_if(lambda: tf.math.equal(global_step % log_freq, 0)):
            # Collect experience.
            time_step, policy_state = data_collection_driver.run(
                time_step=time_step,
                policy_state=policy_state
            )
            # Now the replay buffer should have data in it so we can collect the data and train the
            # agent.
            experience = replay_buffer.gather_all()
            agent.train(experience)
            # Clear the replay buffer and return to play.
            replay_buffer.clear()
            for metric in metrics:
                metric.tf_summaries(
                    train_step=global_step,
                    step_metrics=metrics[:2]
                )
            # Run the policy tracking metrics one at a time each on their own summary writer to
            # enable shared axes on TensorBoard.
            for metric, summary_writer in zip(policy_metrics, policy_summary_writers):
                with summary_writer.as_default():
                    tf.summary.scalar(name=metric.name, data=metric.result(), step=global_step)

        if eval_summary_writer and eval_metrics and eval_env:
            if i > 0 and global_step % eval_freq == 0:
                evaluate_policy(
                    eval_metrics,
                    eval_env,
                    agent.policy,
                    per_step_metrics=per_step_eval_metrics,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix="Metrics",
                    logging=True,
                    tf_log_stream_path=tf_log_stream_path
                )
        # Periodically save the model provided that we have the infrastructure in place.
        if save_model and i > 0 and (i + 1) % save_freq == 0:
            checkpoint_manager.save(i + 1)
        if i % (num_iters // 100) == 0:
            print(f"\tCompleted: {i / num_iters * 100} %")
    checkpoint_manager.save(num_iters)


def main(params):
    """Puts everything together to set up and run the experiment."""
    # Attain a step counter which will be incremented every time the agent is trained.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # Attain separate environments for training and evaluation.
    env, gamma, load, max_ep_len, action_dims = get_environment(
        params.env_name,
        params.agent_name,
        params.episode_len_to_min_drain_time_ratio,
        params.terminal_discount_factor,
        params.action_repetitions,
        params.parallel_envs,
        params.env_param_overrides,
        params.rl_agent_params
    )
    eval_params = params.env_param_overrides.copy()
    eval_params['job_gen_seed'] = (eval_params.get('job_gen_seed', 0) or 0) + 1234
    eval_env, _, _, _, _ = get_environment(
        params.env_name,
        params.agent_name,
        params.episode_len_to_min_drain_time_ratio,
        params.terminal_discount_factor,
        params.action_repetitions,
        1,
        params.env_param_overrides,
        params.rl_agent_params,
        seed=eval_params['job_gen_seed']
    )

    # Update parameters with calculated values so that they can be logged.
    params.load = load
    params.max_ep_len = max_ep_len

    # Get the summary writers.
    train_summary_writer, action_prob_summary_writers, eval_summary_writer = \
        set_up_summary_writers(params.logdir, action_dims)
    # Set the training summary writer as the default. This means that unless otherwise stated any
    # summary made on the main thread will be logged using train_summary_writer.
    train_summary_writer.set_as_default()
    # Set up the agent, replay buffer and metrics.
    if params.agent_name.lower() == 'reinforce':
        agent = get_reinforce_agent(
            env, gamma, debug=not params.no_debug, agent_params=params.rl_agent_params
        )
    elif params.agent_name.lower() == 'ppo':
        assert 'num_epochs' in params.rl_agent_params, "PPO Agents require a num_epochs parameter."
        agent = get_ppo_agent(
            env,
            params.rl_agent_params['num_epochs'],
            gamma,
            debug=not params.no_debug,
            agent_params=params.rl_agent_params
        )
    else:
        raise NotImplementedError(f"RL Agent of type {params.agent_name} not currently supported.")

    with open(os.path.join(params.logdir, 'params.txt'), 'w') as param_file:
        param_file.write(get_args_string(params))

    replay_buffer = get_replay_buffer(env, agent)
    # Write logs to a file rather than the console.
    tf_log_stream_path = os.path.join(params.logdir, 'eval_logs.txt')
    metrics, policy_metrics, eval_metrics, per_step_eval_metrics = get_metrics(
        agent=agent,
        batch_size=env.batch_size,
        for_single_server_queue=params.env_name == 'single_server_queue',
        log_file_path=tf_log_stream_path)
    # Combine the method to add experience to the replay buffer with the metrics to track during
    # training to attain the list of observers which will be run by the DynamicEpisodeDriver.
    observers = [replay_buffer.add_batch] + metrics
    # Attain a driver which will run episodes with the policy for data collection (using Boltzmann
    # exploration), save experience to the replay buffer and log metrics to TensorBoard.
    driver = get_collection_driver(env, agent, observers, policy_observers=policy_metrics,
                                   num_episodes=params.episodes_per_iter)
    # Run the full training function.
    train_agent(env, agent, driver, replay_buffer, params.num_iters, global_step,
                eval_env=eval_env, metrics=metrics, eval_metrics=eval_metrics,
                policy_metrics=policy_metrics, per_step_eval_metrics=per_step_eval_metrics,
                policy_summary_writers=action_prob_summary_writers, eval_freq=params.eval_freq,
                eval_summary_writer=eval_summary_writer, log_freq=params.log_freq,
                save_freq=params.save_freq, model_save_path=params.savedir,
                tf_log_stream_path=tf_log_stream_path)


def parse_args() -> argparse.Namespace:
    """Processes command line arguments and collects them in the named tuple returned·"""
    params = argparse.ArgumentParser(description="Experiment Arguments with RL agent.")
    params.add_argument("--env_name", type=str, default="klimov_model",
                        help="Name of environment to run on. "
                             "Must be in the list of implemented scenarios.")
    params.add_argument("--agent_name", type=str, default="ppo",
                        help="Name of the agent to train.")
    params.add_argument("--experiment_name", type=str, default=None,
                        help="A name for the experiment. Used in creating log directories."
                             "If None, the results will be saved in a time stamped folder.")
    params.add_argument("--logdir", type=str, default="../../../../tmp/{}/{}",
                        help="Directory for storing TensorBoard logs.")
    params.add_argument("--savedir", type=str, default=None,
                        help="directory for storing model weights."
                             "If None then will save in subdirectory of logdir.")
    params.add_argument("-melr", "--episode_len_to_min_drain_time_ratio", type=float, default=1.5,
                        help="Maximum number of steps per episode as a proportion of the minimal "
                             "draining time.")
    params.add_argument("--action_repetitions", type=int, default=1,
                        help="Number of time steps each selected action is repeated for.")
    params.add_argument("--log_freq", type=int, default=5,
                        help="Number of training iterations between TensorBoard logging.")
    params.add_argument("--eval_freq", type=int, default=5,
                        help="Number of training iterations between policy evaluation.")
    params.add_argument("--save_freq", type=int, default=500,
                        help="Number of training iterations between model saves. "
                             "The training process will store the 20 most recent models and hence "
                             "the range of iterations captured in saved models will be "
                             "20 * save_freq")
    params.add_argument("-N", "--num_iters", type=int, default=250,
                        help="Number of training iterations.")
    params.add_argument("-n", "--episodes_per_iter", type=int, default=25,
                        help="Number of episodes to run in each training iteration.")
    params.add_argument("-ep", "--env_param_overrides", type=str, default='{}',
                        help="JSON formatted dictionary of environment parameter overrides. "
                             "May be a string or a path to a JSON file.")
    params.add_argument("--rl_agent_params", type=str, default='{"num_epochs": 10}',
                        help="JSON formatted dictionary of agent parameters. "
                             "May be a string or a path to a JSON file.")
    params.add_argument("-df", "--terminal_discount_factor", type=float, default=0.7,
                        help="Compounded discount factor at the final episode step.")
    params.add_argument("--seed", type=int, default=int(time.time()),
                        help="Random seed for numpy and TensorFlow.")
    params.add_argument("--parallel_envs", type=int, default=8,
                        help="Number of environments to run in parallel to speed up data "
                             "collection")

    # Debug logging is on by default.
    params.add_argument("--no_debug", default=False, action="store_true")

    # Collect the parameters in a namespace.
    parsed_params = params.parse_args()

    assert parsed_params.env_name in scenarios.SCENARIO_CONSTRUCTORS, \
        "Scenario passed does not exist."

    # Set the logging folder from the load automatically.
    if "{}" in parsed_params.logdir:
        if parsed_params.experiment_name is None:
            parsed_params.experiment_name = datetime.strftime(datetime.now(),
                                                              "%Y_%m_%d__%H:%M:%S.%f")
        # Set the logging folder from the load automatically.
        parsed_params.logdir = parsed_params.logdir.format(
            parsed_params.env_name + '-' + parsed_params.agent_name,
            parsed_params.experiment_name)
    # If no specific save directory provided then build one based on the logging directory.
    if parsed_params.savedir is None:
        parsed_params.savedir = os.path.join(parsed_params.logdir.format(
            parsed_params.env_name + '-' + parsed_params.agent_name), "models")

    # Support environment parameters passed as a string of a dictionary or as a path to a JSON file.
    if os.path.exists(parsed_params.env_param_overrides):
        with open(parsed_params.env_param_overrides, 'r') as json_file:
            parsed_params.env_param_overrides = json.load(json_file)
    else:
        parsed_params.env_param_overrides = json.loads(parsed_params.env_param_overrides)

    # Load agent parameters.
    if os.path.exists(parsed_params.rl_agent_params):
        with open(parsed_params.rl_agent_params, 'r') as json_file:
            parsed_params.rl_agent_params = json.load(json_file)
    else:
        parsed_params.rl_agent_params = json.loads(parsed_params.rl_agent_params)

    # We want to make sure that the seed is not None
    if parsed_params.seed is None:
        parsed_params.seed = int(time.time())

    # If job_gen_seed has not been specified, then we are going to set it equal to param.seed
    # If the two seeds are different, we will print a warning
    if 'job_gen_seed' in parsed_params.env_param_overrides:
        assert parsed_params.env_param_overrides["job_gen_seed"] is not None
        if parsed_params.env_param_overrides["job_gen_seed"] != parsed_params.seed:
            warn("Seed for environment job generator differs from random seed "
                 "supplied to  rl_experiment_script.py.")
    else:
        parsed_params.env_param_overrides["job_gen_seed"] = parsed_params.seed

    # This means that all list type objects will be cast to numpy arrays before passing to the
    # environment. Should functionality of the ControlledRandomWalk environments change in future
    # this will need to be reviewed.
    for p in parsed_params.env_param_overrides:
        if isinstance(parsed_params.env_param_overrides[p], list):
            parsed_params.env_param_overrides[p] = np.array(parsed_params.env_param_overrides[p])

    # Write the parameters out to a log file for future reference.
    if not os.path.exists(parsed_params.logdir):
        os.makedirs(parsed_params.logdir)

    return parsed_params


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    # TODO: Delete line when TensorFlow Issue 37252 is remedied. Link below
    # TODO: https://github.com/tensorflow/tensorflow/issues/37252
    # monkey_patch_tf_get_seed(args.seed)
    # tf.random.set_seed(args.seed)
    main(args)
    print("Training complete!")
