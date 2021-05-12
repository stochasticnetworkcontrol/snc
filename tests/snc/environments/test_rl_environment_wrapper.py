import pytest
import numpy as np
import tensorflow as tf
from gym import spaces as gym_spaces

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment

import snc.utils.snc_tools as snc
from snc.environments.rl_environment_wrapper \
    import RLControlledRandomWalk, rl_env_from_snc_env
from snc.environments.job_generators. \
    scaled_bernoulli_services_poisson_arrivals_generator  \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from snc.environments.job_generators.discrete_review_job_generator \
    import DeterministicDiscreteReviewJobGenerator
import snc.environments.state_initialiser as stinit
from snc.environments.scenarios import load_scenario


def test_find_coupled_resource_sets_a():
    """
    Test the _find_coupled_resource_sets method of the wrapped RL environment directly.
    The method takes in a binary matrix denoting which resources are affected by which constraints
    and returns a list of sets of resources the actions of which are made dependent by the
    constraints. This is essentially noticing and interpreting the chaining effect of multiple
    constraints on activities.
    """
    # Each resource has one activity and all are independent.
    independent_resource_constraints_matrix = np.eye(4)  # => resource sets are [{0}, {1}, {2}, {3}]

    # Run the method for the test case defined above
    independent_resource_sets = RLControlledRandomWalk._find_coupled_resource_sets(
        independent_resource_constraints_matrix)

    # Assert that the responses are as expected.
    assert independent_resource_sets == [{0}, {1}, {2}, {3}]


def test_find_coupled_resource_sets_b():
    """
    Test the _find_coupled_resource_sets method of the wrapped RL environment directly.
    The method takes in a binary matrix denoting which resources are affected by which constraints
    and returns a list of sets of resources the actions of which are made dependent by the
    constraints. This is essentially noticing and interpreting the chaining effect of multiple
    constraints on activities.
    """

    # The constraints lead to resources 0 and 2, and 1 and 3 being coupled into two dependent sets.
    simple_resource_constraints_matrix = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])  # => Resource sets are [{0, 2}, {1, 3}]

    # Run the method for the test case defined above
    simple_resource_sets = RLControlledRandomWalk._find_coupled_resource_sets(
        simple_resource_constraints_matrix
    )

    # Assert that the responses are as expected.
    assert simple_resource_sets == [{0, 2}, {1, 3}]


def test_find_coupled_resource_sets_c():
    """
    Test the _find_coupled_resource_sets method of the wrapped RL environment directly.
    The method takes in a binary matrix denoting which resources are affected by which constraints
    and returns a list of sets of resources the actions of which are made dependent by the
    constraints. This is essentially noticing and interpreting the chaining effect of multiple
    constraints on activities.
    """

    # This test case tests the chaining effect. The first constraint links resources 0 and 1 then
    # the second links resources 2 and 3 and the third links 1 and 2 which creates a chain such that
    # resources 0, 1, 2 and 3 form one large dependent resource set
    overlapping_resource_constraints_matrix = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 0]
    ])  # => Resource set is {0, 1, 2, 3}

    # Run the method for the test case defined above
    overlapping_resource_sets = RLControlledRandomWalk._find_coupled_resource_sets(
        overlapping_resource_constraints_matrix
    )

    # Assert that the responses are as expected.
    assert overlapping_resource_sets == [{0, 1, 2, 3}]


def test_single_server_queue_spaces():
    """
    Tests the RL environment wrapper for the Single Server Queue. The set up is copied from the
    examples file: snc/stochastic_network_control/environments/examples.py
    """
    # Set up the environment parameters.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)

    # Test that the wrapper sets the spaces up as we would expect.
    assert isinstance(env.action_space, gym_spaces.Tuple)
    assert len(env._rl_action_space.spaces) == 1
    assert isinstance(env._rl_action_space.spaces[0], gym_spaces.Box)
    assert env.action_space.spaces[0].shape == (2,)
    assert isinstance(env.activities_action_space, gym_spaces.Box)
    assert env.activities_action_space.shape == (1,)
    assert isinstance(env.observation_space, gym_spaces.Box)
    assert env.observation_space.shape == (1,)


def test_independent_resource_actions():
    """
    Tests the RL environment wrapper for the ksrs_network_model example as per the examples file
    from which the initial set up code is copied.
    see snc/stochastic_network_control/environments/examples.py for the original code.
    """
    # Set up the environment parameters.
    alpha1, alpha3 = 2, 2
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (0, 0, 0, 0)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True
    seed = 72
    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)

    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # Test that the wrapper sets the spaces up as we would expect.
    assert len(env._rl_action_space.spaces) == 2
    assert np.all(np.array([s.shape[0] for s in env._rl_action_space.spaces]) == 3)
    assert len(env._action_vectors) == 3 + 3
    assert env.observation_space.shape == (4,)


def test_input_queued_switch_spaces():
    """
    Test the RL environment wrapper for the input_queued_switch_3x3_model example.
    This is a good test for resources with activity constraints.
    See snc/stochastic_network_control/environments/examples.py for the original set up code.
    """
    # Set up the environment parameters.
    mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33 = (1,) * 9
    cost_per_buffer = np.ones((9, 1))
    initial_state = np.zeros((9, 1))
    capacity = np.ones((9, 1)) * np.inf
    demand_rate = 0.3 * np.ones((9, 1))
    job_conservation_flag = True
    seed = 72
    buffer_processing_matrix = - np.diag([mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33])
    constraints = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0, 1]])
    constituency_matrix = np.vstack([
        np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1]]),
        constraints]
    )
    index_phys_resources = (0, 1, 2)

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 index_phys_resources=index_phys_resources)

    # Test that the environment wrapper works as expected.
    assert isinstance(env._rl_action_space, gym_spaces.Tuple)
    assert len(env._rl_action_space.spaces) == 1
    # There are overall 34 combinations of activities (or inactivity) that are feasible.
    assert len(env._action_vectors) == 34
    assert env._action_vectors.shape == (34, 9)
    assert snc.is_binary(env._action_vectors)
    assert snc.is_binary(np.matmul(env._action_vectors, constraints.T))


def test_rl_env_with_simple_activity_conditions():
    """
    Sets up a simple case of a tandem model with 4 independent resources each with one action.
    Activity constraints are then added as a way of testing the wrapper's handling of such activity
    constraints.
    """
    # Set up the environment parameters.
    alpha1 = 4
    cost_per_buffer = np.ones((4, 1))
    initial_state = np.zeros((4, 1))
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = False
    seed = 72

    demand_rate = np.array([alpha1, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([
        [-1, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1]
    ])

    constituency_matrix = np.vstack([np.eye(4), np.array([[1, 1, 0, 0]])])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 index_phys_resources=(0, 1, 2, 3))

    # Ensure that the action spaces are as expected.
    assert len(env._rl_action_space.spaces) == 3
    assert env._rl_action_space.spaces[0].shape == (3,)
    assert env._rl_action_space.spaces[1].shape == (2,)


def test_rl_env_with_complex_activity_conditions():
    """
    A more complicated version of the previous test (test_rl_env_with_simple_activity_conditions).
    Tests the handling of more complex activity constraints in an 8 resource and 8 activity tandem
    model.
    """
    # Set up environment parameters.
    alpha1 = 4
    cost_per_buffer = np.ones((8, 1))
    initial_state = np.zeros((8, 1))
    capacity = np.ones((8, 1)) * np.inf
    job_conservation_flag = False
    seed = 72

    demand_rate = np.array([alpha1, 0, 0, 0, 0, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1, -1]
    ])

    constraints_matrix = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0]
    ])
    constituency_matrix = np.vstack([np.eye(8), constraints_matrix])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 index_phys_resources=(0, 1, 2, 3, 4, 5, 6, 7))

    # Test that action spaces are as expected.
    assert len(env._rl_action_space.spaces) == 5
    # First group has 4 resources, and 8 actions.
    assert env._rl_action_space.spaces[0].shape == (8,)
    # Each independent group has only two actions.
    assert env._rl_action_space.spaces[1].shape == (2,)


def test_simple_rl_action_interpretation():
    """
    Tests the mechanism for reading in actions from an RL model.
    This utilises the 4 resource 4 activity tandem model.
    """
    # Tandem model set up without any additional activity constraints
    alpha1 = 4
    cost_per_buffer = np.ones((4, 1))
    initial_state = np.zeros((4, 1))
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = False
    seed = 72

    demand_rate = np.array([alpha1, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([
        [-1, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1]
    ])

    constituency_matrix = np.eye(4)

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 index_phys_resources=(0, 1, 2, 3))

    # Build various test RL actions in increasing complexity
    null_action = env._interpret_rl_action(np.zeros(8))
    binary_action_one = np.zeros(8)
    binary_action_one[1] = 1
    action_one = env._interpret_rl_action(binary_action_one)
    binary_action_two = np.zeros(8)
    binary_action_two[[1, 3, 5, 7]] = 1
    action_two = env._interpret_rl_action(binary_action_two)

    # Test that the actions are interpreted correctly
    assert np.all(null_action == np.zeros(4))
    assert np.all(action_one == np.array([1, 0, 0, 0]))
    assert np.all(action_two == np.ones(4))


def test_complex_rl_action_interpretation():
    """
    Tests the RL environment's action interpretation for the ksrs_network_model example as per the
    examples file from which the initial set up code is copied.
    See snc/stochastic_network_control/environments/examples.py for the original code.
    This tests action interpretation where each resource controls multiple activities.
    """
    # Set up model parameters
    alpha1, alpha3 = 2, 2
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (0, 0, 0, 0)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True
    seed = 72
    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # ----------------- Enumeration of Actions -----------------
    # Zero: First group idles
    # First: First activity on (first action of first group)
    # Second: Fourth activity on (second action of first group)
    # Third: Second group idles
    # Fourth: Second activity (first action of second group)
    # Fifth: Third activity (second action of second group)

    # Build various test RL actions in increasing complexity
    idle_action = env._interpret_rl_action(np.zeros(6))
    binary_action_a = np.zeros(6)
    binary_action_a[2] = 1
    action_a = env._interpret_rl_action(binary_action_a)
    binary_action_b = np.zeros(6)
    binary_action_b[[2, 4]] = 1
    action_b = env._interpret_rl_action(binary_action_b)

    # Check that the null action, a simple action and an action combining activities across resource
    # sets is set up correctly.
    assert np.all(idle_action == np.zeros(4))
    assert np.all(action_a == np.array([0, 0, 0, 1]))
    assert np.all(action_b == np.array([0, 1, 0, 1]))


def test_simple_numpy_action_interpretation():
    """
    Test interpretation of action where there is only one action space. This is done in the single
    server queue setting so that there is one resource and therefore one action space.

    This tests actions as numpy arrays.
    """
    # Set up single server queue.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)
    action = np.array([0, 1])
    assert env.preprocess_action(action) == np.array([1])


def test_simple_tensorflow_action_interpretation():
    """
    Test interpretation of action where there is only one action space. This is done in the single
    server queue setting so that there is one resource and therefore one action space.

    This tests actions as TensorFlow arrays.
    """
    # Set up single server queue.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)
    action = tf.convert_to_tensor([0, 1], tf.float32)
    assert env.preprocess_action(action) == np.array([1])


def test_tensorflow_action_interpretation():
    """
    Tests that tuples of TensorFlow tensors can be handled correctly by the environment.
    """
    # Set up the environment parameters.
    alpha1, alpha3 = 2, 2
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (0, 0, 0, 0)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True
    seed = 72
    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)

    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # Set up a TensorFlow action tuple with an action for each resource and ensure that the
    # environment is able to interpret it and update itself.
    action_tuple = (tf.convert_to_tensor([0, 1, 0]), tf.convert_to_tensor([0, 0, 1]))
    assert np.all(env.preprocess_action(action_tuple) == np.array([[1], [0], [1], [0]]))


def test_action_interpretation_error():
    """
    Tests that when actions are not of an expected format that the correct error is thrown.
    Uses the single server queue for simplicity.
    """
    # Set up single server queue.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)
    misformatted_action_tuple = (1, 2, 3)
    misformatted_action_value = int(1)
    pytest.raises(ValueError, env.preprocess_action, misformatted_action_tuple)
    pytest.raises(ValueError, env.preprocess_action, misformatted_action_value)


def test_stepping_environment_with_action_tuple():
    """
    Tests the RL environment wrapper for the ksrs_network_model example as per the examples file
    from which the initial set up code is copied.
    see snc/stochastic_network_control/environments/examples.py for the original code.
    """
    # Set up the environment parameters.
    alpha1, alpha3 = 2, 2
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (0, 0, 0, 0)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True
    seed = 72
    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)

    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # Set up a numpy action tuple with an action for each resource and ensure that the environment
    # is able to interpret it and update itself.
    action_tuple = (np.array([0, 1, 0]), np.array([0, 0, 1]))
    env.step(action_tuple)


def test_tf_environment_wrapping():
    """
    Test wrapping the RL environment for use with TensorFlow Agents.

    Use Simple Server Queue for simplicity
    """
    # Set up single server queue.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)

    # Try wrapping environment for tf agents.
    tf_env = TFPyEnvironment(GymWrapper(env))
    del tf_env


def test_render_error():
    """
    Test the render method.

    Included to give full test coverage.
    """
    # Set up single server queue.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)
    # Test that the environment fails to render. This will need to be fixed should we wish to
    # visualise episodes in future.
    pytest.raises(NotImplementedError, env.render)


def test_action_processing_in_single_server_queue():
    """
    Tests the interpretation and application of actions in the single server queue with
    deterministic dynamics. This is validated in terms of the environment state and rewards.
    """
    # Set up single server queue with no arrivals and deterministic dynamics.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (10,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.0
    job_conservation_flag = True

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = DeterministicDiscreteReviewJobGenerator(
        sim_time_interval=1,
        demand_rate=demand_rate,
        buffer_processing_matrix=buffer_processing_matrix
    )
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)
    new_state_idle, reward_idle, _, _ = env.step(np.array([1, 0]))
    assert new_state_idle[0] == initial_state[0]
    assert reward_idle == initial_state[0] * -1.
    new_state_active, reward_active, _, _ = env.step(np.array([0, 1]))
    assert new_state_active[0] == initial_state[0] - 1
    assert reward_active == (initial_state[0] - 1) * -1.


def test_action_processing_with_multiple_resource_sets():
    """
    Tests the interpretation and application of actions in a more complex environment with
    deterministic dynamics. This is validated in terms of the environment state and rewards.
    """
    # Set up the environment parameters with no arrivals and deterministic dynamics.
    alpha1, alpha3 = 0, 0
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (100, 100, 100, 100)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True

    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct deterministic job generator.
    job_generator = DeterministicDiscreteReviewJobGenerator(
        sim_time_interval=1,
        demand_rate=demand_rate,
        buffer_processing_matrix=buffer_processing_matrix
    )

    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # Ensure that the idling actions do not change the environment state and are rewarded
    # accordingly.
    idle_action = (np.array([1, 0, 0]), np.array([1, 0, 0]))
    new_state_idle, reward_idle, _, _ = env.step(idle_action)
    assert all((np.array(initial_state) / sum(initial_state)) == new_state_idle)
    assert reward_idle == -1 * np.dot(initial_state, cost_per_buffer)
    # Ensure that the action which activates activities 2 and 4 is interpreted and rewarded
    # correctly.
    active_action = (np.array([0, 0, 1]), np.array([0, 1, 0]))
    new_state_active, reward_active, _, _ = env.step(active_action)
    assert np.allclose(new_state_active, np.array([100, 97, 100, 97]) / 394)
    assert reward_active == -1 * np.dot(env.state.flatten(), cost_per_buffer)


def test_state_scaling_only_for_rl():
    """
    Tests that the state of the underlying ControlledRandomWalk object is not mutated as a side
    effect of the RLControlledRandomWalk wrapper.
    """
    # Set up the environment parameters with no arrivals and deterministic dynamics.
    alpha1, alpha3 = 0, 0
    mu1, mu2, mu3, mu4 = 10, 3, 10, 3
    cost_per_buffer = np.ones((4, 1))
    initial_state = (100, 100, 100, 100)
    capacity = np.ones((4, 1)) * np.inf
    job_conservation_flag = True

    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])

    # Construct deterministic job generator.
    job_generator = DeterministicDiscreteReviewJobGenerator(
        sim_time_interval=1,
        demand_rate=demand_rate,
        buffer_processing_matrix=buffer_processing_matrix
    )
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag)

    # Ensure that scaling is applied to the RL state but not the underlying stated.
    assert np.all(env.state == np.array([[100], [100], [100], [100]]))
    active_action = (np.array([0, 0, 1]), np.array([0, 1, 0]))
    new_state_active, _, _, _ = env.step(active_action)
    assert np.all(env.state == np.array([[100], [97], [100], [97]]))
    assert np.allclose(new_state_active, np.array([100, 97, 100, 97]) / 394)


def test_rl_env_from_snc_env_action_space_dims_simple():
    """
    Tests the formation and stability of the action space dimensions of the environment through
    the RL environment pipeline in a simple setting.
    """
    # Set up the environment parameters.
    cost_per_buffer = np.ones((1, 1))
    initial_state = (0,)
    capacity = np.ones((1, 1)) * np.inf
    demand_rate_val = 0.7
    job_conservation_flag = True
    seed = 72

    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = RLControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                                 state_initialiser, job_conservation_flag,
                                 list_boundary_constraint_matrices)

    _, action_space_dims = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=False)
    _, action_space_dims_tf = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=True)
    assert action_space_dims == action_space_dims_tf
    assert len(env.action_vectors) == sum(action_space_dims)


def test_rl_env_from_snc_env_action_space_dims_multiple_resource_sets():
    """
    Tests the formation and stability of the action space dimensions of the environment through
    the RL environment pipeline in a more complex setting.
    """
    # Set the environment name for this case as the asserts are difficult to make as variables.
    env_name = 'double_reentrant_line_shared_res_homogeneous_cost'
    # Set up the environment parameters.
    # Environment parameters do not affect the test result here.
    env = load_scenario(env_name, job_gen_seed=10).env
    rl_env, action_space_dims = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=False)
    _, action_space_dims_tf = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=True)
    assert action_space_dims == action_space_dims_tf
    assert len(rl_env.action_vectors) == sum(action_space_dims)


def test_rl_env_normalise_obs_property():
    """
    Ensure that the normalise_obs property of RLControlledRandomWalk is set and updated correctly.
    """
    # Set the environment name for this case as the asserts are difficult to make as variables.
    env_name = 'double_reentrant_line_shared_res_homogeneous_cost'
    # Set up the environment parameters.
    # Environment parameters do not affect the test result here.
    env = load_scenario(env_name, job_gen_seed=10).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=False)
    assert rl_env.normalise_obs is True
    rl_env.normalise_obs = False
    assert rl_env.normalise_obs is False


def test_rl_env_normalise_obs_action():
    """
    Ensure that the normalise_obs property of RLControlledRandomWalk is used correctly.
    """
    # Set the environment name for this case as the asserts are difficult to make as variables.
    env_name = 'klimov_model'
    # Set up the environment parameters.
    # Environment parameters do not affect the test result here.
    env = load_scenario(env_name,
                        job_gen_seed=10,
                        override_env_params={"initial_state": [100, 100, 100, 100]}).env
    rl_env, _ = rl_env_from_snc_env(env, discount_factor=0.99, for_tf_agent=False)
    assert rl_env.normalise_obs is True
    s0_normalised = rl_env.reset()
    assert s0_normalised.tolist() == [0.25, 0.25, 0.25, 0.25]
    rl_env.normalise_obs = False
    s0_unnormalised = rl_env.reset()
    assert s0_unnormalised.tolist() == [100, 100, 100, 100]
