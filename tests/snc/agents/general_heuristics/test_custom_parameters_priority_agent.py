import numpy as np
import pytest

import snc.environments.job_generators.discrete_review_job_generator \
    as drjg
import snc.environments.controlled_random_walk as crw
import snc.environments.state_initialiser as si
import snc.agents.general_heuristics.custom_parameters_priority_agent \
    as custom_parameters_priority_agent


def get_default_env_params(state, buffer_processing_matrix=None):
    """ Default environment parameters used in the tests. """
    num_buffers = state.shape[0]
    arrival_rate = np.ones_like(state)
    num_resources = num_buffers
    constituency_matrix = np.zeros((num_resources, num_resources))
    time_interval = 1
    if buffer_processing_matrix is None:
        # default upper triangular buffer_processing_matrix
        buffer_processing_matrix = -np.triu(np.ones((num_buffers, num_resources))) * 0.5

    return {
        "cost_per_buffer": np.zeros_like(state),
        "capacity": np.zeros_like(state),
        "constituency_matrix": constituency_matrix,
        "job_generator": drjg.DeterministicDiscreteReviewJobGenerator(
            arrival_rate, buffer_processing_matrix, sim_time_interval=time_interval
        ),
        "state_initialiser": si.DeterministicCRWStateInitialiser(state),
        "job_conservation_flag": True,
        "list_boundary_constraint_matrices": None,
    }


# tests for the initial asserts
def test_assert_no_parameter_is_true():
    state = 5 * np.ones((1, 1))
    env_params = get_default_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 1))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 1))]
    env = crw.ControlledRandomWalk(**env_params)
    pytest.raises(AssertionError, custom_parameters_priority_agent.CustomParametersPriorityAgent,
                  env, state_option=False, cost_option=False, rate_option=False, name="CPPAgent")


def test_assert_activity_performed_by_multiple_resources():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state)
    env_params["constituency_matrix"] = np.ones((2, 2))
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)
    pytest.raises(AssertionError, custom_parameters_priority_agent.CustomParametersPriorityAgent,
                  env, state_option=False, cost_option=False, rate_option=True, name="CPPAgent")


def test_assert_activity_performed_by_no_resource():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state)
    env_params["constituency_matrix"] = np.zeros((2, 2))
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)
    pytest.raises(AssertionError,
                  custom_parameters_priority_agent.CustomParametersPriorityAgent,
                  env, state_option=False, cost_option=False, rate_option=True, name="CPPAgent")


def test_assert_no_push_model():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 0]])]
    env = crw.ControlledRandomWalk(**env_params)
    pytest.raises(AssertionError, custom_parameters_priority_agent.CustomParametersPriorityAgent,
                  env, state_option=False, cost_option=False, rate_option=True, name="CPPAgent")


# tests for compute_priority_values with all the possible parameters combinations
def test_compute_priority_values_with_state():
    state = np.array([[5], [10]])
    env_params = get_default_env_params(state=state)
    # constituency_matrix and list_boundary_constraint_matrix are dummy matrices not used in the
    # priority_value computation. They are needed to pass the initial asserts (tested above).
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=False,
                                                                           rate_option=False,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-5, -5], [0, -10]])
    assert (priority_value == true_priority_values).all()


def test_compute_priority_values_with_cost():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state=state)
    # constituency_matrix and list_boundary_constraint_matrix are dummy matrices not used in the
    # priority_value computation. They are needed to pass the initial asserts (tested above).
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [3]])
    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=True,
                                                                           rate_option=False,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-2, -2], [0, -3]])
    assert (priority_value == true_priority_values).all()


def test_compute_priority_values_with_rate():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state=state)
    # constituency_matrix and list_boundary_constraint_matrix are dummy matrices not used in the
    # priority_value computation. They are needed to pass the initial asserts (tested above).
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-1/0.5, -1/0.5], [0, -1/0.5]])).all()


def test_compute_priority_values_all_combination():
    state = np.array([[5], [10]])
    env_params = get_default_env_params(state=state)
    # constituency_matrix and list_boundary_constraint_matrix are dummy matrices not used in the
    # priority_value computation. They are needed to pass the initial asserts (tested above).
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [3]])
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=True,
                                                                           rate_option=False,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-10, -10], [0, -30]])
    assert (priority_value == true_priority_values).all()

    # state and rate
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-5/0.5, -5/0.5], [0, -10/0.5]])
    assert (priority_value == true_priority_values).all()

    # cost and rate
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=True,
                                                                           rate_option=True,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-2/0.5, -2/0.5], [0, -3/0.5]])
    assert (priority_value == true_priority_values).all()

    # state, cost, and rate
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=True,
                                                                           rate_option=True,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-10/0.5, -10/0.5], [0, -30/0.5]])
    assert (priority_value == true_priority_values).all()


def test_compute_priority_values_remove_positive_rate():
    state = 5 * np.ones((2, 1))
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-0.5, 0.5], [0, -0.5]]))
    # constituency_matrix and list_boundary_constraint_matrix are dummy matrices not used in the
    # priority_value computation. They are needed to pass the initial asserts (tested above).
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-1/0.5, 0], [0, -1/0.5]])).all()


# tests for map_state_to_action
def test_no_possible_action():
    state = np.array([[0], [0]])
    env_params = get_default_env_params(state=state)
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [3]])
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=True,
                                                                           rate_option=False,
                                                                           name="CPPAgent")
    action = agent.map_state_to_actions(state)
    true_action = np.zeros((2, 1))
    assert (action == true_action).all()


def test_all_activities_are_possible():
    state = np.array([[5], [10]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-5, 0], [0, -5]]))
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [3]])
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=True,
                                                                           cost_option=True,
                                                                           rate_option=False,
                                                                           name="CPPAgent")
    action = agent.map_state_to_actions(state)
    true_action = np.ones((2, 1))
    assert (action == true_action).all()


def test_multiple_activities_one_resource():
    # resource 0 can do activity 0 and activity 2. Activity 0 is the one with the highest value and
    # so the corresponding action is selected. resource 1 can do activity 1 however the related
    # buffer is empty.
    state = np.array([[10], [0]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array(
                                            [[-2, 0, -1], [0, -1, 0]]))
    env_params["constituency_matrix"] = np.array([[1, 0, 1],
                                                  [0, 1, 0]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [3]])
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")

    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-1/2, 0, -1], [0, -1, 0]])
    assert (priority_value == true_priority_values).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([[0], [0], [1]])
    assert (action == true_action).all()


def test_action_on_multiple_buffers_one_is_empty():
    # an activity works on two buffer, the buffer for which the priority value is the lowers in not
    # empty but the second one is. Thus, the activity should not be selected to perform the action.
    state = np.array([[10], [0], [0]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-2, 0],
                                                                           [0, -1],
                                                                           [-0.5, 0]]))
    env_params["constituency_matrix"] = np.array([[1, 0],
                                                  [0, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0, 1]]),
                                                       np.array([[0, 1, 0]])]
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")

    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[-1/2, 0], [0, -1], [-1/0.5, 0]])
    assert (priority_value == true_priority_values).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 0])[:, None]
    assert (action == true_action).all()


def test_second_lowest_is_chosen():
    # a resource (0) has two possible activities (0, 2). The one with the lower priority value (0)
    # works on an empty buffer and so cannot be selected. Thus, the second activity (2) has to be
    # chosen to perform the action.
    state = np.array([[10], [0], [0]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[0, 0, -2],
                                                                           [0, -1, 0],
                                                                           [-1, 0, 0]]))
    env_params["constituency_matrix"] = np.array([[1, 0, 1],
                                                  [0, 1, 0]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0, 1]]),
                                                       np.array([[0, 1, 0]])]
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")

    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[0, 0, -1/2], [0, -1, 0], [-1, 0, 0]])
    assert (priority_value == true_priority_values).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 0, 1])[:, None]
    assert (action == true_action).all()


def test_when_if_is_triggered():
    # the buffer_processing_matrix has no negative value so the if min_value >= 0 is triggered
    state = np.array([[10], [7]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[2, 0, 4],
                                                                           [0, 3, 0]]))
    env_params["constituency_matrix"] = np.array([[1, 0, 1],
                                                  [0, 1, 0]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 0]]),
                                                       np.array([[0, 1]])]
    env = crw.ControlledRandomWalk(**env_params)

    # state and cost
    agent = custom_parameters_priority_agent.CustomParametersPriorityAgent(env,
                                                                           state_option=False,
                                                                           cost_option=False,
                                                                           rate_option=True,
                                                                           name="CPPAgent")

    priority_value = agent.compute_priority_values(state)
    true_priority_values = np.array([[0, 0, 0], [0, 0, 0]])
    assert (priority_value == true_priority_values).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 0, 0])[:, None]
    assert (action == true_action).all()


# tests for the classes without flags
def test_class_state():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-2, 0],
                                                                           [0, -3]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[10], [2]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityState(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-4, 0], [0, -5]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 1])[:, None]
    assert (action == true_action).all()


def test_class_cost():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-3, 0],
                                                                           [0, -2]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[10], [2]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityCost(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-10, 0], [0, -2]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([1, 0])[:, None]
    assert (action == true_action).all()


def test_class_rate():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-2, 0],
                                                                           [0, -3]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [10]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityRate(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-1/2, 0], [0, -1/3]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([1, 0])[:, None]
    assert (action == true_action).all()


def test_class_state_cost():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-3, 0],
                                                                           [0, -2]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[10], [2]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityStateCost(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-40, 0], [0, -10]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([1, 0])[:, None]
    assert (action == true_action).all()


def test_class_state_rate():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-3, 0],
                                                                           [0, -2]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[10], [2]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityStateRate(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-4/3, 0], [0, -5/2]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 1])[:, None]
    assert (action == true_action).all()


def test_class_cost_rate():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-2, 0],
                                                                           [0, -3]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[10], [2]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityCostRate(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-10/2, 0], [0, -2/3]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([1, 0])[:, None]
    assert (action == true_action).all()


def test_class_state_cost_rate():
    state = np.array([[4], [5]])
    env_params = get_default_env_params(state=state,
                                        buffer_processing_matrix=np.array([[-2, 0],
                                                                           [0, -5]]))
    env_params["constituency_matrix"] = np.array([[1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1]])]
    env_params["cost_per_buffer"] = np.array([[2], [10]])
    env = crw.ControlledRandomWalk(**env_params)

    agent = custom_parameters_priority_agent.PriorityStateCostRate(env)
    priority_value = agent.compute_priority_values(state)
    assert (priority_value == np.array([[-8/2, 0], [0, -50/5]])).all()

    action = agent.map_state_to_actions(state)
    true_action = np.array([0, 1])[:, None]
    assert (action == true_action).all()
