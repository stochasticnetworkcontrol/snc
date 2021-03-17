import numpy as np
import pytest

import snc.environments.examples as examples
from snc.agents.activity_rate_to_mpc_actions.feedback_stationary_feasible_mpc_policy \
    import FeedbackStationaryFeasibleMpcPolicy


class TestGetIndActivitiesPerResource:

    @staticmethod
    def perform_test(constituency_matrix, true_activities_per_resource,
                     true_num_activities_per_resource):
        activities_per_resource, num_activities_per_resource = FeedbackStationaryFeasibleMpcPolicy. \
            get_ind_activities_per_resource(constituency_matrix)
        assert activities_per_resource == true_activities_per_resource
        assert num_activities_per_resource == true_num_activities_per_resource

    def test_get_ind_activities_per_resource_double_reentrant_line(self):
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        true_act_per_resource = [[0, 3], [1, 2]]
        true_num_act_per_resource = [2, 2]
        self.perform_test(constituency_matrix, true_act_per_resource, true_num_act_per_resource)

    def test_get_ind_activities_per_resource_case_1(self):
        constituency_matrix = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
        true_act_per_resource = [[0], [1, 2, 3]]
        true_num_act_per_resource = [1, 3]
        self.perform_test(constituency_matrix, true_act_per_resource, true_num_act_per_resource)

    def test_get_ind_activities_per_resource_case_2(self):
        constituency_matrix = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1]])
        true_act_per_resource = [[0], [1, 2, 3], [4]]
        true_num_act_per_resource = [1, 3, 1]
        self.perform_test(constituency_matrix, true_act_per_resource, true_num_act_per_resource)

    def test_get_ind_activities_per_resource_empty_resource(self):
        constituency_matrix = np.array([[1, 0], [0, 0]])
        with pytest.raises(AssertionError):
            _, _ = FeedbackStationaryFeasibleMpcPolicy.get_ind_activities_per_resource(constituency_matrix)

    def test_get_ind_activities_per_resource_with_one_shared_activity(self):
        constituency_matrix = np.array([[1, 0], [1, 1]])
        with pytest.raises(AssertionError):
            _, _ = FeedbackStationaryFeasibleMpcPolicy.get_ind_activities_per_resource(constituency_matrix)


class TestDoesActivityJDrainAnyEmptyBuffer:

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_false_case_1():
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [1]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert not drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_false_case_2():
        j = 1
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [1]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert not drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_false_case_3():
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [0]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert not drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_false_case_4():
        j = 1
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [1]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert not drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_true_case_1():
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [1]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_true_case_2():
        j = 1
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [0]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_true_case_3():
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [0]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert drain_empty_buffer

    @staticmethod
    def test_does_activity_j_drain_any_empty_buffer_true_case_4():
        j = 1
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [0]])
        drain_empty_buffer = FeedbackStationaryFeasibleMpcPolicy.\
            does_activity_j_drain_any_currently_empty_buffer(state, j, buffer_processing_matrix)
        assert drain_empty_buffer


class TestGetActivitiesDrainEmptyBuffers:
    @staticmethod
    def test_get_activities_drain_empty_buffers_empty():
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [1]])
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy.\
            get_activities_drain_currently_empty_buffers(state, buffer_processing_matrix)
        assert not act_drain_empty_buffers

    @staticmethod
    def test_get_activities_drain_empty_buffers_case_1():
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [1]])
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy.\
            get_activities_drain_currently_empty_buffers(state, buffer_processing_matrix)
        assert act_drain_empty_buffers == [0]

    @staticmethod
    def test_get_activities_drain_empty_buffers_case_2():
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[1], [0]])
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy.\
            get_activities_drain_currently_empty_buffers(state, buffer_processing_matrix)
        assert act_drain_empty_buffers == [1]

    @staticmethod
    def test_get_activities_drain_empty_buffers_case_3():
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        state = np.array([[0], [0]])
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy.\
            get_activities_drain_currently_empty_buffers(state, buffer_processing_matrix)
        assert act_drain_empty_buffers == [0, 1]


class TestGetValidActions:
    @staticmethod
    def test_get_valid_actions_empty_case_1():
        act_s = []
        act_drain_empty = []
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert not act_drain_empty_buffers

    @staticmethod
    def test_get_valid_actions_empty_case_2():
        act_s = []
        act_drain_empty = [0]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert not act_drain_empty_buffers

    @staticmethod
    def test_get_valid_actions_empty_case_3():
        act_s = [0]
        act_drain_empty = [0]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert not act_drain_empty_buffers

    @staticmethod
    def test_get_valid_actions_empty_case_4():
        act_s = [0, 1, 2, 3]
        act_drain_empty = [0, 1, 2, 3]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert not act_drain_empty_buffers

    @staticmethod
    def test_get_valid_actions_empty_draining_actions():
        act_s = [0]
        act_drain_empty = []
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0]

    @staticmethod
    def test_get_valid_actions_no_overlap():
        act_s = [0]
        act_drain_empty = [1]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0]

    @staticmethod
    def test_get_valid_actions_case_1():
        act_s = [0, 1]
        act_drain_empty = [1]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0]

    @staticmethod
    def test_get_valid_actions_case_2():
        act_s = [0, 1, 2, 3]
        act_drain_empty = [1]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0, 2, 3]

    @staticmethod
    def test_get_valid_actions_case_3():
        act_s = [0, 1, 2, 3]
        act_drain_empty = [1, 2]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0, 3]

    @staticmethod
    def test_get_valid_actions_case_4():
        act_s = [0, 1, 2, 3]
        act_drain_empty = [1, 2, 3]
        act_drain_empty_buffers = FeedbackStationaryFeasibleMpcPolicy. get_valid_actions(act_s,
                                                                                         act_drain_empty)
        assert act_drain_empty_buffers == [0]


class TestReallocateActivityJ:
    @staticmethod
    def test_reallocate_activity_j_error_not_draining_empty_buffer():
        j = 0
        activity_rates = np.array([[]])
        act_s = [0]
        act_drain_empty = []
        with pytest.raises(AssertionError):
            _ = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(j, activity_rates, act_s,
                                                                          act_drain_empty)

    @staticmethod
    def test_reallocate_activity_j_error_not_belonging_to_resource():
        j = 0
        activity_rates = np.array([[]])
        act_s = [1]
        act_drain_empty = [0]
        with pytest.raises(AssertionError):
            _ = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(j, activity_rates, act_s,
                                                                          act_drain_empty)

    @staticmethod
    def test_reallocate_activity_j_case_1():
        j = 0
        activity_rates = np.array([[0.1], [0.3], [0.4]])
        act_s = [0, 1]
        act_drain_empty = [0]
        true_new_activity_rates = np.array([[0], [0.4], [0.4]])
        new_activity_rates = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(
            j, activity_rates, act_s, act_drain_empty)
        np.testing.assert_almost_equal(new_activity_rates, true_new_activity_rates)

    @staticmethod
    def test_reallocate_activity_j_case_2():
        j = 0
        activity_rates = np.array([[0.1], [0.3], [0.4]])
        act_s = [0, 1, 2]
        act_drain_empty = [0]
        true_new_activity_rates = np.array([[0], [0.35], [0.45]])
        new_activity_rates = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(
            j, activity_rates, act_s, act_drain_empty)
        np.testing.assert_almost_equal(new_activity_rates, true_new_activity_rates)

    @staticmethod
    def test_reallocate_activity_j_case_3():
        j = 1
        activity_rates = np.array([[0.1], [0.3], [0.4]])
        act_s = [0, 1, 2]
        act_drain_empty = [1]
        true_new_activity_rates = np.array([[0.25], [0], [0.55]])
        new_activity_rates = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(
            j, activity_rates, act_s, act_drain_empty)
        np.testing.assert_almost_equal(new_activity_rates, true_new_activity_rates)

    @staticmethod
    def test_reallocate_activity_j_two_out_of_3_drain_empty_buffers():
        j = 1
        activity_rates = np.array([[0.1], [0.3], [0.4]])
        act_s = [0, 1, 2]
        act_drain_empty = [0, 1]
        true_new_activity_rates = np.array([[0.1], [0], [0.7]])
        new_activity_rates = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(
            j, activity_rates, act_s, act_drain_empty)
        np.testing.assert_almost_equal(new_activity_rates, true_new_activity_rates)

    @staticmethod
    def test_reallocate_activity_j_no_valid_activities():
        j = 1
        activity_rates = np.array([[0.1], [0.3], [0.4]])
        act_s = [0, 1, 2]
        act_drain_empty = [0, 1, 2]
        true_new_activity_rates = np.array([[0.1], [0], [0.4]])
        new_activity_rates = FeedbackStationaryFeasibleMpcPolicy.reallocate_activity_j(
            j, activity_rates, act_s, act_drain_empty)
        np.testing.assert_almost_equal(new_activity_rates, true_new_activity_rates)


class TestComputeFeedbackRatesInSimplex:
    @staticmethod
    def test_compute_feedback_rates_in_simplex():
        sum_actions = np.array([[5], [10], [1], [3]])
        activities_per_resource = [[0, 3], [1, 2]]
        num_steps_to_recompute_policy = 10
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        sum_actions_resources = constituency_matrix @ sum_actions
        true_activity_rates = np.array([[0.5], [1], [0.1], [0.3]])
        activity_rates = FeedbackStationaryFeasibleMpcPolicy.\
            clip_to_simplex_and_normalise_feedback_rates(sum_actions, activities_per_resource,
                                                         sum_actions_resources,
                                                         num_steps_to_recompute_policy)
        np.testing.assert_almost_equal(activity_rates, true_activity_rates)

    @staticmethod
    def test_compute_feedback_rates_in_simplex_one_resource_with_zero_actions():
        sum_actions = np.array([[0], [10], [0], [0]])
        activities_per_resource = [[0, 3], [1, 2]]
        num_steps_to_recompute_policy = 30
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        sum_actions_resources = np.squeeze(constituency_matrix @ sum_actions)
        true_activity_rates = np.array([[0], [1/3], [0], [0]])
        activity_rates = FeedbackStationaryFeasibleMpcPolicy.\
            clip_to_simplex_and_normalise_feedback_rates(sum_actions, activities_per_resource,
                                                         sum_actions_resources,
                                                         num_steps_to_recompute_policy)
        np.testing.assert_almost_equal(activity_rates, true_activity_rates)

    @staticmethod
    def test_compute_feedback_rates_in_simplex_all_zero_actions():
        sum_actions = np.zeros((4, 1))
        activities_per_resource = [[0, 3], [1, 2]]
        num_steps_to_recompute_policy = 1
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        sum_actions_resources = np.squeeze(constituency_matrix @ sum_actions)
        true_activity_rates = np.zeros((4, 1))
        activity_rates = FeedbackStationaryFeasibleMpcPolicy.\
            clip_to_simplex_and_normalise_feedback_rates(sum_actions, activities_per_resource,
                                                         sum_actions_resources,
                                                         num_steps_to_recompute_policy)
        np.testing.assert_almost_equal(activity_rates, true_activity_rates)

    @staticmethod
    def test_compute_feedback_rates_in_simplex_one_resource_with_negative_actions_but_positive_sum():
        sum_actions = np.array([[4], [10], [1], [-2]])
        activities_per_resource = [[0, 3], [1, 2]]
        num_steps_to_recompute_policy = 20
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        sum_actions_resources = np.squeeze(constituency_matrix @ sum_actions)
        true_activity_rates = np.array([[0.2], [0.5], [0.05], [0]])
        activity_rates = FeedbackStationaryFeasibleMpcPolicy.\
            clip_to_simplex_and_normalise_feedback_rates(sum_actions, activities_per_resource,
                                                         sum_actions_resources,
                                                         num_steps_to_recompute_policy)
        np.testing.assert_almost_equal(activity_rates, true_activity_rates)

    @staticmethod
    def test_compute_feedback_rates_in_simplex_one_resource_with_negative_sum():
        sum_actions = np.array([[-4], [10], [1], [2]])
        activities_per_resource = [[0, 3], [1, 2]]
        num_steps_to_recompute_policy = 20
        constituency_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        sum_actions_resources = np.squeeze(constituency_matrix @ sum_actions)
        true_activity_rates = np.array([[0], [0.5], [0.05], [0]])
        activity_rates = FeedbackStationaryFeasibleMpcPolicy.\
            clip_to_simplex_and_normalise_feedback_rates(sum_actions, activities_per_resource,
                                                         sum_actions_resources,
                                                         num_steps_to_recompute_policy)
        assert np.all(activity_rates >= 0)
        np.testing.assert_almost_equal(activity_rates, true_activity_rates)


class TestGenerateActionsWithFeedback:
    @staticmethod
    def perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials=1):
        # Double reentrant line.
        np.random.seed(seed=42)
        num_steps_to_recompute_policy = 10
        env = examples.double_reentrant_line_only_shared_resources_model(mu1=4., mu2=2., mu3=4.,
                                                                         mu4=2.)
        mpc_policy = FeedbackStationaryFeasibleMpcPolicy(env.physical_constituency_matrix,
                                                         env.job_generator.buffer_processing_matrix)
        avg_action = np.zeros((env.num_activities, 1))
        for _ in range(num_trials):
            avg_action += mpc_policy.generate_actions_with_feedback(sum_actions, state,
                                                                    num_steps_to_recompute_policy)
        avg_action /= num_trials
        np.testing.assert_almost_equal(avg_action, true_action, decimal=2)

    def test_generate_actions_with_feedback_all_buffers_filled(self):
        sum_actions = np.array([[4], [8], [2], [6]])
        state = 100 * np.ones((4, 1))
        num_trials = 10000
        true_action = np.array([[0.4], [0.8], [0.2], [0.6]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_one_buffer_empty(self):
        sum_actions = np.array([[4], [8], [2], [6]])
        state = np.array([[1], [1], [1], [0]])
        num_trials = 10000
        true_action = np.array([[1], [0.8], [0.2], [0]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_all_buffers_same_reource_empty(self):
        sum_actions = np.array([[4], [8], [2], [6]])
        state = np.array([[0], [1], [1], [0]])
        num_trials = 10000
        true_action = np.array([[0], [0.8], [0.2], [0]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_all_buffers_but_one_empty(self):
        sum_actions = np.array([[4], [8], [2], [6]])
        state = np.array([[0], [1], [0], [0]])
        num_trials = 10
        true_action = np.array([[0], [1], [0], [0]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_all_buffers_empty(self):
        sum_actions = np.array([[4], [8], [2], [6]])
        state = np.zeros((4, 1))
        num_trials = 10
        true_action = np.array([[0], [0], [0], [0]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_negative_some_sum_actions_but_positive_total(self):
        sum_actions = np.array([[-4], [8], [2], [6]])
        state = np.ones((4, 1))
        num_trials = 10000
        true_action = np.array([[0], [0.8], [0.2], [1]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_negative_some_sum_actions_but_positive_empty_buffer(self):
        sum_actions = np.array([[-4], [8], [2], [6]])
        state = np.array([[1], [1], [1], [0]])
        num_trials = 10000
        true_action = np.array([[1], [0.8], [0.2], [0]])
        self.perform_test_double_reentrant_line(sum_actions, state, true_action, num_trials)

    def test_generate_actions_with_feedback_klimov_model(self):
        np.random.seed(seed=42)
        num_steps_to_recompute_policy = 10
        env = examples.klimov_model()
        sum_actions = np.array([[2], [4], [2], [2]])
        state = np.array([[100], [0], [100], [100]])
        num_trials = 10000
        true_action = np.array([[1/3], [0], [1/3], [1/3]])
        mpc_policy = FeedbackStationaryFeasibleMpcPolicy(env.physical_constituency_matrix,
                                                         env.job_generator.buffer_processing_matrix)
        avg_action = np.zeros((env.num_activities, 1))
        for _ in range(num_trials):
            avg_action += mpc_policy.generate_actions_with_feedback(sum_actions, state,
                                                                    num_steps_to_recompute_policy)
        avg_action /= num_trials
        np.testing.assert_almost_equal(avg_action, true_action, decimal=2)
