import numpy as np
import pytest

import src.snc.environments.examples as examples
from src import snc as mw, snc as mw_lp, snc as smw, snc as ps
import src.snc.environments.controlled_random_walk as crw
import src.snc.environments.job_generators.discrete_review_job_generator \
    as jg
import src.snc.environments.state_initialiser as si


class TestGetGainActivityJ:
    @staticmethod
    def perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory):
        theta_j = mw.get_gain_activity_j(
            state, j, buffer_processing_matrix, cost_per_buffer)
        np.testing.assert_almost_equal(theta_j_theory, theta_j)

    def test_get_gain_activity_j_simple(self):
        state = np.array([[1], [1]])
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        cost_per_buffer = np.ones((2, 1))
        theta_j_theory = 0
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)

    def test_get_gain_activity_j_simple_higher_cost(self):
        state = np.array([[1], [1]])
        j = 0
        buffer_processing_matrix = np.array([[-1, 0], [1, -1]])
        cost_per_buffer = np.array([[2], [1]])
        theta_j_theory = 1
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)

    def test_get_gain_activity_j_simple_higher_rate(self):
        state = np.array([[4], [5]])
        j = 0
        buffer_processing_matrix = np.array([[-2, 0], [2, -1]])
        cost_per_buffer = np.ones((2, 1))
        theta_j_theory = -2
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)

    def test_get_gain_activity_j_simple_higher_cost_and_rate(self):
        state = np.array([[8], [5]])
        j = 0
        buffer_processing_matrix = np.array([[-2, 0], [2, -1]])
        cost_per_buffer = np.array([[3], [0.5]])
        theta_j_theory = 43
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)

    def test_get_gain_activity_j_simple_higher_cost_and_rate_swap_col_buffer_processing_mat(self):
        state = np.array([[8], [5]])
        j = 0
        buffer_processing_matrix = np.array([[0, -2], [-1, 2]])
        cost_per_buffer = np.array([[3], [0.5]])
        theta_j_theory = 2.5
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)
        i = 1
        theta_i_theory = 43
        self.perform_test(state, i, buffer_processing_matrix, cost_per_buffer, theta_i_theory)

    def test_get_gain_activity_j_simple_higher_rate_second_action(self):
        state = np.array([[8], [5]])
        j = 1
        buffer_processing_matrix = np.array([[-2, 0], [1, -3]])
        cost_per_buffer = np.array([[0.1], [0.5]])
        theta_j_theory = 7.5
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)

    def test_get_gain_activity_j_simple_higher_rate_second_action_swap_col_buffer_process_mat(self):
        state = np.array([[8], [5]])
        j = 1
        buffer_processing_matrix = np.array([[0, -2], [-3, 1]])
        cost_per_buffer = np.array([[0.1], [0.5]])
        theta_j_theory = -0.9
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)
        i = 0
        theta_i_theory = 7.5
        self.perform_test(state, i, buffer_processing_matrix, cost_per_buffer, theta_i_theory)

    def test_get_gain_activity_j_items_to_more_than_one_buffer(self):
        state = np.array([[8], [5], [3]])
        j = 1
        buffer_processing_matrix = np.array([[0, -2], [-3, 1], [2, 3]])
        cost_per_buffer = np.array([[0.1], [0.5], [0.2]])
        theta_j_theory = - 2.7
        self.perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory)
        i = 0
        theta_i_theory = 6.3
        self.perform_test(state, i, buffer_processing_matrix, cost_per_buffer, theta_i_theory)


class TestGetMaxGainStationS:
    @staticmethod
    def perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                     max_theta_theory, list_max_action_theory):
        max_theta, list_max_action = mw.get_max_gain_station_s(
            ind_actions_s, state, buffer_processing_matrix, cost_per_buffer)
        np.testing.assert_almost_equal(max_theta, max_theta_theory)
        assert list_max_action_theory == list_max_action

    def test_get_max_gain_station_s_0_simple_reentrant_line(self):
        state = np.array([[8], [5], [3]])
        buffer_processing_matrix = np.array([[-2, 0, 0],
                                             [2, -3, 0],
                                             [0, 3, -4]])
        cost_per_buffer = np.array([[0.1], [0.5], [0.8]])
        constituency_matrix = np.array([[1, 0, 1],
                                        [0, 1, 0]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 9.6
        list_max_action_theory = [2]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_s_0_simple_reentrant_line_swap_columns_buffer_process_mat(self):
        state = np.array([[8], [5], [3]])
        buffer_processing_matrix = np.array([[0, -2, 0],
                                             [0, 2, -3],
                                             [-4, 0, 3]])
        cost_per_buffer = np.array([[0.1], [0.5], [0.8]])
        constituency_matrix = np.array([[1, 1, 0],
                                        [0, 0, 1]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 9.6
        list_max_action_theory = [0]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_s_1_simple_reentrant_line(self):
        state = np.array([[8], [5], [3]])
        buffer_processing_matrix = np.array([[-2, 0, 0],
                                             [2, -3, 0],
                                             [0, 3, -4]])
        cost_per_buffer = np.array([[0.1], [0.5], [0.8]])
        constituency_matrix = np.array([[1, 0, 1],
                                        [0, 1, 0]])
        s = 1
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 0.3
        list_max_action_theory = [1]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_s_1_simple_reentrant_line_swap_columns_buffer_process_mat(self):
        state = np.array([[8], [5], [3]])
        buffer_processing_matrix = np.array([[0, -2, 0],
                                             [0, 2, -3],
                                             [-4, 0, 3]])
        cost_per_buffer = np.array([[0.1], [0.5], [0.8]])
        constituency_matrix = np.array([[1, 1, 0],
                                        [0, 0, 1]])
        s = 1
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 0.3
        list_max_action_theory = [2]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_s_0_simple_reentrant_line_two_max_actions(self):
        state = np.array([[8], [5], [1]])
        buffer_processing_matrix = np.array([[-2, 0, 0],
                                             [2, -3, 0],
                                             [0, 3, -2]])
        cost_per_buffer = np.array([[1], [0.5], [5.5]])
        constituency_matrix = np.array([[1, 0, 1],
                                        [0, 1, 0]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 11
        list_max_action_theory = [0, 2]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_s0_simple_reentrant_line_swap_columns_buffer_process_matrix(self):
        state = np.array([[8], [5], [1]])
        buffer_processing_matrix = np.array([[0, -2, 0],
                                             [0, 2, -3],
                                             [-2, 0, 3]])
        cost_per_buffer = np.array([[1], [0.5], [5.5]])
        constituency_matrix = np.array([[1, 1, 0],
                                        [0, 0, 1]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 11
        list_max_action_theory = [0, 1]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_routing(self):
        state = np.array([[8], [5], [1]])
        buffer_processing_matrix = np.array([[-2, -3],
                                             [5, 0],
                                             [0, 2]])
        cost_per_buffer = np.array([[0.2], [0.4], [0.3]])
        constituency_matrix = np.array([[1, 1]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 4.2
        list_max_action_theory = [1]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)

    def test_get_max_gain_station_routing_2_activities_with_same_gain(self):
        state = np.array([[8], [5], [5]])
        buffer_processing_matrix = np.array([[-2, -2],
                                             [1, 0],
                                             [0, 1]])
        cost_per_buffer = np.array([[0.2], [0.4], [0.4]])
        constituency_matrix = np.array([[1, 1]])
        s = 0
        ind_actions_s = np.argwhere(constituency_matrix[s] > 0)
        max_theta_theory = 1.2
        list_max_action_theory = [0, 1]
        self.perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                          max_theta_theory, list_max_action_theory)


class TestMaxWeightWeightPerBufferParameter:

    @staticmethod
    def perform_test_weight_per_buffer(env_model, num_buffers):
        state = np.random.random_sample((num_buffers, 1))
        weight_per_buffer = np.random.random_sample((num_buffers, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1))
        env_1 = env_model(cost_per_buffer=weight_per_buffer)
        agent_1 = mw.MaxWeightAgent(env_1)
        z_star_1 = agent_1.max_weight_policy(state)

        env_2 = env_model(cost_per_buffer=cost_per_buffer)
        agent_1 = mw.MaxWeightAgent(env_2, weight_per_buffer)
        z_star_2 = agent_1.max_weight_policy(state)
        np.testing.assert_almost_equal(z_star_1, z_star_2)

    def test_klimov_model_weight_per_buffer(self):
        self.perform_test_weight_per_buffer(examples.klimov_model, 4)

    def test_simple_reentrant_line_model_weight_per_buffer(self):
        self.perform_test_weight_per_buffer(examples.simple_reentrant_line_model, 3)

    def test_simple_routing_model_buffer3_weight_per_buffer(self):
        self.perform_test_weight_per_buffer(examples.simple_routing_model, 3)

    def test_weight_per_buffer_cost(self):
        cost_per_buffer = np.array([[1], [2], [3]])
        env = examples.simple_reentrant_line_model(cost_per_buffer=cost_per_buffer)
        agent = mw.MaxWeightAgent(env, weight_per_buffer='cost_per_buffer')
        assert np.all(agent.weight_per_buffer == cost_per_buffer)

    def test_weight_per_buffer_nonnegative(self):
        weight_per_buffer = np.array([[0], [2], [3]])
        env = examples.simple_reentrant_line_model()
        with pytest.raises(AssertionError):
            _ = mw.MaxWeightAgent(env, weight_per_buffer=weight_per_buffer)

    def test_weight_per_buffer_different_size_than_num_buffers(self):
        weight_per_buffer = np.ones((2, 1))
        env = examples.simple_reentrant_line_model()
        with pytest.raises(AssertionError):
            _ = mw.MaxWeightAgent(env, weight_per_buffer=weight_per_buffer)


class TestMaxWeightPolicySimulator:

    # tests for the initial asserts
    def test_assert(self):
        bpm = np.array([[-1, 1],
                        [-1, 0]])
        demand_rate = np.array([0.5, 0.7])[:, None]
        env_params = {
            "job_generator": jg.DiscreteReviewJobGenerator(sim_time_interval=0.1,
                                                           demand_rate=demand_rate,
                                                           buffer_processing_matrix=bpm),
            # the following parameters are dummy parameters not used in this test
            "cost_per_buffer": np.array([1, 1])[:, None],
            "capacity": np.array([1, 1])[:, None],
            "constituency_matrix": np.array([[1, 1],
                                             [1, 0]]),
            "state_initialiser": si.DeterministicCRWStateInitialiser(np.array([0.5, 0.7])[:, None]),
        }
        env = crw.ControlledRandomWalk(**env_params)
        pytest.raises(AssertionError, mw.MaxWeightAgent, env)

    @staticmethod
    def perform_test(state, env, z_star_theory):
        agent = mw.MaxWeightAgent(env)
        z_star = agent.max_weight_policy(state)
        np.testing.assert_almost_equal(z_star, z_star_theory)

    def test_klimov_model(self):
        state = np.array([[0], [1], [2], [0]])
        z_star_theory = np.array([[0], [0], [1], [0]])
        self.perform_test(state, examples.klimov_model(alpha1=0.2, alpha2=.3, alpha3=.4, alpha4=.5,
                                                       mu1=1.1, mu2=2.2, mu3=3.3, mu4=4.4,
                                                       cost_per_buffer=np.ones((4, 1)),
                                                       initial_state=(0, 0, 0, 0),
                                                       capacity=np.ones((4, 1)) * np.inf,
                                                       job_conservation_flag=True,
                                                       job_gen_seed=None,
                                                       max_episode_length=None),
                          z_star_theory)

    def test_simple_reentrant_line_model(self):
        state = np.array([[0], [0.5], [1]])
        z_star_theory = np.array([[0], [0], [1]])
        self.perform_test(state,
                          examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                                               cost_per_buffer=np.ones((3, 1)),
                                                               initial_state=np.array(
                                                                   [[0], [0], [0]]),
                                                               capacity=np.ones((3, 1)) * np.inf,
                                                               job_conservation_flag=True,
                                                               job_gen_seed=None,
                                                               max_episode_length=None),
                          z_star_theory)

    def test_simple_routing_model_buffer3_all_negative_theta_j(self):
        state = np.array([[8], [5], [1]])
        z_star_theory = np.array([[1], [1], [0], [0]])
        self.perform_test(state, examples.simple_routing_model(alpha_r=0.2,
                                                               mu1=0.13, mu2=0.07, mu_r=0.2,
                                                               cost_per_buffer=np.ones((3, 1)),
                                                               initial_state=(1, 1, 1),
                                                               capacity=np.ones((3, 1)) * np.inf,
                                                               job_conservation_flag=True,
                                                               job_gen_seed=None,
                                                               max_episode_length=None),
                          z_star_theory)

    def test_simple_routing_model_buffer3_all_positive_but_different_theta_j(self):
        state = np.array([[1], [5], [8]])
        z_star_theory = np.array([[1], [1], [1], [0]])
        self.perform_test(state, examples.simple_routing_model(alpha_r=0.2,
                                                               mu1=0.13, mu2=0.07, mu_r=0.2,
                                                               cost_per_buffer=np.ones((3, 1)),
                                                               initial_state=(1, 1, 1),
                                                               capacity=np.ones((3, 1)) * np.inf,
                                                               job_conservation_flag=True,
                                                               job_gen_seed=None,
                                                               max_episode_length=None),
                          z_star_theory)

    def test_simple_routing_model_buffer3_all_positive_and_equal_theta_j(self):
        state = np.array([[1], [1], [5]])
        z_star_theory = np.array([[1], [1], [1 / 2], [1 / 2]])
        self.perform_test(state, examples.simple_routing_model(alpha_r=0.2,
                                                               mu1=0.13, mu2=0.07, mu_r=0.2,
                                                               cost_per_buffer=np.ones((3, 1)),
                                                               initial_state=(1, 1, 1),
                                                               capacity=np.ones((3, 1)) * np.inf,
                                                               job_conservation_flag=True,
                                                               job_gen_seed=None,
                                                               max_episode_length=None),
                          z_star_theory)

    def test_integration_double_reentrant_line_model(self):
        seed = 42
        np.random.seed(seed)
        initial_state = 50 * np.ones((5, 1))
        env = examples.double_reentrant_line_model(alpha=1, mu1=4, mu2=3, mu3=2, mu4=3, mu5=4,
                                                   cost_per_buffer=np.array([1, 1, 1, 1, 1])
                                                   [:, None],
                                                   initial_state=initial_state,
                                                   capacity=np.ones((5, 1)) * np.inf,
                                                   job_conservation_flag=True,
                                                   job_gen_seed=seed,
                                                   max_episode_length=None)
        agent = mw.MaxWeightAgent(env)
        simulator = ps.SncSimulator(env, agent, discount_factor=0.95)
        data_mw = simulator.run(num_simulation_steps=1000)
        assert np.all(data_mw['state'][-1] < initial_state)

    @staticmethod
    def assert_both_mw_policies_are_equivalent(state, cost_per_buffer, buffer_processing_matrix,
                                               z_star_mw_lp, z_star_mw):
        diag_cost = np.diag(cost_per_buffer[:, -1])
        opt_val_mw_lp = state.T @ diag_cost @ buffer_processing_matrix @ z_star_mw_lp
        opt_val_mw = state.T @ diag_cost @ buffer_processing_matrix @ z_star_mw
        np.testing.assert_almost_equal(opt_val_mw_lp, opt_val_mw, decimal=5)

    def test_compare_maxweight_vs_lp_maxweight_double_reentrant_line_model(self):
        seed = 42
        np.random.seed(seed)
        initial_state = 50 * np.ones((5, 1))
        env = examples.double_reentrant_line_model(alpha=1, mu1=4, mu2=3, mu3=2, mu4=3, mu5=4,
                                                   cost_per_buffer=np.array([1, 1, 1, 1, 1])
                                                   [:, None],
                                                   initial_state=initial_state,
                                                   capacity=np.ones((5, 1)) * np.inf,
                                                   job_conservation_flag=True,
                                                   job_gen_seed=seed,
                                                   max_episode_length=None)
        mw_agent = mw.MaxWeightAgent(env)
        mw_lp_agent = mw_lp.MaxWeightLpAgent(env)

        states = np.random.randint(50, size=(400, 5))
        for s in states:
            z_star_mw_lp, _ = mw_lp_agent.max_weight_policy(s[:, None])
            z_star_mw = mw_agent.max_weight_policy(s[:, None])
            self.assert_both_mw_policies_are_equivalent(s, env.cost_per_buffer,
                                                        env.job_generator.buffer_processing_matrix,
                                                        z_star_mw_lp, z_star_mw)

    def test_compare_maxweight_vs_lp_maxweight_simple_routing_model(self):
        seed = 42
        np.random.seed(seed)
        env = examples.simple_routing_model(alpha_r=0.2,
                                            mu1=0.13, mu2=0.07, mu_r=0.2,
                                            cost_per_buffer=np.ones((3, 1)),
                                            initial_state=(1, 1, 1),
                                            capacity=np.ones((3, 1)) * np.inf,
                                            job_conservation_flag=True,
                                            job_gen_seed=seed,
                                            max_episode_length=None)
        mw_agent = mw.MaxWeightAgent(env)
        mw_lp_agent = mw_lp.MaxWeightLpAgent(env)

        states = np.random.randint(50, size=(400, 3))
        for s in states:
            z_star_mw_lp, _ = mw_lp_agent.max_weight_policy(s[:, None])
            z_star_mw = mw_agent.max_weight_policy(s[:, None])
            self.assert_both_mw_policies_are_equivalent(s, env.cost_per_buffer,
                                                        env.job_generator.buffer_processing_matrix,
                                                        z_star_mw_lp, z_star_mw)

    def test_compare_maxweight_vs_maxweight_scheduling_double_reentrant_line_model(self):
        seed = 42
        np.random.seed(seed)
        initial_state = 50 * np.ones((5, 1))
        env = examples.double_reentrant_line_model(alpha=1, mu1=4, mu2=3, mu3=2, mu4=3, mu5=4,
                                                   cost_per_buffer=np.ones((5, 1)),
                                                   initial_state=initial_state,
                                                   capacity=np.ones((5, 1)) * np.inf,
                                                   job_conservation_flag=True,
                                                   job_gen_seed=seed,
                                                   max_episode_length=None)
        smw_agent = smw.SchedulingMaxWeightAgent(env)
        mw_agent = mw.MaxWeightAgent(env)

        states = np.random.randint(50, size=(400, 5))
        for s in states:
            z_star_mw = mw_agent.max_weight_policy(s[:, None])
            z_star_smw = smw_agent.scheduling_max_weight_policy(s[:, None])
            self.assert_both_mw_policies_are_equivalent(s, env.cost_per_buffer,
                                                        env.job_generator.buffer_processing_matrix,
                                                        z_star_mw, z_star_smw)

    def test_compare_maxweight_vs_maxweight_scheduling_simple_routing_model(self):
        seed = 42
        np.random.seed(seed)
        env = examples.simple_routing_model(alpha_r=0.2,
                                            mu1=0.13, mu2=0.07, mu_r=0.2,
                                            cost_per_buffer=np.ones((3, 1)),
                                            initial_state=(1, 1, 1),
                                            capacity=np.ones((3, 1)) * np.inf,
                                            job_conservation_flag=True,
                                            job_gen_seed=seed,
                                            max_episode_length=None)
        smw_agent = smw.SchedulingMaxWeightAgent(env)
        mw_agent = mw.MaxWeightAgent(env)

        states = np.random.randint(50, size=(400, 3))
        for s in states:
            z_star_mw = mw_agent.max_weight_policy(s[:, None])
            z_star_smw = smw_agent.scheduling_max_weight_policy(s[:, None])
            self.assert_both_mw_policies_are_equivalent(s, env.cost_per_buffer,
                                                        env.job_generator.buffer_processing_matrix,
                                                        z_star_mw, z_star_smw)
