import numpy as np

import snc.environments.examples as examples
import snc.agents.maxweight_variants.scheduling_maxweight_agent as sch_mw
import snc.agents.maxweight_variants.maxweight_lp as mw_lp
import snc.simulation.snc_simulator as ps


class TestGetGainBufferDrainedByActivityJ:
    @staticmethod
    def perform_test(state, j, buffer_processing_matrix, cost_per_buffer, theta_j_theory):
        theta_j = sch_mw.get_gain_buffer_drained_by_activity_j(
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


class TestGetMaxGainStationS:
    @staticmethod
    def perform_test(ind_actions_s, state, buffer_processing_matrix, cost_per_buffer,
                     max_theta_theory, list_max_action_theory):
        max_theta, list_max_action = sch_mw.get_max_gain_station_s(
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


class TestSchedulingMaxWeightPolicySimulator:
    @staticmethod
    def perform_test(state, env, z_star_theory):
        agent = sch_mw.SchedulingMaxWeightAgent(env)
        z_star = agent.scheduling_max_weight_policy(state)
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
        agent = sch_mw.SchedulingMaxWeightAgent(env)
        simulator = ps.SncSimulator(env, agent, discount_factor=0.95)
        data_smw = simulator.run(num_simulation_steps=1000)
        assert np.all(data_smw['state'][-1] < initial_state)

    @staticmethod
    def assert_both_mw_policies_are_equivalent(state, cost_per_buffer, buffer_processing_matrix,
                                               z_star_mw, z_star_smw):
        diag_cost = np.diag(cost_per_buffer[:, -1])
        opt_val_mw = state.T @ diag_cost @ buffer_processing_matrix @ z_star_mw
        opt_val_smw = state.T @ diag_cost @ buffer_processing_matrix @ z_star_smw
        np.testing.assert_almost_equal(opt_val_mw, opt_val_smw, decimal=5)

    def test_compare_maxweight_vs_maxweight_scheduling(self):
        seed = 42
        np.random.seed(seed)
        initial_state = 50 * np.ones((5, 1))
        env = examples.double_reentrant_line_model(initial_state=initial_state,
                                                   job_gen_seed=seed)
        sch_mw_agent = sch_mw.SchedulingMaxWeightAgent(env)
        mw_agent = mw_lp.MaxWeightLpAgent(env)

        states = np.random.randint(50, size=(400, 5))
        for s in states:
            z_star_mw, _ = mw_agent.max_weight_policy(s[:, None])
            z_star_sch_mw = sch_mw_agent.scheduling_max_weight_policy(s[:, None])
            self.assert_both_mw_policies_are_equivalent(s, env.cost_per_buffer,
                                                        env.job_generator.buffer_processing_matrix,
                                                        z_star_mw, z_star_sch_mw)
