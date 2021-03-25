import numpy as np
import snc.environments.examples as examples
import snc.agents.maxweight_variants.maxweight_lp as mw_lp
import snc.simulation.snc_simulator as ps


class TestGetActionsWorkingOnEmptyBuffers:
    @staticmethod
    def perform_test(state, action, buffer_processing_matrix, ind_actions_theory):
        ind_actions = mw_lp.get_actions_working_on_empty_buffers(
            state, action, buffer_processing_matrix)
        assert ind_actions_theory == ind_actions

    def test_assert_not_working_on_empty_buffers_not_draining_actions(self):
        """It is always true because we never drain the buffers."""
        self.perform_test(np.array([[0], [1], [2], [0]]), np.ones((4, 1)), np.eye(4), [])

    def test_assert_not_working_on_empty_buffers(self):
        self.perform_test(np.array([[0], [1], [2], [0]]), np.array([[0], [1], [1], [0]]),
                          - np.eye(4), [])

    def test_assert_not_working_on_empty_buffers_one_fails(self):
        self.perform_test(np.array([[0], [1], [2], [0]]), np.array([[0], [1], [1], [1]]),
                          - np.eye(4), [3])

    def test_assert_not_working_on_empty_buffers_two_actions_drain(self):
        self.perform_test(np.array([[0], [1]]), np.array([[0], [0]]), np.array([[-1, -1], [1, 1]]),
                          [])

    def test_assert_not_working_on_empty_buffers_two_actions_drain_fails(self):
        self.perform_test(np.array([[0], [1], [2], [0]]), np.array([[0], [0], [0], [1]]),
                          np.array([[-1, -1, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 1],
                                    [0, 0, -1, -1]]), [3])


class TestSetActionDrainingEmptyBufferToZero:
    @staticmethod
    def perform_test(opt_val, z_star, state, buffer_processing_matrix, diag_cost, z_star_theory):
        z_star_zeroed = mw_lp.set_action_draining_empty_buffer_to_zero(
            opt_val, z_star, state, buffer_processing_matrix, diag_cost)
        assert np.all(z_star_zeroed == z_star_theory)

    def test_set_action_draining_empty_buffer_to_zero_no_empty_buffers(self):
        opt_val = -2
        z_star = np.array([[1], [1]])
        state = np.array([[1], [1]])
        z_star_theory = np.array([[1], [1]])
        self.perform_test(opt_val, z_star, state, -1 * np.eye(2), np.eye(2), z_star_theory)

    def test_set_action_draining_empty_buffer_to_zero_buffer_one_empty(self):
        opt_val = -1
        z_star = np.array([[1], [1]])
        state = np.array([[0], [1]])
        z_star_theory = np.array([[0], [1]])
        self.perform_test(opt_val, z_star, state, -1 * np.eye(2), np.eye(2), z_star_theory)

    def test_set_action_draining_empty_buffer_to_zero_all_buffers_empty(self):
        opt_val = 0
        z_star = np.array([[1], [1]])
        state = np.array([[0], [0]])
        z_star_theory = np.array([[0], [0]])
        self.perform_test(opt_val, z_star, state, -1 * np.eye(2), np.eye(2), z_star_theory)

    def test_set_action_draining_empty_buffer_to_zero_double_reentrant_line_model(self):
        """
        The output of the LP for this example gives nonzero output to action draining buffer 5,
        which is empty.
        """
        env = examples.double_reentrant_line_model(alpha=0.25, mu1=1, mu2=0.75, mu3=0.5, mu4=0.75,
                                                   mu5=1, initial_state=5 * np.ones((5, 1)))
        opt_val = -21.25
        z_star = np.array([[1], [0], [1], [1], [1]])
        state = np.array([[54], [54], [29], [27], [0]])
        diag_cost = np.eye(5)
        z_star_zeroed = mw_lp.set_action_draining_empty_buffer_to_zero(
            opt_val, z_star, state, env.job_generator.buffer_processing_matrix, diag_cost)
        assert np.all(z_star_zeroed[:-1] == z_star[:-1])
        assert z_star_zeroed[4] == 0


class TestMaxWeightPolicySimulator:
    @staticmethod
    def perform_test_max_weight_policy(state, env, z_star_theory):
        agent = mw_lp.MaxWeightLpAgent(env)
        z_star, _ = agent.max_weight_policy(state)
        np.testing.assert_almost_equal(z_star, z_star_theory)

    def test_klimov_model(self):
        state = np.array([[0], [1], [2], [0]])
        z_star_theory = np.array([[0], [0], [1], [0]])
        self.perform_test_max_weight_policy(state,
                                            examples.klimov_model(alpha1=0.2, alpha2=.3, alpha3=.4,
                                                                  alpha4=.5,
                                                                  mu1=1.1, mu2=2.2, mu3=3.3,
                                                                  mu4=4.4,
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
        self.perform_test_max_weight_policy(state,
                                            examples.simple_reentrant_line_model(
                                                alpha1=9, mu1=22, mu2=10, mu3=22,
                                                cost_per_buffer=np.ones((3, 1)),
                                                initial_state=np.array([[0], [0], [0]]),
                                                capacity=np.ones((3, 1)) * np.inf,
                                                job_conservation_flag=True,
                                                job_gen_seed=None,
                                                max_episode_length=None),
                                            z_star_theory)

    def test_simple_routing_model_buffer3_all_negative_theta_j(self):
        state = np.array([[8], [5], [1]])
        z_star_theory = np.array([[1], [1], [0], [0]])
        self.perform_test_max_weight_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
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
        self.perform_test_max_weight_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
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
        self.perform_test_max_weight_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
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
        agent = mw_lp.MaxWeightLpAgent(env)
        simulator = ps.SncSimulator(env, agent, discount_factor=0.95)
        data = simulator.run(num_simulation_steps=1000)
        assert np.all(data['state'][-1] < initial_state)

    @staticmethod
    def perform_test_max_weight_binary_policy(state, env, z_star_theory,
                                              method, binary_action):
        agent = mw_lp.MaxWeightLpAgent(env, method=method, binary_action=binary_action)
        z_star, _ = agent.max_weight_policy(state)
        equal = 0
        for z in z_star_theory:
            if np.allclose(z, z_star):
                equal = equal + 1
        assert equal == 1

    def test_simple_routing_model_buffer3_all_negative_theta_j_binary(self):
        state = np.array([[8], [5], [1]])
        z_star_theory = np.array([[[1], [1], [0], [0]]])
        self.perform_test_max_weight_binary_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
            mu1=0.13, mu2=0.07, mu_r=0.2,
            cost_per_buffer=np.ones((3, 1)),
            initial_state=(1, 1, 1),
            capacity=np.ones((3, 1)) * np.inf,
            job_conservation_flag=True,
            job_gen_seed=None,
            max_episode_length=None),
                                                   z_star_theory, method='cvx.ECOS_BB',
                                                   binary_action=True)

    def test_simple_routing_model_buffer3_all_positive_but_different_theta_j_binary(self):
        state = np.array([[1], [5], [8]])
        z_star_theory = np.array([[[1], [1], [1], [0]]])
        self.perform_test_max_weight_binary_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
            mu1=0.13, mu2=0.07, mu_r=0.2,
            cost_per_buffer=np.ones((3, 1)),
            initial_state=(1, 1, 1),
            capacity=np.ones((3, 1)) * np.inf,
            job_conservation_flag=True,
            job_gen_seed=None,
            max_episode_length=None),
                                                   z_star_theory, method='cvx.ECOS_BB',
                                                   binary_action=True)

    def test_simple_routing_model_buffer3_all_positive_and_equal_theta_j_binary(self):
        state = np.array([[1], [1], [5]])
        z_star_theory = np.array([[[1], [1], [1], [0]], [[1], [1], [0], [1]]])
        self.perform_test_max_weight_binary_policy(state, examples.simple_routing_model(
            alpha_r=0.2,
            mu1=0.13, mu2=0.07, mu_r=0.2,
            cost_per_buffer=np.ones((3, 1)),
            initial_state=(1, 1, 1),
            capacity=np.ones((3, 1)) * np.inf,
            job_conservation_flag=True,
            job_gen_seed=None,
            max_episode_length=None),
                                                   z_star_theory, method='cvx.ECOS_BB',
                                                   binary_action=True)
