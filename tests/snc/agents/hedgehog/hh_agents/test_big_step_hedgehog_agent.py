import numpy as np

from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
import snc.agents.hedgehog.minimal_draining_time as mdt
import snc.environments.examples as examples
from snc.demand_planning.constant_demand_plan import ConstantDemandPlan
from snc.simulation.utils.load_agents import build_bs_hedgehog_agent, get_hedgehog_hyperparams


class TestBigStepHedgehogAgent:

    @staticmethod
    def get_klimov_environment():
        return examples.klimov_model(alpha1=0.2, alpha2=.3, alpha3=.4, alpha4=.5, mu1=1.1,
                                     mu2=2.2, mu3=3.3, mu4=4.4)

    def perform_test_get_horizon_klimov(self, state, horizon_drain_time_ratio, penalty_grad,
                                        minimum_horizon, true_horizon):
        env = self.get_klimov_environment()
        horizon = BigStepHedgehogAgent.get_horizon(state=state, env=env,
                                                   horizon_drain_time_ratio=horizon_drain_time_ratio,
                                                   penalty_grad=penalty_grad,
                                                   minimum_horizon=minimum_horizon,
                                                   convex_solver="cvx.CPLEX")
        np.testing.assert_almost_equal(horizon, true_horizon)

    def test_get_horizon_klimov_zero_ratio(self):
        state = 1e6 * np.ones((4, 1))
        horizon_drain_time_ratio = 0
        penalty_grad = np.zeros((4, 1))
        minimum_horizon = 10
        true_horizon = minimum_horizon
        self.perform_test_get_horizon_klimov(state, horizon_drain_time_ratio, penalty_grad,
                                             minimum_horizon, true_horizon)

    def test_get_horizon_other_method(self):
        np.random.seed(42)
        state = np.random.random_sample((4, 1)) * 1000
        horizon_drain_time_ratio = 0.1
        penalty_grad = np.zeros((4, 1))
        minimum_horizon = 1
        env = self.get_klimov_environment()
        true_horizon = np.ceil(mdt.compute_minimal_draining_time_computing_workload_from_env(
            state, env, env.job_generator.demand_rate)
            * env.job_generator.sim_time_interval * horizon_drain_time_ratio)
        self.perform_test_get_horizon_klimov(state, horizon_drain_time_ratio, penalty_grad,
                                             minimum_horizon, true_horizon)

    def test_get_horizon_minimum_horizon_is_larger(self):
        np.random.seed(42)
        state = np.random.random_sample((4, 1)) * 1000
        horizon_drain_time_ratio = 0.1
        penalty_grad = np.zeros((4, 1))
        minimum_horizon = 900
        true_horizon = 900
        self.perform_test_get_horizon_klimov(state, horizon_drain_time_ratio, penalty_grad,
                                             minimum_horizon, true_horizon)

    def test_reset_mpc_variables(self):
        np.random.seed(42)
        env = self.get_klimov_environment()
        ha = build_bs_hedgehog_agent(env, 0.9, {})
        ha.current_policy = np.array([0.9, 0.1, 0, 0]).reshape(-1, 1)
        ha.num_steps_to_recompute_policy = 10
        ha.reset_mpc_variables()
        assert np.all(ha.mpc_variables["sum_actions"].ravel() == np.array([9, 1, 0, 0]))

    def test_update_mpc_variables(self):
        np.random.seed(42)
        env = self.get_klimov_environment()
        ha = build_bs_hedgehog_agent(env, 0.9, {})
        ha.current_policy = np.array([0.9, 0.1, 0, 0]).reshape(-1, 1)
        ha.num_steps_to_recompute_policy = 10
        ha.reset_mpc_variables()
        actions = np.array([1, 1, 0, 0]).reshape(-1, 1)
        ha.update_mpc_variables(actions=actions)
        assert np.all(ha.mpc_variables["sum_actions"].ravel() == np.array([8, 0, 0, 0]))
        assert np.all(ha.mpc_variables["total_sum_actions"].ravel() == np.array([1, 1, 0, 0]))
        assert np.all(ha.mpc_variables["total_fluid_sum_actions"] == ha.current_policy)
        assert ha.mpc_variables["total_timesteps"] == 1

        actions = np.array([1, 0, 0, 0]).reshape(-1, 1)
        ha.update_mpc_variables(actions=actions)
        assert np.all(ha.mpc_variables["sum_actions"].ravel() == np.array([7, 0, 0, 0]))
        assert np.all(ha.mpc_variables["total_sum_actions"].ravel() == np.array([2, 1, 0, 0]))
        assert np.all(ha.mpc_variables["total_fluid_sum_actions"] == 2 * ha.current_policy)
        assert ha.mpc_variables["total_timesteps"] == 2


def construct_env_and_big_step_hedgehog_agent(override_dp_params):
    env = examples.simple_reentrant_line_with_demand_model(job_gen_seed=42)
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_hyperparams(**override_dp_params)
    agent = BigStepHedgehogAgent(
        env,
        0.9999,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        po_params,
        si_class,
        dp_params,
        name
    )
    return env, agent


def check_constant_demand_planner_eq(dp1, dp2):
    assert type(dp1) == type(dp2) == ConstantDemandPlan
    assert dp1.__dict__ == dp2.__dict__


def test_constructor_demand_plan_class_push_model():
    dp_params = {
        "DemandPlanningParams": {
            "demand_planning_class_name": "ConstantDemandPlan",
            "params_dict": {
                "ind_surplus_buffers": [4],
                "demand_plan_values": {4: 10}
            }
        }
    }
    env, agent = construct_env_and_big_step_hedgehog_agent(dp_params)

    demand_planner = ConstantDemandPlan(ind_surplus_buffers=[4], demand_plan_values={4: 10})
    check_constant_demand_planner_eq(agent.demand_planner, demand_planner)
