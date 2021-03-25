import numpy as np

import snc.utils.snc_types as types

import snc.agents.activity_rate_to_mpc_actions.mpc_utils as mpc_utils
import snc.agents.maxweight_variants.scheduling_maxweight_agent as sch_mw
from snc.agents.activity_rate_to_mpc_actions.action_mpc_policy import ActionMPCPolicy
from snc.agents.activity_rate_to_mpc_actions.stationary_mpc_policy import StationaryActionMPCPolicy


class DifferentialBackPressureMPC(ActionMPCPolicy):
    def __init__(self, env) -> None:
        super().__init__(env.physical_constituency_matrix)

        self.draining_jobs_rate_matrix = env.job_generator.draining_jobs_rate_matrix
        self.demand_rate = env.job_generator.demand_rate
        self.num_resources = env.num_resources

        self.activities_per_resource, self.num_activities_per_resource \
            = mpc_utils.get_ind_activities_per_resource(self.physical_constituency_matrix)

        self.ind_activities_drain_buffers = \
            mpc_utils.get_ind_activities_drain_each_buffer(self.draining_jobs_rate_matrix)

        self.resource_per_activity = \
            mpc_utils.get_resource_each_activity_belongs(self.physical_constituency_matrix)

        mw_mpc_policy = StationaryActionMPCPolicy(
            physical_constituency_matrix=env.physical_constituency_matrix)

        cost_per_buffer = np.ones(env.cost_per_buffer.shape[0])[:, None]
        self.agent_smw = sch_mw.SchedulingMaxWeightAgent(env, mw_mpc_policy, cost_per_buffer)

    @staticmethod
    def get_active_actions_for_resource_r(activities_r, active_actions):
        return [a for a in activities_r if a in active_actions]

    def build_schedule(self, z_star: types.ActionSpace, sum_actions: types.ActionSpace,
                       state: types.StateSpace):
        threshold = np.minimum(state, - self.draining_jobs_rate_matrix @ sum_actions)
        actions = self.agent_smw.map_state_to_actions(threshold)
        assert np.all(self.physical_constituency_matrix @ actions <= 1)
        return actions

    def obtain_actions(self, z_star: types.ActionSpace, **args) \
            -> types.ActionProcess:
        """
        Return actions dealing with each resource independently.
        :param z_star: Policy to be performed during the given number of time steps.
        :param args: Dictionary that contains:
            - state: Current state, needed to check if some buffers are empty.
            - num_mpc_steps: Number of steps for which we should compute actions.
            - new_policy_flag: Flag to indicate that the policy has changed, so we can reset the
                counter that aims to track the distribution from the previous fluid policy.
        :return: actions: binary indicator matrix with number of rows equal to the number of
            non-idling activities (i.e. num of columns of the constituency matrix), and number of
            columns equal to number of time steps to perform MPC.
        """
        state = args['state']
        sum_actions = args['sum_actions']

        num_mpc_steps = args['num_mpc_steps']
        mpc_utils.check_num_time_steps(num_mpc_steps)
        assert num_mpc_steps == 1  # Current version only works for pure feedback.

        z_star_now = z_star.copy()

        actions = self.build_schedule(z_star_now, sum_actions, state)

        print(f"z_star_now : {np.squeeze(z_star_now)}")

        return actions
