import numpy as np
from typing import Dict, Optional

from snc.agents.activity_rate_to_mpc_actions.feedback_stationary_feasible_mpc_policy import \
    FeedbackStationaryFeasibleMpcPolicy
from snc.agents.agent_interface import AgentInterface
from snc.agents.hedgehog.asymptotic_workload_cov.compute_asymptotic_cov_interface \
    import ComputeAsymptoticCovInterface
from snc.environments import controlled_random_walk as crw
from snc.utils import snc_types as types


class SteadyStatePolicyAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk,
                 mpc_policy_class=FeedbackStationaryFeasibleMpcPolicy,
                 horizon: int = 200,
                 name: str = "SteadyStatePolicyAgent",
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        Draw activities from the distribution given by the steady state policy, which is computed
        as a function of the buffer processing matrix and the demand rate.

        :param env: the environment to stepped through.
        :param mpc_policy_class: Class to create an MPC policy object for this agent.
        :param horizon: Horizon to track the steady-state activity rates. This is implemented as a
            countdown. When the counter gets to zero, it resets and the remaining sum of actions
            resets too, meaning that negative actions are set to zero. This reset helps to forget.
            If horizon is small, we force to track the activity rates in the short term,
            as resets excessive rate of an activity due to empty buffers of other activities of the
            same resource. But this forgetting may make the long term rate different from the
            desired steady-state policy. If horizon is large, the activity rates will track the
            desired steady-state policy in the long term, but the short term activity rates could be
            different from the steady state policy due to empty buffers.
        :param name: Agent identifier.
        :param agent_seed: Random seed to be used in setting up the agent's numpy random state.
        :param mpc_seed: Random seed for MPC policy's random numpy random state.
        """
        super().__init__(env, name, agent_seed)

        self.mpc_policy = mpc_policy_class(
            env.physical_constituency_matrix,
            env.job_generator.buffer_processing_matrix,
            mpc_seed
        )
        self.policy = ComputeAsymptoticCovInterface.compute_steady_state_policy(
            env.job_generator.buffer_processing_matrix,
            env.job_generator.demand_rate,
            env.constituency_matrix
        )

        self.mpc_variables: Dict[str, np.ndarray] = dict()

        assert horizon > 10, f"If horizon < 10, we run the risk of rounding sum_actions to zero, " \
                             f"but provided: {horizon}."
        self.horizon = horizon
        self.num_steps_to_recompute_policy = 0

    def reset_mpc_variables(self) -> None:
        self.mpc_variables["sum_actions"] = np.round(max(1, self.num_steps_to_recompute_policy)
                                                     * self.policy)

    def update_mpc_variables(self, actions) -> None:
        self.mpc_variables["sum_actions"] -= actions

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Dict) \
            -> types.ActionProcess:
        """
        Returns action giving by a nonidling policy, such that each resources acts if the
        boundary conditions on its buffers are satisfied for the given safety_stock, and the
        action is chosen randomly among any of its possible activities.

        :param state: Current state of the system
        :return action: action vector
        """

        if self.num_steps_to_recompute_policy == 0:
            self.num_steps_to_recompute_policy = self.horizon
            self.reset_mpc_variables()

        # Obtain physically feasible actions from MPC policy.
        actions = self.mpc_policy.obtain_actions(
            state=state,
            x_star = state,
            x_eff = state,
            r_idling_set = np.array([]),
            draining_resources = set(),
            mpc_variables=self.mpc_variables,
            num_steps_to_recompute_policy=self.num_steps_to_recompute_policy,
            z_star=self.policy, demand_rate=self.env.job_generator.demand_rate)

        self.update_mpc_variables(actions)
        self.num_steps_to_recompute_policy -= 1
        return actions
