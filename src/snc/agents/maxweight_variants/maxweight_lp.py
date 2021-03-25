from typing import Optional, Tuple, List, Any
import numpy as np
import cvxpy as cvx
from snc.utils import snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.agents.agent_interface import AgentInterface
from snc.agents.activity_rate_to_mpc_actions.stationary_mpc_policy \
    import StationaryActionMPCPolicy


def get_actions_working_on_empty_buffers(state: types.StateSpace,
                                         action: types.ActionSpace,
                                         buffer_processing_matrix: types.BufferMatrix,
                                         eps: float = 1e-6) -> List[int]:
    """
    Get list of indexes of actions with positive value that would work on empty buffers.

    :param state: Current state of the environment.
    :param action: Action vector with positive actions that would drain empty buffers.
    :param buffer_processing_matrix: From job generator (the environment).
    :param eps: Tolerance to state that a value is positive.
    :return: List of indexes of actions poised to work on empty buffers.
    """
    ind_empty_buffers = np.argwhere(state == 0)
    ind_actions_drain_empty_buffers = []
    for i, _ in ind_empty_buffers:
        ind_actions_drain_i = np.argwhere(buffer_processing_matrix[i] < 0)
        for j in ind_actions_drain_i:
            if np.abs(action[j]) > eps:
                ind_actions_drain_empty_buffers.append(j[0])
    return ind_actions_drain_empty_buffers


def set_action_draining_empty_buffer_to_zero(opt_val: float,
                                             z_star: types.ActionSpace,
                                             state: types.StateSpace,
                                             buffer_processing_matrix: types.BufferMatrix,
                                             diag_cost: np.ndarray) -> types.ActionSpace:
    """
    Set positive actions that would work on empty buffers to zero. We also check that the new
    actions (that don't drain empty buffers) are also solutions to the LP.

    :param opt_val: Optimal value resulting from solving the LP.
    :param z_star: Solution provided by the LP solver.
    :param state: Current state of the environment.
    :param buffer_processing_matrix: From job generator (the environment).
    :param diag_cost: Diagonal matrix with entries given by the cost vector.
    :return: Action vector with zero entries in each component that could drain an empty buffer.
    """
    ind_actions_drain_empty_buffers \
        = get_actions_working_on_empty_buffers(state, z_star, buffer_processing_matrix)
    if ind_actions_drain_empty_buffers:
        z_star[ind_actions_drain_empty_buffers] = 0
        new_opt_val = state.T @ diag_cost @ buffer_processing_matrix @ z_star
        np.testing.assert_almost_equal(new_opt_val, opt_val)
    return z_star


class MaxWeightLpAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk,
                 method: str = 'cvx.ECOS',
                 name: str = "MaxWeightLpAgent",
                 binary_action: Optional[bool] = False,
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        MaxWeight policy for general push models.

        :param env: CRW environment.
        :param method: Optimisation method/solver that CVXPY should use to solve the LP.
        :param name: Agent identifier.
        :param binary_action: where the problem is an MIP with binary variables or not.
        :param agent_seed: Agent random seed.
        :param mpc_seed: MPC random seed.
        :return None.
        """
        super().__init__(env, name, agent_seed)
        self.mpc_policy = StationaryActionMPCPolicy(env.physical_constituency_matrix, mpc_seed)
        self.method = method
        self.binary_action = binary_action

    def max_weight_policy(self, state: types.StateSpace) -> Tuple[types.ActionSpace, float]:
        """
        MaxWeight policy expressed as the solution to an LP:
            z_star = arg min -x.T DB z
                        s.t. Cz <= 1, z >= 0.

        :param state: Current state of the environment.
        :return: (z_star, opt_val)
            - z_star: solution to the LP provided by the solver.
            - opt_val: optimal value of the LP.
        """
        _, num_activities = self.constituency_matrix.shape
        z = cvx.Variable((num_activities, 1), boolean=self.binary_action)
        diag_cost = np.diag(self.env.cost_per_buffer[:, -1])
        objective = cvx.Minimize(state.T @ diag_cost @ self.buffer_processing_matrix @ z)
        constraints = [
            self.env.constituency_matrix * z <= np.ones((self.env.num_resources, 1)),
            z >= np.zeros((self.env.num_activities, 1))
        ]
        prob = cvx.Problem(objective, constraints)
        opt_val = prob.solve(solver=eval(self.method))
        z_star = z.value
        z_star = set_action_draining_empty_buffer_to_zero(opt_val, z_star, state,
                                                          self.buffer_processing_matrix, diag_cost)
        return z_star, opt_val

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Converts the continuous action vector to a set of binary actions.

        :param state: Current state of the environment.
        :param override_args: Arguments that can be overriden.
        :return: Set of binary actions.
        """
        z_star, _ = self.max_weight_policy(state)
        actions = self.mpc_policy.obtain_actions(num_mpc_steps=1, z_star=z_star)
        return actions
