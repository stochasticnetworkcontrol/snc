import cvxpy as cvx
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from snc.agents.hedgehog.params import BigStepPenaltyPolicyParams
from snc.agents.hedgehog.policies.big_step_base_policy import BigStepBasePolicy
import snc.agents.hedgehog.policies.policy_utils as policy_utils
from snc.agents.solver_names import SolverNames
import snc.utils.snc_types as types


class BigStepPolicy(BigStepBasePolicy):

    def __init__(
            self,
            cost_per_buffer: types.StateSpace,
            constituency_matrix: types.ConstituencyMatrix,
            demand_rate: types.StateSpace,
            buffer_processing_matrix: types.BufferMatrix,
            workload_mat: types.WorkloadMatrix,
            nu: types.NuMatrix,
            list_boundary_constraint_matrices: List[np.ndarray],
            ind_surplus_buffers: Optional[List[int]],
            policy_params: Optional[BigStepPenaltyPolicyParams] = None,
            debug_info: float = False
    ) -> None:
        """
        Creates big step policy whose main method 'get_policy' returns a schedule that is
        approximately optimal in the discounted cost sense for the number of time steps given
        by `horizon`, when starting at a given state. Note that 'horizon' is the size of the step
        of the big-step policy.

        :param cost_per_buffer: from the environment.
        :param constituency_matrix: from the environment.
        :param demand_rate: from the environment
        :param buffer_processing_matrix: from job generator (the environment).
        :param workload_mat: workload matrix.
        :param nu: matrix with number of rows equal to number of workload vectors, and number of
            columns equal to the number of physical resources. Each row indicates for which
            resources the constraint C z <= 1 is active, i.e. it indicates the set of bottlenecks.
        :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
            indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
        :param ind_surplus_buffers: List of integers indicating the location of the surplus
            buffers in the buffer processing matrix.
        :param policy_params: Named tuple with policy parameters.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        """

        super().__init__(
            cost_per_buffer,
            constituency_matrix,
            demand_rate,
            buffer_processing_matrix,
            workload_mat,
            nu,
            list_boundary_constraint_matrices,
            policy_params,
            debug_info
        )

        self.pull = True if ind_surplus_buffers is not None else False
        self.ind_surplus_buffers = ind_surplus_buffers

        (
            self.boolean_action_flag,
            self.convex_solver,
            self.kappa,
            self.kappa_w
        ) = self.get_policy_params(policy_params)

        (
            self._lp_problem,
            self._state,
            self._horizon,
            self._workload_buffer_mat,
            self._ones_vec_nonidling,
            self._safety_stocks_vec,
            self._allowed_activities,
            self._weight_nonidling_tol,
            self._surplus_target_vec,
            self._z,
            self._safety_stock_tol,
            self._nonidling_tol,
            self._surplus_tol
        ) = self.create_big_step_policy_nonidling_penalty_lp_program()

    @staticmethod
    def get_policy_params(policy_params: BigStepPenaltyPolicyParams) \
            -> Tuple[float, str, float, bool]:
        """
        Extracts parameters from `BigStepPenaltyPolicyParams` named tuple, namely boolean flag
        indicating binary pure-feedback actions, convex solver, nonidling penalty coefficient, and
        safety stock penalty coefficient.

        :param policy_params: Named tuple with policy parameters.
        :return: (kappa, kappa_w, boolean_action_flag)
            - boolean_action_flag: Boolean flag indicating if the solution of each pure feedback
                policy synthesis optimisation problem should be a real or a binary vector.
            - convex_solver: String indicating which solver should be used.
            - kappa: Nonidling penalty coefficient.
            - kappa_w: Safety stock penalty coefficient.
        """
        assert hasattr(policy_params, "boolean_action_flag"), \
            "Policy parameters must include pure-feedback boolean action flag: " \
            "'boolean_action_flag'."
        boolean_action_flag = policy_params.boolean_action_flag
        assert boolean_action_flag in [True, False], \
            f"boolean_action_flag must be a boolean variable, but provided: {boolean_action_flag}."

        assert hasattr(policy_params, "convex_solver"), \
            "Policy parameters must include 'convex_solver'."
        convex_solver = policy_params.convex_solver
        assert convex_solver in SolverNames.CVX, \
            f"Convex solver must be in {SolverNames.CVX}, but provided: {convex_solver}."

        assert hasattr(policy_params, "nonidling_penalty_coeff"), \
            "Policy parameters must include nonidling penalty coefficient: 'kappa_w'."
        kappa_w = policy_params.nonidling_penalty_coeff
        assert kappa_w >= 0, f"Nonidling penalty coefficient 'kappa_w' must be nonnegative, " \
                             f"but provided: {kappa_w}."

        assert hasattr(policy_params, "safety_penalty_coeff"), \
            "Policy parameters must include safety stock penalty coefficient 'kappa'."
        kappa = policy_params.safety_penalty_coeff
        assert kappa >= 0, f"Safety stocks penalty coefficient 'kappa' must be nonnegative, " \
                           f"but provided: {kappa}."

        return boolean_action_flag, convex_solver, kappa, kappa_w

    def get_weight_nonidling_tol(self, draining_bottlenecks) -> types.WorkloadSpace:
        """
        Returns weights for penalty associated to nonidling resources distinguishing bottlenecks,
        which cannot idle in order for the system to be able to drain the network, from other
        workload resources that cannot idle in order to reduce the cost. In particular, the weight
        of the former (draining bottlenecks) is kappa_w, while for the others is just one.

        :param draining_bottlenecks: Set of draining bottlenecks.
        :return weight_nonidling_tol: Array of positive weights.
        """
        weight_nonidling_tol = np.ones((self.num_wl_vec, 1))
        weight_nonidling_tol[list(draining_bottlenecks)] = self.kappa_w
        return weight_nonidling_tol

    def create_big_step_policy_nonidling_penalty_lp_program(self):
        """
        Creates cvxpy optimisation problem and return parameters and variables.
        """
        # Variables.
        if self.boolean_action_flag:
            z_options = {'boolean': True}
        else:
            z_options = {'nonneg': True}
        z = cvx.Variable((self.num_activities, 1), **z_options)
        safety_stock_tol = cvx.Variable((self.num_resources, 1), nonneg=True)
        nonidling_tol = cvx.Variable((self.num_wl_vec, 1), nonneg=True)
        surplus_tol = cvx.Variable((self.num_buffers, 1), nonneg=True)
        # Parameters.
        state = cvx.Parameter((self.num_buffers, 1))
        horizon = cvx.Parameter(nonneg=True)
        workload_buffer_mat = cvx.Parameter((self.num_wl_vec, self.num_activities))
        ones_vec_nonidling = cvx.Parameter((self.num_wl_vec, 1))
        safety_stocks_vec = cvx.Parameter((self.num_resources, 1))
        allowed_activities = cvx.Parameter((self.num_activities, 1))
        weight_nonidling_tol = cvx.Parameter((self.num_wl_vec, 1), nonneg=True)
        surplus_target_vec = cvx.Parameter((self.num_buffers, 1), nonneg=True)

        # Build objective (we can remove demand as it is a constant).
        objective = cvx.Minimize(
            self.cost_per_buffer.T @ self.buffer_processing_matrix @ z
            + self.kappa_w * cvx.sum(cvx.multiply(weight_nonidling_tol, nonidling_tol))
            + self.kappa * cvx.sum(safety_stock_tol)
            + self.kappa * cvx.sum(surplus_tol)
        )

        # Build constraints.
        # Resource, nonnegative future state, and forbidden activities constraints.
        constraints = [
            self.constituency_matrix @ z <= 1,
            state + (self.buffer_processing_matrix @ z + self.demand_rate) * horizon >= 0,
            workload_buffer_mat @ z + ones_vec_nonidling == nonidling_tol,
            z <= allowed_activities,
            z <= 1,
            (state + (self.buffer_processing_matrix @ z + self.demand_rate) * horizon
             + surplus_tol >= surplus_target_vec)
        ]

        # Safety stock threshold constraints.
        for i, boundary_matrix in enumerate(self.list_boundary_constraint_matrices):
            constraints.append(
                boundary_matrix @ (state + (self.buffer_processing_matrix @ z + self.demand_rate)
                                   * horizon) + safety_stock_tol[i]
                >= safety_stocks_vec[i]
            )

        lp_problem = cvx.Problem(objective, constraints)
        return (lp_problem,
                state,
                horizon,
                workload_buffer_mat,
                ones_vec_nonidling,
                safety_stocks_vec,
                allowed_activities,
                weight_nonidling_tol,
                surplus_target_vec,
                z,
                safety_stock_tol,
                nonidling_tol,
                surplus_tol)

    def print_debug_info(self,
                         z_star,
                         weight_nonidling_tol,
                         allowed_activities,
                         nonidling_res,
                         draining_bottlenecks: Set[int]) -> None:
        """
        Print debug information.

        :param z_star: matrix where columns are the actions for each of the horizon.
        :param weight_nonidling_tol: Array of positive weights.
        :param allowed_activities: Binary array of allowed activities.
        :param nonidling_res: List of nonidling resources.
        :param draining_bottlenecks: set of resources that determine the draining time.
        :return: None.
        """
        inst_cost = self.cost_per_buffer.T @ self.buffer_processing_matrix @ z_star
        nonidling_cost = self.kappa_w * np.sum(
            np.multiply(weight_nonidling_tol, self._nonidling_tol.value))
        safety_stock_cost = self.kappa * np.sum(self._safety_stock_tol.value)
        print(f"inst_cost = {inst_cost}")
        print(f"nonidling_cost = {nonidling_cost}")
        print(f"safety_stock_cost = {safety_stock_cost}")
        print(f"kappa_w = {self.kappa_w}")
        print(f"kappa = {self.kappa}")
        print(f"allowed_activities = {allowed_activities.ravel()}")
        print(f"nonidling_res = {nonidling_res}")
        print(f"draining_bottlenecks = {draining_bottlenecks}")
        print(f"weight_nonidling_tol = {weight_nonidling_tol.ravel()}")
        print(f"nonidling_tol = {self._nonidling_tol.value.ravel()}")
        print(f"safety_stock_tol = {self._safety_stock_tol.value.ravel()}")

    def _update_values(
            self,
            state: types.StateSpace,
            safety_stocks_vec: types.ResourceSpace,
            k_idling_set: types.Array1D,
            draining_bottlenecks: Set[int],
            horizon: int,
            demand_plan: Optional[Dict[int, int]] = None
    ) -> Dict:
        """
        Overload...

        :param state: current state of the environment.
        :param safety_stocks_vec: Vector with safety stock levels for current state.
        :param k_idling_set: set of resources that should idle obtained when computing hedging.
        :param draining_bottlenecks: set of resources that determine the draining time.
        :param horizon: number of time steps that this policy should be performed.
        :param demand_plan: Dictionary with keys the identity of the buffers and values the actual
            forecast value.
        """
        assert horizon >= 1, f"Horizon must be >= 1, but provided: {horizon}."
        if self.boolean_action_flag:
            assert horizon == 1, f"Boolean activity rates only make sense for pure feedback " \
                                 f"policy, but provided: horizon={horizon} and " \
                                 f"boolean_action_flag={self.boolean_action_flag}."

        # Find set of nonidling resources as the set of workload dimensions minus k_idling_set.
        nonidling_res = policy_utils.obtain_nonidling_bottleneck_resources(self.num_wl_vec,
                                                                           k_idling_set)

        # Nonidling constraints, given by:   \xi^{i,T} B zeta + 1 = 0.
        workload_buffer_mat, ones_vec_nonidling = self._update_nonidling_parameters(nonidling_res)

        # Give more weight to dynamic bottleneck than to the rest of nonidling resources.
        weight_nonidling_tol = self.get_weight_nonidling_tol(draining_bottlenecks)

        # Forbidden activities.
        ind_forbidden_activities = self.get_index_all_forbidden_activities(nonidling_res)
        allowed_activities = self.get_allowed_activities_constraints(ind_forbidden_activities)

        # Update problem parameters
        self._state.value = state
        self._horizon.value = horizon
        self._workload_buffer_mat.value = workload_buffer_mat
        self._ones_vec_nonidling.value = ones_vec_nonidling
        self._weight_nonidling_tol.value = weight_nonidling_tol
        self._safety_stocks_vec.value = safety_stocks_vec
        self._allowed_activities.value = allowed_activities
        # Update target surplus buffer for pull models.
        if self.pull:
            assert demand_plan is not None, "A valid demand_plan is needed for the surplus buffers."
            self._surplus_target_vec.value = self._update_surplus_parameter(demand_plan)
        else:
            self._surplus_target_vec.value = np.zeros((self.num_buffers, 1))

        info_values = {
            'weight_nonidling_tol': weight_nonidling_tol,
            'allowed_activities': allowed_activities,
            'nonidling_res': nonidling_res,
            'draining_bottlenecks': draining_bottlenecks
        }
        return info_values

    def get_policy(self, state: types.StateSpace, **kwargs) -> Tuple[types.ActionSpace, float]:
        """
        Returns a schedule that is approximately optimal in the discounted cost sense for the
        number of time steps given by 'horizon' when starting at 'state'.

        :param state: current state of the environment.
        :param kwargs: Dictionary with parameters relevant tot he policy like safety stock levels,
            idling set, draining bottlenecks, horizon and demand plan.
        :return (z_star, opt_val):
            - z_star: matrix where columns are the actions for each of the horizon.
            - opt_val: value of the objective cost at z_star.
        """
        info_values = self._update_values(state, **kwargs)
        opt_val = self._lp_problem.solve(solver=eval(self.convex_solver), warm_start=True)
        z_star = self._z.value

        # Debug info
        if self.debug_info:
            self.print_debug_info(z_star, **info_values)

        return z_star, opt_val
