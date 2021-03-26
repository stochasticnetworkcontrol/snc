import cvxpy as cvx
import numpy as np
from typing import Callable, Dict, List, Optional, Set, Tuple

from snc.agents.hedgehog.params import BigStepLayeredPolicyParams
from snc.agents.hedgehog.policies.big_step_layered_policy import BigStepLayeredPolicy
import snc.utils.snc_types as types


class BigStepSurplusLayeredPolicy(BigStepLayeredPolicy):

    def __init__(self,
                 cost_per_buffer: types.StateSpace,
                 constituency_matrix: types.ConstituencyMatrix,
                 demand_rate: types.StateSpace,
                 buffer_processing_matrix: types.BufferMatrix,
                 workload_mat: types.WorkloadMatrix,
                 nu: types.NuMatrix,
                 list_boundary_constraint_matrices: List[np.ndarray],
                 ind_surplus_buffers: Optional[List[int]] = None,
                 policy_params: Optional[BigStepLayeredPolicyParams] = None,
                 debug_info: float = False) -> None:
        """
        Creates big-step layered policy that includes an additional layer to ensure that surplus
        buffers are as close as possible to its desired safety stock level. The main method of the
        class is still 'get_policy', which at any given state return activity rates to be followed
        by for the number of time steps given by `horizon`.

        The computation of policy proceeds in a strictly ordered sequence of layered LP problems,
        each solving for minimum achievable tolerance levels. Resulting tolerance levels for high
        priority constraints are relayed to lower priority LP problems as hard parameters modifying
        fixed constraints.

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
        :return None.
        """
        self.pull = True if ind_surplus_buffers is not None else False
        self.ind_surplus_buffers = ind_surplus_buffers
        self._surplus_lp: Optional[cvx.Problem] = None
        self._surplus_lp_constraints: Optional[List[cvx.constraints]] = None

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

    def _create_lp_objects(self):
        """
        Overloads parent's method initialising the variables and parameters for the surplus safety
        stock problem first. Then, it calls the overloaded parent's method to initialise the
        variables and parameters of the rest LP layers.
        """

        # Activity rates variable, other variables and parameters for surplus LP.
        if self.pull:
            self._z_surplus = cvx.Variable((self.num_activities, 1), nonneg=True)
            self._surplus_target_vec = cvx.Parameter((self.num_buffers, 1), nonneg=True)
            self._surplus_tol_var = cvx.Variable((self.num_buffers, 1), nonneg=True)
            self._surplus_tol_par = cvx.Parameter((self.num_buffers, 1), nonneg=True)
        super()._create_lp_objects()

    def _initialise_lp_problems(self, cum_constraints: List) -> None:
        """
        Overload original method to create the surplus safety stock layer first, so that it takes
        the highest priority and its minimum tolerance is passed as a constraint to the rest of
        layers. Then, launch the overloaded method from the parent class to create the rest of
        layers.
        """
        if self.pull:
            cum_constraints = self._create_surplus_lp_problem(cum_constraints)
        super()._initialise_lp_problems(cum_constraints)

    def _create_surplus_lp_problem(self, cum_constraints: List[Callable]) -> List[Callable]:
        """
        Creates cvxpy optimisation problem that finds the minimum feasible tolerance for satisfying
        stock levels in the surplus buffers at the end of the horizon considered by the fluid model.
        """
        objective = cvx.Minimize(cvx.sum(self._surplus_tol_var))

        # Build constraints.
        # Resource, nonnegative future state, and forbidden activities constraints.
        constraints = [c(self._z_surplus) for c in cum_constraints]

        # Add constraints with tolerance variables to optimize for current LP.
        constraints.append(
            self._state + (
                self.buffer_processing_matrix @ self._z_safe + self.demand_rate
            ) * self._horizon + self._surplus_tol_var >= self._surplus_target_vec
        )
        # Add new constraints with adjustable tolerance parameters for subsequent LPs.
        cum_constraints.append(
            lambda z: self._state + (
                self.buffer_processing_matrix @ z + self.demand_rate
            ) * self._horizon + self._surplus_tol_par >= self._surplus_target_vec
        )
        self._surplus_lp = cvx.Problem(objective, constraints)
        self._surplus_lp_constraints = constraints
        return cum_constraints

    def _update_values(
            self,
            state: types.StateSpace,
            safety_stocks_vec: types.ResourceSpace,
            k_idling_set: types.Array1D,
            draining_bottlenecks: Set[int],
            horizon: int,
            demand_plan: Optional[Dict[int, int]] = None
    ) -> None:
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

        # Update target surplus buffer for pull models.
        if self.pull:
            assert demand_plan is not None, "A valid demand_plan is needed for the surplus buffers."
            self._surplus_target_vec.value = self._update_surplus_parameter(demand_plan)

        super()._update_values(
            state,
            safety_stocks_vec,
            k_idling_set,
            draining_bottlenecks,
            horizon
        )

    def _solve_lp_layers(self) -> float:
        """
        Solve LP in order of priority and update parameters from one layer to the next one.

        :return optimal value after solving all layers.
        """
        if self.pull:
            self._surplus_lp.solve(solver=eval(self.convex_solver), warm_start=True)
            self._surplus_tol_par.value = self._surplus_tol_var.value

        return super()._solve_lp_layers()
