import cvxpy as cvx
import numpy as np
from typing import Callable, List, Optional, Set, Tuple

from snc.agents.hedgehog.params import BigStepLayeredPolicyParams
from snc.agents.hedgehog.policies.big_step_base_policy import BigStepBasePolicy
import snc.agents.hedgehog.policies.policy_utils as policy_utils
from snc.agents.solver_names import SolverNames
import snc.utils.snc_types as types


class BigStepLayeredPolicy(BigStepBasePolicy):

    def __init__(self,
                 cost_per_buffer: types.StateSpace,
                 constituency_matrix: types.ConstituencyMatrix,
                 demand_rate: types.StateSpace,
                 buffer_processing_matrix: types.BufferMatrix,
                 workload_mat: types.WorkloadMatrix,
                 nu: types.NuMatrix,
                 list_boundary_constraint_matrices: List[np.ndarray],
                 policy_params: Optional[BigStepLayeredPolicyParams] = None,
                 debug_info: float = False) -> None:
        """
        Construct big-step layered object whose main method 'get_policy', which at any given state
        return activity rates to be followed by for the number of time steps given by `horizon`.

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
        :param policy_params: Named tuple with policy parameters.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        :return None.
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

        self.convex_solver = self.get_policy_params(policy_params)

        self._create_lp_objects()

        self._draining_lp: Optional[cvx.Problem] = None
        self._draining_lp_constraints: Optional[List[cvx.constraints]] = None
        self._safety_stocks_lp: Optional[cvx.Problem] = None
        self._safety_stocks_lp_constraints: Optional[List[cvx.constraints]] = None
        self._nonidling_lp: Optional[cvx.Problem] = None
        self._nonidling_lp_constraints: Optional[List[cvx.constraints]] = None
        self._cost_lp: Optional[cvx.Problem] = None
        self._cost_lp_constraints: Optional[List[cvx.constraints]] = None
        cum_constraints = self._initialise_cum_constraints()
        self._initialise_lp_problems(cum_constraints)

    @staticmethod
    def get_policy_params(policy_params: BigStepLayeredPolicyParams) -> str:
        """
        Extracts convex_solver parameter from `BigStepLayeredPolicyParams` named tuple.

        :param policy_params: Named tuple with policy parameters.
        :return: convex_solver: String indicating which solver should be used.
        """
        if policy_params is None:
            convex_solver = "cvx.CPLEX"
        else:
            assert hasattr(policy_params, "convex_solver"), \
                "Policy parameters must include 'convex_solver'."
            convex_solver = policy_params.convex_solver

        assert convex_solver in SolverNames.CVX, \
            f"Convex solver must be in {SolverNames.CVX}, but provided: {convex_solver}."

        return convex_solver

    def _create_lp_objects(self):
        """
        Method initialises all the variables and parameters to be used in layered LP problems
        """
        # activity rates variables for the rest of LPs.
        self._z_drain = cvx.Variable((self.num_activities, 1), nonneg=True)
        self._z_safe = cvx.Variable((self.num_activities, 1), nonneg=True)
        self._z_nonidle = cvx.Variable((self.num_activities, 1), nonneg=True)
        self._z = cvx.Variable((self.num_activities, 1), nonneg=True)

        # parameters and variables for draining LP
        self._workload_buffer_mat_draining = cvx.Parameter((self.num_wl_vec, self.num_activities))
        self._ones_vec_draining = cvx.Parameter((self.num_wl_vec, 1))
        self._draining_tol_var = cvx.Variable((self.num_wl_vec, 1), nonneg=True)
        self._draining_tol_par = cvx.Parameter((self.num_wl_vec, 1), nonneg=True)

        # parameters and variables for safety stocks LP
        self._safety_stocks_vec = cvx.Parameter((self.num_resources, 1))
        self._safety_stock_tol_var = cvx.Variable((self.num_resources, 1), nonneg=True)
        self._safety_stock_tol_par = cvx.Parameter((self.num_resources, 1), nonneg=True)

        # parameters and variables for nonidling LP
        self._workload_buffer_mat_nonidling = cvx.Parameter((self.num_wl_vec, self.num_activities))
        self._ones_vec_nonidling = cvx.Parameter((self.num_wl_vec, 1))
        self._nonidling_tol_var = cvx.Variable((self.num_wl_vec, 1), nonneg=True)
        self._nonidling_tol_par = cvx.Parameter((self.num_wl_vec, 1), nonneg=True)

        # parameters for basic constraints
        state = cvx.Parameter((self.num_buffers, 1), nonneg=True)
        horizon = cvx.Parameter(nonneg=True)
        allowed_activities = cvx.Parameter((self.num_activities, 1))

        # Build basic set of constraints.
        # Resource, nonnegative future state, and forbidden activities constraints.
        self._basic_constraints = [
            lambda z: self.constituency_matrix @ z <= 1,
            lambda z: state + (self.buffer_processing_matrix @ z + self.demand_rate) * horizon >= 0,
            lambda z: z <= allowed_activities,
        ]

        self._state = state
        self._horizon = horizon
        self._allowed_activities = allowed_activities

    def _initialise_cum_constraints(self) -> List[Callable]:
        """
        Return list with the basic constraints that are shared by all layers.

        :return List of basic cvx constraints.
        """
        return [c for c in self._basic_constraints]

    def _initialise_lp_problems(self, cum_constraints: List[Callable]) -> None:
        """
        Create the series of fluid model LPs in strict order of priority. Each LP appends
        additional constraints for the higher priorities LPs.
        """
        cum_constraints = self._create_draining_lp_program(cum_constraints)
        cum_constraints = self._create_safety_stocks_lp_problem(cum_constraints)
        cum_constraints = self._create_nonidling_lp_program(cum_constraints)
        self._create_cost_lp_program(cum_constraints)

    def _create_draining_lp_program(self, cum_constraints: List[Callable]) -> List[Callable]:
        """
        Create cvxpy LP that finds the minimum feasible tolerance for satisfying nonilding
        constraints for dynamic (draining) bottlenecks. It augments the list of constraints by
        adding the minimum tolerance as a constraint for the next layer.

        :param cum_constraints: List of cvx constraints.
        :return List of augmented cvx constraints.
        """
        objective = cvx.Minimize(cvx.sum(self._draining_tol_var))

        # Build constraints.
        # Resource, nonnegative future state, and forbidden activities constraints.
        constraints = [c(self._z_drain) for c in cum_constraints]

        # add constraints with tolerance variables to optimize for current LP
        constraints.append(self._workload_buffer_mat_draining @ self._z_drain
                           + self._ones_vec_draining == self._draining_tol_var)
        # add new constraints with adjustable tolerance parameters for subsequent LPs
        cum_constraints.append(lambda z: self._workload_buffer_mat_draining @ z
                               + self._ones_vec_draining == self._draining_tol_par)

        self._draining_lp = cvx.Problem(objective, constraints)
        self._draining_lp_constraints = constraints
        return cum_constraints

    def _create_safety_stocks_lp_problem(self, cum_constraints: List[Callable]) -> List[Callable]:
        """
        Create cvxpy LP that finds the minimum feasible tolerance for satisfying safety stock
        levels at the end state of fluid model after horizon timesteps. It augments the list of
        constraints by adding the minimum tolerance as a constraint for the next layer.

        :param cum_constraints: List of cvx constraints.
        :return List of augmented cvx constraints.
        """
        objective = cvx.Minimize(cvx.sum(self._safety_stock_tol_var))

        constraints = [c(self._z_safe) for c in cum_constraints]

        # create a single matrix transform for all safety stock variables
        a_mat = np.vstack(self.list_boundary_constraint_matrices)

        constraints.append(a_mat @ (self._state + (self.buffer_processing_matrix @ self._z_safe
                                                   + self.demand_rate) * self._horizon)
                           + self._safety_stock_tol_var >= self._safety_stocks_vec)

        cum_constraints.append(lambda z: a_mat @
                               (self._state + (self.buffer_processing_matrix @ z + self.demand_rate)
                                * self._horizon) + self._safety_stock_tol_par
                               >= self._safety_stocks_vec)

        self._safety_stocks_lp = cvx.Problem(objective, constraints)
        self._safety_stocks_lp_constraints = constraints
        return cum_constraints

    def _create_nonidling_lp_program(self, cum_constraints: List[Callable]) -> List[Callable]:
        """
        Create cvxpy LP that finds the minimum feasible tolerance for satisfying nonidling
        constraints for workload directions which are not draining critical. It augments the list of
        constraints by adding the minimum tolerance as a constraint for the next layer.

        :param cum_constraints: List of cvx constraints.
        :return List of augmented cvx constraints.
        """
        objective = cvx.Minimize(cvx.sum(self._nonidling_tol_var))

        # Build constraints.
        constraints = [c(self._z_nonidle) for c in cum_constraints]
        # add constraints with tolerance variables to optimize for current LP
        constraints.append(self._workload_buffer_mat_nonidling @ self._z_nonidle
                           + self._ones_vec_nonidling == self._nonidling_tol_var)
        # add new constraints with adjustable tolerance parameters for subsequent LPs
        cum_constraints.append(lambda z: self._workload_buffer_mat_nonidling @ z
                               + self._ones_vec_nonidling == self._nonidling_tol_par)

        self._nonidling_lp = cvx.Problem(objective, constraints)
        self._nonidling_lp_constraints = constraints
        return cum_constraints

    def _create_cost_lp_program(self, cum_constraints: List[Callable]) -> None:
        """
        Create cvxpy LP that finds the cost optimal activity rates subject to meeting all the
        nonidling and safety stock constraints within the feasible tolerance levels.

        :param cum_constraints: List of cvx constraints.
        :return None.
        """
        objective = cvx.Minimize(
            self.cost_per_buffer.T @ self.buffer_processing_matrix @ self._z)

        # Build constraints.
        constraints = [c(self._z) for c in cum_constraints]
        self._cost_lp = cvx.Problem(objective, constraints)
        self._cost_lp_constraints = constraints

    def _update_values(
            self,
            state: types.StateSpace,
            safety_stocks_vec: types.ResourceSpace,
            k_idling_set: types.Array1D,
            draining_bottlenecks: Set[int],
            horizon: int
    ) -> None:
        """

        :param state: current state of the environment.
        :param safety_stocks_vec: Vector with safety stock levels for current state.
        :param k_idling_set: set of resources that should idle obtained when computing hedging.
        :param draining_bottlenecks: set of resources that determine the draining time.
        :param horizon: number of time steps that this policy should be performed.
        :return None.
        """
        assert horizon >= 1, f"Horizon must be >= 1, but provided: {horizon}."

        # Find set of nonidling resources as the set of workload dimensions minus k_idling_set.
        all_nonidling_w_dirs = policy_utils.obtain_nonidling_bottleneck_resources(self.num_wl_vec,
                                                                                  k_idling_set)
        # Update draining bottleneck LP parameters
        draining_w_dirs = np.array([r for r in all_nonidling_w_dirs if r in draining_bottlenecks])

        self._workload_buffer_mat_draining.value, self._ones_vec_draining.value \
            = self._update_nonidling_parameters(draining_w_dirs)

        # Update other nonidling workload directions LP parameters
        nonidling_w_dirs = np.array([r for r in all_nonidling_w_dirs
                                     if r not in draining_bottlenecks])

        self._workload_buffer_mat_nonidling.value, self._ones_vec_nonidling.value \
            = self._update_nonidling_parameters(nonidling_w_dirs)

        # Forbidden activities.
        ind_forbidden_activities = self.get_index_all_forbidden_activities(all_nonidling_w_dirs)
        allowed_activities = self.get_allowed_activities_constraints(ind_forbidden_activities)

        # Update rest of problem parameters.
        self._state.value = state
        self._horizon.value = horizon
        self._safety_stocks_vec.value = safety_stocks_vec
        self._allowed_activities.value = allowed_activities

    def _solve_lp_layers(self) -> float:
        """
        Solve LP in order of priority and update parameters from one layer to the next one.

        :return optimal value after solving all layers.
        """
        self._draining_lp.solve(solver=eval(self.convex_solver), warm_start=True)
        self._draining_tol_par.value = self._draining_tol_var.value

        self._safety_stocks_lp.solve(solver=eval(self.convex_solver), warm_start=True)
        self._safety_stock_tol_par.value = self._safety_stock_tol_var.value

        self._nonidling_lp.solve(solver=eval(self.convex_solver), warm_start=True)
        self._nonidling_tol_par.value = self._nonidling_tol_var.value

        opt_val = self._cost_lp.solve(solver=eval(self.convex_solver), warm_start=True)
        return opt_val

    def get_policy(self, state: types.StateSpace, **kwargs) -> Tuple[types.ActionSpace, float]:
        """
        Returns a schedule that is approximately optimal in the discounted cost sense for the
        number of time steps given by 'horizon' when starting at 'state'.

        :param state: current state of the environment.
        :param kwargs: Dictionary with parameters relevant tot he policy like safety stock levels,
            idling set, draining bottlenecks and horizon.
        :return (z_star, opt_val):
            - z_star: policy as a vector with activity rates to be used for the given horizon.
            - opt_val: value of the objective cost at z_star.
        """
        self._update_values(state, **kwargs)
        opt_val = self._solve_lp_layers()
        return self._z.value, opt_val
