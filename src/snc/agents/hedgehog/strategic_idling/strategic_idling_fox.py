import cvxpy as cvx
import numpy as np
from typing import Optional, Set, Dict, Any

from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_from_workload \
    as compute_min_drain_time
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingOutput, \
    StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.compute_primal_effective_cost \
    import ComputePrimalEffectiveCost
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import get_dynamic_bottlenecks, \
    is_pull_model
from snc.utils.snc_types import WorkloadSpace, StateSpace


class StrategicIdlingFox(StrategicIdlingHedging):

    def __init__(self,
                 workload_mat: types.WorkloadMatrix,
                 neg_log_discount_factor: float,
                 load: WorkloadSpace,
                 cost_per_buffer: types.StateSpace,
                 list_boundary_constraint_matrices,
                 model_type: str,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 workload_cov: Optional[types.WorkloadCov] = None,
                 debug_info: bool = False) -> None:
        """
        StrategicIdling class is responsible for online identification of idling directions for
        bottlenecks to reduce effective running cost in the network.

        :param workload_mat: workload matrix, with rows being workload vectors.
        :param neg_log_discount_factor: negative log of the discount factor given by environment.
        :param load: vector with loads for every workload vector.
        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param model_type: String indicating if this is a `'pull'` or `'push'` model.
        :param strategic_idling_params: tolerance levels, convex solver and other params
                                        for navigating effective cost space.
        :param workload_cov: asymptotic covariance of the workload process.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        """
        super().__init__(workload_mat, load, cost_per_buffer,
                         model_type, strategic_idling_params, debug_info)

        self.list_boundary_constraint_matrices = list_boundary_constraint_matrices
        self.c_bar_solver = ComputePrimalEffectiveCost(workload_mat, cost_per_buffer, list_boundary_constraint_matrices, convex_solver)
        self._w_star_lp_problem, self._x_star, self._w_param, self._safety_stocks_param,= \
        self._create_find_workload_with_min_eff_cost_by_idling_lp_program()

    def _find_workload_with_min_eff_cost_by_idling(self, w: WorkloadSpace) -> WorkloadSpace:
        self._w_param.value = w
        self._safety_stocks_param.value = np.zeros_like(self._safety_stocks_vec)
        self._w_star_lp_problem.solve(solver=eval(self.strategic_idling_params.convex_solver),
                                      warm_start=True)  # Solve LP.
        x_star = self._x_star.value
        w_star = self._workload_mat @ x_star  # Workload in the boundary of the monotone region.
        self._safety_stocks_param.value = self._safety_stocks_vec
        self._w_star_lp_problem.solve(solver=eval(self.strategic_idling_params.convex_solver),
                                      warm_start=True)  # Solve LP.
        x_star = self._x_star.value
        tol = 1e-6
        assert np.all(w_star >= w - tol)
        return w_star, x_star

    def _get_level_set_for_current_workload(self, w: WorkloadSpace) -> Optional[WorkloadSpace]:
        """
        The effective cost can be represented as a piecewise linear function,
        with coefficients given by the vertexes of the feasible set of the dual
        program of the LP that computes the effective cost. Indeed, the solution to such
        dual program for a given w, gives the linear coefficient at w.

        :param w: current state in workload space, i.e. w = Xi x.
        :return: c_bar: vector defining level set of the effective cost at current w. None is
            returned if the optimisation is unsuccessful.
        """
        c_bar, x_eff, _ = self.c_bar_solver.solve(w, self._safety_stocks_vec)
        return c_bar, x_eff

    def _create_find_workload_with_min_eff_cost_by_idling_lp_program(self):
        x_var = cvx.Variable((self._num_buffers, 1), nonneg=True)  # Variable
        w_par = cvx.Parameter((self._num_bottlenecks, 1))  # Parameter
        safety_stocks_vec = cvx.Parameter((self._num_bottlenecks, 1))
        penalty_coeff_w_star = self.strategic_idling_params.penalty_coeff_w_star
        objective = cvx.Minimize(
            self._cost_per_buffer.T @ x_var
            + penalty_coeff_w_star * cvx.sum(self._workload_mat @ x_var - w_par))
        constraints = [self._workload_mat @ x_var >= w_par]
        constraints.append(self._workload_mat[0,:] @ x_var == w_par[0])

        if add_safety_stocks:
            a_mat = np.vstack(self.list_boundary_constraint_matrices)
            constraints.append(a_mat @ x_var >= safety_stocks_vec)

        constraints.append(x_var >= 1)
        constraints.append(x_var[1:4] >= 20)
        constraints.append(x_var[3] >= 25)
        #constraints.append(x_var[2:4] >= 50)


        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, x_var, w_par, safety_stocks_vec


    def get_allowed_idling_directions(self, state: StateSpace) -> StrategicIdlingOutput:
        """
        Method returns idling decision corresponding to either standard hedging or
        switching curve regimes.

        :param state: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        w = self._workload_mat @ state
        self._verify_offline_preliminaries()
        if self._is_negative_orthant(w) and not self._is_1d_workload_relaxation(w):
            idling_decision_dict = self._negative_workloads(w)
            regime = "negative_workloads"
        else:
            current_workload_vars = self._non_negative_workloads(w)

            if self._is_decision_not_to_idle(current_workload_vars['k_idling_set']):
                idling_decision_dict = current_workload_vars
                regime = "no_dling"
            elif self._is_switching_curve_regime(w, current_workload_vars):
                idling_decision_dict = self._handle_switching_curve_regime(w,
                                                                           current_workload_vars)
                regime = "switching_curve"
            else:
                idling_decision_dict = self._add_standard_hedging(w, current_workload_vars)
                self._verify_standard_hedging_regime(idling_decision_dict)
                regime = "standard_hedging"

        idling_decision = self._get_null_strategic_idling_output(**idling_decision_dict)
        if self.debug_info:
            print(f"beta_star: {idling_decision.beta_star}, "
                  f"k_iling_set: {idling_decision.k_idling_set}, "
                  f"sigma_2_h: {idling_decision.sigma_2_h}, "
                  f"delta_h: {idling_decision.delta_h}, "
                  f"regime: {regime}")
        return idling_decision
