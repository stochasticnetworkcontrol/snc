from copy import deepcopy
from typing import Tuple, Dict
import cvxpy as cvx
import snc.utils.snc_types as types
from snc.agents.solver_names import SolverNames
from snc.simulation.store_data.numpy_encoder import clean_to_serializable


class ComputePrimalEffectiveCost:

    def __init__(self,
                 workload_mat: types.WorkloadMatrix,
                 cost_per_buffer: types.StateSpace,
                 convex_solver: str):

        assert convex_solver in SolverNames.CVX
        self.convex_solver = convex_solver

        self.workload_mat = workload_mat
        self.cost_per_buffer = cost_per_buffer

        self._lp_problem, self._x_eff_var, self._w_param = \
            self._create_compute_primal_effective_cost_lp_program()

    def _create_compute_primal_effective_cost_lp_program(self) \
            -> Tuple[cvx.Problem, cvx.Variable, cvx.Parameter]:
        """
        Creates the linear program that will defines the dual of the effective cost problem.

        :return: (lp_problem, c_bar_var, w_param, constraints):
            - lp_problem: Linear program structure.
            - c_bar_var: c_bar vector variable normal to the the hyperplane that defines the
                effective cost for the current workload.
            - w_param: Current workload parameter.
        """
        num_buffers = self.workload_mat.shape[1]
        x_eff_var = cvx.Variable((num_buffers, 1))
        w_param = cvx.Parameter((num_buffers, 1))
        objective = cvx.Minimize(self.cost_per_buffer.T @ x_eff_var)
        constraints = [self.workload_mat @ x_eff_var == w_param]
        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, x_eff_var, w_param

    def solve(self, w: types.WorkloadSpace):
        """
        Solves the linear program with the current workload using warm start.

        :param w: Current workload.
        :return: (c_bar, x_star, eff_cost)
            - c_bar: vector defining level set of the effective cost at w. None is returned if the
                optimisation is unsuccessful.
            - x_star = effective state, solution to primal program,
            - eff_cost = actual value of the effective cost.
        """
        self._w_param.value = w  # Update parameter value with actual current workload.
        eff_cost = self._lp_problem.solve(solver=eval(self.convex_solver), warm_start=True)
        c_bar = self._lp_problem.constraints[0].dual_value
        x_eff = self._x_eff_var.value
        return c_bar, x_eff, eff_cost

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        return clean_to_serializable(self)
