import cvxpy as cvx
import numpy as np
from typing import Any, Dict, Optional

from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_from_workload \
    as compute_min_drain_time
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore, \
    StrategicIdlingOutput
from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.utils.snc_types import StateSpace, WorkloadMatrix, WorkloadSpace


class StrategicIdlingCoreHorizon(StrategicIdlingCore):
    """
    StrategicIdlingCoreHorizon extends the basic StrategicIdlingCore class by computing minimum
    effective cost for workload values 'horizon' timesteps away to synchronize it with future state
    of big step fluid policy.
    """
    def __init__(self,
                 workload_mat: WorkloadMatrix,
                 load: WorkloadSpace,
                 cost_per_buffer: StateSpace,
                 model_type: str,
                 horizon: int,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 debug_info: bool = False) -> None:

        super().__init__(workload_mat, load, cost_per_buffer, model_type,
                         strategic_idling_params, debug_info)

        self._horizon = horizon

    def _non_negative_workloads(self, w: WorkloadSpace, eps: float = 1e-6) -> Dict[str, Any]:
        """
        Overriden method considers idling decision at a 'drifted' workload position
        after horizon timesteps.

        :param w: current state in workload space, i.e. w = Xi x.
        :param eps: tolerance value to check that we are not in the negative orthant.
        :return: idling decision dictionary with the following fields:
                 w            - current workload position
                 w_star       - the upper bound on workload position to be reached after horizon
                                timesteps
                 c_bar        - local cost sensitivity to idling at 'drifted' workload position
                 v_star       - distance towards projected monotone region boudary (w_star)
                 k_idling_set - set of workload directions which are allowed to idle
        """
        w_drift = w - self._horizon * self.drift

        c_bar, x_eff = self._get_level_set_for_current_workload(w_drift)
        if not self._is_infeasible(c_bar) and self._is_defining_a_monotone_region(c_bar):
            current_workload_vars = {'w': w, 'w_star': w_drift, 'c_bar': c_bar,
                                     'k_idling_set': np.array([])}
            return current_workload_vars

        w_star, x_star = self._find_workload_with_min_eff_cost_by_idling(w_drift)

        v_star = self._get_vector_defining_possible_idling_direction(w_star, w_drift)
        k_idling_set = np.where(v_star > eps)[0]

        current_workload_vars = {'w': w, 'w_star': w_star, 'c_bar': c_bar, 'v_star': v_star,
                                 'k_idling_set': k_idling_set}

        return current_workload_vars

    def _negative_workloads(self, w: WorkloadSpace, eps: float = 1e-6) -> Dict[str, Any]:
        """
        For strategic idling with no hedging when workload has no positive
        components all resources are allowed to idle.

        :param w: current state in workload space, i.e. w = Xi x.
        :param eps: tolerance value to check that we are in the negative orthant.
        """
        assert False, "This class (StrategicIdlingCoreHorizon) is not yet suitable for pull models."
        current_workload_variables = {'w': w,'w_star': w, 'k_idling_set': np.arange(len(w))}
        return current_workload_variables

class StrategicIdlingGTOHorizon(StrategicIdlingCore):
    """
    This class follows a more explicit GTO policy and tries to minimise the instantaneous cost
    without compromising minimum draining time.
    """
    def __init__(self,
                 workload_mat: WorkloadMatrix,
                 load: WorkloadSpace,
                 cost_per_buffer: StateSpace,
                 model_type: str,
                 horizon: int,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 debug_info: bool = False) -> None:

        super().__init__(workload_mat, load, cost_per_buffer, model_type,
                         strategic_idling_params, debug_info)

        self._horizon = horizon

        self._w_star_lp_problem, self._x_star, self._w_param, self._min_drain_w = \
            self._create_find_workload_with_min_draining_time_by_idling_lp_program()

    def _create_find_workload_with_min_draining_time_by_idling_lp_program(self):
        x_var = cvx.Variable((self._num_buffers, 1), nonneg=True)  # Variable
        w_par = cvx.Parameter((self._num_bottlenecks, 1))  # Parameter
        # workload values corresponding to current minimum draining time
        min_drain_w = cvx.Parameter((self._num_bottlenecks, 1))

        penalty_coeff_w_star = self.strategic_idling_params.penalty_coeff_w_star
        objective = cvx.Minimize(
            self._cost_per_buffer.T @ x_var
            + penalty_coeff_w_star * cvx.sum(self._workload_mat @ x_var - w_par))

        constraints = [self._workload_mat @ x_var >= w_par,
                       self._workload_mat @ x_var <= min_drain_w]
        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, x_var, w_par, min_drain_w

    def _find_workload_with_min_draining_time_by_idling(self, w: WorkloadSpace) -> WorkloadSpace:
        """
        Current method sets lower bound constraints in minimum effective cost LP
        by using current workload state minus drift times horizon timesteps.
        It also finds minimum draining time after horizon timesteps and computes corresponding
        values for upper bound workload values.
        """
        min_drain_time = compute_min_drain_time(w, self._load)
        self._w_param.value = w - self._horizon * self.drift
        self._min_drain_w.value = max(0, min_drain_time - self._horizon) * self.drift

        self._w_star_lp_problem.solve(solver=eval(self.strategic_idling_params.convex_solver),
                                      warm_start=True)  # Solve LP.
        x_star = self._x_star.value
        w_star = self._workload_mat @ x_star  # Workload in the boundary of the monotone region.
        return w_star

    def get_allowed_idling_directions(self, state: StateSpace) -> StrategicIdlingOutput:
        """
        Method returns a full set of network bottlenecks excluding the ones which constitute
        the set of dynamic bottlenecks.

        :param state: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        eps = 1e-6

        w = self._workload_mat @ state
        w_min_drain = self._find_workload_with_min_draining_time_by_idling(w)
        v_min_drain = self._get_vector_defining_possible_idling_direction(w_min_drain, w)
        k_idling_set = np.where(v_min_drain > eps)[0]
        idling_decision = {'w': w, 'w_star': w_min_drain, 'k_idling_set': k_idling_set}

        return self._get_null_strategic_idling_output(**idling_decision)
