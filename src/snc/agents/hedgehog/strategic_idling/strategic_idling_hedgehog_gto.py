import cvxpy as cvx
import numpy as np
from typing import Optional, Set, Dict, Any

from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_from_workload \
    as compute_min_drain_time
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingOutput, \
    StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import get_dynamic_bottlenecks, \
    is_pull_model
from snc.utils.snc_types import WorkloadSpace, StateSpace


class StrategicIdlingHedgehogNaiveGTO(StrategicIdlingHedging):
    """
    StrategicIdlingHedgehogNaiveGTO simply ensures that dynamic bottlenecks which determine
    the minimum draining time in the network are not allowed to idle.
    """
    def _is_switching_curve_regime(self, w: WorkloadSpace,
                                   current_workload_variables: Dict[str, Any]) -> bool:
        """
        The switching curve regime is determined by checking if workload at minimum
        cost effective state (w_star) corresponds to longer minimum draining time than
        for current workload.

        :param w: current state in workload space, i.e. w = Xi x.
        :param current_workload_variables: dictionary of relevant variables in workload space.
                                           It must contain w_star.
        :return: bool
        """
        assert not is_pull_model(self.model_type), \
            f"Minimum draining time is computed assuming workload vectors with o_s = 1. " \
            f"But current environment is: {self.model_type}."
        w_star = current_workload_variables['w_star']
        current_min_drain_time = compute_min_drain_time(w, self._load)
        new_min_drain_time = compute_min_drain_time(w_star, self._load)
        tol = 1e-3
        assert new_min_drain_time >= current_min_drain_time - tol, \
            "Something is wrong here! Idling a non-dynamic bottleneck shouldn't increase the " \
            f"draining time, but it has increased by {new_min_drain_time - current_min_drain_time}."
        return new_min_drain_time > current_min_drain_time

    def _handle_switching_curve_regime(self, w: WorkloadSpace,
                                       current_workload_vars: Dict[str, Any]) -> Dict[str, Any]:
        """
        In case switching curve regime has been identified, this method finds
        the current set of dynamic bottlenecks which determine the minimum draining
        time and returns the strategic idling output with idling set of all bottlenecks
        excluding the dynamic ones.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param current_workload_vars: dictionary containing all the current worloadspace variables.
        :return: Strategic idling decision with k_idling set overriden.
        """
        dynamic_bottlenecks = get_dynamic_bottlenecks(w, self._workload_mat, self._load)
        k_idling_set = np.array([i for i in range(len(w)) if i not in dynamic_bottlenecks])
        current_workload_vars['k_idling_set'] = k_idling_set
        return current_workload_vars

    def _verify_standard_hedging_regime(self, current_workload_vars: Dict[str, Any]) -> None:
        """
        Method is called in order to check whether network is indeed facing a standard hedging
        regime by computing the dot product between the drift vector and vector determining
        the boundary of monotone region (psi_plus).
        Positive dot product corresponds to standard hedging regime.

        :param current_workload_vars: dictionary containing all the current workload-space vars.
        """
        hedging_case = current_workload_vars['hedging_case']
        if hedging_case == 'standard':  # Method invalid if psi_plus is altered artificially.
            psi_plus = current_workload_vars['psi_plus']
            drift = (1 - self._load).reshape(-1, 1)
            psi_drift_dot = psi_plus.T @ drift
            eps = 1e-6  # numerical tolerance
            assert psi_drift_dot > -eps

    def get_allowed_idling_directions(self, state: StateSpace, safety_stocks_vec) -> StrategicIdlingOutput:
        """
        Method returns idling decision corresponding to either standard hedging or
        switching curve regimes.

        :param state: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        w = self._workload_mat @ state
        self._safety_stocks_vec = safety_stocks_vec
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


class StrategicIdlingHedgehogGTO(StrategicIdlingHedgehogNaiveGTO):
    """
    This class follows 'StrategicIdlingHedgehogNaiveGTO' but when encountering switching curve
    regime adopts a more explicit GTO policy and tries to minimise the instantaneous cost
    without compromising minimum draining time.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_dyn_bot_set: Optional[Set[int]] = None
        self._min_drain_lp: Optional[cvx.Problem] = None
        self._min_drain_x: Optional[cvx.Variable] = None
        self._workload_critical_mat: Optional[cvx.Parameter] = None
        self._workload_rest_mat: Optional[cvx.Parameter] = None
        self._drain_time_rest_mat: Optional[cvx.Parameter] = None
        self._w_critical: Optional[cvx.Parameter] = None
        self._w_rest: Optional[cvx.Parameter] = None
        self._min_drain_time: Optional[cvx.Parameter] = None

    def _create_find_workload_with_min_draining_time_by_idling_lp_program(self):
        num_resources, num_buffers = self._workload_mat.shape
        n = num_resources
        m = num_buffers
        x_var = cvx.Variable((m, 1), nonneg=True)  # Variable
        workload_critical_mat = cvx.Parameter((n, m)) # workload matrix for dynamic bottlenecks
        workload_rest_mat = cvx.Parameter((n, m)) # workload matrix for remaining bottlenecks
        drain_time_rest_mat = cvx.Parameter((n, m)) # draining time matrix for remaining bottlenecks
        w_critical = cvx.Parameter((n, 1))  # workload vector for dynamic bottlenecks
        w_rest = cvx.Parameter((n, 1))  # workload vector for remaining bottlenecks
        min_drain_time = cvx.Parameter(nonneg=True)  # minimum draining time

        w = w_critical + w_rest  # the full workload vector

        penalty_coeff_w_star = self.strategic_idling_params.penalty_coeff_w_star

        objective = cvx.Minimize(
            self._cost_per_buffer.T @ x_var
            + penalty_coeff_w_star * cvx.sum(self._workload_mat @ x_var - w))
        constraints = [
            # Don't idle critical bottlenecks.
            workload_critical_mat @ x_var == w_critical,
            # Reduce cost by idling (increasing workload).
            workload_rest_mat @ x_var >= w_rest,
            # Don't idle beyond the minimum draining time.
            drain_time_rest_mat @ x_var <= min_drain_time]
        lp_problem = cvx.Problem(objective, constraints)
        self._min_drain_lp = lp_problem
        self._min_drain_x = x_var
        self._workload_critical_mat = workload_critical_mat
        self._workload_rest_mat = workload_rest_mat
        self._drain_time_rest_mat = drain_time_rest_mat
        self._w_critical = w_critical
        self._w_rest = w_rest
        self._min_drain_time = min_drain_time

    def _find_workload_with_min_draining_time_by_idling(self, w: WorkloadSpace) -> WorkloadSpace:
        """
        This method first identifies a set of current dynamic bottlenecks which determine
        the minimum draining time. If this set has changed from the previous call to the method,
        the corresponding LP constraints for dynamic and remaining bottlenecks are updated.
        LP is then solved to yield the target workload vector corresponding to minimum
        effective cost subject to minimum draining time constraint.

        :param w: current state in workload space, i.e. w = Xi x.
        :return: w_min_drain, target workload vector
        """
        dyn_bot_set = get_dynamic_bottlenecks(w, self._workload_mat, self._load)

        ind_dyn_bot = np.array(list(dyn_bot_set))

        # Update parameters of LP only when they change, i.e. when set of dynamic bottlenecks change
        if self._current_dyn_bot_set is None or dyn_bot_set != self._current_dyn_bot_set:

            workload_critical_mat = np.zeros_like(self._workload_mat)
            workload_critical_mat[ind_dyn_bot, :] = self._workload_mat[ind_dyn_bot, :]
            self._workload_critical_mat.value = workload_critical_mat

            workload_rest_mat = self._workload_mat.copy()
            workload_rest_mat[ind_dyn_bot, :] = 0
            self._workload_rest_mat.value = workload_rest_mat

            drift = (1 - self._load).reshape(-1, 1)
            assert drift.shape == (len(self._load), 1)
            drain_time_rest_mat = np.multiply(1/drift, workload_rest_mat)
            self._drain_time_rest_mat.value = drain_time_rest_mat

        self._current_dyn_bot_set = dyn_bot_set

        w_critical = np.zeros_like(w)
        w_critical[ind_dyn_bot, 0] = w[ind_dyn_bot, 0]
        self._w_critical.value = w_critical
        w_rest = w.copy()
        w_rest[ind_dyn_bot, 0] = 0
        self._w_rest.value = w_rest
        assert not is_pull_model(self.model_type)
        self._min_drain_time.value = compute_min_drain_time(w, self._load)

        self._min_drain_lp.solve(solver=eval(self.strategic_idling_params.convex_solver),
                                 warm_start=True)  # Solve LP.
        min_drain_x = self._min_drain_x.value
        w_star = self._workload_mat @ min_drain_x
        return w_star

    def _handle_switching_curve_regime(self, w: WorkloadSpace,
                                       current_workload_variables: Dict[str, Any]) \
            -> Dict[str, Any]:
        """
        In case switching curve regime has been identified, this method determines the set
        idling bottlenecks by solving a min-cost effective state LP which has additional constraint
        ensuring that new lower cost workload configuration does not extended the current
        minimum draining time in the network.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param current_workload_variables: dictionary containing all the current workload-space
            variables to be updated.
        :return: Strategic idling decision with k_idling set overridden.
        """
        if self._min_drain_lp is None:
            self._create_find_workload_with_min_draining_time_by_idling_lp_program()

        eps = 1e-6

        w_min_drain = self._find_workload_with_min_draining_time_by_idling(w)
        v_min_drain = self._get_vector_defining_possible_idling_direction(w_min_drain, w)
        k_idling_set = np.where(v_min_drain > eps)[0]
        current_workload_variables['w_min_drain'] = w_min_drain
        current_workload_variables['k_idling_set'] = k_idling_set
        return current_workload_variables


class StrategicIdlingHedgehogGTO2(StrategicIdlingHedgehogGTO):
    """
    StrategicIdlingHedgehogGTO checks if for current workload location the drift
    will stray into monotone region to identify switching curve regime.
    """
    def _is_switching_curve_regime(self, w: WorkloadSpace,
                                   current_workload_variables: Dict[str, Any]) -> bool:
        """
        Method determines whether network is facing a switching curve regime
        by computing the dot product between the drift vector and vector determining the
        boundary of monotone region (psi_plus).
        Negative dot product corresponds to switching curve regime and positive to standard
        hedging regime

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param current_workload_variables: dict containing all the current workload-space variables.
        :return: bool
        """
        w_star = current_workload_variables['w_star']
        v_star = current_workload_variables['v_star']

        psi_plus, _, _ = self._get_closest_face_and_level_sets(w_star, v_star)
        drift = (1 - self._load).reshape(-1,1)
        psi_drift_dot = psi_plus.T @ drift
        eps = 1e-6  # numerical tolerance
        return psi_drift_dot < -eps


class StrategicIdlingGTO(StrategicIdlingCore):

    def get_allowed_idling_directions(self, state: StateSpace) -> StrategicIdlingOutput:
        """
        Method returns a full set of network bottlenecks excluding the ones which constitute
        the set of dynamic bottlenecks.

        :param state: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        w = self._workload_mat @ state
        dynamic_bottlenecks = get_dynamic_bottlenecks(w, self._workload_mat, self._load)
        k_idling_set = np.array([i for i in range(len(w)) if i not in dynamic_bottlenecks])
        idling_decision = {'k_idling_set': k_idling_set, 'w': w}
        return self._get_null_strategic_idling_output(**idling_decision)
