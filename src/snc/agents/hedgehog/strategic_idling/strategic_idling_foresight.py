import numpy as np
from typing import Optional, Set, Dict, Any

from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_from_workload \
    as compute_min_drain_time
from snc.agents.hedgehog.policies.big_step_w_bound_policy import BigStepWBoundPolicy
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore, \
    StrategicIdlingOutput
from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedgehog_gto \
    import StrategicIdlingHedgehogGTO
from snc.agents.hedgehog.strategic_idling.strategic_idling_horizon \
    import StrategicIdlingCoreHorizon, StrategicIdlingGTOHorizon
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import get_dynamic_bottlenecks
import snc.utils.snc_types as types
from snc.utils.snc_types import WorkloadSpace, StateSpace


class StrategicIdlingForesight(StrategicIdlingHedgehogGTO):
    """
    In general, when facing a switching curve regime, due to buffer cost differences following
    GTO policy by default might be prohibitively expensive in cumulative cost.
    This variant is an extension of Hedgehog-GTO hybrid that estimates the total cumulative policy
    cost until draining using a complete roll out of fluid model under either Hedgehog or GTO
    policies. It then commits to the policy with lower cost until the set of dynamic bottlenecks
    changes, so the policy roll-outs have to be rerun.
    """
    def __init__(self,
                 workload_mat: types.WorkloadMatrix,
                 neg_log_discount_factor: float,
                 load: WorkloadSpace,
                 cost_per_buffer: types.StateSpace,
                 model_type: str,
                 policy_object: BigStepWBoundPolicy,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 workload_cov: Optional[types.WorkloadCov] = None,
                 debug_info: bool = False) -> None:
        """
        :param workload_mat: workload matrix, with rows being workload vectors.
        :param neg_log_discount_factor: negative log of the discount factor used for the simulation.
        :param load: vector with load for every workload vector.
        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param model_type: String indicating if this is a `'pull'` or `'push'` model.
        :param policy_object: Policy to be used for the fluid rollouts.
        :param strategic_idling_params: tolerance levels, convex solver choice and
                                        other parameters for navigating effective cost space.
        :param workload_cov: asymptotic covariance of the workload process.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        """
        super().__init__(workload_mat, neg_log_discount_factor, load, cost_per_buffer, model_type,
                         strategic_idling_params, workload_cov, debug_info)

        self._current_state: Optional[StateSpace] = None
        self._current_regime: Optional[str] = None
        self._original_target_dyn_bot_set: Optional[Set[int]] = None
        self._policy_obj: Optional[BigStepWBoundPolicy] = None

        self._initialise_fluid_policy(policy_object)
        self._compute_num_roll_out_steps()

    def get_allowed_idling_directions(self, state: StateSpace) -> StrategicIdlingOutput:
        """
        Method returns a set of bottlenecks which are allowed to idle for a current
        buffer state.

        :param state: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        self._current_state = state
        return super().get_allowed_idling_directions(state)

    def _has_gto_regime_changed(self, current_workload_variables: Dict[str, Any]) -> bool:
        """
        Method is called whenever the algorithm has been following the GTO policy and checks if the
        current target set of dynamic bottlenecks for GTO idling decision is different from the
        original set when GTO was last chosen by fluid policy roll out cumulative cost estimation.
        If so, the location in workload space has changed and choice of Hedging-vs-GTO trajectory
        needs to be re-evaluated.

        :param current_workload_variables: Dict containing all the current workload-space variables.
        :return: bool
        """
        target_dyn_bot_set = get_dynamic_bottlenecks(
            current_workload_variables['w_min_drain'], self._workload_mat, self._load)
        return not target_dyn_bot_set.issubset(self._original_target_dyn_bot_set)

    def _has_hedging_regime_changed(self, current_workload_variables: Dict[str, Any]) -> bool:
        """
        Method is called whenever the algorithm has been following standard hedging policy and
        checks if the current target set of dynamic bottlenecks for hedging idling decision is
        different from the original set when standard hedging was last chosen by fluid policy
        roll out cumulative cost estimation. If so, the location in workload space has changed and
        choice of Hedgehog-vs-GTO trajectory needs to be re-evaluated.

        :param current_workload_variables: dictionary containing all the current workload-space
            variables.
        :return: bool
        """
        target_dyn_bot_set = get_dynamic_bottlenecks(
            current_workload_variables['w_star'], self._workload_mat, self._load)
        return not target_dyn_bot_set.issubset(self._original_target_dyn_bot_set)

    def _handle_switching_curve_regime(self, w: WorkloadSpace,
                                       current_workload_variables: Dict[str, Any]) \
            -> Dict[str, Any]:
        """
        Method chooses whether to follow standard hedging or GTO policy by running a continuous
        fluid model roll out to estimate the corresponding cumulative cost until draining for each
        policy. After cost comparison, the controller commits to following the cheaper policy until
        draining. However if at any time the set of dynamic bottlenecks changes the roll out
        comparison is performed again.
        """
        cw_vars = current_workload_variables

        recompute_cum_costs = False
        idling_dec_dict: Optional[Dict[str, Any]] = None  # Dictionary with idling decision.

        if self._current_regime is None:
            recompute_cum_costs = True
            idling_dec_dict = cw_vars
        elif self._current_regime == "standard_hedging":
            recompute_cum_costs = self._has_hedging_regime_changed(cw_vars)
            if not recompute_cum_costs:
                idling_dec_dict = self._add_standard_hedging(w, cw_vars)
        elif self._current_regime == "gto":
            idling_dec_dict = super()._handle_switching_curve_regime(w, cw_vars)
            recompute_cum_costs = self._has_gto_regime_changed(idling_dec_dict)

        if recompute_cum_costs:
            hedgehog_cum_cost = self._roll_out_hedgehog_fluid_policy(idling_dec_dict)
            gto_cum_cost = self._roll_out_gto_fluid_policy(idling_dec_dict)

            if hedgehog_cum_cost < gto_cum_cost:
                self._current_regime = "standard_hedging"
                idling_dec_dict = self._add_standard_hedging(w, cw_vars)
                self._original_target_dyn_bot_set = get_dynamic_bottlenecks(
                    idling_dec_dict['w_star'], self._workload_mat, self._load)
            else:
                self._current_regime = "gto"
                idling_dec_dict = super()._handle_switching_curve_regime(w, cw_vars)
                self._original_target_dyn_bot_set = get_dynamic_bottlenecks(
                    idling_dec_dict['w_min_drain'], self._workload_mat, self._load)

            if self.debug_info:
                print("Recomputed roll-out")
                print(f"hedgehog_cum_cost = {hedgehog_cum_cost}")
                print(f"Current regime = {self._current_regime}")
                print(f"gto_cum_cost = {gto_cum_cost}")
                print(f"original_target_dyn_bot_set = {self._original_target_dyn_bot_set}")

        assert idling_dec_dict is not None
        return idling_dec_dict

    def _roll_out_hedgehog_fluid_policy(self, current_workload_variables: Dict[str, Any]) -> float:
        """
        Custom method for selecting appropriate strategic idling object to be used in
        standard Hedgehog fluid policy roll output.

        :return: cumulative cost of standard hedgehog policy roll out.
        """
        min_drain_time = compute_min_drain_time(current_workload_variables['w_star'],
                                                self._load)

        horizon = self._compute_horizon(min_drain_time, self._num_steps)

        strategic_idling_object = StrategicIdlingCoreHorizon(self._workload_mat,
                                                             self._load,
                                                             self._cost_per_buffer,
                                                             self.model_type,
                                                             horizon,
                                                             self.strategic_idling_params)

        return self._roll_out_fluid_policy(strategic_idling_object, horizon)

    def _roll_out_gto_fluid_policy(self, current_workload_variables: Dict[str, Any]) -> float:
        """
        Custom method for selecting appropriate strategic idling object to be used in
        GTO fluid policy roll output.

        :return: cumulative cost of GTO policy roll out
        """
        min_drain_time = compute_min_drain_time(current_workload_variables['w'],
                                                self._load)

        horizon = self._compute_horizon(min_drain_time, self._num_steps)

        strategic_idling_object = StrategicIdlingGTOHorizon(self._workload_mat,
                                                            self._load,
                                                            self._cost_per_buffer,
                                                            self.model_type,
                                                            horizon,
                                                            self.strategic_idling_params)

        return self._roll_out_fluid_policy(strategic_idling_object, horizon)

    def _initialise_fluid_policy(self, policy_obj: BigStepWBoundPolicy):
        self._policy_obj = BigStepWBoundPolicy(policy_obj.cost_per_buffer,
                                               policy_obj.constituency_matrix,
                                               policy_obj.demand_rate,
                                               policy_obj.buffer_processing_matrix,
                                               policy_obj.workload_mat,
                                               policy_obj.convex_solver)

    def _compute_num_roll_out_steps(self) -> None:
        """
        Methods sets the number of steps to roll out the fluid policy until draining.
        This number should be at least equal to separation of timescales factor (1/drift)
        """
        self._num_steps = int(np.ceil(1 / (1-np.max(self._load))))

    @staticmethod
    def _compute_horizon(min_drain_time: float, num_steps: int) -> int:
        """
        Method computes the horizon of one roll out step given the minimum draining time
        of the network.

        :param min_drain_time: time it will take to drain the network according to fluid model.
        :param num_step: in how many roll out steps fluid model will reach drained state.
        :return: int
        """
        return int(np.ceil(min_drain_time / num_steps))

    def _roll_out_fluid_policy(self, strategic_idling_object: StrategicIdlingCore,
                               horizon: int) -> float:
        """
        Method simulates the roll out util draining of fluid model under the various
        policies determined by the received strategic_idling_object and computes the
        total accumulated cost until draining.

        :param strategic_idling_object: object determining the idling decisions of the policy.
        :param horizon: minimum draining time achieved by the provided policy
        :return: float
        """
        demand_rate = self._policy_obj.demand_rate
        x = self._current_state

        cum_cost = 0
        for _ in range(self._num_steps):
            si_output = strategic_idling_object.get_allowed_idling_directions(x)

            w_bound = si_output.w_star

            no_penalty_grad = np.zeros_like(x)
            z_star, _ = self._policy_obj.get_policy(x, no_penalty_grad, w_bound, horizon)

            x_new = x + ((self._policy_obj.buffer_processing_matrix @ z_star + demand_rate)
                         * horizon)

            cum_cost += self._cost_per_buffer.T @ ((x_new + x)/2 * horizon)
            x = x_new

        eps = 1e-6
        assert np.all(x < 1), f"Policy roll-out has ended before draining: final state={x.ravel()}."
        assert np.all(-eps <= x), f"Policy roll-out has drained too much: final state={x.ravel()}."

        return cum_cost
