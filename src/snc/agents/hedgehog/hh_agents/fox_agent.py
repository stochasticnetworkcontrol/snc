import numpy as np
from typing import Optional, Type, Union, Dict

from snc.agents.activity_rate_to_mpc_actions.feedback_mip_feasible_mpc_policy import \
    FeedbackMipFeasibleMpcPolicy
from snc.agents.activity_rate_to_mpc_actions.fox_mpc import FoxMpcPolicy
from snc.agents.activity_rate_to_mpc_actions.feedback_stationary_feasible_mpc_policy import \
    FeedbackStationaryFeasibleMpcPolicy
from snc.agents.hedgehog.class_loader import get_class_from_name
from snc.agents.hedgehog.hh_agents.hedgehog_agent_interface import AsymptoticCovarianceParams, \
    HedgehogAgentInterface, HedgehogHyperParams, WorkloadRelaxationParams
import snc.agents.hedgehog.minimal_draining_time as mdt
from snc.agents.hedgehog.params import \
    BigStepLayeredPolicyParams, \
    BigStepPenaltyPolicyParams, \
    DemandPlanningParams, \
    StrategicIdlingParams
from snc.agents.hedgehog.policies.big_step_layered_policy import BigStepLayeredPolicy
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
from snc.agents.hedgehog.policies.big_step_surplus_layered_policy import BigStepSurplusLayeredPolicy
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_foresight import StrategicIdlingForesight
from snc.environments.controlled_random_walk import ControlledRandomWalk


class FoxAgent(HedgehogAgentInterface):
    def __init__(self,
                 env: ControlledRandomWalk,
                 discount_factor: float,
                 workload_relaxation_params: WorkloadRelaxationParams,
                 hedgehog_hyperparams: HedgehogHyperParams,
                 asymptotic_covariance_params: AsymptoticCovarianceParams,
                 strategic_idling_params: StrategicIdlingParams = StrategicIdlingParams(),
                 policy_params: Optional[
                     Union[BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams]] = None,
                 strategic_idling_class: Type[StrategicIdlingCore] = StrategicIdlingForesight,
                 demand_planning_params: DemandPlanningParams = DemandPlanningParams(),
                 name: str = "BigStepActivityRatesHedgehogAgent",
                 debug_info: bool = False,
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        :param env: Environment to stepped through.
        :param discount_factor: Discount factor for the cost per time step.
        :param workload_relaxation_params: Tuple of parameters defining the first workload
            relaxation.
        :param hedgehog_hyperparams: Named tuple including the following parameters:
            - activity_rates_policy_class_name:
            - mpc_policy_class_name:
            - theta_0: Tuning parameter to compute the safety stock threshold.
            - horizon_drain_time_ratio: Ratio num steps of the planning horizon (a.k.a. size of the
                Big step) over the minimal draining time. If horizon_drain_time_ratio == 0, then
                the horizon equals minimum_horizon (described below).
            - horizon_mpc_ratio: Ratio num steps to follow the activity rates over horizon.
            - minimum_horizon: Minimum horizon length i.e. minimum step size of the big step policy.
        :param asymptotic_covariance_params: Named tuple of parameters used to generate an estimate
            of the asymptotic covariance matrix. It includes
            - num_batch: Number of batches to group data samples for covariance estimation.
            - num_presimulation_steps: Total number of simulated steps for covariance estimation.
        :param strategic_idling_params: Named tuple with the following parameters:
            - convex_solver: Solver to be used at the different steps when finding potential idling
                directions.
            - epsilon: Nonnegative float that gives the angle used to create the artificial cone
                around the monotone region when it is a ray (i.e. it has empty interior).
            - shift_eps: Nonnegative float used as distance that we go before and beyond w_star in
                order to obtain the level sets. It has to be large enough to avoid the lack of
                precision of the convex optimisation solver, and small enough to remain in the
                closest linear part of the piece-wise linear cost.
            - hedging_scaling_factor: Nonnegative float used as scaling factor that multiplies
                the hedging threshold given by the diffusion heuristic.
            - penalty_coeff_w_star: Nonnegative float used as penalty coefficient to encourage the
                solution w_star to be close to the lower boundary of the solution set (useful when
                this is not a singleton).
        :param policy_params: Named tuple with the following parameters. It is specific for each
            policy class.
            - convex_solver: String with the solver to be used by the policy.
            - boolean_action_flag: Indicates if the optimisation variable of the policy synthesis
                problem should be a real or a binary vector.
            - nonidling_penalty_coeff: Coefficient to scale the nonidling penalty.
            - safety_penalty_coeff: Coefficient to scale the safety stock penalty.
        :param strategic_idling_class: Class to be used to make strategic idling decisions (e.g.,
            'StrategicIdlingCore', 'StrategicIdlingHedging', 'StrategicIdlingForesight', etc.)
        :param demand_planning_params: Tuple of parameters that specify demand_planning_class and
            its parameters (if any).
        :param name: Agent identifier used when storing the results.
        :param debug_info: Flat to print instantaneous calculations useful for debugging.
        :param agent_seed: Random seed for agent's random number generator.
        :param mpc_seed: Random seed for MPC policy's random number generator.
        """
        mpc_policy_class = self.get_mpc_policy_class(hedgehog_hyperparams.mpc_policy_class_name)
        mpc_policy_object = mpc_policy_class(
            env.physical_constituency_matrix,
            env.job_generator.buffer_processing_matrix,
            mpc_seed
        )
        super().__init__(
            env,
            mpc_policy_object,
            discount_factor,
            workload_relaxation_params,
            hedgehog_hyperparams,
            asymptotic_covariance_params,
            strategic_idling_params,
            policy_params,
            strategic_idling_class,  # TODO: Get from class name in hyperparams.
            demand_planning_params,
            name,
            debug_info,
            agent_seed
        )

    @staticmethod
    def get_mpc_policy_class(class_name: str) -> Type:
        """
        Return MPC policy class from class name.

        :param class_name: String with MPC policy class name.
        :return: MPC policy class.
        """
        return FoxMpcPolicy

    def _fill_init_big_step_policy_kwargs(self):
        """
        Build dictionary of parameters to be passed when constructing a BigStepLayeredPolicy object.

        :return: Dictionary of parameters.
        """
        kwargs_init = {
            'cost_per_buffer': self.env.cost_per_buffer,
            'constituency_matrix': self.env.constituency_matrix,
            'demand_rate': self.env.job_generator.demand_rate,
            'buffer_processing_matrix': self.env.job_generator.buffer_processing_matrix,
            'workload_mat': self.workload_tuple.workload_mat,
            'nu': self.workload_tuple.nu,
            'list_boundary_constraint_matrices': self.list_boundary_constraint_matrices,
            'policy_params': self.policy_params,
            'debug_info': self.debug_info
        }
        return kwargs_init

    def serialise_get_policy_kwargs(self, **kwargs):
        """
        Serialise the parameters needed to get the activity rates policy. Depending on the
        activity rates class, it passes different parameters.
        It allows: BigStepLayeredPolicy, BigStepPolicy and BigStepSurplusLayeredPolicy classes.

        :return: Dictionary of parameters.
        """
        if self.activity_rates_policy_class == BigStepLayeredPolicy:
            return self._fill_get_policy_big_step_layered_policy_kwargs(**kwargs)
        elif self.activity_rates_policy_class in [BigStepPolicy, BigStepSurplusLayeredPolicy]:
            return self._fill_get_policy_big_step_policy_with_surplus_kwargs(**kwargs)
        else:
            raise ValueError('activity_rates_policy_class not recognised.')

    def reset_mpc_variables(self, **kwargs) -> None:
        self.mpc_variables["sum_actions"] = np.round(max(1, self.num_steps_to_recompute_policy)
                                                     * self.current_policy)
        if self.debug_info:
            print("Current policy: ", self.current_policy.ravel())
            print("Current sum of actions: ", self.mpc_variables["sum_actions"].ravel())
            if "total_fluid_sum_actions" in self.mpc_variables:
                print("Total sum of actions: ", self.mpc_variables["total_sum_actions"].ravel())
                print("Total sum of fluid actions",
                      self.mpc_variables["total_fluid_sum_actions"].ravel().astype(int))
                print("Actual to fluid ratio: ",
                      (self.mpc_variables["total_sum_actions"] /
                       self.mpc_variables["total_fluid_sum_actions"]).flatten(), "\n")

    def update_mpc_variables(self, **kwargs) -> None:
        assert "actions" in kwargs
        actions = kwargs["actions"]
        self.mpc_variables["sum_actions"] -= actions

        # The section below collects diagnostic information and is not used by the policy
        if "total_sum_actions" in self.mpc_variables:
            self.mpc_variables["total_sum_actions"] += actions
            self.mpc_variables["total_timesteps"] += 1
        else:
            self.mpc_variables["total_sum_actions"] = actions.copy()
            self.mpc_variables["total_timesteps"] = 1
        # The section above collects diagnostic information and is not used by the policy
        print(self.mpc_variables["total_sum_actions"].ravel())
        print(np.sqrt(self.mpc_variables["total_sum_actions"][4]/self.mpc_variables["total_sum_actions"][3]))

    def query_hedgehog_policy(self,
                              state: types.StateSpace,
                              env: crw.ControlledRandomWalk,
                              safety_stocks_vec: types.ResourceSpace,
                              draining_time_solver: str,
                              reporter: Optional[rep.Reporter]) \
            -> Tuple[types.ActionProcess, int]:
        """
        Return activity rates for the current state and their horizon.

        :param state: current state.
        :param env: the environment specifying the topology and constraints.
        :param safety_stocks_vec: Safety stocks vector.
        :param draining_time_solver: Convex solver for computing the minimal draining time.
        :param reporter: reporter to store all data.
        :return: (z_star, horizon)
            - z_star: Activity rates.
            - horizon: Horizon for which the activity rates have been computed.
        """
        strategic_idling_tuple = self.strategic_idling_object.get_allowed_idling_directions(state,
                                                                                            safety_stocks_vec)

        draining_bottlenecks = get_dynamic_bottlenecks(
            strategic_idling_tuple.w, self.workload_tuple.workload_mat, self.workload_tuple.load)

        horizon = 1

        # Find activity rates for some horizon given nonidling and safety stock penalties.
        kwargs = {
            'state': state,
            'safety_stocks_vec': safety_stocks_vec,
            'x_eff': strategic_idling_tuple.x_eff,
            'x_star': strategic_idling_tuple.x_star,
            'w_star': strategic_idling_tuple.w_star-strategic_idling_tuple.w,
            'k_idling_set': strategic_idling_tuple.k_idling_set,
            'draining_bottlenecks': draining_bottlenecks,
            'horizon': horizon,
            'demand_plan': self.get_demand_plan()
        }
        kwargs_get_policy = self.serialise_get_policy_kwargs(**kwargs)

        if self.debug_info:
            print(f"horizon: {horizon}")
            print(f"z_star: {np.squeeze(z_star)}")

        if reporter is not None:
            stored_vars = {'strategic_idling_tuple': strategic_idling_tuple, 'horizon': horizon}
            reporter.store(**stored_vars)

        return None, None, kwargs


    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Returns actions (possibly many) given current state. Can take a kwarg dictionary
        of overriding arguments that may be policy specific.

        :param state: Current state of the system.
        :return: Schedule of actions.
        """
        # Compute safety stock target.
        safety_stocks_vec = safety_stocks.obtain_safety_stock_vector(
            self.theta, self.load_ph, self.sigma_2_ph, state, self.debug_info)
        if self.env.model_type == "pull":
            safety_stocks_vec += self.obtain_safety_stock_for_surplus_buffers()

        # Recompute activity rates
        args = {
            "state": state,
            "env": self.env,
            "safety_stocks_vec": safety_stocks_vec,
            "draining_time_solver": self.policy_params.convex_solver,
            "reporter": None
        }
        args.update(override_args)
        _, _, kwargs = self.query_hedgehog_policy(**args)
        self.reset_mpc_variables()

        # Store retrospectively the actual number of MPC steps performed in the previous
        # iteration before recomputing the activity rates. We skip zero since it corresponds
        # with the first iteration, before actually having performed any action yet.
        if self.actual_num_mpc_steps > 0 and args['reporter'] is not None:
            stored_vars = {'num_mpc_steps': self.actual_num_mpc_steps}
            args['reporter'].store(**stored_vars)
            self.actual_num_mpc_steps = 0

        r_idling_set = self._get_resource_idling_set(kwargs['k_idling_set'], kwargs['draining_bottlenecks'])
        draining_resources = set()
        for w_dir in kwargs['draining_bottlenecks']:
            draining_resources = draining_resources.union(self.w_dirs_to_resources[w_dir])

        # Obtain physically feasible actions from MPC policy.
        actions = self.mpc_policy.obtain_actions(
            state=state,
            x_star = kwargs['x_star'],
            x_eff = kwargs['x_eff'],
            w_star = kwargs['w_star'],
            r_idling_set = r_idling_set,
            draining_resources = draining_resources)
        actions.setflags(write=False)

        # Update remaining number of actions to be performed, countdown before recomputing activity
        # rates, and actual number of steps following the current activity rates.
        self.update_mpc_variables(actions=actions)
        self.actual_num_mpc_steps += 1

        return actions

    def _get_resource_idling_set(self,k_idling_set, draining_bottlenecks):
        r_idling_set = set()
        for w_dir in k_idling_set:
            if w_dir in draining_bottlenecks:
                continue
            r_idling_set = r_idling_set.union(self.w_dirs_to_resources[w_dir])

        return r_idling_set
