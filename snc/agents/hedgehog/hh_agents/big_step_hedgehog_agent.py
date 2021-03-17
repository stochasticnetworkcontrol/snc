import numpy as np
from typing import Optional, Type, Union, Dict

from snc.agents.activity_rate_to_mpc_actions.feedback_mip_feasible_mpc_policy import \
    FeedbackMipFeasibleMpcPolicy
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


class BigStepHedgehogAgent(HedgehogAgentInterface):
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
        classes = [
            FeedbackStationaryFeasibleMpcPolicy,
            FeedbackMipFeasibleMpcPolicy
        ]
        return get_class_from_name(class_name, classes, 'MPC policy')

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

    def _fill_init_big_step_policy_with_surplus_kwargs(self):
        """
        Build dictionary of parameters to be passed when constructing a BigStepSurplusLayeredPolicy
        object.

        :return: Dictionary of parameters.
        """
        kwargs_init = self._fill_init_big_step_policy_kwargs()
        kwargs_init['ind_surplus_buffers'] = self.env.ind_surplus_buffers
        return kwargs_init

    def serialise_init_policy_kwargs(self):
        """
        Serialise the parameters for the initialisation of the activity rates class. Depending on
        the activity rates class, it passes different parameters.
        It allows: BigStepLayeredPolicy, BigStepPolicy and BigStepSurplusLayeredPolicy classes.

        :return: Dictionary of parameters.
        """
        if self.activity_rates_policy_class == BigStepLayeredPolicy:
            return self._fill_init_big_step_policy_kwargs()
        elif self.activity_rates_policy_class in [BigStepPolicy, BigStepSurplusLayeredPolicy]:
            return self._fill_init_big_step_policy_with_surplus_kwargs()
        else:
            raise ValueError('activity_rates_policy_class not recognised.')

    @staticmethod
    def _fill_get_policy_big_step_layered_policy_kwargs(**kwargs) -> Dict:
        """
        Build dictionary of parameters to be passed when constructing a BigStepLayeredPolicy object.

        :return: Dictionary of parameters.
        """
        args_get_policy = {
            'state': kwargs['state'],
            'safety_stocks_vec': kwargs['safety_stocks_vec'],
            'k_idling_set': kwargs['k_idling_set'],
            'draining_bottlenecks': kwargs['draining_bottlenecks'],
            'horizon': kwargs['horizon']
        }
        return args_get_policy

    @staticmethod
    def _fill_get_policy_big_step_policy_with_surplus_kwargs(**kwargs) -> Dict:
        """
        Build dictionary of parameters to be passed when constructing a BigStepSurplusLayeredPolicy
        object.

        :return: Dictionary of parameters.
        """
        args_get_policy = BigStepHedgehogAgent._fill_get_policy_big_step_layered_policy_kwargs(
            **kwargs
        )
        args_get_policy['demand_plan'] = kwargs['demand_plan']
        return args_get_policy

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
            self.mpc_variables["total_fluid_sum_actions"] += self.current_policy
            self.mpc_variables["total_timesteps"] += 1
        else:
            self.mpc_variables["total_sum_actions"] = actions.copy()
            self.mpc_variables["total_fluid_sum_actions"] = self.current_policy.copy()
            self.mpc_variables["total_timesteps"] = 1
        # The section above collects diagnostic information and is not used by the policy

    @staticmethod
    def get_horizon(**kwargs) -> int:
        """
        Returns the size of the big step, i.e. the horizon, for which the big step policy will
        compute a schedule. Such size is returned as an integer number of time steps, which is
        computed as a proportion of the minimal draining time for the current state. If the
        proportion coefficient ('horizon_drain_time_ratio') is zero, then it returns one.
        (Implements abstract method).

        :return: horizon: Horizon as a number of time steps. We force: horizon >= minimum_horizon.
        """
        assert "state" in kwargs, "BigStepActivityRatesHedgehogAgent requires a 'state' keyword."
        assert "env" in kwargs, "BigStepActivityRatesHedgehogAgent requires an 'env' keyword."
        assert "horizon_drain_time_ratio" in kwargs, "BigStepActivityRatesHedgehogAgent requires " \
                                                     "a 'horizon_drain_time_ratio' keyword."
        assert "convex_solver" in kwargs, "BigStepActivityRatesHedgehogAgent requires a " \
                                          "'convex_solver' keyword."
        assert "minimum_horizon" in kwargs, "BigStepActivityRatesHedgehogAgent requires a" \
                                            "'minimum_horizon' keyword."
        state = kwargs['state']  # Current state of the environment
        env = kwargs['env']  # Environment.
        # Ration horizon over minimal draining time.
        horizon_drain_time_ratio = kwargs['horizon_drain_time_ratio']
        # Convex solver for computing the minimal draining time.
        convex_solver = kwargs['convex_solver']

        # Minimum number of time steps when computing the big step activity rates policy.
        minimum_horizon = kwargs['minimum_horizon']
        # Tolerance to check that minimal draining time is nonnegative.
        eps = kwargs.get('eps', 1)

        if horizon_drain_time_ratio == 0:  # If want to recompute the activity rates after a fixed
            # number of time steps (possible one, as given by minimum_horizon).
            horizon = int(minimum_horizon)
        else:  # Compute horizon from minimal draining time.
            min_draining_time = mdt.compute_dual_minimal_draining_time_cvxpy(
                state, env.constituency_matrix, env.job_generator.demand_rate,
                env.job_generator.buffer_processing_matrix, convex_solver)
            min_draining_time *= env.job_generator.sim_time_interval
            assert min_draining_time >= -eps
            horizon = int(max([minimum_horizon,
                               np.ceil(horizon_drain_time_ratio * min_draining_time)]))
        return horizon
