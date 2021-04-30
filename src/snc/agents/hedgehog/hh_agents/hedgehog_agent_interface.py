from abc import abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Tuple, Type, Union

from snc.agents.activity_rate_to_mpc_actions.action_mpc_policy import ActionMPCPolicy
from snc.agents.agent_interface import AgentInterface
from snc.agents.hedgehog import safety_stocks
from snc.agents.hedgehog.asymptotic_workload_cov.estimate_asymptotic_cov \
    import EstimateAsymptoticWorkloadCovBatchMeans
from snc.agents.hedgehog.class_loader import get_class_from_name
from snc.agents.hedgehog.params import \
    AsymptoticCovarianceParams, \
    BigStepLayeredPolicyParams, \
    BigStepPenaltyPolicyParams, \
    HedgehogHyperParams, \
    StrategicIdlingParams, \
    WorkloadRelaxationParams, DemandPlanningParams
from snc.agents.hedgehog.policies.big_step_base_policy import BigStepBasePolicy
from snc.agents.hedgehog.policies.big_step_layered_policy import BigStepLayeredPolicy
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_foresight import StrategicIdlingForesight
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedgehog_gto import \
    StrategicIdlingHedgehogGTO, \
    StrategicIdlingHedgehogGTO2, \
    StrategicIdlingHedgehogNaiveGTO, \
    StrategicIdlingGTO
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import get_dynamic_bottlenecks
from snc.agents.hedgehog.workload.workload import \
    compute_load_workload_matrix, \
    WorkloadTuple
from snc.agents.steady_state_agents.steady_state_policy_agent import SteadyStatePolicyAgent
from snc.demand_planning.constant_demand_plan import ConstantDemandPlan
from snc.demand_planning.demand_plan_interface import DemandPlanInterface
from snc.environments import controlled_random_walk as crw
import snc.simulation.store_data.reporter as rep
from snc.simulation.store_data.numpy_encoder import clean_to_serializable
from snc.utils import snc_types as types


class HedgehogAgentInterface(AgentInterface):
    """
    Simulate a whole hedgehog policy on an environment.
    """

    def __init__(self,
                 env: crw.ControlledRandomWalk,
                 mpc_policy: ActionMPCPolicy,
                 discount_factor: float,
                 workload_relaxation_params: WorkloadRelaxationParams,
                 hedgehog_hyperparams: HedgehogHyperParams,
                 asymptotic_covariance_params: AsymptoticCovarianceParams,
                 strategic_idling_params: StrategicIdlingParams,
                 policy_params: Union[BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams],
                 strategic_idling_class: Type[StrategicIdlingCore],
                 demand_planning_params: DemandPlanningParams,
                 name: str,
                 debug_info: bool,
                 agent_seed: Optional[int]) -> None:
        """
        :param env: Environment to stepped through.
        :param mpc_policy: MPC policy object that maps activity rates to actions.
        :param discount_factor: Discount factor for the cost per time step.
        :param workload_relaxation_params: Tuple of parameters defining the
            first workload relaxation.
        :param hedgehog_hyperparams: Tuple of parameters defining penalties
            for safety stocks.
        :param asymptotic_covariance_params: Tuple of parameters used to
            generate an estimate of the asymptotic covariance matrix.
        :param strategic_idling_params: Tuple of parameters that specify the solver to be used at
            the different steps when finding potential idling directions, as well as some tolerance
            parameters.
        :param policy_params: Tuple of parameters to be used by the activity rates policy.
        :param strategic_idling_class: Class to be used to make strategic idling decisions (e.g.,
            'StrategicIdlingCore', 'StrategicIdlingHedging', 'StrategicIdlingForesight', etc.)
        :param demand_planning_params: Tuple of parameters that specify demand_planning_class and
            its parameters (if any).
        :param name: Agent identifier.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        :param: agent_seed: Agent random seed.
        """
        super().__init__(env, name, agent_seed)

        # Parameters.
        self.mpc_policy = mpc_policy

        assert 0 <= discount_factor <= 1
        self.discount_factor = discount_factor
        self.neg_log_discount_factor = - np.log(discount_factor)

        self.workload_relaxation_params = workload_relaxation_params
        self.hedgehog_hyperparams = hedgehog_hyperparams
        self.asymptotic_covariance_params = asymptotic_covariance_params
        self.policy_params = policy_params
        self.strategic_idling_params = strategic_idling_params

        assert strategic_idling_class in [StrategicIdlingCore,
                                          StrategicIdlingForesight,
                                          StrategicIdlingGTO,
                                          StrategicIdlingHedgehogGTO,
                                          StrategicIdlingHedgehogGTO2,
                                          StrategicIdlingHedgehogNaiveGTO,
                                          StrategicIdlingHedging], \
            f"Provided strategic_idling_class not valid: {strategic_idling_class}."
        self.strategic_idling_class = strategic_idling_class

        self.theta = hedgehog_hyperparams.theta_0 * np.ones((env.num_resources, 1))
        self.debug_info = debug_info

        self.activity_rates_policy_class = self.get_activity_rates_policy_class(
            hedgehog_hyperparams.activity_rates_policy_class_name
        )
        self.demand_planner = self.initialise_constant_demand_planner(demand_planning_params)
        self.asymptotic_workload_cov_estimator = EstimateAsymptoticWorkloadCovBatchMeans(env)

        # To be updated when performing offline calculations.
        self.workload_tuple: Optional[WorkloadTuple] = None
        self.workload_cov: Optional[types.WorkloadCov] = None

        self.load_ph: Optional[types.ResourceSpace] = None
        self.sigma_2_ph: Optional[types.ResourceSpace] = None

        self.policy_obj: BigStepBasePolicy = None
        self.strategic_idling_object: Optional[StrategicIdlingCore] = None

        self.current_policy: Optional[types.ActionSpace] = None
        self.mpc_variables: Dict[str, Any] = dict()
        self.num_steps_to_recompute_policy: Optional[int] = None
        self.actual_num_mpc_steps: Optional[int] = None
        self.initialise_policy_and_counters()

    @abstractmethod
    def reset_mpc_variables(self, **kwargs):
        """
        Each agent can initialise various state variables to be used by the MPC policy of each
        specific HH agent implementation.
        """

    @abstractmethod
    def update_mpc_variables(self, **kwargs):
        """
        Each agent can update various state variables to be used by the MPC policy of each
        specific HH agent implementation.
        """

    @staticmethod
    @abstractmethod
    def get_horizon(**kwargs) -> int:
        """
        Returns the size of the big step, i.e. the horizon, for which the big step policy will
        compute a schedule.

        :return: num_time_steps: Horizon as a number of time steps.
        """

    @staticmethod
    def get_activity_rates_policy_class(class_name: str) -> Type:
        """
        Return BigStep policy class from class name.

        :param class_name: String with BigStep policy class name.
        :return: BigStep policy class.
        """
        classes = [
            BigStepLayeredPolicy,
            BigStepPolicy,
        ]
        return get_class_from_name(class_name, classes, 'Activity rates BigStep policy')

    @staticmethod
    def get_demand_planning_class(class_name: str) -> Type:
        """
        Return demand planning class from class name.

        :param class_name: String with demand planning class name.
        :return: Demand planning policy class.
        """
        classes = [
            ConstantDemandPlan,
        ]
        return get_class_from_name(class_name, classes, 'Demand planning')

    def initialise_constant_demand_planner(
            self,
            demand_planning_params: DemandPlanningParams) -> Optional[DemandPlanInterface]:
        """
        Initialise demand planning object without parameters (at least for now), which will yield
        the surplus target levels for pull models.

        :param demand_planning_params: Tuple of parameters that specify demand_planning_class and
            its parameters (if any).
        :return: demand planning object.
        """
        if self.env.model_type == 'push':
            return None

        assert self.env.model_type == 'pull'
        demand_planning_class = self.get_demand_planning_class(
            demand_planning_params.demand_planning_class_name
        )
        assert demand_planning_class in [ConstantDemandPlan], \
            f"Provided demand_planning_class not valid: {demand_planning_class}."
        assert hasattr(demand_planning_params, 'params_dict'), \
            "Missing 'params_dict' for DemandPlanning class constructor."
        assert 'ind_surplus_buffers' in demand_planning_params.params_dict, \
            "Missing 'ind_surplus_buffers' in 'params_dict'."
        return demand_planning_class(**demand_planning_params.params_dict)

    def serialise_init_policy_kwargs(self) -> Dict:
        """
        Build dictionary of parameters to be passed when constructing the policy object, which is
        done after performing the workload tuple offline computations.
        This method is meant to be overloaded by agents that use policies that require more or
        fewer parameters.

        :return: Dictionary of parameters.
        """
        raise NotImplementedError('serialise_init_policy_kwargs method is meant to be overloaded.')

    def initialise_policy_and_counters(self) -> None:
        """
        Initialise current activity rates and counters to zero.

        :return: None.
        """
        self.current_policy = np.zeros((self.env.num_activities, 1))  # Current activity rates.
        self.num_steps_to_recompute_policy = 0  # Triggers recomputing big step policy LP.
        self.actual_num_mpc_steps = 0

    def _initialize_strategic_idling_object(self) -> None:
        """
        Initialise strategic idling object.

        :return: None.
        """
        init_vars = {
            'workload_mat': self.workload_tuple.workload_mat,
            'load': self.workload_tuple.load,
            'cost_per_buffer': self.env.cost_per_buffer,
            'list_boundary_constraint_matrices': self.env.list_boundary_constraint_matrices,
            'model_type': self.env.model_type,
            'strategic_idling_params': self.strategic_idling_params,
            'debug_info': self.debug_info
        }
        if self.strategic_idling_class not in (StrategicIdlingGTO, StrategicIdlingCore):
            init_vars.update({
                'neg_log_discount_factor': self.neg_log_discount_factor,
                'workload_cov': self.workload_cov
            })
        if self.strategic_idling_class == StrategicIdlingForesight:
            init_vars.update({'policy_object': self.policy_obj})

        self.strategic_idling_object = self.strategic_idling_class(**init_vars)

    @staticmethod
    def generate_workload_tuple(env: crw.ControlledRandomWalk,
                                num_workload_vectors: int,
                                load_threshold: float,
                                solver: str) -> WorkloadTuple:
        """
        Generate the workload tuple from the first relaxation.

        :param env: Environment (defining the topology and constraints of the network).
        :param num_workload_vectors: Number of leading workload vectors to choose.
        :param load_threshold: Lower bound on load to choose workload vectors.
        :param solver: Convex optimisation solver to be called when computing a feasible point
            inside the intersection of halfspaces.
        """
        return compute_load_workload_matrix(env, num_workload_vectors, load_threshold,
                                            solver=solver)

    @staticmethod
    def map_workload_to_physical_resources(workload_tuple: WorkloadTuple,
                                           workload_cov: types.WorkloadCov) \
            -> Tuple[types.ResourceSpace, types.ResourceSpace]:
        """
        Map workloads that may correspond to more than one physical resource, to a single physical
        resource. Return a load and covariance for these resources.

        :param workload_tuple: the workload tuple for the system (load, workload mat, nu).
        :param workload_cov: the asymptotic covariance matrix for the workload.
        """

        sigma_2 = workload_cov.diagonal()  # Variance of the workload process
        load_sig = safety_stocks.map_workload_to_physical_resources_with_conservative_max_heuristic(
            workload_tuple.nu, workload_tuple.load, sigma_2)
        load_ph, sigma_2_ph = load_sig
        return load_ph, sigma_2_ph

    def perform_offline_calculations(self) -> None:
        """
        Perform required offline calculations before running the simulation.
        """
        self.workload_tuple = self.generate_workload_tuple(
            self.env,
            self.workload_relaxation_params.num_vectors,
            self.workload_relaxation_params.load_threshold,
            self.workload_relaxation_params.convex_solver
        )

        # Initialise policy, as implemented by children classes.
        self.policy_obj = self.activity_rates_policy_class(**self.serialise_init_policy_kwargs())

        self.workload_cov = self.asymptotic_workload_cov_estimator.estimate_asymptotic_workload_cov(
            self.env.job_generator.buffer_processing_matrix,
            self.workload_tuple,
            self.asymptotic_covariance_params.num_batch,
            self.asymptotic_covariance_params.num_presimulation_steps,
            SteadyStatePolicyAgent(
                self.env, type(self.mpc_policy)
            ),
            self.debug_info
        )

        self._initialize_strategic_idling_object()

        self.load_ph, self.sigma_2_ph = self.map_workload_to_physical_resources(self.workload_tuple,
                                                                                self.workload_cov)

        # Reset trigger to recompute big step policy LP and remaining set of actions (they might've
        # been set if the estimation of the asymptotic covariance was done with this self agent).
        self.initialise_policy_and_counters()

    def get_demand_plan(self) -> Optional[Dict[int, int]]:
        """
        Return demand plan if any.

        :return: Target levels for every buffer
        """
        if self.demand_planner is None:
            return None
        else:
            return self.demand_planner.get_demand_plan()

    def serialise_get_policy_kwargs(self, **kwargs):
        """
        Build dictionary of parameters to be passed when constructing the policy object, which is
        done after performing the workload tuple offline computations.
        This method is meant to be overloaded by agents that use policies that require more or
        fewer parameters.

        :return: Dictionary of parameters.
        """
        raise NotImplementedError('serialise_get_policy_kwargs method is meant to be overloaded.')

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

        # Compute horizon length as a ratio of the minimal draining time.
        horizon = self.get_horizon(
            state=state,
            env=env,
            horizon_drain_time_ratio=self.hedgehog_hyperparams.horizon_drain_time_ratio,
            convex_solver=draining_time_solver,
            minimum_horizon=self.hedgehog_hyperparams.minimum_horizon
        )

        # Find activity rates for some horizon given nonidling and safety stock penalties.
        kwargs = {
            'state': state,
            'safety_stocks_vec': safety_stocks_vec,
            'x_eff': strategic_idling_tuple.x_eff,
            'x_star': strategic_idling_tuple.x_star,
            'k_idling_set': strategic_idling_tuple.k_idling_set,
            'draining_bottlenecks': draining_bottlenecks,
            'horizon': horizon,
            'demand_plan': self.get_demand_plan()
        }
        #kwargs_get_policy = self.serialise_get_policy_kwargs(**kwargs)
        #z_star, _ = self.policy_obj.get_policy(**kwargs_get_policy)

        if self.debug_info:
            print(f"horizon: {horizon}")
            print(f"z_star: {np.squeeze(z_star)}")

        if reporter is not None:
            stored_vars = {'strategic_idling_tuple': strategic_idling_tuple, 'horizon': horizon}
            reporter.store(**stored_vars)

        return kwargs

    @staticmethod
    def get_num_steps_to_recompute_policy(current_horizon: float,
                                          horizon_mpc_ratio: float,
                                          minimum_num_steps: float) -> int:
        """
        Compute number of time steps before recomputing the the activity rates. The result will be
        integer lower bounded by parameter 'minimum_num_steps'.

        :param current_horizon: Horizon for which the activity rates remain valid.
        :param horizon_mpc_ratio: Ratio of number of MPC steps over horizon.
        :param minimum_num_steps: Minimum number of time steps before recomputing the policy. When
            using this function, this is currently equal to the minimum horizon, but could be a
            different parameter in the future.
        :return:
        """
        assert minimum_num_steps >= 1, f"Minimum number of time steps must be >= 1, but provided:" \
                                       f" {minimum_num_steps}."
        return int(np.maximum(np.ceil(current_horizon * horizon_mpc_ratio), minimum_num_steps))

    def obtain_safety_stock_for_surplus_buffers(self):
        pass

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Returns actions (possibly many) given current state. Can take a kwarg dictionary
        of overriding arguments that may be policy specific.

        :param state: Current state of the system.
        :return: Schedule of actions.
        """

        # If any resource is starving or countdown ends, then recompute activity rates.
        if True:#self.num_steps_to_recompute_policy == 0:

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
            #self.current_policy, current_horizon =
            kwargs = self.query_hedgehog_policy(**args)
            # Reset countdown timer to recomputing the activity rates.
            #self.num_steps_to_recompute_policy = self.get_num_steps_to_recompute_policy(
                #current_horizon,
                #self.hedgehog_hyperparams.horizon_mpc_ratio,
                #self.hedgehog_hyperparams.minimum_horizon
            #)
            # Reset number times each action has to be performed before recomputing activity rates.
            self.reset_mpc_variables()

            # Store retrospectively the actual number of MPC steps performed in the previous
            # iteration before recomputing the activity rates. We skip zero since it corresponds
            # with the first iteration, before actually having performed any action yet.
            if self.actual_num_mpc_steps > 0 and args['reporter'] is not None:
                stored_vars = {'num_mpc_steps': self.actual_num_mpc_steps}
                args['reporter'].store(**stored_vars)
                self.actual_num_mpc_steps = 0

        # Obtain physically feasible actions from MPC policy.
        actions = self.mpc_policy.obtain_actions(
            state=state,
            x_star = kwargs['x_star'],
            x_eff = kwargs['x_eff'],
            k_idling_set = kwargs['k_idling_set'],
            mpc_variables=self.mpc_variables,
            num_steps_to_recompute_policy=self.num_steps_to_recompute_policy,
            z_star=self.current_policy,
            demand_rate=self.env.job_generator.demand_rate
        )
        actions.setflags(write=False)

        # Update remaining number of actions to be performed, countdown before recomputing activity
        # rates, and actual number of steps following the current activity rates.
        self.update_mpc_variables(actions=actions)
        self.num_steps_to_recompute_policy -= 1
        self.actual_num_mpc_steps += 1

        return actions

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        return clean_to_serializable(self)
