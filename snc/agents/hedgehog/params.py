from collections import namedtuple


AsymptoticCovarianceParams = namedtuple(
    'AsymptoticCovarianceParams',
    (
        'num_batch',
        'num_presimulation_steps'
    ),
    defaults=(
        200,  # num_batch
        10000  # num_presimulation_steps
    )
)

BigStepLayeredPolicyParams = namedtuple(
    'BigStepLayeredPolicyParams',
    (
        'convex_solver',
    ),
    defaults=(
        "cvx.CPLEX",
    )  # convex_solver
)

BigStepPenaltyPolicyParams = namedtuple(
    'BigStepPenaltyPolicyParams',
    (
        'convex_solver',
        'boolean_action_flag',  # Flag indicating if activity rates must be binary.
        'nonidling_penalty_coeff',  # Coefficient to scale the nonidling penalty.
        'safety_penalty_coeff',  # Coefficient to scale the safety stock penalty.
    ),
    defaults=(
        "cvx.CPLEX",  # convex_solver
        False,  # boolean_action_flag
        1000,  # nonidling_penalty_coeff
        10,  # safety_penalty_coeff
    )
)

DemandPlanningParams = namedtuple(
    'DemandPlanningParams',
    (
        'demand_planning_class_name',  # Demand planning class name for setting surplus buffers
                                       # target stock levels. Only needed for pull models.
        'params_dict',  # Dictionary with the parameters for constructing the DemandPlanning object.
    ),
    defaults=(
        None,  # demand_planning_class_name
        dict(),  # params_dict
    )
)

HedgehogHyperParams = namedtuple(
    'HedgehogHyperParams',
    (
        'activity_rates_policy_class_name',  # Big step policy class name.
        "mpc_policy_class_name",  # MPC policy class name.
        'theta_0',  # Tuning parameter to compute the safety stock threshold.
        "horizon_drain_time_ratio",  # Ratio num steps horizon over the minimal draining time.
        "horizon_mpc_ratio",  # Ratio num steps to follow the activity rates over horizon.
        "minimum_horizon",  # Minimum horizon length i.e. minimum step size of the big step policy.
    ),
    defaults=(
        'BigStepLayeredPolicy',  # activity_rates_policy_class_name
        'FeedbackStationaryFeasibleMpcPolicy',  # mpc_policy_class_name
        0.5,  # theta_0
        0,  # horizon_drain_time_ratio
        1,  # horizon_mpc_ratio
        100,  # minimum_horizon
    )
)

StrategicIdlingParams = namedtuple(
    'StrategicIdlingParams',
    (
        'strategic_idling_class',
        'convex_solver',
        'epsilon',
        'shift_eps',
        'hedging_scaling_factor',
        'penalty_coeff_w_star'
    ),
    defaults=(
        'StrategicIdlingForesight',  # strategic_idling_class
        'cvx.CPLEX',  # convex_solver
        0.05,  # epsilon
        1e-2,  # shift_eps
        1,  # hedging_scaling_factor
        1e-5  # penalty_coeff_w_star
    )
)

WorkloadRelaxationParams = namedtuple(
    'WorkloadRelaxationParams',
    (
        'num_vectors',
        'load_threshold',
        'convex_solver'
    ),
    defaults=(
        None,  # num_vectors
        None,  # load_threshold
        'cvx.CPLEX'  # convex_solver
    )
)
