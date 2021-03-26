from typing import Optional, Type
from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
from snc.agents.hedgehog.params import (
    AsymptoticCovarianceParams,
    BigStepPenaltyPolicyParams,
    DemandPlanningParams,
    HedgehogHyperParams,
    StrategicIdlingParams,
    WorkloadRelaxationParams
)
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.environments.controlled_random_walk import ControlledRandomWalk


class PureFeedbackStationaryHedgehogAgent(BigStepHedgehogAgent):
    def __init__(self,
                 env: ControlledRandomWalk,
                 discount_factor: float,
                 workload_relaxation_params: WorkloadRelaxationParams,
                 hedgehog_hyperparams: HedgehogHyperParams,
                 asymptotic_covariance_params: AsymptoticCovarianceParams,
                 strategic_idling_params: StrategicIdlingParams = StrategicIdlingParams(
                     'StrategicIdlingHedging', 'cvx.CPLEX', 0.01, 0.01, 1, 1e-5),
                 policy_params: BigStepPenaltyPolicyParams = BigStepPenaltyPolicyParams(
                     'cvx.CPLEX', False, 100, 1000),
                 strategic_idling_class: Type[StrategicIdlingCore] = StrategicIdlingHedging,
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
        :param hedgehog_hyperparams: Tuple of parameters defining penalties
            for safety stocks.
        :param asymptotic_covariance_params: Tuple of parameters used to
            generate an estimate of the asymptotic covariance matrix.
        :param strategic_idling_params: Tuple of parameters that specify the solver to be used at
            the different steps when finding potential idling directions, as well as some tolerance
            parameters.
        :param policy_params: Tuple of parameters that specify the solver to be used by the policy.
        :param strategic_idling_class: Class to be used to make strategic idling decisions. Pure
            feedback Hedgehog only accepts StrategicIdlingCore or StrategicIdlingHedging.
        :param name: Agent identifier.
        :param debug_info: Flat to print instantaneous calculations useful for debugging.
        :param agent_seed: Random seed for agent's random number generator.
        :param mpc_seed: Random seed for MPC policy's random number generator.
        """
        self.validate_params(hedgehog_hyperparams, policy_params, strategic_idling_class)

        super().__init__(
            env,
            discount_factor,
            workload_relaxation_params,
            hedgehog_hyperparams,
            asymptotic_covariance_params,
            strategic_idling_params,
            policy_params,
            strategic_idling_class,
            demand_planning_params,
            name,
            debug_info,
            agent_seed,
            mpc_seed
        )

    @staticmethod
    def validate_policy_params(policy_params: BigStepPenaltyPolicyParams) -> None:
        """

        :param policy_params: Tuple of parameters that specify the solver to be used by the policy.
        :return: None.
        """
        assert not policy_params.boolean_action_flag, \
            f"Stationary MPC requires boolean_action_flag = False, so that activity rates are not" \
            f" constrained to binary vector, but provided: {policy_params.boolean_action_flag}."

    @staticmethod
    def validate_hedgehog_hyperparams(hedgehog_hyperparams: HedgehogHyperParams) -> None:
        """
        Assert the horizon parameters are for pure feedback.

        :param hedgehog_hyperparams: Tuple of parameters defining penalties for safety stocks.
        :return: None.
        """
        assert hedgehog_hyperparams.horizon_drain_time_ratio == 0, \
            f"PF Hedgehog agents require horizon_drain_time_ratio = 0, but provided: " \
            f"{hedgehog_hyperparams.horizon_drain_time_ratio}."

        assert hedgehog_hyperparams.horizon_mpc_ratio == 0, \
            f"PF Hedgehog agents require 'horizon_mpc_ratio' = 0, but provided: " \
            f"{hedgehog_hyperparams.horizon_mpc_ratio}."

        assert hedgehog_hyperparams.minimum_horizon == 1, \
            f"PF Hedgehog agents require 'horizon_mpc_ratio' = 0, but provided: " \
            f"{hedgehog_hyperparams.minimum_horizon}."

    def validate_params(self, hedgehog_hyperparams: HedgehogHyperParams,
                        policy_params: BigStepPenaltyPolicyParams,
                        strategic_idling_class: Type[StrategicIdlingCore]) -> None:
        """
        Assert that agent parameters are valid for this agent.

        :param hedgehog_hyperparams: Tuple of parameters defining penalties for safety stocks.
        :param policy_params: Tuple of parameters that specify the solver to be used by the policy.
        :param strategic_idling_class: Class to be used to make strategic idling decisions. Pure
            feedback Hedgehog only accepts StrategicIdlingCore or StrategicIdlingHedging.
        :return: None.
        """
        self.validate_policy_params(policy_params)
        self.validate_hedgehog_hyperparams(hedgehog_hyperparams)
        assert strategic_idling_class in (StrategicIdlingCore, StrategicIdlingHedging)

    @staticmethod
    def get_horizon(**kwargs) -> int:
        """
        Returns always one.

        :return: constant value 1.
        """
        return 1
