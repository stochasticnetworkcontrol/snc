from typing import Optional, Union

from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
from snc.agents.hedgehog.params import AsymptoticCovarianceParams, \
    BigStepLayeredPolicyParams, \
    BigStepPenaltyPolicyParams, \
    HedgehogHyperParams, \
    WorkloadRelaxationParams, DemandPlanningParams
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedgehog_gto import StrategicIdlingGTO
from snc.environments.controlled_random_walk import ControlledRandomWalk


class BigStepHedgehogGTOAgent(BigStepHedgehogAgent):
    def __init__(self,
                 env: ControlledRandomWalk,
                 discount_factor: float,
                 workload_relaxation_params: WorkloadRelaxationParams,
                 hedgehog_hyperparams: HedgehogHyperParams,
                 asymptotic_covariance_params: AsymptoticCovarianceParams,
                 policy_params: Optional[
                     Union[BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams]] = None,
                 demand_planning_params: DemandPlanningParams = DemandPlanningParams(),
                 name: str = "BigStepHedgehogGTOAgent",
                 debug_info: bool = False,
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        See the description of class parameters in the docstring of parent class
        `BigStepHedgehogAgent`. All of them are exactly the same but `strategic_idling_params` is
        excluded as irrelevant for this type of agent.
        """
        super().__init__(env=env,
                         discount_factor=discount_factor,
                         workload_relaxation_params=workload_relaxation_params,
                         hedgehog_hyperparams=hedgehog_hyperparams,
                         asymptotic_covariance_params=asymptotic_covariance_params,
                         policy_params=policy_params,
                         strategic_idling_class=StrategicIdlingGTO,
                         demand_planning_params=demand_planning_params,
                         name=name,
                         debug_info=debug_info,
                         agent_seed=agent_seed,
                         mpc_seed=mpc_seed)
