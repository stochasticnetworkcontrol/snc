from snc.agents.hedgehog.hh_agents.\
    pure_feedback_stationary_hedgehog_agent import PureFeedbackStationaryHedgehogAgent
from snc.agents.hedgehog.params import BigStepPenaltyPolicyParams


class PureFeedbackMIPHedgehogAgent(PureFeedbackStationaryHedgehogAgent):
    @staticmethod
    def validate_policy_params(policy_params: BigStepPenaltyPolicyParams) -> None:
        """
        Assert that the boolean constraint flag is True.

        :param policy_params: Tuple of parameters that specify the solver to be used by the policy.
        :return: None.
        """

        assert policy_params.boolean_action_flag, \
            f"MIP requires boolean_action_flag = True, so activity rates are binary," \
            f"but provided: {policy_params.boolean_action_flag}."
