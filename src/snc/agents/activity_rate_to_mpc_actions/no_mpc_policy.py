import numpy as np

from snc.agents.activity_rate_to_mpc_actions.action_mpc_policy import ActionMPCPolicy
from snc.utils import snc_tools
import snc.utils.snc_types as types


class NoMPCPolicy(ActionMPCPolicy):
    """
    This class is used to bypass the MPC policy, i.e., it outputs an action exactly equal to the
    input activity rates. This is useful when the input activity rates are already binary and
    satisfy the physical constraints, e.g., with the Pure Feedback Hedgehog agent.
    """

    def obtain_actions(self, **kwargs) -> types.ActionProcess:
        """
        Check that activity rates are already boolean and return then back.

        :param kwargs: Dictionary that can contain:
            - 'z_star': Activity rates with boolean values representing the actions to be executed
                with probability zero or one.
        :return: The same input activity rates.
        """
        z_star = kwargs['z_star']
        assert snc_tools.is_approx_binary(z_star), f"Activity rates must be binary, but " \
                                                   f"provided: {z_star}."
        assert np.all(self.constituency_matrix @ z_star <= 1)
        return z_star
