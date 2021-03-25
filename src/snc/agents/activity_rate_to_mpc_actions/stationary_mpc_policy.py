import numpy as np
from typing import Optional

from src.snc import ActionMPCPolicy
import src.snc.agents.activity_rate_to_mpc_actions.mpc_utils as mpc_utils
from src import snc as agents_utils, snc as types


class StationaryActionMPCPolicy(ActionMPCPolicy):
    """
    This class is used to obtain actions by considering policy as a probability distribution over
    the action set. The probability distribution is normalised for each resource independently.
    This class doesn't use feedback of how many actions of each kind have already been generated.
    Hence, the distance between the desired activity rates (z_star) and the actual frequency depends
    on the number of samples. More formally, if we think the frequency of the actual actions as a
    sample estimate of the rates, the variance of such estimate decreases as sqrt(T), where T is the
    number of samples: E(z_star - actual frequency)^2 = O(T).
    """

    def __init__(self, physical_constituency_matrix: types.ConstituencyMatrix,
                 mpc_seed: Optional[int] = None) -> None:
        """
        :param physical_constituency_matrix: Matrix with those rows from the constituency matrix
            that represent physical resources, so that it must have a single one per column. If the
            constituency_matrix of the problem has multiple ones per column (e.g. the "3x3 input
            queued switch"), then the matrix should not be used with this class.
        :param mpc_seed: MPC policy random seed.
        """
        # Ensure that constituency_matrix has a single one per column.
        assert agents_utils.has_orthogonal_rows(physical_constituency_matrix), \
            "Physical constituency matrix must have orthogonal rows."

        super().__init__(physical_constituency_matrix, mpc_seed)

    def build_stationary_policy_matrix(self, z_star: types.ActionSpace, eps: float = 1e-4) \
            -> np.ndarray:
        """
        Returns a matrix with rows representing randomised policies, one per resource.

        :param z_star: Policy to be performed during the given number of time steps.
        :param eps: Tolerance for checking the entries of the policy lie in the unit interval.
        :return policy_matrix: Matrix with rows being the stationary policies, one per resource,
            and columns the different activities, including an extra column for idling.
        """
        # Although policy is not probability distribution, it must lie in [0, 1].
        assert np.all(z_star >= -eps) and np.all(z_star <= 1 + eps)
        np.clip(z_star, 0, 1, out=z_star)

        # Weight each column of the constituency matrix (activity) with its corresponding activity
        # rate.
        weighted_constituency_matrix = \
            self.constituency_matrix @ np.diag(z_star.reshape((self.num_activities,)))
        resource_activity = np.sum(weighted_constituency_matrix, axis=1)
        assert np.all(resource_activity >= -eps) and np.all(resource_activity <= 1 + eps)

        # Policy is the solution of the fluid model, which satisfies C z <= 1. If Cz < 1, then the
        # solution can be understood as that the resource should idle some time i.e. 1 - Cz.
        # Compute the probability of idling
        z_star_idle = 1 - resource_activity[:, None]
        np.clip(z_star_idle, 0, 1, out=z_star_idle)

        # Build matrix with probability distribution over activities, including idling as rightmost
        # column, for each resource, and make it non-negative.
        # Normalised, to prevent numerical instabilities with probabilities norm to 1 +/- eps
        policy_matrix = np.hstack((weighted_constituency_matrix, z_star_idle))
        normed_policy_matrix = policy_matrix / np.sum(policy_matrix, axis=1)[:, None]
        assert np.all(np.abs(normed_policy_matrix - policy_matrix) < eps)

        return normed_policy_matrix

    @staticmethod
    def draw_samples_from_stationary_policy_matrix(policy_matrix, num_mpc_steps: int,
                                                   np_random=np.random):
        """
        Draw action samples from the probability distributions of each resource.

        :param policy_matrix: Matrix with rows being the stationary policies, one per resource,
            and columns the different activities, including an extra column for idling.
        :param num_mpc_steps: Number of actions to return, as columns of returned actions matrix.
        :param np_random: Random generator of the class passed by obtain_actions. If nothing is
            passed, it takes numpy random generator by default.
        :return: actions: Matrix whose columns are action vectors, as many as num_mpc_steps.
        """

        num_activities = policy_matrix.shape[1] - 1  # subtract idling activity

        actions = np.zeros((num_activities, num_mpc_steps))  # Initialise output.
        for stationary_policy_resource_s in policy_matrix:
            # Draw actions for that resource
            action_resource_s = np_random.choice(range(num_activities + 1), num_mpc_steps,
                                                 p=stationary_policy_resource_s)
            for t, a in enumerate(action_resource_s):
                if a < num_activities:  # If it is not the idle action
                    actions[a, t] = 1
        return actions

    def obtain_actions(self, **kwargs) -> types.ActionProcess:
        """
        Return actions dealing with each resource independently.

        :param kwargs: Dictionary that can contain:
            - 'z_star': Stationary policy vector, such that constituency_matrix @ z_star <= 1.
            - 'num_mpc_steps': Number of samples to be drawn, each corresponding to a single action.
                When this parameter is specified and it is greater than one, then the method will
                return a schedule of actions of length 'num_mpc_steps'. If this parameter is not
                passed, then it will be set to one.
        :return: actions: binary indicator matrix with number of rows equal to the number of
            activities (i.e. num of columns of the constituency matrix), and number of columns equal
            to 'num_mpc_steps'.
        """
        num_mpc_steps = kwargs.get('num_mpc_steps', 1)
        mpc_utils.check_num_time_steps(num_mpc_steps)

        assert 'z_star' in kwargs, "Activity rates 'z_star' are required to be passed as parameter."
        stationary_policy_matrix = self.build_stationary_policy_matrix(kwargs['z_star'])
        actions = self.draw_samples_from_stationary_policy_matrix(stationary_policy_matrix,
                                                                  num_mpc_steps,
                                                                  self.np_random)
        return actions
