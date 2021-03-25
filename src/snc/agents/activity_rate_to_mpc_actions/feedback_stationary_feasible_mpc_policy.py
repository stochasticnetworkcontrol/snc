import numpy as np
from typing import List, Optional, Tuple

from src.snc import ActionMPCPolicy
from src import snc as agents_utils, snc as policy_utils, snc as types


class FeedbackStationaryFeasibleMpcPolicy(ActionMPCPolicy):
    # TODO: Allow to choose actions deterministically, e.g., for a resource take the action with
    #  largest rate.
    def __init__(self,
                 physical_constituency_matrix: types.ConstituencyMatrix,
                 buffer_processing_matrix: types.BufferMatrix,
                 mpc_seed: Optional[int] = None) -> None:
        """
        Obtain feasible binary actions from activity rates with feedback on how many actions have
        been performed so far for a given horizon. The actions are drawn from a probability
        distribution that aims to match the activity rates after some horizon. The feedback allows
        to adjust the distribution so that underperformed activities are emphasised. Feasible
        actions refer to those that drain nonempty buffers. In the case that some action is
        infeasible, then the corresponding resource performs other activity.

        :param physical_constituency_matrix: Constituency matrix from environment. We assume it has
            orthogonal rows.
        :param buffer_processing_matrix: Buffer processing matrix from environment.
        :return: None.
        """
        # Ensure that constituency_matrix has a single one per column.
        assert agents_utils.has_orthogonal_rows(physical_constituency_matrix), \
            "Physical constituency matrix must have orthogonal rows."
        super().__init__(physical_constituency_matrix, mpc_seed)

        self.buffer_processing_matrix = buffer_processing_matrix

        self.activities_per_resource, self.num_activities_per_resource \
            = self.get_ind_activities_per_resource(physical_constituency_matrix)

    @staticmethod
    def get_ind_activities_per_resource(physical_constituency_matrix: types.ConstituencyMatrix) -> \
            Tuple[List[List[int]], List[int]]:
        """
        Return the index of activities per resource and the total number of activities per resource.

        :param physical_constituency_matrix: Constituency matrix from environment. We assume it has
            orthogonal rows.
        :return: (activities_per_resource, num_activities_per_resource):
            - activities_per_resource: List of lists of activities per resource.
            - num_activities_per_resource: List of number of activities per resource.
        """
        assert agents_utils.has_orthogonal_rows(physical_constituency_matrix), \
            "Physical constituency matrix must have orthogonal rows."

        activities_per_resource = []  # type: List[List[int]]
        num_activities_per_resource = []  # type: List[int]

        for c in physical_constituency_matrix:
            activities_c = np.nonzero(c)[0]
            assert activities_c.size > 0

            activities_per_resource += [activities_c.tolist()]
            num_activities_per_resource += [activities_c.size]

        return activities_per_resource, num_activities_per_resource

    @staticmethod
    def does_activity_j_drain_any_currently_empty_buffer(
            state: types.StateSpace, j: int, buffer_processing_matrix: types.BufferMatrix) -> bool:
        """
        Check if activity j drains an empty buffer.

        :param state: Current state.
        :param j: Index activity.
        :param buffer_processing_matrix: Buffer processing matrix.
        :return: True if activity j drains an empty buffer, or False otherwise.
        """
        empty_buffers = np.where(state < 1)[0].tolist()
        if not empty_buffers:
            return False
        else:
            it_drains_empty_buffer = False
            for b in empty_buffers:
                if j in policy_utils.get_index_activities_that_can_drain_buffer(
                        b, buffer_processing_matrix).tolist():
                    it_drains_empty_buffer = True
        return it_drains_empty_buffer

    @staticmethod
    def get_activities_drain_currently_empty_buffers(
            state: types.StateSpace, buffer_processing_matrix: types.BufferMatrix) -> List[int]:
        """
        Return list of activities that will attempt to drain empty buffers.

        :param state: Current state.
        :param buffer_processing_matrix: Buffer processing matrix.
        :return: List of indexes of activities draining empty buffers.
        """
        act_drain_empty_buffers = []  # type: List[int]
        num_activities = buffer_processing_matrix.shape[1]
        for j in range(num_activities):
            if FeedbackStationaryFeasibleMpcPolicy.does_activity_j_drain_any_currently_empty_buffer(
                    state, j, buffer_processing_matrix):
                act_drain_empty_buffers += [j]
        return act_drain_empty_buffers

    @staticmethod
    def get_valid_actions(act_s: List[int], act_drain_empty: List[int]):
        """
        For a given list of activities of some resource, return  only those activities that don't
        drain empty buffers.
        This is actually a generic function that takes two lists as input, and returns the elements
        of the first argument that are not in the second argument.

        :param act_s: List of activities of the s-th resource.
        :param act_drain_empty: List of activities that don't drain empty buffers.

        """
        return [x for x in act_s if x not in act_drain_empty]  # List of valid actions.

    @staticmethod
    def reallocate_activity_j(j: int, activity_rates: types.ActionSpace,
                              act_s: List[int], act_drain_empty: List[int]):
        """
        Adjust the activity rates by reallocating the activity rate of action j, which drains an
        empty buffer, to other actions that belong to the same resource but don't drain any empty
        buffer.

        :param j: Index of the activity whose rate we want to reallocate.
        :param activity_rates: Vector of activity rates.
        :param act_s: List of actions of the s-th resource.
        :param act_drain_empty: List of not valid actions, which drain empty buffers.
        :return: new_activity_rates: Vector of activity rates where the mass of j has been evenly
            added to other activities of the same resource.
        """
        assert j in act_drain_empty, f"Only reallocate activity j={j} if it drains an empty buffer."
        assert j in act_s, f"Something is wrong: action j={j} should belong to act_s=[{act_s}]."

        new_activity_rates = activity_rates.copy()
        valid_act_s = FeedbackStationaryFeasibleMpcPolicy.get_valid_actions(act_s, act_drain_empty)
        if valid_act_s:
            new_activity_rates[valid_act_s] += activity_rates[j] / len(valid_act_s)
        new_activity_rates[j] = 0
        return new_activity_rates

    @staticmethod
    def clip_to_simplex_and_normalise_feedback_rates(sum_actions: types.ActionSpace,
                                                     activities_per_resource: List[List[int]],
                                                     sum_actions_per_resource: np.ndarray,
                                                     num_steps_to_recompute_policy: int) \
            -> np.ndarray:
        """
        Returns rates for remaining activities, with entries given by the number of times each
        activity has to be performed normalised by the sum of rates for the resource to which the
        activity belongs.

        :param sum_actions: Vector with number of times that each action has to be executed in
            the remaining number of steps given by `num_steps_to_recompute_policy`.
        :param activities_per_resource: List of lists of activities per resource.
        :param sum_actions_per_resource: Sum of number of times all the activities that belong to
            each resource have to be performed.
        :param num_steps_to_recompute_policy: Remaining number of steps before recomputing the
            policy.
        :return: Weighting diagonal matrix.
        """
        rate_remain_act = np.zeros((sum_actions.shape[0], 1))
        for i, s in enumerate(activities_per_resource):
            if sum_actions_per_resource[i] > 0:
                for j in s:
                    rate_remain_act[j] = sum_actions[j] / num_steps_to_recompute_policy
                    rate_remain_act[j] = np.clip(rate_remain_act[j], 0, 1)
        assert np.all(0 <= rate_remain_act) and np.all(rate_remain_act <= 1)
        return rate_remain_act

    def choose_action_from_rate_distribution(self, num_activities, rates):
        return self.np_random.choice(num_activities, 1, p=rates)[0]

    def generate_actions_with_feedback(self, sum_actions: types.ActionSpace,
                                       state: types.StateSpace,
                                       num_steps_to_recompute_policy: int,
                                       tol: float = 1e-7):
        """
        Return a single action vector with the following features:
        - It never works on an empty buffer.
        - It is drawn from a distribution obtained as follows:
            i) First convert remaining actions to activity rates, clip the negative values (i.e.
                actions that have been performed more often than required) to zero, and normalised
                the number of remaining actions for each activity by the total remaining number of
                remaining actions of the resource it belongs to.
                    d[i] = 0 if sum_actions[i] <= 0, and
                    d[i] = sum_actions[i] / [C @ sum_actions]_{s}, for all i such that C_{s,i} = 1.
            ii) For any action i that serves an empty buffer, we reallocate d[i] to the rest of
                activities of the same resource that drain nonempty buffers. Let us say that
                valid_activities is the set of activities that don't drain an empty buffer. Then:
                    d[i] = 0, if i not in valid_activities
                    d[j] = d[j] + sum_{i not in valid_activities} d[i] / |valid_activities|
            iii) For each resource s:
                1) [C @ d]_s in {0, 1} (this is always feasible if the rows of C are orthogonal),
                2) [C @ d]_s = 0 if and only if [C @ sum_actions]_s <= 0.

        :param state: Current state, needed to check if some buffers are empty.
        :param sum_actions: Vector with number of times that each action has to be executed in
            the remaining number of steps given by `num_steps_to_recompute_policy`.
        :param num_steps_to_recompute_policy: Remaining number of steps before recomputing the
            policy.
        :param tol: Tolerance to check inequalities.
        :return: action: Binary action vector satisfying feasibility constraints.
        """
        action = np.zeros_like(sum_actions)  # Initialise action vector to be returned.

        # Compute sum_actions_per_resource.
        sum_actions_per_resource = self.constituency_matrix @ sum_actions
        # If [C @ sum_actions]_s > 0, then the s-th resource shouldn't idle.
        nonidling = sum_actions_per_resource > tol

        # Obtain rates for remaining activities: d.
        rate_remain_act = self.clip_to_simplex_and_normalise_feedback_rates(
            sum_actions, self.activities_per_resource, sum_actions_per_resource,
            num_steps_to_recompute_policy)

        act_drain_empty_buffers = self.get_activities_drain_currently_empty_buffers(
            state, self.buffer_processing_matrix)

        for s in range(self.num_resources):
            if nonidling[s]:  # if [C @ sum_actions]_s > 0
                act_s = self.activities_per_resource[s]

                for j in act_s:
                    if j in act_drain_empty_buffers:  # Reallocate rate to other activities: d.
                        rate_remain_act = self.reallocate_activity_j(
                            j, rate_remain_act, act_s, act_drain_empty_buffers)

                # Normalise per resource so that: [C @ d]_s = 1.
                rate_remain_act[act_s] /= np.sum(rate_remain_act[act_s])
                # TODO the line above produces NAN that should instead be a 0
                # Choose one valid action for resource s, such that [C @ action]_s = 1.
                valid_act = self.get_valid_actions(act_s, act_drain_empty_buffers)
                if valid_act:
                    j = self.choose_action_from_rate_distribution(
                        self.num_activities_per_resource[s],
                        np.squeeze(rate_remain_act[act_s], axis=1))
                    action[act_s[j]] = 1
                    assert 1 - tol < self.constituency_matrix[s] @ action < 1 + tol, \
                        "Constraint C @ rate_remain_act <= 1 does not hold."
        return action

    def obtain_actions(self, **kwargs) -> types.ActionProcess:
        """
        This method implements the abstract method from super class `ActionMPCPolicy`.
        It first gathers the feedback information namely the number of times each activity has to
        be performed (i.e. 'sum_actions') and the current state ('state'). Then, it calls its own
        method to return a single action vector.

        :return: actions: binary indicator matrix with number of rows equal to the number of
            non-idling activities (i.e. num of columns of the constituency matrix), and number of
            columns equal to number of time steps to perform MPC.
        """
        assert 'mpc_variables' in kwargs, "Ensure MPC variables dict is passed"
        assert 'sum_actions' in kwargs["mpc_variables"], \
            "Number of times each activity has to be performed ('sum_actions') is required to be " \
            "passed as parameter."
        assert 'state' in kwargs, "Current state ('state') is required to be passed as parameter."

        actions = self.generate_actions_with_feedback(kwargs["mpc_variables"]['sum_actions'],
                                                      kwargs['state'],
                                                      kwargs['num_steps_to_recompute_policy'])
        return actions
