import numpy as np
import random
from typing import List, Optional, Tuple

from snc.agents.activity_rate_to_mpc_actions.action_mpc_policy import ActionMPCPolicy
import snc.agents.agents_utils as agents_utils
import snc.agents.hedgehog.policies.policy_utils as policy_utils
import snc.utils.snc_types as types


class FoxMpcPolicy(ActionMPCPolicy):
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
        self.n_activities = buffer_processing_matrix.shape[1]
        self.activities_per_resource, self.num_activities_per_resource \
            = self.get_ind_activities_per_resource(physical_constituency_matrix)
        self.exit_activities = self.get_exit_activities(buffer_processing_matrix)
        self.activities_to_target_buffers, self.activities_to_source_buffers \
            = self.get_target_and_source_buffers(buffer_processing_matrix,self.exit_activities)


    @staticmethod
    def get_exit_activities(buffer_processing_matrix):
        exit_activities = set()
        for a in range(buffer_processing_matrix.shape[1]):
            if np.all(buffer_processing_matrix[:,a] <= 0):
                exit_activities.add(a)
        return exit_activities

    @staticmethod
    def get_target_and_source_buffers(buffer_processing_matrix,exit_activities):
        n_activities = buffer_processing_matrix.shape[1]
        activities_to_target_buffers = {}
        activities_to_source_buffers = {}
        for a in range(n_activities):
            act_vector = buffer_processing_matrix[:,a]
            activities_to_source_buffers[a] = int(np.where(act_vector < 0)[0][0])
            if a in exit_activities:
                continue
            activities_to_target_buffers[a] = int(np.where(act_vector > 0)[0][0])
        return activities_to_target_buffers, activities_to_source_buffers

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
        assert 'state' in kwargs, "Current state ('state') is required to be passed as parameter."

        state = kwargs["state"]
        x_eff = kwargs["x_eff"]
        x_star = kwargs["x_star"]
        print(state.ravel().astype(int),x_eff.ravel().astype(int),x_star.ravel().astype(int))
        idling_set = kwargs["r_idling_set"]
        draining_resources = kwargs["draining_resources"]
        print(idling_set)
        print(self.activities_per_resource)
        x_target = x_star if len(idling_set) > 0 else x_eff
        buffer_weights = list(np.maximum(x_target - state, 0).astype(int).ravel())
        print(buffer_weights)
        actions = np.zeros((self.n_activities,1))

        for r in draining_resources:
            actions_list = self._get_actions_list(r,state,buffer_weights,False)
            random.shuffle(actions_list)
            decided_action,_ = actions_list[0]
            actions[decided_action,0] = 1


        for r in range(len(self.activities_per_resource)):
            if r in draining_resources:
                continue
            actions_list = self._get_actions_list(r,state,buffer_weights, r in idling_set)
            if not actions_list:
                continue
            random.shuffle(actions_list)
            print(r,actions_list)
            decided_action,_ = actions_list[0]


            actions[decided_action,0] = 1
        print()
        return actions

    def _get_actions_list(self,r,state,buffer_weights,r_in_idling_set):

        actions_list = []
        max_weight = 0

        has_starved_activities = False
        for a in self.activities_per_resource[r]:
            if a in self.exit_activities:
                continue
            source_buffer = self.activities_to_source_buffers[a]
            target_buffer = self.activities_to_target_buffers[a]
            buffer_weight = buffer_weights[target_buffer]
            if buffer_weight > 0 and state[source_buffer,0] == 0:
                has_starved_activities = True

        for a in self.activities_per_resource[r]:
            if state[self.activities_to_source_buffers[a]] == 0:
                continue
            if a in self.exit_activities:
                source_buffer = self.activities_to_source_buffers[a]
                weight = max(1,int(state[source_buffer,0]))
                if weight == max_weight:
                    actions_list.append((a,weight))
                elif weight > max_weight:
                    actions_list = [(a,weight)]
                    max_weight = weight
                continue

            source_buffer = self.activities_to_target_buffers[a]
            target_buffer = self.activities_to_target_buffers[a]
            buffer_weight = buffer_weights[target_buffer]
            if r_in_idling_set and buffer_weight == 0:# and not has_starved_activities:
                continue
            if buffer_weight == max_weight:
                actions_list.append((a,buffer_weight))
            elif buffer_weight > max_weight:
                actions_list = [(a,buffer_weight)]
                max_weight = buffer_weight
        print(has_starved_activities)
        return actions_list
