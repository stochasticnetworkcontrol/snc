import numpy as np
from typing import Dict, Any, Optional

from snc.utils import snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.agents.agent_interface import AgentInterface


class CustomActivityPriorityAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk, priorities: Dict,
                 name: str = "CustomActivityPriorityAgent", seed: Optional[int] = None) -> None:
        """
        Priority policy such that some activities have priority over others. For resources where
        no priorities are given, activities are chosen randomly.

        :param env: the environment to stepped through.
        :param priorities: a dictionary where the keys are the resources and the values are the
            activity with the highest priority. If the value is None, then no priority is given to
            any activity.
        :param name: Agent identifier.
        :param seed: Random seed to initialise numpy random state for the agent.
        """
        super().__init__(env, name, seed)
        self.buffer_processing_matrix = self.env.job_generator.buffer_processing_matrix
        num_resources, _ = self.env.constituency_matrix.shape

        self.priorities = {}  # type: Dict
        for resource in np.arange(num_resources):
            priority_activity = priorities.get(resource, None)
            if priority_activity is not None:
                assert self.constituency_matrix[resource, priority_activity] == 1
            self.priorities[resource] = priority_activity

    def sample_random_actions(self, state: types.StateSpace, action: types.ActionSpace,
                              activities: types.Array1D):
        """
        Returns action updated with randomly chosen activities given a list of activities to
        choose from and such that the activity chosen can be made (buffer not empty).

        :param state: Current state of the system.
        :param action: Current action.
        :param activities: List of possible activities.
        :return action: Updated action vector.
        """
        updated_action = action.copy()
        possible_actions = []
        for activity in activities:
            buffers_drained_by_activity = np.argwhere(
                self.buffer_processing_matrix[:, activity] < 0.)[:, 0]
            if np.all(state[buffers_drained_by_activity] > 0):
                possible_actions.append(activity)
        if possible_actions:
            ind = int(self.np_random.randint(0, len(possible_actions)))
            updated_action[possible_actions[ind]] = 1
        return updated_action

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Returns action such that the resources work on their highest priority activity first. If
        there are no priorities for this resource or if its priority activity can't be done
        (buffer empty), then the remaining activities are chosen randomly.

        :param state: Current state of the system.
        :return action: Action vector.
        """
        _, num_activities = self.constituency_matrix.shape
        action = np.zeros((num_activities, 1))

        # For each resource
        for resource_s, constituency_s in enumerate(self.constituency_matrix):
            priority_activity = self.priorities[resource_s]
            if priority_activity is not None:
                buffers_drained_by_priority_activity = np.argwhere(
                    self.buffer_processing_matrix[:, priority_activity] < 0.)[:, 0]
                if np.all(state[buffers_drained_by_priority_activity] > 0):
                    action[priority_activity] = 1
                else:
                    constituency_s_wo_priority_activity = constituency_s.copy()
                    constituency_s_wo_priority_activity[priority_activity] = 0
                    other_activities = np.argwhere(constituency_s_wo_priority_activity == 1)[:, 0]
                    if other_activities.size:
                        action = self.sample_random_actions(state, action, other_activities)
            else:
                all_activities = np.argwhere(constituency_s == 1)[:, 0]
                action = self.sample_random_actions(state, action, all_activities)

        return action
