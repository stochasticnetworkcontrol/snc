import numpy as np
from typing import Dict, Optional

from snc.utils import snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.agents.agent_interface import AgentInterface


class RandomNonIdlingAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk, safety_stock=10.0,
                 name: str = "RandomNonIdlingAgent", agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy such that every resource that satisfies boundary conditions (i.e.
        it does not starve) works on any of its activities. The activity is randomly selected.

        :param env: the environment to stepped through.
        :param safety_stock: minimum number of customers in a buffer before a resource can work.
        :param name: Agent identifier.
        :param seed: Random seed to be used in setting up the agent's numpy random state.
        """
        super().__init__(env, name, agent_seed)
        self.safety_stock = safety_stock

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Dict) \
            -> types.ActionProcess:
        """
        Returns action giving by a nonidling policy, such that each resources acts if the
        boundary conditions on its buffers are satisfied for the given safety_stock, and the
        action is chosen randomly among any of its possible activities.

        :param state: Current state of the system
        :return action: action vector
        """
        _, num_activities = self.constituency_matrix.shape
        action = np.zeros((num_activities, 1))
        for constituency_s, boundary_constraint_s in zip(self.constituency_matrix,
                                                         self.list_boundary_constraint_matrices):
            if np.all(np.dot(boundary_constraint_s, state) > self.safety_stock):
                possible_actions = np.nonzero(constituency_s)[0]
                ind = self.np_random.randint(0, possible_actions.size)
                action[possible_actions[ind]] = 1
        return action
