import numpy as np
from typing import Dict, Optional

from src.snc.utils import snc_types as types
from src.snc.environments import controlled_random_walk as crw
from src.snc import AgentInterface


class LongestBufferPriorityAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk, safety_stock: Optional[float] = 10.0,
                 name: str = "LongestBufferPriorityAgent", seed: Optional[int] = None) -> None:
        """
        Non-idling policy such that every resource works on the buffer with largest amount of
        customers, and that are above the specified safety_stock. If there are multiple buffers with
        the same largest size and/or multiple activities, each resource chooses among them randomly.

        :param env: the environment to stepped through.
        :param safety_stock: minimum number of customers in a buffer before a resource can work.
        :param name: Agent identifier.
        :param seed: Random seed to be used in setting up the agent's numpy random state.
        """
        super().__init__(env, name, seed)
        self.buffer_processing_matrix = env.job_generator.buffer_processing_matrix
        self.safety_stock = safety_stock

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Dict) \
            -> types.ActionProcess:
        """
        Returns action such that the resources work on their largest buffers, when they are above
        the specified safety_stock. If there are multiple buffers with the same largest size and/or
        multiple activities, each resource chooses among them randomly.

        :param state: Current state of the system.
        :return action: Action vector.
        """
        _, num_activities = self.constituency_matrix.shape
        action = np.zeros((num_activities, 1))

        # For each resource
        for constituency_s, boundary_constraint_matrix_s in zip(
                self.constituency_matrix, self.list_boundary_constraint_matrices):
            # 1. Figure out which buffer has larger size.
            # Collapse matrix into a single binary row that indicates in which buffers this resource
            # can work on.
            boundary_constraint_s = np.sum(boundary_constraint_matrix_s, axis=0) > 0
            # Get elements of the state controlled by this resource.
            state_s = boundary_constraint_s[:, None] * state
            # Get index of buffers with highest size controlled by this resource. Result are given
            # as two coordinates with column=0, so we only need the row.
            largest_ind = np.argwhere(state_s == np.amax(state_s))[:, 0]
            # If there are multiple, choose one at random
            if largest_ind.size > 1:
                ind = self.np_random.randint(0, largest_ind.size)
                largest_ind = largest_ind[ind]
            else:
                largest_ind = largest_ind[0]  # Convert array to scalar in any case.

            # 2. If there are enough customers, find out which activity works on this buffer.
            if state[largest_ind] > self.safety_stock:
                # Get activities controlled by this resource that influence this buffer.
                activities = self.buffer_processing_matrix[largest_ind, :] * constituency_s
                possible_actions = np.nonzero(activities)[0]
                if possible_actions.size > 1:
                    ind = self.np_random.randint(0, possible_actions.size)
                    possible_actions = possible_actions[ind]
                action[possible_actions] = 1
        return action
