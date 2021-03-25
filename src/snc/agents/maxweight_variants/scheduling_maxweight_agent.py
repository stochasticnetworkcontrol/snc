from typing import Tuple, List, Any, Optional
import numpy as np
from src.snc.utils import snc_types as types
from src.snc.environments import controlled_random_walk as crw
from src.snc import AgentInterface
from src.snc \
    import StationaryActionMPCPolicy


def get_gain_buffer_drained_by_activity_j(state: types.StateSpace, j: int,
                                          buffer_processing_matrix: types.BufferMatrix,
                                          cost_per_buffer: np.ndarray) -> float:
    """
    Get gain (pressure) of draining some buffer j.

    :param state: Current state of the environment.
    :param j: Index of activity under evaluation.
    :param buffer_processing_matrix: From job generator (the environment).
    :param cost_per_buffer: Cost per unit of inventory per buffer (from the environment).
    :return: Gain of draining buffer j.
    """
    ind_drained_buffer = np.argwhere(buffer_processing_matrix[:, j] < 0)
    assert ind_drained_buffer.size == 1  # Scheduling: only one drained buffer per activity.
    ind_drained_buffer = ind_drained_buffer[0, 0]
    mu_j = np.abs(buffer_processing_matrix[ind_drained_buffer, j])
    ind_buffer_plus = np.argwhere(buffer_processing_matrix[:, j] > 0)
    assert ind_buffer_plus.size <= 1  # Standard scheduling, no routing. Could be zero if exit node.
    if ind_buffer_plus.size == 0:
        cost_state_next_j = 0
        mu_j_plus = 0
    else:
        ind_buffer_plus = ind_buffer_plus[0, 0]
        cost_state_next_j = state[ind_buffer_plus, 0] * cost_per_buffer[ind_buffer_plus, 0]
        mu_j_plus = np.abs(buffer_processing_matrix[ind_buffer_plus, j])
    theta_j = mu_j * cost_per_buffer[ind_drained_buffer, 0] * state[ind_drained_buffer, 0] \
              - mu_j_plus * cost_state_next_j
    return theta_j


def get_max_gain_station_s(ind_actions_s: types.Array1D, state: types.StateSpace,
                           buffer_processing_matrix: types.BufferMatrix,
                           cost_per_buffer: np.ndarray) -> Tuple[float, List[int]]:
    """
    Get maximum gain for some station (resource) s.

    :param ind_actions_s: Station index.
    :param state: Current state of the environment.
    :param buffer_processing_matrix: From job generator (the environment).
    :param cost_per_buffer: Cost per unit of inventory per buffer (from the environment).
    :return: (max_theta, list_max_action)
        - max_theta: Maximum gain.
        - list_max_action: List of actions that provide maximum gain.
    """
    max_theta = - np.inf
    list_max_action = []
    for j in ind_actions_s:
        theta_j = get_gain_buffer_drained_by_activity_j(state, j[0],
                                                        buffer_processing_matrix, cost_per_buffer)
        if theta_j == max_theta:
            list_max_action.append(j[0])  # Append  maximising action
        elif theta_j > max_theta:
            list_max_action = [int(j[0])]  # Delete previous maximising action, and add current one.
            max_theta = theta_j
    return max_theta, list_max_action


class SchedulingMaxWeightAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk,
                 name: str = "SchedulingMaxWeightAgent",
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        Alternative description of the MaxWeight policy based on Prop. 4.8.1 (CTCN book online
        edition). This description only works for scheduling problems (i.e., with invertible B
        matrix) and only one draining activity per buffer.

        :param env: CRW environment.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :param mpc_seed: MPC random seed.
        """
        super().__init__(env, name, agent_seed)
        self.mpc_policy = StationaryActionMPCPolicy(env.physical_constituency_matrix, mpc_seed)

    def scheduling_max_weight_policy(self, state: types.StateSpace, eps: float = 1e-6) \
            -> types.ActionSpace:
        """
        Compute MaxWeight policy for scheduling problems based on the back-pressure formulation.

        :param state: Current state of the environment.
        :param eps: Tolerance to state if max gain is negative, and if state is nonempty.
        :return: MaxWeight policy.
        """
        num_activities = self.env.constituency_matrix.shape[1]
        z_star = np.zeros((num_activities, 1))
        for s in self.env.constituency_matrix:
            ind_actions_s = np.argwhere(s > 0)
            max_theta_s, list_max_action = get_max_gain_station_s(
                ind_actions_s, state, self.env.job_generator.buffer_processing_matrix,
                self.env.cost_per_buffer)
            if max_theta_s < -eps:
                z_star[ind_actions_s, :] = 0
            else:
                num_positive_actions = 0
                ind_positive_actions = []
                for j in list_max_action:
                    ind_drained_buffer = np.argwhere(
                        self.env.job_generator.buffer_processing_matrix[:, j] < 0)
                    if state[ind_drained_buffer] >= 1 - eps:
                        ind_positive_actions.append(j)
                        num_positive_actions += 1
                if num_positive_actions > 0:
                    z_star[ind_positive_actions] = 1 / num_positive_actions
        return z_star

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Converts the continuous action vector to a set of binary actions.

        :param state: Current state of the environment.
        :param override_args: Arguments that can be overriden.
        :return: Set of binary actions.
        """
        z_star = self.scheduling_max_weight_policy(state)
        actions = self.mpc_policy.obtain_actions(z_star=z_star, num_mpc_steps=1)
        return actions
