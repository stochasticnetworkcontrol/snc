from typing import Tuple, List, Any, Optional, Union
import numpy as np
from snc.utils import snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.agents.agent_interface import AgentInterface
from snc.agents.activity_rate_to_mpc_actions.stationary_mpc_policy \
    import StationaryActionMPCPolicy


def get_gain_activity_j(state: types.StateSpace, j: int,
                        buffer_processing_matrix: types.BufferMatrix,
                        weight_per_buffer: np.ndarray) -> float:
    """
    Get gain (pressure) for activity j.

    :param state: Current state of the environment.
    :param j: Index of activity under evaluation.
    :param buffer_processing_matrix: From job generator (the environment).
    :param weight_per_buffer: Cost per unit of inventory per buffer (from the environment).
    :return: Gain of draining buffer j.
    """
    current_buffers_cost = np.multiply(weight_per_buffer, state)
    theta_j = - np.sum(np.multiply(current_buffers_cost, buffer_processing_matrix[:, j][:, None]))
    return theta_j


def get_max_gain_station_s(ind_activities_s: types.Array1D, state: types.StateSpace,
                           buffer_processing_matrix: types.BufferMatrix,
                           weight_per_buffer: np.ndarray) -> Tuple[float, List[int]]:
    """
    Get maximum gain for some station (resource) s.

    :param ind_activities_s: Station index.
    :param state: Current state of the environment.
    :param buffer_processing_matrix: From job generator (the environment).
    :param weight_per_buffer: Weight per unit of inventory per buffer.
    :return: (max_theta, list_max_action)
        - max_theta: Maximum gain.
        - list_max_action: List of actions that provide maximum gain.
    """
    max_theta = - np.inf
    list_max_action = []
    for j in ind_activities_s:
        theta_j = get_gain_activity_j(state, j[0], buffer_processing_matrix, weight_per_buffer)
        if theta_j == max_theta:
            list_max_action.append(j[0])  # Append  maximising action
        elif theta_j > max_theta:
            list_max_action = [int(j[0])]  # Delete previous maximising action, and add current one.
            max_theta = theta_j
    return max_theta, list_max_action


class MaxWeightAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk,
                 weight_per_buffer: Optional[Union[str, types.StateSpace]] = None,
                 name: str = "MaxWeightAgent",
                 agent_seed: Optional[int] = None,
                 mpc_seed: Optional[int] = None) -> None:
        """
        MaxWeight policy based on Chapter 6.4 (CTCN book online edition).
        This only works for scheduling and routing problems,
        where each activity drains only one buffer.

        NOTE: in case of a buffer managed by multiple resources, the job_conservation_flag has to be
         True otherwise the buffer may have negative value.
         Moreover, in this case, the algorithm performance may be affected in a negative way.

        :param env: CRW environment.
        :param weight_per_buffer: Vector whose entries weight the difference pressure at each buffer
            that MaxWeight aims to balance. For the case where this is given by the cost per buffer
            given by the environment, this is passed as a string: 'cost_per_buffer'.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :param mpc_seed: MPC random seed.
        """
        # verify that each activity drains exactly one buffer
        bpm = env.job_generator.buffer_processing_matrix
        assert np.all(np.sum(np.where(bpm < 0, -1, 0), axis=0) >= -1), \
            f'Buffer processing matrix not allowed: {bpm}.' \
            'Current version only works for networks where each activity drains exactly one ' \
            'buffer (i.e., only works for scheduling and/or routing).'

        if weight_per_buffer is None:  # Set BackPressure by default.
            self.weight_per_buffer = np.ones_like((env.num_buffers, 1))
        elif isinstance(weight_per_buffer, types.StateSpace):
            assert weight_per_buffer.size == env.num_buffers, \
                f'Length of weight_per_buffer = {weight_per_buffer.size}, but it should' \
                f' equal num_buffers = {env.num_buffers}.'
            assert np.all(weight_per_buffer > 0), \
                f'weight_per_buffer: {weight_per_buffer} must have positive entries.'
            self.weight_per_buffer = weight_per_buffer
        elif isinstance(weight_per_buffer, str):
            if weight_per_buffer == 'cost_per_buffer':
                self.weight_per_buffer = env.cost_per_buffer
            else:
                raise ValueError(
                    'weight_per_buffer must be a valid string (i.e. "cost_per_buffer") '
                    f'or  numpy array. Current value is: {weight_per_buffer}.')
        else:
            assert isinstance(weight_per_buffer, types.StateSpace), \
                'weight_per_buffer must be a valid string (i.e. "cost_per_buffer") or  numpy ' \
                f'array. Current value is {weight_per_buffer}.'

        super().__init__(env, name, agent_seed)
        self.mpc_policy = StationaryActionMPCPolicy(env.physical_constituency_matrix, mpc_seed)

    def max_weight_policy(self, state: types.StateSpace, eps: float = 1e-6) \
            -> types.ActionSpace:
        """
        Compute MaxWeight policy for scheduling and routing problems based on the back-pressure
        formulation.

        :param state: Current state of the environment.
        :param eps: Tolerance to state if max gain is negative, and if state is nonempty.
        :return: MaxWeight policy.
        """
        num_activities = self.env.constituency_matrix.shape[1]
        z_star = np.zeros((num_activities, 1))
        for s in self.env.constituency_matrix:
            ind_activities_s = np.argwhere(s > 0)
            max_theta_s, list_max_activity = get_max_gain_station_s(
                ind_activities_s, state, self.env.job_generator.buffer_processing_matrix,
                self.weight_per_buffer)
            if max_theta_s < -eps:
                z_star[ind_activities_s, :] = 0
            else:
                num_positive_actions = 0
                ind_positive_actions = []
                for j in list_max_activity:
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
        z_star = self.max_weight_policy(state)
        actions = self.mpc_policy.obtain_actions(z_star=z_star, num_mpc_steps=1)
        return actions
