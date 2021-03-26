import numpy as np
from typing import Optional, Any

from snc.utils import snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.agents import agents_utils
from snc.agents.agent_interface import AgentInterface


class CustomParametersPriorityAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk, state_option: bool, cost_option: bool,
                 rate_option: bool, name: str, agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy for push models where each activity can be done only by one resource.
        Buffers on which to work and activities to performed are decided based on a custom
        combination of buffers state, buffer cost, and activity rate. Flags are used to specify
        which combination of such parameters is used to determine priority. In order to be able to
        work on a buffer, its state has to be higher than zero. If there are multiple options with
        the same value, each resource chooses among them randomly.

        :param env: the environment to stepped through.
        :param state_option: whether the current state is considered when computing the priority
        values.
        :param cost_option: whether the cost is considered when computing the priority values.
        :param rate_option: whether the activities rate is considered when computing the priority
        values.
        :param name: Agent identifier.
        :return: None.
        """
        # verify that at least an option has been selected
        assert any([state_option, cost_option, rate_option])
        # verify that each activity can be performed by only one resource
        assert agents_utils.has_orthogonal_rows(env.constituency_matrix), \
            "Constituency matrix must have orthogonal rows."
        # verify that the environment is a push model
        for resource_constraints in env.list_boundary_constraint_matrices:
            assert np.all(np.sum(resource_constraints, axis=1) >= 1)
        super().__init__(env, name, agent_seed)
        self._state_option = state_option
        self._cost_option = cost_option
        self._rate_option = rate_option
        self.env = env

    def compute_priority_values(self, state: types.StateSpace) -> np.ndarray:
        # we care only about activities that can work on buffers, i.e., negative value in the
        # buffer_processing_matrix
        bpm = self.env.job_generator.buffer_processing_matrix
        priority_values = np.where(bpm < 0, -1, 0)
        if self._rate_option:
            # positive elements in bpm are set to 1 to avoid division by zero.
            priority_values = \
                np.divide(priority_values, np.where(bpm < 0, -bpm, 1))
        if self._state_option:
            priority_values = np.multiply(priority_values, state)
        if self._cost_option:
            priority_values = np.multiply(priority_values, self.env.cost_per_buffer)
        return priority_values

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Returns action such that buffers on which to work and activities to performed are
        decided based on the custom combination of parameters. If there are multiple options
        with the same value, each resource chooses among them randomly.

        :param state: Current state of the system.
        :param override_args: extra policy-specific arguments not needed for this heuristic.
        :return action: Action vector.
        """
        _, num_activities = self.env.job_generator.buffer_processing_matrix.shape
        action = np.zeros((num_activities, 1))
        priority_values = self.compute_priority_values(state)

        # For each resource
        for resource_constituency in self.constituency_matrix:
            activities_of_resource, = np.where(resource_constituency == 1)
            search_action = True
            while search_action:
                min_value = np.amin(priority_values[:, activities_of_resource])
                # check if there is no possible action
                if min_value >= 0:
                    search_action = False
                else:
                    # create a submatrix of priority_values with only the columns corresponding to
                    # the activities of the resource considered
                    restricted_priority_values = priority_values[:, activities_of_resource]
                    _, active_activity = \
                        np.where(restricted_priority_values == min_value)
                    # randomly select the active action if multiple have the same lowest value
                    i = self.np_random.choice(active_activity)
                    buffers_with_active_action = np.where(restricted_priority_values[:, i] < 0)
                    if all(state[buffers_with_active_action] > 0):
                        action[activities_of_resource[i]] = 1
                        search_action = False
                    else:
                        # at least one buffer on which the activity works is empty and thus
                        # such activity is prohibited
                        priority_values[:, activities_of_resource[i]] = 0

        return action


class PriorityState(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityState",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities that work on longer buffers.

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=True, cost_option=False, rate_option=False,
                         name=name, agent_seed=agent_seed)


class PriorityCost(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityCost",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities that work on buffers with higher cost.

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=False, cost_option=True, rate_option=False,
                         name=name, agent_seed=agent_seed)


class PriorityRate(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityRate",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities with lower rate.

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=False, cost_option=False, rate_option=True,
                         name=name, agent_seed=agent_seed)


class PriorityStateCost(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityStateCost",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities that work on buffers with higher
        (length * cost).

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=True, cost_option=True, rate_option=False,
                         name=name, agent_seed=agent_seed)


class PriorityStateRate(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityStateRate",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities with higher (buffers states / activity rate).

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=True, cost_option=False, rate_option=True,
                         name=name, agent_seed=agent_seed)


class PriorityCostRate(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityCostRate",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities with higher (buffer cost / activity rate).

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=False, cost_option=True, rate_option=True,
                         name=name, agent_seed=agent_seed)


class PriorityStateCostRate(CustomParametersPriorityAgent):

    def __init__(self, env: crw.ControlledRandomWalk, name: str = "PriorityStateCostRate",
                 agent_seed: Optional[int] = None) -> None:
        """
        Non-idling policy that prioritises activities with higher
        (buffer length * buffer cost / activity rate).

        :param env: the environment to stepped through.
        :param name: Agent identifier.
        :param agent_seed: Agent random seed.
        :return: None.
        """
        super().__init__(env=env, state_option=True, cost_option=True, rate_option=True,
                         name=name, agent_seed=agent_seed)
