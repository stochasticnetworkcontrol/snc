from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from typing import Dict, Any, Optional
import snc.utils.snc_types as types
from snc.environments import controlled_random_walk as crw
from snc.simulation.store_data.numpy_encoder import clean_to_serializable


class AgentInterface(ABC):

    def __init__(self, env: crw.ControlledRandomWalk, name: str, agent_seed: Optional[int] = None)\
            -> None:
        """
        Interface for all policies.

        :param env: The environment in which to simulate the policy
        :param name: The name of the agent.
        :param agent_seed: Random seed used to set up individual random state.
        """
        self.env = env

        # Note: all these objects are available from the `env` object
        # but caching them here, means they are stored as top level parameters during saving.
        self.buffer_processing_matrix = env.job_generator.buffer_processing_matrix
        self.constituency_matrix = env.constituency_matrix
        self.demand_rate = env.job_generator.demand_rate
        self.list_boundary_constraint_matrices = env.list_boundary_constraint_matrices
        self.name = name
        # Note that this random generator is different from the job random generator.
        self.np_random = np.random.RandomState(agent_seed)

    @abstractmethod
    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Returns actions (possibly many) given current state. Can take a kwarg dictionary
        of overriding arguments that may be policy specific.

        :param state: Current state of the system
        :param override_args: extra policy-specific arguments to override defaults (if needed)
        """
        pass

    def perform_offline_calculations(self) -> None:
        """
        Perform any offline calculations before running the simulation.
        """
        pass

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        return clean_to_serializable(self)
