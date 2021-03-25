from copy import deepcopy
from typing import Dict

import numpy as np
from abc import ABC, abstractmethod
from src import snc as snc_types


class CRWStateInitialiser(ABC):
    """
    Sets the initial state for each buffer.
    """
    @abstractmethod
    def get_initial_state(self) -> snc_types.StateSpace:
        pass

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON Encoder.
        """
        d = deepcopy(self.__dict__)
        return d


class UniformRandomCRWStateInitialiser(CRWStateInitialiser):
    """
    Sets the initial state for each buffer uniformly at random in the interval between 0 and
    capacity.
    """
    def __init__(self, num_buffers: int, capacity: snc_types.StateSpace):
        """
        :param num_buffers: Number of buffers in the system.
        :param capacity: Maximum number of jobs allowed at each buffer.
        """
        self.num_buffers = num_buffers
        self.capacity = capacity

    def get_initial_state(self) -> snc_types.StateSpace:
        return np.random.randint(0, self.capacity, size=(self.num_buffers, 1))


class DeterministicCRWStateInitialiser(CRWStateInitialiser):
    """
    Sets the initial state deterministically according to a given value.
    """

    def __init__(self, initial_state: snc_types.StateSpace):
        """
        :param initial_state: Initial number of jobs per buffer.
        """
        self.initial_state = np.array(initial_state)

    def get_initial_state(self) -> snc_types.StateSpace:
        initial_state = np.asarray(self.initial_state)
        return initial_state.reshape((len(self.initial_state), 1))
