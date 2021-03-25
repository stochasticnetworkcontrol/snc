from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from typing import Optional, Dict

import snc.utils.snc_types as types
from snc.simulation.store_data.numpy_encoder import clean_to_serializable


class ActionMPCPolicy(ABC):

    def __init__(
            self,
            constituency_matrix: types.ConstituencyMatrix,
            mpc_seed: Optional[int] = None
    ) -> None:
        """
        Interface to generate actions during some control interval given some MPC policy.

        :param constituency_matrix: Matrix with those rows from the constituency matrix
            that represent physical resources, so that it must have a single one per column. If the
            constituency_matrix of the problem has multiple ones per column (e.g. the "3x3 input
            queued switch"), then the matrix should not be used with this class.
        :param mpc_seed: MPC policy random seed.
        """
        self.np_random = np.random.RandomState(mpc_seed)

        num_resources, num_activities = constituency_matrix.shape
        assert num_resources > 0
        assert num_activities > 0
        self.num_resources = num_resources
        self.num_activities = num_activities
        self.constituency_matrix = constituency_matrix

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON Encoder.
        """
        return clean_to_serializable(self)

    @abstractmethod
    def obtain_actions(self, **kwargs) -> types.ActionProcess:
        """
        Returns matrix with columns being the actions for each time-step of the MPC horizon.

        :param kwargs: Dictionary containing the parameters needed by each MPC policy.
        :return: actions: binary indicator matrix with number of rows equal to the number of
            non-idling activities (i.e. num of columns of the constituency matrix), and number of
            columns equal to number of time steps to perform MPC.
        """
        pass
