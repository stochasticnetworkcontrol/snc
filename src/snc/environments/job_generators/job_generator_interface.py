from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

from numpy.random.mtrand import RandomState

import numpy as np

from src.snc import BufferRoutesType, SupplyNodeType, ExitNodeType
from src import snc as snc_types


class JobGeneratorInterface(ABC):
    """Interface to generate job arrivals, and serviced and routed jobs."""

    def __init__(self,
                 demand_rate: snc_types.StateSpace,
                 buffer_processing_matrix: snc_types.BufferMatrix,
                 job_gen_seed: Optional[int] = None) -> None:
        """
        Generates job arrivals, and processed and routed jobs according to demand and service rates
        and to the routing matrix.
        :param demand_rate: Array with demand rate at every buffer, alpha.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :param job_gen_seed: The number to initialise the numpy RandomState
        """

        assert np.all(demand_rate >= 0), "Demand must be nonnegative."
        self.num_buffers = demand_rate.shape[0]
        assert self.num_buffers > 0
        assert demand_rate.shape[1] == 1
        self._demand_rate = demand_rate

        bpm_num_buffers, self.num_activities = buffer_processing_matrix.shape
        assert bpm_num_buffers == self.num_buffers, "Number of buffers in demand_rate and " \
                                                    "buffer_processing matrix are different."
        self._buffer_processing_matrix = buffer_processing_matrix

        self.seed = job_gen_seed
        self.np_random = RandomState(self.seed)

        self.routes, self.supply_nodes, self.demand_nodes, self.exit_nodes = \
            self.get_routes_supply_demand_exit_nodes()

    def to_serializable(self) -> Dict:
        """Return a serializable object, that can be used by a JSON Encoder"""
        sr = {str(k): v for k,v in self.__dict__.items() if k != 'np_random'}
        for k in ['routes', 'supply_nodes', 'demand_nodes', 'exit_nodes']:
            sr[k] = str(sr[k])
        return sr

    @property
    def demand_rate(self) -> snc_types.StateSpace:
        """Demand rate at each buffer."""
        return self._demand_rate

    @demand_rate.setter
    def demand_rate(self, demand_rate: np.ndarray) -> None:
        assert demand_rate.shape == (self.num_buffers, 1)
        self._demand_rate = demand_rate

    @property
    def buffer_processing_matrix(self) -> snc_types.BufferMatrix:
        """Matrix that defines the influence of each activity (column) in each buffer (row)."""
        return self._buffer_processing_matrix

    @property
    def draining_jobs_rate_matrix(self) -> snc_types.BufferMatrix:
        """Matrix similar to buffer_processing_matrix but only for routing and demand nodes, i.e. we
        have removed the positive rates that correspond to input or supply nodes."""
        draining_jobs_rate_matrix = np.minimum(0, self._buffer_processing_matrix)
        assert np.all(draining_jobs_rate_matrix <= 0)
        return draining_jobs_rate_matrix

    @abstractmethod
    def get_arrival_jobs(self) -> snc_types.StateSpace:
        """Returns number of new job arrivals at each buffer."""
        pass

    @abstractmethod
    def get_instantaneous_drained_jobs_matrix(self) -> snc_types.BufferMatrix:
        """
        Returns the number of processed jobs with given rate (take absolute value to make rate
        positive). The entries are negative to indicate that they leave the buffer.
        """
        pass

    @abstractmethod
    def get_supplied_jobs(self, rate: float) -> np.int64:
        """
        Returns the number of supplied jobs with given rate. The entries are positive to indicate
        that they add to the buffer.
        """
        pass

    def get_routing_job_matrix(self) -> snc_types.BufferMatrix:
        """Returns the possibly stochastic B(t) matrix."""
        # Set the actual number of serviced jobs. They must have negative sign back since they are
        # leaving jobs (output).
        routing_jobs_matrix = self.get_instantaneous_drained_jobs_matrix()
        # For routing, set number of jobs leaving a buffer equal to the number jobs that input the
        # next buffer in the route
        for destination, origin in self.routes.items():  # Change sign to indicate input.
            routing_jobs_matrix[destination] = - routing_jobs_matrix[origin]
        # For demand nodes, same amount is retired from both surplus and deficit buffers.
        for surplus, deficit in self.demand_nodes.items():
            routing_jobs_matrix[deficit] = routing_jobs_matrix[surplus]

        # For supply nodes, set the number of supplied jobs input to the buffer
        for supply in self.supply_nodes:
            supply_rate = self.buffer_processing_matrix[supply]
            routing_jobs_matrix[supply] = self.get_supplied_jobs(supply_rate)

        return routing_jobs_matrix

    def get_routes_supply_demand_exit_nodes(self) -> Tuple[BufferRoutesType, SupplyNodeType,
                                                           BufferRoutesType, ExitNodeType]:
        """
        Returns which elements in the buffer processing matrix that correspond to routes,
        supply_nodes, demand_nodes and exit_nodes.

        :return (routes, supply_nodes, demand_nodes, exit_nodes):
            - buffer_routes: Dict[Tuple, Tuple], where the key indicates positions where we are
              routing, and the value denotes where we are routing from, e.g. {(1, 1): (0, 1)}
              indicates that we are routing from (0, 1) to (1, 1). Moreover, these tuples represent
              position in the buffer processing matrix, such that (0, 1) indicates buffer 0 when
              using activity 1.
            - supply_nodes: List[Tuple], where Tuple gives the location of the node in the buffer
              routing matrix.
            - demand_nodes:  Dict[Tuple, Tuple], where the keys indicates positions (row and column)
              of surplus and the values the positions of the associated deficit nodes.
            - exit_nodes: List[Tuple, Tuple] that gathers the nodes where jobs leave the system
              (i.e., they are not routed further.)
        """
        routes = {}
        supply_nodes = []
        demand_nodes = {}
        exit_nodes = []
        for ci in range(self.buffer_processing_matrix.shape[1]):  # per column
            c = self.buffer_processing_matrix[:, ci]
            input_node = np.flatnonzero(c > 0)
            output_node = np.flatnonzero(c < 0)
            assert output_node.size > 0 or input_node.size > 0, 'Must consume or produce work'
            if input_node.size > 0:
                assert output_node.size <= 1, 'Model assumption: Pull from at most one node or none'
                if output_node.size == 1:  # There is one output and one input, so this is a route.
                    # Iterate if we push to multiple buffers
                    for i in range(input_node.size):
                        routes[(input_node[i], ci)] = (output_node[0], ci)
                else:  # There is at least one input but no output, so this is a supply node.
                    supply_nodes.append((input_node[0], ci))
            elif output_node.size == 1:  # No input and one output, this is an exit node.
                exit_nodes.append((output_node[0], ci))
            elif output_node.size == 2:  # No input and two outputs, these are virtual demand nodes.
                # Distinguish surplus and deficit buffer.
                s, d = output_node
                if np.argwhere(self.buffer_processing_matrix[s] > 0).size > 0:
                    surplus = s
                    deficit = d
                else:
                    surplus = d
                    deficit = s
                assert np.argwhere(self.buffer_processing_matrix[deficit] > 0).size == 0, \
                    "Deficit buffer cannot receive items from other buffers."
                demand_nodes[(surplus, ci)] = (deficit, ci)
        self._check_buffer_routes(routes)
        return routes, supply_nodes, demand_nodes, exit_nodes

    def _check_buffer_routes(self, buffer_routes: Dict) -> None:
        for destination, origin in buffer_routes.items():
            assert destination[1] == origin[1]  # The activity (column) must be the same.
            assert destination[0] < self.num_buffers
            assert destination[1] < self.num_activities
            assert origin[0] < self.num_buffers
            assert origin[1] < self.num_activities

    def reset_seed(self, job_gen_seed: Optional[int] = None):
        if job_gen_seed is not None:
            self.seed = job_gen_seed
        self.np_random.seed(self.seed)
