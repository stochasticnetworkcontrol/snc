from collections import defaultdict
import numpy as np
import random
from typing import Any, DefaultDict, Dict, List, Optional, ValuesView, Tuple

from snc.environments.controlled_random_walk import ControlledRandomWalk
from snc.environments.job_generators.job_generator_interface import JobGeneratorInterface
from snc.environments.state_initialiser import CRWStateInitialiser
import snc.utils.snc_types as snc_types


class ClosedLoopCRW(ControlledRandomWalk):

    def __init__(self,
                 demand_to_supplier_routes: Dict[int, Tuple[int, int]],
                 ind_surplus_buffers: List[int],
                 cost_per_buffer: snc_types.StateSpace,
                 capacity: snc_types.StateSpace,
                 constituency_matrix: snc_types.ConstituencyMatrix,
                 job_generator: JobGeneratorInterface,
                 state_initialiser: CRWStateInitialiser,
                 list_boundary_constraint_matrices: Optional[List[np.ndarray]] = None,
                 index_phys_resources: Optional[Tuple] = None,
                 max_episode_length: Optional[int] = None) -> None:
        """
        Environment that extends controlled random walk (CRW) pull models to modelling closed
        networks by routing satisfied demand to suppliers.
        Physically realizable CRWs ensure that resources can only process items that exist in
        their respective buffers. This is called job conservation and is a feature that can be
        controlled by setting the `job_conservation_flag` parameter to True in the original CRW
        class.
        Closed loop CRWs have the additional constraint that each supplier has a buffer that
        receives the satisfied demand and can only process and provide to the network what is
        available in such buffer.

        :param demand_to_supplier_routes: Dictionary with each key being the demand buffer and
            values being a tuple with the supply resource the buffer is connected to and the
            delay for that route, respectively.
        :param ind_surplus_buffers: List of integers indicating the location of the surplus
            buffers in the buffer processing matrix.
        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param capacity: List with maximum number of jobs allowed at each buffer.
        :param constituency_matrix:  Matrix whose s-th row corresponds to resource s; and each
            entry, C_{si} in {0, 1}, specifies whether activity i
            corresponds to resource s (examples of activities are scheduling a buffer or routing to
            another resource).
        :param job_generator: object to generate events from
        :param state_initialiser: initialiser for state
        :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
            indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
            If this parameter is not passed, it assumes no boundary constraints by default.
        :param index_phys_resources: Tuple indexing the rows of constituency matrix that correspond
            to physical resources (as opposed to other rows that might represent coupled constraints
            among physical resources). If this parameter is not passed, then it considers all rows
            of the constituency matrix as corresponding to physical resources by default.
        :param max_episode_length: Integer number of time steps to allow in each episode. Defaults
            to None meaning unlimited steps per episode (i.e. non-terminating episodes).
        """

        job_conservation_flag = True
        model_type = 'pull'
        super().__init__(
            cost_per_buffer,
            capacity,
            constituency_matrix,
            job_generator,
            state_initialiser,
            job_conservation_flag,
            list_boundary_constraint_matrices,
            model_type,
            index_phys_resources,
            ind_surplus_buffers,
            max_episode_length
        )

        # Get supply resources and buffers ids and check consistency with the constituency matrix
        # and the job generator.
        supply_ids, demand_ids = self.get_supply_and_demand_ids(demand_to_supplier_routes)
        assert self.are_demand_ids_unique(demand_ids)
        assert self.is_demand_to_supplier_routes_consistent_with_job_generator(
            supply_ids,
            demand_ids,
            constituency_matrix,
            job_generator.supply_nodes,
            job_generator.demand_nodes.values()
        )
        self.supply_ids = supply_ids
        self.demand_ids = demand_ids
        self.demand_to_supplier_routes = demand_to_supplier_routes

        # Filter phantom supply nodes detected by the job generator.
        assert self.is_supply_ids_consistent_with_job_generator(
            self.supply_ids,
            job_generator.supply_nodes,
            constituency_matrix
        )
        self.supply_nodes = job_generator.supply_nodes

        # Track initial total number of items in the network.
        self.num_initial_items = self.get_num_items_state_without_demand()

        # Build supplier and in-transit buffers.
        self.supply_buffers = self.initialise_supply_buffers(self.supply_ids)
        self.in_transit_parcels: DefaultDict[int, List[Tuple]] = defaultdict(list)

        # Associate activities to resources and to buffers.
        self.activity_to_res, self.resource_to_act \
            = self.get_activity_supply_resource_association(self.supply_nodes, constituency_matrix)
        self.activity_to_supply_buffer = self.get_supply_activity_to_buffer_association(
            self.supply_nodes
        )

    @staticmethod
    def get_supply_and_demand_ids(
            demand_to_supplier_routes: Dict[int, Tuple[int, int]]
    ) -> Tuple[List[int], List[int]]:
        """
        Get user input specifying which demand node feeds each supply buffer and return two sorted
        lists with supply resource and demand buffer ids each.

        :param demand_to_supplier_routes: Dictionary with each key being the demand buffer and
            values being a tuple with the supplier the buffer is connected to and the delay for
            that route, respectively.
        :return (supply_ids, demand_ids):
            - supply_ids: Sorted list of supply resources ids (rows of the constituency matrix).
            - demand_ids: Sorted list of demand buffers ids (rows of the buffer processing matrix).
        """
        supply_ids = sorted(list({v[0] for v in demand_to_supplier_routes.values()}))
        demand_ids = sorted(list(demand_to_supplier_routes.keys()))
        return supply_ids, demand_ids

    @staticmethod
    def are_demand_ids_unique(demand_ids: List[int]) -> bool:
        """
        Check if elements (demand ids) in the list are unique.

        :param demand_ids: Sorted list of demand buffers ids (rows of the buffer processing matrix).
        :return True if all elements are unique or false if at least one is repeated.
        """
        for d in demand_ids:
            if demand_ids.count(d) > 1:
                return False
        return True

    @staticmethod
    def is_demand_to_supplier_routes_consistent_with_job_generator(
            supply_ids: List[int],
            demand_ids: List[int],
            constituency_matrix: snc_types.ConstituencyMatrix,
            jg_supply_nodes: snc_types.SupplyNodeType,
            jg_demand_nodes: ValuesView[Tuple[int, int]]
    ) -> bool:
        """
        Check that provided user input specifying which demand node feeds each supply buffer is
        consistent with the supply and demand nodes detected by the job generator. Supply ids
        provided by the user have to be a subset, while the demand buffer ids provided by the user
        have to be equal to those detected by the job generator.

        :param supply_ids: Sorted list of supply resources ids (rows of the constituency matrix).
        :param demand_ids: Sorted list of demand buffers ids (rows of the buffer processing matrix).
        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i corresponds to resource s (examples of
            activities are scheduling a buffer or routing to another resource).
        :param jg_supply_nodes: Dictionary of supply nodes detected by the job generator.
        :param jg_demand_nodes List of demand nodes detected by the job generator.
        :return Bool indicating consistency.
        """
        # Get supply resources ids from constituency matrix and job generator.
        jg_supply_set_ids = set()
        for _, action in jg_supply_nodes:
            resource = np.argwhere(constituency_matrix[:, action])[:, 0]
            assert resource.size == 1, "A given action can belong to at most one supplier."
            jg_supply_set_ids.add(resource[0])

        # Get demand buffers ids from job generator.
        jg_demand_ids = sorted([d[0] for d in jg_demand_nodes])

        # Check they are consistent with those obtained form demand_to_supplier_routes parameter.
        # Supply ids have to be a subset, while demand buffer ids have to be equal.
        return set(supply_ids).issubset(jg_supply_set_ids) and demand_ids == jg_demand_ids

    @staticmethod
    def is_supply_ids_consistent_with_job_generator(
            supply_ids: List[int],
            jg_supply_nodes: snc_types.SupplyNodeType,
            constituency_matrix: snc_types.ConstituencyMatrix,
    ) -> bool:
        """
        Checks the passed ids correspond to the supply resources automatically detected by the
        job generator.

        :param supply_ids: Sorted list of supply resources ids (rows of the constituency matrix).
        :param jg_supply_nodes: Dictionary of supply nodes detected by the job generator.
        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i corresponds to resource s (examples of
            activities are scheduling a buffer or routing to another resource).
        :return True if supply_ids are consistent with jg_supply_nodes or False otherwise.
        """
        jg_supply_ids = set()
        for _, action in jg_supply_nodes:
            resource = np.argwhere(constituency_matrix[:, action])[:, 0]
            assert resource.size == 1, \
                f"Only one supplier per activity allowed, but activity {action} belongs to " \
                f"multiple resources: {resource}."
            jg_supply_ids.add(resource[0])
        return sorted(list(jg_supply_ids)) == supply_ids

    @staticmethod
    def initialise_supply_buffers(supply_ids: List[int]) -> Dict[int, int]:
        """
        Return dictionary with keys supply buffer ids and values set to zero.

        :param supply_ids: Sorted list of supply resources ids (rows of the constituency matrix).
        :return Dictionary of supply buffer ids with their value set to zero.
        """
        return {i: 0 for i in supply_ids}

    @staticmethod
    def get_activity_supply_resource_association(
            supply_nodes: snc_types.SupplyNodeType,
            constituency_matrix: snc_types.ConstituencyMatrix
    ) -> Tuple[Dict[int, int], DefaultDict[int, List[int]]]:
        """
        Return two dictionaries associating actions and resources.

        :param supply_nodes: List of tuples of length 2 automatically detected by the job generator
            with first value the buffer being supplied and second value the supply action id.
        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i corresponds to resource s (examples of
            activities are scheduling a buffer or routing to another resource).
        :return (activity_to_resource, resource_to_activity):
            - activity_to_resource: Dict associating actions (keys) to resources (values).
            - resource_to_activity: Dict associating resources (keys) to actions (values).
        """
        activity_to_resource: Dict[int, int] = {}
        resource_to_activity = defaultdict(list)

        for node in supply_nodes:
            action = node[1]
            resource = np.flatnonzero(constituency_matrix[:, action])
            assert resource.size == 1, \
                f"Supply activity {action} belongs to multiple resources. That is not allowed!"
            activity_to_resource[action] = resource[0]
            resource_to_activity[resource[0]].append(action)

        for key in resource_to_activity.keys():
            resource_to_activity[key] = sorted(resource_to_activity[key])
        return activity_to_resource, resource_to_activity

    @staticmethod
    def get_supply_activity_to_buffer_association(
            supply_nodes: snc_types.SupplyNodeType
    ) -> Dict[int, int]:
        """
        Returns dictionary associating supply activities with receiving buffers.

        :param supply_nodes: List of tuples of length 2 automatically detected by the job generator
            with first value the buffer being supplied and second value the supply action id.
        :return Dict associating each action (key) to buffers (value).
        """
        activity_to_buffer: Dict[int, int] = {}
        for node in supply_nodes:
            buffer = node[0]
            action = node[1]
            activity_to_buffer[action] = buffer
        return activity_to_buffer

    def sum_supplier_outbound(
            self,
            routing_jobs_matrix: snc_types.BufferMatrix
    ) -> DefaultDict[int, int]:
        """
        Returns the total number of items supplied by each supply resource according to the
        `routing_jobs_matrix` from the job generator, without taking into account the amount of
        items available in the supply buffers.

        :param routing_jobs_matrix: Matrix similar to the buffer processing matrix but with the
            actual samples drawn by the job generator.
        :return Dict with keys and values being the resource id and the amount of supplied items
            according to the routing_jobs_matrix, respectively.
        """
        sum_outbound: DefaultDict[int, int] = defaultdict(int)
        for supply in self.supply_nodes:
            action = supply[1]
            resource = self.activity_to_res[action]
            sum_outbound[resource] += routing_jobs_matrix[supply]
        return sum_outbound

    def truncate_routing_matrix_supplier(
            self,
            supply_id: int,
            routing_jobs_matrix: snc_types.BufferMatrix,
            supply_buffer: int,
    ) -> snc_types.BufferMatrix:
        """
        When supply buffer of a given supplier is lower than suggested by `routing_job_matrix`, this
        method ensures the number of items that are supplied does not exceed the number of available
        items. When truncating the amount of supplied items, the buffers receiving these items are
        chosen with random priority.

        :param supply_id: Supply resource id.
        :param routing_jobs_matrix: Matrix similar to the buffer processing matrix but with the
            actual samples drawn by the job generator.
        :param supply_buffer: Number of items available at buffer of supply
        :return truncated routing matrix.
        """
        # Allocate items in the supply buffer to activities randomly.
        res_id_actions = self.resource_to_act[supply_id].copy()
        random.shuffle(res_id_actions)

        # Follow randomised order of activities.
        for action in res_id_actions:

            # Find receiving buffer.
            buffer = self.activity_to_supply_buffer[action]

            # Truncate to the available items in the supply buffer.
            if routing_jobs_matrix[buffer, action] > supply_buffer:
                routing_jobs_matrix[buffer, action] = supply_buffer

            # Update the supply buffer for the next action.
            supply_buffer -= routing_jobs_matrix[buffer, action]

        return routing_jobs_matrix

    def ensure_jobs_conservation(
            self,
            routing_jobs_matrix: snc_types.BufferMatrix,
            state_plus_arrivals: snc_types.StateSpace
    ) -> snc_types.BufferMatrix:
        """
        Overload original method from parent class to ensure that supply resources do not supply
        more items than what is available at their buffers. While doing that it also updates the
        number of items available at the supply buffers.
        Then, it runs the same method from the parent class to truncate the `routing_jobs_matrix`
        when more items are being serviced than those available in the buffers.

        :param routing_jobs_matrix: Matrix similar to the buffer processing matrix but with the
            actual samples drawn by the job generator.
        :param state_plus_arrivals: Current state plus new arrivals as given by the job generator.
        :return Adjusted routing_jobs_matrix.
        """
        sum_outbound = self.sum_supplier_outbound(routing_jobs_matrix)
        for i in self.supply_ids:
            if self.supply_buffers[i] < sum_outbound[i]:
                routing_jobs_matrix = self.truncate_routing_matrix_supplier(
                    i,
                    routing_jobs_matrix,
                    self.supply_buffers[i]
                )
                self.supply_buffers[i] = 0
            else:
                self.supply_buffers[i] -= sum_outbound[i]

        return super().ensure_jobs_conservation(routing_jobs_matrix, state_plus_arrivals)

    def get_num_items_state_without_demand(self) -> int:
        """
        Returns the number of items in the current state without counting the demand buffers out.

        :return Number of items in the state without demand.
        """
        state_nd = self.state.copy()
        state_nd[self.demand_ids] = 0
        return np.sum(state_nd)

    def get_num_items_supply_buff(self) -> int:
        """
        Returns the total number of items available in the supplier buffers.

        :return Total number of items available in the supplier buffers.
        """
        num_items_supply_buff = 0
        for val in self.supply_buffers.values():
            num_items_supply_buff += val
        return num_items_supply_buff

    def get_num_items_in_transit_to_suppliers(self) -> int:
        """
        Returns the total number of items in-transit to any supplier.

        :return Total number of items in-transit.
        """
        num_items_in_transit = 0
        for parcels in self.in_transit_parcels.values():
            for parcel in parcels:
                amount = parcel[1]
                num_items_in_transit += amount
        return num_items_in_transit

    def assert_remains_closed_network(self) -> None:
        """
        Asserts the network is closed. Meaning that the total number of items in the network,
        computed as the sum of those in any standard buffer (i.e., represented by the state) plus
        those in the supply buffers and in-transit are equal to the initial total number of items.

        :return None.
        """
        num_items_supply_buff = self.get_num_items_supply_buff()
        num_items_in_transit = self.get_num_items_in_transit_to_suppliers()
        num_items_state_nd = self.get_num_items_state_without_demand()
        current_total_num_items = num_items_state_nd + num_items_supply_buff + num_items_in_transit

        assert current_total_num_items == self.num_initial_items, \
            f"Network is not closed! Total number of items has changed:" \
            f"Current={current_total_num_items} != Initial={self.num_initial_items}."

    @staticmethod
    def get_satisfied_demand(
            drained_amount: snc_types.StateSpace,
            demand_ids: List[int]
    ) -> Dict[int, int]:
        """
        Return number of items drained from each demand buffer.

        :param drained_amount: Number of jobs drained by all buffers in the network. This is
            obtained from the step() method of the parent class.
        :param demand_ids: Sorted list of demand buffers ids, as rows of buffer processing matrix.
        :return Dictionary with keys the ids of the demand buffers and values the amount of items
            that have been drained from those buffers.
        """
        satisfied_demand = dict()
        for i in demand_ids:
            satisfied_demand[i] = np.abs(drained_amount[i])
        return satisfied_demand

    def fill_in_transit_to_suppliers(self, satisfied_demand: Dict[int, int]) -> None:
        """
        Fill in transit buffers with items coming from the demand and their associated delays.

        :param satisfied_demand: Dictionary with keys being delivery times and value being a tuple
            with the parcel info namely destination (supply resource ID) and amount of items.
        :return None.
        """
        for demand_id, amount in satisfied_demand.items():
            if amount > 0:
                # Get id of destination and time of arrival.
                supply_id, toa = self.demand_to_supplier_routes[demand_id]
                delivery_time = self._t + toa
                # Add parcel destination and amount to be delivered at the desired time.
                self.in_transit_parcels[delivery_time].append((supply_id, amount))

    def fill_supply_buffers(self) -> None:
        """
        Fill supply buffers with what arrives from what was in-transit from the satisfied demand.

        :return None.
        """
        if self._t in self.in_transit_parcels:
            parcel_lst = self.in_transit_parcels.pop(self._t)
            for supply_id, amount in parcel_lst:
                self.supply_buffers[supply_id] += amount

    def step(self, action: snc_types.ActionSpace) \
            -> Tuple[snc_types.StateSpace, float, bool, Any]:
        """
        Overload `step` method from `ControlledRandomWalk` parent class to fill the supply and
        in-transit buffers.
        Supply buffers are first drained by the overloaded ensure_jobs_conservation method. Then,
        in-transit buffers are filled with what comes from the satisfied demand. Finally, supply
        buffers are re-filled again with what gets out of the in-transit buffers.

        :param action: Current action performed by the agent.
        :return: (state, reward, done, extra_data):
            - state: New state after performing action in current state.
            - reward: Reward (negative cost) obtained after the state transition.
            - done: Indicates if maximum number of steps has been reached. Currently used for
                training RL agents. It can implement other termination conditions in the future.
            - extra_data: Dictionary including new arrivals, total number of precessed jobs,
                processed jobs that have been added to any buffer (from suppliers or routed from
                other buffers), and processed jobs that have been drained from any buffer.
        """
        self.state, reward, done, extra_data = super().step(action)

        satisfied_demand = self.get_satisfied_demand(extra_data['drained'], self.demand_ids)
        self.fill_in_transit_to_suppliers(satisfied_demand)
        self.fill_supply_buffers()
        self.assert_remains_closed_network()
        return self.state, reward, done, extra_data
