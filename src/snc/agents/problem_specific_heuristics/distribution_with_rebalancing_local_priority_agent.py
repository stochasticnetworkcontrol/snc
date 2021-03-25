import numpy as np
from typing import Dict, Optional, Set, Tuple, Any

from src.snc.utils import snc_types as types
from src.snc.environments import controlled_random_walk as crw
from src.snc import AgentInterface


class DistributionWithRebalancingLocalPriorityAgent(AgentInterface):

    def __init__(self, env: crw.ControlledRandomWalk,
                 safety_stocks: Optional[types.StateSpace] = None,
                 name: str = "DistributionWithRebalancingLocalPriorityAgent") -> None:
        """
        Heuristic policy for a distribution with rebalancing example where
        where:
        - a supplier produces items and can send them to one warehouse,
        - a warehouse can send items to one manufacturer,
        - a warehouse can send items to other warehouses
        - a manufacturer consumes the item
        In this example we consider 3 suppliers, 3 warehouses, and 2 manufacturers in total.

        The policy is deterministic and only built for the specific example. The policy depends only
        on the state and safety stocks (ss). It is simple.

        Consider the example as 3 'chains' of buffers ['warehouse' -> ('supply', 'demand')]:
        * 1. IF 'supply' < ss[supply]:
                refill TAKE ACTIONS '->warehouse',
                IF warehouse > 0
                    refill TAKE ACTION 'warehouse->supply'
        * 2. IF 'warehouse' < ss[warehouse] - refill TAKE ACTIONS '->warehouse'
        * 2. IF 'warehouse' > ss[warehouse] - refill REMOVE ACTIONS '->warehouse'
        * 3. IF 'demand' >  ss[demand] (=0) AND related supply > 0:
                drain  TAKE ACTIONS 'demand+supply->'
        * 4. IF 'warehouse' >= ss[warehouse] AND NOT 'warehouse' CURRENTLY TAKING ACTIONS:
                FIND 'warehouse_min', the warehouse with MIN(w - ss[w])
                IF  'warehouse_min' < ss[warehouse_min]
                    TAKE ACTION 'w->warehouse_min'
                    IF 'w == ss[w]'
                        TAKE ACTION '->w'

        :param env: the environment to stepped through.
        :param safety_stocks: the safety stocks used for the policy.
        :param name: Agent identifier.
        :return: None.
        """
        super(DistributionWithRebalancingLocalPriorityAgent, self).__init__(env, name)
        self.num_activities = self.constituency_matrix.shape[1]

        if safety_stocks is not None:
            self.safety_stocks = safety_stocks
        else:
            self.safety_stocks = None

        self.map_activity_to_index = {
            'act1': 0, 'act2': 1, 'act3': 2, 'act5': 3,
            'act7': 4, 'act8': 5, 'act9': 6, 'act11': 7,
            'act13': 8, 'act14': 9, 'act15': 10,
            'act17': 11, 'act19': 12, 'act20': 13, 'act21': 14,
        }

    def perform_offline_calculations(self) -> None:
        """
        Calculate our safety stocks given the parameters of the network and arrivals.
        """
        if self.safety_stocks is not None:
            return
        else:
            self.safety_stocks = np.array([10, 5, 0, 10, 5, 0, 10, 5, 0])[:, None]

    def get_greedy_actions_no_rebalancing(self, state: types.StateSpace) -> Set[str]:
        """
        Get the actions (expressed as strings from the diagram) that correspond to the greedy
            refilling actions, based on safety stock levels.
        NB: this corresponds to finding the actions from steps 1-3 of the algorithm as above.

        :param state: The current state of the environment
        :returns: a set of the actions to be taken {'act1', 'act7'} etc
        """
        actions = set()  # type: Set[str]

        greedy_refill_actions = {
            # buffer: ({in_deficit_actions}, {in_surplus_actions},
            #          feeder_buffer, {empty_feeder_action})
            'buff1': ({'act3'}, {'-act3'}, None, set()),
            'buff4': ({'act9'}, {'-act9'}, None, set()),
            'buff7': ({'act15'}, {'-act15'}, None, set()),
            'buff2': ({'act1', 'act3'}, set(), 'buff1', {'-act1'}),
            'buff5': ({'act7', 'act9'}, set(), 'buff4', {'-act7'}),
            'buff8': ({'act13', 'act15'}, set(), 'buff7', {'-act13'}),
            'buff3': (set(), {'act2'}, 'buff2', {'-act2'}),
            'buff6': (set(), {'act8'}, 'buff5', {'-act8'}),
            'buff9': (set(), {'act14'}, 'buff8', {'-act14'}),
        }  # type: Dict[str, Tuple[Set[str], Set[str], Optional[str], Set[str]]]

        for i, (x, ss) in enumerate(zip(state, self.safety_stocks), 1):
            buff = 'buff{}'.format(i)
            if x < ss:
                a = greedy_refill_actions[buff][0]
            elif x > ss:
                a = greedy_refill_actions[buff][1]
            else:
                a = set()

            feeder_buff = greedy_refill_actions[buff][2]

            if feeder_buff is not None:
                feeder_index = int(feeder_buff[4:]) - 1
                if state[feeder_index, 0] == 0:
                    a.update(greedy_refill_actions[buff][3])

            actions.update(a)

        removals = {a for a in actions if
                    a.startswith('-')}  # actions:  # {'act1', 'act3', '-act3'}
        removals |= {r[1:] for r in removals}  # removals {'-act3'} -> {'-act3', 'act3'}
        return actions - removals

    def get_rebalancing_actions(self, state: types.StateSpace, action_list: Set[str]) -> Set[str]:
        """
        Get the actions (expressed as strings from the diagram) that correspond to rebalancing
            actions, taking into account the current greedy actions already chosen.
        NB: this corresponds to finding the actions from step 4 of the algorithm as above.

        :param state: The current state of the environment.
        :param action_list: A set of the actions currently taken as a result of greedy refilling.
        :returns: a set of the actions to be taken {'act21'} for rebalancing.
        """

        warehouse_indices = np.array([1, 4, 7]) - 1
        warehouse_rebalance_actions = [
            # ((refill_supply, refill_warehouse), {to_warehouse: rebalance_action})
            (('act1', 'act3'), {1: 'act5', 2: 'act19'}),
            (('act7', 'act9'), {2: 'act11', 0: 'act20'}),
            (('act13', 'act15'), {0: 'act17', 1: 'act21'}),
        ]

        warehouse_balance = state[warehouse_indices] - self.safety_stocks[warehouse_indices]
        is_warehouse_in_deficit = warehouse_balance < 0

        actions = set()
        for w, in_deficit in enumerate(is_warehouse_in_deficit):
            if in_deficit:
                # skip rebalance if in deficit
                continue

            refill_actions, rebalance_action_map = warehouse_rebalance_actions[w]
            refill_supply_action, refill_warehouse_action = refill_actions
            if refill_supply_action in action_list:
                # skip rebalance if already refilling supply buffer
                continue

            w_min = np.argmin(warehouse_balance)
            if warehouse_balance[w_min] < 0:
                actions.add(rebalance_action_map[int(w_min)])
                if warehouse_balance[w] == 0:
                    # if at safety stock level and rebalancing, refill warehouse
                    actions.add(refill_warehouse_action)
        return actions

    def map_state_to_actions(self, state: types.StateSpace, **override_args: Any) \
            -> types.ActionProcess:
        """
        Map the state to set of actions {'act1', 'act13'} as enumerated in the diagram. Then
        convert these to the indices of the actual action vector.

        :param state: Current state of the system.
        :return action: action vector.
        """
        action = np.zeros((self.num_activities, 1))

        action_list = self.get_greedy_actions_no_rebalancing(state)
        action_list.update(self.get_rebalancing_actions(state, action_list))

        for a in action_list:
            action[self.map_activity_to_index[a], 0] = 1

        return action
