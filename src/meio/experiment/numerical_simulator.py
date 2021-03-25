"""
Module implements numerical simulator of GSM replenishment policy
"""

from typing import Dict, List, Optional

from scipy.stats import poisson
import numpy as np

import meio.gsm.tree_gsm as tree_gsm
from meio.gsm.tree_gsm import GSM_Policy
from meio.gsm.types import GSM_Stage_Policy


np.random.seed(seed=8675309)


def _get_stage_replenishment_time(stage: tree_gsm.Stage, stage_policy: GSM_Stage_Policy) -> int:
    """
    Function simply computes net replenishment time from stage's lead time and its gsm policy
    service time 's' and internal service time 'si'
    """
    s = stage_policy["s"]
    si = stage_policy["si"]
    l = stage.lead_time

    tau = si + l - s
    return tau


def compute_base_stocks(stages: Dict[str, tree_gsm.Stage],
                        gsm_policy: GSM_Policy,
                        daily_demand_mean: float,
                        sla: float) -> Dict[str, int]:
    """
    Function to compute basestock inventory levels for all stages in the network
    from the provided gsm_policy and required SLA asuming Poisson distributed daily demand
    Base stock is the initial amount of stock in the inventory.
    """
    # TODO Compute basestocks for different undelying demand distributions, not only poisson
    lam = daily_demand_mean

    base_stocks = {}
    for stage_id in stages:
        tau = _get_stage_replenishment_time(stages[stage_id], gsm_policy[stage_id])
        base_stocks[stage_id] = np.ceil(poisson.ppf(sla, tau * lam))

    return base_stocks


def compute_safety_stocks(stages: Dict[str, tree_gsm.Stage],
                          gsm_policy: GSM_Policy,
                          daily_demand_mean: float,
                          sla: float) -> Dict[str, int]:
    """
    Function to compute average running safety stock levels for all stages in the network
    from the provided gsm_policy and required SLA asuming Poisson distributed daily demand
    Safety stock is the running average amount of items in the inventory
    """
    # TODO Compute safety stocks for different undelying demand distributions, not only poisson
    lam = daily_demand_mean

    safety_stocks = {}
    for stage_id in stages:
        tau = _get_stage_replenishment_time(stages[stage_id], gsm_policy[stage_id])
        safety_stocks[stage_id] = np.ceil(poisson.ppf(sla, tau * lam)) - np.ceil(tau * lam)

    return safety_stocks


def simulate(stages: Dict[str, tree_gsm.Stage],
             gsm_policy: GSM_Policy,
             base_stocks: Dict[str, int],
             capacity_constraints: Dict[str, int],
             demand_history: np.ndarray,
             stockout_stages: Optional[List[str]]=None) -> Dict[str, np.ndarray]:
    """
    :param stages: serial network to simulate
    :param gsm_policy:
    :param base_stocks: basestock setting for each stage in the network
    :param capacity_constraints: dictionary of maximum throughoutput capacity for each stage
    :param demand_history: simulated demand sequence
    :param stockout_stages: List of stages for which to propagate the stockouts donwstream
                            If None, all stages are propagated

    :returns: dictionary of inventory position histories for all stages
    """
    for stage_id in stages:
        assert len(stages[stage_id].down_stages) <= 1, "network needs to be serial line"
        assert len(stages[stage_id].up_stages) <= 1, "network needs to be serial line"

    # get ordered list of stages starting from the most downstream one
    ordered_stages_list = tree_gsm.GuaranteedServiceModel._order_stages(stages)
    n = len(demand_history)

    inv_pos_histories = {}  # type: Dict[str,np.ndarray]
    actual_replen_histories = {}  # type: Dict[str, np.ndarray]

    if stockout_stages is None:
        stockout_stages = ordered_stages_list

    # propagate orders
    repl_orders_histories = {}
    service_histories = {}
    sequence = demand_history
    min_cap_constraint = np.inf  # no truncation for demand
    for stage_id in ordered_stages_list:
        service_histories[stage_id] = sequence
        cap_constraint = capacity_constraints.get(stage_id, np.inf)
        if cap_constraint < min_cap_constraint:
            sequence = truncate_and_conserve(sequence, cap_constraint)
            min_cap_constraint = cap_constraint
        repl_orders_histories[stage_id] = sequence

    # propagate replenishments
    for stage_id in reversed(ordered_stages_list):
        stage = stages[stage_id]
        s = gsm_policy[stage_id]["s"]
        si = gsm_policy[stage_id]["si"]
        lt = stages[stage_id].lead_time

        b = base_stocks[stage_id]
        service_history = np.zeros(n + s + 1)
        service_history[-n:] = service_histories[stage_id]
        replen_history = np.zeros(n + si + lt + 1)
        replen_history[-n:] = repl_orders_histories[stage_id]

        for up_stage_id in stage.up_stages:
            # TODO figure out how to manage multiple sources and destinations
            replen_history[-n:] = actual_replen_histories[up_stage_id][-n:]
            if stage_id in capacity_constraints:
                cap_constraint = capacity_constraints[stage_id]
                replen_history = truncate_and_conserve(replen_history, cap_constraint)

        inv_pos_history = b + np.cumsum(replen_history[:len(service_history)]-service_history)
        # shape of inv_position is the same as service history: (n + s + 1)
        inv_pos_histories[stage_id] = inv_pos_history

        if stage_id in stockout_stages:

            pos = inv_pos_history.copy()
            pos[inv_pos_history < 0] = 0
            neg = -inv_pos_history.copy()
            neg[inv_pos_history > 0] = 0

            required_replen = neg[:-1] + service_history[1:]
            available_replen = pos[:-1] + replen_history[1:len(pos)]

            actual_replen = np.minimum(available_replen, required_replen)

            actual_replen_histories[stage_id] = actual_replen
        else:
            actual_replen_histories[stage_id] = service_history[1:]

    return inv_pos_histories


def truncate_and_conserve(sequence: np.ndarray, max_capacity: int) -> np.ndarray:
    """
    Function for redistributing excessive daily demand values above max_capacity
    into the future timesteps where demand is below max_capacity
    """
    assert len(sequence.shape) == 1, "sequence must be zero dimensional"
    new_sequence = np.zeros_like(sequence)
    cum_val = 0
    for i, val in enumerate(sequence):
        val = sequence[i]
        delta = val - max_capacity

        if delta <= 0:
            if cum_val == 0:
                new_sequence[i] = val

            elif abs(delta) < cum_val:
                # not enough slack capacity to clear surplus
                cum_val += delta
                new_sequence[i] = max_capacity

            elif abs(delta) >= cum_val:
                new_sequence[i] = val + cum_val
                cum_val = 0

        else:
            cum_val += delta
            new_sequence[i] = max_capacity

    return new_sequence
