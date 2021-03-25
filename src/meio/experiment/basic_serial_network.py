"""
This module implements the experimental setup as described in
Graves and Schoenmeyr 2016 "Strategic Safety-Stock Placement in Supply Chains
with Capacity Constraints" section 4. Numerical Experiments

The purpose of implementing this simple experiment is threefold:
   - replicate results state in the paper and verify validity of GSM modules
   - have a controllable environment to investigate cascading stocks
   - benchmark different GSM policies as well as Hedgehog ones
"""

from typing import Dict,List
import numpy as np

from src.meio import Stage

ADDED_COST_PROFILES = {
    "upstream_heavy":(0.36, 0.28, 0.20, 0.12, 0.04),
    "constant":(0.2, 0.2, 0.2, 0.2, 0.2),
    "downstream_heavy":(0.04, 0.12, 0.20, 0.28, 0.36),
}

LEAD_TIME_PROFILES = {
    "upstream_heavy":(36, 28, 20, 12, 4),
    "constant":(20, 20, 20, 20, 20),
    "downstream_heavy":(4, 12, 20, 28, 36),
}

CAPACITY_CONSTRAINTS = (42, 45, 50, 60, 70)

DEMAND_STAGE_PARAMS = {
    "mean":40,
    "std": 20,
    "thres":2
}


def create_serial_stages(added_cost_prof: str, lead_time_prof: str) -> Dict[str, Stage]:
    """
    This function takes as inputs desired added cost and lead time profiles
    from the available selection and initialises individual stages objects
    corresponding to experimental setup
    """

    added_costs = ADDED_COST_PROFILES[added_cost_prof]
    lead_times = LEAD_TIME_PROFILES[lead_time_prof]
    stages = {} # type: Dict[str,Stage]

    stages["1"] = Stage(_id="1", lead_time=lead_times[-1], max_s_time=0,
                        added_cost=added_costs[-1],
                        up_stages={"2":1}, down_stages={},
                        is_ext_demand_stage=True,
                        demand_mean=DEMAND_STAGE_PARAMS["mean"],
                        demand_std=DEMAND_STAGE_PARAMS["std"],
                        demand_thres=DEMAND_STAGE_PARAMS["thres"])

    for i in range(2, 6):
        stages[str(i)] = Stage(_id=str(i), lead_time=lead_times[-i],
                               max_s_time=np.inf,
                               added_cost=added_costs[-i],
                               up_stages={str(i+1):1} if i < 5 else {},
                               down_stages={str(i-1):1},
                               risk_pool=2)

    return stages


def iterate_experimental_profiles():
    for added_cost_prof in ADDED_COST_PROFILES:
        for lead_time_prof in LEAD_TIME_PROFILES:
            yield {"added_cost_prof":added_cost_prof, "lead_time_prof":lead_time_prof}


def create_serial_line_from_lead_times(lead_times: List[int]):
    N = len(lead_times)
    stages = {}

    for i in range(1,N+1):
        stages[str(i)] = Stage(_id=str(i), lead_time=lead_times[i-1],
                               max_s_time=np.inf,
                               added_cost=0,
                               up_stages={str(i+1):1} if i < N else {},
                               down_stages={str(i-1):1} if i > 1 else {},
                               risk_pool=2)

    return stages
