from typing import Dict, List

from src.snc.demand_planning.demand_plan_interface import DemandPlanInterface


class ConstantDemandPlan(DemandPlanInterface):
    def __init__(self,
                 ind_surplus_buffers: List[int],
                 demand_plan_values: Dict[int, int]):

        ind_surplus_dp = sorted(list(demand_plan_values.keys()))
        assert sorted(ind_surplus_buffers) == ind_surplus_dp, \
            f"Indexes of surplus buffers in 'demand_plan_values': {sorted(demand_plan_values)}, " \
            f"don't match those in 'ind_surplus_buffers': {ind_surplus_dp}."
        super().__init__(ind_surplus_dp)
        self.demand_plan_values = demand_plan_values

    def get_demand_plan(self, **kwargs) -> Dict[int, int]:
        return self.demand_plan_values
