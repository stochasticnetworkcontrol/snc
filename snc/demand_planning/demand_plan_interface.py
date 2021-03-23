from typing import Dict, List


class DemandPlanInterface:
    def __init__(self,
                 ind_surplus_buffers: List[int]):
        self.ind_surplus_buffers = ind_surplus_buffers

    def get_demand_plan(self, **kwargs) -> Dict[int, int]:
        """
        Main method that provides demand_plan.
        """
        raise NotImplementedError('This method is meant to be overloaded.')
