import pytest

from src.snc import ConstantDemandPlan


def test_constant_demand_plan():
    ind_surplus_buffers = [1, 2]
    demand_plan_values = {1: 3, 2: 5}
    dp = ConstantDemandPlan(ind_surplus_buffers, demand_plan_values)
    assert demand_plan_values == dp.get_demand_plan()


def test_constant_demand_plan_mismatch():
    ind_surplus_buffers = [1, 3]
    demand_plan_values = {1: 3, 2: 5}
    with pytest.raises(AssertionError):
        _ = ConstantDemandPlan(ind_surplus_buffers, demand_plan_values)
