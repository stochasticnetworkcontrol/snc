import json
import os
from itertools import permutations
import numpy as np

import pytest

import meio.gsm.tree_gsm as tree_gsm
import meio.gsm.dag_gsm as dag_gsm
from meio.gsm.common_test_utils import setup_coc_network, setup_skip_network
from meio.gsm.tree_gsm import GuaranteedServiceModelTree
from meio.gsm.utils import read_supply_chain_from_txt, create_gsm_instance, GSM
from meio.gsm.common_test_utils import (assert_solution_policies_equal,
                                        compare_with_brute_force_solution,
                                        check_labeling_invariance, check_additional_constraints,
                                        setup_cyclic_network)
from meio.experiment.basic_serial_network import create_serial_stages


dirname = os.path.dirname(__file__)


def test_cyclic_handling():
    stages = setup_cyclic_network()
    with pytest.raises(tree_gsm.IncompatibleGraphTopology):
        # Should fail because of cycles
        GuaranteedServiceModelTree(stages)


def test_skip_handling():
    stages = setup_skip_network()
    with pytest.raises(tree_gsm.IncompatibleGraphTopology):
        # Should fail because of cycles
        GuaranteedServiceModelTree(stages)


def test_coc_handling():
    stages = setup_coc_network()
    with pytest.raises(tree_gsm.IncompatibleGraphTopology):
        # Should fail because of bipartite graph not a tree
        GuaranteedServiceModelTree(stages)


@pytest.mark.parametrize("supply_chain_filename, assertion_failure", [
    ("bulldozer.txt", False),
    ("food.txt", True),
    ("toys.txt", True),
    ("semiconductors.txt", True),
    ("chemicals.txt", True),
])
def test_check_tree_topology(supply_chain_filename,assertion_failure):
    supply_chain_filename = os.path.join(dirname, supply_chain_filename)
    stages = read_supply_chain_from_txt(supply_chain_filename)
    if assertion_failure:
        with pytest.raises(tree_gsm.IncompatibleGraphTopology):
            tree_gsm.GuaranteedServiceModelTree(stages)
    else:
        tree_gsm.GuaranteedServiceModelTree(stages)


@pytest.mark.parametrize("supply_chain_name, inv_holding_rate, total_cost", [
    ("bulldozer", 0.3, 633000),
    ("battery", 0.25, 853000)
])
def test_against_ground_truth(supply_chain_name, inv_holding_rate, total_cost):

    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "{}.txt".
                                                             format(supply_chain_name)))

    sol = gsm.find_optimal_solution()
    sol = sol.serialize()
    policy = sol["policy"]

    inventory_costs = tree_gsm.compute_expected_inventory_costs(policy, stages)

    true_solution_filename = os.path.join(dirname, "{}_solution.json".format(supply_chain_name))
    with open(true_solution_filename,"r") as f:
        true_solution = json.load(f)

    assert len(policy) == len(true_solution)
    acc_total_cost = 0
    for stage_id, pol in true_solution:
        assert policy[stage_id]["s"] == pol["s"], pol["s"]
        if pol["cost"] == 0:
            assert inventory_costs[stage_id] == 0
        else:
            assert abs(inventory_costs[stage_id]*inv_holding_rate-pol["cost"])/pol["cost"] <= 0.001

        acc_total_cost += pol["cost"]

    assert abs(acc_total_cost - total_cost)/total_cost <= 0.001 

    assert abs(sol["cost"]*inv_holding_rate-total_cost)/total_cost <= 0.001


@pytest.mark.parametrize("scenario, service_time_constraints, correct_cost", [
    [1, [], 71757],
    [2, [("Imager", 0)], 78000],
    [3, [("Imager", 0), ("Transfer", 0), ("Build_Test_Pack", 0)], 89000],
    [4, [("Imager", 0), ("Transfer", 0)], 81000]])
def test_against_gound_truth_camera(scenario, service_time_constraints, correct_cost):

    inventory_holding_rate = 0.24

    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "digital_camera.txt"))

    for stage_id,max_s_time in service_time_constraints:
        gsm.stages[stage_id].max_s_time = max_s_time

    sol = gsm.find_optimal_solution()
    sol = sol.serialize()

    solution_cost = sol["cost"]*inventory_holding_rate
    assert abs(solution_cost-correct_cost)/correct_cost < 0.01

    policy = sol["policy"]
    inventories = tree_gsm.compute_expected_inventories(policy, gsm.stages)

    if scenario == 2:
        for stage_id in stages:

            if stage_id in ["Camera",
                            "Imager",
                            "Circuit_Board",
                            "Other_Parts_L_60",
                            "Other_Parts_M_60",
                            "Build_Test_Pack"]:

                assert policy[stage_id]["s"] == 0
                assert inventories[stage_id] > 0
            else:
                assert inventories[stage_id] == 0
                if stage_id == "Transfer":
                    assert policy[stage_id]["s"] == 2

                elif stage_id == "Ship":
                    assert policy[stage_id]["s"] == 5

    elif scenario == 4:
        for stage_id in stages:
            if stage_id in ["Build_Test_Pack","Ship"]:
                assert inventories[stage_id] == 0
            else:
                assert inventories[stage_id] > 0


def test_optimal_solution_invariance_to_labeling():
    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "bulldozer.txt"))

    check_labeling_invariance(gsm_obj=gsm)


@pytest.mark.parametrize("supply_chain_name, constraints_list", [
    ("bulldozer",[(("Fans", "max_s"), 6),
                  (("Chassis&Platform", "max_s"), 5)]),
    ("bulldozer",[(("Fans", "max_s"), 6),
                  (("Chassis&Platform", "max_s"), 5),
                  (("Common_Subassembly", "min_si"), 16)]),
    ("bulldozer",[(("Fans", "max_s"), 6),
                  (("Chassis&Platform", "max_s"), 5),
                  (("Common_Subassembly", "min_si"), 16),
                  (("Pin_Assembly", "max_s"), 5)]),
])
def test_additional_constraints(supply_chain_name,constraints_list):
    supply_chain_filename = os.path.join(dirname, "{}.txt".format(supply_chain_name))
    stages = read_supply_chain_from_txt(supply_chain_filename)
    gsm = tree_gsm.GuaranteedServiceModelTree(stages)

    unconstrained_solution = gsm.find_optimal_solution()

    monotone_solution = check_additional_constraints(gsm_obj=gsm,
                                                     constraints_list=constraints_list,
                                                     unconstrained_solution=unconstrained_solution,
                                                     monotone_increase=True)
    monotone_done = False
    for const_list in permutations(constraints_list):
        if (not monotone_done) and const_list == constraints_list:
            monotone_done = True
        else:
            solution = check_additional_constraints(gsm_obj=gsm, constraints_list=const_list)
            assert abs(solution.cost - monotone_solution.cost)/monotone_solution.cost <= 0.001


@pytest.mark.parametrize("supply_chain_name, stage_modify, new_rate", [
    ("bulldozer_small_convergent", "Case&Frame", 550),
    ("battery_small_divergent", "Pack_SKU_A", 0.01),
   ("bulldozer_small_linear", "Main_Assembly", 1200)
])
def test_vs_brute_force(supply_chain_name, stage_modify, new_rate):
    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "{}.txt".
                                                             format(supply_chain_name)))

    sol_1 = gsm.find_optimal_solution()
    sol_1 = sol_1.serialize()

    bf_solution_filename_1 = os.path.join(dirname, "brute_force_{}_sol_1.json".
                                          format(supply_chain_name))
    compare_with_brute_force_solution(sol_1, stages, bf_solution_filename_1, recompute=True)

    # change the balance of costs
    stages[stage_modify].cost_rate = new_rate
    sol_2 = gsm.find_optimal_solution()
    sol_2 = sol_2.serialize()

    bf_solution_filename_2 = os.path.join(dirname, "brute_force_{}_sol_2.json".
                                          format(supply_chain_name))
    compare_with_brute_force_solution(sol_2, stages, bf_solution_filename_2, recompute=True)

    with pytest.raises(AssertionError):
        assert_solution_policies_equal(sol_1["policy"],sol_2["policy"], stages)


def test_expected_inventories_and_basestocks_computations():
    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "bulldozer.txt"))

    sol = gsm.find_optimal_solution()
    expected_inventories = tree_gsm.compute_expected_inventories(sol.policy, stages)
    base_stocks = tree_gsm.compute_base_stocks(sol.policy, stages)
    assert len(stages) == len(base_stocks)
    assert len(base_stocks) == len(expected_inventories)


@pytest.mark.parametrize("supply_chain_filename", [
    "bulldozer.txt",
    "food.txt",
    "toys.txt",
    "semiconductors.txt",
    "chemicals.txt",
])
def test_compute_basic_solution_cost(supply_chain_filename):
    supply_chain_filename = os.path.join(dirname, supply_chain_filename)
    stages = read_supply_chain_from_txt(supply_chain_filename)
    gsm = dag_gsm.GuaranteedServiceModelDAG(stages)
    sol = gsm.find_optimal_solution()

    basic_sol_cost = tree_gsm.compute_basic_solution_cost(stages)

    if sol.cost > basic_sol_cost:
        np.testing.assert_approx_equal(sol.cost,basic_sol_cost)

    additional_constraints = {}
    for stage_id in stages:
        additional_constraints[(stage_id,"max_s")] = 0

    const_sol = gsm.find_optimal_solution(constraints=additional_constraints)

    np.testing.assert_approx_equal(const_sol.cost,basic_sol_cost)


def test_gsm_solution_class():
    stages, gsm = create_gsm_instance(GSM.Tree, os.path.join(dirname, "bulldozer.txt"))

    sol = gsm.find_optimal_solution()
    sol_dict = sol.serialize()

    sol_2 = tree_gsm.GSM_Solution(**sol_dict)
    sol_2_dict = sol_2.serialize()

    assert_solution_policies_equal(sol_dict["policy"],sol_2_dict["policy"],stages)
    assert sol_dict["cost"] == sol_2_dict["cost"]


def test_against_basic_serial_network_experiment():
    """
    Hand coded answers were taken from Graves and Schoenmeyr 2016 (Table 5, No constraint row)
    """
    stages = create_serial_stages(added_cost_prof="constant",
                                  lead_time_prof="upstream_heavy")

    gsm = GuaranteedServiceModelTree(stages)
    solution = gsm.find_optimal_solution(root_stage_id="4")
    # there is another policy with the same cost
    # it just happens that lead time and cost numbers add up this way
    assert int(solution.cost) == 368
    safety_stocks = tree_gsm.compute_expected_inventories(solution.policy, stages)
    assert int(safety_stocks["5"]) == 240
    assert int(safety_stocks["1"]) == 320


@pytest.mark.parametrize("added_cost_prof, lead_time_prof, uncap_cost", [
    ("upstream_heavy","upstream_heavy",400),
    ("upstream_heavy","constant",400),
    ("upstream_heavy","downstream_heavy",400),
    ("constant","upstream_heavy",368),
    ("constant","constant",394),
    ("constant","downstream_heavy",400),
    ("downstream_heavy","upstream_heavy",268),
    ("downstream_heavy","constant",346),
    ("downstream_heavy","downstream_heavy",392),
])
def test_against_basic_serial_network_experiments_uncap(added_cost_prof,lead_time_prof,uncap_cost):
    """
    Hand coded answers were taken from Graves and Schoenmeyr 2016 (Table 4)
    """
    stages = create_serial_stages(added_cost_prof=added_cost_prof,
                                  lead_time_prof=lead_time_prof)
    gsm = GuaranteedServiceModelTree(stages)
    solution = gsm.find_optimal_solution()
    assert int(solution.cost) == uncap_cost or np.ceil(solution.cost) == uncap_cost


@pytest.mark.parametrize("added_cost_prof, lead_time_prof, cap_loc, cap_cost, tol", [
    ("upstream_heavy","upstream_heavy","1",400*0.89,2),
    ("upstream_heavy","constant","1",400*0.91,1),
    ("upstream_heavy","downstream_heavy","1",400*0.92,1),
    ("constant","upstream_heavy","1",368*0.73,1),
    ("constant","constant","1",394*0.86,1),
    ("constant","downstream_heavy","1",400*0.91,1),
    ("downstream_heavy","upstream_heavy","1",268*0.65,1),
    ("downstream_heavy","constant","1",346*0.78,2),
    ("downstream_heavy","downstream_heavy","1",392*0.86,2),

    ("upstream_heavy","upstream_heavy","2",400*1.03,2),

    # these two configs give a bit too high of an error which cannot be attributed to rounding up
    ("upstream_heavy","constant","2",400*1.09,5),
    ("upstream_heavy","downstream_heavy","2",400*1.10,6),

    ("constant","upstream_heavy","2",368*0.87,1),
    ("constant","constant","2",394*1.03,3),
    ("constant","downstream_heavy","2",400*1.08,2),
    ("downstream_heavy","upstream_heavy","2",268*0.81,2),
    ("downstream_heavy","constant","2",346*0.96,3),
    ("downstream_heavy","downstream_heavy","2",392*1.01,2),

    ("upstream_heavy","upstream_heavy","3",400*1.06,3),
    ("upstream_heavy","constant","3",400*1.08,3),
    ("upstream_heavy","downstream_heavy","3",400*1.09,4),
    ("constant","upstream_heavy","3",368*0.93,2),
    ("constant","constant","3",394*1.01,1),
    ("constant","downstream_heavy","3",400*1.06,1),
    ("downstream_heavy","upstream_heavy","3",268*0.91,1),
    ("downstream_heavy","constant","3",346*0.95,2),
    ("downstream_heavy","downstream_heavy","3",392*0.98,1),

    ("upstream_heavy","upstream_heavy","4",400*1.04,1),
    ("upstream_heavy","constant","4",400*1.06,1),
    ("upstream_heavy","downstream_heavy","4",400*1.07,3),
    ("constant","upstream_heavy","4",368*0.95,2),
    ("constant","constant","4",394*0.99,2),
    ("constant","downstream_heavy","4",400*1.04,2),
    ("downstream_heavy","upstream_heavy","4",268*0.97,1),
    ("downstream_heavy","constant","4",346*0.98,2),
    ("downstream_heavy","downstream_heavy","4",392*0.98,1),

    ("upstream_heavy","upstream_heavy","5",400*0.99,1),
    ("upstream_heavy","constant","5",400*1.03,1),
    ("upstream_heavy","downstream_heavy","5",400*1.04,1),
    ("constant","upstream_heavy","5",368*0.98,1),
    ("constant","constant","5",394*0.98,1),
    ("constant","downstream_heavy","5",400*1.01,1),
    ("downstream_heavy","upstream_heavy","5",268*1.0,2),
    ("downstream_heavy","constant","5",346*1.0,2),
    ("downstream_heavy","downstream_heavy","5",392*1.0,1),
])
def test_against_basic_serial_network_experiments_cap(added_cost_prof,lead_time_prof,cap_loc,cap_cost,tol):
    """
    Hand coded answers were taken from Graves and Schoenmeyr 2016 (Table 4) where capacity constraint is
    (c=45).
    """
    stages = create_serial_stages(added_cost_prof=added_cost_prof,
                                  lead_time_prof=lead_time_prof)
    stages[cap_loc].cap_constraint = 45
    gsm = GuaranteedServiceModelTree(stages,propagate_bounds=True)
    solution = gsm.find_optimal_solution()
    back_orders_correction = 30  # see text on page of 13 of Graves and Schoenmeyr
    cost_correction = int(back_orders_correction * stages[cap_loc].cost_rate)
    assert abs(int(solution.cost)-(np.ceil(cap_cost) + cost_correction)) <= tol


@pytest.mark.parametrize("added_cost_prof, lead_time_prof, cap_loc, cap_cost, exp_safety_stocks", [
    ("constant","upstream_heavy","1",300,{"1":100,"2":60,"3":100,"4":140,"5":180}),
    ("constant","upstream_heavy","2",344,{"1":80,"2":140,"3":100,"4":140,"5":180}),
    ("constant","upstream_heavy","3",360,{"1":160,"2":0,"3":179,"4":140,"5":180}),
    ("constant","upstream_heavy","4",361,{"1":240,"2":0,"3":0,"4":212,"5":180}),
    ("constant","upstream_heavy","5",368,{"1":320,"2":0,"3":0,"4":0,"5":240}),
])
def test_against_basic_serial_network_experiments_cap_location(added_cost_prof,lead_time_prof,cap_loc,cap_cost,exp_safety_stocks):
    """
    Hand coded answers were taken from Graves and Schoenmeyr 2016 (Table 5)
    """
    stages = create_serial_stages(added_cost_prof=added_cost_prof,
                                  lead_time_prof=lead_time_prof)
    stages[cap_loc].cap_constraint = 45
    gsm = GuaranteedServiceModelTree(stages,propagate_bounds=True)

    solution = gsm.find_optimal_solution()
    safety_stocks = tree_gsm.compute_expected_inventories(solution.policy, stages)
    print(tree_gsm.compute_replenishment_times(solution.policy, stages))

    assert abs(int(solution.cost)-cap_cost) <= 1

    for stage_id in exp_safety_stocks:
        assert abs(safety_stocks[stage_id]-exp_safety_stocks[stage_id]) <= 1
