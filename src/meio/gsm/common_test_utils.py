import io
import json
from typing import Optional, List

import numpy as np

import src.meio.gsm.tree_gsm as tree_gsm
from src.meio import GSM_Constraints
from src.meio import create_supply_chain_network_from_iterator


def setup_cyclic_network():
    a_b_c = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                        'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                        'A,11,,575.0,2,,,,C,1\n'+\
                        'B,11,,575.0,2,,,,A,1\n'+\
                        'C,11,,575.0,2,,,,B,1')
    stages = create_supply_chain_network_from_iterator(a_b_c)
    assert len(stages) == 3
    return stages


def setup_skip_network():
    a_b_c = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                        'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                        'A,11,,575.0,2,,,,\n'+\
                        'B,11,,575.0,2,,,,A,1\n'+\
                        'C,11,,575.0,2,10,10,1.645,A,1,B,1')
    stages = create_supply_chain_network_from_iterator(a_b_c)
    assert len(stages) == 3
    return stages


def setup_coc_network():
    a_b_c_d = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                          'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                          'A,11,,575.0,2,,,,\n'+\
                          'B,11,,575.0,2,,,,\n'+\
                          'C,11,,575.0,2,10,10,1.645,A,1,B,1\n'+\
                          'D,11,,575.0,2,10,10,1.645,A,1,B,1')
    stages = create_supply_chain_network_from_iterator(a_b_c_d)
    assert len(stages) == 4
    return stages


def setup_diamond_network():
    a_b_c_d = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                          'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                          'A,2,,1,2,,,,\n' + \
                          'B,11,,2,2,,,,A,1\n' + \
                          'C,11,,2,2,,,,A,1\n' + \
                          'D,11,0,5,2,2,1,1.645,C,1,B,1')
    stages = create_supply_chain_network_from_iterator(a_b_c_d)
    assert len(stages) == 4
    return stages


def setup_diamond_network_extra_edge():
    # like diamond but has an additional edge A->D
    a_b_c_d = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                          'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                          'A,2,,1,2,,,,\n' + \
                          'B,11,,2,2,,,,A,1\n' + \
                          'C,11,,4,2,,,,A,1\n' + \
                          'D,11,0,5,2,2,1,1.645,C,1,B,1,A,1')
    stages = create_supply_chain_network_from_iterator(a_b_c_d)
    assert len(stages) == 4
    return stages


def setup_ext_diamond_network():
    a_b_c_d = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                          'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                          'A,2,,1,2,,,,\n' + \
                          'B,11,,2,2,,,,A,1\n' + \
                          'C,11,,2,2,,,,A,1\n' + \
                          'E,11,,2,2,10,5,1.645,A,1\n' + \
                          'D,11,0,5,2,2,1,1.645,C,1,B,1')
    stages = create_supply_chain_network_from_iterator(a_b_c_d)
    assert len(stages) == 5
    return stages


def setup_dist_network():
    a_b_c_d = io.StringIO('stage_id,lead_time,max_s_time,cost_rate,risk_pool,'+\
                          'ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n'+\
                          'A,2,,1,2,,,,\n' + \
                          'B,11,,2,2,,,,A,1\n' + \
                          'C,11,,2,2,,,,A,1\n' + \
                          'D,11,,2,2,,,,B,1\n' + \
                          'E,11,,2,2,20,10,1.645,C,1\n' + \
                          'F,11,,2,2,30,11,1.645,D,1\n' + \
                          'G,11,0,5,2,25,12,1.645,D,1')
    stages = create_supply_chain_network_from_iterator(a_b_c_d)
    assert len(stages) == 7
    return stages


def generate_policy(stage):
    for s in range(min(stage.max_s_bound, stage.max_s_time)+1):
        for si in range(max(0, s-stage.lead_time), max(0, stage.max_s_bound - stage.lead_time) + 1):
            yield {"s":s, "si":si}


def find_brute_force_solution(stages):
    """
    This test function is hardcoded for simple topologies consisting exactly of
    three stages and cannot be used for a different number of stages
    """
    assert len(stages) == 3

    min_cost = np.inf

    stage_ids = list(stages.keys())

    optimal_solutions = []
    counter = 0
    for i_policy in generate_policy(stages[stage_ids[0]]):
        for j_policy in generate_policy(stages[stage_ids[1]]):
            for k_policy in generate_policy(stages[stage_ids[2]]):
                solution = {
                    stage_ids[0]:i_policy,
                    stage_ids[1]:j_policy,
                    stage_ids[2]:k_policy
                }
                counter += 1
                try:
                    tree_gsm.verify_solution_policy(solution, stages)
                except AssertionError:
                    continue

                solution_cost = tree_gsm.compute_total_inventory_cost(solution, stages)

                if solution_cost == np.inf:
                    continue

                if np.isclose(solution_cost, min_cost):
                    optimal_solutions.append(solution)
                    continue

                if solution_cost < min_cost:
                    optimal_solutions = [solution]
                    min_cost = solution_cost
                    #print(counter,min_cost)

    return {"cost":min_cost, "optimal_solutions":optimal_solutions}


def compare_with_brute_force_solution(sol,stages,bf_solution_filename,recompute=True):

    if recompute:
        bf_solutions = find_brute_force_solution(stages)
        with open(bf_solution_filename,"w") as f:
            json.dump(bf_solutions,f)
    else:
        with open(bf_solution_filename,"r") as f:
            bf_solutions =  json.load(f)

    np.testing.assert_almost_equal(sol["cost"],bf_solutions["cost"])

    match_found = False
    for bf_sol in bf_solutions["optimal_solutions"]:
        if solution_policies_equal(sol["policy"],bf_sol,stages):
            match_found = True

    assert match_found


def assert_solution_policies_equal(sol1, sol2, stages) -> None:
    for stage_id in stages:
        assert stage_id in sol1
        assert stage_id in sol2
        assert sol1[stage_id]["s"] == sol2[stage_id]["s"],\
            "stage `{}` has different s values: {} {}".\
            format(stage_id,sol1[stage_id]["s"],sol2[stage_id]["s"])

        assert sol1[stage_id]["si"] == sol2[stage_id]["si"],\
            "stage `{}` has different si values: {} {}".\
                format(stage_id,sol1[stage_id]["si"],sol2[stage_id]["si"])


def solution_policies_equal(sol1, sol2, stages, verbose=False) -> bool:
    for stage_id in stages:
        if sol1[stage_id]["s"] != sol2[stage_id]["s"]:
            if verbose:
                print("stage `{}` has different s values: {} {}".\
                      format(stage_id,sol1[stage_id]["s"],sol2[stage_id]["s"]))
            return False

        if sol1[stage_id]["si"] != sol2[stage_id]["si"]:
            if verbose:
                print("stage `{}` has different si values: {} {}".\
                      format(stage_id,sol1[stage_id]["si"],sol2[stage_id]["si"]))
            return False

    return True


def check_labeling_invariance(gsm_obj: tree_gsm.GuaranteedServiceModel,
                              selected_roots: Optional[List[str]]=None,
                              additional_constraints_dict: Optional[GSM_Constraints]=None) -> None:
    solutions = []
    if selected_roots is None:
        selected_roots = list(gsm_obj.stages.keys())

    assert selected_roots
    for stage_id in selected_roots:
        sol = gsm_obj.find_optimal_solution(root_stage_id=stage_id,
                                            constraints=additional_constraints_dict)

        tree_gsm.verify_solution_policy(sol.policy, gsm_obj.stages)
        assert list(sol.policy.keys())[0] == stage_id
        solution_cost = tree_gsm.compute_total_inventory_cost(sol.policy, gsm_obj.stages)
        np.testing.assert_approx_equal(solution_cost,sol.cost)

        solutions.append(sol)

    assert solutions
    sol_ref = solutions[0]
    assert (sol_ref.policy.keys() == gsm_obj.stages.keys()), \
        "number of stages in policy is unequal to actual amount"

    for sol in solutions[1:]:
        np.testing.assert_approx_equal(sol_ref.cost,sol.cost)

        assert list(sol_ref.policy.keys()) != list(sol.policy.keys())

        assert (sol_ref.policy.keys() == sol.policy.keys())

        assert_solution_policies_equal(sol_ref.policy,sol.policy,gsm_obj.stages)


def check_additional_constraints(gsm_obj,
                                 constraints_list,
                                 unconstrained_solution=None,
                                 monotone_increase=False) -> tree_gsm.GSM_Solution:

    if unconstrained_solution is None:
        unconstrained_solution = gsm_obj.find_optimal_solution()

    prev_solution = unconstrained_solution

    new_constraints = {}
    for c,v in constraints_list:
        new_constraints[c] = v
        check_labeling_invariance(gsm_obj=gsm_obj,
                                  additional_constraints_dict=new_constraints)

        constrained_solution = gsm_obj.find_optimal_solution(constraints=new_constraints)

        assert constrained_solution.cost > unconstrained_solution.cost

        if monotone_increase:
            assert constrained_solution.cost > prev_solution.cost
        else:
            assert constrained_solution.cost >= prev_solution.cost

        for (stage_id,constr),val in new_constraints.items():
            min_or_max,s_or_si = constr.split("_")
            assert min_or_max in ["min","max"]
            assert s_or_si in ["s","si"]

            if min_or_max == "min":
                assert constrained_solution.policy[stage_id][s_or_si] >= val
            else:
                assert constrained_solution.policy[stage_id][s_or_si] <= val

        prev_solution = constrained_solution

    return prev_solution


def traverse_backtrack_cache(gsm_object):
    for k,stage_id in gsm_object.labels_stages.items():
        print()
        print(k,stage_id)
        for i,j in gsm_object._backtrack_cache[stage_id]["dependencies"].items():
            print(i,j)
        for i,j in gsm_object._backtrack_cache[stage_id]["s_si"].items():
            print(i,j)
        input()
