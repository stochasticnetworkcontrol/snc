import os
from collections import namedtuple
from typing import Dict

import numpy as np
import pytest
from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree

from src import meio as dag_gsm
import src.meio.gsm.tree_gsm as tree_gsm
from src.meio import (solution_policies_equal, setup_cyclic_network,
                      setup_skip_network, setup_coc_network,
                      setup_diamond_network, setup_dist_network,
                      setup_ext_diamond_network,
                      setup_diamond_network_extra_edge)
from src.meio import GuaranteedServiceModelDAG
from src.meio.gsm.tree_gsm import GuaranteedServiceModelTree, Stage
from src.meio import read_supply_chain_from_txt


dirname = os.path.dirname(__file__)


def test_cyclic_handling():
    stages = setup_cyclic_network()
    with pytest.raises(tree_gsm.IncompatibleGraphTopology):
        # Should fail because of cycles
        GuaranteedServiceModelDAG(stages)


def test_diamond_handling():
    stages = setup_diamond_network()
    gsm = GuaranteedServiceModelDAG(stages)


def test_diamond_demand_bounds():
    stages = setup_diamond_network()
    gsm = GuaranteedServiceModelDAG(stages)

    for stage_id,stage in gsm.stages.items():
        assert len(stage.demand_stages_phis) == 1
        assert "D" in stage.demand_stages_phis,stage.demand_stages_phis
        if stage_id == "A":
            assert stage.demand_stages_phis["D"] == 2
        else:
            assert stage.demand_stages_phis["D"] == 1

    stage_tau_val_list = [("D",10,25.20),
                          ("C",11,27.45),
                          ("B",12,29.69),
                          ("A",13,63.86)
    ]
    for stage_id,tau,d_bound in stage_tau_val_list:
        np.testing.assert_approx_equal(gsm.stages[stage_id].demand_bound_func(tau),d_bound,4)


def test_ext_diamond_demand_bounds():
    """
    The true reference demand bound values in this and all other "test_..._demand_bounds" tests
    were computed by looking at the topology and using the pooling formula from
    Appendix 2.4. Demand Propagation in Humair and Willems 2011 "Optimizing inventory in DAGs"
    and the demand_bound_function for demand stages as in original tree GSM paper (Grave and Willems 2000)
    """
    stages = setup_ext_diamond_network()
    gsm = GuaranteedServiceModelDAG(stages)

    for stage_id,stage in gsm.stages.items():

        if stage_id == "A":
            assert set(stage.demand_stages_phis.keys()) == set(["D","E"])
            assert stage.demand_stages_phis["D"] == 2
            assert stage.demand_stages_phis["E"] == 1
        elif stage_id in ["B","C","D"]:
            assert len(stage.demand_stages_phis.keys()) == 1
            assert stage.demand_stages_phis["D"] == 1
        else:
            assert stage_id == "E"
            assert len(stage.demand_stages_phis.keys()) == 1
            assert stage.demand_stages_phis["E"] == 1

    stage_tau_val_list = [("D",10,25.20),
                          ("C",11,27.45),
                          ("B",12,29.69),
                          ("A",13,213.94)
    ]
    for stage_id,tau,d_bound in stage_tau_val_list:
        np.testing.assert_approx_equal(gsm.stages[stage_id].demand_bound_func(tau),d_bound,4)


def test_skip_handling():
    stages = setup_skip_network()
    gsm = GuaranteedServiceModelDAG(stages)


def test_skip_demand_bounds():
    stages = setup_skip_network()
    gsm = GuaranteedServiceModelDAG(stages)

    for stage_id,stage in gsm.stages.items():
        assert len(stage.demand_stages_phis) == 1
        assert "C" in stage.demand_stages_phis
        if stage_id == "A":
            assert stage.demand_stages_phis["C"] == 2
        else:
            assert stage.demand_stages_phis["C"] == 1


    stage_tau_val_list = [("C",5,86.78),
                          ("B",11,164.56),
                          ("A",7,227.04)
    ]
    for stage_id,tau,d_bound in stage_tau_val_list:
        np.testing.assert_approx_equal(gsm.stages[stage_id].demand_bound_func(tau),d_bound,4)


def test_dist_demand_bounds():
    stages = setup_dist_network()
    gsm = GuaranteedServiceModelDAG(stages)

    for stage_id,stage in gsm.stages.items():
        assert all(v == 1 for v in stage.demand_stages_phis.values())
        if stage_id == "A":
            assert set(stage.demand_stages_phis.keys()) == set(["F","G","E"])
        elif stage_id in ["B","D"]:
            assert set(stage.demand_stages_phis.keys()) == set(["F","G"])
        elif stage_id in ["C","E"]:
            assert set(stage.demand_stages_phis.keys()) == set(["E"])
        else:
            assert len(stage.demand_stages_phis) ==1
            assert stage_id in stage.demand_stages_phis



    stage_tau_val_list = [("B",10,634.68),
                          ("C",11,274.55),
                          ("A",13,1088.31)
    ]
    for stage_id,tau,d_bound in stage_tau_val_list:
        np.testing.assert_approx_equal(gsm.stages[stage_id].demand_bound_func(tau),d_bound,4)



def test_coc_handling():
    stages = setup_coc_network()
    gsm = GuaranteedServiceModelDAG(stages)
    for stage_id,stage in gsm.stages.items():
        if stage_id in ["A","B"]:
            assert set(stage.demand_stages_phis.keys()) == set(["C","D"])
        else:
            assert len(stage.demand_stages_phis) == 1
            assert stage_id in stage.demand_stages_phis


def test_spanning_tree_basic_diamond():
    stages = setup_diamond_network()
    # It seems to be non-deterministic
    # This causes infinite loop in the recursion in ordered_stages
    num_random_restarts = 100
    for i in range(num_random_restarts):
        spanning_tree_stages,_ = GuaranteedServiceModelDAG._find_spanning_tree(stages)
        assert ((spanning_tree_stages['C'].down_stages == {'D': 1} and
                 spanning_tree_stages['B'].down_stages == {}) or
                (spanning_tree_stages['B'].down_stages == {'D': 1} and
                 spanning_tree_stages['C'].down_stages == {})), 'B->C or C->D removed'


def test_spanning_tree_skip():
    stages = setup_skip_network()
    gsm = GuaranteedServiceModelDAG(stages)
    St = namedtuple('St', ['id', 'up', 'down'])
    l = ((St('A', [], ['B','C']), St('B', ['A'], []), St('C', ['A'], [])),  # no B->C
         (St('A', [], ['C']), St('C', ['A','B'], []), St('B', [], ['C'])),  # no A->B
         (St('A', [], ['B']), St('B', ['A'], ['C']), St('C', ['B'],[]))   # no A->C
         )
    # try all possible permutations for spanning tree
    optimal_cost = None
    for spanning_tree in l:
        for s in spanning_tree:
            gsm.tree_stages[s.id].up_stages = {u: 1 for u in s.up }
            gsm.tree_stages[s.id].down_stages = {d: 1 for d in s.down}
        print(gsm.tree_stages)
        gsm.tree_gsm = GuaranteedServiceModelTree(gsm.tree_stages, initialise_bounds=False)
        soln = gsm.find_optimal_solution()
        assert optimal_cost is None or np.allclose(soln.cost, optimal_cost)
        optimal_cost = soln.cost


def test_spanning_tree_basic_scenario():
    stages = setup_diamond_network()
    gsm = GuaranteedServiceModelDAG(stages)
    St = namedtuple('St', ['id', 'up', 'down'])
    l = (((St('A', [], ['B', 'C']),
           St('B', ['A'], []),
           St('C', ['A'], ['D']),
           St('D', ['C'], [])),set([("B","D")])),  # no B->D
         ((St('A', [], ['B', 'C']),
           St('C', ['A'], []),
           St('B', ['A'], ['D']),
           St('D', ['B'], [])),set([("C","D")])),  # no C->D
         ((St('A', [], ['C']),
          St('C', ['A'], ['D']),
          St('B', [], ['D']),
           St('D', ['B', 'C'], [])),set([("A","B")])),  # no A->B
         ((St('A', [], ['B']),
           St('B', ['A'], ['D']),
           St('C', [], ['D']),
           St('D', ['B', 'C'], [])),set([("A","C")]))   # no A->C
         )
    # try all possible permutations for spanning tree
    optimal_cost = None
    for spanning_tree,removed_links in l:
        for s in spanning_tree:
            gsm.tree_stages[s.id].up_stages = {u: 1 for u in s.up }
            gsm.tree_stages[s.id].down_stages = {d: 1 for d in s.down}
            gsm._ordered_removed_links = gsm._order_removed_links(removed_links)
        print(gsm.tree_stages)
        gsm.tree_gsm = GuaranteedServiceModelTree(gsm.tree_stages, initialise_bounds=False)
        soln = gsm.find_optimal_solution()
        assert optimal_cost is None or np.allclose(soln.cost, optimal_cost)
        optimal_cost = soln.cost


def calc_edge_cost(stages: Dict[str, Stage]):
    cost = 0
    for stage_id,stage in stages.items():
        for down_stage_id in stage.down_stages:
            down_stage = stages[down_stage_id]
            cost += GuaranteedServiceModelDAG._get_edge_cost(stage, down_stage)
    return cost


def test_compute_spanning_tree_cost():
    stages = setup_diamond_network_extra_edge()
    with pytest.raises(tree_gsm.IncompatibleGraphTopology):
        # Should fail because diamond is not a tree
        GuaranteedServiceModelDAG._compute_spanning_tree_cost(stages)

    gsm = GuaranteedServiceModelDAG(stages)
    MST,_ = gsm._find_MST(stages)
    MST_cost = GuaranteedServiceModelDAG._compute_spanning_tree_cost(MST)
    assert MST_cost == calc_edge_cost(MST)

def check_mst_vs_scipy(stages_full: Dict[str, Stage], mst: Dict[str, Stage]) -> None:
    """
    compare our MST solution to scipy baseline
    :param stages_full: full network
    :param mst: mst solution
    """
    assign_ids = np.array([stage_id for stage_id in sorted(stages_full.keys())])
    N = assign_ids.size
    # create cost array
    cost = np.zeros((len(stages_full), len(stages_full)))
    for stage_id, stage in stages_full.items():
        for down_stage_id in stage.down_stages:
            down_stage = stages_full[down_stage_id]
            i = np.flatnonzero(assign_ids == stage_id)
            j = np.flatnonzero(assign_ids == down_stage_id)
            cost[j, i] = cost[i, j] = GuaranteedServiceModelDAG._get_edge_cost(stage, down_stage)
    C = minimum_spanning_tree(cost).toarray()
    assert C.sum() == calc_edge_cost(mst), 'different costs for MST and scipy solvers'
    for i, c in enumerate(C):
        for d in range(N):
            if assign_ids[d] in mst[assign_ids[i]].down_stages.keys():
                assert C[i, d] != 0, 'scipy said there is no edge'
            else:
                assert C[i, d] == 0, 'scipy said there must be an edge'


def test_MST_spanning_diamond():
    stages = setup_diamond_network_extra_edge()
    gsm = GuaranteedServiceModelDAG(stages)
    MST,_ = gsm._find_MST(stages)
    MST_cost = GuaranteedServiceModelDAG._compute_spanning_tree_cost(MST)
    assert MST_cost == calc_edge_cost(MST)
    assert MST_cost == -83.895

    St = namedtuple('St', ['id', 'up', 'down'])
    l = ((St('A', [], ['B', 'C']), St('B', ['A'], []), St('C', ['A'], ['D']), St('D', ['C'], [])),  # no B->D,A->D
         (St('A', [], ['C', 'D']), St('C', ['A'], []), St('B', [], ['D']), St('D', ['B','A'], [])),  # no C->D,A->B
         (St('A', [], ['D']), St('C', [], ['D']), St('B', [], ['D']), St('D', ['B', 'C', 'A'], [])),  # no A->B,A->C
         (St('A', [], ['B']), St('B', ['A'], ['D']), St('C', [], ['D']), St('D', ['B', 'C'], [])),  # no A->C,A->D
         (St('A', [], ['C']), St('B', [], ['D']), St('C', ['A'], ['D']), St('D', ['B', 'C'], []))   # no A->D,A->B
         )
    # try some permutations for spanning tree
    for i,spanning_tree in enumerate(l):
        for s in spanning_tree:
            gsm.tree_stages[s.id].up_stages = {u: 1 for u in s.up }
            gsm.tree_stages[s.id].down_stages = {d: 1 for d in s.down}
        assert GuaranteedServiceModelDAG._compute_spanning_tree_cost(gsm.tree_stages) >= MST_cost

    check_mst_vs_scipy(stages, MST)

@pytest.mark.parametrize("supply_chain_filename",
                         [
                             "bulldozer.txt",
                             "chemicals.txt",
                             "semiconductors.txt",
])
def test_find_spanning_tree(supply_chain_filename):
    supply_chain_filename = os.path.join(dirname, supply_chain_filename)
    stages = read_supply_chain_from_txt(supply_chain_filename)

    gsm = dag_gsm.GuaranteedServiceModelDAG(stages)

@pytest.mark.parametrize("supply_chain_filename",
                         [
                             "bulldozer.txt",
                             "chemicals.txt",
                             "semiconductors.txt",
])
def test_find_minimum_spanning_tree(supply_chain_filename):
    supply_chain_filename = os.path.join(dirname, supply_chain_filename)
    stages = read_supply_chain_from_txt(supply_chain_filename)

    gsm = dag_gsm.GuaranteedServiceModelDAG(stages)
    MST,_ = gsm._find_MST(stages)


def test_against_tree_GSM():
    supply_chain_filename = os.path.join(dirname, "bulldozer.txt")
    stages = read_supply_chain_from_txt(supply_chain_filename)

    gsm_1 = tree_gsm.GuaranteedServiceModelTree(stages)
    sol_1 = gsm_1.find_optimal_solution()

    gsm_2 = dag_gsm.GuaranteedServiceModelDAG(stages)
    sol_2 = gsm_2.find_optimal_solution()

    np.testing.assert_approx_equal(sol_1.cost,sol_2.cost)
    assert solution_policies_equal(sol_1.policy,sol_2.policy,stages,verbose=False)


@pytest.mark.parametrize("supply_chain_filename, root_ids_list", [
    ("toys.txt", ["Part_0003","Manuf_0003","Dist_0011"]),
    ("food.txt", ["Part_0002","Part_0007","Retail_0003","Manuf_0005"]),
])
def test_root_invariance(supply_chain_filename,root_ids_list):
    supply_chain_filename = os.path.join(dirname, supply_chain_filename)
    stages = read_supply_chain_from_txt(supply_chain_filename)
    gsm_1 = dag_gsm.GuaranteedServiceModelDAG(stages)
    sol_1 = gsm_1.find_optimal_solution(root_stage_id=root_ids_list[0])

    for root_stage_id in root_ids_list[1:]:
        gsm_2 = dag_gsm.GuaranteedServiceModelDAG(stages)
        sol_2 = gsm_2.find_optimal_solution(root_stage_id=root_stage_id)
        np.testing.assert_approx_equal(sol_1.cost,sol_2.cost)
        assert solution_policies_equal(sol_1.policy,sol_2.policy,stages,verbose=False)


@pytest.mark.parametrize("supply_chain_name, ref_cost, ref_stock",
                         [
                             ("bulldozer.txt", 630000,None),
                             ("chemicals.txt",33500,946),
                             ("semiconductors.txt",9510000,None),
                             ("toys.txt",49000,None),
                             ("food.txt",2010000,630507),
                             ("cutlery.txt",77800,805)
])
def test_dag_gsm(supply_chain_name, ref_cost, ref_stock):
    supply_chain_filename = os.path.join(dirname, supply_chain_name)
    stages = read_supply_chain_from_txt(supply_chain_filename)

    print(supply_chain_name)
    gsm = dag_gsm.GuaranteedServiceModelDAG(stages)
    solution = gsm.find_optimal_solution()

    base_stocks = tree_gsm.compute_base_stocks(solution.policy, stages)
    print(sum(i for i in base_stocks.values()),ref_stock)
    print(solution.cost,ref_cost,ref_cost/solution.cost)
    print()


@pytest.mark.parametrize("supply_chain_name",
                         [
                             "bulldozer.txt",
                             "chemicals.txt",
                             "semiconductors.txt",
                             "toys.txt",
                             "food.txt",
                             "cutlery.txt",
])
def test_init_with_basic_solution(supply_chain_name):
    supply_chain_filename = os.path.join(dirname, supply_chain_name)
    stages = read_supply_chain_from_txt(supply_chain_filename)

    gsm = dag_gsm.GuaranteedServiceModelDAG(stages)
    sol = gsm.find_optimal_solution()
    sol_init = gsm.find_optimal_solution_with_basic_solution_init()
    np.testing.assert_approx_equal(sol.cost,sol_init.cost)
    # TODO: We could also assert number of iterations using
    # find_optimal_solution_with_basic_solution_init is smaller or equal to
    # find_optimal_solution.
    # Measuring runtimes as we were doing is unreliable and causes the test
    # to fail sporadically.


def test_complexity_bound():
    links_ranges = {
        ("A","B"):(0,6),
        ("C","B"):(0,5),
        ("C","D"):(0,5)
    }
    stages_log2_c = dag_gsm.GuaranteedServiceModelDAG._compute_log2_complexities(links_ranges)
    assert len(stages_log2_c) == 2
    assert "B" in stages_log2_c
    assert "C" in stages_log2_c
    assert 2**(stages_log2_c["B"]) == 33
    assert 2**(stages_log2_c["C"]) == 13
