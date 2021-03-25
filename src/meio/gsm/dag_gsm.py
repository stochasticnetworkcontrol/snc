"""
Guaranteed service model for supply chains with directed acyclic graph structure as described in:
Optimising Strategic Safety Stock Placement in General Acyclic Networks, Humair and Willems, 2011.
"""
from collections import OrderedDict, defaultdict
from typing import Any
from typing import List, Dict, Tuple, Set, Optional

import numpy as np

from meio.gsm.tree_gsm import (GuaranteedServiceModel, GuaranteedServiceModelTree,
                               Stage, GSM_Solution, IncompatibleGraphTopology)
from meio.gsm.tree_gsm import (compute_total_inventory_cost, correct_policy_downstream,
                               verify_solution_policy, find_unsatisfied_constraints,
                               compute_basic_solution_cost, compute_base_stocks)
from meio.gsm.types import GSM_Constraints, GSM_Policy


class GuaranteedServiceModelDAG(GuaranteedServiceModel):
    """
    Class that runs a variant of branch-and-bound algoritm on top of tree_GSM optimisation to find
    the optimal service rates which satisfy all the conflicting constraints of DAG topology
    """
    def __init__(self, stages: Dict[str, Stage], use_random_ST: bool=False) -> None:
        """
        :param stages: dictionary of supply chain stages

        :raises: IncompatibleGraphTopology:  If the network does not in fact have a compatible
                                             topology
        :raises: InconsistentGSMConfiguration:  Network topology labels not as expected.
        """
        super().__init__(stages)
        if use_random_ST:
            self._use_random_ST = True
            self.tree_stages,removed_links = GuaranteedServiceModelDAG._find_spanning_tree(
                self.stages)
        else:
            self._use_random_ST = False
            self.tree_stages,removed_links = GuaranteedServiceModelDAG._find_MST(self.stages)

        self._ordered_removed_links = self._order_removed_links(removed_links)

        self.tree_gsm = GuaranteedServiceModelTree(self.tree_stages, initialise_bounds=False)

    def _generate_cost_function(self, label: int, stage: Stage):
        pass

    def _backtrack_optimal_policy(self, last_stage_id: str, last_stage_s: int):
        pass

    def _verify_graph_topology(self) -> None:
        """
        Method to verify that the provided supply network does not contain cycles.
        Method recursively removes stages from the network which do not have any upstream stages
        and if at some iteration no new such stages appear in the network, this means it contains
        at least one cycle

        :raises:  IncompatibleGraphTopology:  If the network does not in fact have a compatible
        topology for this GSM class
        """

        stages_parents = {s_id:set(self.stages[s_id].up_stages.keys()) for s_id in self.stages}
        Q = [s_id for s_id in self.stages if not self.stages[s_id].up_stages]

        if not Q:
            raise IncompatibleGraphTopology("Supply chain has no stages without upstream stages")

        remaining_stages = set(self.stages.keys())
        while True:
            new_Q = []
            for s_id in Q:
                for d_s_id in self.stages[s_id].down_stages:
                    stages_parents[d_s_id].remove(s_id)
                    if not stages_parents[d_s_id]:
                        new_Q.append(d_s_id)
                remaining_stages.remove(s_id)

            if remaining_stages:
                if not new_Q:
                    raise IncompatibleGraphTopology("there is a cycle between stages {}".\
                                               format(remaining_stages))
            else:
                if new_Q:
                    raise IncompatibleGraphTopology("network is not a DAG")
                break

            Q = new_Q

    @staticmethod
    def _get_edge_cost(up_stage: Stage,down_stage: Stage) -> float:
        """
        Function takes the two connected stages and computes the heuristic
        edge weight as described in the appendix of Humair and Willems 2011
        section '5.1 Selecting a spanning tree'
        """
        assert up_stage.id in down_stage.up_stages
        assert down_stage.id in up_stage.down_stages

        assert not up_stage.cost_rate is None
        assert not up_stage.demand_bound_func is None
        assert not up_stage.demand_mean is None

        assert not down_stage.cost_rate is None
        assert not down_stage.demand_bound_func is None
        assert not down_stage.demand_mean is None

        edge_k_times_sigma = down_stage.demand_bound_func(tau=1) - down_stage.demand_mean
        edge_cost_rate = up_stage.cost_rate + down_stage.cost_rate
        edge_cost = -edge_k_times_sigma*edge_cost_rate
        assert edge_cost <= 0

        return edge_cost

    def _get_link_cost(self, link: Tuple[str,str]) -> float:
        """
        Function is the wrapper around '_get_edge_cost' when
        graph edge id is passed as input
        """
        up_stage_id,down_stage_id = link
        up_stage = self.stages[up_stage_id]
        down_stage = self.stages[down_stage_id]
        return GuaranteedServiceModelDAG._get_edge_cost(up_stage,down_stage)

    def _order_removed_links(self,removed_links: Set[Tuple[str,str]]) -> List[Tuple[str,str]]:
        """Method sorts the set of removed links using the heuristic weighting scheme"""
        return list(sorted(removed_links,key=self._get_link_cost))

    @staticmethod
    def _find_MST(stages: Dict[str,Stage]) -> Tuple[Dict[str,Stage],Set[Tuple[str,str]]]:
        """
        Method replicates the stages but removes some connections
        to ensure that the resulting graph is a minimum spanning tree
        with edge weights as defined in the appendix of Humair and Willems 2011
        section '5.1 Selecting a spanning tree'
        """
        #compute and sort edge costs for each individual stage
        stages_edges = {}
        min_edge_cost = 0.
        min_edge_cost_stage_id = None
        for stage_id,stage in stages.items():
            edges = []
            for down_stage_id in stage.down_stages:
                down_stage = stages[down_stage_id]

                edge_cost = GuaranteedServiceModelDAG._get_edge_cost(stage,down_stage)

                if edge_cost < min_edge_cost:
                    min_edge_cost = edge_cost
                    min_edge_cost_stage_id = stage_id

                edges.append(("down",down_stage_id,edge_cost))

            for up_stage_id in stage.up_stages:
                up_stage = stages[up_stage_id]

                edge_cost = GuaranteedServiceModelDAG._get_edge_cost(up_stage,stage)

                if edge_cost < min_edge_cost:
                    min_edge_cost = edge_cost
                    min_edge_cost_stage_id = up_stage_id

                edges.append(("up",up_stage_id,edge_cost))

            #sort stage edges in the descending order of negative costs
            stages_edges[stage_id] = sorted(edges,key=lambda x:-x[2])

        assert not min_edge_cost_stage_id is None

        #run Prim's algorithm to find MST
        added_links = set()
        visited_set = set([min_edge_cost_stage_id])

        current_stage_id = min_edge_cost_stage_id
        min_cut_edge = stages_edges[min_edge_cost_stage_id].pop()
        while True:
            up_down,new_stage_id,edge_cost = min_cut_edge
            visited_set.add(new_stage_id)

            assert up_down in ["up","down"]

            if up_down == "up":
                link = (new_stage_id,current_stage_id)
            else:
                link = (current_stage_id,new_stage_id)

            added_links.add(link)

            if len(visited_set) == len(stages):
                break

            next_min_cut_edge_cost = 0.
            next_min_cut_stage_id = None
            for stage_id in visited_set:

                edges = stages_edges[stage_id]

                while edges:
                    _,stage_min_cost_edge_neighbour_id,stage_min_edge_cost = edges[-1]

                    if stage_min_cost_edge_neighbour_id in visited_set:
                        edges.pop()
                        continue

                    if stage_min_edge_cost <= next_min_cut_edge_cost:
                        next_min_cut_edge_cost = stage_min_edge_cost
                        next_min_cut_stage_id = stage_id

                    break

            if next_min_cut_stage_id is None:
                print(len(visited_set),len(stages))

            assert not next_min_cut_stage_id is None
            min_cut_edge = stages_edges[next_min_cut_stage_id].pop()
            current_stage_id = next_min_cut_stage_id

        #initialise new tree like stages graph
        tree_stages = {}  # type: Dict[str,Stage]
        removed_links = set()
        for stage_id in stages:
            stage = stages[stage_id]
            tree_stage = stage._replicate()
            new_u_stages = {}  # type: Dict[str,int]
            for u_stage_id in stage.up_stages:
                link = (u_stage_id,stage_id)
                if not link in added_links:
                    removed_links.add(link)
                    continue

                new_u_stages[u_stage_id] = stage.up_stages[u_stage_id]

            tree_stage.up_stages = new_u_stages

            new_d_stages = {}  # type: Dict[str,int]
            for d_stage_id in stage.down_stages:
                link = (stage_id,d_stage_id)
                if not link in added_links:
                    removed_links.add(link)
                    continue

                new_d_stages[d_stage_id] = stage.down_stages[d_stage_id]

            tree_stage.down_stages = new_d_stages

            tree_stages[stage_id] = tree_stage

        return tree_stages,removed_links

    @staticmethod
    def _compute_spanning_tree_cost(stages: Dict[str,Stage]) -> float:
        GuaranteedServiceModelTree(stages,initialise_bounds=False)
        tree_cost = 0.
        for _,stage in stages.items():
            for u_stage_id in stage.up_stages:
                u_stage = stages[u_stage_id]
                tree_cost += GuaranteedServiceModelDAG._get_edge_cost(u_stage,stage)

        return tree_cost

    def _get_links_enumeration_ranges(self,additional_constraints: GSM_Constraints
                                     ) -> Dict[Tuple[str,str],int]:
        """
        Used in old complexity bound function

        Function loops through the ordered list of removed links, computes
        the complete enumeration range of S and SI values along this link,
        and shortens it if additional constraints on S and SI values have been
        passed.
        """
        links_ranges = OrderedDict() # type: Dict[Tuple[str,str],int]
        for link in self._ordered_removed_links:
            up_stage_id,down_stage_id = link
            down_stage = self.stages[down_stage_id]
            r_start = 0
            assert not down_stage.max_s_bound is None
            r_end = down_stage.max_s_bound - down_stage.lead_time + 1

            if (up_stage_id,"max_s") in additional_constraints:
                r_end = min(r_end,additional_constraints[(up_stage_id,"max_s")])

            if (down_stage_id,"min_si") in additional_constraints:
                r_start = max(r_start,additional_constraints[(down_stage_id,"min_si")])

            links_ranges[link] = max(1,r_end - r_start)

        return links_ranges

    def _get_link_enumeration_range(self,link: Tuple[str,str],
                                    additional_constraints: GSM_Constraints) -> Tuple[int,int]:
        """
        Function gets the complete enumeration range start and end values
        along this link, and shortens it if additional constraints on S and SI
        values have been passed.
        """
        up_stage_id,down_stage_id = link
        up_stage = self.stages[up_stage_id]
        assert not up_stage.max_s_bound is None
        r_start = 0
        r_end = up_stage.max_s_bound

        if (up_stage_id,"max_s") in additional_constraints:
            r_end = min(r_end,additional_constraints[(up_stage_id,"max_s")])

        if (down_stage_id,"min_si") in additional_constraints:
            r_start = max(r_start,additional_constraints[(down_stage_id,"min_si")])

        return r_start,r_end

    def _collect_removed_links_enumeration_ranges(self,additional_constraints: GSM_Constraints
                                                 ) -> Dict[Tuple[str,str],Tuple[int,int]]:
        """
        Method loops through the list of removed links and gets their enumeration range
        end points
        """
        links_ranges = OrderedDict() # type: Dict[Tuple[str,str],Tuple[int,int]]
        for link in self._ordered_removed_links:
            links_ranges[link] = self._get_link_enumeration_range(link,additional_constraints)

        return links_ranges

    @staticmethod
    def _compute_log2_complexities(links_ranges: Dict[Tuple[str,str],Tuple[int,int]]
                                  ) -> Dict[str, int]:
        """
        Function loops through the provided dict of links and their enumeration ranges,
        and computes the upper bound on enumeration complexity
        """
        # collect links into clusters defined by up and down
        # TODO: can we have meaningful types for these two?
        s_link_clusters = defaultdict(set)  # type: Dict[Any, Any]
        si_link_clusters = defaultdict(set) # type: Dict[Any, Any]
        for link in links_ranges:
            up_stage_id,down_stage_id = link
            s_link_clusters[up_stage_id].add(link)
            si_link_clusters[down_stage_id].add(link)

        # remove redundant link references
        overlap_s_clusters = set() # type: Set[str]
        for up_stage_id in list(s_link_clusters.keys()):
            links = s_link_clusters[up_stage_id]
            if len(links) == 1:
                del s_link_clusters[up_stage_id]
            else:
                for _,down_stage_id in links:
                    if len(si_link_clusters[down_stage_id]) == 1:
                        del si_link_clusters[down_stage_id]
                    else:
                        overlap_s_clusters.add(up_stage_id)

        # collect complexities of si clusters
        stages_log2_c = OrderedDict() # type: Dict[str, int]
        added_links = set() # type: Set[Tuple[str,str]]
        for down_stage_id,links in si_link_clusters.items():
            r_start = links_ranges[list(links)[0]][0]
            max_r_end_link = max(links,key=lambda link:links_ranges[link][-1])
            max_r_end = links_ranges[max_r_end_link][-1]

            c = 2**(np.ceil(np.log2(max(1,max_r_end-r_start)))+1)
            added_links.add(max_r_end_link)
            links.remove(max_r_end_link)
            if links:
                for x in range(r_start+1,max_r_end+1):
                    n_combs = 1
                    for link in links:
                        link_r_end = links_ranges[link][-1]
                        #TODO How to ensure no multiplication by zero?
                        n_combs *= np.ceil(np.log2(max(1,min(link_r_end,x)-r_start)))+1

                    c += n_combs

            stages_log2_c[down_stage_id] = np.log2(max(1,c))
            added_links.update(links)

        # collect complexities of s clusters
        for up_stage_id,links in s_link_clusters.items():
            r_end = links_ranges[list(links)[0]][-1]

            min_r_start_link = min(links,key=lambda link:links_ranges[link][0])
            min_r_start = links_ranges[min_r_start_link][0]
            c = 2**(np.ceil(np.log2(max(1,r_end - min_r_start)))+1)

            if up_stage_id in overlap_s_clusters:
                overlap_links = [link for link in links if link in added_links]
                min_overlap_r_start_link = min(overlap_links,key=lambda link:links_ranges[link][0])
                min_overlap_r_start = links_ranges[min_overlap_r_start_link][0]
                c -= 2**(np.ceil(np.log2(max(1,r_end - min_overlap_r_start)))+1)
                for link in overlap_links:
                    if link in links:
                        links.remove(link)

            if links:
                for x in range(min_r_start+1,r_end+1):
                    n_combs = 1
                    for link in links:
                        link_r_start = links_ranges[link][0]
                        #TODO How to ensure no multiplication by zero?
                        #TODO Check all '+1'
                        n_combs *= np.ceil(np.log2(max(1,r_end+1-max(link_r_start,x))))+1

                    c += n_combs

            stages_log2_c[up_stage_id] = np.log2(max(1,c))
            added_links.update(links)

        assert len(added_links) == len(links_ranges)

        return stages_log2_c


    def _compute_log2_complexity_bound(self, problem_constraints: GSM_Constraints,
                                       f_tight_version: bool=True) -> float:
        """
        Function gets the enumeration ranges of removed links given the additional
        constraints.

        It then computes the log2 of upper bound on total number of branch and bound
        recursion steps

        For an explanation of this upper bound see handwritten notes at google drive:
           SNC/Handwritten notes/GSM/Dag GSM Complexity/
        """
        if f_tight_version:
            links_ranges = self._collect_removed_links_enumeration_ranges(problem_constraints)
            stage_log2_c = self._compute_log2_complexities(links_ranges)
            return sum(log2_c for log2_c in stage_log2_c.values())
        else:
            links_ranges_loose = self._get_links_enumeration_ranges(problem_constraints)
            log2_complexity_bound = 0
            for _,link_range in links_ranges_loose.items():
                log2_complexity_bound += np.ceil(np.log2(link_range)) + 1

            return log2_complexity_bound

    @staticmethod
    def _find_spanning_tree(stages: Dict[str,Stage]) -> Tuple[Dict[str,Stage],Set[Tuple[str,str]]]:
        """
        Method replicates the stages but removes some connections
        to ensure that the resulting graph is a spanning tree
        """

        tree_stages = {}  # type: Dict[str,Stage]
        init_stage_id = sorted(stages.keys())[0]
        stack = [init_stage_id]  # type: List[str]
        encountered_stages = {init_stage_id}
        encountered_links = set()  # type: Set[Tuple[str,str]]
        removed_links = set()  # type: Set[Tuple[str,str]]

        #find loopy links using DFS
        while stack:
            stage_id = stack.pop()  # type: str
            stage = stages[stage_id]
            neighbours = list(stage.up_stages) + list(stage.down_stages)
            for n_stage_id in neighbours:
                # s = (stage_id, n_stage_id)  # type: Tuple[str,str]
                if n_stage_id in stage.up_stages:
                    link = (n_stage_id,stage_id)
                else:
                    link = (stage_id,n_stage_id)

                if link in encountered_links:
                    continue

                if link in removed_links:
                    continue

                if n_stage_id in encountered_stages:
                    removed_links.add(link)
                    continue

                encountered_links.add(link)
                encountered_stages.add(n_stage_id)
                stack.append(n_stage_id)

        #initialise new tree like stages graph
        for stage_id in stages:
            stage = stages[stage_id]
            tree_stage = stage._replicate()
            new_u_stages = {}  # type: Dict[str,int]
            for u_stage_id in stage.up_stages:
                link = (u_stage_id,stage_id)
                if link in removed_links:
                    continue

                new_u_stages[u_stage_id] = stage.up_stages[u_stage_id]

            tree_stage.up_stages = new_u_stages

            new_d_stages = {}  # type: Dict[str,int]
            for d_stage_id in stage.down_stages:
                link = (stage_id,d_stage_id)

                if link in removed_links:
                    continue

                new_d_stages[d_stage_id] = stage.down_stages[d_stage_id]

            tree_stage.down_stages = new_d_stages

            tree_stages[stage_id] = tree_stage

        return tree_stages,removed_links

    @staticmethod
    def _print_progress(init_log2_complexity_bound: float,
                        completed_problems_count: int,
                        removed_problems_count: float) -> None:


        p_r = np.exp2(np.log2(removed_problems_count) - init_log2_complexity_bound)
        assert p_r <= 1.,(np.log2(removed_problems_count),init_log2_complexity_bound)

        if p_r < 1:
            remaining_problems_log2_count = np.log2(1-p_r) + init_log2_complexity_bound

            p_c = min(1,np.exp2(np.log2(completed_problems_count) - remaining_problems_log2_count))
        else:
            p_c = 1.
        print("\nProblems completed: ",p_c)
        print("Problems removed: ", p_r)


    def _run_branch_and_bound(self, root_stage_id: Optional[str]=None,
                              constraints: Optional[GSM_Constraints]=None,
                              init_with_basic_solution: bool=False) -> GSM_Solution:
        """
        Main method that runs the iterative analogue of branch and bound.

        It corresponds to Routine R of GNA on page 783 in the main reference,
        but instead of using recursive calls to this subroutine it uses stack
        to process spanning sub_problems
        """
        problem_counter = 0

        initial_problem_constraints = {}  # type: GSM_Constraints
        if constraints is not None:
            initial_problem_constraints = constraints

        global_min_cost = np.inf
        global_min_policy = {}  # type: GSM_Policy
        if init_with_basic_solution:
            global_min_cost = compute_basic_solution_cost(self.stages)
            global_min_policy = {stage_id:{"s":0,"si":0} for stage_id in self.stages}

        init_log2_complexity_bound = self._compute_log2_complexity_bound(
            initial_problem_constraints)

        removed_problems = 0

        stack = [(problem_counter,initial_problem_constraints)]
        while stack:
            _,problem_constraints = stack.pop()
            log2_complexity_bound = self._compute_log2_complexity_bound(problem_constraints)
            assert log2_complexity_bound <= init_log2_complexity_bound,log2_complexity_bound

            #step 2 of routine R
            solution = self.tree_gsm.find_optimal_solution(root_stage_id=root_stage_id,
                                                           constraints=problem_constraints)

            #step 3 of routine R
            if np.isinf(solution.cost):
                removed_problems += np.exp2(log2_complexity_bound)
                self._print_progress(init_log2_complexity_bound,
                                     problem_counter+1,
                                     removed_problems)
                continue

            if solution.cost >= global_min_cost:
                removed_problems += np.exp2(log2_complexity_bound)
                self._print_progress(init_log2_complexity_bound,
                                     problem_counter+1,
                                     removed_problems)
                continue

            #step 4 of routine R
            broken_constraints = find_unsatisfied_constraints(solution.policy, self.stages)

            if not broken_constraints:
                global_min_cost = solution.cost
                global_min_policy = solution.policy

                removed_problems += np.exp2(log2_complexity_bound)
                self._print_progress(init_log2_complexity_bound,
                                     problem_counter+1,
                                     removed_problems)

                continue

            #step 5 of routine R
            corrected_solution_policy = correct_policy_downstream(solution.policy,self.stages)
            corrected_solution_cost = compute_total_inventory_cost(corrected_solution_policy,
                                                                   self.stages)

            if corrected_solution_cost < global_min_cost:
                global_min_cost = corrected_solution_cost
                global_min_policy = corrected_solution_policy

            #step 5(a)
            if self._use_random_ST:
                broken_link = list(broken_constraints.keys())[0]
            else:
                broken_link = min(broken_constraints.keys(),key=self._get_link_cost)

            u_stage_id,d_stage_id = broken_link
            s_value,si_value = broken_constraints[broken_link]
            assert s_value > si_value

            new_level = si_value + int((s_value-si_value)/2.)

            #step 5(b)
            lowerbound_problem_constraints = {**problem_constraints}
            for d_stage_id in self.stages[u_stage_id].down_stages:
                lowerbound_problem_constraints[(d_stage_id,"min_si")] = new_level+1 # higher than

            problem_counter += 1
            stack.append((problem_counter,lowerbound_problem_constraints))

            upperbound_problem_constraints = {**problem_constraints}
            upperbound_problem_constraints[(u_stage_id,"max_s")] = new_level # lower than or equal

            problem_counter += 1
            stack.append((problem_counter,upperbound_problem_constraints))

        base_stocks = compute_base_stocks(global_min_policy, self.stages)
        solution = GSM_Solution(cost=global_min_cost,
                                policy=global_min_policy,
                                base_stocks=base_stocks)

        return solution

    def find_optimal_solution(self, root_stage_id: Optional[str]=None,
                              constraints: Optional[GSM_Constraints]=None) -> GSM_Solution:
        solution = self._run_branch_and_bound(root_stage_id=root_stage_id,
                                              constraints=constraints)

        policy_cost = compute_total_inventory_cost(solution.policy,self.stages)
        assert np.isclose(policy_cost,solution.cost)
        verify_solution_policy(solution.policy,self.stages)

        return solution


    def find_optimal_solution_with_basic_solution_init(self,
                                                       root_stage_id: Optional[str]=None,
                                                       constraints: Optional[GSM_Constraints]=None
                                                      ) -> GSM_Solution:
        solution = self._run_branch_and_bound(root_stage_id=root_stage_id,
                                              constraints=constraints,
                                              init_with_basic_solution=True)

        policy_cost = compute_total_inventory_cost(solution.policy,self.stages)
        assert np.isclose(policy_cost,solution.cost)
        verify_solution_policy(solution.policy,self.stages)

        return solution
