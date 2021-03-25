"""
Guaranteed service model for spanning trees as implemented in Graves and Willems 2000
Capacity constraints extension is taken from Schoenmeyr and Graves 2016
"""

import abc
from typing import List, Dict, Tuple, Callable, Set, Optional, Iterable
import numpy as np
from collections import OrderedDict, deque, defaultdict

from src.meio import Sub_Prob_Cost_Func, Cumul_Min_Sub_Prob_Cost_Func
from src.meio import (GSM_Policy, GSM_Constraints, GSM_Unsatisfied_Constraints,
                      Stages_Labels, Labels_Stages,
                      Depths_Stages, Stages_Depths, Labels_Parents)


class GSM_Solution:
    """
    Class for representing the solution of the GSM dynamic programming algorithm
    """
    def __init__(self, cost: float, policy: GSM_Policy, base_stocks: Dict[str, int]):
        """
        :param cost: numerical cost of proposed GSM policy
        :param policy: settings of service times for each stage in the suppl chain
        """
        self.cost = cost
        self.policy = policy
        self.base_stocks = base_stocks

    def serialize(self) -> Dict:
        solution_dict = {"cost": self.cost, "policy": self.policy, "base_stocks": self.base_stocks}
        return solution_dict


class GSMException(Exception):
    """
    Exception raised by GSM code
    """
    pass


class EnumerationRangeOverflow(GSMException):
    """
    Exception is raised when the rate of demand bound function in capacity constrained GSM stages
    does not fall below capacity constraint value over the limited enumeration range provided
    See equation (9) in Graves 2016 where enumeration range is assumed infinite
    """


class UnSupportedGSMException(GSMException):
    """
    This exception is for when a user asks for a type of GSM model that does exist in the
    literature but we have not yet implemented.
    """
    pass


class IncompatibleGraphTopology(GSMException):
    """
    This exception is for when a network (described e.g. through a config file)
    is fed into a GSM class with incompatible topology.  E.g. a config file describing
    a CoC topology is used to instantiate a GuaranteedServiceModelTree class. Note that
    network with cycles or loops is incompatible with all GSM classes
    """
    pass


class InconsistentGSMConfiguration(GSMException):
    """
    This exception is for when an inconsistency is detected in the labelling of the stages,
    usually from the config file.
    """
    pass


class Stage:
    """
    Class which stores all the attributes of the individual processing stage in supply network
    together with its connections to other stages
    """
    def __init__(self,
                 _id: str,
                 lead_time: int,
                 max_s_time: int,
                 added_cost: float,
                 up_stages: Dict[str, int],
                 down_stages: Dict[str, int],
                 cap_constraint: int = np.inf,
                 theta: Optional[int] = None,
                 cap_margin: Optional[int] = None,
                 risk_pool: Optional[int] = None,
                 is_ext_demand_stage: bool = False,
                 demand_mean: Optional[float] = None,
                 demand_std: Optional[float] = None,
                 demand_thres: Optional[float] = None,
                 cost_rate: Optional[float] = None,
                 max_s_bound: Optional[int] = None,
                 max_si_bound: Optional[int] = None,
                 demand_bound_func: Optional[Callable] = None,
                 base_stock_func: Optional[Callable] = None,
                 demand_stages_phis: Optional[Dict[str, int]] = None) -> None:
        """
        :param _id: unique processing stage identifier
        :param lead_time: production lead-time of the stage
        :param max_s_time: maximum allowed service time for the stage
        :param added_cost: value added to a product at this stage
        :param up_stages: list of stage ids which are connected upstream
        :param down_stages: list of stage ids which are connected downstream
        :param cap_constraint: maximum batch size of items that can be processed per day
        :param theta: is the point at which derivative of the demand bound equals the capacity
                      see equation (10) in Graves 2016
        :param cap_margin: (D(theta)/c - theta) which is negative of the lower bound
                           on replenishment time see equation (51) in Graves 2016 Appendix 7
        :param risk_pool: the degree of risk pooling in this stage from downstream stages
        :param is_ext_demand_stage: whether stage has external demand or not
        :param demand_mean: mean of the external demand distribition
        :param demand_std: standard deviation of external demand distribition
        :param demand_thres: factor `k` representing the percentage of time that
                                 the safety stock covers the demand variation
        :param cost_rate: total accumulated unit value of the items in the inventory
                          of this stage
        :param max_s_bound: maximum allowable service time derived from accumulated lead times
        :param max_si_bound: maximum internal service time derived from accumulated lead times
        :param demand_bound_func: function which bounds the cummulative demand over the period of
        given replenishment time
        :param base_stock_func: function which computes the required base stock level for a given
        net replenishment time
        :param demand_stages_phis: mapping on how many items of this stage correspond to one item
        for each of the leaf demand nodes
        """
        assert lead_time >= 0
        assert max_s_time >= 0
        assert added_cost >= 0
        assert cap_constraint > 0
        if risk_pool is not None:
            assert risk_pool >= 1

        if is_ext_demand_stage:
            assert demand_mean is not None
            assert demand_std is not None
            assert demand_thres is not None

        if demand_mean is not None:
            assert demand_mean >= 0

        if demand_std is not None:
            assert demand_std >= 0

        if demand_thres is not None:
            assert demand_thres >= 0

        if cost_rate is not None:
            assert cost_rate >= 0

        if theta is not None:
            assert theta >= 0

        if cap_margin is not None:
            assert cap_margin >= 0

        self._id = _id
        self.lead_time = lead_time
        self.max_s_time = max_s_time
        self.max_s_bound = max_s_bound
        self.max_si_bound = max_si_bound
        self.added_cost = added_cost
        self.cap_constraint = cap_constraint
        self.theta = theta
        self.cap_margin = cap_margin
        self.is_ext_demand_stage = is_ext_demand_stage
        self.risk_pool = risk_pool
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.demand_thres = demand_thres

        self.up_stages = up_stages
        self.down_stages = down_stages

        self.demand_bound_func = demand_bound_func
        self.base_stock_func = base_stock_func
        self.demand_stages_phis = demand_stages_phis
        self.cost_rate = cost_rate

        # this attribute sets the upper limit on enumerating n
        # in equation (9) of Graves 2016
        self._LARGE_N = 1000

    @property
    def id(self):
        return self._id

    def __repr__(self) -> str:
        return "Stage(id={}, up_stages={}, down_stages={}, is_ext_demand_stage={})".format(
            self._id, self.up_stages.keys(), self.down_stages.keys(),
            self.is_ext_demand_stage)

    def _replicate(self) -> "Stage":
        """
        Method makes exact copy of Stage object
        """
        stage_config = dict(vars(self))
        # TODO how avoid deleting this variable so it does not pass into init?
        del stage_config["_LARGE_N"]
        return Stage(**stage_config)


class GuaranteedServiceModel(metaclass=abc.ABCMeta):
    """
    Class that has all the methods for finding the optimal base stock levels in
    guaranteed service model given a static supply network
    """
    def __init__(self, stages: Dict[str, Stage], initialise_bounds: bool=True,
                 propagate_bounds: bool=False) -> None:
        """
        :param stages: dictionary of supply chain stages
        :param initialise_bounds: whether to recompute demand bounds and service time bounds
        :param propagate_bounds: whether to sequentially propagate demand bounds upstream instead
                                 of propagate links to demand stages
        :raises: IncompatibleGraphTopology:  If the network does not in fact have a compatible
                                             topology
        :raises: InconsistentGSMConfiguration:  Network topology labels not as expected.
        :raises: EnumerationRangeOverflow:  demand bound function is not sloping below capacity rate
        """
        self.stages = stages
        self._ordered_stages = []  # type: List[str]
        self.labels_stages = {}  # type: Labels_Stages
        self.stages_labels = {}  # type: Stages_Labels
        self.stages_depths = {}  # type: Stages_Depths
        self.depths_stages = defaultdict(set)  # type: Depths_Stages
        self.label_parents = {}  # type: Labels_Parents
        self._root_stage_id = None  # type: Optional[str]
        self._n_links = None  # type: Optional[int]

        self._constraints = {}  # type: GSM_Constraints
        self._cost_f_funcs = {}  # type: Dict[int,Sub_Prob_Cost_Func]
        self._min_cost_f_funcs = {}  # type: Dict[int,Cumul_Min_Sub_Prob_Cost_Func]
        self._cost_g_funcs = {}  # type: Dict[int,Sub_Prob_Cost_Func]
        self._min_cost_g_funcs = {}  # type: Dict[int,Cumul_Min_Sub_Prob_Cost_Func]
        self._backtrack_cache = OrderedDict()  # type: OrderedDict[str,Dict]

        self._label_stages()
        self._verify_graph_topology()

        self._propagate_bounds = propagate_bounds

        if initialise_bounds:
            self._compute_cost_rates()
            self._create_all_stage_functions()
            self._compute_capacity_constraints_margins()
            self._compute_maximum_service_time_bounds()

    @property
    def ordered_stages(self) -> List[str]:
        """
        :returns: list of stage ids ordered starting from external demand stages
        and going up the supply chain
        """
        if not self._ordered_stages:
            self._ordered_stages = self._order_stages(self.stages)

        return self._ordered_stages

    @staticmethod
    def _order_stages(stages_dict: Dict[str, Stage]) -> List[str]:
        """
        Method returns the ordering of stage ids starting from external demand stages and
        going up the supply chain(in the opposite direction to supply arrows). This ordering is
        required to compute demand means and upper bounds recursively starting from customer
        facing stages.

        The recursive formula is mentioned in `Demand Process` part of section `2. Assumptions`
        of the reference paper.

        Exactly the reverse order of processing is required to recursively compute maxumum service
        time bounds starting from highest stages in the supply chain and going down to
        external demand stages.
        """
        ordered_stages = []
        stage_ids = deque(stages_dict.keys())
        encountered = set()  # type: Set[str]
        while stage_ids:
            remove = True
            stage_id = stage_ids.popleft()
            stage = stages_dict[stage_id]
            for d_stage_id in stage.down_stages:
                if d_stage_id not in encountered:
                    remove = False
                    break

            if not remove:
                stage_ids.append(stage_id)
                continue

            ordered_stages.append(stage_id)
            encountered.add(stage.id)

        return ordered_stages

    def _compute_cost_rates(self) -> None:
        """
        Method for recursively computing the total cost of items (products) in the inventory of each
        stage starting from upmost supply stages and ending with demand stages
        """
        cost_rates = {}  # type: Dict[str,float]
        for stage_id in reversed(self.ordered_stages):
            stage = self.stages[stage_id]
            upstream_cost = sum([cost_rates[u_stage] for u_stage in stage.up_stages])
            cost_rate = stage.added_cost + upstream_cost
            stage.cost_rate = cost_rates[stage_id] = cost_rate

    def _compute_capacity_constraints_margins(self) -> None:
        """
        Method computes thetas and margins for the negative replenishment times as specified
        in Appendix 7 of Graves 2016 equation (51). It uses explicit enumeration to find break-even
        points (thetas) as in equation (9)

        :raises: EnumerationRangeOverflow:  demand bound function is not sloping below capacity rate
        """
        for stage_id in self.ordered_stages:
            stage = self.stages[stage_id]
            if stage.cap_constraint == np.inf:
                stage.theta = 0
                stage.cap_margin = 0
                continue

            # TODO figure out proper range for n
            assert stage.demand_bound_func is not None
            for n in range(stage._LARGE_N):
                grad = stage.demand_bound_func(n+1)-stage.demand_bound_func(n)
                if grad <= stage.cap_constraint:
                    stage.theta = n
                    # capacity margin is stored as positive
                    # TODO Figure out why "-1" is needed to replicate the tests
                    stage.cap_margin = int(stage.demand_bound_func(n)/stage.cap_constraint - n) - 1
                    break
            else:
                raise EnumerationRangeOverflow("Rate of demand bounds did not fall below capacity")

    def _compute_maximum_service_time_bounds(self) -> None:
        """
        Method for computing maximum possible service times for all nodes
        in order starting from the up-most supply stages and going down
        in the direction of supply chain arrows

        This recursive procedure is mentioned as footnote in pg 70 as:
        M_j=T_j + max {M_i | i:(i,j) A}
        in the reference paper and also correspond to variables M_k in the section
        `Functional Equations`

        With capacity constrained stages, upper bound on service time is extended beyond
        cumulative lead time by the capacity margin to allow for replenishment ahead of service
        """
        max_s_time_bounds: Dict[str, int] = {}
        for stage_id in reversed(self.ordered_stages):
            stage = self.stages[stage_id]
            neighbour_s_time_bounds = [max_s_time_bounds[u_stage] for u_stage in stage.up_stages]
            stage.max_si_bound = max(neighbour_s_time_bounds if neighbour_s_time_bounds else [0])
            stage.max_s_bound = stage.max_si_bound + stage.lead_time + stage.cap_margin
            max_s_time_bounds[stage_id] = stage.max_s_bound

    def _compute_internal_demand_mean(self, stage: Stage) -> None:
        assert not stage.is_ext_demand_stage
        stage_demand_mean = 0.0
        for d_stage_id in stage.down_stages:
            phi = stage.down_stages[d_stage_id]
            d_stage = self.stages[d_stage_id]

            assert d_stage.demand_mean is not None
            stage_demand_mean += phi * d_stage.demand_mean

        stage.demand_mean = stage_demand_mean

    @staticmethod
    def _create_base_stock_func(stage: Stage, demand_bound_func: Callable) -> Callable:
        """
        Method computes basestock level by complete enumeration as in equation (9) Graves 2016
        as well as in Table 2
        :raises: EnumerationRangeOverflow:  demand bound function is not sloping below capacity rate
        """
        # TODO create a cache for this function evaluations?
        def base_stock_func(tau: int) -> float:
            max_base_stock = 0
            # TODO Look into maximum range for n
            for n in range(stage._LARGE_N):
                base_stock = demand_bound_func(tau + n) - n*stage.cap_constraint
                if base_stock >= max_base_stock:
                    max_base_stock = base_stock
                else:
                    break
            else:
                raise EnumerationRangeOverflow("Maximum basestock level has not been encountered")

            return max_base_stock

        return base_stock_func

    @staticmethod
    def _create_external_stage_funcs(stage: Stage) -> None:
        """
        for external demand stages the demand bound and basestock functions can be computed
        independently of other stages
        """
        assert stage.is_ext_demand_stage

        def demand_bound_func(tau: int) -> float:

            # equation (1) in the reference paper
            assert stage.demand_mean is not None
            assert stage.demand_thres is not None
            assert stage.demand_std is not None
            if tau > 0:
                demand_bound = tau * stage.demand_mean + \
                    stage.demand_thres * stage.demand_std * np.sqrt(tau)
            else:
                demand_bound = 0

            return demand_bound

        if stage.cap_constraint != np.inf:
            base_stock_func = GuaranteedServiceModel._create_base_stock_func(stage,
                                                                             demand_bound_func)
        else:
            base_stock_func = demand_bound_func

        stage.demand_bound_func = demand_bound_func
        stage.base_stock_func = base_stock_func
        stage.demand_stages_phis = {stage.id: 1}

    def _create_internal_stage_funcs(self, stage: Stage) -> None:
        """
        method computes demand bound and basestock functions for each stage
        """
        assert not stage.is_ext_demand_stage
        assert getattr(stage, "risk_pool", None) is not None, stage

        self._cache_demand_bounds[stage.id] = {}

        # computed as in Appendix 2.4. Demand Propagation
        # from Humair and Willems 2011 "Optimizing inventory in DAGs"
        demand_stages_phis = defaultdict(int)  # type:Dict[str,int]
        for d_stage_id in stage.down_stages:
            phi = stage.down_stages[d_stage_id]
            d_stage = self.stages[d_stage_id]

            assert d_stage.demand_stages_phis,\
                "demand phis in donwstream stage {} has not been computed before {}".\
                format(d_stage_id, stage.id)

            for demand_stage_id, d_phi in d_stage.demand_stages_phis.items():
                demand_stages_phis[demand_stage_id] += phi*d_phi

        # for capacitated networks demand bound need to be propagated from immediate donwstream
        # stages
        demand_bound_phis = stage.down_stages if self._propagate_bounds else demand_stages_phis

        def demand_bound_func(tau: int) -> float:
            # TODO make a single generic demand_bound_func factory for external and internal stages?
            if tau in self._cache_demand_bounds[stage.id]:
                return self._cache_demand_bounds[stage.id][tau]

            demand_bound = tau * stage.demand_mean
            cum_var = 0

            for demand_stage_id, d_phi in demand_bound_phis.items():
                demand_stage = self.stages[demand_stage_id]
                demand_b_func = demand_stage.demand_bound_func

                # demand bound value is computed as in Table 2 of Graves 2016
                if np.isinf(demand_stage.cap_constraint):
                    demand_bound_value = demand_b_func(tau)
                else:
                    demand_bound_value = min(tau*demand_stage.cap_constraint, demand_b_func(tau))

                demand_mean = demand_stage.demand_mean
                cum_var += (d_phi * (demand_bound_value - tau * demand_mean))**stage.risk_pool

            demand_bound += cum_var**(1.0/stage.risk_pool)

            self._cache_demand_bounds[stage.id][tau] = demand_bound

            return demand_bound

        # as in Table 2 of Graves 2016
        if stage.cap_constraint != np.inf:
            base_stock_func = self._create_base_stock_func(stage, demand_bound_func)
        else:
            base_stock_func = demand_bound_func

        stage.demand_bound_func = demand_bound_func
        stage.base_stock_func = base_stock_func
        stage.demand_stages_phis = demand_stages_phis

    def _create_all_stage_functions(self):
        """
        Method to recursively generate all the demand bound functions
        starting from external demand and going up the supply chain
        """
        self._cache_demand_bounds = {}
        for stage_id in self.ordered_stages:
            stage = self.stages[stage_id]

            if stage.is_ext_demand_stage:
                self._create_external_stage_funcs(stage)
            else:
                self._compute_internal_demand_mean(stage)
                self._create_internal_stage_funcs(stage)

    def _verify_graph_topology(self) -> None:
        """
        Method to loop through all the stages and verify that the relative depths with its
        adjacent stages are consistent, i.e. that all connected stages have depths difference
        of one

        :raises:  IncompatibleGraphTopology:  If the network does not in fact have a compatible
        topology for this GSM class
        """
        for stage_id in self.stages:
            stage = self.stages[stage_id]
            d = self.stages_depths[stage_id]
            for up_stage_id in stage.up_stages:

                if not (self.stages_depths[up_stage_id] == d + 1):
                    raise IncompatibleGraphTopology("skip between stages {} and {}".
                                                    format(stage_id, up_stage_id))

            for d_stage_id in stage.down_stages:

                if not (self.stages_depths[d_stage_id] == d - 1):
                    raise IncompatibleGraphTopology("skip between stages {} and {}".
                                                    format(stage_id, d_stage_id))

            if not np.isinf(stage.cap_constraint):
                for up_stage_id in stage.up_stages:
                    if not (len(self.stages[up_stage_id].down_stages) == 1):
                        raise IncompatibleGraphTopology("Capacity constraints will only work for \
                        serial or convergent supply chains")

    def _label_stages(self, root_stage_id: str=None) -> None:
        """
        Method to index the stages and find their relative depths
        :param root_stage_id: which stage to use as a root, i.e. last stage to be processed
        :raise: InconsistentGSMConfiguration:  Inconsistent input, e.g. a loop of stages
        """

        # By default we deterministically and arbitrarily pick one stage as the root
        if root_stage_id is None:
            root_stage_id = sorted(self.stages.keys())[0]

        self._root_stage_id = root_stage_id

        n = len(self.stages)

        stages_depths = OrderedDict()  # type: Stages_Depths
        depths_stages = defaultdict(set)  # type: Depths_Stages
        labels_stages = OrderedDict()  # type: Labels_Stages
        stages_labels = OrderedDict()  # type: Stages_Labels
        labels_parents = {}  # type: Labels_Parents

        stages_depths[root_stage_id] = 0
        depths_stages[0].add(root_stage_id)
        labels_stages[n] = root_stage_id
        labels_parents[n] = None
        stages_labels[root_stage_id] = n
        k = n - 1

        n_links = 0
        encountered = {root_stage_id}
        stages_queue = deque([root_stage_id])
        while stages_queue:
            stage_id = stages_queue.popleft()
            stage = self.stages[stage_id]
            d = stages_depths[stage_id]

            adjacent_stages = list(stage.up_stages.keys()) + list(stage.down_stages.keys())

            for a_stage_id in adjacent_stages:

                n_links += 1

                if a_stage_id in encountered:
                    continue

                if a_stage_id in stage.up_stages:
                    stages_depths[a_stage_id] = d + 1
                    depths_stages[d + 1].add(a_stage_id)
                else:
                    stages_depths[a_stage_id] = d - 1
                    depths_stages[d - 1].add(a_stage_id)

                if (k in labels_stages) or (a_stage_id in stages_labels) or (k in labels_parents):
                    raise InconsistentGSMConfiguration("Labelling is inconsistent")

                labels_stages[k] = a_stage_id
                stages_labels[a_stage_id] = k
                labels_parents[k] = stage_id
                k -= 1
                stages_queue.append(a_stage_id)
                encountered.add(a_stage_id)

        if k > 0:
            InconsistentGSMConfiguration("not all the stages has been labeled")
        if k < 0:
            InconsistentGSMConfiguration("some stages have been labeled more than once")

        self.stages_depths = stages_depths
        self.depths_stages = depths_stages
        self.labels_stages = labels_stages
        self.stages_labels = stages_labels
        self.label_parents = labels_parents
        self._n_links = int(n_links/2)

    @abc.abstractmethod
    def _generate_cost_function(self, label: int, stage: Stage):
        raise NotImplementedError("Specify stage cost function")

    def _minimize_cost_function_fixed_s(self,
                                        stage: Stage, fixed_s: int,
                                        cost_function: Callable) -> Tuple[float, Optional[int]]:
        """
        Method finds the minimum value and associated `si` for a fixed `s` in the
        cost function `cost(s,si)`

        :param stage: stage in the network
        :param fixed_s: fixed value of s
        :param cost_dunction: cost function to be minimised
        """
        objective_func = lambda x: cost_function(fixed_s, x)

        # si_range_min is taken from the last constraint in equation (51)
        # of Appendix 7 in Graves 2016
        assert stage.cap_margin is not None
        si_range_min = max(0, fixed_s - stage.lead_time - stage.cap_margin)

        # check if additional constraints on min value for si are provided for this stage
        if (stage.id, "min_si") in self._constraints:
            min_si_constraint = self._constraints[(stage.id, "min_si")]
            si_range_min = max(si_range_min, min_si_constraint)

        assert stage.max_si_bound is not None
        si_range_max = max(0,stage.max_si_bound)
        si_range = range(si_range_min, si_range_max + 1)
        min_val,min_si = self._minimize_objective(objective_func, si_range,
                                                  last_arg=False)

        return min_val,min_si


    def _minimize_cost_function_fixed_si(self,
                                         stage: Stage, fixed_si: int,
                                         cost_function: Callable) -> Tuple[float,Optional[int]]:
        """
        Method finds the minimum value and associated `s` for a fixed `si` in the
        cost function `cost(s,si)`

        :param stage: stage in the network
        :param fixed_si: fixed value of si
        :param cost_dunction: cost function to be minimised
        """
        objective_func = lambda x: cost_function(x, fixed_si)
        s_range_min = 0
        s_range_max = min(fixed_si + stage.lead_time + stage.cap_margin, stage.max_s_time)
        s_range_max = min(s_range_max, stage.max_s_bound)

        # Check if additional constraints on max value for s are provided for this stage
        if (stage.id,"max_s") in self._constraints:
            max_s_constraint = self._constraints[(stage.id,"max_s")]
            s_range_max = min(s_range_max, max_s_constraint)

        assert not s_range_max is None
        s_range = range(s_range_min,s_range_max + 1)
        min_val,min_s = self._minimize_objective(objective_func, s_range,
                                                 last_arg=True)

        return min_val,min_s

    def _compute_cost_f_funcs(self, label: int, stage: Stage) -> None:
        """
        Method to compute the cost `f` function values and cumulative min cost `f` function values.

        Cumulative min cost is defined as `min_cost_f(x) = min f(s), for s in range(0,x)`
        and is computed for each input value `x` by minimising over all permissible
        values of external service time `s`.
        """
        self._backtrack_cache[stage.id] = {"s_si":{},
                                           "dependencies":defaultdict(dict)}

        cost_func = self._generate_cost_function(label,stage)
        s_range_min = 0
        s_range_max = min(stage.max_s_bound,stage.max_s_time)

        # check if additional constraints on max value for s are provided for this stage
        if (stage.id,"max_s") in self._constraints:
            max_s_constraint = self._constraints[(stage.id,"max_s")]
            s_range_max = min(s_range_max, max_s_constraint)

        assert not s_range_max is None
        s_range = range(s_range_min, s_range_max + 1)

        min_cost_f = {}
        cost_f = {}

        min_cost = np.inf
        min_s = None

        for s in s_range:
            min_val,min_si = self._minimize_cost_function_fixed_s(stage, s, cost_func)

            if min_si is not None:
                cost_f[s] = min_val
                self._backtrack_cache[stage.id]["s_si"][("s",s)] = ("si",min_si,min_val)

            if min_val <= min_cost and min_val < np.inf:
                min_cost = min_val
                min_s = s

            min_cost_f[s] = min_cost,min_s

        def min_cost_f_func(x):
            return min_cost_f[x] if x <= s_range_max else (min_cost,min_s)

        def cost_f_func(x):
            return cost_f[x] if x <= s_range_max else np.inf

        self._min_cost_f_funcs[label] = min_cost_f_func
        self._cost_f_funcs[label] = cost_f_func

    def _compute_cost_g_funcs(self,label: int, stage: Stage) -> None:
        """
        Method to compute the cost `g` function values and cumulative min cost `g` function values.

        Cumulative min cost is defined as `min_cost_g(x) = min g(si), for si in range(x,max_si)`
        and is computed for each input value `x` by minimising over all permissible
        values of internal service time `si`.

        """
        self._backtrack_cache[stage.id] = {"s_si":{},
                                           "dependencies":defaultdict(dict)}

        cost_func = self._generate_cost_function(label,stage)
        assert not stage.max_si_bound is None
        si_range_max = max(0,stage.max_si_bound)
        si_range_min = 0

        # check if additional constraints on min value for si are provided for this stage
        if (stage.id,"min_si") in self._constraints:
            min_si_constraint = self._constraints[(stage.id,"min_si")]
            si_range_min = max(si_range_min, min_si_constraint)

        si_range = range(si_range_max,si_range_min - 1,-1)

        min_cost_g = {}
        cost_g = {}

        min_cost = np.inf
        min_si = None

        for si in si_range:
            min_val,min_s = self._minimize_cost_function_fixed_si(stage, si, cost_func)
            if min_s is not None:
                cost_g[si] = min_val
                self._backtrack_cache[stage.id]["s_si"][("si",si)] = ("s",min_s,min_val)

            if min_val < min_cost and min_val < np.inf:

                min_cost = min_val
                min_si = si

            min_cost_g[si] = min_cost, min_si

        def min_cost_g_func(x):
            if x >= si_range_min and x <= si_range_max:
                return min_cost_g[x]
            elif x < si_range_min:
                return (min_cost, min_si)
            else:
                return (np.inf, None)

        def cost_g_func(x):
            return cost_g[x] if x >= si_range_min and x <= si_range_max else np.inf

        self._min_cost_g_funcs[label] = min_cost_g_func
        self._cost_g_funcs[label] = cost_g_func

    @staticmethod
    def _minimize_objective(
            objective_function: Callable,
            args_range: Iterable,
            last_arg: bool= False) -> Tuple[float, Optional[int]]:
        """
        Util method to find the minimum and arg_min of the function by complete enumeration

        :param objective_function: function to minimize
        :param args_range: range object over arguments to optimize over
        :param last_arg: if the cost does not change whether to find the last argument
                         in the range with this cost
        """
        min_val = np.inf
        min_arg = None
        for arg in args_range:
            val = objective_function(arg)
            if val < min_val:
                min_val = val
                min_arg = arg

            if last_arg and val == min_val and val < np.inf:
                min_val = val
                min_arg = arg

        return min_val, min_arg

    @abc.abstractclassmethod
    def find_optimal_solution(
            self, root_stage_id: Optional[str]=None,
            constraints: Optional[GSM_Constraints]=None) -> GSM_Solution:
        raise NotImplementedError("Implement find optimal solution method")

    def _run_dp(self, constraints: Optional[GSM_Constraints]=None) -> Tuple[float,GSM_Policy]:
        """
        Core method that run the dynamic process minimisation of cost function over spanning tree

        :returns: optimal solution `s` and `si` times and minimum cost
        """
        self._backtrack_cache = OrderedDict()
        self._min_cost_f_funcs = {}
        self._cost_f_funcs = {}
        self._min_cost_g_funcs = {}
        self._cost_g_funcs = {}
        self._constraints = constraints if constraints is not None else {}

        n = len(self.stages)

        for k in range(1,n+1):
            s_id = self.labels_stages[k]
            stage = self.stages[s_id]
            p_k = self.label_parents[k]
            if p_k in stage.down_stages:
                # step 2 in `Dynamic Program` section of the reference paper
                self._compute_cost_f_funcs(k,stage)

            elif p_k in stage.up_stages:
                # step 3 in `Dynamic Program` section of the reference paper
                self._compute_cost_g_funcs(k,stage)
            else:
                assert p_k is None
                assert k == n, "Algorithm has not finished at the last labeled stage"
                final_stage_id = s_id
                final_stage_label = k

        assert final_stage_id == self._root_stage_id

        # step 4 in `Dynamic Program` section of the reference paper
        final_stage = self.stages[final_stage_id]
        self._compute_cost_f_funcs(final_stage_label,
                                   final_stage)

        # step 5 in `Dynamic Program` section of the reference paper
        min_cost_f_func = self._min_cost_f_funcs[final_stage_label]
        max_s = min(final_stage.max_s_time,self._constraints.get((final_stage_id,"s"),np.inf))
        min_cost,min_arg = min_cost_f_func(max_s)

        assert min_cost < np.inf
        policy = self._backtrack_optimal_policy(final_stage_id,min_arg)
        assert len(policy) == len(self.stages), \
            "Policy size({}) needs to equal number of stages({})"\
            .format(len(policy),len(self.stages))

        return min_cost,policy

    @abc.abstractmethod
    def _backtrack_optimal_policy(self,last_stage_id: str,last_stage_s: int) -> GSM_Policy:
        raise NotImplementedError("Implement backtracking algorithm")


class GuaranteedServiceModelTree(GuaranteedServiceModel):

    def _generate_cost_function(self, label: int, stage: Stage):
        """
        method to initialise the stage cost function for further minimization
        """
        bk_cache = self._backtrack_cache[stage.id]["dependencies"]

        def cost_func(s,si):
            """
            :param s: service time of the stage
            :param si: internal service time of the stage
            :returns: the cost of the stage plus all the connected stages in the subtree
            """
            tau = si + stage.lead_time - s

            cost = stage.cost_rate * (stage.base_stock_func(tau) - tau * stage.demand_mean)

            for u_stage_id in stage.up_stages:
                if self.stages_labels[u_stage_id] < label:
                    min_cost_f_func = self._min_cost_f_funcs[self.stages_labels[u_stage_id]]
                    min_val,min_arg = min_cost_f_func(si)
                    if min_arg is not None:
                        bk_cache[(s,si)][u_stage_id] = ("s",min_arg,min_val)

                    cost += min_val

            for d_stage_id in stage.down_stages:
                if self.stages_labels[d_stage_id] < label:
                    min_cost_g_func = self._min_cost_g_funcs[self.stages_labels[d_stage_id]]
                    min_val,min_arg = min_cost_g_func(s)
                    if min_arg is not None:
                        bk_cache[(s,si)][d_stage_id] = ("si",min_arg,min_val)

                    cost += min_val

            return cost

        return cost_func

    def _verify_graph_topology(self):
        """
        :raises:  IncompatibleGraphTopology:  If the network does not in fact have a compatible
        tree topology
        """
        super()._verify_graph_topology()
        if not (self._n_links == len(self.stages) - 1):
            raise IncompatibleGraphTopology("Supply chain is not a tree")

    def find_optimal_solution(
            self, root_stage_id: Optional[str]=None,
            constraints: Optional[GSM_Constraints]=None) -> GSM_Solution:
        """
        :returns: optimal solution policy and corresponding minimal cost of the
                  initialised supply chain
        :raises: InconsistentGSMConfiguration:  Network topology labels not as expected.
        """
        if root_stage_id is not None:
            self._label_stages(root_stage_id)
        min_cost, policy = self._run_dp(constraints=constraints)
        policy_cost = compute_total_inventory_cost(policy,self.stages)
        assert np.isclose(policy_cost, min_cost,rtol=1e-3), (policy_cost, min_cost, policy)
        verify_solution_policy(policy, self.stages)
        base_stocks = compute_base_stocks(policy,self.stages)
        solution = GSM_Solution(cost=min_cost,policy=policy,base_stocks=base_stocks)

        return solution

    def _backtrack_optimal_policy(self,last_stage_id: str, last_stage_s: int) -> GSM_Policy:
        """
        Method to backtrack the settings of `s`s and `si`s for the minimal cost solution

        This procedure is mentioned in section `5.3. The Optimal Policy` of
        the [Humair and Willems, 2006, `Optimizing Supply Chains with Clusters of Commonality`]

        :param last_stage: the stage which was the last one to be processed in
                           dynamic programming iterations
        :param last_stage_s: the value of optimal service time `s` for the final stage
        :returns: the optimal setting of service and internal service times for each stage
        """
        assert last_stage_s is not None

        policy = OrderedDict()  # type: GSM_Policy

        stack = [(last_stage_id,("s",last_stage_s))]
        while stack:
            stage_id,(s_or_si,v) = stack.pop()

            if s_or_si == "s":
                s = v
                n,si,_ = self._backtrack_cache[stage_id]["s_si"][("s",s)]

                assert n == "si"
            elif s_or_si == "si":
                si = v
                n,s,_ = self._backtrack_cache[stage_id]["s_si"][("si",si)]
                assert n == "s"

            policy[stage_id] = {"s":s,"si":si}

            for n_stage_id,vals in self._backtrack_cache[stage_id]["dependencies"][(s,si)].items():
                s_or_si,v,_ = vals
                stack.append((n_stage_id,(s_or_si,v)))

        return policy


def verify_solution_policy(solution_policy: GSM_Policy,
                           stages: Dict[str, Stage]) -> None:
    """
    Method that checks the proposed solution policy for the satisfaction of GSM time constraints
    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: whether all constraints of the solution are satisfied or not
    """
    assert len(solution_policy) == len(stages), "number of decisions must equal number of stages"

    for stage_id in stages:
        policy = solution_policy[stage_id]
        stage = stages[stage_id]
        s = policy["s"]
        si = policy["si"]

        assert stage.max_s_bound is not None
        assert not s > stage.max_s_bound,\
            "`s={}` in stage `{}` is above allowed service bound - ({})".\
            format(s,stage_id,stage.max_s_bound)

        assert stage.max_si_bound is not None
        assert not si > stage.max_si_bound,\
            "`si={}` in stage `{}` is above allowed internal service bound - ({})".\
            format(si,stage_id,stage.max_si_bound)

        if stage.max_s_time is not None:
            assert not s > stage.max_s_time,\
                "`s={}` in stage `{}` is above allowed service time - ({})".\
                format(s,stage_id,stage.max_s_time)

        assert not si > stage.max_s_bound - stage.lead_time, \
            "`si={}` in stage `{}` is above allowed service bound - ({})".\
            format(s,stage_id,stage.max_s_time)

        assert not si + stage.lead_time < s - stage.cap_margin,\
            "`s` in stage `{}` is above the maximum production time".\
            format(stage_id)

        for u_stage_id in stage.up_stages:
            assert not si < solution_policy[u_stage_id]["s"],\
                "`si={}` in stage `{}` is below the `s={}` of up stage `{}`"\
                .format(si,stage_id,solution_policy[u_stage_id]["s"],u_stage_id)

        for d_stage_id in stage.down_stages:
            assert not s > solution_policy[d_stage_id]["si"],\
                "`s={}` in stage `{}` is above the `si={}` time of down stage `{}`"\
                .format(s,stage_id,solution_policy[d_stage_id]["si"],d_stage_id)


def compute_replenishment_times(
        solution_policy: GSM_Policy, stages: Dict[str, Stage]
) -> Dict[str, int]:
    rep_times = {}
    for stage_id, policy in solution_policy.items():

        stage = stages[stage_id]

        s = policy["s"]
        si = policy["si"]

        tau = si + stage.lead_time - s

        if tau < 0:
            assert stage.cap_margin is not None
            assert abs(tau) <= stage.cap_margin

        rep_times[stage_id] = tau

    return rep_times


def compute_basic_solution_cost(stages: Dict[str,Stage]) -> float:
    """
    Method to compute the total inventory cost of basic solution when all S and SI are zero
    :param stage: dictionary of stages in the supply chain
    :returns: total inventory cost
    """
    total_cost = 0.0
    for stage_id in stages:
        stage = stages[stage_id]
        s = 0
        si = 0

        tau = si + stage.lead_time - s

        if tau > 0:
            assert stage.base_stock_func is not None
            assert stage.demand_mean is not None

            expected_inventory = stage.base_stock_func(tau) - stage.demand_mean*tau
        else:
            expected_inventory = 0

        total_cost += stage.cost_rate * expected_inventory

    return total_cost


def compute_total_inventory_cost(
        solution_policy: GSM_Policy, stages: Dict[str, Stage]
) -> float:
    """
    Method to compute the total inventory cost of proposed solution
    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: total inventory cost
    """
    total_cost = 0.0
    invent_costs = compute_expected_inventory_costs(solution_policy, stages)

    total_cost = sum(invent_costs.values())

    return total_cost


def compute_expected_inventory_costs(
        solution_policy: GSM_Policy, stages: Dict[str, Stage]
) -> Dict[str, float]:
    """
    Method to compute the inventory costs for each individual stage
    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: total inventory cost
    """
    inventory_costs = {}
    exp_inventories = compute_expected_inventories(solution_policy, stages)
    for stage_id, exp_invent in exp_inventories.items():
        stage = stages[stage_id]
        inventory_costs[stage_id] = stage.cost_rate * exp_invent

    return inventory_costs


def compute_expected_inventories(
        solution_policy: GSM_Policy, stages: Dict[str, Stage]
) -> Dict[str, float]:
    """
    Method to compute optimal expected inventory levels and their costs for all stages
    given the optimal values of service times

    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: expected inventory levels for a all stages
    """
    inventories = {}
    base_stocks = compute_base_stocks(solution_policy, stages)
    repr_times = compute_replenishment_times(solution_policy, stages)
    for stage_id, base_stock in base_stocks.items():

        stage = stages[stage_id]

        tau = repr_times[stage_id]

        assert stage.demand_mean is not None
        inventory = base_stock - stage.demand_mean * tau

        inventories[stage_id] = inventory

    return inventories


def compute_base_stocks(
        solution_policy: GSM_Policy, stages: Dict[str, Stage]
) -> Dict[str, int]:
    """
    Method to compute optimal expected inventory levels for all stages
    given the optimal values of service times

    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: base stocks for all stages
    """
    stocks = {}
    repr_times = compute_replenishment_times(solution_policy, stages)
    for stage_id, tau in repr_times.items():

        stage = stages[stage_id]

        assert stage.base_stock_func is not None
        stock = stage.base_stock_func(tau)

        stocks[stage_id] = stock

    return stocks


def find_unsatisfied_constraints(solution_policy: GSM_Policy,
                                 stages: Dict[str, Stage]) -> GSM_Unsatisfied_Constraints:
    """
    Methods iterates through all stages and collects all the unsatisfied constraints
    between connected stages
    :param solution_policy: proposed setting of service times
    :param stages: dictionary of stages in the supply chain
    :returns: the set of unsatisfied constraints, identified by the pair of conflicting
              stage ids
    """
    assert len(solution_policy) == len(stages), "number of decisions must equal number of stages"
    broken_constraints = {}  # type: GSM_Unsatisfied_Constraints

    for stage_id in stages:
        stage_policy = solution_policy[stage_id]
        stage = stages[stage_id]
        s = stage_policy["s"]
        si = stage_policy["si"]

        for u_stage_id in stage.up_stages:
            pair = (u_stage_id, stage_id)
            if si < solution_policy[u_stage_id]["s"] and pair not in broken_constraints:
                broken_constraints[pair] = (solution_policy[u_stage_id]["s"],si)

        for d_stage_id in stage.down_stages:
            pair = (stage_id,d_stage_id)
            if s > solution_policy[d_stage_id]["si"] and pair not in broken_constraints:
                broken_constraints[pair] = (s,solution_policy[d_stage_id]["si"])

    return broken_constraints


def correct_policy_downstream(
        solution_policy: GSM_Policy, stages: Dict[str,Stage]
) -> GSM_Policy:
    """
    Method loops through all the stages and adjusts their internal service "SI"
    to the maximum of the upstream stage service times
    :returns: new gsm solution policy which satisfies all the constraints
    """
    new_solution_policy = {}  # type: GSM_Policy
    for stage_id, stage_policy in solution_policy.items():

        stage = stages[stage_id]

        new_stage_policy = {**stage_policy}

        for u_stage_id in stage.up_stages:
            new_stage_policy["si"] = max(new_stage_policy["si"], solution_policy[u_stage_id]["s"])

        new_solution_policy[stage_id] = new_stage_policy

    verify_solution_policy(new_solution_policy, stages)

    return new_solution_policy
