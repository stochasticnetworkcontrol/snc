import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union

from src import snc as types
from src.snc import BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams
from src.snc import clean_to_serializable


class BigStepBasePolicy:

    def __init__(self,
                 cost_per_buffer: types.StateSpace,
                 constituency_matrix: types.ConstituencyMatrix,
                 demand_rate: types.StateSpace,
                 buffer_processing_matrix: types.BufferMatrix,
                 workload_mat: types.WorkloadMatrix,
                 nu: types.NuMatrix,
                 list_boundary_constraint_matrices: List[np.ndarray],
                 policy_params: Optional[Union[BigStepLayeredPolicyParams,
                                               BigStepPenaltyPolicyParams]] = None,
                 debug_info: float = False) -> None:
        """
        This is a parent class containing the functionality common to all big step fluid policies.
        It cannot be used for generating the policy itself and child policy classes
        need to implement their custom 'get_policy' method.

        :param cost_per_buffer: from the environment.
        :param constituency_matrix: from the environment.
        :param demand_rate: from the environment
        :param buffer_processing_matrix: from job generator (the environment).
        :param workload_mat: workload matrix.
        :param nu: matrix with number of rows equal to number of workload vectors, and number of
            columns equal to the number of physical resources. Each row indicates for which
            resources the constraint C z <= 1 is active, i.e. it indicates the set of bottlenecks.
        :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
            indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        :return None.
        """
        self.cost_per_buffer = cost_per_buffer
        self.constituency_matrix = constituency_matrix
        self.demand_rate = demand_rate
        self.buffer_processing_matrix = buffer_processing_matrix
        self.workload_mat = workload_mat
        self.nu = nu
        self.list_boundary_constraint_matrices = list_boundary_constraint_matrices

        self.policy_params = policy_params
        self.debug_info = debug_info

        self.num_resources, self.num_activities = constituency_matrix.shape
        self.num_buffers = cost_per_buffer.shape[0]
        self.num_wl_vec = self.workload_mat.shape[0]  # Number of dimensions in workload space.

        self.pushing_buffer_processing_matrix = np.maximum(0, self.buffer_processing_matrix)

        self.load_ph = None  # type: Optional[types.ResourceSpace]
        self.sigma_2_ph = None  # type: Optional[types.ResourceSpace]

    def update_safety_stock_params(self, load_ph: types.ResourceSpace,
                                   sigma_2_ph: types.ResourceSpace) -> None:
        """
        :param load_ph: the load vector corresponding to physical resources.
        :param sigma_2_ph: the variance corresponding to physical resources.
        :return: None.
        """
        self.load_ph = load_ph
        self.sigma_2_ph = sigma_2_ph

    @staticmethod
    def get_forbidden_activities_per_bottleneck(xi_s: types.StateSpace,
                                                nu_s: types.ResourceSpace,
                                                buffer_processing_matrix: types.BufferMatrix,
                                                constituency_matrix: types.ConstituencyMatrix,
                                                tol: float = 1e-6) -> List[int]:
        """
        Return indexes of forbidden activities for some bottleneck (workload direction).

        :param xi_s: Workload vector associated to the resource under study.
        :param nu_s: Constraints vector associated to the resource under study.
        :param buffer_processing_matrix: From the job generator.
        :param constituency_matrix: From the environment.
        :param tol: Tolerance for checking positivity.
        :return: List of forbidden activities in order to ensure this bottleneck is not idling.
        """
        active_constraints = buffer_processing_matrix.T @ xi_s + constituency_matrix.T @ nu_s
        forbidden_activities_s = np.where(active_constraints > tol)[0]
        return forbidden_activities_s.tolist()

    def get_index_all_forbidden_activities(self, nonidling_res) -> Set[int]:
        """
        Return Set with indexes of forbidden activities for the current list of nonidling resources.

        :param nonidling_res: List of nonidling resources.
        :return: Set with indexes of forbidden activities for all nonidling resources.
        """
        ind_forbidden_activities = []
        for s in nonidling_res:
            fa_s = self.get_forbidden_activities_per_bottleneck(
                self.workload_mat[s],
                self.nu[s],
                self.buffer_processing_matrix,
                self.constituency_matrix
            )
            if len(fa_s) > 0:
                ind_forbidden_activities.extend(fa_s)
        return set(ind_forbidden_activities)

    def get_allowed_activities_constraints(self, ind_forbidden_activities) -> types.ActionSpace:
        """
        Takes set of indexes of forbidden activities and return a binary array of allowed actions,
        with zeros in the components with indexes corresponding to forbidden activities, and one
        everywhere else.

        :param ind_forbidden_activities: set of indexes of forbidden activities.
        :return allowed_activities: Binary array of allowed activities.
        """
        allowed_activities = np.ones((self.num_activities, 1))
        for i in ind_forbidden_activities:
            allowed_activities[i] = 0
        return allowed_activities

    def _update_surplus_parameter(self, demand_plan: Dict[int, int]) -> types.StateSpace:
        """
        Returns a vector with a recommended stock level for the surplus buffers that matches the
        demand plan. The demand plan is passed as a dictionary.

        :param demand_plan: Dictionary with keys the identity of the buffers and values the actual
            forecast value.
        :return
        """
        target_surplus = np.zeros((self.num_buffers, 1))
        # Replace surplus buffers with their demand planned values.
        target_surplus[list(demand_plan.keys())] = np.array(list(demand_plan.values()))[:, None]
        return target_surplus

    def _update_nonidling_parameters(self, nonidling_set: np.ndarray) \
            -> Tuple[types.WorkloadMatrix, types.WorkloadSpace]:
        """
        Method creates the new pair of 'workload_buffer_mat' and 'ones_vec_nonidling' arrays
        to update corresponding LP parameters
        """

        # draining constraints, given by:   \xi^{i,T} B zeta + 1 = 0.
        workload_buffer_mat = np.zeros((self.num_wl_vec, self.num_activities))
        ones_vec_nonidling = np.zeros((self.num_wl_vec, 1))
        num_nonidling = nonidling_set.size
        if num_nonidling > 0:
            # Include workload vectors for the nonidling resources.
            workload_buffer_mat[nonidling_set, :] = \
                self.workload_mat[nonidling_set, :] @ self.buffer_processing_matrix
            # Set value to one for the corresponding rows.
            ones_vec_nonidling[nonidling_set, :] = np.ones((num_nonidling, 1))

        return workload_buffer_mat, ones_vec_nonidling

    def get_policy(self, state: types.StateSpace, **kwargs) -> Tuple[types.ActionSpace, float]:
        """
        Returns a vector of activity rates that is approximately optimal in the discounted cost
        sense for the number of time steps given by 'horizon', when starting at 'state'.

        :param state: current state of the environment.
        :param kwargs: Dictionary with parameters relevant tot he policy like safety stock levels,
            idling set, draining bottlenecks, horizon or demand plan.
        :return (z_star, opt_val):
            - z_star: policy as a vector with activity rates to be used for the given horizon.
            - opt_val: value of the objective cost at z_star.
        """
        raise NotImplementedError("This method is meant to be overloaded.")

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON Encoder.
        """
        return clean_to_serializable(self)
