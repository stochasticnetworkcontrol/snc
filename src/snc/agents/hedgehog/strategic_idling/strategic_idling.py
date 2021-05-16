import cvxpy as cvx
import numpy as np
from typing import Optional, List, NamedTuple, Dict, Any

from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.agents.hedgehog.strategic_idling.compute_dual_effective_cost \
    import ComputeDualEffectiveCost
from snc.agents.solver_names import SolverNames
from snc.simulation.store_data.numpy_encoder import clean_to_serializable
from snc.utils.snc_types import Array1D, StateSpace, WorkloadMatrix, WorkloadSpace


StrategicIdlingOutput = NamedTuple('StrategicIdlingOutput',
                                   [('w', WorkloadSpace),
                                    ('x_eff', StateSpace),
                                    ('x_star', StateSpace),
                                    ('beta_star', float),
                                    ('k_idling_set', Array1D),
                                    ('sigma_2_h', float),
                                    ('psi_plus', WorkloadSpace),
                                    ('height_process', np.ndarray),
                                    ('w_star', WorkloadSpace),
                                    ('c_plus', WorkloadSpace),
                                    ('c_bar', WorkloadSpace),
                                    ('psi_plus_cone_list', List[WorkloadSpace]),
                                    ('beta_star_cone_list', List[float]),
                                    ('delta_h', float),
                                    ('lambda_star', float),
                                    ('theta_roots', float)])


class StrategicIdlingCore(object):
    """
    StrategicIdlingCore class is responsible for online identification of idling directions for
    bottlenecks to reduce effective running cost in the network. It does not compute hedging
    thresholds. It is the ancestor class for more involved StrategicIdling classes.
    """
    def __init__(self,
                 workload_mat: WorkloadMatrix,
                 load: WorkloadSpace,
                 cost_per_buffer: StateSpace,
                 list_boundary_constraint_matrices,
                 model_type: str,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 debug_info: bool = False) -> None:
        """
        :param workload_mat: workload matrix, with rows being workload vectors.
        :param load: vector with loads for every workload vector.
        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param model_type: String indicating if this is a `'pull'` or `'push'` model.
        :param strategic_idling_params: tolerance levels and convex solver choice for navigating
            effective cost space.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        """
        self._workload_mat = workload_mat
        self._load = load
        self._cost_per_buffer = cost_per_buffer

        assert model_type in ['push', 'pull']
        self.model_type = model_type

        self.check_strategic_idling_parameters(strategic_idling_params)
        self.strategic_idling_params = strategic_idling_params

        self.debug_info = debug_info

        self._num_bottlenecks, self._num_buffers = workload_mat.shape

        convex_solver = strategic_idling_params.convex_solver
        self.c_bar_solver = ComputeDualEffectiveCost(workload_mat, cost_per_buffer, convex_solver)

        self._w_star_lp_problem, self._x_star, self._w_param, = \
            self._create_find_workload_with_min_eff_cost_by_idling_lp_program()

    @property
    def drift(self):
        return 1 - self._load.reshape(-1, 1)

    @staticmethod
    def check_strategic_idling_parameters(strategic_idling_params):
        assert strategic_idling_params is not None
        assert strategic_idling_params.convex_solver in SolverNames.CVX, \
            f'Convex solver {strategic_idling_params.convex_solver} is not valid. ' \
            f'It must be one of these options: {SolverNames.CVX}.'
        assert strategic_idling_params.penalty_coeff_w_star >= 0, \
            'Penalty coefficient used to ensure w_star close to the lower boundary of the ' \
            f'monotone region must be nonnegative: .'

    def _get_null_strategic_idling_output(self, **overrides) -> StrategicIdlingOutput:
        """
        There is no need to compute the idling directions when we are in the monotone region, so we
        return null values.

        :return: (beta_star, k_idling_set, sigma_2_h, psi_plus, ...)
            - w: current state in workload space, i.e. w = Xi x.
            - beta_star: hedging threshold.
            - k_idling_set: set of possibly idling directions.
            - sigma_2_h: asymptotic variance of the 1-dim "height" process.
            - psi_plus: vector normal to the closest face.
            - height_process: current state of height process variable
            - w_star: Projection of workload on to the nearest cone face
            - c_plus: dual cost vector in the monotone region
            - c_bar: dual cost vector at the current workload
            - psi_plus_cone_list: list of encountered psi_plus vectors
            - beta_star_cone_list: list of encountered beta_star values
            - delta_h: drift of the 1-dim "height" process
            - lambda_star: the price of random oscillations along the closest face
            - theta_roots: root of quadratic equation used for computing hedging
        """
        assert 'w' in overrides, "Current workload variable is not being returned"
        w = overrides['w']
        x_eff = overrides.get('x_eff', np.array([]))
        x_star = overrides.get('x_star', np.array([]))
        beta_star = overrides.get('beta_star', 0)
        k_idling_set = overrides.get('k_idling_set', np.array([]))
        sigma_2_h = overrides.get('sigma_2_h', 0)
        psi_plus = overrides.get('psi_plus', None)
        height_process = overrides.get('height_process', 0.)
        w_star = overrides.get('w_star', None)
        c_plus = overrides.get('c_plus', None)
        c_bar = overrides.get('c_bar', None)
        psi_plus_cone_list = overrides.get('psi_plus_cone_list',
                                           getattr(self, 'psi_plus_cone_list', []))
        beta_star_cone_list = overrides.get('beta_star_cone_list',
                                            getattr(self, 'beta_star_cone_list', []))
        delta_h = overrides.get('delta_h', 0)
        lambda_star = overrides.get('lambda_star', 0)
        theta_roots = overrides.get('theta_roots', None)
        return StrategicIdlingOutput(w, x_eff, x_star, beta_star, k_idling_set, sigma_2_h, psi_plus,
                                     height_process, w_star, c_plus, c_bar,
                                     psi_plus_cone_list, beta_star_cone_list,
                                     delta_h, lambda_star, theta_roots)

    @staticmethod
    def _is_decision_not_to_idle(k_idling_set: np.ndarray) -> bool:
        """
        Method determines whether all the bottlenecks have already been eliminated
        from possible idling action.

        :param k_idling_set: a set of bottlenecks that are permitted to idle.
        :return: bool
        """
        return k_idling_set.size == 0

    @staticmethod
    def _is_1d_workload_relaxation(w: WorkloadSpace) -> bool:
        """
        Check if the current workload relaxation occurs in one dimension.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :return: bool
        """
        return w.shape[0] == 1

    @staticmethod
    def _is_negative_orthant(w: WorkloadSpace, eps: float = 1e-6) -> bool:
        """
        Check if current workload state, w, is in the negative quadrant, i.e.,
        all its components are nonpositive and at least one is strictly negative.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param eps: tolerance value to check that we are not in the negative quadrant.
        :return: bool
        """
        # casting to bool to satisfy mypy disagreement with type np.bool_
        return bool(np.all(w <= eps) and np.any(w < -eps))

    @staticmethod
    def _is_infeasible(c_bar: Optional[WorkloadSpace]) -> bool:
        """
        We know that a given w is infeasible if the corresponding c_bar vector (solution to the dual
        of the effective cost) is None

        :param c_bar: vector defining a level set.
        :return: True if the c_bar vector is None. False otherwise.
        """
        return c_bar is None

    @staticmethod
    def _is_defining_a_monotone_region(c_bar: WorkloadSpace, eps: float = 1e-7) -> bool:
        """
        Checks whether a c_bar vector defines a level set in the region where the effective cost is
        monotone. If c_bar is componentwise nonnegative, then it defines a level set where
        any increment in the workload makes an increment in the effective cost.

        :param c_bar: vector normal to the level set.
        :param eps: tolerance for assessing that a component is null.
        :return: True if c_bar defines a level set of the monotone region. False otherwise.
        """
        # casting to bool to satisfy mypy disagreement with type np.bool_
        return bool(np.all(c_bar >= - eps))

    @staticmethod
    def _is_w_inside_monotone_region(w: WorkloadSpace, w_star: WorkloadSpace,
                                     c_bar: WorkloadSpace) -> bool:
        """
        Given a workload, w, and its projection onto the monotone region, w_star, it checks whether
        w is already inside the monotone region. When w == w_star, then we conclude that there is no
        other workload greater than w that reduces the effective cost. However, depending on the
        actual instance of the LP that we solved in order to obtain w_star, and the solver we used,
        it could happen that even when w != w_star, they still have the same effective cost.
        So a more robust test to check whether we are inside the monotone region, is to check
        whether w and w_star give the same effective cost. Recall that the effective cost can
        be quickly computed for any w by doing the dot product:
            c_bar.T @ w

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param w_star: projection of w onto the monotone region.
        :param c_bar: vector normal to the level set.
        :return: True if w is in the monotone region. False otherwise.
        """
        return np.abs(c_bar.T @ (w_star - w)) < 1e-03

    def _get_level_set_for_current_workload(self, w: WorkloadSpace) -> Optional[WorkloadSpace]:
        """
        The effective cost can be represented as a piecewise linear function,
        with coefficients given by the vertexes of the feasible set of the dual
        program of the LP that computes the effective cost. Indeed, the solution to such
        dual program for a given w, gives the linear coefficient at w.

        :param w: current state in workload space, i.e. w = Xi x.
        :return: c_bar: vector defining level set of the effective cost at current w. None is
            returned if the optimisation is unsuccessful.
        """
        c_bar, x_eff, _ = self.c_bar_solver.solve(w)
        return c_bar, x_eff

    @staticmethod
    def _get_vector_defining_possible_idling_direction(w_star: WorkloadSpace,
                                                       w: WorkloadSpace) -> WorkloadSpace:
        """
        Returns vector in the projection direction from w to w_star, which defines the resources in
        which we might be able to relax the nonidling constraints

        :param w_star: projection of w onto the closest face along the direction of minimum cost.
        :param w: current state in workload space, i.e. w = Xi @ x.
        :return: v_star: vector in the projection direction from w to w_star. If we are in
            the monotone region, then w = w_star and v_star is a vector of np.nan.
        """
        return w_star - w

    def _create_find_workload_with_min_eff_cost_by_idling_lp_program(self, add_safety_stocks=False):
        x_var = cvx.Variable((self._num_buffers, 1), nonneg=True)  # Variable
        w_par = cvx.Parameter((self._num_bottlenecks, 1))  # Parameter
        penalty_coeff_w_star = self.strategic_idling_params.penalty_coeff_w_star
        objective = cvx.Minimize(
            self._cost_per_buffer.T @ x_var
            + penalty_coeff_w_star * cvx.sum(self._workload_mat @ x_var - w_par))
        constraints = [self._workload_mat @ x_var >= w_par]

        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, x_var, w_par

    def _find_workload_with_min_eff_cost_by_idling(self, w: WorkloadSpace) -> WorkloadSpace:
        self._w_param.value = w
        self._w_star_lp_problem.solve(solver=eval(self.strategic_idling_params.convex_solver),
                                      warm_start=True)  # Solve LP.
        x_star = self._x_star.value
        w_star = self._workload_mat @ x_star  # Workload in the boundary of the monotone region.
        tol = 1e-6
        assert np.all(w_star >= w - tol)
        return w_star, x_star

    def _non_negative_workloads(self, w: WorkloadSpace, eps: float = 1e-6) -> Dict[str, Any]:
        """
        Performs all steps needed to obtain the hedging threshold for the closest face.

        :param eps: tolerance value to check that we are not in the negative orthant.
        """
        # In 1-dim workload space, the monotone region is the cone of nonnegative real numbers.
        # Thus, it is not needed to use the cone envelope. In other words, projecting on
        # w_star=0 is OK in 1-dim.
        if not self._is_1d_workload_relaxation(w):
            assert not self._is_negative_orthant(w)
            if not np.any(w > eps):
                return {'w': w, 'w_star': w, 'k_idling_set': np.array([])}

        c_bar, x_eff = self._get_level_set_for_current_workload(w)

        if self._is_infeasible(c_bar):
            return {'w': w, 'w_star': w, 'x_eff': x_eff, 'k_idling_set': np.array([])}
        elif self._is_defining_a_monotone_region(c_bar):
            current_workload_vars = {'w': w, 'w_star': w, 'c_bar': c_bar,
                                     'x_eff': x_eff,
                                     'k_idling_set': np.array([])}
            return current_workload_vars

        w_star, x_star = self._find_workload_with_min_eff_cost_by_idling(w)

        if self._is_w_inside_monotone_region(w, w_star, c_bar):
            # Since c_bar doesn't define a monotone region, w is already at the boundary.
            current_workload_vars = {'w': w, 'w_star': w_star, 'c_bar': c_bar, 'x_eff': x_eff,
                                     'x_star': x_star, 'k_idling_set': np.array([])}
            return current_workload_vars

        v_star = self._get_vector_defining_possible_idling_direction(w_star, w)
        k_idling_set = np.where(v_star > eps)[0]

        current_workload_vars = {'w': w, 'w_star': w_star, 'c_bar': c_bar, 'v_star': v_star, 'x_star': x_star,
                                 'x_eff': x_eff, 'k_idling_set': k_idling_set}

        return current_workload_vars

    def _negative_workloads(self, w: WorkloadSpace, eps: float = 1e-6) -> Dict[str, Any]:
        """
        For strategic idling with no hedging when workload has no positive
        components all resources are allowed to idle.

        :param w: current state in workload space, i.e. w = Xi x.
        :param eps: tolerance value to check that we are in the negative orthant.
        """
        assert self._is_negative_orthant(w, eps)
        current_workload_variables = {'w': w,'w_star': w, 'k_idling_set': np.arange(len(w))}
        return current_workload_variables

    def get_allowed_idling_directions(self, state: StateSpace) -> StrategicIdlingOutput:
        """
        Method projects current worload onto the full monotone effective cost cone in order
        to identify allowed idling directions.

        :param state: current buffer state of the network.
        :return: StrategicIdlingOutput
            - k_idling_set: set of possibly idling directions.
        """
        w = self._workload_mat @ state
        if self._is_negative_orthant(w) and not self._is_1d_workload_relaxation(w):
            idling_decision_dict = self._negative_workloads(w)
        else:
            idling_decision_dict = self._non_negative_workloads(w)

        return self._get_null_strategic_idling_output(**idling_decision_dict)

    def to_serializable(self) -> Dict:
        """Return a serializable object, that can be used by a JSON Encoder"""
        return clean_to_serializable(self)
