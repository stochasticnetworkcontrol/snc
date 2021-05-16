import itertools
import numpy as np
from random import shuffle
from typing import Tuple, Optional, List, Dict, Any

from snc.agents.hedgehog.strategic_idling.compute_dual_effective_cost \
    import ComputeDualEffectiveCost
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore, \
    StrategicIdlingOutput
from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import is_pull_model
from snc.agents.solver_names import SolverNames
from snc.utils.exceptions import e_assert
import snc.utils.exceptions as exceptions
from snc.utils.snc_types import StateSpace, WorkloadSpace
import snc.utils.snc_types as types


class StrategicIdlingHedging(StrategicIdlingCore):

    def __init__(self,
                 workload_mat: types.WorkloadMatrix,
                 neg_log_discount_factor: float,
                 load: WorkloadSpace,
                 cost_per_buffer: types.StateSpace,
                 model_type: str,
                 strategic_idling_params: Optional[StrategicIdlingParams] = None,
                 workload_cov: Optional[types.WorkloadCov] = None,
                 debug_info: bool = False) -> None:
        """
        StrategicIdling class is responsible for online identification of idling directions for
        bottlenecks to reduce effective running cost in the network.

        :param workload_mat: workload matrix, with rows being workload vectors.
        :param neg_log_discount_factor: negative log of the discount factor given by environment.
        :param load: vector with loads for every workload vector.
        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param model_type: String indicating if this is a `'pull'` or `'push'` model.
        :param strategic_idling_params: tolerance levels, convex solver and other params
                                        for navigating effective cost space.
        :param workload_cov: asymptotic covariance of the workload process.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        """
        super().__init__(workload_mat, load, cost_per_buffer,
                         model_type, strategic_idling_params, debug_info)
        self._workload_cov = workload_cov
        self._neg_log_discount_factor = neg_log_discount_factor

        self.check_strategic_idling_parameters(strategic_idling_params)
        self.strategic_idling_params = strategic_idling_params

        self._psi_plus_cone_list: Optional[List[WorkloadSpace]] = None
        self._beta_star_cone_list: Optional[List[float]] = None

        # Create linear programs that will be used at each iteration.
        convex_solver = strategic_idling_params.convex_solver
        self.c_minus_solver = ComputeDualEffectiveCost(workload_mat, cost_per_buffer, convex_solver)
        self.c_plus_solver = ComputeDualEffectiveCost(workload_mat, cost_per_buffer, convex_solver)

        if workload_cov is not None:
            self.update_workload_cov(workload_cov)

    @property
    def psi_plus_cone_list(self):
        return self._psi_plus_cone_list

    @property
    def beta_star_cone_list(self):
        return self._beta_star_cone_list

    @staticmethod
    def check_strategic_idling_parameters(strategic_idling_params):
        assert strategic_idling_params is not None
        assert strategic_idling_params.convex_solver in SolverNames.CVX, \
            f'Convex solver {strategic_idling_params.convex_solver} is not valid. ' \
            f'It must be one of these options: {SolverNames.CVX}.'
        assert strategic_idling_params.epsilon >= 0, \
            f'epsilon parameter to create the artificial cone when monotone region is empty must ' \
            f'be nonnegative: {strategic_idling_params.epsilon}.'
        assert strategic_idling_params.shift_eps >= 0, \
            f'shift_eps parameter to compute distance around w_star when obtaining the level sets' \
            f'that define the monotone region must be nonnegative: ' \
            f'{strategic_idling_params.shift_eps}.'
        assert strategic_idling_params.penalty_coeff_w_star >= 0, \
            'Penalty coefficient used to ensure w_star close to the lower boundary of the ' \
            f'monotone region must be nonnegative: .'
        assert strategic_idling_params.hedging_scaling_factor >= 0, \
            'Scaling factor to multiply the hedging threshold resulting from the diffusion ' \
            f'heuristic must be nonnegative: {strategic_idling_params.penalty_coeff_w_star}. '

    def update_workload_cov(self, workload_cov: Optional[types.WorkloadCov] = None) -> None:
        self._workload_cov = workload_cov
        self._compute_cone_envelope()

    def _compute_cone_envelope(self, max_points: Optional[int] = 200) -> None:
        """
        Pre-computes a conservative envelope of the monotone region of the effective cost cone. This
        cone envelope is only used to have an idea of the hedging threshold when the current
        workload is in the negative quadrant. The cone envelope has as many faces as dimensions on
        the workload space. This function returns the normal vector and the hedging threshold for
        each face of the cone envelope.

        :param max_points: Maximum number of points to compute the cone envelope.
        """
        assert max_points is None or (isinstance(max_points, int) and max_points > 0)

        self._psi_plus_cone_list = []
        self._beta_star_cone_list = []

        if not is_pull_model(self.model_type):
            return

        w_list = self._build_workloads_for_computing_cone_envelope(self._workload_mat,
                                                                   max_points=max_points)
        for w in w_list:
            # Update `psi_plus_cone_list` and `beta_star_cone_list` inside `standard_hedging`.
            if not self._is_negative_orthant(w) or self._is_1d_workload_relaxation(w):
                output = self._non_negative_workloads(w)
                if not self._is_decision_not_to_idle(output['k_idling_set']):
                    _ = self._add_standard_hedging(w, output)

        assert self.psi_plus_cone_list and self.beta_star_cone_list

    @staticmethod
    def _compute_height_process(w: WorkloadSpace, psi_plus: WorkloadSpace) -> np.ndarray:
        """
        One way of infering height process using the psi_plus vector. Height process is
        one particular system state observable instrumental to idling decision.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param psi_plus: vector normal to the closest face.
        """
        assert psi_plus is not None
        assert w is not None
        return np.squeeze(-(psi_plus.T @ w))

    @staticmethod
    def _get_height_process_statistics(psi_plus: WorkloadSpace, workload_cov: types.WorkloadCov,
                                       load: types.WorkloadSpace) -> Tuple[float, float]:
        """
        Obtain drift and variance of the (asymptotically large workload) 1-dim "height" process
        along the "reflection" direction on the closest face (to the current state in workload
        space).

        :param psi_plus: vector normal to the closest face.
        :param workload_cov: asymptotic covariance matrix of the workload process.
        :param load: vector with loads for every workload vector.
        :return:
            - delta_h: drift of the 1-dim "height" process.
            - sigma_2_h: asymptotic variance of the 1-dim "height" process.
        """
        drift = 1 - load
        delta_h = (drift.T @ psi_plus)[0]
        sigma_2_h = (psi_plus.T @ workload_cov @ psi_plus)[0]
        return delta_h, sigma_2_h

    @staticmethod
    def _is_monotone_region_a_ray(c_plus: WorkloadSpace, eps: float = 1e-7) -> bool:
        """
        The monotone region is a ray if c_plus has both positive and negative components.

        :param c_plus: vector defining the level set a bit farther than the projection w_star.
        :param eps: tolerance to evaluate the sign of each components of the c_plus vector.
        :return: True if c_plus has both negative and positive components.
        """
        # casting to bool to satisfy mypy disagreement with type np.bool_
        return bool(np.any(c_plus < -eps) and np.any(c_plus > eps))

    @staticmethod
    def _get_c_plus_when_monotone_region_is_a_ray_or_boundary(c_minus: WorkloadSpace,
                                                              w_star: WorkloadSpace,
                                                              epsilon: float) -> WorkloadSpace:
        """
        When the monotone effective-cost region is a ray or it is the boundary of the feasible
        workload, we build a proper cone around such ray. To do so, we take w_star as the direction
        defining the level sets of this artificial proper cone, and we scale with some
        hyperparameter (epsilon) which is positively correlated with the width of the cone
        (the higher epsilon, the bigger width of the cone and thus, the more conservative).
        If epsilon is equal to zero, then the returned c_plus will define a psi_plus which is
        orthogonal to the ray.

        :param c_minus: c_bar vector corresponding to current workload (i.e. "below" the ray)
        :param w_star: Projection of workload onto the ray
        :param epsilon: Hyperparameter to create the artificial cone that includes the ray.
        :return: c_plus: Vector in the direction of the ray or boundary.
        """
        assert epsilon >= 0
        norm_w_star = np.linalg.norm(w_star)
        # epsilon_nought is the scalar projection of c_minus onto w_star
        epsilon_nought = (c_minus.T @ w_star) / (norm_w_star ** 2)
        assert epsilon_nought >= 0
        c_plus = (epsilon_nought + epsilon / norm_w_star) * w_star
        return c_plus

    @staticmethod
    def _get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(c_minus: WorkloadSpace,
                                                                         w_star: WorkloadSpace,
                                                                         epsilon: float) \
            -> Tuple[WorkloadSpace, WorkloadSpace]:
        """
        When the monotone effective-cost region is a ray, we build a proper cone around such ray.
        To do so, we take w_star as the direction defining the level sets of this artificial proper
        cone, and we scale with some hyperparameter (epsilon) that gives the width of the cone
        (the higher epsilon, the more conservative).
        Similarly, when the monotone effective-cost region is the boundary of the feasible workload
        space, we take w_star as the direction for c_plus, and build an artificial cone with
        one face defined by the boundary of the feasible region.

        :param c_minus: c_bar vector corresponding to current workload (i.e. "below" the ray)
        :param w_star: Projection of workload onto the ray
        :param epsilon: Hyperparameter to create the artificial cone that includes the ray.
        :return:
            - psi_plus: vector normal to the closest face.
            - c_plus: Vector in the direction of the ray or boundary.
        """
        assert epsilon >= 0
        c_plus = StrategicIdlingHedging._get_c_plus_when_monotone_region_is_a_ray_or_boundary(
            c_minus, w_star, epsilon)
        # @TODO: If np.any(c_plus < -eps), then we need another method to compute psi_plus.
        #   See notes with new algorithm for that.
        psi_plus = c_plus - c_minus
        return psi_plus, c_plus

    @staticmethod
    def _is_w_inside_artificial_monotone_region(w: WorkloadSpace, psi_plus: WorkloadSpace) \
            -> bool:
        """
        When the monotone region is a ray, we build an artificial cone containing it, and obtain new
        c_plus and psi_plus vectors. This test checks whether the current workload state, w, is
        inside the new artificial cone.

        :param w: current state in workload space, i.e. w = Xi @ x.
        :param psi_plus: vector normal to the closest face.
        :return: True if w is in the monotone region. False otherwise.
        """
        return psi_plus.T @ w >= 0

    @staticmethod
    def _get_price_lambda_star(c_plus: WorkloadSpace, psi_plus: WorkloadSpace,
                               eps: float = 1e-6) -> float:
        """
        Returns lambda_star i.e. the price of random oscillations along the closest face. There are
        multiple ways of computing lambda_star. This function takes the minimum of the component-
        wise ratio of c_plus over psi_plus. Alternative methods are implemented in alt_methods_test.
        We have proved this method only when w is outside the monotone region, so that w_star > w, a
        nd Slater's condition holds.

        :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
        :param psi_plus: vector normal to the closest face.
        :param eps: tolerance to evaluate positive components of psi_plus.
        :return: lambda_star: price of random oscillations along the closest face.
        """
        e_assert(bool(np.all(c_plus >= -eps)), exceptions.ArraySignError(array_name="c_plus",
                                                                         all_components=True,
                                                                         positive=True,
                                                                         strictly=False))
        c_plus = np.array(np.clip(c_plus, a_min=0., a_max=None))  # Remove numerical errors
        e_assert(bool(np.any(c_plus > eps)), exceptions.ArraySignError(array_name="c_plus",
                                                                       all_components=False,
                                                                       positive=True,
                                                                       strictly=True))
        ratio = np.divide(c_plus[psi_plus > eps], psi_plus[psi_plus > eps])
        # Reshape to avoid 'np.where' thinking that there are 2-dim
        ratio = ratio.reshape((ratio.shape[0],))
        e_assert(ratio.size > 0, exceptions.EmptyArrayError(array_name="ratio"))
        lambda_star = float(np.nanmin(ratio))  # We take 0/0 = inf.
        return lambda_star

    def _compute_hedging_threshold(self, c_plus: WorkloadSpace, psi_plus: WorkloadSpace,
                                   eps: float = 1e-5) -> Tuple[float, float, float, float, float]:
        """
        Returns the hedging threshold as a single scalar for the 1-dim height process along the
        reflection direction, with respect to the closest face.

        :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
        :param psi_plus: vector normal to the closest face.
        :param eps: tolerance to evaluate positive components of psi_plus.
        :return:
            - beta_star: hedging threshold.
            - sigma_2_h: asymptotic variance of the 1-dim "height" process.
            - delta_h: drift of the high process.
            - lambda_star: price of random oscillations along the closest face.
            - theta_roots: positive root to the quadratic equation used to compute beta_star.
        """
        delta_h, sigma_2_h = self._get_height_process_statistics(psi_plus,
                                                                 self._workload_cov, self._load)
        lambda_star = self._get_price_lambda_star(c_plus, psi_plus)

        assert -eps <= lambda_star < 1 + eps
        lambda_star = float(np.clip(lambda_star, a_min=0., a_max=1.))

        # Obtain positive root to the quadratic equation:
        # 0.5 * sigma_2_H * theta^2 - delta_h * theta - discount_factor = 0
        coeff = [0.5 * sigma_2_h, - delta_h, - self._neg_log_discount_factor]
        theta_roots = np.roots(coeff)
        theta_roots = theta_roots[theta_roots > 0]
        assert theta_roots.size == 1
        beta_star = (1 / theta_roots) * np.log(1 + lambda_star / (1 - lambda_star))
        beta_star *= self.strategic_idling_params.hedging_scaling_factor
        return beta_star, sigma_2_h, delta_h, lambda_star, theta_roots

    def _get_closest_face_and_level_sets(self,
                                         w_star: WorkloadSpace,
                                         v_star: WorkloadSpace) \
            -> Tuple[Optional[WorkloadSpace], Optional[WorkloadSpace], Optional[WorkloadSpace]]:
        """
        Obtains two alternative descriptions of the face of the monotone region that is closest to
        the current state in workload space, namely the normal vector to the face, and the
        level sets whose intersection define the face.

        :param w_star: projection of w onto the closest face along the direction of minimum cost.
        :param v_star: direction of projection from w to w_star.
        :return:
            - psi_plus: vector normal to the closest face.
            - c_plus: vector normal to the level set in the monotone region 'right above' the face.
            - c_minus: vector normal to the level set 'right below' the face.
        """
        shift_eps = self.strategic_idling_params.shift_eps
        assert not np.any(np.isnan(v_star))
        assert 0 < shift_eps < 1
        shift_eps /= (1 + np.linalg.norm(v_star))

        w_r_minus = w_star - v_star * shift_eps  # Stop bit before w_star (convex comb).
        w_r_plus = w_star + v_star * shift_eps  # Go a bit farther than w_star.

        c_minus, _, _ = self.c_minus_solver.solve(w_r_minus)
        c_plus, _, _ = self.c_plus_solver.solve(w_r_plus)

        if c_plus is not None and c_minus is not None:
            psi_plus = c_plus - c_minus
        else:
            psi_plus = None
        return psi_plus, c_plus, c_minus

    @staticmethod
    def _get_possible_idling_directions(w: WorkloadSpace,
                                        beta_star: float,
                                        psi_plus: WorkloadSpace,
                                        v_star: WorkloadSpace,
                                        eps: float = 1e-6) -> types.IdlingSet:
        """
        Return set of directions where we are allowed to idle. If we are far enough (more than
        beta_star) from the closest face (given by psi_plus), then we are allowed to idle in
        any convex combination of the directions that lead from w to w_star.

        Positive elements of v_star determine the set of bottlenecks for which short term idling
        will reduce instantaneous cost. However we only allow this idling when sufficiently far from
        the critical boundary. This sufficient separation is determined by hedging threshold.

        :param: w: current state in workload space, i.e. w = Xi @ x.
        :param: beta_star: hedging threshold.
        :param: psi_plus: vector orthogonal to the boundary of the monotone region.
        :param: v_star: direction from w to w_star.
        :return: k_idling_set: set of allowed idling directions.
        """
        if psi_plus.T @ w < - beta_star:
            k_idling_set = np.where(v_star > eps)[0]  # np.where returns a tuple.
        else:
            k_idling_set = np.array([])
        return k_idling_set

    @staticmethod
    def _build_state_list_for_computing_cone_envelope(num_buffers: int,
                                                      init_x: int,
                                                      max_points: Optional[int] = None) \
            -> List[np.ndarray]:
        """
        Returns a list of states with all possible combinations of buffers with positive value.

        :param num_buffers: number of buffers, i.e. dim of state space.
        :param init_x: initial value for the different components of the states in the list.
        :param max_points: Maximum number of points to compute the cone envelope.
        :return: state_list: List of states that will be used to compute workloads from which
                 we will obtain a set of faces that will define the cone envelope.
        """
        assert num_buffers > 0
        assert init_x > 0
        state_list = []
        num_points = 0
        buff_list_1 = list(range(num_buffers))
        shuffle(buff_list_1)
        buff_list_2 = list(range(num_buffers))
        shuffle(buff_list_2)
        for l in buff_list_1:  # For each dimension of the workload space.
            for s in itertools.combinations(buff_list_2, l + 1):
                state = np.zeros((num_buffers, 1))
                for i in s:
                    state[i] = init_x
                state_list.append(state)

                if max_points is not None and num_points == max_points - 1:
                    return state_list
                num_points += 1
        return state_list

    @staticmethod
    def _build_workloads_for_computing_cone_envelope(workload_mat: types.WorkloadMatrix,
                                                     init_x: int = 10,
                                                     max_points: Optional[int] = None) \
            -> List[np.ndarray]:
        """
        Returns a list of workload values for which we will compute the closest faces
        in the monotone region. Such faces will define the initial version of the cone envelope.
        The length of the list equals the number of all possible combinations of buffers in state
        space. By building workloads from valid states, we ensure that they are feasible.

        :param workload_mat: workload matrix.
        :param init_x: initial value for the different components of the states in the list.
        :param max_points: Maximum number of points to compute the cone envelope.
        :return: w_list: list of workload values.
        """
        num_buffers = workload_mat.shape[1]
        state_list = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(
            num_buffers, init_x, max_points)
        w_list = []
        for s in state_list:
            w_list.append(workload_mat @ s)
        return w_list

    def _add_face_to_cone_envelope(self,
                                   psi_plus: WorkloadSpace,
                                   beta_star: float,
                                   eps: float = 1e-2) -> None:
        """
        Add the face given by psi_plus to the cone envelope if it was not already included.

        :param psi_plus: vector normal to the closest face.
        :param beta_star: hedging threshold.
            Threshold for each face of the pre-computed cone envelope.
        :param eps: Tolerance to verify that same face implies same hedging threshold.
        :return: (psi_plus_cone_list, beta_star_cone_list)
            - psi_plus_cone_list: Updated matrix with rows equal to the vector normal
        to the faces of the pre-computed cone envelope.
            - beta_star_cone_list: Updated vector with elements equal to the hedging
        threshold for each face of the pre-computed cone envelope.
        """
        for i, (p, b) in enumerate(zip(self._psi_plus_cone_list, self._beta_star_cone_list)):
            if np.linalg.norm(psi_plus - p) / np.linalg.norm(psi_plus) < eps:
                if b > 1e-6 and abs(beta_star - b) / b < 0.05:
                    return
                else:
                    if beta_star > b:
                        self.beta_star_cone_list[i] = beta_star  # Update to be more conservative.
                    return

        self._psi_plus_cone_list.append(psi_plus)
        self._beta_star_cone_list.append(beta_star)

    def _add_standard_hedging(self, w: WorkloadSpace,
                              current_workload_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs all steps needed to obtain the hedging threshold for the closest face.

        :param w: current state in workload space, i.e. w = Xi x.
        :current_workload_variables: dictionary containing all the current workload space variables
        :return: StrategicIdlingOutput
            - beta_star: hedging threshold.
            - k_idling_set: set of possibly idling directions.
            - sigma_2_h: asymptotic variance of the 1-dim "height" process.
            - psi_plus: vector normal to the closest face.
            - delta_h: drift projected in the direction of the "height" process.
            - lambda_star: price of random oscillations along the closest face.
            - theta_roots: root of quadratic equation used for computing hedging.
        """
        c_bar = current_workload_variables['c_bar']
        w_star = current_workload_variables['w_star']
        v_star = current_workload_variables['v_star']

        psi_plus, c_plus, c_minus = self._get_closest_face_and_level_sets(w_star, v_star)
        hedging_case = 'standard'

        if self._is_infeasible(c_plus):
            psi_plus, c_plus = \
                self._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
                    c_minus, w_star, self.strategic_idling_params.epsilon)
            hedging_case = 'infeasible'
        elif self._is_monotone_region_a_ray(c_plus):
            psi_plus, c_plus = \
                self._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
                    c_minus, w_star, self.strategic_idling_params.epsilon)
            hedging_case = 'empty_interior'

        height_process = self._compute_height_process(w, psi_plus)
        current_workload_variables['psi_plus'] = psi_plus
        current_workload_variables['c_plus'] = c_plus
        current_workload_variables['hedging_case'] = hedging_case
        current_workload_variables['height_process'] = height_process

        if self._is_w_inside_artificial_monotone_region(w, psi_plus):
            return current_workload_variables

        beta_star, sigma_2_h, delta_h, lambda_star, theta_roots \
            = self._compute_hedging_threshold(c_plus, psi_plus)
        k_idling_set = self._get_possible_idling_directions(w, beta_star, psi_plus, v_star)

        if is_pull_model(self.model_type) and not self._is_1d_workload_relaxation(w):
            # Update cone envelope with the current closest face if needed.
            self._add_face_to_cone_envelope(psi_plus, beta_star)

        current_workload_variables['beta_star'] = beta_star
        current_workload_variables['k_idling_set'] = k_idling_set
        current_workload_variables['sigma_2_h'] = sigma_2_h
        current_workload_variables['delta_h'] = delta_h
        current_workload_variables['lambda_star'] = lambda_star
        current_workload_variables['theta_roots'] = theta_roots


        return current_workload_variables

    def _negative_workloads(self, w: WorkloadSpace, eps: float = 1e-6) \
            -> Dict[str, Any]:
        """
        This function is only used when the current workload, w, is in the negative orthant,
        such that if we are below any of the faces of the cone envelope, then we are allowed
        to idle in the direction towards the origin.

        :param w: current state in workload space, i.e. w = Xi x.
        :param eps: tolerance value to check that we are in the negative orthant.
        :return: k_idling_set: set of possibly idling directions.
        """
        assert self._is_negative_orthant(w, eps)
        # Cone envelope lists must be non-empty.
        assert self.psi_plus_cone_list and self.beta_star_cone_list
        k_idling_set = np.array([])
        beta_star = 0.
        c_bar = self._get_level_set_for_current_workload(w)
        for psi_plus, beta_star in zip(self.psi_plus_cone_list, self.beta_star_cone_list):
            height_process = self._compute_height_process(w, psi_plus)
            if height_process >  beta_star:
                k_idling_set = np.arange(c_bar.size)  # Use [0] since np.where returns a tuple.
                break  # We are out of the monotone region

        current_workload_variables = {'w': w,
                                      'k_idling_set': k_idling_set,
                                      'beta_star': beta_star,
                                      'c_bar': c_bar,
                                      'height_process': height_process,
                                      'psi_plus_cone_list': self.psi_plus_cone_list,
                                      'beta_star_cone_list': self.beta_star_cone_list}

        return current_workload_variables

    def _verify_offline_preliminaries(self) -> None:
        assert self._workload_cov is not None, \
                "update workload covariance first to run policy with hedging"

    def get_allowed_idling_directions(self, x: StateSpace, safety_stocks_vec) -> StrategicIdlingOutput:
        """
        Method projects current worload onto the full monotone effective cost cone, or
        projects onto the precomputed envelope of the monotone effective cost cone.

        :param x: current buffer state of the network.
        :return: set of allowed idling resources with auxiliary variables
        """
        w = self._workload_mat @ x
        self._safety_stocks_vec = safety_stocks_vec
        self._verify_offline_preliminaries()
        if self._is_negative_orthant(w):
            idling_decision_dict = self._negative_workloads(w)
        else:
            current_workload_variables = self._non_negative_workloads(w)

            if self._is_decision_not_to_idle(current_workload_variables['k_idling_set']):
                idling_decision_dict = current_workload_variables
            else:
                idling_decision_dict = self._add_standard_hedging(w, current_workload_variables)

        idling_decision = self._get_null_strategic_idling_output(**idling_decision_dict)

        if self.debug_info:
            print(f"beta_star: {idling_decision.beta_star}, "
                  f"k_iling_set: {idling_decision.k_idling_set}, "
                  f"sigma_2_h: {idling_decision.sigma_2_h}, "
                  f"delta_h: {idling_decision.delta_h}")
        return idling_decision
