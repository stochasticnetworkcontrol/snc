import cvxpy as cvx
import numpy as np
from scipy.optimize import linprog
from typing import Tuple, List, Optional

from src.snc import StrategicIdlingHedging
from src.snc import compute_vertexes_cdd
from src.snc.environments import ControlledRandomWalk
from src import snc as exceptions, snc as types


# --------------------------------------------------------------------------------------------------
#
# The following tests are useful when testing the workload relaxation.
# They provide alternative methods of computing some of the variables.
#
# --------------------------------------------------------------------------------------------------


def compute_network_load(env: ControlledRandomWalk) -> float:
    """Computes the network load, i.e. the highest load associated to any (pooled) resource.
    Perform Eq. (6.5) from CTCN book.
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods."""
    rho = cvx.Variable()
    zeta = cvx.Variable((env.num_activities, 1))
    objective = cvx.Minimize(rho)
    constraints = [
        env.job_generator.buffer_processing_matrix * zeta + env.job_generator.demand_rate
        == np.zeros((env.num_buffers, 1)),
        env.constituency_matrix * zeta <= rho * np.ones((env.num_resources, 1)),
        zeta >= np.zeros((env.num_activities, 1))
    ]
    prob = cvx.Problem(objective, constraints)
    network_load = prob.solve(solver=cvx.SCS, eps=1e-7)
    return network_load


def compute_network_load_and_bottleneck_workload(env: ControlledRandomWalk) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """Computes the network load (i.e. the highest load associated to any (pooled) resource) and its
    associated workload vector.
    Perform Eq. (6.18) from CTCN book with x=alpha, so network_load=W(alpha).
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods."""
    xi = cvx.Variable((env.num_buffers, 1))
    nu = cvx.Variable((env.num_resources, 1))
    objective = cvx.Maximize(xi.T * env.job_generator.demand_rate)
    constraints = [
        -env.job_generator.buffer_processing_matrix.T * xi - env.constituency_matrix.T * nu
        <= np.zeros((env.num_activities, 1)),
        np.ones((env.num_resources, 1)).T * nu <= 1,
        nu >= np.zeros((env.num_resources, 1))
    ]
    prob = cvx.Problem(objective, constraints)
    network_load = prob.solve(solver=cvx.CPLEX)
    return network_load, xi, nu, constraints


# --------------------------------------------------------------------------------------------------
#
# The following tests are useful when testing strategic idling.
# They provide alternative methods of computing some of the variables.
#
# --------------------------------------------------------------------------------------------------


def compute_effective_cost_cvxpy(w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
                                 cost_per_buffer: types.StateSpace) \
        -> Tuple[types.WorkloadSpace, types.StateSpace, float]:
    num_buffers = cost_per_buffer.shape[0]
    x = cvx.Variable((num_buffers, 1))
    objective = cvx.Minimize(cost_per_buffer.T @ x)
    constraints = [workload_mat @ x == w,
                   x >= np.zeros((num_buffers, 1))]
    prob = cvx.Problem(objective, constraints)
    eff_cost = prob.solve(solver=cvx.SCS, eps=1e-8)
    c_bar = - constraints[0].dual_value # Dual var for equality constraint has opposite sign
    assert abs(c_bar.T @ w - eff_cost) < 1e-6
    return c_bar, x.value, eff_cost


def compute_dual_effective_cost_cvxpy(w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
                                      cost_per_buffer: types.StateSpace,
                                      method: str = 'cvx.ECOS', verbose=False) \
        -> Tuple[types.WorkloadSpace, types.StateSpace, float]:
    num_resources = workload_mat.shape[0]
    c_bar = cvx.Variable((num_resources, 1))
    objective = cvx.Maximize(c_bar.T @ w)
    constraints = [workload_mat.T @ c_bar <= cost_per_buffer]
    prob = cvx.Problem(objective, constraints)
    eff_cost = prob.solve(solver=eval(method), verbose=verbose)
    x = constraints[0].dual_value
    return c_bar.value, x, eff_cost


def compute_effective_cost_scipy(w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
                                 cost_per_buffer: types.StateSpace,
                                 method: str = 'revised simplex') \
        -> Tuple[types.WorkloadSpace, types.StateSpace, float]:
    res = linprog(c=cost_per_buffer, A_eq=workload_mat, b_eq=w, bounds=(0, None), method=method)
    if res.success:
        eff_cost = res.fun
        x = res.x[:, None]
    else:
        x = None
        if res.status == 2:  # 2 : Problem appears to be infeasible
            eff_cost = None
        elif res.status == 3:  # 3 : Problem appears to be unbounded
            eff_cost = np.inf
    return None, x, eff_cost


def compute_dual_effective_cost_scipy(w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
                                      cost_per_buffer: types.StateSpace,
                                      method: str = 'revised simplex') \
        -> Tuple[Optional[types.WorkloadSpace], types.StateSpace, float]:
    """
    The effective cost can be represented as a piecewise linear function, with coefficients given
    by the vertexes of the feasible set of the dual program of the LP that computes the effective
    cost. Indeed, the solution to such dual program for a given w, gives the linear coefficient
    at w. The output of this function is a tuple that follows the following interface:
        (
            c_bar = level set, solution to dual program,
            x = effective state, solution to primal program,
            eff_cost = actual value of the effective cost.
        )
    This method uses SciPy to solve the dual LP, which does not return the dual variable (of this
    dual problem), which would be the solution to the primal program. Thus, we return None.

    :param w: current state in workload space, i.e. w = Xi x.
    :param workload_mat: workload matrix.
    :param cost_per_buffer: cost per unit of inventory per buffer.
    :param method:
    :return: (c_bar, None, eff_cost)
        - c_bar: vector defining level set of the effective cost at w. Return None is returned if
            the optimisation is unsuccessful.
    """
    # We remove the columns of the workload matrix which are all zeros because they correspond to
    # inequalities which are always satisfied as we assume that the cost per buffer is always
    # positive. This is needed for scipy linprog because if not done, the method will return
    # status 4: Numerical difficulties encountered. The other methods may not need this step
    non_zeros_columns = np.logical_not(np.all(np.isclose(workload_mat, 0.), axis=0))
    workload_mat = workload_mat[:, non_zeros_columns]
    cost_per_buffer = cost_per_buffer[non_zeros_columns, :]
    res = linprog(c=-w, A_ub=workload_mat.T, b_ub=cost_per_buffer, bounds=(None, None),
                  method=method)
    if res.success:
        eff_cost = - res.fun  # Dual problem is: max c_bar @ w; while here we do: min - c_bar @ w
        c_bar = res.x[:, None]
    else:
        c_bar = None
        if res.status == 2:  # Problem appears to be infeasible
            eff_cost = None
        elif res.status == 3:  # Problem appears to be unbounded
            eff_cost = np.inf
        elif res.status == 4:  # We should not return anything
            raise exceptions.ScipyLinprogStatusError("Scipy encountered numerical difficulties")
    return c_bar, None, eff_cost


def find_workload_with_min_eff_cost_by_idling_scipy(
        w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
        cost_per_buffer: types.StateSpace, method: str = 'interior-point') \
        -> Optional[types.WorkloadSpace]:
    """
    Returns the workload with minimimum effective cost that is achievable when idling from the
    current workload state w. We can think of this as a projection onto the region where the
    effective cost is monotone along the idling directions. It does so by solving an LP that
    minimises the effective cost, subject to the constraint that w_star >= w.

    :param w: current state in workload space, i.e. w = Xi x.
    :param workload_mat: workload matrix.
    :param cost_per_buffer: cost per unit of inventory per buffer.
    :param method: method to solve the LP.
    :return: w_star: projection vector onto the monotone region. None is returned if the
        optimisation is unsuccessful.
    """
    res = linprog(c=cost_per_buffer, A_ub=-workload_mat, b_ub=-w, bounds=(0, None),
                  method=method)
    if res.success:
        x = res.x[:, None]
        w_star = workload_mat @ x
    else:
        w_star = None
    return w_star


def get_all_effective_cost_linear_vectors(workload_mat: types.WorkloadMatrix,
                                          cost_per_buffer: types.StateSpace) -> np.ndarray:
    # Feasible set is:          workload_mat.T * psi <= cost_per_buffer
    num_resources, _ = workload_mat.shape
    a_mat = workload_mat.T
    b_vec = cost_per_buffer
    vertexes = compute_vertexes_cdd(a_mat, b_vec)
    # Clean vertexes that are not feasible solutions.
    v = []
    for i in range(vertexes.shape[0]):
        # Check the vertex is not nan.
        if not np.any(np.isnan(vertexes[i])):
            # Check that the vertex is a feasible solution.
            transformed_vertex = a_mat @ vertexes[i].reshape((num_resources, 1))
            if np.all(transformed_vertex - 1e-10 <= b_vec):
                v.append(vertexes[i])
    if not v:
        raise ValueError("No valid cost vectors!")
    result = np.array(v)
    return result


def project_workload_on_monotone_region_along_minimal_cost_cvxpy(
        w: types.WorkloadSpace, workload_mat: types.WorkloadMatrix,
        cost_per_buffer: types.StateSpace) -> types.WorkloadSpace:
    num_buffers = cost_per_buffer.shape[0]
    x = cvx.Variable((num_buffers, 1))
    objective = cvx.Minimize(cost_per_buffer.T @ x)
    constraints = [workload_mat * x >= w,
                   x >= np.zeros((num_buffers, 1))]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(solver=cvx.SCS, eps=1e-10)
    w_star = np.dot(workload_mat, x.value)
    return w_star


def get_price_lambda_star_strong_duality(w: np.ndarray, w_star: np.ndarray, c_plus: np.ndarray,
                                         psi_plus: np.ndarray) -> np.ndarray:
    """
    Computes lambda_star based on strong duality, so it is only valid if Slater's condition holds,
    which is not the case when the monotone region is a ray.
    We have proved this method only when w is outside the monotone region, so that w_star > w, and
    Slater's condition holds.

    :param w: current state in workload space.
    :param w_star: projection of w onto the closest face along the direction of minimum cost.
    :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
    :param psi_plus: vector normal to the closest face.
    :return: lambda_star: price of random oscillations along the closest face.
    """
    assert not StrategicIdlingHedging._is_w_inside_artificial_monotone_region(w, psi_plus)
    lambda_star_mat = (c_plus.T @ (w_star - w)) / - (psi_plus.T @ w)
    lambda_star = lambda_star_mat.item()
    return lambda_star


def get_price_lambda_star_lp_1_cvxpy(w: np.ndarray, c_plus: np.ndarray, psi_plus: np.ndarray) \
        -> float:
    """
    Computes lambda_star based on dual program of the projection of w_star.

    :param w: current state in workload space.
    :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
    :param psi_plus: vector normal to the closest face.
    :return: lambda_star: price of random oscillations along the closest face.
    """
    assert not StrategicIdlingHedging._is_w_inside_artificial_monotone_region(w, psi_plus)
    num_wl = w.shape[0]
    lambda_var = cvx.Variable(1)
    v_dagger_var = cvx.Variable((num_wl, 1))
    objective = cvx.Maximize(v_dagger_var.T @ w)
    constraints = [c_plus - v_dagger_var - lambda_var * psi_plus == 0,
                   v_dagger_var >= 0]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(solver=cvx.SCS, eps=1e-8)
    lambda_star = lambda_var.value[0]
    if prob.status != 'optimal':
        lambda_star = None
    return lambda_star


def get_price_lambda_star_lp_2_cvxpy(w: np.ndarray, c_plus: np.ndarray, psi_plus: np.ndarray) \
        -> float:
    """
    Computes lambda_star based on dual program of the projection of w_star with only one dual
    variable.

    :param w: current state in workload space.
    :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
    :param psi_plus: vector normal to the closest face.
    :return: lambda_star: price of random oscillations along the closest face.
    """
    assert not StrategicIdlingHedging._is_w_inside_artificial_monotone_region(w, psi_plus)
    lambda_var = cvx.Variable(1)
    objective = cvx.Maximize((c_plus - lambda_var * psi_plus).T @ w)
    constraints = [c_plus - lambda_var * psi_plus >= 0]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(solver=cvx.SCS, eps=1e-8)
    lambda_star = lambda_var.value[0]
    if prob.status != 'optimal':
        lambda_star = None
    return lambda_star


def get_price_lambda_star_lp_scipy(w: np.ndarray, c_plus: np.ndarray, psi_plus: np.ndarray) \
        -> float:
    """
    Computes lambda_star based on dual program of the projection of w_star. It is expressed as
    minimisation to be compatible with SciPy.

    :param w: current state in workload space.
    :param c_plus: vector normal to the level set in the monotone region 'right above' the face.
    :param psi_plus: vector normal to the closest face.
    :return: lambda_star: price of random oscillations along the closest face.
    """
    assert not StrategicIdlingHedging._is_w_inside_artificial_monotone_region(w, psi_plus)
    res = linprog(c=psi_plus.T @ w, A_ub=psi_plus, b_ub=c_plus, bounds=(None, None),
                  method='simplex')
    lambda_star = res.x[:, None]
    return lambda_star[0]
