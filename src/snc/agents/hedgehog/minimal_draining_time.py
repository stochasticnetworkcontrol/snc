from typing import Tuple

import cvxpy as cvx
import numpy as np
from scipy.optimize import linprog
from snc.agents.hedgehog.workload.workload import compute_load_workload_matrix
from snc.agents.solver_names import SolverNames
from snc.environments.controlled_random_walk import ControlledRandomWalk
import snc.utils.snc_types as types


def compute_minimal_draining_time_from_workload(w: types.WorkloadSpace,
                                                load: types.WorkloadSpace) -> float:
    """
    Method computes minimum achievable draining time for a given workload state.
    Can be used only for strictly push models

    :param w: current state in workload space, i.e. w = Xi @ x.
    :param load: vector with loads for every workload vector.
    :return: continuous draining time
    """
    drift = (1 - load).reshape(-1, 1)
    return np.max((1/drift) * w)


def compute_minimal_draining_time_computing_workload(initial_state: types.StateSpace,
                                                     workload_mat: types.WorkloadMatrix,
                                                     demand_rate: types.StateSpace) -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.8) from CTCN book (online version). This equation consider only the xi_s
    which are workload vectors (o_s=1) and is not the general Eq. (7.4). Thus, it can only be
    used for push model.
    This method is used by the big step policy to estimate the horizon of for which the schedule is
    optimised.
    """
    rho = workload_mat @ demand_rate
    minimal_draining_time = np.max(np.divide(workload_mat @ initial_state,
                                             np.ones(rho.shape) - rho))
    return minimal_draining_time


def compute_minimal_draining_time_computing_workload_from_env(
        initial_state: types.StateSpace,
        env: ControlledRandomWalk,
        demand_rate: types.StateSpace) -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.8) from CTCN book (online version). This equation consider only the xi_s which
    are workload vectors (o_s=1) and is not the general Eq. (7.4). Thus, it can only be used for
    push model.
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods.
    """
    workload_mat = compute_load_workload_matrix(
        env=env, num_wl_vec=None, load_threshold=None, feasible_tol=1e-10).workload_mat
    minimal_draining_time = compute_minimal_draining_time_computing_workload(initial_state,
                                                                             workload_mat,
                                                                             demand_rate)
    return minimal_draining_time


def compute_minimal_draining_time_cvxpy(initial_state: types.StateSpace,
                                        constituency_matrix: types.ConstituencyMatrix,
                                        demand_rate: types.StateSpace,
                                        buffer_processing_matrix: types.BufferMatrix,
                                        convex_solver: str = "cvx.CPLEX") \
        -> Tuple[float, types.ActionSpace]:
    """
    Computes the minimal draining time for an initial state by performing Eq. (6.16) from CTCN book
    (online version). This method works for push and pull models.
    """
    assert convex_solver in SolverNames.CVX
    num_buffers, num_activities = buffer_processing_matrix.shape
    num_resources = constituency_matrix.shape[0]
    t = cvx.Variable()
    z = cvx.Variable((num_activities, 1))
    objective = cvx.Minimize(t)
    constraints = [
        initial_state + buffer_processing_matrix @ z +
        demand_rate * t == np.zeros((num_buffers, 1)),
        constituency_matrix @ z <= t * np.ones((num_resources, 1)),
        z >= np.zeros((num_activities, 1))
    ]
    prob = cvx.Problem(objective, constraints)
    minimal_draining_time = prob.solve(solver=eval(convex_solver))
    return minimal_draining_time, z.value


def compute_minimal_draining_time_from_env_cvxpy(initial_state: types.StateSpace,
                                                 env: ControlledRandomWalk) -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.16) from CTCN book (online version). This method works for push and pull models.
    """
    min_drain_time, _ = compute_minimal_draining_time_cvxpy(
        initial_state, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix)
    return min_drain_time


def compute_dual_minimal_draining_time_cvxpy(initial_state: types.StateSpace,
                                             constituency_matrix: types.ConstituencyMatrix,
                                             demand_rate: types.StateSpace,
                                             buffer_processing_matrix: types.BufferMatrix,
                                             convex_solver: str = "cvx.ECOS") -> float:
    """
    Computes the minimal draining time for an initial state by performing Eq. (6.18) from CTCN book
    (online version). This method works for push and pull models.
    """
    num_buffers, num_activities = buffer_processing_matrix.shape
    num_resources = constituency_matrix.shape[0]
    xi = cvx.Variable((num_buffers, 1))
    nu = cvx.Variable((num_resources, 1))
    objective = cvx.Maximize(xi.T @ initial_state)
    constraints = [
        - buffer_processing_matrix.T @ xi - constituency_matrix.T @ nu
        <= np.zeros((num_activities, 1)),
        - demand_rate.T @ xi + np.ones((num_resources, 1)).T @ nu <= np.ones((1, 1)),
        nu >= np.zeros((num_resources, 1))
    ]
    prob = cvx.Problem(objective, constraints)
    minimal_draining_time = prob.solve(solver=eval(convex_solver))
    return minimal_draining_time


def compute_dual_minimal_draining_time_from_env_cvxpy(initial_state: types.StateSpace,
                                                      env: ControlledRandomWalk) -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.17) from CTCN book (online version). This method works for push and pull
    models.
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods.
    """
    return compute_dual_minimal_draining_time_cvxpy(initial_state, env.constituency_matrix,
                                                    env.job_generator.demand_rate,
                                                    env.job_generator.buffer_processing_matrix)


def compute_minimal_draining_time_from_env_scipy(initial_state: types.StateSpace,
                                                 env: ControlledRandomWalk,
                                                 method: str = 'interior-point') -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.16) from CTCN book (online version). This method works for push and pull
    models.
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods.
    """
    assert method in SolverNames.SCIPY
    res = linprog(c=np.vstack((np.zeros((env.num_activities, 1)), np.ones((1, 1)))),
                  A_eq=np.hstack((- env.job_generator.buffer_processing_matrix,
                                  - env.job_generator.demand_rate)).tolist(),
                  b_eq=initial_state.tolist(),
                  A_ub=np.hstack((env.constituency_matrix,
                                  - np.ones((env.num_resources, 1)))).tolist(),
                  b_ub=np.zeros((env.num_resources, 1)).tolist(),
                  bounds=(0, None), method=method)
    if res.success:
        minimal_draining_time = res.fun
    else:
        if res.status == 2:  # 2 : Problem appears to be infeasible
            minimal_draining_time = None
        elif res.status == 3:  # 3 : Problem appears to be unbounded
            minimal_draining_time = np.inf
    return minimal_draining_time


def compute_dual_minimal_draining_time_scipy(initial_state: types.StateSpace,
                                             env: ControlledRandomWalk,
                                             method: str ='simplex') -> float:
    """
    Computes the minimal draining time for an initial state.
    Perform Eq. (6.17) from CTCN book (online version). This method works for push and pull
    models.
    This method is not used by the algorithm, but just by the tests for checking the results of
    comparing the same magnitudes computed via different methods.
    """
    assert method in SolverNames.SCIPY

    A_first_constraint_ub = np.hstack((- env.job_generator.buffer_processing_matrix.T,
                                       - env.constituency_matrix.T))
    b_first_constraint_ub = np.zeros((env.num_activities, 1))

    A_nu_distribution_ub = np.hstack((- env.job_generator.demand_rate.T,
                                      np.ones((env.num_resources, 1)).T))
    b_nu_distribution_ub = np.ones((1, 1))

    A_nu_positive_ub = np.hstack((np.zeros((env.num_resources, env.num_activities)),
                                  - np.eye(env.num_resources)))
    b_nu_positive_ub = np.zeros((env.num_resources, 1))

    A_ub = np.vstack((A_first_constraint_ub, A_nu_distribution_ub, A_nu_positive_ub)).tolist()
    b_ub = np.vstack((b_first_constraint_ub, b_nu_distribution_ub, b_nu_positive_ub)).tolist()
    # We put a negative sign in front of the initial_state because the dual is a maximisation but
    # scipy only do minimisation
    res = linprog(c=np.vstack((- initial_state, np.zeros((env.num_resources, 1)))), A_ub=A_ub,
                  b_ub=b_ub, bounds=(None, None), method=method)
    if res.success:
        minimal_draining_time = - res.fun
    else:
        if res.status == 2:  # 2 : Problem appears to be infeasible
            minimal_draining_time = None
        elif res.status == 3:  # 3 : Problem appears to be unbounded
            minimal_draining_time = np.inf
    return minimal_draining_time
