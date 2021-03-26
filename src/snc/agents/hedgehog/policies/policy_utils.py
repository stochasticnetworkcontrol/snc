from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import cvxpy as cvx
from scipy.optimize import linprog
import snc.utils.snc_types as types


def obtain_nonidling_bottleneck_resources(num_bottlenecks: int,
                                          k_idling_set: types.Array1D) -> types.Array1D:
    """
    Returns the bottlenecks for which we enforce the nonidling constraint by subtracting the
    directions that are allowed to idle from all the possible directions.

    :param num_bottlenecks: Num of dimensions in workload space.
    :param k_idling_set: Set of directions in which we are allowed to idle.
    :return: Set of workload dimension in which we shouldn't idle.
    """
    assert num_bottlenecks >= k_idling_set.size, \
        f"Size of k_idling_set must be less or equal than num_wl_vec, since it is a subset." \
        f"But provided k_idling_set.size={k_idling_set.size} > {num_bottlenecks}=num_wl_vec."
    nonidling_resources = np.setdiff1d(np.array(range(num_bottlenecks)), k_idling_set)
    return nonidling_resources


def add_draining_bottlenecks_to_nonidling_resources(draining_bottlenecks: Set[int],
                                                    nonidling_res: types.Array1D) -> types.Array1D:
    """
    Takes the set of nonidling resources and adds the draining bottlenecks if they weren't in the
    set. Draining bottlenecks are those workload resources that determine the minimum draining time.

    :param draining_bottlenecks: set of resources that determine the draining time.
    :param nonidling_res: Set of workload dimension in which we shouldn't idle.
    """
    for i in draining_bottlenecks:
        if i not in nonidling_res:
            nonidling_res = np.hstack((nonidling_res, i))
    return nonidling_res


def get_index_activities_that_can_drain_buffer(b: int,
                                               buffer_processing_matrix: types.BufferMatrix) \
        -> types.Array1D:
    """
    Returns an array with the indexes of the activities that can drain buffer b. This is
    obtained from the negative entries of the b-th row of the buffer processing matrix.

    :param b: index of the buffer.
    :param buffer_processing_matrix:
    :return: indexes of activities that can drain buffer b
    """
    return np.where(buffer_processing_matrix[b] < 0)[0]


def get_index_non_exit_activities(buffer_processing_matrix: types.BufferMatrix) -> List[int]:
    """
    Returns the indexes of the activities that do not drain exit buffers.

    :param buffer_processing_matrix: from job generator (the environment).
    :return: non_exit_activities: indexes of activities that do not drain exit resources.
    """
    non_exit_activities = []
    for i, b in enumerate(buffer_processing_matrix.T):
        if np.any(b > 0):
            non_exit_activities.append(i)
    return non_exit_activities


def obtain_general_empty_buffers_constraints_matrix_form(
        state: types.StateSpace, buffer_processing_matrix: types.BufferMatrix) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns general nonempty buffer constraints in matrix form. It is general in the sense that
    ensures that routing models, where multiple activities can drain the same buffer, never
    drain beyond the actual capacity of the buffers.

    :param state: current state of the environment.
    :param buffer_processing_matrix: from job generator (the environment).
    :return: (a_mat, b_vec)
        - a_mat: List of rows that will be added to the matrix of the inequality constraints.
        - b_vec: List of elements that will be added to the vector of inequality constraints.
    """
    num_buffers, num_activities = buffer_processing_matrix.shape
    a_mat = []
    b_vec = []
    for b in range(num_buffers):
        act_b = get_index_activities_that_can_drain_buffer(
            b, buffer_processing_matrix)
        if act_b.size > state[b]:  # Buffer might be drained more than its the current size.
            new_a_mat_row = np.zeros((num_activities,))
            new_a_mat_row[act_b] = np.ones((act_b.size,))
            new_b_vec_row = state[b]
            a_mat.append(new_a_mat_row)
            b_vec.append(new_b_vec_row)
    return a_mat, b_vec


def feedback_policy_nonidling_penalty_scipy(
        state: types.StateSpace,
        cost_per_buffer: types.StateSpace,
        constituency_matrix: types.ConstituencyMatrix,
        buffer_processing_matrix: types.BufferMatrix,
        workload_mat: types.WorkloadMatrix,
        safety_stocks_vec: types.ResourceSpace,
        k_idling_set: types.Array1D,
        draining_bottlenecks: Set[int],
        kappa_w: float,
        list_boundary_constraint_matrices: List[np.ndarray],
        allowed_activities: types.ActionSpace,
        method: str = 'revised simplex',
        demand_rate: Optional[types.StateSpace] = None,
        horizon: Optional[int] = None) -> Tuple[types.ActionSpace, float]:
    """
    Returns an action that is approximately optimal in the discounted cost sense for the given
    state, relaxing the non-idling equality constraints to with some additive nonnegative tolerance
    to ensure feasibility, and adding a penalty cost in the objective that minimises the tolerance.

    :param state: current state of the CRW.
    :param cost_per_buffer: from the environment.
    :param constituency_matrix: from the environment.
    :param buffer_processing_matrix: from from job generator (the environment).
    :param workload_mat: workload matrix.
    :param safety_stocks_vec: Vector with safety stock levels for current state.
    :param k_idling_set: set of resources that should idle obtained when computing hedging.
        resources.
    :param draining_bottlenecks: set of resources that determine the draining time.
    :param kappa_w: nonidling penalty constant (>> safety stock penalty), the same for all
    :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
        indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
    :param horizon: number of time steps that this policy should be performed.
    :param allowed_activities: Binary vector indicating which activities are allowed.
    :param method: preferred method for SciPy solver.
    :param demand_rate: from the environment
    :return (z_star, opt_val):
        - z_star: matrix where columns are the actions for each of the given horizon.
        - opt_val: value of the objective cost at z_star. Note that this opt_val doesn't include the
            constant term: penalty_grad.T @ buffer_processing_matrix.
    """

    assert np.round(horizon) == horizon
    assert horizon >= 0
    assert demand_rate is not None
    assert kappa_w >= 0

    num_resources, num_activities = constituency_matrix.shape
    num_buffers = buffer_processing_matrix.shape[0]

    num_bottlenecks = workload_mat.shape[0]
    nonidling_res = obtain_nonidling_bottleneck_resources(num_bottlenecks, k_idling_set)
    nonidling_res = add_draining_bottlenecks_to_nonidling_resources(draining_bottlenecks,
                                                                    nonidling_res)
    num_nonidle = nonidling_res.size
    ones_weighted = np.ones((num_nonidle, 1))
    for i in range(num_nonidle):
        if nonidling_res[i] in draining_bottlenecks:
            ones_weighted[i] *= kappa_w
    zeros_buff_mat = np.zeros((num_buffers, num_nonidle))
    zeros_vec = np.zeros((num_activities, 1))

    # For any constant value, we have: min_x f(x) = min_x f(x) + constant. So we can safely ignore
    # the kappa_w constant that results from expanding the objective. Indeed, SciPy.linprog does
    # not allow to include any constant in the formulation.
    c = np.squeeze(np.hstack((buffer_processing_matrix, zeros_buff_mat)).T  @ cost_per_buffer
                   + kappa_w * np.vstack((zeros_vec, ones_weighted)))

    if num_nonidle > 0:
        zeros_nonidling_act = np.zeros((num_nonidle, num_activities))
        zeros_nonidling_tol = np.zeros((num_nonidle, num_nonidle))
        b_feasible_eq = -np.ones((num_nonidle, 1))
        A_feasible_eq = (
            np.hstack((workload_mat[nonidling_res, :] @ buffer_processing_matrix,
                       zeros_nonidling_tol))
            - np.hstack((zeros_nonidling_act, np.eye(num_nonidle)))
        )
        b_eq = b_feasible_eq.reshape((num_nonidle,))
        A_eq = A_feasible_eq.tolist()
    else:
        b_eq = None
        A_eq = None

    # scipy inequalities in upper-bound form b >= Az
    b_ub_vec = np.ones((num_resources, 1))
    A_ub_mat = constituency_matrix  # type: np.ndarray

    # Safety stock constraints
    tol = 1e-6
    for i, boundary_matrix in enumerate(list_boundary_constraint_matrices):
        if np.any(boundary_matrix > tol):
            b_ub_vec = np.vstack((
                b_ub_vec, boundary_matrix @ (state + demand_rate * horizon) - safety_stocks_vec[i]))
            A_ub_mat = np.vstack((A_ub_mat, - horizon * boundary_matrix @ buffer_processing_matrix))

    # Forbidden activities constraints.
    b_ub_vec = np.vstack((b_ub_vec, allowed_activities))
    A_ub_mat = np.vstack((A_ub_mat, np.eye(num_activities)))

    # Nonempty buffer constraints.
    b_ub_vec = np.vstack((b_ub_vec, state + demand_rate * horizon))
    A_ub_mat = np.vstack((A_ub_mat, - horizon * buffer_processing_matrix))

    A_ub_mat = np.hstack((A_ub_mat, np.zeros((A_ub_mat.shape[0], num_nonidle))))

    b_ub = b_ub_vec.tolist()
    A_ub = A_ub_mat.tolist()

    # It is not needed to include the non-negativity of z , since this is included by default in
    # linprog, i.e. bounds=(0, None) (non-negative). We make the default parameter explicit though.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method=method, bounds=(0, None))
    opt_val = res.fun + kappa_w * num_nonidle
    if res['success']:
        z_star = res.x[0:-num_nonidle]
    else:
        z_star = None
    return z_star, opt_val


def feedback_policy_nonidling_constraint_cvx(
        state: types.StateSpace,
        cost_per_buffer: types.StateSpace,
        constituency_matrix: types.ConstituencyMatrix,
        buffer_processing_matrix: types.BufferMatrix,
        workload_mat: types.WorkloadMatrix,
        safety_stocks_vec: types.ResourceSpace,
        k_idling_set: types.Array1D,
        draining_bottlenecks: Set[int],
        list_boundary_constraint_matrices: List[np.ndarray],
        allowed_activities: types.ActionSpace,
        demand_rate: types.StateSpace,
        horizon: float,
        demand_plan: Optional[Dict[int, int]] = None,
        convex_solver: str = 'cvx.ECOS') -> Tuple[types.ActionSpace, float]:
    """
    Returns an action that is approximately optimal in the discounted cost sense for the given
    state, if it exists under hard non-idling constraints. The main drawback with this approach is
    that the feasible might be empty.

    :param state: current state of the CRW.
    :param cost_per_buffer: from the environment.
    :param constituency_matrix: from the environment.
    :param buffer_processing_matrix: from from job generator (the environment).
    :param workload_mat: workload matrix.
    :param safety_stocks_vec: Vector with safety stock levels for current state.
    :param k_idling_set: set of resources that should idle obtained when computing hedging.
    :param draining_bottlenecks: set of resources that determine the draining time.
    :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
            indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
    :param allowed_activities: Binary vector indicating which activities are allowed.
    :param demand_rate: demand rate vector from environment.
    :param horizon: number of time steps that this policy should be performed.
    :param demand_plan: Dictionary with keys the identity of the buffers and values the actual
        forecast value.
    :param convex_solver: preferred solver for CVX wrapper.
    :return (z_star, opt_val):
        - z_star: matrix where columns are the actions for each of the given horizon.
        - opt_val: value of the objective cost at z_star
    """
    num_resources, num_activities = constituency_matrix.shape
    num_buffers = buffer_processing_matrix.shape[0]

    z = cvx.Variable((num_activities, 1), nonneg=True)
    objective = cvx.Minimize(cost_per_buffer.T @ buffer_processing_matrix @ z)

    constraints = []

    # Surplus constraints for pull models.
    if demand_plan is not None:
        surplus_target_vec = np.zeros((num_buffers, 1))
        surplus_target_vec[list(demand_plan.keys())] = np.array(list(demand_plan.values()))[:, None]
        constraints.append(
            state + (buffer_processing_matrix @ z + demand_rate) * horizon >= surplus_target_vec
        )
    # Nonidling constraints.
    num_bottlenecks = workload_mat.shape[0]
    nonidling_res = obtain_nonidling_bottleneck_resources(num_bottlenecks, k_idling_set)
    nonidling_res = add_draining_bottlenecks_to_nonidling_resources(draining_bottlenecks,
                                                                    nonidling_res)
    num_nonidle = nonidling_res.size
    if num_nonidle > 0:
        workload_buffer_mat = workload_mat[nonidling_res, :] @ buffer_processing_matrix
        constraints.append(workload_buffer_mat @ z == -np.ones((num_nonidle, 1)))

    # Buffer, action feasibility, and forbidden activities constraints.
    constraints += [
        state + (buffer_processing_matrix @ z + demand_rate) * horizon >= 0,  # Feasible next state.
        constituency_matrix @ z <= np.ones((num_resources, 1)),  # Resource constraints.
        z <= allowed_activities  # Forbidden activities.
    ]

    # Safety stock threshold constraints.
    for boundary_matrix in list_boundary_constraint_matrices:
        if np.any(boundary_matrix > 0):
            constraints.append(
                boundary_matrix @ (
                    state + (buffer_processing_matrix @ z + demand_rate) * horizon)
                >= safety_stocks_vec
            )

    # Nonempty buffer constraints, also for routing (when buffers can be drained by many actions)
    num_buffers = buffer_processing_matrix.shape[0]
    for b in range(num_buffers):
        act_b = get_index_activities_that_can_drain_buffer(
            b, buffer_processing_matrix)
        if act_b.size > state[b]:
            constraints.append(cvx.sum(z[act_b]) <= state[b])

    prob = cvx.Problem(objective, constraints)
    opt_val = prob.solve(solver=eval(convex_solver))
    z_star = z.value
    return z_star, opt_val


def feedback_policy_nonidling_constraint_scipy(
        state: types.StateSpace,
        cost_per_buffer: types.StateSpace,
        constituency_matrix: types.ConstituencyMatrix,
        buffer_processing_matrix: types.BufferMatrix,
        workload_mat: types.WorkloadMatrix,
        safety_stocks_vec: types.ResourceSpace,
        k_idling_set: types.Array1D,
        draining_bottlenecks: Set[int],
        list_boundary_constraint_matrices: List[np.ndarray],
        allowed_activities: types.ActionSpace,
        demand_rate: types.StateSpace,
        horizon: int,
        method: str = 'revised simplex') -> Tuple[types.ActionSpace, float]:
    """
    Alternative method that uses a different optimisation library.
    Returns an action that is approximately optimal in the discounted cost sense for the given
    state, if it exists under hard non-idling constraints. The main drawback with this approach is
    that the feasible might be empty.

    :param state: current state of the CRW.
    :param cost_per_buffer: from the environment.
    :param constituency_matrix: from the environment.
    :param buffer_processing_matrix: from from job generator (the environment).
    :param workload_mat: workload matrix.
    :param safety_stocks_vec: Vector with safety stock levels for current state.
    :param k_idling_set: set of resources that should idle obtained when computing hedging.
    :param draining_bottlenecks: set of resources that determine the draining time.
    :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
        indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
    :param allowed_activities: Binary vector indicating which activities are allowed.
    :param demand_rate: from the environment.
    :param horizon: number of time steps that this policy should be performed.
    :param method: preferred method for SciPy solver.
    :return (z_star, opt_val):
        - z_star: matrix where columns are the actions for each of the given horizon.
        - opt_val: value of the objective cost at z_star
    """
    assert horizon >= 1
    assert demand_rate is not None

    num_resources, num_activities = constituency_matrix.shape

    c = np.squeeze(buffer_processing_matrix.T @ cost_per_buffer)

    # scipy inequalities must be in upper-bound form b >= Az
    b_ub_vec = np.ones((num_resources, 1))
    A_ub_mat = constituency_matrix

    # Safety stock constraints
    for i, boundary_matrix in enumerate(list_boundary_constraint_matrices):
        if np.any(boundary_matrix > 0):
            b_ub_vec = np.vstack((
                b_ub_vec, boundary_matrix @ (state + demand_rate * horizon) - safety_stocks_vec[i]))
            A_ub_mat = np.vstack((A_ub_mat, - horizon * boundary_matrix @ buffer_processing_matrix))

    # Forbidden activities constraints.
    b_ub_vec = np.vstack((b_ub_vec, allowed_activities))
    A_ub_mat = np.vstack((A_ub_mat, np.eye(num_activities)))

    # Nonempty buffer constraints.
    b_ub_vec = np.vstack((b_ub_vec, state + demand_rate * horizon))
    A_ub_mat = np.vstack((A_ub_mat, - horizon * buffer_processing_matrix))

    b_ub = b_ub_vec.tolist()
    A_ub = A_ub_mat.tolist()

    # scipy equalities must be in upper-bound form   b = Az
    num_bottlenecks = workload_mat.shape[0]
    nonidling_res = obtain_nonidling_bottleneck_resources(num_bottlenecks, k_idling_set)
    nonidling_res = add_draining_bottlenecks_to_nonidling_resources(draining_bottlenecks,
                                                                    nonidling_res)
    num_nonidle = nonidling_res.size
    if num_nonidle > 0:
        b_feasible_eq = -np.ones((num_nonidle, 1))
        A_feasible_eq = workload_mat[nonidling_res, :] @ buffer_processing_matrix

        b_eq = b_feasible_eq.reshape((num_nonidle,))
        A_eq = A_feasible_eq.tolist()
    else:
        b_eq = None
        A_eq = None
    # It is not needed to include the non-negativity of z , since this is included by default in
    # linprog, i.e. bounds=(0, None) (non-negative). We make the default parameter explicit though.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=method)
    opt_val = res.fun
    z_star = res.x
    return z_star, opt_val
