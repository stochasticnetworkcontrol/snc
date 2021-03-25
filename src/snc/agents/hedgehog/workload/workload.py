import cdd
from collections import namedtuple
import cvxpy as cvx
import numpy as np
from scipy.spatial import HalfspaceIntersection
from typing import Optional

from src.snc \
    import ValidateNegativeWorkloadVectors
from src.snc.environments import ControlledRandomWalk
from src import snc as types

WorkloadTuple = namedtuple('WorkloadTuple', ['load', 'workload_mat', 'nu'])


def compute_feasible_point(a_mat: types.Matrix, b_vec: types.ColVector, max_r=np.inf,
                           solver='cvx.SCS') \
        -> types.ColVector:
    """
    Computes feasible point inside the intersection of halfspaces given as: A*x <= b.
    In particular, it computes the Chebyshev center of a polyhedron. The returned feasible is used
    as input to compute the vertexes of the polyhedron.

    :param a_mat: A matrix in A*x <= b.
    :param b_vec: b vector in A*x <= b.
    :param max_r: Maximum distance to the boundary. This is useful when the feasible set is open.
    :param solver: Convex optimisation solver to be called by CVX.
    :return:
    """
    r = cvx.Variable()  # Radius
    # Chebyshev center of the polyhedron. Defined as array here, rather than as a column vector with
    # dimensions (solution_length, 1).
    solution_length = a_mat.shape[1]
    x = cvx.Variable(solution_length)
    objective = cvx.Maximize(r)
    # Define constraints.
    constraints = []
    for i in range(a_mat.shape[0]):
        constraints.append(a_mat[i, :] * x + r * cvx.norm(a_mat[i, :]) <= b_vec[i])
        if max_r < np.inf:
            constraints.append(r <= max_r)
    # Define and solve problem
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=eval(solver))
    # Check tests if so, since order of vectors can be different.
    return x.value


def compute_load_workload_matrix(env: ControlledRandomWalk,
                                 num_wl_vec: Optional[int] = None,
                                 load_threshold: Optional[float] = None,
                                 feasible_tol: float = 1e-10,
                                 use_cdd_for_vertices: bool = True,
                                 use_demand_in_lp: bool = False,
                                 solver: str = 'cvx.CPLEX',
                                 tol: float = 1e-4) -> WorkloadTuple:
    """
    Compute the load vector, the associated workload matrix, and the associated dual
    variables.
    :param env: CRW environment
    :param num_wl_vec: gives the number of workload vectors to keep, ranked by higher load.
    :param load_threshold: gives the number of workload vectors to keep specified as a threshold
        load value, such that it gives all workload vectors whose corresponding load is above
        this threshold.
    :param feasible_tol: Tolerance when verifying that workload vector is inside the feasible
        solution.
    :param use_cdd_for_vertices: Use cdd lib to calculate vertices, avoiding QHull errors
    :param use_demand_in_lp: (EXPERIMENTAL) Use the demand/arrival rate in our LP solution for
        workload vectors. This is feature should be switched off mainly, as it needs more
        investigation.
    :param solver: Convex optimisation solver to be called by CVX.
    :param tol: Tolerance to detect negative workload components.
    :return: WorkloadTuple
            - load - vector with loads for every workload vector.
            - workload - workload matrix, with rows being workload vectors.
            - nu - matrix with each row being a transposed dual variable.
    """
    assert (num_wl_vec is None and load_threshold is not None) or \
           (num_wl_vec is not None and load_threshold is None) or \
           (num_wl_vec is None and load_threshold is None)
    if num_wl_vec is not None:
        assert num_wl_vec > 0  # It must return at least one.
    if load_threshold is not None:
        assert 0 <= load_threshold <= 1  # It must return at least one.
    num_buffers = env.job_generator.demand_rate.shape[0]

    if use_demand_in_lp:
        demand_block_matrix = -env.job_generator.demand_rate.T
    else:
        demand_block_matrix = np.zeros((1, env.num_buffers))

    # Build constraints as A*x <= b
    a_mat = np.concatenate((
        np.concatenate((-env.job_generator.buffer_processing_matrix.T, -env.constituency_matrix.T),
                       axis=1),
        np.concatenate((demand_block_matrix, np.ones((1, env.num_resources))), axis=1),
        np.concatenate((np.zeros((env.num_resources, env.num_buffers)), -np.eye(env.num_resources)),
                       axis=1)
    ), axis=0)
    b_vec = np.concatenate((np.zeros((env.num_activities, 1)), np.ones((1, 1)),
                            np.zeros((env.num_resources, 1))), axis=0)

    if use_cdd_for_vertices:
        vertexes = compute_vertexes_cdd(a_mat, b_vec)
    else:
        # Find feasible point and use it to compute all vertexes of the feasible set.
        feasible_point = compute_feasible_point(a_mat, b_vec, solver=solver)
        vertexes = compute_vertexes(a_mat, b_vec, feasible_point.reshape((env.num_buffers +
                                                                          env.num_resources,)))

    # Sort vertexes by load
    workload_mat = vertexes[:, 0:num_buffers]
    load = np.dot(workload_mat, env.job_generator.demand_rate)
    v_load = np.hstack((load, vertexes))
    v_load_sorted_raw = v_load[v_load[:, 0].argsort()[::-1]]

    # Clean vertexes removing nan loads, and zero or unfeasible workload vectors.
    v = np.zeros((0, v_load_sorted_raw.shape[1]))
    for i in range(v_load_sorted_raw.shape[0]):
        # If the load is not nan, and the workload vector is not all zeros, then they are valid.
        if not np.any(np.isnan(v_load_sorted_raw[i])) \
                and not np.allclose(v_load_sorted_raw[i, 1: 1 + env.num_buffers], 0):
            # Be sure the workload vector is inside the feasible solution (plus some tolerance).
            x_i = v_load_sorted_raw[i, 1:].reshape((env.num_buffers + env.num_resources, 1))
            if np.all(np.dot(a_mat, x_i) <= b_vec + feasible_tol):
                v = np.vstack((v, v_load_sorted_raw[i]))
    if v.size == 0:
        raise ValueError("Workload matrix is empty!")

    # Split the variables and return them
    if num_wl_vec is not None:
        # Return just the first 'num_wl_vec' vectors with higher load
        load = v[0:num_wl_vec, 0]
        workload_mat = v[0:num_wl_vec, 1:1 + num_buffers]
        nu = v[0:num_wl_vec, 1 + num_buffers:]
    elif load_threshold is not None:
        # Return the vectors with load >= num_wl_vec_or_load_threshold
        ind_wl = v[:, 0] >= load_threshold
        load = v[ind_wl, 0]
        workload_mat = v[ind_wl, 1:1 + num_buffers]
        nu = v[ind_wl, 1 + num_buffers:]
    else:
        # Return all workload vectors
        load = v[:, 0]
        workload_mat = v[:, 1:1 + num_buffers]
        nu = v[:, 1 + num_buffers:]

    # If there are workload vectors with negative components.
    if np.any(workload_mat <= - tol):
        valid_workload = ValidateNegativeWorkloadVectors(env.constituency_matrix,
                                                         env.job_generator.buffer_processing_matrix,
                                                         workload_mat, tol=tol)
        # Find which of these correspond to draining vectors.
        valid_workload_vec = valid_workload.identify_draining_workload_vectors(workload_mat)
        if np.any(valid_workload_vec == 0):
            load = load[valid_workload_vec == 1]
            workload_mat = workload_mat[valid_workload_vec == 1, :]
            nu = nu[valid_workload_vec == 1, :]

    return WorkloadTuple(load, workload_mat, nu)


def compute_full_workload_matrix_velocity_projection_cdd(env: ControlledRandomWalk,
                                                         use_whole_velocity_space: bool = False,
                                                         debug_info: bool = False) \
        -> types.WorkloadMatrix:
    """
    Compute all the xi_s using the definition based on the velocity space
    (Def 6.1.1 in CTCN online) but return only the workload vectors (o_s=1).
    The general method is to describe the velocity space V_0 (Eq 6.10 in CTCN online) as an
    intersection of halfspaces (i.e list of inequalities). Then, using the Double Description Method
    (of Motzkin et al.) implemented in cdd, to rewrite this intersection of halfspaces as a convex
    hull and cones. Next, we project the vertexes of the convex hull using a linear transformation.
    Finally, we rewrite (using cdd) the projected vertexes in an intersection of halfspaces and we
    scale the halfspaces to have a b_vec of only zeros or ones. The halfspaces which have a b_vec
    equal to one are the workload vectors.

    :param use_whole_velocity_space:
    :param env: CRW environment
    :param use_whole_velocity_space_proj: Whether we use v=B zeta to define the initial intersection
        of halfspaces and project only on v, or we only use zeta >= 0 and C zeta <= 1 to define the
        initial intersection of halfspaces and project on B.
    :param debug_info: Boolean flag indicating whether to print debug info.
    """
    # Define the intersection of halfspaces as [b_vec a_mat] where a_mat @ variables <= b_vec.
    if use_whole_velocity_space:
        # The variables are [v zeta].
        a_mat = np.vstack((np.hstack((np.eye(env.num_buffers),
                                      -env.job_generator.buffer_processing_matrix)),
                           np.hstack((-np.eye(env.num_buffers),
                                      env.job_generator.buffer_processing_matrix)),
                           np.hstack((np.zeros((env.num_resources, env.num_buffers)),
                                      env.constituency_matrix)),
                           np.hstack((np.zeros((env.num_activities, env.num_buffers)),
                                      -np.eye(env.num_activities)))))
        b_vec = np.vstack((np.zeros((env.num_buffers, 1)),
                           np.zeros((env.num_buffers, 1)),
                           np.ones((env.num_resources, 1)),
                           np.zeros((env.num_activities, 1))))
    else:
        # The variables are [zeta].
        a_mat = np.vstack((env.constituency_matrix, -np.eye(env.num_activities)))
        b_vec = np.vstack((np.ones((env.num_resources, 1)), np.zeros((env.num_activities, 1))))

    halfspaces = np.hstack((b_vec, -a_mat))
    h_representation_mat = cdd.Matrix(halfspaces)
    h_representation_mat.rep_type = cdd.RepType.INEQUALITY
    polyhedron = cdd.Polyhedron(h_representation_mat)

    # Rewrite the intersection of halfspaces as the sum of a convex hull and cones.
    v_representation = polyhedron.get_generators()
    v_representation_array = np.array(v_representation)
    # This matrix has an extra leading column, indicating whether a point is
    # a vertex row vector, or a ray row vector. (Rays are unbounded). All vertices
    # appear before all rays.
    # [[ 1, x_11, x_12, ..., x_1n],  -  vertex
    #  [ 1, x_21, x_22, ..., x_2n],  -  vertex
    #  [ 0, x_31, x_32, ..., x_3n]]  -  ray
    # The polyhedron must be bounded, so we only have a convex hull (i.e only vertexes).
    assert np.sum(v_representation_array[:, 0] == 0.) == 0
    vertexes = v_representation_array[:, 1:]

    # Project the vertexes using a linear transform.
    if use_whole_velocity_space:
        # Select only the v variable.
        projector = np.hstack((np.eye(env.num_buffers), np.zeros((env.num_buffers,
                                                                  env.num_activities))))
    else:
        # Project on B.
        projector = env.job_generator.buffer_processing_matrix
    projected_vertexes = np.dot(projector, vertexes.T)

    projected_v_representation = np.hstack((np.ones((vertexes.shape[0], 1)), projected_vertexes.T))
    projected_v_representation_mat = cdd.Matrix(projected_v_representation)
    projected_v_representation_mat.rep_type = cdd.RepType.GENERATOR
    projected_polyhedron = cdd.Polyhedron(projected_v_representation_mat)

    # Rewrite the projected vertexes as an intersection of halfspaces.
    projected_h_representation = projected_polyhedron.get_inequalities()
    projected_h_representation_array = np.array(projected_h_representation)
    # The values very close to zero are set to zero to avoid numerical instabilities when we scale
    # the halfspaces.
    id_almost_zero = np.where(np.isclose(projected_h_representation_array, 0.))
    projected_h_representation_array[id_almost_zero] = 0.
    repeated_column = np.tile(projected_h_representation_array[:, [0]],
                              projected_h_representation_array.shape[1])
    # Scale the halfspaces to have b_vec (i.e o_s) equal to zeros or ones.
    projected_h_representation_array = np.divide(projected_h_representation_array, repeated_column)
    rows_b_equal_one = np.where(np.isclose(projected_h_representation_array[:, 0], 1.))[0]
    # The workload vectors are the rows such that b_vec equal to one (i.e xi_s such that o_s=1)
    workload_mat = projected_h_representation_array[rows_b_equal_one, 1:]

    if debug_info:
        print(projected_vertexes.shape)
        print("projected_h_representation_array.shape: ", projected_h_representation_array.shape)
        print("projected_h_representation_array[:, 0]: ", projected_h_representation_array[:, 0])
        print("projected_h_representation_array: ", projected_h_representation_array)
        print("rows_b_equal_one: ", rows_b_equal_one)
        print("workload_mat_cdd.shape: ", workload_mat.shape)
        print("workload_mat_cdd: ", workload_mat)

    return workload_mat


def compute_vertexes(a_mat: types.Matrix, b_vec: types.ColVector,
                     feasible_point: types.ColVector) -> types.Matrix:
    """
    Compute all vertexes of a polyhedron. Constraints have to be expressed as:  A*x + b <= 0.

    :param a_mat: A matrix in A*x + b <= 0.
    :param b_vec: b vector in A*x + b <= 0.
    :param feasible_point: Initial feasible point.
    :return: vertexes: Matrix with rows given by the vertexes of the feasible set.
    """
    halfspaces = np.hstack((a_mat, -b_vec))
    halfspaces_intersections = HalfspaceIntersection(halfspaces, feasible_point)
    vertexes = halfspaces_intersections.intersections
    return vertexes


def compute_vertexes_cdd(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Compute all vertexes of a polyhedron. Constraints must be expressed as:  A*x <= b."""
    halfspaces = np.hstack((b_vec, -a_mat))
    matrix = cdd.Matrix(halfspaces)
    matrix.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(matrix)
    vertexes_and_edges = poly.get_generators()
    # This matrix has an extra leading column, indicating whether a point is
    # a vertex row vector, or a ray row vector. (Rays are unbounded). All vertices
    # appear before all rays.
    # [[ 1, x_11, x_12, ..., x_1n],  -  vertex
    #  [ 1, x_21, x_22, ..., x_2n],  -  vertex
    #  [ 0, x_31, x_32, ..., x_3n]]  -  ray
    vertexes_and_edges = np.array(vertexes_and_edges)
    rows_are_vertices = np.where(vertexes_and_edges[:, 0] == 1.0)
    vertexes = vertexes_and_edges[rows_are_vertices][:, 1:]
    return vertexes


def describe_workload_space_as_halfspaces(workload_mat):
    """
    Returns the workload space as the intersection of halfspaces. It starts building the V-cone
    description as the nonnegative spanning of the workload vectors:
        P = nonneg(Xi^T).
    Then it transforms to its equivalent H-cone description as the intersection of halfspaces:
        P = {x | A x >= 0}.

    :param workload_mat: Workload matrix.
    :return halfspaces: Matrix with rows representing vector normals to each of the hyperplanes.
    """
    num_buffers = workload_mat.shape[1]
    # Build V-cone: P = nonneg(\Xi^T).
    generator = np.hstack((np.zeros((num_buffers, 1)), workload_mat.T))
    matrix = cdd.Matrix(generator)
    matrix.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(matrix)
    # Transform to H-cone: P = {x | A x <= 0}.
    b_vec_neg_a_mat = np.array(poly.get_inequalities())  # Get the matrix [b -A].
    if b_vec_neg_a_mat.size > 0:
        halfspaces = b_vec_neg_a_mat[:, 1:]  # Get -A such that -A x >= 0.
    else:
        halfspaces = None
    return halfspaces
