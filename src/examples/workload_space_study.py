import argparse
import numpy as np

from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_cvxpy
from snc.agents.hedgehog.strategic_idling.compute_dual_effective_cost \
    import ComputeDualEffectiveCost
from snc.agents.hedgehog.strategic_idling.strategic_idling \
    import get_default_strategic_idling_params, StrategicIdlingCore
import snc.agents.hedgehog.workload.workload as wl
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.simulation.utils import validation_utils
from snc.simulation.utils.load_env import load_env, load_env_params
import snc.utils.alt_methods_test as utils


def parse_arg_path():
    """
    Obtain path to json file passed as a command line argument.
    """
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--env_param_path', type=str,
                        help='Path to json file with environment parameters.')
    parser.add_argument('--env_name', type=str, help='Environment name (must be key of '
                                                     'scenarios.SCENARIO_CONSTRUCTORS dict).')
    parser.add_argument('--num_wl_vec', type=int, help='Number of workload vectors to keep.')
    return parser.parse_args()


def find_linearly_independent_workload_matrix(workload_mat):
    """
    TODO: TO BE TESTED
    """
    ld = []
    li = [0]
    k = 1
    for i in range(workload_mat.shape[1]):
        for j in range(i + 1, workload_mat.shape[1]):
            inner_product = np.inner(
                workload_mat[:, i],
                workload_mat[:, j]
            )
            norm_i = np.linalg.norm(workload_mat[:, i])
            norm_j = np.linalg.norm(workload_mat[:, j])

            if np.abs(inner_product - norm_j * norm_i) < 1e-5:
                ld.append(j)
            else:
                li.append(j)
    li = list(set(li))
    li_workload_mat = workload_mat[:, li]
    return li_workload_mat, k


def find_intersection_level_set_with_workload_space(workload_mat, c_bar_vectors):
    """
    Returns intersection of c_bar level sets with boundary of workload space.

    :param workload_mat: Workload matrix.
    :param c_bar_vectors: Matrix with rows being c_bar vectors.
    :return: List of intersection points.
    """
    w_int_list = []
    li_workload_mat, k = find_linearly_independent_workload_matrix(workload_mat)

    for w_vec in li_workload_mat.T:
        for c_bar in c_bar_vectors:
            w_int = w_vec * (np.sum(c_bar)) / (w_vec @ c_bar)
            w_int_list.append(w_int)
    return w_int_list


def find_intersection_two_hyperplanes(normal_vectors_matrix):
    """
    Returns intersection of two planes given by their normal vectors.

    :param normal_vectors_matrix: Matrix with two rows, one per normal vector.
    :return: Vector orthogonal to the intersection. In 2-D, this is a point.
    """
    num_vec, dim = normal_vectors_matrix.shape
    assert num_vec == dim
    return np.linalg.inv(normal_vectors_matrix) @ np.ones((dim, 1))


def find_intersection_c_bar_hyperplanes(c_bar_vectors):
    """
    Returns intersection of the planes defined by the c_bar vectors. The number of intersection
    points grow very quickly with the number of vectors.
    TODO: TO BE TESTED.
    TODO: Allow to select which planes to be studied.

    :param c_bar_vectors: Matrix with rows being the c_bar vectors.
    :return w_int_list: List of intersections among c_bar vectors.
    """
    num_c_bar, dim = c_bar_vectors.shape

    w_int_list = []
    for i in range(num_c_bar - 1):
        for j in range(i+1, num_c_bar):
            c_bar_i = c_bar_vectors[i, :]
            c_bar_j = c_bar_vectors[j, :]
            w_int = find_intersection_two_hyperplanes(np.vstack((c_bar_i, c_bar_j)))
            if np.all(np.abs(w_int) < 1e10):
                w_int_list.append(w_int)

    return w_int_list


def study_workload_space(env, num_wl_vec):
    """
    Compute and returns the workload tuple, including the workload matrix and the load, all the
    c_bar vectors, and the intersection of the planes defined by the c_bar vectors.

    :param env: CRW environment object.
    :param num_wl_vec: Number of workload vectors.
    :return: (workload_tuple, workload_space, c_bar_vectors_vertexes, w_int)
    """
    workload_tuple = wl.compute_load_workload_matrix(env, num_wl_vec=num_wl_vec)
    workload_space = wl.describe_workload_space_as_halfspaces(workload_tuple.workload_mat)

    min_drain_time, z = compute_minimal_draining_time_cvxpy(
        env.job_generator.demand_rate, env.constituency_matrix, 0,
        env.job_generator.buffer_processing_matrix)

    c_bar_vectors = utils.get_all_effective_cost_linear_vectors(
        workload_tuple.workload_mat, env.cost_per_buffer)

    int_c_bar_workload_space = find_intersection_level_set_with_workload_space(
        workload_tuple.workload_mat, c_bar_vectors)

    try:
        int_c_bar = find_intersection_c_bar_hyperplanes(c_bar_vectors)
    except:
        int_c_bar = []

    return (
        workload_tuple,
        workload_space,
        min_drain_time,
        z,
        c_bar_vectors,
        int_c_bar_workload_space,
        int_c_bar
    )


def compute_min_cost_idling_projection(env, workload_tuple):
    """
    For the initial state, it computes the workload and the minimum cost state that can be achieved
    by idling.

    :param env: CRW environment.
    :param workload_tuple: Tuple with workload matrix, load, and nu matrix.
    :return:
        - initial_state
        - w
        - c_bar_vec
        - x_eff
        - x_star
        - w_star
        - v_star
        - si: StrategicIdlingHedging object.
    """

    si_params = get_default_strategic_idling_params()
    initial_state = env.state_initialiser.initial_state.reshape(-1, 1)
    w = workload_tuple.workload_mat @ initial_state

    # Obtain effective state for current workload.
    c_bar_solver = ComputeDualEffectiveCost(workload_tuple.workload_mat,
                                            env.cost_per_buffer,
                                            'cvx.CPLEX')
    c_bar_vec, x_eff, _ = c_bar_solver.solve(w)

    # Obtain minimum instantaneous cost idling projection.
    si = StrategicIdlingHedging(workload_tuple.workload_mat,
                                -np.log(0.99999),
                                workload_tuple.load,
                                env.cost_per_buffer,
                                env.model_type,
                                si_params,
                                np.eye(w.size)
                                )
    si._w_param.value = w
    w_star = si._find_workload_with_min_eff_cost_by_idling(w)
    x_star = si._x_star.value
    v_star = si._get_vector_defining_possible_idling_direction(w_star, w)

    return (
        initial_state,
        w,
        c_bar_vec,
        x_eff,
        x_star,
        w_star,
        v_star,
        si
    )


def compute_standard_hedging(si_obj, w, w_star, v_star):
    """
    For the initial state, it computes the workload and the minimum cost state that can be achieved
    by idling.
    TODO: Instead of relying on the small level calls, we can use the same high level call used by
     HedgehogAgentInterface, and print the information by extending the use of debug_info in the
     strategic idling class.

    :param si_obj: StrategicIdlingHedging object.
    :param w: Current workload vector.
    :param w_star: Minimum cost projection on the monotone region along the idling direction.
    :param v_star: Vector from w to w_star, whose positive components denote possible idling dir.
    :return:
        - c_minus
        - c_plus
        - psi_plus
        - height_process
        - beta_star
        - k_idling_set
    """
    psi_plus, c_plus, c_minus = si_obj._get_closest_face_and_level_sets(w_star, v_star)
    try:
        height_process = si_obj._compute_height_process(w, psi_plus)
        beta_star, sigma_2_h, delta_h, lambda_star, theta_roots \
            = si_obj._compute_hedging_threshold(c_plus, psi_plus)
        beta_star_flag = True
        k_idling_set = si_obj._get_possible_idling_directions(w, beta_star, psi_plus, v_star)
    except:
        height_process = 0
        beta_star = 0
        beta_star_flag = False
        k_idling_set = []
    return (
        c_minus,
        c_plus,
        psi_plus,
        height_process,
        beta_star,
        beta_star_flag,
        k_idling_set
    )


def print_workload_space(
        workload_tuple,
        workload_space,
        min_drain_time,
        z_min_drain_time,
        constituency_matrix,
        c_bar_vectors,
        int_c_bar_workload_space,
        int_c_bar):
    """
    Print workload space characterisation.
    """
    print("\n\n----------------\nWorkload space.\n----------------")
    validation_utils.print_workload_to_physical_resources_indexes(workload_tuple.nu)
    print(f"\nload = {np.squeeze(workload_tuple.load)}")
    print("\nWorkload matrix:")
    print(workload_tuple.workload_mat)
    print("\nnu:")
    print(workload_tuple.nu)

    print("\nWorkload space as set of inequalities, A x >= 0:")
    print(workload_space)

    print(f"\nPer-bottleneck idleness allowed for minimum draining time:\n "
          f"{workload_tuple.nu @ (constituency_matrix @ z_min_drain_time - min_drain_time)}")

    print("\nc_bar_vectors (computed as vertexes of: Xi^T y <= c):")
    print(c_bar_vectors)

    print("\nIntersection points between c_bar and the boundary of the workload space:")
    for i, w in enumerate(int_c_bar_workload_space):
        print(f"w_int_c_bar_boundary_{i} = {np.squeeze(w)}")

    if len(int_c_bar) > 0:
        print("\nIntersection points between c_bar planes:")
        for i, w in enumerate(int_c_bar):
            print(f"w_int_c_bar_{i} = {np.squeeze(w)}")
    else:
        print("\nOnly one c_bar planes (no c_bar planes intersection).")


def print_idling_projection(
        x,
        w,
        c_bar_vec,
        x_eff,
        x_star,
        w_star,
        v_star
):
    """
    Print idling projection.
    """
    print("\n\n----------------\nIdling projection.\n----------------")
    print(f"\nx =      {np.int32(np.squeeze(x))}")
    print(f"x_eff =  {np.int32(np.squeeze(x_eff))}")
    print(f"x_star = {np.int32(np.squeeze(x_star))}")
    print(f"\nw = {np.squeeze(w)}")
    print(f"w_star = {np.squeeze(w_star)}")
    print(f"\nv_star = {np.squeeze(v_star)}")
    print(f"\nc_bar = {np.squeeze(c_bar_vec)}")


def print_hedging(
        c_minus,
        c_plus,
        psi_plus,
        height_process,
        beta_star,
        beta_star_flag,
        k_idling_set
):
    print("\n\n----------------\nHedging.\n----------------")
    print(f"\nc_minus = {np.squeeze(c_minus)}")
    print(f"\nc_plus = {np.squeeze(c_plus)}")
    print(f"\npsi_plus = {np.squeeze(psi_plus)}")
    print(f"\nheight_process = {np.squeeze(height_process)}\n")
    if not beta_star_flag:
        print(f"It wasn't possible to compute hedging.")
    print(f"beta_star = {np.squeeze(beta_star)}")
    print(f"\nk_idling_set = {np.squeeze(k_idling_set)}")


def main_script(env_name, env_param_json_path, num_wl_vec):
    """
    Run script. It requires path and env_name passed as command line argument:

    python workload_space_characterisation.py \
        --env_name product_demo_beer_kegs \
        --env_param_path "/path_to/snc/simulation/example_json/env/product_demo_beer_kegs.json" \
        --num_wl_vec 3

    :param env_name: Name environment example as defined in scenarios.SCENARIO_CONSTRUCTORS list.
    :param env_param_json_path: Absolute path to the json file with environment's parameters.
    :param num_wl_vec: Number of workload vectors.
    """
    if env_param_json_path is not None:
        env_param = load_env_params(env_param_json_path)
    else:
        env_param = None

    env = load_env(env_name, env_param)

    (
        workload_tuple,
        workload_space,
        min_drain_time,
        z_min_drain_time,
        c_bar_vectors_vertexes,
        int_c_bar_workload_space,
        int_c_bar
     ) = study_workload_space(env, num_wl_vec)

    print_workload_space(
        workload_tuple,
        workload_space,
        min_drain_time,
        z_min_drain_time,
        env.constituency_matrix,
        c_bar_vectors_vertexes,
        int_c_bar_workload_space,
        int_c_bar
    )

    initial_state, w, c_bar, x_eff, x_star, w_star, v_star, si_obj \
        = compute_min_cost_idling_projection(env, workload_tuple)
    print_idling_projection(initial_state, w, c_bar, x_eff, x_star, w_star, v_star)
    print(f"\nc.T @ x = {np.squeeze(env.cost_per_buffer.T @ initial_state)}")
    print(f"c.T @ x_eff = {np.squeeze(env.cost_per_buffer.T @ x_eff)}")
    print(f"c_bar.T @ w = {np.squeeze(c_bar.T @ w)}")

    c_minus, c_plus, psi_plus, height_proc, beta_star, beta_star_flag, k_idling_set \
        = compute_standard_hedging(si_obj, w, w_star, v_star)
    print_hedging(c_minus, c_plus, psi_plus, height_proc, beta_star, beta_star_flag, k_idling_set)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    parse_args = parse_arg_path()
    if parse_args.env_name is None:
        raise ValueError("You must provide --env_name command line parameters.")

    main_script(parse_args.env_name, parse_args.env_param_path, parse_args.num_wl_vec)
