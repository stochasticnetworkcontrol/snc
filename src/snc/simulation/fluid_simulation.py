import argparse
import numpy as np
from snc.agents.activity_rate_to_mpc_actions.feedback_mip_feasible_mpc_policy \
    import FeedbackMipFeasibleMpcPolicy
from snc.agents.hedgehog.minimal_draining_time import compute_minimal_draining_time_from_workload \
    as compute_min_drain_time
from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.agents.hedgehog.policies.big_step_w_bound_policy import BigStepWBoundPolicy
from snc.agents.hedgehog.strategic_idling.strategic_idling_horizon import StrategicIdlingCoreHorizon
import snc.agents.hedgehog.workload.workload as wl
from snc.environments.controlled_random_walk import ControlledRandomWalk
from snc.simulation.utils.load_env import load_env, load_env_params


def parse_arg_path():
    """
    Obtain path to json file passed as a command line argument.
    """
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--env_param_json_file', type=str,
                        help='Path to json file with environment parameters.')
    parser.add_argument('--env_name', type=str, help='Environment name (must be key of '
                                                     'scenarios.SCENARIO_CONSTRUCTORS dict).')
    parser.add_argument('--num_wl_vec', type=int, help='Number of workload vectors to keep.')
    parser.add_argument('--mpc', action="store_true", help='Simulate with MPC policy.')
    return parser.parse_args()


def print_action_update(total_fluid, total_actions, action, sum_actions):
    activity_ratio = np.squeeze(np.divide(total_actions, total_fluid))
    print(f"action: {np.squeeze(action)}")
    print(f"sum_actions: {np.squeeze(sum_actions)}")
    print(f"activity_ratio: {activity_ratio}")


def print_state_update(i, x, cost_per_buffer, workload_mat):
    print(f"\nstep: {i}")
    print(f"state: {x.ravel().astype(int)}")
    print(f"instantaneous cost: {np.round(cost_per_buffer.T @ x)}")
    print(f"w: {np.squeeze(workload_mat @ x)}")


def simulate_fluid_model(env: ControlledRandomWalk,
                         num_wl_vec: int,
                         mpc: bool) -> None:
    """
    Simulate fluid model. This is very useful when comparing with the trajectory of the stochastic
    and integer system.

    :param env: CRW environment object.
    :param num_wl_vec: Number of workload vectors to keep in the workload matrix.
    :param mpc: Flag indicating if using MPC in the simulation.
    :return: None.
    """
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)
    print(f"load: {load}")
    print(f"workload_mat: {workload_mat}")

    strategic_idling_params = StrategicIdlingParams()
    si_object_w_star = StrategicIdlingCoreHorizon(workload_mat,
                                                  load,
                                                  env.cost_per_buffer,
                                                  env.model_type,
                                                  0,
                                                  strategic_idling_params)

    initial_state = env.state_initialiser.initial_state
    target_w_star = si_object_w_star.get_allowed_idling_directions(initial_state).w_star
    min_drain_time = compute_min_drain_time(target_w_star, load)

    n_steps = int(np.ceil(1 / (1-np.max(load))))
    horizon = int(np.maximum(np.ceil(min_drain_time / n_steps), 1))
    si_object = StrategicIdlingCoreHorizon(workload_mat,
                                           load,
                                           env.cost_per_buffer,
                                           env.model_type,
                                           horizon,
                                           strategic_idling_params)

    if mpc:
        mpc_policy = FeedbackMipFeasibleMpcPolicy(env.constituency_matrix,
                                                  env.job_generator.buffer_processing_matrix)
        num_activities = env.constituency_matrix.shape[1]
        total_actions = np.zeros((num_activities, 1))
        total_fluid = np.zeros((num_activities, 1))

    policy_object = BigStepWBoundPolicy(env.cost_per_buffer,
                                        env.constituency_matrix,
                                        env.job_generator.demand_rate,
                                        env.job_generator.buffer_processing_matrix,
                                        workload_mat,
                                        'cvx.CPLEX')
    x = np.array(initial_state, dtype=float)

    cum_cost = 0
    print("Start simulation")
    for i in range(n_steps):

        si_output = si_object.get_allowed_idling_directions(x)

        w_bound = si_output.w_star

        no_penalty_grad = np.zeros_like(x)
        z_star, _ = policy_object.get_policy(x, no_penalty_grad, w_bound, horizon)
        print(f"z_star: {np.squeeze(z_star)}")

        if mpc:
            sum_actions = np.ceil(z_star * horizon)
            total_fluid += sum_actions
            for h in range(horizon):
                action = mpc_policy.generate_actions_with_feedback(sum_actions, x)
                x += policy_object.buffer_processing_matrix @ action + env.job_generator.demand_rate
                cum_cost += env.cost_per_buffer.T @ x
                print_action_update(total_fluid, total_actions, action, sum_actions)
                sum_actions -= action
                total_actions += action
                print_state_update(i * horizon + h, x, env.cost_per_buffer, workload_mat)
        else:
            x_new = x + ((policy_object.buffer_processing_matrix @ z_star
                          + env.job_generator.demand_rate) * horizon)
            cum_cost += env.cost_per_buffer.T @ ((x_new + x) / 2 * horizon)
            x = x_new
            print_state_update(i, x, env.cost_per_buffer, workload_mat)

    print(f"final state: {x.ravel().astype(int)}")
    print(f"total cost {cum_cost / 1e3}")


def cli_simulate_fluid():
    """
    Command line interface for "simulate_fluid".
    The script takes three command line parameters:
    --env_name: Name of the environment as described in scenarios.py.
    --num_wl_vec: Number of workload vectors to be considered.
    --env_param_json_file: Path to json file with the environment parameters.
    --mpc: Indicates whether we should simulate MPC, i.e., integer actions.

    For example, the script can be launched as follows following:
        python /project_path/snc/simulation/fluid_simulator.py \
            --num_wl_vec 4 \
            --env_name simple_link_constrained_model \
            --env_param_json_file /project_path/.../simple_link_routing.json
    """
    np.set_printoptions(precision=3, suppress=True)
    parse_args = parse_arg_path()
    if parse_args.env_name:
        if parse_args.env_param_json_file:
            env_param = load_env_params(parse_args.env_param_json_file)  # Load json parameters.
        else:
            env_param = None  # Use default parameters.
        env = load_env(parse_args.env_name, env_param)
        simulate_fluid_model(env, parse_args.num_wl_vec, parse_args.mpc)
    else:
        print("You must provide --env_name command line parameter.")


if __name__ == "__main__":
    cli_simulate_fluid()
