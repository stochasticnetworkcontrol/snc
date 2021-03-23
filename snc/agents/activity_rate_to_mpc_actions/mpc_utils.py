
def check_num_time_steps(num_mpc_steps: int) -> None:
    assert isinstance(num_mpc_steps, int), "Number of MPC steps is not integer."
    assert num_mpc_steps >= 1, "Number of MPC steps is zero or negative."
