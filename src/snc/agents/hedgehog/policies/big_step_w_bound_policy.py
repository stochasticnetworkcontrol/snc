from typing import Tuple, Optional, Dict

import numpy as np
import cvxpy as cvx
from src import snc as types
from src.snc import clean_to_serializable


class BigStepWBoundPolicy:

    def __init__(self,
                 cost_per_buffer: types.StateSpace,
                 constituency_matrix: types.ConstituencyMatrix,
                 demand_rate: types.StateSpace,
                 buffer_processing_matrix: types.BufferMatrix,
                 workload_mat: types.WorkloadMatrix,
                 convex_solver: str = 'cvx.CPLEX',
                 norm1_penalty_flag: bool = False) -> None:
        """
        This variant of BigStepPolicy (will be inhereted through refactoring) computes fluid
        policy rates with end state after horizon timesteps constrained by upper bound values
        in workload space (w_bound). This is different from current approach in BigStepPolicy
        where a non-idling penalty is used to extend myopic fluid policy.

        :param cost_per_buffer: from the environment.
        :param constituency_matrix: from the environment.
        :param demand_rate: from the environment
        :param buffer_processing_matrix: from job generator (the environment).
        :param workload_mat: workload matrix.
        :param convex_solver: method to solve the LP.
        :param norm1_penalty_flag: boolean that specifies whether we want to compute the norm 1 of
            the calling nonidling penalty vector. This is useful for testing purposes, when
            comparing the solution of this method with that of an LP with nonidling constraints,
            while using random matrices.
        """
        self.cost_per_buffer = cost_per_buffer
        self.constituency_matrix = constituency_matrix
        self.demand_rate = demand_rate
        self.buffer_processing_matrix = buffer_processing_matrix
        self.workload_mat = workload_mat

        self._convex_solver = convex_solver
        self.norm1_penalty_flag = norm1_penalty_flag

        self.num_activities = constituency_matrix.shape[1]
        self.num_buffers = cost_per_buffer.shape[0]
        self.num_bottlenecks = self.workload_mat.shape[0]  # Number of dimensions in workload space.

        self.pushing_buffer_processing_matrix = np.maximum(0, self.buffer_processing_matrix)

        self.load_ph = None  # type: Optional[types.ResourceSpace]
        self.sigma_2_ph = None  # type: Optional[types.ResourceSpace]

        self._lp_problem, self._state, self._horizon, self._penalty_grad, \
            self._w_bound, self._z \
            = self.create_big_step_policy_nonidling_penalty_lp_program()

    @property
    def convex_solver(self):
        return self._convex_solver

    @convex_solver.setter
    def convex_solver(self, convex_solver: str):
        self._convex_solver = convex_solver

    def update_safety_stock_params(self, load_ph: types.ResourceSpace,
                                   sigma_2_ph: types.ResourceSpace) -> None:
        """
        :param load_ph: the load vector corresponding to physical resources.
        :param sigma_2_ph: the variance corresponding to physical resources.
        :return: None.
        """
        self.load_ph = load_ph
        self.sigma_2_ph = sigma_2_ph

    def create_big_step_policy_nonidling_penalty_lp_program(self):
        z = cvx.Variable((self.num_activities, 1), nonneg=True)
        h = cvx.Variable(nonneg=True)  # tolerance variable
        state = cvx.Parameter((self.num_buffers, 1))
        horizon = cvx.Parameter(nonneg=True)
        penalty_grad = cvx.Parameter((self.num_buffers, 1))
        w_bound = cvx.Parameter((self.num_bottlenecks, 1))

        cost_vec = self.cost_per_buffer.T @ self.buffer_processing_matrix \
                   + penalty_grad.T @ self.pushing_buffer_processing_matrix

        tol_penalty_coeff = 1e3
        tolerance_penalty = tol_penalty_coeff * np.sum(self.cost_per_buffer)

        # We aim to move in the direction (velocity) that minimises the cost. But note that we
        # can remove the demand rate term from the objective as it is just a constant that don't
        # influence the solution.
        obj_equation = cost_vec * z + tolerance_penalty * h
        objective = cvx.Minimize(obj_equation)

        constraints = [
            # Resource constraint.
            self.constituency_matrix @ z <= 1,
            # Nonnegative future state.
            state + (self.buffer_processing_matrix @ z + self.demand_rate) * horizon >= 0,
            # Future state in workload space not exceeding provided bound on workload values
            self.workload_mat @ (state + (self.buffer_processing_matrix @ z + self.demand_rate) \
                                 * horizon) - h <= w_bound]

        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, state, horizon, penalty_grad, w_bound, z

    def big_step_policy_nonidling_penalty(self, state: types.StateSpace,
                                          penalty_grad: types.ColVector,
                                          w_bound: types.WorkloadSpace,
                                          horizon: int):
        """
        Returns a schedule that is approximately optimal in the discounted cost sense for the
        number of time steps given by 'horizon' when starting at 'state'.

        :param state: current state of the environment.
        :param penalty_grad: the output of the safety stock.
        :param w_bound: provided bounds on workload values not to be exceeeded by fluid policy
        :param horizon: number of time steps that this policy should be performed.
        :return (z_star, opt_val):
            - z_star: matrix where columns are the actions for each of the horizon.
            - opt_val: value of the objective cost at z_star.
        """
        assert horizon >= 1, f"Horizon must be >= 1, but provided: {horizon}."

        # Update problem parameters
        self._state.value = state
        self._horizon.value = horizon
        self._penalty_grad.value = penalty_grad
        self._w_bound.value = w_bound

        opt_val = self._lp_problem.solve(solver=eval(self._convex_solver), warm_start=True)
        z_star = self._z.value
        return z_star, opt_val

    def get_policy(self,
                   state: types.StateSpace,
                   penalty_grad: types.ColVector,
                   w_bound: types.WorkloadSpace,
                   horizon: int) -> Tuple[types.ColVector, float]:
        """
        Returns a vector of activity rates that is approximately optimal in the discounted cost
        sense for the number of time steps given by 'horizon', when starting at 'state'.
        This method just calls 'big_step_policy_nonidling_penalty' and it is been included to be
        easily overloaded by other policies, like the pure feedback policy.

        :param state: current state of the environment.
        :param penalty_grad: the output of the safety stock.
        :param w_bound: workload destination after moving horizon timesteps.
        :param horizon: horizon for computing the activity rates.
        :return (z_star, opt_val):
            - z_star: policy as a vector with activity rates to be used for the given horizon.
            - opt_val: value of the objective cost at z_star.
        """
        z_star, opt_val = self.big_step_policy_nonidling_penalty(state, penalty_grad, w_bound,
                                                                 horizon)
        return z_star, opt_val

    def to_serializable(self) -> Dict:
        """Return a serializable object, that can be used by a JSON Encoder"""
        return clean_to_serializable(self)
