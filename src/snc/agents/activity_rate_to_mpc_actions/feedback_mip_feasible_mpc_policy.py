import cvxpy as cvx
import numpy as np
from typing import Optional, List, Tuple

from src.snc import ActionMPCPolicy
from src.snc.agents.solver_names import SolverNames
from src import snc as types


class FeedbackMipFeasibleMpcPolicy(ActionMPCPolicy):
    def __init__(
            self,
            constituency_matrix: types.ConstituencyMatrix,
            buffer_processing_matrix: types.BufferMatrix,
            mpc_seed: Optional[int] = None,
            nonidling_penalty_coeff: float = 1e5,
            mip_solver: str = "cvx.CPLEX"
    ) -> None:
        """
        Obtain feasible binary actions from activity rates with feedback on how many actions have
        been performed so far for a given horizon. The actions are obtained as the result of a mixed
        integer program that matches the activity rates after some horizon, while ensuring that
        the matching is done in a balanced manner. The feedback allows to count which actions remain
        underperformed so that they can be taken more often. Feasible actions refer to those that
        drain nonempty buffers and satisfy any other problem specific constraint. Indeed, the main
        benefit of this class over `FeedbackStationaryFeasibleMpcPolicy` is that it allows to handle
        complex action constraints easily.

        :param constituency_matrix: Constituency matrix from environment.
        :param buffer_processing_matrix: Buffer processing matrix from environment.
        :param nonidling_penalty_coeff: Penalty coefficient.
        :param mip_solver: Mixed integer program (MIP) solver.
        :return: None.
        """
        assert nonidling_penalty_coeff >= 0
        self.nonidling_penalty_coeff = nonidling_penalty_coeff
        assert mip_solver in SolverNames.CVX
        self.mip_solver = mip_solver

        super().__init__(constituency_matrix, mpc_seed)

        self.buffer_processing_matrix = buffer_processing_matrix
        self.buffer_draining_matrix = np.minimum(0, np.floor(buffer_processing_matrix))
        self.abs_buffer_draining_matrix = np.abs(self.buffer_draining_matrix)

        (
            self._mip_problem,
            self._sum_actions,
            self._state,
            self._action,
            self._nonidling_constituency_mat,
            self._nonidling_ones,
            self._bias_counter
        ) = self.create_mip()

        self._bias_counter.value = np.zeros((self.num_activities, 1))

        self.actions_drain_each_buffer = self.get_actions_drain_each_buffer()

    def create_mip(self, bias_penalty_coeff: float = 1e-3):
        """
        Create MIP that will be solved to obtain the action at every iteration.

        :param bias_penalty_coeff: Penalty coefficient to balance how actions are chosen.
        :return: Tuple with problem variables and parameters:
            - mip_problem: CVX problem.
            - sum_actions: CVX parameter with feedback on how many actions of each kind remain to be
                taken for the current horizon.
            - state: CVX parameter representing the current state. Needed to ensure feasibility.
            - action: CVX variable with action to be taken by the agent.
            - nonidling_constituency_mat: CVX parameter representing a modified constituency matrix
                with resources that shouldn't idle, so that they can switch to other actions in the
                case that the preferred ones are infeasible.
            - nonidling_ones: CVX parameter used to ensure nonidling resources.
            - bias_counter: CVX parameter to balance the order in which activities are taken.
        """
        num_buffers = self.buffer_processing_matrix.shape[0]
        # Parameters
        sum_actions = cvx.Parameter((self.num_activities, 1))
        state = cvx.Parameter((num_buffers, 1))
        nonidling_constituency_mat = cvx.Parameter((self.num_resources, self.num_activities))
        nonidling_ones = cvx.Parameter((self.num_resources, 1))
        bias_counter = cvx.Parameter((self.num_activities, 1))
        # Variable
        action = cvx.Variable((self.num_activities, 1), boolean=True)

        nonidling_penalty = cvx.sum(nonidling_ones - nonidling_constituency_mat @ action)
        objective = cvx.Minimize(
            cvx.sum(sum_actions - cvx.multiply(sum_actions, action))
            + self.nonidling_penalty_coeff * nonidling_penalty
            + bias_penalty_coeff * cvx.sum(bias_counter - cvx.multiply(bias_counter, action))
        )
        constraints = [
            state + self.buffer_draining_matrix @ action >= 0,  # Nonempty buffers.
            self.constituency_matrix @ action <= 1  # Feasible actions (defined boolean above).
        ]
        mip_problem = cvx.Problem(objective, constraints)
        return (
            mip_problem,
            sum_actions,
            state,
            action,
            nonidling_constituency_mat,
            nonidling_ones,
            bias_counter
        )

    def get_actions_drain_each_buffer(self, tol: float = 1e-3) -> List[np.ndarray]:
        """
        Get all actions that drain each buffer.

        :param tol: tolerance to check if an action drains a buffer.
        :return List of length equal to the number of buffers, whose i-th element contains an array
            with the indexes of the activities draining the i-th buffer.
        """
        actions_drain_each_buffer = list()
        for b in self.buffer_draining_matrix:
            actions_drain_each_buffer.append(np.where(b <= - (1 - tol))[0])
        return actions_drain_each_buffer

    def update_bias_counter(
            self,
            state: types.StateSpace,
            action: types.ActionSpace,
            sum_actions: types.ActionSpace
    ) -> None:
        """
        In order to remove systematic bias when the MIP chooses solutions, we count the number of
        times that each action has been dismissed when it had the same right to be chosen than
        others. We first identify buffers with less items than the amount that could be drained if
        all actions for which sum_actions > 0 would be performed at the same time. For each of this
        conflicting buffers, we identify which actions where dismissed. We reduce the bias counter
        for actions that were chosen and increase for those which were dismissed. We make the
        counter nonnegative to give only positive incentive to take dismissed actions (no need to
        add negative bias to the chosen actions). Finally, we reset the counter for those actions
        that are not required anymore (sum_actions = 0), so that there is no incentive in taking
        them.

        :param state: Current state.
        :param action: Current action.
        :param sum_actions: Number of times each action remains to be taken in the current horizon.
        :return: None.
        """
        counter = self._bias_counter.value
        dismissed_actions_counter = np.zeros((self.num_activities, 1))
        unit_sum_actions = np.minimum(1, sum_actions)

        chosen_actions = np.where(action > 0)[0]
        counter[chosen_actions] -= 1  # Reduce bias (it can become negative bias).

        conflicting_buffers = state < self.abs_buffer_draining_matrix @ unit_sum_actions
        for i, conflicting_b in enumerate(conflicting_buffers):
            if conflicting_b:
                dismissed_actions = np.setdiff1d(self.actions_drain_each_buffer[i], chosen_actions)
                dismissed_actions_counter[dismissed_actions] += 1
        counter += dismissed_actions_counter  # Increase bias.

        counter = np.maximum(0, counter)  # Make it nonnegative.
        counter = np.multiply(counter, unit_sum_actions)  # Reset actions not required anymore.
        self._bias_counter.value = counter

    def get_nonidling_resources(self,
                                sum_actions: types.ActionSpace
                                ) -> Tuple[types.ConstituencyMatrix, types.ColVector]:
        """
        First, it gets the number of times each action remain to be taken in the current horizon.
        Second, it identifies which resources they belong to, so that we assume these resources
        shouldn't idle. Finally, it returns a matrix and a vector that allow to represent nonidling
        constraint for these resources in the case that the preferred actions are infeasible.
        TODO: Currently it sets nonidling all resources with sum_actions >0 for at least one of its
         activities. This is a very aggressive approach. We will be less aggressive by setting the
         set of resources that shouldn't idle as a subset of the resources that have been identified
         as such by `BigStepPolicy`. This has to be done for both Hedging and GTO idling strategies.

        :param sum_actions: Number of times each action remains to be taken in the current horizon.
        :return: (nonidling_constituency_mat, nonidling_ones)
            - nonidling_constituency_mat: Matrix with rows corresponding to nonidling resources
                equal to those of the constituency matrix, and zero everywhere else.
            - nonidling_ones: Vector with ones in the rows corresponding to nonidling resources and
                zero everywhere else.
        """
        nonidling_constituency_mat = np.zeros_like(self.constituency_matrix)
        nonidling_ones = np.zeros((self.num_resources, 1))

        ind_nonidling = np.where(self.constituency_matrix @ sum_actions > 0)[0]
        nonidling_constituency_mat[ind_nonidling] = self.constituency_matrix[ind_nonidling]
        nonidling_ones[ind_nonidling] = 1
        return nonidling_constituency_mat, nonidling_ones

    def generate_actions_with_feedback(self,
                                       sum_actions: types.ActionSpace,
                                       state: types.StateSpace) -> types.ActionProcess:
        """
        Gets the number of times each action remain to be taken in the current horizon and current
        state and returns the binary action to be taken.

        :param sum_actions: Number of times each action remains to be taken in the current horizon.
        :param state: Current state.
        :return: Binary action to be taken.
        """
        self._sum_actions.value = sum_actions
        self._state.value = state
        self._nonidling_constituency_mat.value, self._nonidling_ones.value \
            = self.get_nonidling_resources(sum_actions)

        self._mip_problem.solve(solver=eval(self.mip_solver), warm_start=True)
        action = self._action.value
        self.update_bias_counter(state, action, sum_actions)
        return action

    def obtain_actions(self, **kwargs) -> types.ActionProcess:
        """
        This method implements the abstract method from super class `ActionMPCPolicy`.
        It first gathers the feedback information namely the number of times each activity has to
        be performed (i.e. 'sum_actions') and the current state ('state'). Then, it calls its own
        method to return a single action vector.

        :return: actions: binary indicator matrix with number of rows equal to the number of
            non-idling activities (i.e. num of columns of the constituency matrix), and number of
            columns equal to number of time steps to perform MPC.
        """
        assert 'mpc_variables' in kwargs, "Ensure MPC variables dict is passed"
        assert 'sum_actions' in kwargs["mpc_variables"], \
            "Number of times each activity has to be performed ('sum_actions') is required to be " \
            "passed as parameter."
        assert 'state' in kwargs, "Current state ('state') is required to be passed as parameter."

        actions = self.generate_actions_with_feedback(
            kwargs["mpc_variables"]['sum_actions'],
            kwargs['state']
        )
        return actions
