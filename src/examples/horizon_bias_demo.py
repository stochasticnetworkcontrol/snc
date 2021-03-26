import cvxpy as cvx
import numpy as np

def main():
    """
    The purpose of this script is to demonstrate an evident bias which emerges in big step LP
    in conjunction with non-idling constraints (or penalties). A simple re-etrant line is taken
    as exemplary environment with high initial state located at the lowest effective cost.
    Big step horizon is taken to be 10% of the draining time.

    Two LP solutions are compared:one with non-idling constraint penalty imposed on resource 2
    and one where it is not. In the first case the obtained minimum solution to LP in the form of
    zeta_star results in the INCREASE of instanteneous cost after the horizon period. Since resource
    2 is forced to operate at is maximum capacity (never to idle) instantaneous cost increases
    because system deviates from the effective state, even if the system is draining overall.

    In case where non-idling penalty is not imposed on resource 2, the obtained zeta_star is
    consistent with both ensuring minimal draining time (since resource 1 is forced to non-idle)
    and ensuring that effective state is maintained towards the end of policy horizon. Consequently
    the instantaneous cost is reduced below the the initial value.

    Additional metrics such as inverse loading are displayed to corroborate the claim that in
    presence of non-idling constraints for too long horizon, LP results in the skewed solution
    for zeta_star, which does not maintain the minimum load with respect to arrivals rate.
    """

    # define network rates for a simple re-entrant line
    a = 0.33
    m1 = m3 = (2*a)*1.03
    m2 = a * 1.05

    print("\nHere are our simple re-entrant line processing rates")
    print(f"a = {a}")
    print("m1 = m3 = (2*a)*1.03")
    print("m2 = a * 1.05")
    print("We define the activity inverse loading as 'zeta_i*m_i/a' \
    \nFor example when resource 1 is working on activity 1 half of the time (zeta_1=0.5) \
    \nthe inverse loading of activity 1 is 0.5*m1/a = 0.5*2*a*1.03/a = 1.03")

    # specify re-entrant line topology matrices
    buffer_processing_matrix = np.array([[-m1,0,0],[+m1,-m2,0],[0,+m2,-m3]])
    # rows of the constituency matrix
    resource_1 = np.array([1,0,1])
    resource_2 = np.array([0,1,0])

    cost_per_buffer = np.array([1.5,1,2])[:,np.newaxis]

    demand_rate = np.array([a,0,0])[:,np.newaxis]

    # initial state corresponds to draining time of 100000
    x_0 = np.array([1000,0,0])[:,np.newaxis]

    horizon = 10000 # 0.1 of draining time

    # non-idling penalty coefficient
    kappa_w = 1e3

    # define fluid policy LP
    def solve_z_star(idle_resource_2):
        z = cvx.Variable((3, 1), nonneg=True)

        cost_vec = cost_per_buffer.T @ buffer_processing_matrix

        if idle_resource_2:
            penalty_nonidling_vec = cvx.abs(resource_1 @ z - 1)
        else:
            penalty_nonidling_vec = cvx.abs(resource_1 @ z - 1) + cvx.abs(resource_2 @ z - 1)

        penalty_nonidling = cvx.sum(penalty_nonidling_vec)

        obj_equation = cost_vec @ z + kappa_w * penalty_nonidling
        objective = cvx.Minimize(obj_equation)

        constraints = [
            # Resource constraint.
            resource_1 @ z <= 1,
            resource_2 @ z <= 1,
            # Nonnegative future state.
            x_0 + (buffer_processing_matrix @ z + demand_rate) * horizon >= 0]

        cvx.Problem(objective, constraints).solve()

        z_star = z.value
        return z_star



    print("\nHere is the initial state:")
    print(x_0.astype(int))

    print("\nFirst compute z_star for non_idling constraint active for resource 2. \
    \nAnd then apply this policy over the horizon duration to find new state.")
    z_star = solve_z_star(idle_resource_2=False)

    d_x = (buffer_processing_matrix @ z_star + demand_rate) * horizon

    x_new = x_0 + d_x

    print("\nHere is the new state:")
    print(x_new.astype(int))

    print("\nHere is the change in instantaneous cost if policy is executed over the whole horizon")
    d_c = cost_per_buffer.T @ d_x
    print(d_c)
    print("\nCost increased even though system is draining. \
    \nThat's because the system has deviated from effective state.")

    print("\nLooking at z_star rates:")
    print(z_star)
    print("We can see that activity 2 is not idling but activities 1 and 3 are not balanced, \
    \nmeaning that system has not been operating at the maximum draining capacity. \
    \nFor this network configuration resource 1 is the limiting bottleneck and operates at its \
    \nmaximum draining capacity when zeta_1 = zeta_3 = 0.5, i.e. when operation of activities \
    \n1 and 3 is balanced.")
    print("\nLooking at the inverse loading of activities we can see that activity 3 is operating \
    \nat a lower inverse loading than optimum (1.03). The value of 1.03 is optimal because we designed it to be \
    \nthis way by setting 'm1 = m3 = (2*a)*1.03', i.e. resource 1 in the long run cannot operate at \
    \ncapacity above its mean serving rate a*1.03")
    print(buffer_processing_matrix @ np.diag(z_star.ravel())/a)


    print("\nNow we compute z_star with non_idling constraint not active for resource 2")
    z_star = solve_z_star(idle_resource_2=True)

    d_x = (buffer_processing_matrix @ z_star + demand_rate) * horizon

    x_new = x_0 + d_x

    print("\nHere is the new state:")
    print(x_new.astype(int))

    print("\nHere is the change in instantaneous cost if policy is executed over the whole horizon")
    d_c = cost_per_buffer.T @ d_x
    print(d_c)
    print("\nCost decreased and new state is in the effective state")

    print("\nNow activity 2 (resource 2) is not working full time but instead balances the maximum \
    \nthroughput of resource 1, which is achieved when activities 1 and 3 are perfectly balanced \
    \n(zeta_1 = zeta_3 = 0.5)")
    print(z_star)

    print("\nLooking at activity inverse loading we can see that the flow through the network is \
    \nbalanced at the optimum inverse loading of 1.03")
    print(buffer_processing_matrix @ np.diag(z_star.ravel())/a)


if __name__=="__main__":
    main()
