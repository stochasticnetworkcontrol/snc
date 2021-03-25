from src.snc.agents.hedgehog.policies import policy_utils


def get_allowed_activities(big_step_policy, num_bottlenecks, k_idling_set):
    nonidling_res = policy_utils.obtain_nonidling_bottleneck_resources(num_bottlenecks,
                                                                       k_idling_set)
    ind_forbidden_activities = big_step_policy.get_index_all_forbidden_activities(nonidling_res)
    return big_step_policy.get_allowed_activities_constraints(ind_forbidden_activities)
