from typing import Dict, Union, Tuple, DefaultDict, Set, Optional, Callable

# Type aliases

# aliases for generic and tree GSM
Stages_Labels = Dict[str,int] # str is stage_id and int is the labeling order
                                       # in DP algorithm
Labels_Stages = Dict[int,str] # inverse if Stages_Labels
Stages_Depths = Dict[str,int] # str in stage_id and int is relative depth of the stage
Depths_Stages = DefaultDict[int,Set[str]] # inverse of Stages_Depths
Labels_Parents = Dict[int,Optional[str]] # int is stage's label in DP algorithm and its parent is
                                         #  either stage_id or None for root stage

Sub_Prob_Cost_Func = Callable[[int],float] # int is internal or external service time value
                                           # and return float is is the minimum cost
                                           # of corresponding sub-problem

Cumul_Min_Sub_Prob_Cost_Func = Callable[[int],Tuple[float,int]] # int is the internal or external
                                                          # service time and return type is a tulple
                                                           #(min_cost,service time at min_cost)

# aliases for CoC GSM
Min_Cluster_Cost_Func = Callable[[int,int],float] # input ints are values for 's' and 'si'
                                                  # and output is min_cost for the cluster
                                                  # subproblem



# aliases for GSM policies and solutions
GSM_Stage_Policy = Dict[str,int] # str is either "s" or "si", int is service time setting
GSM_Policy = Dict[str,GSM_Stage_Policy] # str is stage_id
GSM_Constraints = Dict[Tuple[str,str],int] # tuple is (stage_id,"max_s" or "min_si"),
                                           # int is the bound value
GSM_Unsatisfied_Constraints = Dict[Tuple[str,str],Tuple[int,int]]
