from collections import OrderedDict
from enum import Enum, unique
import numpy as np
from typing import Dict, Union, Iterator, Type, Tuple

from meio.gsm.dag_gsm import GuaranteedServiceModelDAG
from meio.gsm.tree_gsm import Stage, GuaranteedServiceModelTree, GuaranteedServiceModel


def create_supply_chain_network_from_iterator(supply_chain: Iterator) -> Dict[str, Stage]:
    """
    Method to read from an iterator and initialise the dictionary of stages forming the supply chain

    The data file is a comma separated txt with one stage per row and
    the following field names for columns:
    - stage_id
    - lead_time
    - max_s_time
    - cost_rate
    - cap_constraint
    - risk_pool
    - ext_demand_mean
    - ext_demand_std
    - ext_demand_thres
    - up_stages (up_stage_1,phi_1,up_stage_2,phi_2,...,up_stage_n,phi_n)

    :returns: dictionary of stage objects with keys being the unique ids of the stages
    """
    stage_configs = OrderedDict()  # type: OrderedDict[str,Dict]
    u_stages = {}  # type: Dict[str,Dict[str,int]]
    for i, row in enumerate(supply_chain):
        if i == 0:
            continue
        line = row.strip("\n").split(",")
        stage_config = {}  # type: Dict[str, Union[str, int, float, Dict[str, int]]]
        stage_config['_id'] = stage_id = line[0]
        stage_config['lead_time'] = int(float(line[1]))
        stage_config["max_s_time"] = int(line[2]) if line[2] != "" else np.inf
        stage_config['added_cost'] = float(line[3])
        if line[4] != "":
            stage_config['risk_pool'] = int(line[4])
        if line[5] != "":
            stage_config['is_ext_demand_stage'] = True
            stage_config['demand_mean'] = float(line[5])
            stage_config['demand_std'] = float(line[6])
            stage_config['demand_thres'] = float(line[7])

        up_stages = {}  # type: Dict[str, int]
        up_stages_list = line[8:]
        if len(up_stages_list) > 1:
            for s in range(0, len(up_stages_list), 2):
                up_stage_id = str(up_stages_list[s])
                phi = int(up_stages_list[s+1])
                up_stages[up_stage_id] = phi

        stage_config["up_stages"] = u_stages[stage_id] = up_stages

        stage_configs[stage_id] = stage_config

    d_stages = {stage_id:{} for stage_id in u_stages}  # type: Dict[str, Dict[str, int]]
    for stage_id,up_stages in u_stages.items():
        for up_stage_id,phi in up_stages.items():
            d_stages[up_stage_id][stage_id] = phi

    for stage_id in stage_configs:
        stage_configs[stage_id]["up_stages"] = u_stages[stage_id]
        stage_configs[stage_id]["down_stages"] = d_stages[stage_id]

    stages = OrderedDict((stage_id,Stage(**stage_config)) for stage_id, stage_config
                         in stage_configs.items())
    return stages


def read_supply_chain_from_txt(supply_chain_txt_file: str) -> Dict[str, Stage]:
    """
    Method to read from file and initialise the dictionary of stages forming the supply chain
    :returns: dictionary of stage objects with keys being the unique ids of the stages
    """
    with open(supply_chain_txt_file, "r") as f:
        stages = create_supply_chain_network_from_iterator(f)
    return stages


@unique
class GSM(Enum):
    Tree = 'Tree'  # Spanning tree
    CoC = 'CoC'   # Clusters of commonality
    DAG = 'DAG'   # Directed Acyclic graphs


def create_gsm_instance(gsm_type: GSM, supply_chain_filename: str) \
        -> Tuple[Dict[str, Stage], GuaranteedServiceModel]:
    """
    GSM Factory method.  Does not necessarily check for compatibility of given GSM type with
    the network topology described in the config file; any checking is a bonus of the Tree model
    construction.

    :param gsm_type:  The type of the GSM model (e.g. spanning tree or clusters of commonality).
    :param supply_chain_filename:  The name of the config file defining the topology and other
                                   parameters.
    :raise UnSupportedGSMException: Not all types of GSM can be created with this utility.
    :raise IncompatibleGraphTopology:  The specified type is inconsistent with the topology
                                       described in the config file.
    :raises: InconsistentGSMConfiguration:  Network topology labels not as expected.
    :return: The stages of the nework and the gsm model of the appropriate type, if config is
             compatible with what was asked for.
    """

    stages = read_supply_chain_from_txt(supply_chain_filename)
    creator = {
        'Tree': GuaranteedServiceModelTree,
        'DAG': GuaranteedServiceModelDAG
    }  # type: Dict[str, Type[GuaranteedServiceModel]]
    return stages, creator[gsm_type.value](stages)
