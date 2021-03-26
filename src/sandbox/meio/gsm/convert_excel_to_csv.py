import pandas as pd
import numpy as np
import sys
import os

from scipy.stats import norm
from math import ceil

from collections import OrderedDict, defaultdict


def load_into_excel(filename):
    """This function returns an excel file in the current directory
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)
    return pd.ExcelFile(filename)


def init_network_description():
    network_description = {}
    network_description['total_stages'] = 0
    network_description['stages_with_stoch_lead'] = 0
    network_description['stages_demand'] = 0
    network_description['stages_with_broken_down_stoch_lead'] = 0
    return network_description


def parse_willems_graphs(excel_file, supply_chain_number, how_many_std_to_lead_time=0):
    """
    Parse the graphs loaded in the willems excel sheet

    :param excel_file: Excel file from Willems et al with all 38 supply chains
    :param supply_chain_number: which supply chain to load
    :param how_many_std_to_lead_time: how many standard deviations to add to the mean
    :return:
    """
    df1 = excel_file.parse("{}_SD".format(supply_chain_number))

    stage_configs = {}
    network_description = init_network_description()

    for r, row in df1.iterrows():
        network_description['total_stages'] += 1
        stage_config = OrderedDict()
        stage_config["stage_id"] = row["Stage Name"]
        # stochastic lead times
        if np.isnan(row['stdDev stageTime']):
            lead_time_std = 0
        else:
            lead_time_std = row['stdDev stageTime']
            network_description['stages_with_stoch_lead'] += 1

        if not np.isnan(row['stageTime_1']):
            network_description['stages_with_broken_down_stoch_lead'] += 1

        stage_config["lead_time"] = row["stageTime"] + how_many_std_to_lead_time * lead_time_std
        stage_config["max_s_time"] = row["maxServiceTime"]
        stage_config["cpst_rate"] = row["stageCost"]
        stage_config["risk_pool"] = 2
        stage_config["ext_demand_mean"] = row["avgDemand"]
        stage_config["ext_demand_std"] = row["stdDevDemand"]
        stage_config["ext_demand_thres"] = row["serviceLevel"]

        if str(stage_config["max_s_time"]) != "nan":
            stage_config["max_s_time"] = int(stage_config["max_s_time"])

        if str(stage_config["ext_demand_mean"]) != "nan":
            network_description['stages_demand'] += 1
            stage_config["risk_pool"] = "nan"
            stage_config["ext_demand_thres"] = "{:.3f}".format(
                norm.ppf(stage_config["ext_demand_thres"]))

        stage_configs[stage_config["stage_id"]] = stage_config

    up_stages = defaultdict(list)
    df1 = excel_file.parse("{}_LL".format(supply_chain_number))
    for r, row in df1.iterrows():
        stage_id = row[1]
        up_stages[stage_id].append(row[0])
    return stage_configs, up_stages, network_description


def write_to_csv(filename, stage_configs, up_stages):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)

    with open(filename, "w") as f:
        f.write("stage_id,lead_time,max_s_time,cost_rate,risk_pool,ext_demand_mean,ext_demand_std,ext_demand_thres,up_stages\n")
        for stage_config in sorted(stage_configs.values(), key=lambda x: x['stage_id']):
            line = ",".join(str(i) if str(i) !=
                            "nan" else "" for i in stage_config.values())
            for u_stage in up_stages[stage_config["stage_id"]]:
                line += ",{},1".format(u_stage)

            line += "\n"

            f.write(line)


def main():
    willems_file = "../../../meio/willems_dataset/data/MSOM-06-038-R2 Data Set in Excel.xls"
    data_xls = load_into_excel(willems_file)
    dest =  os.path.join("../../../meio/willems_dataset", "data")
    os.makedirs(dest, exist_ok=True)
    no_graphs_in_dataset = 38
    for i in range(1, no_graphs_in_dataset + 1):
        no = "{:02d}".format(i)  # '01', '02',.. '38'
        out_filename = os.path.join(dest,"willems_{}.csv".format(no))
        stage_configs, up_stages, _ = parse_willems_graphs(data_xls, no)
        write_to_csv(out_filename, stage_configs, up_stages)


if __name__ == "__main__":
    main()
