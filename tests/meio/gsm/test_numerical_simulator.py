import numpy as np
import os.path
from scipy.stats import poisson
from meio.gsm.utils import read_supply_chain_from_txt
from meio.experiment.numerical_simulator import simulate, compute_base_stocks, truncate_and_conserve


def test_numerical_simulator():
    n = 100000  # length of demand simulation

    lam = 10
    sla = 0.95

    # below is the explicit implementation of two stage system
    # index 1 stands for "Demand" stage
    # index 2 stands for "Dist" stage

    # simulate demand history
    np.random.seed(seed=8675309)
    d = np.random.poisson(size=n, lam=lam)

    # service times
    s_1 = 0
    s_2 = 3

    # internal service times
    si_1 = s_2
    si_2 = 0

    # lead times
    l_1 = 0
    l_2 = 30

    # net replenishment times
    tau_1 = si_1+l_1-s_1
    tau_2 = si_2+l_2-s_2

    # servicing history
    s_1_q = np.zeros(n+s_1+1)
    s_1_q[-n:] = d
    # replenishment history
    r_1_q = np.zeros(n+si_1+l_1+1)
    r_1_q[-n:] = d
    r_1_q_indep = r_1_q.copy()  # copy for the indep_simulation

    s_2_q = np.zeros(n+s_2+1)
    s_2_q[-n:] = d
    r_2_q = np.zeros(n+si_2+l_2+1)
    r_2_q[-n:] = d

    # basestocks
    b_1 = poisson.ppf(sla,tau_1 * lam)
    b_2 = poisson.ppf(sla,tau_2 * lam)

    # inventory position histories
    i_1_q_indep = b_1+np.cumsum(r_1_q_indep[:len(s_1_q)]-s_1_q)
    i_2_q_indep = b_2+np.cumsum(r_2_q[:len(s_2_q)]-s_2_q)

    ref_indep_corr = np.corrcoef(i_1_q_indep[100:n],i_2_q_indep[100:n])[0,1]

    # decoupled stockout rates
    indep_demand_stockout_rate = np.mean(i_1_q_indep < 0)
    indep_dist_stouckout_rate = np.mean(i_2_q_indep < 0)

    # compute actual replenishments from Dist to Demand
    pos = i_2_q_indep.copy()
    neg = -i_2_q_indep.copy()
    pos[pos<0] = 0
    neg[neg<0] = 0

    required_r = neg[:len(s_2_q)-1]+s_2_q[1:]
    available_r = pos+r_2_q[1:len(pos)+1]
    r_1_q_actual = np.minimum(available_r[:len(required_r)],required_r)
    r_1_q[-n:] = r_1_q_actual[-n:]

    i_1_q_casc = b_1+np.cumsum(r_1_q[:len(s_1_q)]-s_1_q)
    casc_demand_stockout_rate = np.mean(i_1_q_casc < 0)

    ref_casc_corr = np.corrcoef(i_1_q_casc[100:n],i_2_q_indep[100:n])[0,1]

    # now lets use the simulator to replicate the stats above
    data_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(data_path, "../../../meio/experiment/basic_serial_network_config.txt")
    stages = read_supply_chain_from_txt(path)
    policy = {"Dist": {"s": 3, "si": 0}, "Demand": {"s": 0, "si": 3}}

    base_stocks = compute_base_stocks(stages,policy,lam,sla)
    assert base_stocks["Demand"] == 39
    assert base_stocks["Dist"] == 297

    indep_inv_histories = simulate(stages,policy,base_stocks,{},d,stockout_stages=[])
    casc_inv_histories = simulate(stages,policy,base_stocks,{},d,stockout_stages=None)

    np.testing.assert_almost_equal(np.mean(indep_inv_histories["Demand"]<0),
                                   indep_demand_stockout_rate,5)
    np.testing.assert_almost_equal(np.mean(indep_inv_histories["Dist"]<0),
                                   indep_dist_stouckout_rate,5)

    indep_corr = np.corrcoef(indep_inv_histories["Demand"][100:n],
                             indep_inv_histories["Dist"][100:n])[0][1]
    np.testing.assert_almost_equal(indep_corr,ref_indep_corr,5)

    np.testing.assert_almost_equal(np.mean(casc_inv_histories["Demand"]<0),0.06437,5)
    np.testing.assert_almost_equal(np.mean(casc_inv_histories["Dist"]<0),0.05176,5)
    casc_corr = np.corrcoef(casc_inv_histories["Demand"][100:n],
                            casc_inv_histories["Dist"][100:n])[0][1]
    np.testing.assert_almost_equal(casc_corr,ref_casc_corr,5)


def test_truncate_and_conserve():
    np.random.seed(seed=8675309)
    for _ in range(10):
        n = 100000
        lam=10
        max_capacity=12
        sequence = np.random.poisson(size=n, lam=lam)
        trun_seq = truncate_and_conserve(sequence,max_capacity)
        assert trun_seq.max() == max_capacity
        diff = sequence-max_capacity
        trun_diff = trun_seq - max_capacity
        cum_diff = np.cumsum(diff)
        cum_trun_diff = np.cumsum(trun_diff)
        check_points = np.where(trun_diff < 0)[0]
        if len(check_points) == 0:
            continue
        for i in check_points:
            assert cum_diff[i] == cum_trun_diff[i]
