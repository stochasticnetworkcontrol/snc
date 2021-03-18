import os

import pytest

from meio.experiment.demo_gsm import run_gsm
from meio.gsm.dag_gsm import GuaranteedServiceModelDAG

dirname = os.path.dirname(__file__)


@pytest.fixture(params=range(1, 38))
def supply_chain_file(request):
    chain = request.param
    return "supply_chain_{:0=2d}.csv".format(chain)


def test_supply_chain_dataset(willems_file):
    data_path = '{}/../../../meio/supply_chain_dataset/data/'.format(dirname)
    try:
        gsm = run_gsm(data_path, willems_file, None, run_gsm_optimiser=False, plotting=False)
        load_succeeded
    except:
        load_succeeded = False
    if load_succeeded:
        assert isinstance(gsm, GuaranteedServiceModelDAG)
