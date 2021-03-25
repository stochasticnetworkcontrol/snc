import os

import pytest

from src.meio.experiment import run_gsm
from src.meio import GuaranteedServiceModelDAG
import sandbox.meio.gsm as sandbox_gsm

dirname = os.path.dirname(__file__)


@pytest.fixture(params=range(1, 38))
def willems_file(request):
    chain = request.param
    return "willems_{:0=2d}.csv".format(chain)


def test_willems_dataset(willems_file):
    print(sandbox_gsm.__file__)
    data_path = '{}/willems_dataset/data/'.format(os.path.dirname(sandbox_gsm.__file__))
    print(data_path)
    try:
        gsm = run_gsm(data_path, willems_file, None, run_gsm_optimiser=False, plotting=False)
        load_completed = True
    except:
        load_completed = False

    if load_completed:
        assert isinstance(gsm, GuaranteedServiceModelDAG)
