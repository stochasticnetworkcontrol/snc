import pytest
import tensorflow as tf
from _pytest.nodes import Item
from tf_agents.system.default.multiprocessing_core import handle_test_main


def multiprocessing(fn):
    def wrapped():
        handle_test_main(fn)

    return wrapped


def pytest_runtest_call(item: Item):
    item.runtest = multiprocessing(item.runtest)
    return item
