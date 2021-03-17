import tensorflow as tf
from random import Random
import os
import psutil

MAXINT = 2**32 - 1


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss


def test_memory_leaking():
    """
    Test to ensure that the memory leak is present.

    If this test fails then the issue has been resolved and this test as well as
    snc.experiment.experiment_utils.monkey_patch_tf_get_seed should be deleted.

    This should also then lead you to remove gorilla from the dependencies and remove all calls
    to snc.experiment.experiment_utils.monkey_patch_tf_get_seed

    The underlying TensorFlow issue is at https://github.com/tensorflow/tensorflow/issues/37252
    """
    m0 = get_memory_usage()
    random_state = Random()
    for _ in range(10000):
        a = random_state.randint(0, MAXINT)
        tf.Variable(a)
    m1 = get_memory_usage()
    assert m1 > m0
