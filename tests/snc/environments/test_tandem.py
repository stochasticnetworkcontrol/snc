import numpy as np

from snc.environments import examples


def test_tandem_2_blocks():
    env = examples.tandem_demand_model(
        n=2,
        d=.3,
        mu=np.array([.7, .6, .5, .3]),
        cost_per_buffer=np.array([1, 1, 1, 1])[:, None],
        initial_state=np.array([3, 3, 3, 2])[:, None],
        capacity=np.ones((4, 1)) * np.inf,
        job_conservation_flag=True
    )
    assert (env.job_generator.buffer_processing_matrix == np.array([[-.7, 0, 0, .3],
                                                  [.7, -.6, 0, 0],
                                                  [0, .6, -.5, 0],
                                                  [0, 0, -.5, 0]])).all()
    assert (env._constituency_matrix == np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])).all()
    assert (env.job_generator.demand_rate == np.array([0, 0, 0, .3])[:, None]).all()
    assert (env.list_boundary_constraint_matrices[0] ==
            np.hstack((np.ones((1, 1)), np.zeros((1, 3))))).all()
    assert (env.list_boundary_constraint_matrices[1] ==
            np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 2))))).all()
    assert (env.list_boundary_constraint_matrices[2] == np.zeros((1, 4))).all()
    assert (env.list_boundary_constraint_matrices[3] == np.zeros((1, 4))).all()


def test_tandem_3_blocks():
    env = examples.tandem_demand_model(n=3, d=.3, mu=np.array([.8, .7, .6, .5, .3]),
                                       cost_per_buffer=np.array([1, 1, 1, 1, 1])[:, None],
                                       initial_state=np.array([3, 3, 3, 3, 2])[:, None],
                                       capacity=np.ones((5, 1)) * np.inf, job_conservation_flag=True)
    assert (env.job_generator.buffer_processing_matrix == np.array([[-.8, 0, 0, 0, .3],
                                                  [.8, -.7, 0, 0, 0],
                                                  [0, .7, -.6, 0, 0],
                                                  [0, 0, .6, -.5, 0],
                                                  [0, 0, 0, -.5, 0]])).all()
    assert (env._constituency_matrix == np.array([[1, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0],
                                             [0, 0, 1, 0, 0],
                                             [0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 1]])).all()
    assert (env.job_generator.demand_rate == np.array([0, 0, 0, 0, .3])[:, None]).all()
    assert (env.list_boundary_constraint_matrices[0] ==
            np.hstack((np.ones((1, 1)), np.zeros((1, 4))))).all()
    assert (env.list_boundary_constraint_matrices[1] ==
            np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 3))))).all()
    assert (env.list_boundary_constraint_matrices[2] ==
            np.hstack((np.zeros((1, 2)), np.ones((1, 1)), np.zeros((1, 2))))).all()
    assert (env.list_boundary_constraint_matrices[3] == np.zeros((1, 5))).all()
    assert (env.list_boundary_constraint_matrices[4] == np.zeros((1, 5))).all()
