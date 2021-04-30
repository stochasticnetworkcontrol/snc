import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.specs.distribution_spec import DistributionSpec

from snc.agents.rl.multi_headed_softmax_policy import (
    MultiHeadedCategoricalActionNetwork,
    OneHotCategoricalProjectionNetwork
)


def test_initialisation_single_head():
    """
    Test network initialisation with a single action subspace i.e. one head.
    """
    # Set up some (IO) space shapes.
    num_actions = 5
    observation_dim = 10
    # Use TensorSpecs to be compatible with TensorFlow.
    input_tensor_spec = tf.TensorSpec(
        shape=tf.TensorShape((observation_dim,)),
        dtype=tf.dtypes.float32,
        name="input"
    )
    output_tensor_spec = BoundedTensorSpec(
        shape=tf.TensorShape((5,)),
        dtype=tf.dtypes.float32,
        minimum=0,
        maximum=1,
        name="action"
    )
    # Instantiate the network.
    network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec,
        output_tensor_spec,
        action_subspace_dimensions=(num_actions,),
        hidden_units=(64,)
    )
    # Ensure that the network has set up some layers.
    assert hasattr(network, "_shared_layers") and network._shared_layers is not None
    del network


def test_initialisation_multiple_heads():
    """
    Test network initialisation with multiple action subspaces i.e. multiple heads.
    """
    # Set up the (IO) space shapes.
    observation_dim = 10
    # Ensure that the action subspace dimensions sum to the overall num_actions
    action_subspace_dimensions = (3, 5)
    input_tensor_spec = tf.TensorSpec(
        shape=tf.TensorShape((observation_dim,)),
        dtype=tf.dtypes.float32,
        name="input"
    )
    output_tensor_spec = (BoundedTensorSpec(shape=tf.TensorShape((3,)), dtype=tf.dtypes.float32,
                                            name="action_subspace_1", minimum=0, maximum=1),
                          BoundedTensorSpec(shape=tf.TensorShape((5,)), dtype=tf.dtypes.float32,
                                            name="action_subspace_2", minimum=0, maximum=1))
    # Use TensorSpecs to be compatible with TensorFlow.
    network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec,
        output_tensor_spec,
        action_subspace_dimensions=action_subspace_dimensions,
        hidden_units=(64,)
    )
    # Ensure that the network has set up some layers.
    assert hasattr(network, "_shared_layers") and network._shared_layers is not None
    del network


def test_forward_pass_single_head():
    """
    Test a forward pass through a single headed action network.
    """
    # Set up the network as in the single-headed test above.
    num_actions = 5
    observation_dim = 10
    input_tensor_spec = tf.TensorSpec(
        shape=tf.TensorShape((observation_dim,)),
        dtype=tf.dtypes.float32,
        name="input"
    )
    output_tensor_spec = BoundedTensorSpec(
        shape=tf.TensorShape((5,)),
        dtype=tf.dtypes.float32,
        minimum=0,
        maximum=1,
        name="action"
    )
    network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec,
        output_tensor_spec,
        action_subspace_dimensions=(num_actions,),
        hidden_units=(64,)
    )
    # Test that zeros as input yields zeros as output. This follows from the biases being
    # initialised to zero.
    zeros_input = np.zeros((1, observation_dim))
    zeros_output = network(zeros_input, step_type=None)[0].logits
    assert np.all(zeros_output == 0)
    # Test that random inputs yield non-zero outputs.
    random_input = np.random.random((1, observation_dim))
    random_output = network(random_input, step_type=None)[0].logits
    assert np.all(random_output != 0)


def test_forward_pass_multiple_heads():
    """
    Test a forward pass through a multi-headed action network.
    """
    # Set up the network as in the multi-headed test above.
    batch_size = 1
    observation_dim = 10
    action_subspace_dimensions = (3, 5)
    input_tensor_spec = tf.TensorSpec(
        shape=tf.TensorShape((observation_dim,)),
        dtype=tf.dtypes.float32,
        name="input"
    )
    output_tensor_spec = (BoundedTensorSpec(shape=tf.TensorShape((3,)), dtype=tf.dtypes.float32,
                                            name="action_subspace_1", minimum=0, maximum=1),
                          BoundedTensorSpec(shape=tf.TensorShape((5,)), dtype=tf.dtypes.float32,
                                            name="action_subspace_2", minimum=0, maximum=1))
    network = MultiHeadedCategoricalActionNetwork(
        input_tensor_spec,
        output_tensor_spec,
        action_subspace_dimensions=action_subspace_dimensions,
        hidden_units=(64,)
    )
    # Test that zeros input yields zeros output as per the biases being zero.
    # Also test that the network returns values for each head and that the shapes of the outputs
    # are as we would expect. We check logits as these are the network's raw outputs.
    zeros_input = np.zeros((1, observation_dim))
    zeros_output = network(zeros_input, step_type=None)[0]
    assert len(zeros_output) == 2
    assert zeros_output[0].logits.shape == (batch_size, 1, 3)
    assert zeros_output[1].logits.shape == (batch_size, 1, 5)
    assert np.all(zeros_output[0].logits == 0) and np.all(zeros_output[1].logits == 0)
    # Perform the same tests with random inputs ensuring non-zero outputs.
    random_input = np.random.random((1, observation_dim))
    random_output = network(random_input, step_type=None)[0]
    assert len(random_output) == 2
    assert random_output[0].logits.shape == (batch_size, 1, 3)
    assert random_output[1].logits.shape == (batch_size, 1, 5)
    assert np.all(random_output[0].logits != 0) and np.all(random_output[1].logits != 0)


def test_one_hot_categorical_projection_network():
    """
    Test the networks used as action heads.
    This tests initialisation and the forward pass.
    """
    # Set up for a single action head with 5 actions in the subspace.
    num_actions = 5
    sample_spec = BoundedTensorSpec(
        shape=tf.TensorShape((num_actions,)),
        dtype=tf.dtypes.float32,
        minimum=0,
        maximum=1,
        name="action"
    )
    action_head = OneHotCategoricalProjectionNetwork(sample_spec, num_actions)
    # Test the initialisation.
    assert hasattr(action_head, "_projection_layer") and action_head._projection_layer is not None
    assert hasattr(action_head, "_output_spec") and isinstance(action_head._output_spec,
                                                               DistributionSpec)
    # Test the forward pass (assuming a final output of the shared layers of dimension 64).
    inputs = tf.convert_to_tensor(np.random.random((1, 100, 64)))
    num_batch_dims = 2
    action_dist, _ = action_head(inputs, num_batch_dims)
    assert isinstance(action_dist, tfp.distributions.OneHotCategorical)
    assert action_dist.event_shape == num_actions
    # Ensure that there are two trainable weights, the weights matrix and a bias of a single linear
    # layer.
    assert len(action_head.trainable_weights) == 2
    assert action_head.trainable_weights[0].shape == (64, 5)
    assert action_head.trainable_weights[1].shape == (5,)


def test_one_hot_categorical_projection_init_errors():
    """
    Test that the action head fails to initialise if not provided with an unbounded TensorSpec.
    """
    num_actions = 5
    sample_spec = tf.TensorSpec(
        shape=tf.TensorShape((num_actions,)),
        dtype=tf.dtypes.float32,
        name="action"
    )
    pytest.raises(ValueError, OneHotCategoricalProjectionNetwork, sample_spec, num_actions)
