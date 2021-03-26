from typing import Tuple, List, Union, Optional, Any

from tf_agents.utils import nest_utils
from tf_agents.specs.tensor_spec import TensorSpec

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks.network import DistributionNetwork
from tf_agents.networks import utils as network_utils
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.specs import tensor_spec


class OneHotCategoricalProjectionNetwork(DistributionNetwork):
    """
    An action head which maps from hidden activations to a distribution over an action subspace.
    """

    def __init__(self,
                 sample_spec: tf.TensorSpec,
                 num_actions: int,
                 name: str = 'OneHotCategoricalProjectionNetwork') -> None:
        """
        Initialises a head of a multi-headed action network which will return a distribution over
        actions related to this head.

        :param sample_spec: A specification for the results of sampling from this head of the
            policy.
        :param num_actions: The dimensionality of this head's action space.
        :param name: A name fo this head.
        """
        output_shape = (1, num_actions)
        output_spec = self._output_distribution_spec(output_shape, sample_spec, name)
        super(OneHotCategoricalProjectionNetwork, self).__init__(
            # We don't need these, but base class requires them.
            input_tensor_spec=None,
            state_spec=(),
            output_spec=output_spec,
            name=name)
        self._projection_layer = tf.keras.layers.Dense(num_actions, activation=None)

        if not tensor_spec.is_bounded(sample_spec):
            raise ValueError('sample_spec must be bounded. Got: %s.' % type(sample_spec))

        self._sample_spec = sample_spec
        self._output_shape = tf.TensorShape(output_shape)

    @staticmethod
    def _output_distribution_spec(output_shape: Tuple[int, ...], sample_spec: tf.TensorSpec,
                                  network_name: str) -> DistributionSpec:
        """
        Determines the specification of the distribution this output head will produce.

        :param output_shape: The shape of actions to be produced.
        :param sample_spec: A specification for the results of sampling from this head of the
            policy.
        :param network_name: A name for the network.
        :return: A specification of the distribution that this action head will produce which is
            compatible with TensorFlow probability and TensorFlow Agents.
        """
        input_param_spec = {
            'logits':
                tensor_spec.TensorSpec(
                    shape=output_shape,
                    dtype=tf.float32,
                    name=network_name + '_logits')
        }

        return DistributionSpec(
            tfp.distributions.OneHotCategorical,
            input_param_spec,
            sample_spec=sample_spec,
            dtype=sample_spec.dtype)

    def call(self, inputs: tf.Tensor, batch_dims: int) -> tfp.distributions.OneHotCategorical:
        """
        Maps from a shared layer of hidden activations of the overall action net (inputs) to a
        distribution over actions for this head alone.

        :param inputs: The hidden activation from the final shared layer of the action network.
        :param batch_dims: The number of batch dimensions in the inputs.
        :return: A (OneHotCategorical) distribution over actions for this head.
        """
        # outer_rank is needed because the projection is not done on the raw observations so getting
        # the outer rank is hard as there is no spec to compare to.
        # BatchSquash is used to flatten and unflatten a tensor caching the original batch
        # dimension(s).
        batch_squash = network_utils.BatchSquash(batch_dims)
        # We project the logits via a linear transformation to the right dimension for the action
        # head.
        inputs = batch_squash.flatten(inputs)
        logits = self._projection_layer(inputs)
        logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
        logits = batch_squash.unflatten(logits)
        # We finally return the appropriate TensorFlow distribution.
        return self.output_spec.build_distribution(logits=logits)

    @property
    def projection_layer(self):
        return self._projection_layer


class MultiHeadedCategoricalActionNetwork(DistributionNetwork):
    def __init__(
            self,
            input_tensor_spec: TensorSpec,
            output_tensor_spec: TensorSpec,
            action_subspace_dimensions: Tuple[int, ...],
            hidden_units: Optional[Tuple[int]] = (64,)) -> None:
        """
        An action network for a TensorFlow Agent. Inherits from the tf_agents base network.
        This network takes in the state/observations from the environment and produces a set of
        distributions over one-hot actions in each action subspace. This is implemented using
        TensorFlow Probability (a `tfp.distributions.OneHotCategorical` distribution for each head)
        so that actions can be sampled easily alongside access to all of the other probability
        distribution utilities (including taking the mode when acting greedily).

        The network architecture is set up such that the final output is a tuple of tensors, one
        output by each head where each head maps from some shared hidden activations to an action
        space in a single linear layer with no activation. This form of a tuple of actions, one for
        each head, is required for compatibility with the wider `tf_agents` framework.

        :param input_tensor_spec: Specification for the input Tensors (usually environment states/
            observations)
        :param output_tensor_spec: Specification for the output tensor i.e. actions.
        :param action_subspace_dimensions: Tuple of the dimensions of each action subspace as ints.
        :param hidden_units: The number of units to use in the hidden layer(s) of the network.
        """
        # Define a function to be used to map TensorSpecs to TensorFlow distributions.
        def set_up_head_distribution(spec: tf.TensorSpec, action_dimension: int)\
                -> OneHotCategoricalProjectionNetwork:
            return OneHotCategoricalProjectionNetwork(spec, action_dimension)

        if len(action_subspace_dimensions) == 1:
            action_heads = (set_up_head_distribution(output_tensor_spec,
                                                     action_subspace_dimensions[0]),)
            output_spec = action_heads[0].output_spec
        else:
            action_heads = tf.nest.map_structure(set_up_head_distribution, output_tensor_spec,
                                                 action_subspace_dimensions)
            output_spec = tf.nest.map_structure(lambda head: head.output_spec, action_heads)

        # Run the standard Network initialisation per standard TensorFlow Agents.
        # State specification is the specification of the policy/network state as would be required
        # for an RNN. We do not require this here so pass an empty tuple.
        super(MultiHeadedCategoricalActionNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name="MultiHeadedCategoricalActionNetwork"
        )
        # Set up the layers of the network which then feeds into the action heads. This is done
        # after the init of the parent class to avoid bugs experienced when this was done earlier.
        self._shared_layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu)
                               for h in hidden_units]

        # Collect the weights from the projection layers of the action heads to ensure that they are
        # trained, tracked and loaded as trainable variables.
        self._head_projection_layers = [head.projection_layer for head in action_heads]

        # Set up the private network attributes.
        self._output_tensor_spec = output_tensor_spec
        self._action_subspace_dimensions = action_subspace_dimensions
        self._action_heads = action_heads
        self._hidden_units = hidden_units

    @property
    def output_tensor_spec(self) -> tf.TensorSpec:
        return self._output_tensor_spec

    def call(
            self,
            observations: Union[tf.Tensor, np.ndarray],
            step_type: Optional[Any],
            network_state: Union[Tuple, Tuple[Union[tf.Tensor, np.ndarray]]] = ()
    ) -> Tuple[
        Union[tfp.distributions.OneHotCategorical, Tuple[tfp.distributions.OneHotCategorical]],
        Union[Tuple, Tuple[Union[tf.Tensor, np.ndarray]]]]:
        """
        Run a forward pass of the action network mapping observations to a distribution over
        actions.

        :param observations: Tensor/Array of observation values from the environment.
        :param step_type: Not used in this network. Kept as an argument to be consistent with the
            standard TensorFlow Agents interface.
        :param network_state: The state of the network. Not required here as this network has no
            state since it is not recurrent.
        :return: A distribution over actions and the current network state.
        """
        # Use shared layers to attain inputs shared across each head.
        hidden_activations = tf.cast(observations, tf.float32)
        for layer in self._shared_layers:
            hidden_activations = layer(hidden_activations)

        # Determine the number of batch dimensions. Since this requires comparison to the input
        # tensor spec and the batch dimensions are preserved by the shared linear layers we
        # calculate batch dimensions based on the supplied observations.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

        # Attain a nested set of actions i.e. a tuple of actions one for each head.
        action_dist = tf.nest.map_structure(
            lambda proj_net: proj_net(hidden_activations, outer_rank), self._action_heads)

        # If there is only one action head unpack the tuple of 1 to attain the singular action
        # distribution itself.
        if len(self._action_subspace_dimensions) == 1:
            action_dist = action_dist[0]

        return action_dist, network_state
