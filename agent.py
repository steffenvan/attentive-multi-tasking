from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sonnet as snt
import tensorflow as tf
import itertools
import collections
import contextlib
import functools
import os
import sys
import vtrace
import numpy as np
import utilities_atari
from utilities_atari import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss

nest = tf.contrib.framework.nest
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits normalized_vf denormalized_vf')

def res_net_convolution(frame):
    for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
        # Downscale.
        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(frame)
        conv_out = tf.nn.pool(
        conv_out,
        window_shape=[3, 3],
        pooling_type='MAX',
        padding='SAME',
        strides=[2, 2])
        # Residual block(s).
        for j in range(num_blocks):
            with tf.variable_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.nn.relu(conv_out)
                conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                conv_out = tf.nn.relu(conv_out)
                conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                conv_out += block_input
    return conv_out

def shallow_convolution(frame):
    conv_out = frame
    # Downscale.
    conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)
    return conv_out

class FeedForwardAgent(snt.AbstractModule):
    def __init__(self, num_actions):
        super(FeedForwardAgent, self).__init__(name="feed_forward_agent")
        self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
        self._num_actions  = num_actions
        self._mean         = tf.get_variable("mean", dtype=tf.float32, initializer=tf.constant([0.0]))
        self._mean_squared = tf.get_variable("mean_squared", dtype=tf.float32, initializer=tf.constant([1.0]))
        self._std          = tf.sqrt(self._mean_squared - tf.square(self._mean))
        self._beta         = 3e-4

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)
        frame /= 255
        
        with tf.variable_scope('convnet'):
            conv_out = shallow_convolution(frame)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        reward = tf.expand_dims(reward, -1)
        output = tf.concat([conv_out, reward, one_hot_last_action], axis=1)
        return output

    def _head(self, torso_output):

        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(torso_output)
        linear = snt.Linear(self._number_of_games, name='baseline')
        last_linear_layer_vf = linear(torso_output)

        normalized_vf = last_linear_layer_vf
        denormalized_vf = tf.stop_gradient(self._std) * normalized_vf + tf.stop_gradient(self._mean)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1,
                                    output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, [1], name='new_action')

        return AgentOutput(new_action, policy_logits, normalized_vf, denormalized_vf) 

    def _build(self, input_):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                (action, env_output))
        outputs = self.unroll(actions, env_outputs)
        squeezed = nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        return squeezed

    @snt.reuse_variables
    def unroll(self, actions, env_outputs):
        # _, _, done, _ = env_outputs
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
        output = snt.BatchApply(self._head, name='batch_apply_unroll')(tf.stack(torso_outputs))
        return output
    
    def update_moments(self, vs, env_id):
        """
        This function computes the normalization statistics for the actor and critic updates, and 
        updates the statistics according to the Multi-task PopArt paper (Hessel et al., 2018). 
        https://arxiv.org/abs/1809.04474

        Args: 
            vs:     Vtrace corrected value estimates. 
            env_id: single game id. Used to pair the value function and specific game. 

        Returns:
            A tuple of the updated first and second moments according to equation (6) 
            in (Hessel et al., 2018)
        """
        with tf.variable_scope("feed_forward_agent/batch_apply_unroll/baseline", reuse=True):
            weight = tf.get_variable("w")
            bias = tf.get_variable("b")

        def update_step(mm, _tuple):
            mean, mean_squared = mm
            gvt, env_id = _tuple
            env_id = tf.reshape(env_id, [1, 1])

            # According to equation (6) in (Hessel et al., 2018).
            # Matching the specific game with it's current vtrace corrected value estimate. 
            first_moment   = tf.reshape((1 - self._beta) * tf.gather(mean, env_id) + self._beta * gvt, [1])
            second_moment  = tf.reshape((1 - self._beta) * tf.gather(mean_squared, env_id) + self._beta * tf.square(gvt), [1])

            # Matching the moments to the specific environment, so we only update the statistics for the specific game. 
            n_mean         = tf.tensor_scatter_update(mean, env_id, first_moment)
            n_mean_squared = tf.tensor_scatter_update(mean_squared, env_id, second_moment)
            return n_mean, n_mean_squared

        # The batch may contain different games, so we need to 
        # ensure that the vtrace corrected value estimate matches the current game. 
        def update_batch(mm, gvt):
            return tf.foldl(update_step, (gvt, env_id), initializer=mm)

        new_mean, new_mean_squared = tf.foldl(update_batch, vs, initializer=(self._mean, self._mean_squared))
        new_std = tf.sqrt(new_mean_squared - tf.square(new_mean)) 

        # According to equation (9) in Multi-task PopArt paper
        weight_update = weight * self._std / new_std
        bias_update   = (self._std * bias + self._mean - new_mean) / new_std 
        
        # And here we perform the Preserving outputs precisely (Pop) updates. 
        new_weight = tf.assign(weight, weight_update)
        new_bias = tf.assign(bias, bias_update)
        with tf.control_dependencies([new_weight, new_bias]):
            new_mean = tf.assign(self._mean, new_mean)
            new_mean_squared = tf.assign(self._mean_squared, new_mean_squared)

        return new_mean, new_mean_squared

# TODO: fix the PopArt normalization for the LSTM agent. 
class LSTMAgent(snt.RNNCore):

    def __init__(self, num_actions):
        super(LSTMAgent, self).__init__(name="agent")

        self._num_actions = num_actions
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        init_state = self._core.zero_state(batch_size, tf.float32)
        return init_state
            
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)
        frame /= 255
        
        with tf.variable_scope('convnet'):
            conv_out = res_net_convolution(frame)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        output = tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)

        return output 

    # The last layer of the neural network
    def _head(self, core_output):
        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(
            core_output)
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)


        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1,
                                    output_dtype=tf.int32)

        new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        squeezed = nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        return squeezed, core_state

    # Just needs to know if the episode has ended. 
    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs
        # Instructions are in here <-- coming from _torso, which returns a 
        # tensor of convolutional output, one_hot_action, 
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self.initial_state(tf.shape(actions)[1])
        core_output_list = []
        for input_, is_done in zip(tf.unstack(torso_outputs), tf.unstack(done)):
        # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, is_done),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)
        output = snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state
        return output

def agent_factory(agent_name):
  specific_agent = {
    'FeedForwardAgent'.lower(): FeedForwardAgent,
    'LSTMAgent'.lower(): LSTMAgent,
  }

  return specific_agent[agent_name.lower()]