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
from utilities_atari import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss

nest = tf.contrib.framework.nest
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')

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

        self._num_actions  = num_actions
        self._mean         = tf.get_variable("mean", shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer())
        self._mean_squared = tf.get_variable("mean_squared", shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer())
        self._std          = tf.get_variable("standard-deviation", dtype=tf.float32, initializer=tf.eye(256))
        self._beta         = 3e-4


    def initial_state(self, batch_size):

        return tf.constant(0, shape=[1, 1])

    def _torso(self, input_):
        last_action, env_output = input_

        reward, _, _, frame = env_output
        # print("Instruction is: ", instruction)

        # Convert to floats.
        frame = tf.to_float(frame)
        frame /= 255
        
        with tf.variable_scope('convnet'):
            conv_out = shallow_convolution(frame)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        output = tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)
        return output

    def _head(self, torso_output):

        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(torso_output)

        unormalized_baseline = snt.Linear(1, name='baseline')(torso_output)
        baseline = tf.squeeze(unormalized_baseline, axis=-1)

        # baseline = self._std * baseline + self._mean

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1,
                                    output_dtype=tf.int32)

        new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, initial_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                (action, env_output))
        outputs, initial_state = self.unroll(actions, env_outputs, initial_state)
        squeezed = nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        return squeezed, initial_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, initial_state):
        _, _, done, _ = env_outputs
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
        output = snt.BatchApply(self._head)(tf.stack(torso_outputs)), initial_state
        return output

    def update_moments(self, vtrace_corrected_return):
        old_mean = self._mean
        old_mean_squared = self._mean_squared
        self._mean = (1 - self._beta) * old_mean + self._beta * vtrace_corrected_return     
        self._mean_squared = (1 - self._beta) * old_mean + self._beta * tf.square(vtrace_corrected_return)

        return old_mean, old_mean_squared

    def compute_sigma(self):
        sigma = tf.sqrt(self._mean_squared - tf.square(self._mean))
        return sigma


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
        # print("Instruction is: ", instruction)

        # Convert to floats.
        frame = tf.to_float(frame)
        frame /= 255
        
        with tf.variable_scope('convnet'):
            conv_out = res_net_convolution(frame)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        # TODO: Use the normalization here. 
        # conv_out = normlize(conv_out) 
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
        # print("squeezed (_build): ", squeezed)
        return squeezed, core_state

    # Just needs to know if the episode has ended. 
    # This is used in build by 
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
        # self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, is_done in zip(tf.unstack(torso_outputs), tf.unstack(done)):
        # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, is_done),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)
        # print("core output list (unroll): ", len(core_output_list))
        output = snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state
        # print("Output is (unroll): ", output)
        return output

def agent_factory(agent_name):
  supported_agent = {
    'FeedForwardAgent'.lower(): FeedForwardAgent,
    'LSTMAgent'.lower(): LSTMAgent,
  }
  return supported_agent[agent_name.lower()]