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
import dmlab30_utilities
# from utilities_atari import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss

nest = tf.contrib.framework.nest
AgentOutput = collections.namedtuple('AgentOutput',
                                    'action policy_logits un_normalized_vf normalized_vf')
ImpalaAgentOutput = collections.namedtuple('AgentOutput',
                                             'action policy_logits baseline')
def shallow_convolution(frame):
    conv_out = frame
    conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)
    return conv_out

def res_net_convolution(frame):
    conv_out = frame
    for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
        # Downscale.
        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
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

def bigger_shallow_convolution(frame):
      conv_out = frame
      conv_out = snt.Conv2D(32, 8, stride=4)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = snt.Conv2D(64, 4, stride=2)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = snt.Conv2D(64, 3, stride=1)(conv_out)
      return conv_out

def pnn_convolution(frame):
    conv_out = frame
    conv_out = snt.Conv2D(12, 8, stride=4)(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.Conv2D(12, 4, stride=2)(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.Conv2D(12, 3, stride=1)(conv_out)
    return conv_out

class ImpalaSubNetworks(snt.AbstractModule):
  """Subnetworks"""

  def __init__(self, num_actions):
    super(ImpalaSubNetworks, self).__init__(name='impala_subnetworks')
    self._num_actions = num_actions
    self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
    self.sub_networks = 3

  def _torso(self, input_):
    last_action, env_output, level_name = input_
    reward, _, _, frame = env_output

    frame = tf.to_float(frame)
    frame /= 255

    with tf.variable_scope('shared_convnet'):
      shared_conv_out = frame
      shared_conv_out = snt.Conv2D(32, 3, stride=2)(shared_conv_out)
      shared_conv_out = tf.nn.relu(shared_conv_out)

    one_hot_task = tf.one_hot(level_name, self._number_of_games)

    values_fc_list = []
    hidden_fc_list = []
    weights_fc_list = []
    
    for i in range(self.sub_networks):
      with tf.variable_scope("sub_network_" + str(i)):
        conv_out     = snt.Conv2D(32, 4, stride=3)(shared_conv_out)
        conv_out     = snt.BatchFlatten()(conv_out)
        conv_out     = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=256)

        value_fc     = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=1, activation_fn=None)
        
        hidden_fc    = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=self._num_actions, activation_fn=None)
        hidden_fc    = tf.expand_dims(hidden_fc, axis=1)

        # Concat the one-hot-encoding and shared non-linear layer
        # tau          = tf.expand_dims(one_hot_task, axis=1)
        tau          = one_hot_task
        tau          = tf.reshape(tau, [-1, 1, self._number_of_games])

        conv_out     = tf.reshape(conv_out, [tf.shape(tau)[0], -1, 256])
        tau          = tf.tile(tau, [1, tf.shape(conv_out)[1], 1])
        weights      = tf.concat(values=[conv_out, tau], axis=2)
        weights      = tf.reshape(weights, [-1, self._number_of_games + 256])
        weight_fc    = tf.contrib.layers.fully_connected(inputs=weights, num_outputs=16)
        weight_fc    = tf.contrib.layers.fully_connected(inputs=weight_fc, num_outputs=1, activation_fn=None)

        values_fc_list.append(value_fc)
        hidden_fc_list.append(hidden_fc)
        weights_fc_list.append(weight_fc)

    values_fc_list   = tf.concat(values=values_fc_list, axis=1)
    weights_fc_list  = tf.concat(values=weights_fc_list, axis=1)
    hidden_fc_list   = tf.concat(values=hidden_fc_list, axis=1)
    weights_soft_max = tf.nn.softmax(weights_fc_list)

    values_softmaxed = tf.reduce_sum(weights_soft_max * values_fc_list, axis=1)
    hidden_softmaxed = tf.reduce_sum(tf.expand_dims(weights_soft_max, axis=2) * hidden_fc_list, axis=1)

    return values_softmaxed, hidden_softmaxed
    # Append clipped last reward and one hot last action.
    # clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    # one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    # return tf.concat(
    #     [conv_out, clipped_reward, one_hot_last_action],
    #     axis=1)

  def _head(self, core_output):
    (values, hidden), level_name = core_output
    policy_logits = snt.Linear(self._number_of_games * self._num_actions, name='policy_logits')(hidden) 
    level_name    = tf.reshape(level_name, [-1, 1, 1])
    policy_logits = tf.reshape(policy_logits, [tf.shape(level_name)[0], -1, self._number_of_games, self._num_actions])
    level_name    = tf.tile(level_name, [1, tf.shape(policy_logits)[1], 1])
    print("LEVEL NAME: ", level_name)
    print("POLICY LOGITS 1: ", policy_logits)
    policy_logits = tf.batch_gather(policy_logits, level_name)
    policy_logits = tf.reshape(policy_logits, [tf.shape(values)[0], self._num_actions])
    # print("POLICY LOGITS: ", policy_logits)
    # print("VALUES: ", values)

    # policy_logits   = snt.Linear(self._num_actions, name='policy_logits')(hidden) 
    baseline = values
    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1,
                                output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return ImpalaAgentOutput(new_action, policy_logits, baseline)

  def _build(self, input_):
    action, env_output, level_name = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs = self.unroll(actions, env_outputs, level_name)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

  @snt.reuse_variables
  def unroll(self, actions, env_outputs, level_name):
    _, _, done, _ = env_outputs
    print("ENV OUTPUTS: ", env_outputs)
    print("ACTIONS: ", actions)
    
    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs, level_name))
    # print("TORSO OUTPUTS: ", torso_outputs)
    # print("LEVEL NAME: ", level_name)
    return snt.BatchApply(self._head)((torso_outputs, level_name))


class ImpalaFFRelational(snt.AbstractModule):
  """."""

  def __init__(self, num_actions):
    super(ImpalaFFRelational, self).__init__(name='impala_feed_forward_agent_relational')
    
    self._num_actions = num_actions

    coord_list = []
    self.n_entities = 11
    for y in range(self.n_entities):
        for x in range(self.n_entities):
            coord_list.append(float(y))
            coord_list.append(float(x))

    self.coord_list = tf.constant(coord_list, shape=[1, self.n_entities, self.n_entities, 2], name="coord_list")


  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, frame = env_output
    frame = tf.to_float(frame)
    frame /= 255
    batch_size = tf.shape(frame)[0]
    
    h = 9
    # TODO: Check whether the dimensions of the queries needs to be modified
    d_q = 32
    d_v = 32
    q_dim = h * d_q
    k_dim = h * d_q
    v_dim = h * d_v

    with tf.variable_scope('convnet'):
      conv_out = frame
      conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.concat([tf.broadcast_to(self.coord_list, [batch_size, self.n_entities, self.n_entities, 2]), conv_out], axis=3)
    conv_out = tf.reshape(conv_out, [batch_size, self.n_entities*self.n_entities, 34])

    queries = snt.BatchApply(snt.Linear(q_dim))(conv_out)
    keys    = snt.BatchApply(snt.Linear(k_dim))(conv_out)
    values  = snt.BatchApply(snt.Linear(v_dim))(conv_out)

    def extract_head(input_tensor, dim):
        input_tensor = tf.reshape(input_tensor, [batch_size, self.n_entities*self.n_entities, h, dim])
        input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
        input_tensor = tf.reshape(input_tensor, [batch_size * h, self.n_entities*self.n_entities, dim])
        return input_tensor

    queries = extract_head(queries, d_q)
    keys    = extract_head(keys, d_q)
    values  = extract_head(values, d_v)

    dot_prod_attention = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) / tf.sqrt(float(d_q))
    dot_prod_attention_sm = tf.nn.softmax(dot_prod_attention)

    # softmax(Q, K^T) * V
    attention_qkv = tf.matmul(dot_prod_attention_sm, values)
    attention_qkv = tf.reshape(attention_qkv, [batch_size, h, self.n_entities*self.n_entities, d_v])
    attention_qkv = tf.transpose(attention_qkv, [0, 2, 1, 3])
    attention_qkv = tf.reshape(attention_qkv, [batch_size, self.n_entities*self.n_entities, h*d_v])

    # Creating fully connected layers before the residual connection
    # entities_mods = tf.contrib.layers.fully_connected(inputs=attention_qkv, num_outputs=34)
    entities_mods = snt.BatchApply(snt.nets.MLP([384, 384, 34]))(attention_qkv)
    conv_out += entities_mods
    
    conv_out = tf.reshape(conv_out, [batch_size, self.n_entities, self.n_entities, 34])
    conv_out = tf.keras.layers.MaxPool2D(pool_size=(self.n_entities, self.n_entities), padding='valid')(conv_out)
    conv_out = tf.squeeze(conv_out, axis=[1, 2])
    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)


    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat(
        [conv_out, clipped_reward, one_hot_last_action],
        axis=1)

  def _head(self, core_output):
    policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1,
                                output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return ImpalaAgentOutput(new_action, policy_logits, baseline)

  def _build(self, input_):
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs = self.unroll(actions, env_outputs)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

  @snt.reuse_variables
  def unroll(self, actions, env_outputs):
    _, _, done, _ = env_outputs
    torso_outputs = snt.BatchApply(self._torso, name="batch_apply_torso")((actions, env_outputs))

    return snt.BatchApply(self._head)(torso_outputs)


class ImpalaFeedForward(snt.AbstractModule):
  """Agent with Simple CNN."""

  def __init__(self, num_actions):
    super(ImpalaFeedForward, self).__init__(name='impala_feed_forward_agent')

    self._num_actions = num_actions

  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, frame = env_output

    frame = tf.to_float(frame)
    frame /= 255

    with tf.variable_scope('convnet'):
      conv_out = frame
      conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)

    conv_out = tf.nn.relu(conv_out)
    # conv_out = snt.BatchApply(tf.keras.layers.MaxPool2D(pool_size=(9, 9), padding='valid'))(conv_out)
    conv_out = snt.BatchFlatten()(conv_out)
    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat(
        [conv_out, clipped_reward, one_hot_last_action],
        axis=1)

  def _head(self, core_output):
    policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1,
                                output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return ImpalaAgentOutput(new_action, policy_logits, baseline)

  def _build(self, input_):
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs = self.unroll(actions, env_outputs)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

  @snt.reuse_variables
  def unroll(self, actions, env_outputs):
    _, _, done, _ = env_outputs

    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

    return snt.BatchApply(self._head)(torso_outputs)


def agent_factory(agent_name):
  specific_agent = {
    'ImpalaFeedForward'.lower(): ImpalaFeedForward,
    # 'PopArtFeedForward'.lower(): PopArtFeedForward,
    'ImpalaFFRelational'.lower(): ImpalaFFRelational,
    'ImpalaSubNetworks'.lower(): ImpalaSubNetworks
    # 'ImpalaLSTM'.lower(): ImpalaLSTM,
    # 'PopArtLSTM'.lower(): PopArtLSTM
  }

  return specific_agent[agent_name.lower()]
