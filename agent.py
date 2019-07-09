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
import utilities_atari
import dmlab30_utilities
import self_attention
FLAGS = tf.app.flags.FLAGS

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

def create_attention_weights(conv_out, tau, num_games):
  tau          = tf.reshape(tau, [-1, 1, num_games])
  conv_out     = tf.reshape(conv_out, [tf.shape(tau)[0], -1, 256])
  tau          = tf.tile(tau, [1, tf.shape(conv_out)[1], 1])
  # Concatting 
  weights      = tf.concat(values=[conv_out, tau], axis=2)
  weights      = tf.reshape(weights, [-1, num_games + 256])
  weight_fc    = tf.contrib.layers.fully_connected(inputs=weights, num_outputs=16)
  weight_fc    = tf.contrib.layers.fully_connected(inputs=weight_fc, num_outputs=1, activation_fn=None)
  return weight_fc


class ImpalaSubNetworks(snt.AbstractModule):
  """Subnetworks"""

  def __init__(self, num_actions):
    super(ImpalaSubNetworks, self).__init__(name='impala_subnetworks')
    self._num_actions = num_actions
    self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
    self.sub_networks = 2
    self.use_simplified = FLAGS.use_simplified

  def _torso(self, input_):
    last_action, env_output, level_name = input_
    reward, _, _, frame = env_output

    frame = tf.to_float(frame)
    frame /= 255

    one_hot_task = tf.one_hot(level_name, self._number_of_games)

    values_fc_list = []
    hidden_fc_list = []
    weights_fc_list = []
    
    for i in range(self.sub_networks):
      with tf.variable_scope("sub_network_" + str(i)):
        conv_out     = snt.Conv2D(32, 4, stride=3)(frame)
        conv_out     = snt.BatchFlatten()(conv_out)
        conv_out     = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=256)

        value_fc     = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=1, activation_fn=None)
        hidden_fc    = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=self._num_actions, activation_fn=None)
        hidden_fc    = tf.expand_dims(hidden_fc, axis=1)
        
        # No seperate attention module. 
        if self.use_simplified == 1:
          weights_fc = create_attention_weights(conv_out, one_hot_task, self._number_of_games)
          weights_fc_list.append(weights_fc)
        
        values_fc_list.append(value_fc)
        hidden_fc_list.append(hidden_fc)

    # Using seperate attention module
    if not self.use_simplified == 1:
      with tf.variable_scope("attention_net"):
        conv_out        = snt.Conv2D(32, 4, stride=3)(frame)
        conv_out        = snt.BatchFlatten()(conv_out)
        conv_out        = tf.contrib.layers.fully_connected(inputs=conv_out, num_outputs=256)
        weights_fc      = create_attention_weights(conv_out, one_hot_task, self._number_of_games)        
        weights_fc_list = weights_fc
    else:
      weights_fc_list  = tf.concat(values=weights_fc_list, axis=1)
  
    values_fc_list   = tf.concat(values=values_fc_list, axis=1)
    hidden_fc_list   = tf.concat(values=hidden_fc_list, axis=1)
    
    weights_soft_max = tf.nn.softmax(weights_fc_list)
    values_softmaxed = tf.reduce_sum(weights_soft_max * values_fc_list, axis=1)
    hidden_softmaxed = tf.reduce_sum(tf.expand_dims(weights_soft_max, axis=2) * hidden_fc_list, axis=1)

    return values_softmaxed, hidden_softmaxed

  def _head(self, core_output):
    (values, hidden), level_name = core_output
    policy_logits = snt.Linear(self._number_of_games * self._num_actions, name='policy_logits')(hidden) 
    level_name    = tf.reshape(level_name, [-1, 1, 1])
    policy_logits = tf.reshape(policy_logits, [tf.shape(level_name)[0], -1, self._number_of_games, self._num_actions])
    level_name    = tf.tile(level_name, [1, tf.shape(policy_logits)[1], 1])

    policy_logits = tf.batch_gather(policy_logits, level_name)
    policy_logits = tf.reshape(policy_logits, [tf.shape(values)[0], self._num_actions])

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
    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs, level_name))
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


class SelfAttentionSubnet(snt.AbstractModule):
  """Agent with Simple CNN."""

  def __init__(self, num_actions):
    super(SelfAttentionSubnet, self).__init__(name='self_attention_subnet')

    self._num_actions = num_actions
    self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
    self.sub_networks = FLAGS.subnets
    self.use_simplified = FLAGS.use_simplified

  def _torso(self, input_):
    last_action, env_output, level_name = input_
    reward, _, _, frame = env_output

    frame = tf.to_float(frame)
    frame /= 255
    batch_size = tf.shape(frame)[0]

    fc_out_list = []
    weight_list = []

    one_hot_task = tf.one_hot(level_name, self._number_of_games)
    tau          = tf.reshape(one_hot_task, [-1, 1, self._number_of_games])
    frame_2      = tf.reshape(frame, [tf.shape(tau)[0], -1, 84 * 84 * 4])
    tau          = tf.tile(tau, [1, tf.shape(frame_2)[1], 1])
    tau          = tf.reshape(tau, [-1, self._number_of_games])

    h = 4
    d_k = 6
    d_v = 4
    q_dim = h * d_k
    k_dim = h * d_k
    v_dim = h * d_v

    for i in range(self.sub_networks):
      with tf.variable_scope('subnetwork_' + str(i)):
        conv_out = frame
        conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
        conv_out = tf.nn.relu(conv_out)
        # conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)
        conv_out = self_attention.augmented_conv2d(conv_out, 32, 2, d_k * h, d_v * h, h, True, batch_size)
        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv_out)
        # print("CONV OUT 0: ", conv_out)
        # print("CONV OUT 1: ", conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        fc_out   = conv_out
        fc_out   = snt.Linear(256)(fc_out)
        fc_out   = tf.expand_dims(fc_out, axis=1)

        conv_out = tf.concat(values=[conv_out, tau], axis=1)
        weight   = snt.Linear(1, name='weights')(conv_out)
        
        fc_out_list.append(fc_out)
        weight_list.append(weight)

    fc_out_list = tf.concat(values=fc_out_list, axis=1)
    weight_list = tf.concat(values=weight_list, axis=1)

    weights_soft_max = tf.nn.softmax(weight_list)
    hidden_softmaxed = tf.reduce_sum(tf.expand_dims(weights_soft_max, axis=2) * fc_out_list, axis=1)
    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat(
        [hidden_softmaxed, clipped_reward, one_hot_last_action],
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
    action, env_output, level_name = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs = self.unroll(actions, env_outputs, level_name)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

  @snt.reuse_variables
  def unroll(self, actions, env_outputs, level_name):
    _, _, done, _ = env_outputs

    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs, level_name))

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
    'ImpalaSubNetworks'.lower(): ImpalaSubNetworks,
    'SelfAttentionSubnet'.lower(): SelfAttentionSubnet
    # 'ImpalaLSTM'.lower(): ImpalaLSTM,
    # 'PopArtLSTM'.lower(): PopArtLSTM
  }

  return specific_agent[agent_name.lower()]
