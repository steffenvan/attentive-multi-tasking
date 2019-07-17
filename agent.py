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

class ImpalaSubnet(snt.AbstractModule):
  """Agent with subnetworks."""

  def __init__(self, num_actions):
    super(ImpalaSubnet, self).__init__(name='subnet_agent')

    self._num_actions = num_actions
    self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
    self.sub_networks = FLAGS.subnets
    self.use_simplified = FLAGS.use_simplified

  def _torso(self, input_):
    last_action, env_output, level_name = input_
    reward, _, _, frame = env_output

    frame = tf.to_float(frame)
    frame /= 255

    fc_out_list = []
    weight_list = []

    one_hot_task = tf.one_hot(level_name, self._number_of_games)
    tau          = tf.reshape(one_hot_task, [-1, 1, self._number_of_games])
    frame_2      = tf.reshape(frame, [tf.shape(tau)[0], -1, 84 * 84 * 4])
    tau          = tf.tile(tau, [1, tf.shape(frame_2)[1], 1])
    tau          = tf.reshape(tau, [-1, self._number_of_games])

    for i in range(self.sub_networks):
      conv_out = frame
      conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = snt.Conv2D(32, 4, stride=2)(conv_out)
      conv_out = tf.nn.relu(conv_out)
      
      with tf.variable_scope('subnetwork_' + str(i)):
        fc_out   = snt.BatchFlatten()(conv_out)
        fc_out   = snt.Linear(256)(fc_out)
        fc_out   = tf.expand_dims(fc_out, axis=1)

        conv_out = tf.keras.layers.GlobalMaxPooling2D()(conv_out)
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

class SelfAttentionSubnet(snt.AbstractModule):
  """Agent with subnetworks and self-attention."""

  def __init__(self, num_actions):
    super(SelfAttentionSubnet, self).__init__(name='self_attention_subnet')

    self._num_actions = num_actions
    self._number_of_games = len(utilities_atari.ATARI_GAMES.keys())
    self.sub_networks = FLAGS.subnets

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

    # TODO: Will have to experiment with these variables. 
    h   = 4
    d_k = 24
    d_v = 24    

    for i in range(self.sub_networks):
      with tf.variable_scope('subnetwork_' + str(i)):
        conv_out = frame
        conv_out = snt.Conv2D(16, 8, stride=4)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Applying self attention 
        conv_out = self_attention.augmented_conv2d(conv_out, 32, 2, d_k, d_v, h, True, batch_size)
        conv_out = tf.nn.relu(conv_out)
        conv_out = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        fc_out   = snt.Linear(256)(conv_out)
        fc_out   = tf.expand_dims(fc_out, axis=1)

        conv_out = tf.concat(values=[conv_out, tau], axis=1)
        weight   = snt.Linear(1, name='attention_weight')(conv_out)
        
        fc_out_list.append(fc_out)
        weight_list.append(weight)

    fc_out_list         = tf.concat(values=fc_out_list, axis=1) # (84, 1, 256)
    weight_list         = tf.concat(values=weight_list, axis=1) # (84, 1)

    weights_soft_max    = tf.nn.softmax(weight_list) # (84, 1)
    hidden_softmaxed    = tf.reduce_sum(tf.expand_dims(weights_soft_max, axis=2) * fc_out_list, axis=1) # (84, 256)

    # Append clipped last reward and one hot last action.
    clipped_reward      = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat(
        [hidden_softmaxed, clipped_reward, one_hot_last_action],
        axis=1)

  def _head(self, core_output):
    core_output, level_name = core_output
    baseline_games = snt.Linear(self._number_of_games)(core_output)
  
    # adding time dimension
    level_name     = tf.reshape(level_name, [-1, 1, 1])
    baseline_games = tf.reshape(baseline_games, [tf.shape(level_name)[0], -1, self._number_of_games])

    level_name = tf.tile(level_name, [1, tf.shape(baseline_games)[1], 1])
    baseline = tf.batch_gather(baseline_games, level_name)
    baseline = tf.reshape(baseline, [tf.shape(core_output)[0]])

    policy_logits = snt.Linear(self._number_of_games * self._num_actions, name='policy_logits')(core_output) 
    policy_logits = tf.reshape(policy_logits, [tf.shape(level_name)[0], -1, self._number_of_games, self._num_actions])
    policy_logits = tf.batch_gather(policy_logits, level_name)
    policy_logits = tf.reshape(policy_logits, [tf.shape(core_output)[0], self._num_actions])

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
    'ImpalaSubnet'.lower(): ImpalaSubnet,
    'SelfAttentionSubnet'.lower(): SelfAttentionSubnet,
    'ImpalaFeedForward'.lower(): ImpalaFeedForward,
    # 'PopArtFeedForward'.lower(): PopArtFeedForward,
    # 'ImpalaLSTM'.lower(): ImpalaLSTM,
    # 'PopArtLSTM'.lower(): PopArtLSTM
  }

  return specific_agent[agent_name.lower()]
