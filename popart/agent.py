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
sys.path.append('../')
from utils import atari_utils

nest = tf.contrib.framework.nest
PopArtAgentOutput = collections.namedtuple('AgentOutput',
                                    'action policy_logits un_normalized_vf normalized_vf')

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

class PopArtFeedForward(snt.AbstractModule):
    def __init__(self, num_actions):
        super(PopArtFeedForward, self).__init__(name="popart_feed_forward")
        self._number_of_games = len(atari_utils.ATARI_GAMES.keys())
        self._num_actions  = num_actions
        self._mean         = tf.get_variable("mean", dtype=tf.float32, initializer=tf.tile(tf.constant([0.0]), multiples=[self._number_of_games]), trainable=False)
        self._mean_squared = tf.get_variable("mean_squared", dtype=tf.float32, initializer=tf.tile(tf.constant([1.0]), multiples=[self._number_of_games]), trainable=False)
        self._std          = nest.map_structure(tf.stop_gradient, 
                                                tf.sqrt(self._mean_squared - tf.square(self._mean)))
        self._beta         = 3e-4
        self._stable_rate  = 0.1
        self._epsilon      = 1e-4

    def _torso(self, input_):
        last_action, env_output, level_name = input_
        reward, _, _, frame = env_output

        # Convert to floats.
        frame = tf.to_float(frame)
        frame /= 255

        # Matching PNN's architecture       
        with tf.variable_scope('convnet'):
            conv_out = shallow_convolution(frame)

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        output = tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)
        return output

    def _head(self, torso_output):
        torso_output, level_name = torso_output

        normalized_vf_games    = snt.Linear(self._number_of_games, name='baseline')(torso_output)
        un_normalized_vf_games = self._std * normalized_vf_games + self._mean

        # Adding time dimension
        level_name     = tf.reshape(level_name, [-1, 1, 1])

        # Reshaping as to seperate the time and batch dimensions
        # We need to know the length of the time dimension, because it may differ in the initialization
        # E.g the learner and actors have different size batch/time dimension
        normalized_vf    = tf.reshape(normalized_vf_games, [tf.shape(level_name)[0], -1, self._number_of_games])
        un_normalized_vf = tf.reshape(un_normalized_vf_games, [tf.shape(level_name)[0], -1, self._number_of_games])
        
        # Tile the time dimension
        level_name       = tf.tile(level_name, [1, tf.shape(normalized_vf)[1], 1])
        normalized_vf    = tf.batch_gather(normalized_vf, level_name)    # (batch_size, time, 1)
        un_normalized_vf = tf.batch_gather(un_normalized_vf, level_name)    # (batch_size, time, 1)
        # Reshape to the batch size - because Sonnet's BatchApply expects a batch_size * time dimension. 
        normalized_vf    = tf.reshape(normalized_vf, [tf.shape(torso_output)[0]])
        un_normalized_vf = tf.reshape(un_normalized_vf, [tf.shape(torso_output)[0]])
        
        # Sample an action from the policy.
        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(torso_output)
        new_action = tf.random.categorical(policy_logits, num_samples=1, 
                                          dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')
        return PopArtAgentOutput(new_action, policy_logits, un_normalized_vf, normalized_vf) 

    def _build(self, input_):
        action, env_output, level_name = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                (action, env_output))
        outputs = self.unroll(actions, env_outputs, level_name)
        squeezed = nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        return squeezed

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, level_name):
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs, level_name))
        output = snt.BatchApply(self._head, name='batch_apply_unroll')((torso_outputs, level_name))
        return output
    
    def update_moments(self, vs, env_id):

        with tf.variable_scope("popart_feed_forward/batch_apply_unroll/baseline", reuse=True):
            weight = tf.get_variable("w")
            bias = tf.get_variable("b")

        def update_step(mm, _tuple):
            mean, mean_squared = mm
            gvt, _env_id = _tuple
            _env_id = tf.reshape(_env_id, [1, 1])

            # According to equation (6) in the PopArt-IMPALA paper
            # Matching the specific game with it's current vtrace corrected value estimate. 
            first_moment   = tf.reshape((1 - self._beta) * tf.gather(mean, _env_id) + self._beta * gvt, [1])
            second_moment  = tf.reshape((1 - self._beta) * tf.gather(mean_squared, _env_id) + self._beta * tf.square(gvt), [1])

            # Matching the moments to the specific environment, so we only update the statistics for the specific game. 
            n_mean         = tf.tensor_scatter_update(mean, _env_id, first_moment)
            n_mean_squared = tf.tensor_scatter_update(mean_squared, _env_id, second_moment)
            return n_mean, n_mean_squared

        # The batch may contain different games, so we need to ensure that 
        # the vtrace corrected value estimate matches the current game. 
        def update_batch(mm, gvt):
            return tf.foldl(update_step, (gvt, env_id), initializer=mm)

        new_mean, new_mean_squared = tf.foldl(update_batch, vs, initializer=(self._mean, self._mean_squared))
        new_std = tf.sqrt(new_mean_squared - tf.square(new_mean))
        new_std = tf.clip_by_value(new_std, self._epsilon, 1e6)

        # According to equation (9) in (Hessel et al., 2018)

        weight_update = weight * self._std / new_std
        bias_update   = (self._std * bias + self._mean - new_mean) / new_std 
        # Preserving outputs precisely (Pop). 
        new_weight = tf.assign(weight, weight_update)
        new_bias = tf.assign(bias, bias_update)
                
        with tf.control_dependencies([new_weight, new_bias]):
            new_mean = tf.assign(self._mean, new_mean)
            new_mean_squared = tf.assign(self._mean_squared, new_mean_squared)

        return new_mean, new_mean_squared


def agent_factory(agent_name):
  specific_agent = {
    'PopArtFeedForward'.lower(): PopArtFeedForward,
  }

  return specific_agent[agent_name.lower()]