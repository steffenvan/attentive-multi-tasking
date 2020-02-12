# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import sys
from .atari_wrappers import wrap_deepmind, make_atari

import numpy as np
import tensorflow as tf

import gym

nest = tf.contrib.framework.nest

StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step acc_episode_reward acc_episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')

action_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15 ,16 ,17]
# To convert the shape from (84, 84, 4) -> (4, 84, 84)
class TransposeWrapper(gym.ObservationWrapper):
  def observation(self, observation):
    transposed_obs = np.transpose(np.array(observation), axes=(2, 0, 1))
    return transposed_obs

def create_env(env_id, episode_life=True, clip_rewards=False, frame_stack=True, scale=False):
  env = make_atari(env_id)
  env = wrap_deepmind(env, episode_life, clip_rewards, frame_stack, scale)
  env = TransposeWrapper(env)
  return env

def get_observation_spec(env_id):
  env = create_env(env_id)
  obs_shape = env.observation_space.shape
  return obs_shape

def get_action_set(level_name):
  env  = create_env(level_name)
  return [i for i in range(env.action_space.n)]

class PyProcessAtari(object):

    def __init__(self, env_id, config, seed):
      self._env = create_env(env_id)
      
    def _reset(self):
      return self._env.reset()
    
    def _transpose_obs(self, observation):
      return observation.swapaxes(2, 0)

    def initial(self):
      initial_obs = self._transpose_obs(self._reset())
      return initial_obs

    def step(self, action):
      # If the current action exceeds the range of the specific game's action set length -> NOOP
      if action >= self._env.action_space.n:
        obs, reward, is_done, info = self._env.step(0)
      else: 
        obs, reward, is_done, info = self._env.step(action)
        
#      self._env.render()
      if is_done:
        obs = self._reset() 
      
      reward = np.float32(reward)
      obs = self._transpose_obs(obs)
      acc_raw_reward = np.float32(info['acc_raw_reward'])
      acc_raw_step = np.int32(info['acc_raw_step'])
      return reward, is_done, obs, acc_raw_reward, acc_raw_step

    def close(self):
      self._env.close()

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
      """Returns a nest of `TensorSpec` with the method's output specification."""

      env_id = constructor_kwargs['env_id']

      observation_spec = tf.contrib.framework.TensorSpec(get_observation_spec(env_id), tf.uint8)

      if method_name == 'initial':
        return observation_spec

      elif method_name == 'step':
        return (
            tf.contrib.framework.TensorSpec([], tf.float32),
            tf.contrib.framework.TensorSpec([], tf.bool),
            observation_spec,
            tf.contrib.framework.TensorSpec([], tf.float32),
            tf.contrib.framework.TensorSpec([], tf.int32),
        )

class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_reward = tf.constant(0.)
      initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0), tf.constant(0.), tf.constant(0))
      initial_done = tf.constant(True)
  
      initial_observation = self._env.initial()

      initial_output = StepOutput(
          initial_reward,
          initial_info,
          initial_done,
          initial_observation)

      # Control dependency to make sure the next step can't be taken before the
      # initial output has been read from the environment.
      with tf.control_dependencies(nest.flatten(initial_output)):
        initial_flow = tf.constant(0, dtype=tf.int64)
      initial_state = (initial_flow, initial_info)
      return initial_output, initial_state

  def step(self, action, state):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      flow, info = nest.map_structure(tf.convert_to_tensor, state)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([flow]):
        reward, done, observation , acc_reward, acc_step = self._env.step(action)

      with tf.control_dependencies(nest.flatten(observation)):
        new_flow = tf.add(flow, 1)

      # When done, include the reward in the output info but not in the
      # state for the next step.
      new_info = StepOutputInfo(info.episode_return + reward,
                                info.episode_step + 1,
                                acc_episode_reward=acc_reward,
                                acc_episode_step=acc_step)
      new_state = new_flow, nest.map_structure(
          lambda a, b: tf.where(done, a, b),
          StepOutputInfo(tf.constant(0.), tf.constant(0),
          new_info.acc_episode_reward,
          new_info.acc_episode_step),
          new_info)

      output = StepOutput(reward, new_info, done, observation)

      return output, new_state

