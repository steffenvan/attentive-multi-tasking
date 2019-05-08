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

"""Utilities for Atari."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

ATARI_GAMES = collections.OrderedDict([
  # ('BeamRiderNoFrameskip-v4', 'BeamRiderNoFrameskip-v4'),
  ('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-v4'),
  # ('PongNoFrameskip-v4', 'PongNoFrameskip-v4'),
  # ('QbertNoFrameskip-v4', 'QbertNoFrameskip-v4'), 
  # ('SeaquestNoFrameskip-v4', 'SeaquestNoFrameskip-v4'),
  # ('SpaceInvadersNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4'),
    # ('BoxingNoFrameskip-v4', 'BoxingNoFrameskip-v4'),
])

HUMAN_SCORES_ATARI = {
  # 'BeamRiderNoFrameskip-v4': 16926.5,
  'BreakoutNoFrameskip-v4': 30.5,
  # 'PongNoFrameskip-v4': 14.6,
  # 'QbertNoFrameskip-v4': 13455.0,
  # 'SeaquestNoFrameskip-v4': 42054.7,
  # 'SpaceInvadersNoFrameskip-v4': 1668.7,
  # 'BoxingNoFrameskip-v4': 6.0,
}


RANDOM_SCORES_ATARI = {
  # 'BeamRiderNoFrameskip-v4': 0.5,
  'BreakoutNoFrameskip-v4': 1.0,
  # 'PongNoFrameskip-v4': 1.2,
  # 'QbertNoFrameskip-v4': 232.0,
  # 'SeaquestNoFrameskip-v4': 101.0,
  # 'SpaceInvadersNoFrameskip-v4': 42.0,
  # 'BoxingNoFrameskip-v4': 0.5,
}

ALL_LEVELS_ATARI = frozenset([
 'BeamRiderNoFrameskip-v4',
 'BreakoutNoFrameskip-v4',
 'PongNoFrameskip-v4',
 'QbertNoFrameskip-v4',
 'SeaquestNoFrameskip-v4',
 'SpaceInvadersNoFrameskip-v4',
])


# specific_action_set = {
#   "BeamRiderNoFrameskip-v4": ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE', 'LEFTFIRE'),
#   "BreakoutNoFrameskip-v4":   ("NOOP", 'FIRE', 'RIGHT', 'LEFT'),
#   "PongNoFrameskip-v4":           ("NOOP", 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'),
#   "QbertNoFrameskip-v4": ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'),
#   "SeaquestNoFrameskip-v4": ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 
#                           'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'),
#   "SpaceInvadersNoFrameskip-v4": ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
# }


def _transform_level_returns(level_returns):
  """Converts training level names to test level names."""
  new_level_returns = {}
  for level_name, returns in level_returns.iteritems():
    new_level_returns[ATARI_GAMES.get(level_name, level_name)] = returns

  test_set = set(ATARI_GAMES.values())
  diff = test_set - set(new_level_returns.keys())
  if diff:
    raise ValueError('Missing levels: %s' % list(diff))

  for level_name, returns in new_level_returns.iteritems():
    if level_name in test_set:
      if not returns:
        raise ValueError('Missing returns for level: \'%s\': ' % level_name)
    else:
      tf.logging.info('Skipping level %s for calculation.', level_name)
  return new_level_returns


def compute_human_normalized_score(level_returns, per_level_cap):
  """Computes human normalized score.

  Levels that have different training and test versions, will use the returns
  for the training level to calculate the score. E.g.
  'env_id_train' will be used for
  'env_id_test'.

  Args:
    level_returns: A dictionary from level to list of episode returns.
    per_level_cap: A percentage cap (e.g. 100.) on the per level human
      normalized score. If None, no cap is applied.

  Returns:
    A float with the human normalized score in percentage.

  Raises:
    ValueError: If a level is missing from `level_returns` or has no returns.
  """
  new_level_returns = _transform_level_returns(level_returns)

  def human_normalized_score(level_name, returns):
    score = np.mean(returns)
    human = HUMAN_SCORES_ATARI[level_name]
    random = RANDOM_SCORES_ATARI[level_name]
    human_normalized_score = (score - random) / (human - random) * 100
    if per_level_cap is not None:
      human_normalized_score = min(human_normalized_score, per_level_cap)

    return human_normalized_score

  new_score = [human_normalized_score(k, v) for k, v in new_level_returns.items()]
  return np.mean(new_score)


def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))

def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)
