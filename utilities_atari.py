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
  # ('AlienNoFrameskip-v4', 'AlienNoFrameskip-v4'),
  # ('AmidarNoFrameskip-v4','AmidarNoFrameskip-v4'),
  # ('AssaultNoFrameskip-v4','AssaultNoFrameskip-v4'),
  # ('AsterixNoFrameskip-v4','AsterixNoFrameskip-v4'),
  # ('AsteroidsNoFrameskip-v4','AsteroidsNoFrameskip-v4'),
  # ('AtlantisNoFrameskip-v4','AtlantisNoFrameskip-v4'),
  # ('BankHeistNoFrameskip-v4','BankHeistNoFrameskip-v4'),
  # ('BattleZoneNoFrameskip-v4','BattleZoneNoFrameskip-v4'),
  # ('BeamRiderNoFrameskip-v4', 'BeamRiderNoFrameskip-v4'),
  # ('BerzerkNoFrameskip-v4', 'BerzerkNoFrameskip-v4'),
  # ('BowlingNoFrameskip-v4','BowlingNoFrameskip-v4'),
  # ('BoxingNoFrameskip-v4','BoxingNoFrameskip-v4'),
  # ('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-v4'),
  # ('CentipedeNoFrameskip-v4','CentipedeNoFrameskip-v4'),
  # ('ChopperCommandNoFrameskip-v4','ChopperCommandNoFrameskip-v4'),
  # ('CrazyClimberNoFrameskip-v4','CrazyClimberNoFrameskip-v4'),
  # ('DefenderNoFrameskip-v4','DefenderNoFrameskip-v4'),
  # ('DemonAttackNoFrameskip-v4','DemonAttackNoFrameskip-v4'),
  # ('DoubleDunkNoFrameskip-v4','DoubleDunkNoFrameskip-v4'),
  # ('EnduroNoFrameskip-v4','EnduroNoFrameskip-v4'),
  # ('FishingDerbyNoFrameskip-v4','FishingDerbyNoFrameskip-v4'),
  # ('FreewayNoFrameskip-v4', 'FreewayNoFrameskip-v4'),
  # ('FrostbiteNoFrameskip-v4','FrostbiteNoFrameskip-v4'),
  # ('GopherNoFrameskip-v4', 'GopherNoFrameskip-v4'),
  # ('GravitarNoFrameskip-v4','GravitarNoFrameskip-v4'),
  # ('HeroNoFrameskip-v4','HeroNoFrameskip-v4'),
  # ('IceHockeyNoFrameskip-v4','IceHockeyNoFrameskip-v4'),
  # ('JamesbondNoFrameskip-v4','JamesbondNoFrameskip-v4'),
  # ('KangarooNoFrameskip-v4','KangarooNoFrameskip-v4'),
  # ('KrullNoFrameskip-v4', 'KrullNoFrameskip-v4') ,
  # ('KungFuMasterNoFrameskip-v4','KungFuMasterNoFrameskip-v4'),
  # ('MontezumaRevengeNoFrameskip-v4','MontezumaRevengeNoFrameskip-v4'),
  # ('MsPacmanNoFrameskip-v4','MsPacmanNoFrameskip-v4'),
  # ('NameThisGameNoFrameskip-v4','NameThisGameNoFrameskip-v4'),
  # ('PhoenixNoFrameskip-v4', 'PhoenixNoFrameskip-v4'),
  # ('PitfallNoFrameskip-v4', 'PitfallNoFrameskip-v4'),
  ('PongNoFrameskip-v4', 'PongNoFrameskip-v4'),
  # ('PrivateEyeNoFrameskip-v4','PrivateEyeNoFrameskip-v4'),
  # ('QbertNoFrameskip-v4', 'QbertNoFrameskip-v4'), 
  # ('RoadRunnerNoFrameskip-v4','RoadRunnerNoFrameskip-v4'),
  ('RiverraidNoFrameskip-v4', 'RiverraidNoFrameskip-v4'),
  # ('RobotankNoFrameskip-v4','RobotankNoFrameskip-v4'),
  ('SeaquestNoFrameskip-v4', 'SeaquestNoFrameskip-v4'),
  # ('SkiingNoFrameskip-v4','SkiingNoFrameskip-v4'),
  # ('SolarisNoFrameskip-v4','SolarisNoFrameskip-v4'),
  # ('SpaceInvadersNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4'),
  # ('StarGunnerNoFrameskip-v4','StarGunnerNoFrameskip-v4'),
 # ('SurroundNoFrameskip-v4','SurroundNoFrameskip-v4'),
  # ('TennisNoFrameskip-v4', 'TennisNoFrameskip-v4'),
  # ('TimePilotNoFrameskip-v4','TimePilotNoFrameskip-v4'),
  # ('TutankhamNoFrameskip-v4', 'TutankhamNoFrameskip-v4'),
  # ('UpNDownNoFrameskip-v4','UpNDownNoFrameskip-v4'),
  # ('VentureNoFrameskip-v4','VentureNoFrameskip-v4'),
  # ('VideoPinballNoFrameskip-v4','VideoPinballNoFrameskip-v4'),
  # ('WizardOfWorNoFrameskip-v4','WizardOfWorNoFrameskip-v4'),
  # ('YarsRevengeNoFrameskip-v4','YarsRevengeNoFrameskip-v4'),
  # ('ZaxxonNoFrameskip-v4','ZaxxonNoFrameskip-v4'),
])



HUMAN_SCORES_ATARI = {
  # 'AlienNoFrameskip-v4': 7127.7,
  # 'AmidarNoFrameskip-v4': 1719.5,
  # 'AssaultNoFrameskip-v4': 742.0,
  # 'AsterixNoFrameskip-v4': 8503.3,
  # 'AsteroidsNoFrameskip-v4': 47388.7,
  # 'AtlantisNoFrameskip-v4': 29028.1,
  # 'BankHeistNoFrameskip-v4': 753.1,
  # 'BattleZoneNoFrameskip-v4': 37187.5,
  # 'BeamRiderNoFrameskip-v4': 16926.5,
  # 'BerzerkNoFrameskip-v4': 2630.4,
  # 'BowlingNoFrameskip-v4': 160.7,
  # 'BoxingNoFrameskip-v4': 12.1,
  # 'BreakoutNoFrameskip-v4': 30.5,
  # 'CentipedeNoFrameskip-v4': 12017.0,
  # 'ChopperCommandNoFrameskip-v4':7387.8,
  # 'CrazyClimberNoFrameskip-v4': 35829.4,
  # 'DefenderNoFrameskip-v4': 18688.9,
  # 'DemonAttackNoFrameskip-v4': 1971.0,
  # 'DoubleDunkNoFrameskip-v4': -16.4,
  # 'EnduroNoFrameskip-v4': 860.5,
  # 'FishingDerbyNoFrameskip-v4': -38.7,
  # 'FreewayNoFrameskip-v4': 29.6,
  # 'FrostbiteNoFrameskip-v4': 4334.7,
  # 'GopherNoFrameskip-v4': 2412.5, 
  # 'GravitarNoFrameskip-v4': 3351.4, 
  # 'HeroNoFrameskip-v4': 30826.4,
  # 'IceHockeyNoFrameskip-v4': 0.9,
  # 'JamesbondNoFrameskip-v4': 302.8,
  # 'KangarooNoFrameskip-v4': 3035.0,
  # 'KrullNoFrameskip-v4': 2665.5,
  # 'KungFuMasterNoFrameskip-v4': 22736.3,
  # 'MontezumaRevengeNoFrameskip-v4': 4753.3,
  # 'MsPacmanNoFrameskip-v4': 6951.6,
  # 'NameThisGameNoFrameskip-v4': 8049.0,
  # 'PhoenixNoFrameskip-v4': 7242.6,
  # 'PitfallNoFrameskip-v4': 64363,
  'PongNoFrameskip-v4': 14.6,
  # 'PrivateEyeNoFrameskip-v4': 69571.3,
  # 'QbertNoFrameskip-v4': 13455.0,
  'RiverraidNoFrameskip-v4': 17118.0,
  # 'RoadRunnerNoFrameskip-v4': 7845.0,
  # 'RobotankNoFrameskip-v4': 11.9,
  'SeaquestNoFrameskip-v4': 42054.7,
  # 'SkiingNoFrameskip-v4': -4336.9,
  # 'SolarisNoFrameskip-v4': 12326.7,
  # 'SpaceInvadersNoFrameskip-v4': 1668.7,
  # 'StarGunnerNoFrameskip-v4': 10250.0, 
  #'SurroundNoFrameskip-v4': 6.5,
  # 'TennisNoFrameskip-v4': -8.3,
  # 'TimePilotNoFrameskip-v4': 5229.2,
  # 'TutankhamNoFrameskip-v4': 167.6,
  # 'UpNDownNoFrameskip-v4': 11693.2,
  # 'VentureNoFrameskip-v4': 1187.5,
  # 'VideoPinballNoFrameskip-v4': 17667.9,
  # 'WizardOfWorNoFrameskip-v4': 4756.5,
  # 'YarsRevengeNoFrameskip-v4': 54576.9,
  # 'ZaxxonNoFrameskip-v4': 9173.3 
}

RANDOM_SCORES_ATARI = {
  # 'AlienNoFrameskip-v4': 227.8, 
  # 'AmidarNoFrameskip-v4': 5.8,
  # 'AssaultNoFrameskip-v4': 222.4,
  # 'AsterixNoFrameskip-v4': 210.0,
  # 'AsteroidsNoFrameskip-v4': 719.1,
  # 'AtlantisNoFrameskip-v4': 12850.0,
  # 'BankHeistNoFrameskip-v4': 14.2,
  # 'BattleZoneNoFrameskip-v4': 2360.0,
  # 'BeamRiderNoFrameskip-v4': 363.9,
  # 'BerzerkNoFrameskip-v4': 123.7,
  # 'BowlingNoFrameskip-v4': 23.1,
  # 'BoxingNoFrameskip-v4': 0.1,
  # 'BreakoutNoFrameskip-v4': 1.7,
  # 'CentipedeNoFrameskip-v4': 2090.9,
  # 'ChopperCommandNoFrameskip-v4': 811.0,
  # 'CrazyClimberNoFrameskip-v4': 10780.5,
  # 'DefenderNoFrameskip-v4': 2874.5, 
  # 'DemonAttackNoFrameskip-v4': 152.1,
  # 'DoubleDunkNoFrameskip-v4': -18.6,
  # 'EnduroNoFrameskip-v4': 0.0,
  # 'FishingDerbyNoFrameskip-v4': -91.7,
  # 'FreewayNoFrameskip-v4': 0.0,
  # 'FrostbiteNoFrameskip-v4': 65.2,
  # 'GopherNoFrameskip-v4': 257.6,
  # 'GravitarNoFrameskip-v4': 173.0,
  # 'HeroNoFrameskip-v4': 1027.0,
  # 'IceHockeyNoFrameskip-v4': -11.2,
  # 'JamesbondNoFrameskip-v4': 29.0,
  # 'KangarooNoFrameskip-v4': 52.0,
  # 'KrullNoFrameskip-v4': 1598.0,
  # 'KungFuMasterNoFrameskip-v4': 258.5,
  # 'MontezumaRevengeNoFrameskip-v4': 0.0,
  # 'MsPacmanNoFrameskip-v4': 307.3,
  # 'NameThisGameNoFrameskip-v4': 2292.3,
  # 'PitfallNoFrameskip-v4': 761.4,
    'PongNoFrameskip-v4': -229.4,
  # 'PhoenixNoFrameskip-v4': -20.7,
  # 'PrivateEyeNoFrameskip-v4': 24.9,
  # 'QbertNoFrameskip-v4': 163.9,
    'RiverraidNoFrameskip-v4': 1338.5,
  # 'RobotankNoFrameskip-v4': 11.5,
  # 'RoadRunnerNoFrameskip-v4': 2.2,
    'SeaquestNoFrameskip-v4': 68.4,
  # 'SkiingNoFrameskip-v4': -17098.1,
  # 'SolarisNoFrameskip-v4': 1236.3,
  # 'SpaceInvadersNoFrameskip-v4': 148.0,
  # 'StarGunnerNoFrameskip-v4': 664.0,
#  'SurroundNoFrameskip-v4': -10.0,
  # 'TennisNoFrameskip-v4': -23.8,
  # 'TimePilotNoFrameskip-v4': 3568.0,
  # 'TutankhamNoFrameskip-v4': 11.4,
  # 'UpNDownNoFrameskip-v4': 533.4,
  # 'VentureNoFrameskip-v4': 0.0,
  # 'VideoPinballNoFrameskip-v4': 16256.9,
  # 'WizardOfWorNoFrameskip-v4': 563.5,
  # 'YarsRevengeNoFrameskip-v4': 3092.9,
  # 'ZaxxonNoFrameskip-v4': 32.5
}

ALL_LEVELS_ATARI = frozenset([ 
  # 'AlienNoFrameskip-v4',
  # 'AmidarNoFrameskip-v4',
  # 'AssaultNoFrameskip-v4',
  # 'AsterixNoFrameskip-v4',
  # 'AsteroidsNoFrameskip-v4',
  # 'AtlantisNoFrameskip-v4',
  # 'BankHeistNoFrameskip-v4',
  # 'BattleZoneNoFrameskip-v4',
  # 'BeamRiderNoFrameskip-v4',
  # 'BerzerkNoFrameskip-v4',
  # 'BowlingNoFrameskip-v4',
  # 'BoxingNoFrameskip-v4',
  # 'BreakoutNoFrameskip-v4',
  # 'CentipedeNoFrameskip-v4',
  # 'ChopperCommandNoFrameskip-v4',
  # 'CrazyClimberNoFrameskip-v4',
  # 'DefenderNoFrameskip-v4',
  # 'DemonAttackNoFrameskip-v4',
  # 'DoubleDunkNoFrameskip-v4',
  # 'EnduroNoFrameskip-v4',
  # 'FishingDerbyNoFrameskip-v4',
  # 'FreewayNoFrameskip-v4',
  # 'FrostbiteNoFrameskip-v4',
  # 'GopherNoFrameskip-v4',
  # 'GravitarNoFrameskip-v4',
  # 'HeroNoFrameskip-v4',
  # 'IceHockeyNoFrameskip-v4'
  # 'JamesbondNoFrameskip-v4',
  # 'KangarooNoFrameskip-v4',
  # 'KrullNoFrameskip-v4',
  # 'KungFuMasterNoFrameskip-v4',
  # 'MontezumaRevengeNoFrameskip-v4',
  # 'MsPacmanNoFrameskip-v4',
  # 'NameThisGameNoFrameskip-v4',
  # 'PitfallNoFrameskip-v4',
  'PongNoFrameskip-v4',  
  # 'PhoenixNoFrameskip-v4',
  # 'PrivateEyeNoFrameskip-v4',
  # 'QbertNoFrameskip-v4' 
  'RiverraidNoFrameskip-v4',
  # 'RoadRunnerNoFrameskip-v4',
  # 'RobotankNoFrameskip-v4',
  'SeaquestNoFrameskip-v4',
  # 'SkiingNoFrameskip-v4',
  # 'SolarisNoFrameskip-v4',
  # 'SpaceInvadersNoFrameskip-v4',
  # 'StarGunnerNoFrameskip-v4',
  #'SurroundNoFrameskip-v4',
  # 'TennisNoFrameskip-v4',
  # 'TimePilotNoFrameskip-v4',
  # 'TutankhamNoFrameskip-v4',
  # 'UpNDownNoFrameskip-v4',
  # 'VentureNoFrameskip-v4',
  # 'VideoPinballNoFrameskip-v4',
  # 'WizardOfWorNoFrameskip-v4',
  # 'YarsRevengeNoFrameskip-v4',
  # 'ZaxxonNoFrameskip-v4'
])

def _transform_level_returns(level_returns):
  """Converts training level names to test level names."""
  new_level_returns = {}
  for level_name, returns in level_returns.iteritems():
    new_level_returns[ATARI_GAMES.get(level_name, level_name)] = returns
    
  diff = test_set - set(new_level_returns.keys())
  if diff:
    raise ValueError('Missing levels: %s' % list(diff))
  for level_name, returns in new_level_returns.iteritems():
    if level_name in test_set:
      if returns == None:
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
