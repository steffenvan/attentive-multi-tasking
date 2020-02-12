import tensorflow as tf

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/popart_agent', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Agent
flags.DEFINE_string('agent_name', 'popartfeedforward', 'Which learner to use')

# Atari environments
flags.DEFINE_integer('width', 84, 'Width of observation')
flags.DEFINE_integer('height', 84, 'Height of observation')

# Environment settings
flags.DEFINE_integer('total_environment_frames', int(6e8),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 16, 'Number of actors.')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 20, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('queue_capacity', 1, 'queue capacity.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_string('level_name', 'PongNoFrameskip-v4', 'level name')
flags.DEFINE_integer('multi_task', 1, 'Training on multiple games')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.01, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('gradient_clipping', 40.0, 'Negative means no clipping')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0006, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .01, 'RMSProp epsilon.')