" IMPALA for Atari" 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import os
import sys

# from loss_utility import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss
# import dmlab30
import utilities_atari
from utilities_atari import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss
import environments
import numpy as np
import py_process
import sonnet as snt
import tensorflow as tf
import vtrace
from agent import Agent, build_actor, build_learner
# import agent

try:
  import dynamic_batching
except tf.errors.NotFoundError:
  tf.logging.warning('Running without dynamic batching.')

from six.moves import range

nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Atari environments

flags.DEFINE_integer('width', 84, 'Width of observation')
flags.DEFINE_integer('height', 84, 'Height of observation')

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')

def is_single_machine():
    return FLAGS.task == -1

def create_atari_environment(env_id, seed, is_test=False):
#   print("Before env proxy")
  config = {
      'width': 84,
      'height': 84,
      'level': env_id,
      'logLevel': 'warn'
  }
  env_proxy = py_process.PyProcess(environments.PyProcessAtari, 
                                   env_id, config, FLAGS.num_action_repeats, seed)

  environment = environments.FlowEnvironment(env_proxy.proxy)
  return environment


@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""
  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs

def train(action_set, level_names):
  """Train."""
  if is_single_machine():
    local_job_device = ''
    shared_job_device = ''
    is_actor_fn = lambda i: True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    filters = []
    # print("Type of atari data structure: ", type(level_names))
    # print("Should be atari games: ", level_names[0])
  else:
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # actors. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']
    })
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
    filters = [shared_job_device, local_job_device]

  # Only used to find the actor output structure.
  with tf.Graph().as_default():
    
    env = create_atari_environment(level_names[0], seed=1)

    agent = Agent(len(action_set))
    structure = build_actor(agent, env, level_names[0], action_set)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]    
    shapes = [t.shape.as_list() for t in flattened_structure]


  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/cpu'), \
       pin_global_variables(global_variable_device):
    tf.set_random_seed(1)  # Makes initialization deterministic.

    # Create Queue and Agent on the learner.
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
      agent = Agent(len(action_set))

      if is_single_machine() and 'dynamic_batching' in sys.modules:
        # For single machine training, we use dynamic batching for improved GPU
        # utilization. The semantics of single machine training are slightly
        # different from the distributed setting because within a single unroll
        # of an environment, the actions may be computed using different weights
        # if an update happens within the unroll.
        old_build = agent._build
        @dynamic_batching.batch_fn
        def build(*args):
          # print("experiment.py: args: ", args)
          with tf.device('/gpu'):
            return old_build(*args)
        tf.logging.info('Using dynamic batching.')
        agent._build = build

    # Build actors and ops to enqueue their output.
    enqueue_ops = []
    for i in range(FLAGS.num_actors):
      if is_actor_fn(i):
        level_name = level_names[i % len(level_names)]
        tf.logging.info('Creating actor %d with level %s', i, level_name)
        env = create_atari_environment(level_name, seed=i + 1)
        # TODO: Modify to atari environment
        actor_output = build_actor(agent, env, level_name, action_set)
        
        # print("Actor output is: ", actor_output)
        with tf.device(shared_job_device):
          enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

    # If running in a single machine setup, run actors with QueueRunners
    # (separate threads).
    if is_learner and enqueue_ops:
      tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    # Build learner.
    if is_learner:
      # Create global step, which is the number of environment frames processed.
      tf.get_variable(
          'num_environment_frames',
          initializer=tf.zeros_initializer(),
          shape=[],
          dtype=tf.int64,
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      # Create batch (time major) and recreate structure.
      dequeued = queue.dequeue_many(FLAGS.batch_size)
      dequeued = nest.pack_sequence_as(structure, dequeued)

      def make_time_major(s):
        return nest.map_structure(
            lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

      dequeued = dequeued._replace(
          env_outputs=make_time_major(dequeued.env_outputs),
          agent_outputs=make_time_major(dequeued.agent_outputs))

      with tf.device('/gpu'):
        # Using StagingArea allows us to prepare the next batch and send it to
        # the GPU while we're performing a training step. This adds up to 1 step
        # policy lag.
        flattened_output = nest.flatten(dequeued)
        area = tf.contrib.staging.StagingArea(
            [t.dtype for t in flattened_output],
            [t.shape for t in flattened_output])
        stage_op = area.put(flattened_output)
        data_from_actors = nest.pack_sequence_as(structure, area.get())

        # Unroll agent on sequence, create losses and update ops.
        output = build_learner(agent, data_from_actors.agent_state,
                               data_from_actors.env_outputs,
                               data_from_actors.agent_outputs)

    # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
    config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config,
        hooks=[py_process.PyProcessHook()]) as session:

      if is_learner:
        # Logging.
        # TODO: Modify this to be able to handle atari
        # env_returns = {env_id: [] for env_id in env_ids}
        level_returns = {level_name: [] for level_name in level_names}
        # COMMENT OUT SUMMARY WRITER IF NEEDED
        # summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

        # Prepare data for first run.
        session.run_step_fn(
            lambda step_context: step_context.session.run(stage_op))

        # Execute learning and track performance.
        num_env_frames_v = 0
        total_episode_frames = 0
        average_frames = 24000
        # TODO: Modify to Atari 
        total_episode_return = 0.0
        while num_env_frames_v < FLAGS.total_environment_frames:
        #  print("(atari_experiment.py) num_env_frames: ", num_env_frames_v)
          level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
              (data_from_actors.level_name,) + output + (stage_op,))
          level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)
          total_episode_frames = num_env_frames_v
          for level_name, episode_return, episode_step in zip(
              level_names_v[done_v],
              infos_v.episode_return[done_v],
              infos_v.episode_step[done_v]):
            episode_frames = episode_step * FLAGS.num_action_repeats

            total_episode_return += episode_return
            tf.logging.info('Level: %s Episode return: %f after %d frames',
                            level_name, episode_return, num_env_frames_v)
            summary = tf.summary.Summary()
            summary.value.add(tag=level_name + '/episode_return',
                              simple_value=episode_return)
            summary.value.add(tag=level_name + '/episode_frames',
                              simple_value=episode_frames)
            # summary_writer.add_summary(summary, num_env_frames_v)
            # TODO: refactor better

            level_returns[level_name].append(episode_return)

          # Calculate total reward after last X frames
          if total_episode_frames % average_frames == 0:
            with open("logging.txt", "a+") as f:
              f.write("Total frames:%d total_return: %f last %d frames\n" % (num_env_frames_v, total_episode_return, average_frames))

            # tf.logging.info('total return %f last %d frames', 
            #                 total_episode_return, average_frames)
            total_episode_return = 0 
            total_episode_frames = 0

          current_episode_return_list = min(map(len, level_returns.values())) 
          if current_episode_return_list >= 1:
            no_cap = utilities_atari.compute_human_normalized_score(level_returns,
                                                            per_level_cap=None)
            cap_100 = utilities_atari.compute_human_normalized_score(level_returns,
                                                             per_level_cap=100)
            if total_episode_frames % average_frames == 0:
              with open("multi-actors-output.txt", "a+") as f:
                  # f.write("num env frames: %d\n" % num_env_frames_v)
                  f.write("total_return %f last %d frames\n" % (total_episode_return, average_frames))
                  f.write("no cap: %f after %d frames\n" % (no_cap, num_env_frames_v))
                  f.write("cap 100: %f after %d frames\n" % (cap_100, num_env_frames_v))
         #   print("(atari_experiment) No cap: ", no_cap)
         #   print("(atari_experiment) cap 100: ", cap_100)

            summary = tf.summary.Summary()
            summary.value.add(
                tag='atari/training_no_cap', simple_value=no_cap)
            summary.value.add(
                tag='atari/training_cap_100', simple_value=cap_100)
            # summary_writer.add_summary(summary, num_env_frames_v)

            # Clear level scores.
            # TODO refactor 
            level_returns = {level_name: [] for level_name in level_names}

      else:
        # Execute actors (they just need to enqueue their output).
        while True:

          session.run(enqueue_ops)

def test(action_set, level_names):
  """Test."""

  level_returns = {level_name: [] for level_name in level_names}
  with tf.Graph().as_default():
    agent = Agent(len(action_set))
    outputs = {}
    for level_name in level_names:
      env = create_atari_environment(level_name, seed=1, is_test=True)
      outputs[level_name] = build_actor(agent, env, level_name, action_set)

    with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir,
        hooks=[py_process.PyProcessHook()]) as session:
      for level_name in level_names:
        tf.logging.info('Testing level: %s', level_name)
        while True:
          done_v, infos_v = session.run((
              outputs[level_name].env_outputs.done,
              outputs[level_name].env_outputs.info
          ))
          returns = level_returns[level_name]
          returns.extend(infos_v.episode_return[1:][done_v[1:]])

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break


  no_cap = utilities_atari.compute_human_normalized_score(level_returns,
                                                  per_level_cap=None)
  cap_100 = utilities_atari.compute_human_normalized_score(level_returns,
                                                    per_level_cap=100)
  tf.logging.info('No cap.: %f Cap 100: %f', no_cap, cap_100)


ATARI_MAPPING = collections.OrderedDict([
  ('Boxing-v0', 'Boxing-v0')
    # ('Pong-v0', 'Pong-v0'),
    # ('Breakout-v0', 'Breakout-v0'),
    # ('Breakout-v0', 'Breakout-v0')
])


beam_rider_action_values = ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE', 'LEFTFIRE')
breakout_action_values = ('NOOP', 'FIRE', 'RIGHT', 'LEFT')
pong_action_values = ("NOOP", 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
qbert_action_values = ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN')
seauqest_action_values = ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 
                          'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE')
spaceInvaders_action_values = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
boxing_action_values = ('NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE')

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    action_set = boxing_action_values
    if FLAGS.mode == 'train':
      train(action_set, ATARI_MAPPING.keys()) 
    else:
      test(action_set, ATARI_MAPPING.keys())

def get_seed():
  global seed 
  seed = 1

if __name__ == '__main__':
    get_seed()
    tf.app.run()    


