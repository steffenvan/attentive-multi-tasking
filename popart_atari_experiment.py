" IMPALA for Atari" 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import os
import sys
#from more_itertools import one 
import utilities_atari
import atari_environment
import numpy as np
import py_process
import sonnet as snt
import tensorflow as tf
import vtrace
from agent import agent_factory

try:
  import dynamic_batching

except tf.errors.NotFoundError:
  tf.logging.warning('Running without dynamic batching.')

from six.moves import range

nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', 'popart-multi', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Environment settings
flags.DEFINE_integer('width', 84, 'Width of observation')
flags.DEFINE_integer('height', 84, 'Height of observation')

# Train settings
flags.DEFINE_integer('total_environment_frames', int(5e8),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 10, 'Number of actors.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 80, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('queue_capacity', 1, 'tensorflow queue capacity')
flags.DEFINE_string('level_name', 'SeaquestNoFrameskip-v4', 'level name')
flags.DEFINE_string('agent_name', 'FeedForwardAgent', 'Which learner to use')

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

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')

ActorOutputFeedForward = collections.namedtuple(
    'ActorOutputFeedForward', 'level_name env_outputs agent_outputs')


# Used to map the level name -> number for indexation
game_id = {}
games = utilities_atari.ATARI_GAMES.keys()
for i, game in enumerate(games):
  game_id[game] = i

print("GAMES: ", game_id)
def is_single_machine():
    return FLAGS.task == -1

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


def build_actor(agent, env, level_name, action_set):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  # initial_agent_state = agent.initial_state(1)

  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output = agent((initial_action, nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)))
  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state, (initial_env_state, initial_env_output, initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_output = input_

    # Run agent.
    action = agent_output[0]
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output = agent((action, batched_env_output))

    # Convert action index to the native action.
    action = agent_output[0][0]
    raw_action = tf.gather(action_set, action)
    env_output, env_state = env.step(raw_action, env_state)

    return env_state, env_output, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  _, first_env_output, first_agent_output = first_values

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    # first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    # Removed for now 
    # Use the extra state information if it's the LSTM agent
    # if hasattr(initial_agent_state, 'c') and hasattr(initial_agent_state, 'h'):
    #   output = ActorOutput(
    #       level_name=level_name, agent_state=first_agent_state,
    #       env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)
    
    output = ActorOutputFeedForward(
        level_name=level_name, 
        env_outputs=full_env_outputs,
        agent_outputs=full_agent_outputs)
    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)

def build_learner(agent, env_outputs, agent_outputs, env_id):
  """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """

  # Need to map the game name, e.g 'BreakoutNoFrameSkip-v4' to an integer.  
  def get_single_game_info(_tuple):
    single_env_id, game_info = _tuple
    return game_info[single_env_id]

  # Retrieve the specific games in the current batch. 
  def get_batch_value(batch):
    return tf.map_fn(get_single_game_info, (env_id, batch), dtype=tf.float32)

  learner_outputs = agent.unroll(agent_outputs.action, env_outputs)
  un_normalized_vf = learner_outputs.un_normalized_vf
  normalized_vf   = learner_outputs.normalized_vf
  

  game_specific_un_normalized_vf = tf.map_fn(get_batch_value, un_normalized_vf, dtype=tf.float32)
  # game_specific_un_normalized_vf = tf.reduce_sum(game)
  game_specific_normalized_vf   = tf.map_fn(get_batch_value, normalized_vf, dtype=tf.float32)

  # Ensure the learner separates the value functions for each game. 
  # According to equation (10) in (Hessel et al., 2018). 
  learner_outputs = learner_outputs._replace(un_normalized_vf=game_specific_un_normalized_vf,
                                             normalized_vf=game_specific_normalized_vf) 
  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.un_normalized_vf[-1]
 
  # At this point, the environment outputs at time step `t` are the inputs that
  # lead to the learner_outputs at time step `t`. After the following shifting,
  # the actions in agent_outputs and learner_outputs at time step `t` is what
  # leads to the environment outputs at time step `t`.
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)
  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

  if FLAGS.reward_clipping == 'abs_one':
    clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  elif FLAGS.reward_clipping == 'soft_asymmetric':
    squeezed = tf.tanh(rewards / 5.0)
    # Negative rewards are given less weight than positive rewards.
    clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.to_float(~done) * FLAGS.discounting
  game_specific_mean = tf.gather(agent._mean, env_id)
  game_specific_std = tf.gather(agent._std, env_id)

  # Compute V-trace returns and weights.
  # Note, this is put on the CPU because it's faster than on GPU. It can be
  # improved further with XLA-compilation or with a custom TensorFlow operation.
  with tf.device('/cpu'):
    vtrace_returns = vtrace.from_logits(
        behaviour_policy_logits=agent_outputs.policy_logits,
        target_policy_logits=learner_outputs.policy_logits,
        actions=agent_outputs.action,
        discounts=discounts,
        rewards=clipped_rewards,
        un_normalized_values=learner_outputs.un_normalized_vf,
        normalized_values=learner_outputs.normalized_vf,
        mean=game_specific_mean,
        std=game_specific_std,
        bootstrap_value=bootstrap_value)

  # First term of equation (7) in (Hessel et al., 2018)
  normalized_vtrace = (vtrace_returns.vs - game_specific_mean) / game_specific_std

  normalized_vtrace = nest.map_structure(tf.stop_gradient, normalized_vtrace)


  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  total_loss = compute_policy_gradient_loss(
      learner_outputs.policy_logits, agent_outputs.action,
      vtrace_returns.pg_advantages)

  baseline_loss = compute_baseline_loss(
       normalized_vtrace - learner_outputs.normalized_vf)
  # Using the average GvT 
  baseline_loss = tf.divide(baseline_loss, FLAGS.unroll_length)

  total_loss += FLAGS.baseline_cost * baseline_loss
  total_loss += FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits)

  # Optimization
  num_env_frames = tf.train.get_global_step()

  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)

  optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                        FLAGS.momentum, FLAGS.epsilon)

  # Use reward clipping for atari games only 
  if FLAGS.gradient_clipping > 0.0:
    # gradients, variables = zip(*optimizer.compute_gradients(total_loss))
    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
    # print("VARIABLES: ", variables)
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gradient_clipping)
    print("GRADIENTS: ", gradients)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
  else:
    train_op = optimizer.minimize(total_loss)

  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(
        FLAGS.batch_size * FLAGS.unroll_length)

  # Adding a few summaries.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)

  return (done, infos, num_env_frames_and_train) + (agent.update_moments(vtrace_returns.vs, env_id))


def create_atari_environment(env_id, seed, is_test=False):

  config = {
      'width': FLAGS.width,
      'height': FLAGS.height
  }

  if is_test:
    config['allowHoldOutLevels'] = 'true'
    # Mixer seed for evalution, see
    # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
    config['mixerSeed'] = 0x600D5EED

  process = py_process.PyProcess(atari_environment.PyProcessAtari, env_id, config)
  proxy_env = atari_environment.FlowEnvironment(process.proxy)
  return proxy_env

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
  config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters) 
  if is_learner:
    config.gpu_options.allow_growth = True
  
  Agent = agent_factory(FLAGS.agent_name)
  with tf.Graph().as_default():
    env = create_atari_environment(level_names[0], seed=1)
    agent = Agent(len(action_set))
    structure = build_actor(agent, env, level_names[0], action_set)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]    
    shapes = [t.shape.as_list() for t in flattened_structure]

  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/gpu'), \
       pin_global_variables(global_variable_device):
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

    # Create Queue and Agent on the learner.
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(FLAGS.queue_capacity, dtypes, shapes, shared_name='buffer')
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
        # specific_action_set = atari_environment.get_action_set(level_name)
        # tf.logging.info('Current game: {} with action set: {}'.format(level_name, specific_action_set))
        actor_output = build_actor(agent, env, level_name, action_set)
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

        # Returns an ActorOutput tuple -> (level name, agent_state, env_outputs, agent_output)
        data_from_actors = nest.pack_sequence_as(structure, area.get())

        # Converting the tensor of level names to a normal integer used for indexing. 
        level_names_index = tf.map_fn(lambda y: tf.py_function(lambda x: game_id[x.numpy()], [y], Tout=tf.int32), data_from_actors.level_name, dtype=tf.int32)
        level_names_index = tf.reshape(level_names_index, [FLAGS.batch_size])
        # If LSTM agent, we use the hidden states
        # if hasattr(data_from_actors, 'agent_state'):
        #   agent_state = data_from_actors.agent_state

        # Unroll agent on sequence, create losses and update ops.
        output = build_learner(agent,
                               data_from_actors.env_outputs,
                               data_from_actors.agent_outputs,
                               level_names_index)
        
    # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
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
        level_returns = {level_name: [] for level_name in level_names}
        total_level_returns = {level_name: 0.0 for level_name in level_names}
        summary_dir = os.path.join(FLAGS.logdir, "logging")
        summary_writer = tf.summary.FileWriterCache.get(summary_dir)

        # Prepare data for first run.
        session.run_step_fn(
            lambda step_context: step_context.session.run(stage_op))

        # Execute learning and track performance.
        num_env_frames_v = 0
        total_episode_frames = 0
        
        # Log the total return every *average_frames*.  
        average_frames = 24000 
        # total_episode_return = 0.0
        while num_env_frames_v < FLAGS.total_environment_frames:
          level_names_v, done_v, infos_v, num_env_frames_v, mean, _, std, _ = session.run(
              (data_from_actors.level_name,) + output + (agent._std, ) + (stage_op,))

          level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)
          # print("LEVEL NAMES: ", level_names_v)
          total_episode_frames = num_env_frames_v

          for level_name, episode_return, episode_step, acc_episode_reward, acc_episode_step in zip(
              level_names_v[done_v],
              infos_v.episode_return[done_v],
              infos_v.episode_step[done_v],
              infos_v.acc_episode_reward[done_v],
              infos_v.acc_episode_step[done_v]):

            episode_frames = episode_step * FLAGS.num_action_repeats

            tf.logging.info('Level: %s Episode return: %f Acc return: %f after %d frames',
                            level_name, episode_return, acc_episode_reward, num_env_frames_v)
            # print('game: {} mean: {} \n std: {}'.format(game_id[level_name], mean[game_id[level_name]], std[game_id[level_name]]))
            summary = tf.summary.Summary()
            summary.value.add(tag=level_name + '/episode_return',
                              simple_value=episode_return)
            summary.value.add(tag=level_name + '/episode_frames',
                              simple_value=episode_frames)
            summary.value.add(tag=level_name + '/acc_episode_return',
                                simple_value=acc_episode_reward)
            summary.value.add(tag=level_name + '/acc_episode_frames',
                                simple_value=acc_episode_step)
            summary.value.add(tag=level_name + '/env_mean', 
                              simple_value=mean[game_id[level_name]])
            summary.value.add(tag=level_name + '/env_std',
                              simple_value=std[game_id[level_name]])
            summary_writer.add_summary(summary, num_env_frames_v)

            level_returns[level_name].append(episode_return)

            # tf.logging.info('total return %f last %d frames', 
            #                 total_episode_return, average_frames)
          if min(map(len, level_returns.values())) >= 1:
            # for level_name
            # print("Game: {} episode_return: {}".format(level_name, level_returns[level_name]))
            no_cap = utilities_atari.compute_human_normalized_score(level_returns,
                                                            per_level_cap=None)
            cap_100 = utilities_atari.compute_human_normalized_score(level_returns,
                                                             per_level_cap=100)
            # if total_episode_frames % average_frames == 0:
            #   with open("multi-actors-output.txt", "a+") as f:
            #       # f.write("total_return %f last %d frames\n" % (total_episode_return, average_frames))
            #       f.write("no cap: %f after %d frames\n" % (no_cap, num_env_frames_v))
            #       f.write("cap 100: %f after %d frames\n" % (cap_100, num_env_frames_v))

            summary = tf.summary.Summary()
            summary.value.add(
                tag=(level_name + '/training_no_cap'), simple_value=no_cap)
            summary.value.add(
                tag=(level_name + '/training_cap_100'), simple_value=cap_100)

            summary_writer.add_summary(summary, num_env_frames_v)

            # Clear level scores.
            # Add the episode returns before resetting for logging purposes. 
            # level_returns = {level_name: sum(level_returns[level_name]) for level_name in level_names}
            # for level_name in level_names:
            #   total_level_returns[level_name] += level_returns[level_name]
                      
            level_returns = {level_name: [] for level_name in level_names}

          # Calculate total reward after last X frames
          # if total_episode_frames % average_frames == 0:
          #   for level_name in level_names: 
          #     outputs = os.path.join("outputs", level_name + ".txt")
          #     with open(outputs, "a+") as f:
          #       f.write("%s: total episode return: %f last %d frames\n" % (level_name, total_level_returns[level_name], num_env_frames_v))
          #     total_level_returns[level_name] = 0.0
          #   total_episode_frames = 0

      else:
        # Execute actors (they just need to enqueue their output).
        while True:
          session.run(enqueue_ops)

def test(action_set, level_names):
  """Test."""

  Agent = agent_factory(FLAGS.agent_name)
  level_returns = {level_name: [] for level_name in level_names}
  with tf.Graph().as_default():
    agent = Agent(len(action_set))
    outputs = {}
    for level_name in level_names:
      env = create_atari_environment(level_name, seed=1, is_test=True)
      outputs[level_name] = build_actor(agent, env, level_name, action_set)

    logdir = "multi-task"
    # tf.logging.info("LOGDIR IS: {}".format(logdir))
    with tf.train.SingularMonitoredSession(
        checkpoint_dir=logdir,
        hooks=[py_process.PyProcessHook()]) as session:
      for level_name in level_names:
        tf.logging.info('Testing level: %s', level_name)
        while True:
          done_v, infos_v = session.run((
              outputs[level_name].env_outputs.done,
              outputs[level_name].env_outputs.info
          ))
          returns = level_returns[level_name]
          if infos_v.episode_return[1:][done_v[1:]]: 
            tf.logging.info("Return: {}".format(level_returns[level_name]))
          returns.extend(infos_v.episode_return[1:][done_v[1:]])

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break

  no_cap = utilities_atari.compute_human_normalized_score(level_returns,
                                                  per_level_cap=None)
  cap_100 = utilities_atari.compute_human_normalized_score(level_returns,
                                                    per_level_cap=100)
  tf.logging.info('No cap.: %f Cap 100: %f', no_cap, cap_100)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # action_set = atari_environment.ATARI_ACTION_SET
    test_action_set = atari_environment.get_action_set(FLAGS.level_name)
    action_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15 ,16 ,17]
    if FLAGS.mode == 'train':
      train(action_set, games) 
    else:
      test(test_action_set, [FLAGS.level_name])

if __name__ == '__main__':
    tf.app.run()    
