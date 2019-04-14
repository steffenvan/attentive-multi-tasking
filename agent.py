from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sonnet as snt
import tensorflow as tf

import collections
import contextlib
import functools
import os
import sys
import vtrace
from utilities_atari import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss

nest = tf.contrib.framework.nest
flags = tf.app.flags

FLAGS = tf.app.flags.FLAGS

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')

flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 1, 'Number of actors.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 20, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

class Agent(snt.RNNCore):

    def __init__(self, num_actions):
        super(Agent, self).__init__(name="agent")

        self._num_actions = num_actions
        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        init_state = self._core.zero_state(batch_size, tf.float32)
        return init_state

    def _instruction(self, instruction):
        splitted = tf.string_split(instruction)
        dense = tf.sparse_tensor_to_dense(splitted, default_value='')
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(dense, '')), axis=1)

        # To int64 hash buckets. Small risk of having collisions. Alternatively, a
        # vocabulary can be used.
        num_hash_buckets = 1000
        buckets = tf.string_to_hash_bucket_fast(dense, num_hash_buckets)
        # Embed the instruction. Embedding size 20 seems to be enough.
        embedding_size = 20
        embedding = snt.Embed(num_hash_buckets, embedding_size)(buckets)

        # Pad to make sure there is at least one output.
        padding = tf.to_int32(tf.equal(tf.shape(embedding)[1], 0))
        embedding = tf.pad(embedding, [[0, 0], [0, padding], [0, 0]])

        core = tf.contrib.rnn.LSTMBlockCell(64, name='language_lstm')
        output, _ = tf.nn.dynamic_rnn(core, embedding, length, dtype=tf.float32)

        # Return last output.
        return tf.reverse_sequence(output, length, seq_axis=1)[:, 0]
            
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output
        # print("Instruction is: ", instruction)

        # Convert to floats.
        frame = tf.to_float(frame)

        frame /= 255
        with tf.variable_scope('convnet'):
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

                # # Residual block(s).
                # for j in range(num_blocks):
                #     with tf.variable_scope('residual_%d_%d' % (i, j)):
                #         block_input = conv_out
                #         conv_out = tf.nn.relu(conv_out)
                #         conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                #         conv_out = tf.nn.relu(conv_out)
                #         conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
                #         conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)

        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        # print("Clipped reward shape: ", tf.shape(clipped_reward))
        # print("One hot shape: ", tf.shape(one_hot_last_action))
        output = tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)
        return output 

    # This is the LSTM that outputs the value function and the policy
    def _head(self, core_output):
        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(
            core_output)
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1,
                                    output_dtype=tf.int32)

        # sess = tf.InteractiveSession()

        new_action = tf.squeeze(new_action, 1, name='new_action')
        # tf.print(new_action, [new_action], message="this is after squeezing")


        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        squeezed = nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        # print("squeezed (_build): ", squeezed)
        return squeezed, core_state

    # Just need sto know if the episode has ended. 
    # This is used in build by 
    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs
        # Instructions are in here <-- coming from _torso, which returns a 
        # tensor of convolutional output, one_hot_action, 
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self.initial_state(tf.shape(actions)[1])
        # self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
        # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)
        # print("core output list (unroll): ", len(core_output_list))
        output = snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state
        # print("Output is (unroll): ", output)
        return output

def build_actor(agent, env, level_name, action_set):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  # print("initial env output: ", initial_env_output)
  # print("initial env state: ", initial_env_state)
  initial_agent_state = agent.initial_state(1)
  # print("initial agent state: ", initial_agent_state)
  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output, _ = agent(
      (initial_action,
       nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
      initial_agent_state)
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
      create_state, (initial_env_state, initial_env_output, initial_agent_state,
                     initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_state, agent_output = input_

    # Run agent.
    action = agent_output[0]
    # action = tf.Print(action.shape, [action], "Action is: ")
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output, agent_state = agent((action, batched_env_output), agent_state)

    # Convert action index to the native action.
    action = agent_output[0][0]
    # Changed from 
    # action = tf.Print(agent_output[0][0], [agent_output[0][0]], "Action is: ")
    raw_action = action

    # raw_action = tf.Print(tf.shape(raw_action), [tf.shape(raw_action)], "Raw action shape is: ")


    env_output, env_state = env.step(raw_action, env_state)

    return env_state, env_output, agent_state, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  _, first_env_output, first_agent_state, first_agent_output = first_values

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, _, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    output = ActorOutput(
        level_name=level_name, agent_state=first_agent_state,
        env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)
    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)

def build_learner(agent, agent_state, env_outputs, agent_outputs):
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
  learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                    agent_state)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, the environment outputs at time step `t` are the inputs that
  # lead to the learner_outputs at time step `t`. After the following shifting,
  # the actions in agent_outputs and learner_outputs at time step `t` is what
  # leads to the environment outputs at time step `t`.
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)
  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

  # if FLAGS.reward_clipping == 'abs_one':
  #   clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  # elif FLAGS.reward_clipping == 'soft_asymmetric':
  #   squeezed = tf.tanh(rewards / 5.0)
  #   # Negative rewards are given less weight than positive rewards.
  #   clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.to_float(~done) * FLAGS.discounting

  # Compute V-trace returns and weights.
  # Note, this is put on the CPU because it's faster than on GPU. It can be
  # improved further with XLA-compilation or with a custom TensorFlow operation.
  with tf.device('/cpu'):
    vtrace_returns = vtrace.from_logits(
        behaviour_policy_logits=agent_outputs.policy_logits,
        target_policy_logits=learner_outputs.policy_logits,
        actions=agent_outputs.action,
        discounts=discounts,
        rewards=rewards,
        values=learner_outputs.baseline,
        bootstrap_value=bootstrap_value)

  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  total_loss = compute_policy_gradient_loss(
      learner_outputs.policy_logits, agent_outputs.action,
      vtrace_returns.pg_advantages)
  total_loss += FLAGS.baseline_cost * compute_baseline_loss(
      vtrace_returns.vs - learner_outputs.baseline)
  total_loss += FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits)

  # Optimization
  num_env_frames = tf.train.get_global_step()
  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)
  optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                        FLAGS.momentum, FLAGS.epsilon)
  train_op = optimizer.minimize(total_loss)

  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(
        FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)

  # Adding a few summaries.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)
  print("(atari_experiment.py) done: ", done)
  print("(atari_experiment.py) infos: ", done)
  return done, infos, num_env_frames_and_train
