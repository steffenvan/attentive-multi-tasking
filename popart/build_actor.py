import tensorflow as tf
nest = tf.contrib.framework.nest
import sys
sys.path.append('../')
from utils import atari_utils
from .flags import *
import collections

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')

ActorOutputFeedForward = collections.namedtuple(
    'ActorOutputFeedForward', 'level_name level_name_as_idx env_outputs agent_outputs')

# Used to map the level name -> number for indexation
game_id = {}
games = atari_utils.ATARI_GAMES.keys()
for i, game in enumerate(games):
  game_id[game] = i

def build_actor(agent, env, level_name, action_set):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  # initial_agent_state = agent.initial_state(1)

  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output = agent((initial_action, 
                              nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output),
                              tf.constant(game_id[level_name], shape=[1])))
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
    agent_output = agent((action, batched_env_output, tf.constant(game_id[level_name], shape=[1])))

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
    
    output = ActorOutputFeedForward(
        level_name=level_name, 
        level_name_as_idx=game_id[level_name],
        env_outputs=full_env_outputs,
        agent_outputs=full_agent_outputs)
    # No backpropagation should be done here.

    return nest.map_structure(tf.stop_gradient, output)