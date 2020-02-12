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

    logdir = FLAGS.logdir
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

  no_cap = atari_utils.compute_human_normalized_score(level_returns,
                                                  per_level_cap=None)
  cap_100 = atari_utils.compute_human_normalized_score(level_returns,
                                                    per_level_cap=100)
  tf.logging.info('No cap.: %f Cap 100: %f', no_cap, cap_100)