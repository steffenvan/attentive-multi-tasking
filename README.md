# Attentive multi-tasking

## Current status
- PopArt normalization in a multi-task environment according to [(Hessel et al., 2018)](https://arxiv.org/abs/1809.04474) 

## TODO 
- Add PNN?

## Running the agent

### Dependencies

- [TensorFlow][tensorflow] >=1.9.0
- [DeepMind Sonnet][sonnet].
- [Atari](http://gym.openai.com/) 
- [DeepMind Lab][deepmind_lab] (if you want to try the original implementation `experiment.py`).  
There is a [Dockerfile][dockerfile] that serves as a reference for the
pre-requisites and commands needed to run the code.

### Local single machine training on multiple atari games. 

```
python atari_experiment.py --num_actors=10 --batch_size=5 \
    --entropy_cost=0.01 --learning_rate=0.0006 \
    --total_environment_frames=2000000000
```

### Run the agent in a distributed setting 
Use a multiplexer to execute the following commands. 

#### Learner (for Atari)

```sh
python atari_experiment.py --job_name=learner --task=0 --num_actors=30 \
    --level_name=BreakoutNoFrameSkip-v4 --batch_size=32 --entropy_cost=0.01 \
    --learning_rate=0.0006 \
    --total_environment_frames=2000000000 
```
#### Actor(s)

```sh
for i in $(seq 0 29); do
  python atari_experiment.py --job_name=actor --task=$i \
      --num_actors=30 &
done;
wait
```
#### Test Score 
Test it across 10 episodes using: 

```sh
python atari_experiment.py --mode=test --level_name=BreakoutNoFrameSkip-v4 \
    --test_num_episodes=10
```

This work is an extension to [IMPALA](https://arxiv.org/abs/1804.00168]) (Espeholt et al. 2018) and [IMPALA with PopArt](https://arxiv.org/abs/1809.04474) (Hessel et al. 2018) and their recent result with improving distributed deep reinforcement learning.  

[arxiv]: https://arxiv.org/abs/1802.01561
[deepmind_lab]: https://github.com/deepmind/lab
[sonnet]: https://github.com/deepmind/sonnet
[learning_nav]: https://arxiv.org/abs/1804.00168
[generate_images]: https://deepmind.com/blog/learning-to-generate-images/
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[dmlab30]: https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
