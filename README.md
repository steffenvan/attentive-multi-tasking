# Bachelor Thesis - Attentive multi-tasking

## Current status
#### Boxing
- Agent stabilizes after ~50 mio frames. 
- Achieves 100 % in terms of median human normalised score. 
#### Multi-task learning
- Can now train the agent on multiple Atari games at once. 
- Train the MT-setting on the CPU-cluster. There are only 32 CPU's, how to train on the 57 Atari games? 
- Put the learner the GPU when it arrives (after Easter)

## TODO 
- Add PopArt to the architecture (by 21st of April)
- Add PNN to the architecture (by 25th of April).

## Running the Code

### Prerequisites

- [TensorFlow][tensorflow] >=1.9.0-dev20180530, the environment
- [DeepMind Lab][deepmind_lab] (if you want to try the original implementation `experiment.py`).  
- [DeepMind Sonnet][sonnet].
- [Atari](http://gym.openai.com/) 
There is a [Dockerfile][dockerfile] that serves as a reference for the
pre-requisites and commands needed to run the code.

### Single Machine Training on a Single Level

#### Training on `Boxing-v0`. 
Run the code on [Boxing](https://gym.openai.com/envs/Boxing-v0/)

Adjust the number of actors (i.e. number of environments) and batch size to
match the size of the machine it runs on. A single actor, including DeepMind
Lab, requires a few hundred MB of RAM.

### To run it in a distributed setting 
Use a multiplexer and execute the following commands in different windows. 

#### Learner (for Atari)

```sh
python atari_experiment.py --job_name=learner --task=0 --num_actors=30 \
    --level_name=Boxing-v0 --batch_size=10 --entropy_cost=0.01 \
    --learning_rate=0.0006 \
    --total_environment_frames=2000000000 --reward_clipping=soft_asymmetric
```
#### Actor(s)

```sh
for i in $(seq 0 29); do
  python atari_experiment.py --job_name=actor --task=$i \
      --num_actors=30 --level_name=Boxing-v0 &
done;
wait
```
#### Test Score 
Test it across 10 episodes using: 

```sh
python atari_experiment.py --mode=test --level_name=Boxing-v0 \
    --test_num_episodes=10
```

Training on the specific *Atari* game. Across 10 runs with different seeds
but identical hyperparameters, we observed between 45 and 50 capped human
normalized training score with different seeds (`--seed=[seed]`). Test scores
are usually an absolute of ~2% lower.


This work is an extension to [IMPALA](https://arxiv.org/abs/1804.00168]) (Espeholt et al. 2018) and [IMPALA with PopArt](https://arxiv.org/abs/1809.04474) (Hessel et al. 2018) and their recent result with improving distributed deep reinforcement learning.  

[arxiv]: https://arxiv.org/abs/1802.01561
[deepmind_lab]: https://github.com/deepmind/lab
[sonnet]: https://github.com/deepmind/sonnet
[learning_nav]: https://arxiv.org/abs/1804.00168
[generate_images]: https://deepmind.com/blog/learning-to-generate-images/
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[dmlab30]: https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
