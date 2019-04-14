# My bachelor thesis  - Attentive multi-tasking

## Current status
- Now running on Atari games (only one at a time though).
- The reward quickly converges to a fixed (non-optimal) value after ~1600 frames. 

## TODO 
- Modify `environments.py` to render the agents actions (by 11th of April).
- Train on the 6 Atari games and obtain the results (by 16th of April).
- Add PNN to the architecture (by 25th of April).
- More...


## Running the Code

### Prerequisites

- [TensorFlow][tensorflow] >=1.9.0-dev20180530, the environment
- [DeepMind Lab][deepmind_lab] (if you want to try the original implementation `experiment.py`).  
- [DeepMind Sonnet][sonnet].
- [Atari](http://gym.openai.com/) 
We include a [Dockerfile][dockerfile] that serves as a reference for the
prerequisites and commands needed to run the code.

### Single Machine Training on a Single Level

#### Training on `Pong-v0`. 
Run the code on [Pong](https://gym.openai.com/envs/Pong-v0/):
```sh
python atari_experiment.py --num_actors=8 --batch_size=4
```

Adjust the number of actors (i.e. number of environments) and batch size to
match the size of the machine it runs on. A single actor, including DeepMind
Lab, requires a few hundred MB of RAM.


#### Learner (for Atari)

```sh
python atari_experiment.py --job_name=learner --task=0 --num_actors=16 \
    --level_name=Pong-v0 --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric
```

#### Test Score (Doesn't work for Atari at the moment)

```sh
python atari_experiment.py --mode=test --level_name=Pong-v0 --dataset_path=[...] \
    --test_num_episodes=10
```

#### Actor(s)

```sh
for i in $(seq 0 15); do
  python atari_experiment.py --job_name=actor --task=$i \
      --num_actors=16 --level_name=Pong-v0 &
done;
wait
```

### Distributed Training on DMLab-30

Training on the full *Atari*. Across 10 runs with different seeds
but identical hyperparameters, we observed between 45 and 50 capped human
normalized training score with different seeds (`--seed=[seed]`). Test scores
are usually an absolute of ~2% lower.


This work is an extension to Deepmind recent result (Epseholt et al. 2018): 

[arxiv]: https://arxiv.org/abs/1802.01561
[deepmind_lab]: https://github.com/deepmind/lab
[sonnet]: https://github.com/deepmind/sonnet
[learning_nav]: https://arxiv.org/abs/1804.00168
[generate_images]: https://deepmind.com/blog/learning-to-generate-images/
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[dmlab30]: https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
