# Bachelor Thesis - Attentive multi-tasking

## Current status
- Showing results with Boxing.
- Agent stabilizes after ~50 mio frames. 

## TODO 
- Train on the 6 Atari games and obtain the results (by 16th of April).
- Add PNN to the architecture (by 25th of April).
- Add PopArt to the architecture (by 27th of April)


## Running the Code

### Prerequisites

- [TensorFlow][tensorflow] >=1.9.0-dev20180530, the environment
- [DeepMind Lab][deepmind_lab] (if you want to try the original implementation `experiment.py`).  
- [DeepMind Sonnet][sonnet].
- [Atari](http://gym.openai.com/) 
We include a [Dockerfile][dockerfile] that serves as a reference for the
prerequisites and commands needed to run the code.

### Single Machine Training on a Single Level

#### Training on `Boxing-v0`. 
Run the code on [Boxing](https://gym.openai.com/envs/Boxing-v0/)

Adjust the number of actors (i.e. number of environments) and batch size to
match the size of the machine it runs on. A single actor, including DeepMind
Lab, requires a few hundred MB of RAM.

### To run it in a distributed setting 
I recommend using tmux or any multiplexer to run it easily. 

##### Start with one window:
#### Learner (for Atari)

```sh
python atari_experiment.py --job_name=learner --task=0 --num_actors=24 \
    --level_name=Boxing-v0 --batch_size=8 --entropy_cost=0.01 \
    --learning_rate=0.0006 \
    --total_environment_frames=2000000000 --reward_clipping=soft_asymmetric
```
##### And another
#### Actor(s)

```sh
for i in $(seq 0 23); do
  python atari_experiment.py --job_name=actor --task=$i \
      --num_actors=24 --level_name=Boxing-v0 &
done;
wait
```
#### Test Score 

```sh
python atari_experiment.py --mode=test --level_name=Boxing-v0 \
    --test_num_episodes=10
```

Training on the specific *Atari* game. Across 10 runs with different seeds
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
