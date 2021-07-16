![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(soon)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![license MIT](https://img.shields.io/badge/licence-MIT-green)
[![discord badge](https://img.shields.io/badge/discord-join-6E60EF)](https://discord.gg/GB2vwsF)

# MuZero General

A commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) (Nov 2019) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to add a [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the hyperparameters and the game class. Please refer to the [documentation](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a state of the art RL algorithm for board games (Chess, Go, ...) and Atari games.
It is the successor to [AlphaZero](https://arxiv.org/abs/1712.01815) but without any knowledge of the environment underlying dynamics. MuZero learns a model of the environment and uses an internal representation that contains only the useful information for predicting the reward, value, policy and transitions. MuZero is also close to [Value prediction networks](https://arxiv.org/abs/1707.03497). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works).

## Disclaimer

**This repository is fork** of base [MuZero implementation](https://github.com/werner-duvaud/muzero-general). Main target of fork allow higher customiztion and simple usage as library, more simular to OpenAI [stable-baseelines](https://stable-baselines3.readthedocs.io/en/master/).

## Getting started

### Installation

```bash
pip install muzero-baseline
```

### Preapare game and configuration

```py
from muzero_baseline.games.abstract_game import AbstractGame

# Create config for agent and network

class MuZeroConfig:
  def __init__(self): 
    self.seed = 0  # Seed for numpy, torch and the game
    self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

    ### Game
    self.observation_shape = (1, 1, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
    self.players = list(range(1))  # List of players. You should only edit the length
    self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

    # ...

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed = None):
        
        self.env = gym.make("CartPole-v1")

        if seed is not None:
            self.env.seed(seed)

    # ...
```

More examples of configs and games can be found in [games folder](https://github.com/werner-duvaud/muzero-general/tree/master/games), you can adapt them for you needs.

More information is also available in [wiki](https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization).

### Initialize MuZero instance

```py
from muzero_baseline.muzero import MuZero

# Initialize config
config = MuZeroConfig()
# Game object will be initialized in each thread separetly
mz = MuZero(TraidingGame, config)
```

### Train agent

```py
mz.train()
```

During training agent will save metrics and chekpoints of netowork and replay buffer in `results` folder.

### Metrics can accessed though tensorboard

```py
%load_ext tensorboard
%tensorboard --logdir ./results 
```

### Test agent

```py
mz.test()
```

For test in same thread

```py
mz.test_direct()
```

### Load existing model

```py
mz.load_model(
    checkpoint_path = 'results/2021-07-15--16-06-15/model.checkpoint', 
    replay_buffer_path = 'results/2021-07-15--16-06-15/replay_buffer.pkl'
)
```

## Features

* [x] Residual Network and Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Multi-Threaded/Asynchronous/[Cluster](https://docs.ray.io/en/latest/cluster-index.html) with [Ray](https://github.com/ray-project/ray)
* [X] Multi GPU support for the training and the selfplay
* [x] TensorBoard real-time monitoring
* [x] Model weights automatically saved at checkpoints
* [x] Single and two player mode
* [x] Commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation)
* [x] Easily adaptable for new games
* [x] [Examples](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py) of board games, Gym and Atari games (See [list of implemented games](https://github.com/werner-duvaud/muzero-general#games-already-implemented))
* [x] [Pretrained weights](https://github.com/werner-duvaud/muzero-general/tree/master/results) available
* [ ] Windows support (Experimental / Workaround: Use the [notebook](https://github.com/werner-duvaud/muzero-general/blob/master/notebook.ipynb) in [Google Colab](https://colab.research.google.com))

### Further improvements

These improvements are active research, they are personal ideas and go beyond MuZero paper. We are open to contributions and other ideas.

* [x] [Hyperparameter search](https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization)
* [x] [Continuous action space](https://github.com/werner-duvaud/muzero-general/tree/continuous)
* [x] [Tool to understand the learned model](https://github.com/werner-duvaud/muzero-general/blob/master/diagnose_model.py)
* [ ] Support of stochastic environments
* [ ] Support of more than two player games
* [ ] RL tricks (Never Give Up,  Adaptive Exploration, ...)

## Demo

All performances are tracked and displayed in real time in [TensorBoard](https://www.tensorflow.org/tensorboard) :

![cartpole training summary](https://github.com/werner-duvaud/muzero-general/blob/master/docs/cartpole-training-summary.png)

Testing Lunar Lander :

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/lunarlander-training-preview.png)

## Games already implemented

* Cartpole      (Tested with the fully connected network)
* Lunar Lander  (Tested in deterministic mode with the fully connected network)
* Gridworld     (Tested with the fully connected network)
* Tic-tac-toe   (Tested with the fully connected network and the residual network)
* Connect4      (Slightly tested with the residual network)
* Gomoku
* Twenty-One / Blackjack    (Tested with the residual network)
* Atari Breakout

Tests are done on Ubuntu with 16 GB RAM / Intel i7 / GTX 1050Ti Max-Q. We make sure to obtain a progression and a level which ensures that it has learned. But we do not systematically reach a human level. For certain environments, we notice a regression after a certain time. The proposed configurations are certainly not optimal and we do not focus for now on the optimization of hyperparameters. Any help is welcome.

## Code structure

![code structure](https://github.com/werner-duvaud/muzero-general/blob/master/docs/code-structure-werner-duvaud.png)

Network summary:

<p align="center">
<a href="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png">
<img src="https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png" width="250"/>
</a>
</p>



## Authors

* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
* [Contributors](https://github.com/werner-duvaud/muzero-general/graphs/contributors)

Please use this bibtex if you want to cite this repository (master branch) in your publications:

```bash
@misc{muzero-general,
  author       = {Werner Duvaud, Aurèle Hainaut},
  title        = {MuZero General: Open Reimplementation of MuZero},
  year         = {2019},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/werner-duvaud/muzero-general}},
}
```

## Getting involved

* [GitHub Issues](https://github.com/werner-duvaud/muzero-general/issues): For reporting bugs.
* [Pull Requests](https://github.com/werner-duvaud/muzero-general/pulls): For submitting code contributions.
* [Discord server](https://discord.gg/GB2vwsF): For discussions about development or any general questions.
