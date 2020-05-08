<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# MuZero General

A commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to edit the [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the parameters and the game class. Please refer to the [documentation](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a model based reinforcement learning algorithm, successor of AlphaZero. It learns to master games without knowing the rules. It only knows actions and then learn to play and master the game. It is at least more efficient than similar algorithms like [AlphaZero](https://arxiv.org/abs/1712.01815), [SimPLe](https://arxiv.org/abs/1903.00374) and [World Models](https://worldmodels.github.io). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works).

## Features

* [x] Residual Network and Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Multi-Threaded/Asynchronous mode with [Ray](https://github.com/ray-project/ray)
* [x] CPU/GPU support
* [x] TensorBoard real-time monitoring
* [x] Model weights automatically saved at checkpoints
* [x] Single and multiplayer mode
* [x] Commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation)
* [x] Easily adaptable for new games
* [x] [Examples](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py) of board games, Gym and Atari games (See [list of implemented games](https://github.com/werner-duvaud/muzero-general#games-already-implemented))
* [x] [Pretrained weights](https://github.com/werner-duvaud/muzero-general/tree/master/results) available
* [ ] Windows support (Workaround: Use the [notebook](https://github.com/werner-duvaud/muzero-general/blob/master/notebook.ipynb) in Google Colab)

### Further improvements
These improvements are active research, they are personal ideas and go beyond MuZero paper. We are open to contributions and other ideas.

* [ ] Better hyperparameters tuning and improve stability
* [x] [Continuous action space](https://github.com/werner-duvaud/muzero-general/tree/continuous)
* [ ] End user tool to exploit the results
* [ ] Support stochastic environments
* [ ] Better integration with more than two player games
* [ ] Latest RL tricks (Never Give Up,  Adaptive Exploration, ...)

## Demo

All performances are tracked and displayed in real time in TensorBoard :

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
* Atari Breakout

Tests are done on Ubuntu with 16 GB RAM / Intel i7 / GTX 1050Ti Max-Q. We make sure to obtain a progression and a level which ensures that it has learned. But we do not systematically reach a human level. For certain environments, we notice a regression after a certain time. The proposed configurations are certainly not optimal and we do not focus for now on the optimization of hyperparameters. Any help is welcome.

## Code structure

![code structure](https://github.com/werner-duvaud/muzero-general/blob/master/docs/how-it-works-werner-duvaud.png)

See also: [MuZero network summary](https://github.com/werner-duvaud/muzero-general/blob/master/docs/muzero-network-werner-duvaud.png)

## Getting started
### Installation

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

### Run

```bash
python muzero.py
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

## Authors

* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
* [Contributors](https://github.com/werner-duvaud/muzero-general/graphs/contributors)


## Getting involved

* [GitHub Issues](https://github.com/werner-duvaud/muzero-general/issues): For reporting bugs.
* [Pull Requests](https://github.com/werner-duvaud/muzero-general/pulls): For submitting code contributions.
* [Discord server](https://discord.gg/GB2vwsF): For discussions about development or any general questions.
