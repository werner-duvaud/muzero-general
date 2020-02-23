<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# MuZero General

A commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) and the associated [pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to edit the [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the parameters and the game class. Please refer to the [documentation](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a model based reinforcement learning algorithm, successor of AlphaZero. It learns to master games without knowing the rules. It only knows actions and then learn to play and master the game. It is at least more efficient than similar algorithms like [AlphaZero](https://arxiv.org/abs/1712.01815), [SimPLe](https://arxiv.org/abs/1903.00374) and [World Models](https://arxiv.org/abs/1803.10122). See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works)

## Features

* [x] Residual Network and Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Multi-Threaded with [Ray](https://github.com/ray-project/ray)
* [x] CPU/GPU support
* [x] TensorBoard real-time monitoring
* [x] Single and multiplayer mode
* [x] Commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation)
* [x] Easily adaptable for new games
* [x] [Examples](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py) of board and Gym games (See [list below](https://github.com/werner-duvaud/muzero-general#games-already-implemented-with-pretrained-network-available))
* [x] [Pretrained weights](https://github.com/werner-duvaud/muzero-general/tree/master/pretrained) available
* [ ] Improve TensorBoard logging (tree depth, ...)
* [ ] Atari games
* [ ] Appendix Reanalyse of the paper
* [ ] Windows support ([workaround by ihexx](https://github.com/ihexx/muzero-general))

## Demo

All performances are tracked and displayed in real time in tensorboard :

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/cartpole_training_summary.png)

Testing Lunar Lander :

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/lunarlander_training_preview.png)

## Games already implemented

* Cartpole
* Lunar Lander
* Connect4
* Gomoku

## Code structure

![code structure](https://github.com/werner-duvaud/muzero-general/blob/master/docs/how-it-works-werner-duvaud.png)

## Getting started
### Installation

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

### Run

Run:
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
