<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# MuZero General

A flexible, commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) and the associated [pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environments (like [gym](https://github.com/openai/gym)). You only need to edit the [game file](https://github.com/werner-duvaud/muzero-general/tree/master/games) with the parameters and the game class. Please refer to the documentation and the [example](https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py).

MuZero is a model based reinforcement learning algorithm, successor of AlphaZero. It learns to master games without knowing the rules. It only knows actions and then learn to play and master the game. It is at least more efficient than similar algorithms like [AlphaZero](https://arxiv.org/abs/1712.01815), [SimPLe](https://arxiv.org/abs/1903.00374) and [World Models](https://arxiv.org/abs/1803.10122).

It uses [PyTorch](https://github.com/pytorch/pytorch) and [Ray](https://github.com/ray-project/ray) for running the different components simultaneously. There is a complete GPU support.

There are four components which are classes that run simultaneously in a dedicated thread.
The `shared storage` holds the latest neural network weights, the `self-play` uses those weights to generate self-play games and store them in the `replay buffer`. Finally, those played games are used to `train` a network and store the weights in the shared storage. The circle is complete. See [How it works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works)

Those components are launched and managed from the MuZero class in `muzero.py` and the structure of the neural network is defined in `models.py`.

All performances are tracked and displayed in real time in tensorboard.

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/cartpole_training_summary.png)

## Games already implemented with pretrained network available
* Lunar Lander
* Cartpole

![lunarlander training preview](https://github.com/werner-duvaud/muzero-general/blob/master/docs/lunarlander_training_preview.png)

## Getting started
### Installation
```bash
cd muzero-general
pip install -r requirements.txt
```

### Training
Edit the end of muzero.py:
```python
muzero = Muzero("cartpole")
muzero.train()
```
Then run:
```bash
python muzero.py
```
To visualize the training results, run in a new bash:
```bash
tensorboard --logdir ./
```

### Testing
Edit the end of muzero.py:
```python
muzero = Muzero("cartpole")
muzero.load_model()
muzero.test()
```
Then run:
```bash
python muzero.py
```

## Coming soon
* [ ] Atari mode with residual network
* [ ] Live test policy & value tracking 
* [ ] [Open spiel](https://github.com/deepmind/open_spiel) integration
* [ ] Checkers game
* [ ] TensorFlow mode

## Authors
* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
