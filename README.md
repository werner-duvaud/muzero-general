<p>
<img src="https://img.shields.io/badge/licence-MIT-green">
<img src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# MuZero General

A flexible, commented and [documented](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) implementation of MuZero based on the Google DeepMind [paper](https://arxiv.org/abs/1911.08265) and the associated [pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py).
It is designed to be easily adaptable for every games or reinforcement learning environnements (like [gym](https://github.com/openai/gym)). You only need to edit the game file with the parameters and the game class. Please refer to the documentation and the tutorial.

MuZero is a model based reinforcement learning algorithm, successor of AlphaZero. It learns to master games whithout knowing the rules. It only know actions and then learn to play and master the game. It is at least more efficient than similar algorithms like [AlphaZero](https://arxiv.org/abs/1712.01815), [SimPLe](https://arxiv.org/abs/1903.00374) and [World Models](https://arxiv.org/abs/1803.10122).

It uses [PyTorch](https://github.com/pytorch/pytorch) and [Ray](https://github.com/ray-project/ray) for self-playing on multiple threads. A synchronous mode (easier for debug) will be released. There is a complete GPU support.
The code has three parts, muzero.py with the entry class, self-play.py with the replay-buffer and the MCTS classes, and network.py with the neural networks and the shared storage classes.

## Games already implemented with pretrained network available
* Lunar Lander
* Cartpole

## Getting started
### Installation
```bash
cd muzero-general
pip install -r requirements.txt
```

### Training
Edit the end of muzero.py :
```python
muzero = Muzero("cartpole")
muzero.train()
```
Then run :
```bash
python muzero.py
```

### Testing
Edit the end of muzero.py :
```python
muzero = Muzero("cartpole")
muzero.load_model()
muzero.test()
```
Then run :
```bash
python muzero.py
```

## Coming soon
* [ ] Convolutionnal / Atari mode
* [ ] Performance tracking
* [ ] Synchronous mode
* [ ] [Open spiel](https://github.com/deepmind/open_spiel) integration
* [ ] Checkers game
* [ ] TensorFlow mode

## Authors
* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
