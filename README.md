![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(soon)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![license MIT](https://img.shields.io/badge/licence-MIT-green)
[![discord badge](https://img.shields.io/badge/discord-join-6E60EF)](https://discord.gg/GB2vwsF)

![ci-testing workflow](https://github.com/werner-duvaud/muzero-general/workflows/CI%20testing%20continuous/badge.svg)

# Continuous MuZero General

Adaptation of MuZero General for continuous action space environments like [MuJoCo](https://github.com/openai/mujoco-py) and [PyBullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet).

## Features

* [x] Multi-dimension continuous action space
* [x] Fully connected network and Residual Network

## Demo

Testing MuJoCo InvertedDoublePendulum-v2:

![inverted-double-pendulum training preview](https://github.com/werner-duvaud/muzero-general/blob/continuous/docs/mujoco-inverted-double-pendulum-training-preview.png)

## Games already implemented

* MuJoCo InvertedPendulum-v2      (Tested with the fully connected network)
* MuJoCo InvertedDoublePendulum-v2      (Tested with the fully connected network)
* MuJoCo Swimmer-v2      (Tested with the fully connected network)
* MuJoCo Hopper-v2
* MuJoCo Walker2d-v2
* PyBullet InvertedPendulumBulletEnv-v0     (Tested with the fully connected network)
* PyBullet InvertedDoublePendulumBulletEnv-v0     (Tested with the fully connected network)
* PyBullet HopperBulletEnv-v0


Tests are done on Ubuntu with 16 GB RAM / Intel i7 / GTX 1050Ti Max-Q. We make sure to obtain a progression and a level which ensures that it has learned. But we do not systematically reach a human level. For certain environments, we notice a regression after a certain time. The proposed configurations are certainly not optimal and we do not focus for now on the optimization of hyperparameters. Any help is welcome.

## Getting started
### Installation

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general
git checkout continuous

pip install -r requirements.txt
```

For MuJoCo environments, follow the [instructions here](https://github.com/openai/mujoco-py#install-and-use-mujoco-py) for the installation.

### Run

```bash
python muzero.py
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

## Authors

* Xuxi Yang
* Werner Duvaud
* Aurèle Hainaut
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
