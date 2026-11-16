# MuZero General

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-306998)](requirements.txt)
[![MIT license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An educational and commented implementation of [MuZero](https://arxiv.org/abs/1911.08265), based on DeepMind's 2019 paper and [original pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py), built with PyTorch and Ray.

MuZero builds on [AlphaZero](https://arxiv.org/abs/1712.01815) by learning the dynamics used for planning from experience. Its network represents observations as a latent state, predicts how actions change that state, and estimates rewards, values and policies. Monte Carlo tree search (MCTS) uses these predictions to choose actions.

For an introduction, watch the [MuZero explanatory video](https://youtu.be/We20YSAJZSE) or read [How MuZero works](https://github.com/werner-duvaud/muzero-general/wiki/How-MuZero-works) and the [project wiki](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation).

The implementation supports fully connected and residual networks, parallel self-play, prioritized replay, reanalysis, and single-player environments with continuous actions. See [Continuous version](#continuous-version) for the branch-specific policy and action conventions.

To adapt it to a new environment, add a [game file](games) containing its configuration and `Game` adapter. See [Inverted Pendulum](games/mujoco_IP.py) for an example and [Add an environment](#add-an-environment) for the interface.

## Trained examples

Discrete examples below are from [main](https://github.com/werner-duvaud/muzero-general/tree/main); their game adapters are not included in this branch. The continuous checkpoints run on this branch.

| [CartPole](https://github.com/werner-duvaud/muzero-general/blob/main/games/cartpole.py) | [Lunar Lander](https://github.com/werner-duvaud/muzero-general/blob/main/games/lunarlander.py) | [Breakout](https://github.com/werner-duvaud/muzero-general/blob/main/games/breakout.py) |
| :---: | :---: | :---: |
| <img src="checkpoints/cartpole/episode.gif" height="200" alt="CartPole policy balancing for 500 steps"> | <img src="checkpoints/lunarlander/episode.gif" height="200" alt="Lunar Lander policy completing a successful landing with a score of 304.46"> | <img src="checkpoints/breakout/episode.gif" height="200" alt="Breakout policy scoring 428 points"> |
| [Learning curve](checkpoints/cartpole/training.png) · [Checkpoint](checkpoints/cartpole/model.checkpoint) | [Learning curve](checkpoints/lunarlander/training.png) · [Checkpoint](checkpoints/lunarlander/model.checkpoint) | [Learning curve](checkpoints/breakout/training.png) · [Checkpoint](checkpoints/breakout/model.checkpoint) |
| **[Tic-tac-toe](https://github.com/werner-duvaud/muzero-general/blob/main/games/tictactoe.py)** | **[Tic-tac-toe (OpenSpiel)](https://github.com/werner-duvaud/muzero-general/blob/main/games/openspiel.py)** | **[MiniGrid Empty-Random-5x5](https://github.com/werner-duvaud/muzero-general/blob/main/games/gridworld.py)** |
| <img src="checkpoints/tictactoe/episode.gif" height="200" alt="Tic-tac-toe policy drawing against an optimal opponent, first as X then as O"> | <img src="checkpoints/openspiel/episode.gif" height="200" alt="OpenSpiel tic-tac-toe policy drawing against an optimal opponent, first as X then as O"> | <img src="checkpoints/gridworld/episode.gif" height="200" alt="MiniGrid policy reaching the goal in 7 steps"> |
| [Learning curve](checkpoints/tictactoe/training.png) · [Checkpoint](checkpoints/tictactoe/model.checkpoint) | [Learning curve](checkpoints/openspiel/training.png) · [Checkpoint](checkpoints/openspiel/model.checkpoint) | [Learning curve](checkpoints/gridworld/training.png) · [Checkpoint](checkpoints/gridworld/model.checkpoint) |
| **[Inverted Pendulum (continuous branch)](https://github.com/werner-duvaud/muzero-general/tree/continuous)** | **[HalfCheetah (continuous branch)](https://github.com/werner-duvaud/muzero-general/tree/continuous)** | |
| <img src="checkpoints/mujoco_IP/episode.gif" height="200" alt="Continuous Inverted Pendulum policy balancing for 1,000 steps"> | <img src="checkpoints/mujoco_halfcheetah/episode.gif" height="200" alt="HalfCheetah policy scoring 951.7"> | |
| [Learning curve](checkpoints/mujoco_IP/training.png) · [Checkpoint](checkpoints/mujoco_IP/model.checkpoint) | [Learning curve](checkpoints/mujoco_halfcheetah/training.png) · [Checkpoint](checkpoints/mujoco_halfcheetah/model.checkpoint) | |

Both tic-tac-toe checkpoints are certified non-losing from the starting board in either seat; each drew all 200 MCTS evaluation games against an optimal opponent.

## Quick start

### Install

Use Python 3.12 or later. The included checkpoints were trained on Linux with Python 3.12.

```bash
git clone --branch continuous https://github.com/werner-duvaud/muzero-general.git
cd muzero-general
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

See the official [PyTorch installation guide](https://docs.pytorch.org/get-started/locally/) for GPU setup. MCTS tree visualization also needs the system [Graphviz executable](https://graphviz.readthedocs.io/en/stable/manual.html#installation), in addition to its Python package.

### Train

Choose an environment:

```bash
python muzero.py mujoco_IP
python muzero.py mujoco_IDP
python muzero.py mujoco_halfcheetah
```

Each command starts a fresh model using that game's default config, runs the configured training budget, then evaluates the selected checkpoint. Game defaults use one GPU when CUDA is available, otherwise CPU. Use `--cpu` to force CPU execution.

From Python:

```python
import muzero

agent = muzero.MuZero("mujoco_IP")
agent.train()
```

### Evaluate a checkpoint

```bash
python muzero.py mujoco_IP --evaluate
python muzero.py mujoco_halfcheetah --evaluate
```

These commands load `checkpoints/<env>/model.checkpoint` and evaluate it on CPU, without training or rendering.

### Monitor and render

Run TensorBoard in another terminal:

```bash
tensorboard --logdir ./results
```

For interactive rendering, human play, model diagnostics or hyperparameter search, open the menu:

```bash
python muzero.py
```

Run `python muzero.py --help` for all command-line options.

## Configuration and saved runs

Each [game file](games) contains a complete, commented `MuZeroConfig`. All games share the same parameter names; values are tuned per environment. Edit the Python config to change the default recipe. [config.py](config.py) provides shared validation and temperature-schedule helpers.

`--seed` changes the training seed; evaluation seeds are configured separately.

For your machine, tune `num_workers` and `num_threads` for CPU capacity; reduce `batch_size`, `selfplay_batch_size`, and `reanalyse_batch_size` to lower GPU memory use, or `replay_buffer_size` to lower RAM use.

By default, runs are saved under `results/<env>/<timestamp>/`:

| File | Contents |
| --- | --- |
| `model.checkpoint` | Latest saved training state, including weights and optimizer state |
| `model-best.checkpoint` | Best validation checkpoint, when validation is enabled |
| `replay_buffer.pkl.gz` | Collected games for resuming training |
| `config.json`, `run.json` | Effective configuration and runtime/source metadata |
| `validation.json`, `evaluation.json` | Validation history and final evaluation results |

TensorBoard event files are saved in the same directory. With `checkpoint_selection="best"`, the final evaluation uses the best validation checkpoint, which may precede the last gradient update.

Replay snapshots can be many gigabytes for image-based environments.

To evaluate your own run or resume interrupted training, replace `RUN` with its directory name:

```bash
python muzero.py mujoco_IP --evaluate --checkpoint results/mujoco_IP/RUN/model-best.checkpoint

python muzero.py mujoco_IP --checkpoint results/mujoco_IP/RUN/model.checkpoint --replay-buffer results/mujoco_IP/RUN/replay_buffer.pkl.gz
```

Resume with both the checkpoint and replay buffer to retain collected experience. To train beyond the saved budget, increase `training_steps` through `--config`.

## Other environments

The following discrete adapters remain available on [main](https://github.com/werner-duvaud/muzero-general/tree/main):

- [MiniGrid](https://github.com/werner-duvaud/muzero-general/blob/main/games/gridworld.py) (`gridworld`) and [a simple grid world](https://github.com/werner-duvaud/muzero-general/blob/main/games/simple_grid.py) (`simple_grid`).
- [Tic-tac-toe](https://github.com/werner-duvaud/muzero-general/blob/main/games/tictactoe.py), [Connect4](https://github.com/werner-duvaud/muzero-general/blob/main/games/connect4.py) and [Gomoku](https://github.com/werner-duvaud/muzero-general/blob/main/games/gomoku.py).
- [Twenty-One](https://github.com/werner-duvaud/muzero-general/blob/main/games/twentyone.py) (`twentyone`), a simplified card game.
- [OpenSpiel](https://github.com/werner-duvaud/muzero-general/blob/main/games/openspiel.py) (`openspiel`), configured for tic-tac-toe by default. The adapter supports deterministic single-player or alternating two-player zero-sum games with 1D–3D observation tensors. Chance-based games such as backgammon and simultaneous-action games are unsupported.
- [An Atari template](https://github.com/werner-duvaud/muzero-general/blob/main/games/atari.py) (`atari`), configured for Breakout by default. Use `breakout` for the tuned recipe and packaged checkpoint shown above.

## Add an environment

1. Copy [games/mujoco_IP.py](games/mujoco_IP.py) to `games/my_game.py`, keeping the common config fields.
2. Set the observation shape, number of action dimensions (`action_space`), `players = [0]`, network and training parameters. Observations must have shape `(channels, height, width)`; reshape a vector of length `n` to `(1, 1, n)`.
3. Implement `Game` using [AbstractGame](games/abstract_game.py): initialization, `reset()`, `step(action)`, `legal_actions()` and `render()`. `step()` returns `(observation, reward, done)`. Return `None` from `legal_actions()` and implement `close()` to release resources. For Gymnasium Box environments, inherit [ContinuousGame](games/continuous_game.py) and set `env_id`.
4. Run `python muzero.py my_game`.

Actions are finite pre-tanh vectors; the game adapter maps them to environment bounds. Wrap Gymnasium's `terminated` and `truncated` flags into `done`. Trajectory indexing and training-target alignment are documented in [game_history.py](game_history.py).

## Continuous version

This branch adapts main's batched search and training to continuous control. For the continuous-action extension of MuZero, see [Continuous Control for Searching and Planning with a Learned Model](https://arxiv.org/abs/2006.07430) by Xuxi Yang, Werner Duvaud and Peng Wei. The orchestration, value/reward targets, validation, evaluation and checkpoint selection follow main; policy heads and replay targets use continuous actions.

`action_space` is the number of action dimensions. The policy predicts Gaussian means and bounded log standard deviations. Search and replay store pre-tanh vectors; dynamics and stacked observations use `tanh(action)`. The game adapter scales those normalized controls to the environment's Box bounds once.

Each node samples `num_sampled_actions` candidates. Progressive widening grows the set with visits according to `pw_alpha`; zero keeps a fixed candidate count. `node_prior="uniform"` is the default for candidates sampled from the policy. The optional `"density"` setting uses normalized Gaussian densities in pre-tanh coordinates, rather than an importance-corrected Sampled MuZero prior.

Replay stores candidate vectors with normalized visit weights. Training minimizes visit-weighted negative log likelihood, with optional squashed-policy entropy controlled by `entropy_loss_weight`. Padding has zero probability mass; reanalysis replaces candidates and weights together. Both fully connected and residual networks support vector actions.

Evaluation reports original environment rewards. Training divides rewards by `reward_scale`. Discrete checkpoints and historical continuous checkpoints/replay buffers from before this action format are incompatible.

## Code guide

| File | Role |
| --- | --- |
| [muzero.py](muzero.py) | CLI, worker orchestration, training and checkpoint loading |
| [self_play.py](self_play.py) | MCTS, action selection and self-play |
| [models.py](models.py) | Fully connected and residual networks |
| [trainer.py](trainer.py) | Losses, recurrent unrolling and optimizer updates |
| [replay_buffer.py](replay_buffer.py) | Replay sampling, eviction and reanalysis |
| [game_history.py](game_history.py) | Trajectories, observation stacks and training targets |
| [evaluation.py](evaluation.py) | Validation, checkpoint selection and evaluation reports |
| [random_bootstrap.py](random_bootstrap.py) | Random-episode collection and replay initialization |
| [diagnose_model.py](diagnose_model.py) | Compare learned dynamics with real environment transitions |

Format code with [Ruff](https://docs.astral.sh/ruff/): `ruff format --target-version py312 .`

## Further reading

- [Continuous Control for Searching and Planning with a Learned Model](https://arxiv.org/abs/2006.07430) — Xuxi Yang, Werner Duvaud and Peng Wei.
- [MuZero paper](https://arxiv.org/abs/1911.08265) and [original pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py).
- [Project wiki](https://github.com/werner-duvaud/muzero-general/wiki/MuZero-Documentation) and [explanatory video](https://youtu.be/We20YSAJZSE).
- [Network diagram](docs/muzero-network-werner-duvaud.png).
- [EfficientZero](https://arxiv.org/abs/2111.00210) and [Sampled MuZero](https://arxiv.org/abs/2104.06303).

## Authors and citation

Werner Duvaud, Aurèle Hainaut, Paul Lenoir, and [contributors](https://github.com/werner-duvaud/muzero-general/graphs/contributors). Released under the [MIT license](LICENSE).

```bibtex
@misc{muzero-general,
  author       = {Werner Duvaud, Aurèle Hainaut},
  title        = {MuZero General: Open Reimplementation of MuZero},
  year         = {2019},
  publisher    = {GitHub},
  url          = {https://github.com/werner-duvaud/muzero-general},
}
```
