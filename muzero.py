import argparse
import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import random_bootstrap
import replay_buffer
import self_play
import shared_storage
import trainer
from evaluation import (
    Validation,
    file_sha256,
    runtime_metadata,
    summarize_evaluation,
    write_json,
)


class MuZero:
    """Main class to manage MuZero.

    New instances start with random weights. Call load_model to restore weights
    or replay data before training or evaluation.

    Args:
        game_name (str): Module name under games, without the .py suffix.
        config (dict or MuZeroConfig, optional): Overrides for the game's defaults,
            or a complete replacement configuration object.
        split_resources_in (int): Number of concurrent experiments sharing GPUs.

    Examples:
        >>> agent = MuZero("cartpole")
        >>> report = agent.train()
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name.
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.game_name = game_name
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        self._config_override_keys = set(
            config if isinstance(config, dict) else vars(config) if config else ()
        )

        # Overwrite the config.
        if config:
            if type(config) is dict:
                self.config.update(config)
            else:
                self.config = config
        self.config.validate()

        # Fix random generator seed.
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.set_num_threads(self.config.num_threads)

        # Manage GPUs.
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(
            num_gpus=total_gpus,
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {
                    name: str(self.config.num_threads)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                }
            },
        )

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "weights_version": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "bootstrap_training_steps": 0,
            "training_step_offset": 0,
            "played_steps_offset": 0,
            "terminate": False,
            "game_name": self.game_name,
            "config": self.config.to_dict(),
            "training_origin": {"type": "fresh_self_play"},
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """Collect initial replay, warm up, and launch MuZero training workers.

        Args:
            log_in_tensorboard (bool): If True, block while monitoring training,
                writing TensorBoard metrics, and managing validation and shutdown.
                If False, return after launching workers; the caller must monitor
                training and call terminate_workers. Initial collection and warmup
                finish before returning in either mode.

        Returns:
            dict or None: Final evaluation report after a completed, monitored run
            with evaluation enabled; otherwise None.
        """
        self.config.validate()
        self.checkpoint.pop("training_steps_completed", None)
        self.checkpoint["game_name"] = self.game_name
        self.checkpoint["config"] = self.config.to_dict()
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)
            source_checkpoint = getattr(self, "source_checkpoint", None)
            if source_checkpoint:
                self.checkpoint["training_origin"] = {
                    "type": "resumed_self_play",
                    "checkpoint": str(source_checkpoint),
                    "checkpoint_sha256": file_sha256(source_checkpoint),
                    "parent": self.checkpoint.get(
                        "training_origin", {"type": "unknown"}
                    ),
                }
                self.source_checkpoint = None
            write_json(self.config.results_path / "config.json", self.config.to_dict())
            write_json(
                self.config.results_path / "run.json",
                {
                    "game": self.game_name,
                    "runtime": runtime_metadata(),
                    "training_origin": self.checkpoint["training_origin"],
                },
            )

        self.initialize_replay_buffer()

        reanalyse_enabled = (
            self.config.use_last_model_value or self.config.reanalyse_policy
        )

        # Manage GPUs.
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + reanalyse_enabled * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers.
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        warmup_target = min(
            self.checkpoint.get("bootstrap_training_steps", 0),
            self.config.training_steps,
        )
        if self.replay_buffer and self.checkpoint["training_step"] < warmup_target:
            try:
                ray.get(
                    self.training_worker.warmup.remote(
                        self.replay_buffer_worker,
                        self.shared_storage_worker,
                        warmup_target,
                    )
                )
            except BaseException:
                self.terminate_workers()
                raise
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())
            if self.config.save_model:
                self.save_replay_buffer()

        if reanalyse_enabled:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Game,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers.
        worker_tasks = [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        worker_tasks.append(
            self.training_worker.continuous_update_weights.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )
        )
        if reanalyse_enabled:
            worker_tasks.append(
                self.reanalyse_worker.reanalyse.remote(
                    self.replay_buffer_worker, self.shared_storage_worker
                )
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
                worker_tasks,
            )
            if (
                self.checkpoint.get(
                    "training_steps_completed", self.checkpoint["training_step"]
                )
                >= self.config.training_steps
                and self.config.evaluation_episodes
            ):
                return self.evaluate()

    def initialize_replay_buffer(self):
        """Prepare initial replay and warmup counters at training step zero.

        Collect top random episodes only when replay is empty. Existing replay can
        also be warmed up at step zero. Resumed checkpoints past step zero skip
        this initialization.
        """
        warmup_steps = getattr(self.config, "random_bootstrap_training_steps", 0)
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(
                "random_bootstrap_training_steps must be a nonnegative integer"
            )
        if self.checkpoint["training_step"] != 0:
            return
        if not self.replay_buffer:
            self.replay_buffer = random_bootstrap.bootstrap_replay_buffer(
                self.Game, self.config
            )
            if self.replay_buffer:
                self.checkpoint["num_played_games"] = len(self.replay_buffer)
                self.checkpoint["num_played_steps"] = sum(
                    len(history.action_history)
                    for history in self.replay_buffer.values()
                )
        if self.replay_buffer:
            warmup_target = min(warmup_steps, self.config.training_steps)
            self.checkpoint.update(
                {
                    "bootstrap_training_steps": warmup_target,
                    "training_step_offset": warmup_target,
                    "played_steps_offset": self.checkpoint["num_played_steps"],
                }
            )
            if self.config.save_model:
                self.save_replay_buffer()

    def save_replay_buffer(self):
        """Atomically save replay data and counters as replay_buffer.pkl."""
        path = self.config.results_path / "replay_buffer.pkl"
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with open(temporary_path, "wb") as replay_buffer_file:
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                replay_buffer_file,
            )
        temporary_path.replace(path)

    def logging_loop(self, num_gpus, worker_tasks=None):
        """Monitor workers, log metrics, and validate frozen weight snapshots.

        Stop training and save replay on completion or interruption. When configured,
        restore the snapshot with the best validation mean after training finishes.
        """
        # Launch the test worker to get performance metrics.
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        test_task = self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )
        worker_tasks = list(worker_tasks or []) + [test_task]

        # Write everything in TensorBoard.
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard.
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation.
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        validation = (
            Validation(self.Game, self.config)
            if self.config.validation_interval
            else None
        )
        next_validation = self.checkpoint["training_step"]
        try:
            while info["training_step"] < self.config.training_steps:
                if worker_tasks:
                    ready, _ = ray.wait(worker_tasks, num_returns=1, timeout=0)
                    for task in ready:
                        ray.get(
                            task
                        )  # Surface worker failures instead of waiting indefinitely.
                        worker_tasks.remove(task)
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                if validation:
                    report = validation.poll()
                    if report:
                        writer.add_scalar(
                            "4.Validation/Mean_environment_reward",
                            report["mean_reward"],
                            report["training_step"],
                        )
                    if (
                        not validation.tasks
                        and info["training_step"] >= next_validation
                    ):
                        snapshot = ray.get(
                            self.shared_storage_worker.get_checkpoint.remote()
                        )
                        if (
                            not validation.history
                            or validation.history[-1]["training_step"]
                            != snapshot["weights_version"]
                        ):
                            validation.start(snapshot)
                        next_validation = (
                            info["training_step"] + self.config.validation_interval
                        )
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    info["total_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    info["mean_value"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    info["episode_length"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward",
                    info["muzero_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward",
                    info["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    info["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    max(
                        0,
                        info["training_step"]
                        - self.checkpoint.get("training_step_offset", 0),
                    )
                    / max(
                        1,
                        info["num_played_steps"]
                        - self.checkpoint.get("played_steps_offset", 0),
                    ),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f"Last test reward: {info['total_reward']:.2f}. Training step: {info['training_step']}/{self.config.training_steps}. Played games: {info['num_played_games']}. Loss: {info['total_loss']:.2f}",
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
            if validation:
                validation.poll(wait=True)
                snapshot = ray.get(self.shared_storage_worker.get_checkpoint.remote())
                if (
                    not validation.history
                    or validation.history[-1]["training_step"]
                    != snapshot["weights_version"]
                ):
                    validation.start(snapshot)
                    validation.poll(wait=True)
        except KeyboardInterrupt:
            pass
        finally:
            if validation:
                validation.cancel()
            self.terminate_workers()
            writer.close()
            if self.config.save_model:
                self.save_replay_buffer()
        completed = self.checkpoint["training_step"]
        replay_counts = {
            key: self.checkpoint[key]
            for key in ("num_played_games", "num_played_steps", "num_reanalysed_games")
        }
        if (
            validation
            and self.config.checkpoint_selection == "best"
            and validation.best_checkpoint is not None
        ):
            self.checkpoint = dict(validation.best_checkpoint)
            # Replay continues to contain the final collected data when an
            # earlier set of weights is selected. Preserve its insertion IDs.
            self.checkpoint.update(replay_counts)
            self.source_checkpoint = validation.best_path
            print(
                f"Selected checkpoint at step {self.checkpoint['training_step']} from {completed} updates."
            )
        self.checkpoint["training_steps_completed"] = completed
        if (
            validation
            and self.config.checkpoint_selection == "best"
            and getattr(self, "source_checkpoint", None)
        ):
            temporary = self.source_checkpoint.with_suffix(".checkpoint.tmp")
            torch.save(self.checkpoint, temporary)
            temporary.replace(self.source_checkpoint)

    def terminate_workers(self):
        """Stop workers and retrieve their training state.

        Save the latest checkpoint when enabled. This does not shut down Ray.
        """
        if self.shared_storage_worker:
            ray.get(self.shared_storage_worker.set_info.remote("terminate", True))
            if self.training_worker:
                try:
                    # Queued behind training: flush its last completed update.
                    ray.get(
                        self.training_worker.publish_checkpoint.remote(
                            self.shared_storage_worker
                        )
                    )
                except ray.exceptions.RayActorError:
                    # A crashed trainer leaves the last published snapshot usable.
                    pass
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            state = ray.get(self.replay_buffer_worker.get_state.remote())
            self.replay_buffer = state.pop("buffer")
            self.checkpoint.update(state)
            if self.shared_storage_worker:
                ray.get(self.shared_storage_worker.set_info.remote(state))
        if self.shared_storage_worker and self.config.save_model:
            ray.get(self.shared_storage_worker.save_checkpoint.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """Test the model in a Ray actor.

        Args:
            render (bool): Display the game using its rendering interface.
            opponent (str, optional): "self", "human", "expert", or "random".
                None uses the configured opponent.
            muzero_player (int, optional): Player controlled by MuZero. None uses
                the configured player.
            num_tests (int): Number of episodes to average.
            num_gpus (float): GPU resources for the actor; zero uses the CPU.

        Returns:
            float: Mean return in environment reward units, from muzero_player's
            perspective for multiplayer games. No root exploration noise is used.
        """
        opponent = opponent if opponent is not None else self.config.opponent
        muzero_player = (
            muzero_player if muzero_player is not None else self.config.muzero_player
        )
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            numpy.random.randint(10000),
            render,
        )
        results = []
        for i in range(num_tests):
            print(f"Testing {i + 1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                        False,
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [self_play.player_reward(history, muzero_player) for history in results]
            )
        return result * self.config.reward_scale

    def evaluate(self, output_path=None):
        """Evaluate frozen weights with greedy MCTS on configured consecutive seeds.

        Inference runs on CPU without root exploration noise. Results use environment
        reward units and are written as JSON with checkpoint and runtime metadata.

        Args:
            output_path (str or Path, optional): Report destination. Defaults to
                results_path/evaluation.json.

        Returns:
            dict: Episode results, aggregate statistics, and target comparison.

        Raises:
            ValueError: evaluation_episodes is zero.
        """
        if not self.config.evaluation_episodes:
            raise ValueError("evaluation_episodes must be positive to evaluate")
        config = copy.deepcopy(self.config)
        config.selfplay_on_gpu = False
        runtime = runtime_metadata()
        pool_size = min(config.num_workers, config.evaluation_episodes)
        checkpoint = ray.put(self.checkpoint)
        workers = []
        episodes = []
        seeds = iter(
            range(
                config.evaluation_seed,
                config.evaluation_seed + config.evaluation_episodes,
            )
        )
        pending = {}
        try:
            for _ in range(pool_size):
                seed = next(seeds)
                worker = self_play.SelfPlay.options(num_cpus=0, num_gpus=0).remote(
                    checkpoint,
                    self.Game,
                    config,
                    seed,
                )
                workers.append(worker)
                pending[worker.evaluate_episode.remote(seed)] = worker
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                for task in ready:
                    worker = pending.pop(task)
                    episode = ray.get(task)
                    episodes.append(episode)
                    print(
                        f"Evaluation {len(episodes)}/{config.evaluation_episodes}: "
                        f"seed={episode['seed']}, reward={episode['reward']:.2f}",
                        flush=True,
                    )
                    seed = next(seeds, None)
                    if seed is not None:
                        pending[worker.evaluate_episode.remote(seed)] = worker
            ray.get([worker.close_game.remote() for worker in workers])
        finally:
            for worker in workers:
                ray.kill(worker)
        episodes.sort(key=lambda episode: episode["seed"])
        report = summarize_evaluation(episodes, config.evaluation_target)
        checkpoint_path = getattr(self, "source_checkpoint", None)
        if checkpoint_path is None and config.save_model:
            checkpoint_path = config.results_path / "model.checkpoint"
        report.update(
            {
                "game": self.game_name,
                "training_step": self.checkpoint["training_step"],
                "training_steps_completed": self.checkpoint.get(
                    "training_steps_completed",
                    self.checkpoint["training_step"],
                ),
                "training_origin": self.checkpoint.get(
                    "training_origin", {"type": "unknown"}
                ),
                "policy": "MCTS",
                "temperature": 0,
                "root_exploration_noise": False,
                "reward_units": "environment",
                "config": config.to_dict(),
                "runtime": runtime,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_sha256": (
                    file_sha256(checkpoint_path)
                    if checkpoint_path and pathlib.Path(checkpoint_path).is_file()
                    else None
                ),
            }
        )
        output_path = output_path or config.results_path / "evaluation.json"
        write_json(output_path, report)
        status = (
            "no target configured"
            if report["target_met"] is None
            else ("target met" if report["target_met"] else "target NOT met")
        )
        print(f"Mean environment reward: {report['mean_reward']:.2f}; {status}.")
        print(f"Evaluation saved to {output_path}")
        return report

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """Load a model and/or a saved replay buffer.

        This method keeps the current config, except for compatible restoration of
        residual state normalization when not explicitly overridden. The CLI restores
        embedded config before constructing MuZero. Without a replay file, replay
        and collection counters are cleared while a loaded training step is kept.

        Args:
            checkpoint_path (str or Path, optional): Saved model.checkpoint file.
            replay_buffer_path (str or Path, optional): Saved replay_buffer.pkl file.

        Raises:
            ValueError: The checkpoint belongs to a different game.
        """
        # Load checkpoint.
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if self.checkpoint.get("game_name", self.game_name) != self.game_name:
                raise ValueError(
                    "Checkpoint game does not match the selected environment"
                )
            if (
                self.config.network == "resnet"
                and "resnet_state_normalization" not in self._config_override_keys
            ):
                self.config.resnet_state_normalization = self.checkpoint.get(
                    "config", {}
                ).get("resnet_state_normalization", "channel")
            self.source_checkpoint = checkpoint_path.resolve()
            self.checkpoint.setdefault(
                "weights_version", self.checkpoint["training_step"]
            )
            self.checkpoint.setdefault("training_origin", {"type": "unknown"})
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer.
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print("Using empty buffer.")
            self.replay_buffer = {}
            if not checkpoint_path:
                self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0
            self.checkpoint["bootstrap_training_steps"] = 0
            self.checkpoint["training_step_offset"] = self.checkpoint["training_step"]
            self.checkpoint["played_steps_offset"] = 0

    def diagnose_model(self, horizon):
        """Compare an imagined trajectory with the same actions in the environment.

        Display diagnostic plots and wait for user input before closing them.

        Args:
            horizon (int): Maximum number of actions to examine.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    """Create initial CPU weights without reserving GPU resources in Ray."""

    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU.

    def __init__(self):
        pass

    def get_initial_weights(self, config):
        """Return seeded network weights on CPU and a printable model summary."""
        numpy.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.num_threads)
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """Search for hyperparameters by launching parallel experiments.

    Evaluate completed experiments with test and save the best observed weights
    alongside the recommended parameters. Those parameters may differ from the
    configuration that produced the saved weights.

    Args:
        game_name (str): Game module name under games.
        parametrization (nevergrad.p.Parameter): Search space for config overrides.
        budget (int): Total number of training experiments.
        parallel_experiments (int): Maximum number of concurrent experiments.
        num_tests (int): Episodes used to evaluate each experiment.

    Returns:
        dict: Recommended config overrides.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments.
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train(False)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(False)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights).
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters.
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


def load_model_menu(muzero, game_name):
    """Prompt for checkpoint and replay paths, then load them into the agent."""
    # Configure running options.
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / game_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        replay_buffer_path = options[choice] / "replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


def main(argv=None):
    """Run the command-line interface for training or checkpoint evaluation.

    Args:
        argv (list[str], optional): Arguments excluding the program name.
            None reads arguments from sys.argv.

    Returns:
        int: Zero on success, two when an evaluation target is missed, or 130
        when training is interrupted.
    """
    games = sorted(
        path.stem
        for path in (pathlib.Path(__file__).parent / "games").glob("*.py")
        if "class MuZeroConfig(" in path.read_text()
    )
    parser = argparse.ArgumentParser(
        description="Train an environment using its complete default config, then evaluate it.",
        epilog="Examples: python muzero.py cartpole | python muzero.py lunarlander | "
        "python muzero.py breakout | python muzero.py cartpole --evaluate",
    )
    parser.add_argument("env", choices=games)
    parser.add_argument("overrides", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        help="Config overrides as a JSON object or path to a JSON file",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a checkpoint without training",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="Load weights; evaluation defaults to checkpoints/ENV/model.checkpoint",
    )
    parser.add_argument(
        "--replay-buffer",
        type=pathlib.Path,
        help="Replay buffer to resume training",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Directory for checkpoints, config, logs and evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Training seed (evaluation seeds are configured separately)",
    )
    parser.add_argument("--episodes", type=int, help="Number of evaluation episodes")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for all workers")
    args = parser.parse_args(argv)
    if args.config and args.overrides:
        parser.error("Supply config overrides once, using --config")
    if args.evaluate and args.replay_buffer:
        parser.error("--replay-buffer is only used for training")
    if args.evaluate and args.checkpoint is None:
        args.checkpoint = (
            pathlib.Path(__file__).parent
            / "checkpoints"
            / args.env
            / "model.checkpoint"
        )
    for path in (args.checkpoint, args.replay_buffer):
        if path is not None and not path.is_file():
            parser.error(f"File does not exist: {path}")
    config = {}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            config.update(checkpoint["config"])
        elif (args.checkpoint.parent / "config.json").is_file():
            config.update(
                json.loads((args.checkpoint.parent / "config.json").read_text())
            )
        # Loading a checkpoint should write to a new run, never its old results path.
        config.pop("results_path", None)
    overrides = args.config or args.overrides
    if overrides:
        try:
            config.update(
                json.loads(overrides)
                if overrides.lstrip().startswith("{")
                else json.loads(pathlib.Path(overrides).read_text())
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"Invalid config: {error}")
    if args.output is not None:
        config["results_path"] = args.output
    if args.seed is not None:
        config["seed"] = args.seed
    if args.episodes is not None:
        config["evaluation_episodes"] = args.episodes
    if args.cpu or args.evaluate:
        config.update(
            max_num_gpus=0,
            train_on_gpu=False,
            selfplay_on_gpu=False,
            reanalyse_on_gpu=False,
        )
    try:
        agent = MuZero(args.env, config)
        if args.checkpoint or args.replay_buffer:
            agent.load_model(args.checkpoint, args.replay_buffer)
        print(f"Environment: {args.env}; output: {agent.config.results_path}")
        report = agent.evaluate() if args.evaluate else agent.train()
        if (
            not args.evaluate
            and agent.checkpoint.get(
                "training_steps_completed", agent.checkpoint["training_step"]
            )
            < agent.config.training_steps
        ):
            return 130
        return 2 if report and report["target_met"] is False else 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())

    print("\nWelcome to MuZero! Here's a list of games:")
    # Let user pick a game.
    games = [
        filename.stem
        for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
        if filename.name != "abstract_game.py"
    ]
    for i in range(len(games)):
        print(f"{i}. {games[i]}")
    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i) for i in range(len(games))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    # Initialize MuZero.
    choice = int(choice)
    game_name = games[choice]
    muzero = MuZero(game_name)

    while True:
        # Configure running options.
        options = [
            "Train",
            "Load pretrained model",
            "Diagnose model",
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Hyperparameter search",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        if choice == 0:
            muzero.train()
        elif choice == 1:
            load_model_menu(muzero, game_name)
        elif choice == 2:
            muzero.diagnose_model(30)
        elif choice == 3:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 4:
            muzero.test(render=True, opponent="human", muzero_player=0)
        elif choice == 5:
            env = muzero.Game(render=True)
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                env.render()
        elif choice == 6:
            # Define here the parameters to tune.
            # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
            muzero.terminate_workers()
            del muzero
            budget = 20
            parallel_experiments = 2
            lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
            discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
            parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
            best_hyperparameters = hyperparameter_search(
                game_name, parametrization, budget, parallel_experiments, 20
            )
            muzero = MuZero(game_name, best_hyperparameters)
        else:
            break
        print("\nDone")

    ray.shutdown()
