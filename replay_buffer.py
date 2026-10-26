import copy
import time

import numpy
import ray
import torch

import models
from game_history import GameHistory
from self_play import MCTS

PRIORITY_EPSILON = 1e-6


def normalize_probabilities(probabilities):
    """Normalize weights, using uniform probabilities for invalid or zero totals."""
    total = numpy.sum(probabilities)
    if total > 0 and numpy.isfinite(total):
        return probabilities / total
    return numpy.full(len(probabilities), 1 / len(probabilities))


@ray.remote
class ReplayBuffer:
    """Ray actor storing FIFO replay games and sampling MuZero training batches."""

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        for game_history in self.buffer.values():
            self.initialize_priorities(game_history)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed.
        numpy.random.seed(self.config.seed)

    def initialize_priorities(self, game_history):
        """Prepare writable priorities for new games or replay resumed with PER."""
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk.
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value
                            - game_history.compute_target_value(
                                i,
                                self.config.td_steps,
                                self.config.discount,
                                self.config.use_last_model_value,
                            )
                        )
                        + PRIORITY_EPSILON
                    ) ** self.config.PER_alpha
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
            game_history.game_priority = numpy.max(game_history.priorities)

    def save_game(self, game_history, shared_storage=None):
        """Store a nonempty trajectory, initialize priorities, and evict oldest games.

        Update collection counters and optionally publish them to shared storage.
        """
        if not game_history.root_values:
            return
        self.initialize_priorities(game_history)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = next(iter(self.buffer))
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        """Return the current mapping from game IDs to trajectories."""
        return self.buffer

    def get_state(self):
        """Return replay data and total collected game/transition counters."""
        return {
            "buffer": self.buffer,
            "num_played_games": self.num_played_games,
            "num_played_steps": self.num_played_steps,
        }

    def get_batch(self, use_reanalysis=True):
        """Sample a batch of stacked observations, unroll targets, and loss masks.

        Args:
            use_reanalysis (bool): Allow refreshed values and policies when configured.
                False uses original targets, as required during initial warmup.

        Returns:
            tuple: Sampled [game ID, position] pairs and a batch tuple containing
            observations, actions, values, rewards, policies, importance weights,
            value masks, reward masks, and policy masks. Importance weights are None
            when prioritized replay is disabled. Actions have K unroll steps; targets
            and masks have K + 1 prediction positions.
        """
        stacked_channels = (
            self.config.observation_shape[0] * (self.config.stacked_observations + 1)
            + self.config.stacked_observations
        )
        observation_batch = numpy.empty(
            (
                self.config.batch_size,
                stacked_channels,
                *self.config.observation_shape[1:],
            ),
            dtype="float32",
        )
        (
            index_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            value_mask_batch,
            reward_mask_batch,
            policy_mask_batch,
        ) = ([], [], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for batch_index, (game_id, game_history, game_prob) in enumerate(
            self.sample_n_games(self.config.batch_size)
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            use_reanalysed_policy = (
                use_reanalysis
                and self.config.reanalyse_policy
                and game_history.reanalysed_child_visits is not None
                and numpy.random.random() < self.config.reanalyse_policy_ratio
            )
            (
                values,
                rewards,
                policies,
                actions,
                value_masks,
                reward_masks,
                policy_masks,
            ) = game_history.make_target(
                game_pos,
                self.config.num_unroll_steps,
                self.config.td_steps,
                self.config.discount,
                self.config.action_space,
                use_reanalysed_value=use_reanalysis
                and self.config.use_last_model_value,
                use_reanalysed_policy=use_reanalysed_policy,
                train_reward=self.config.train_reward,
            )

            index_batch.append([game_id, game_pos])
            game_history.get_stacked_observations(
                game_pos,
                self.config.stacked_observations,
                len(self.config.action_space),
                getattr(self.config, "observation_scale", 1),
                observation_batch[batch_index],
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            value_mask_batch.append(value_masks)
            reward_mask_batch.append(reward_masks)
            policy_mask_batch.append(policy_masks)
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # *_mask_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                numpy.asarray(action_batch, dtype="int64"),
                numpy.asarray(value_batch, dtype="float32"),
                numpy.asarray(reward_batch, dtype="float32"),
                numpy.asarray(policy_batch, dtype="float32"),
                weight_batch,
                numpy.asarray(value_mask_batch, dtype="float32"),
                numpy.asarray(reward_mask_batch, dtype="float32"),
                numpy.asarray(policy_mask_batch, dtype="float32"),
            ),
        )

    def sample_game(self, force_uniform=False):
        """Sample a game uniformly or according to replay priorities.

        See paper appendix Training.

        Return (game ID, history, probability). The probability is None for uniform
        sampling. force_uniform overrides prioritized replay.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs = normalize_probabilities(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = list(self.buffer)[game_index]

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        """Sample n_games with replacement and return (ID, history, probability) tuples.

        Probabilities are None when sampling uniformly.
        """
        game_ids = list(self.buffer)
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [self.buffer[game_id].game_priority for game_id in game_ids],
                dtype="float32",
            )
            game_probs = normalize_probabilities(game_probs)
            game_prob_dict = dict(zip(game_ids, game_probs))
            selected_games = numpy.random.choice(game_ids, n_games, p=game_probs)
        else:
            selected_games = numpy.random.choice(game_ids, n_games)
            game_prob_dict = {}
        return [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]

    def sample_position(self, game_history, force_uniform=False):
        """Sample a trajectory position uniformly or according to its priorities.

        See paper appendix Training.

        Return (position, probability), with probability None for uniform sampling.
        force_uniform overrides prioritized replay.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = normalize_probabilities(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def sample_reanalyse_batch(self):
        """Sample a uniform game and distinct positions for target refresh.

        Return a tuple of game ID, positions, observation/action history windows,
        legal actions, and players, or None when the game has no searchable positions.
        """
        game_id, game_history, _ = self.sample_game(force_uniform=True)
        game_length = len(game_history.root_values)
        if not game_length:
            return None
        positions = numpy.random.choice(
            game_length,
            min(self.config.reanalyse_batch_size, game_length),
            replace=False,
        )
        windows = []
        for position in positions:
            start = max(0, position - self.config.stacked_observations)
            windows.append(
                (
                    game_history.observation_history[start : position + 1],
                    game_history.action_history[start:position],
                )
            )
        return (
            game_id,
            positions,
            windows,
            [game_history.legal_actions_history[i] for i in positions],
            [game_history.to_play_history[i] for i in positions],
        )

    def update_reanalysed_targets(self, game_id, positions, values, policies):
        """Store refreshed values and policies at the requested positions.

        Ignore updates for games that have already been evicted.
        """
        # The game could have been removed since it was sampled.
        if game_id not in self.buffer:
            return

        game_history = self.buffer[game_id]
        if values is not None:
            if game_history.reanalysed_predicted_root_values is None:
                game_history.reanalysed_predicted_root_values = numpy.array(
                    game_history.root_values, dtype="float32"
                )
            game_history.reanalysed_predicted_root_values[positions] = values
        if policies is not None:
            if game_history.reanalysed_child_visits is None:
                game_history.reanalysed_child_visits = copy.deepcopy(
                    game_history.child_visits
                )
            for position, policy in zip(positions, policies):
                game_history.reanalysed_child_visits[position] = policy

    def update_priorities(self, priorities, index_info):
        """Update position and game priorities from training errors.

        Ignore updates for evicted games and truncate unroll priorities at episode
        end. Prioritized sampling follows Distributed Prioritized Experience Replay:
        https://arxiv.org/abs/1803.00933.
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training.
            if game_id in self.buffer:
                # Update position priorities.
                priority = numpy.maximum(priorities[i, :], PRIORITY_EPSILON)
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities.
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )


@ray.remote
class Reanalyse:
    """Ray actor that updates replay targets with the latest network.

    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed.
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.set_num_threads(self.config.num_threads)

        # Initialize the network.
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()
        self.model_version = initial_checkpoint["weights_version"]

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        """Refresh replay targets until training finishes or stop is requested."""
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            status = ray.get(
                shared_storage.get_info.remote(["training_step", "terminate"])
            )
            if (
                status["training_step"] >= self.config.training_steps
                or status["terminate"]
            ):
                return
            time.sleep(0.1)

        while True:
            status = ray.get(
                shared_storage.get_info.remote(
                    ["training_step", "terminate", "weights_version"]
                )
            )
            if (
                status["training_step"] >= self.config.training_steps
                or status["terminate"]
            ):
                break
            if self.model_version != status["weights_version"]:
                self.model.set_weights(
                    ray.get(shared_storage.get_info.remote("weights"))
                )
                self.model_version = status["weights_version"]

            batch = ray.get(replay_buffer.sample_reanalyse_batch.remote())
            if batch is None:
                continue
            game_id, positions, windows, legal_actions, players = batch
            observations = []
            for observation_history, action_history in windows:
                history = GameHistory()
                history.observation_history = observation_history
                history.action_history = action_history
                observations.append(
                    history.get_stacked_observations(
                        -1,
                        self.config.stacked_observations,
                        len(self.config.action_space),
                        getattr(self.config, "observation_scale", 1),
                    )
                )
            observations = numpy.asarray(observations)

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze).
            if self.config.use_last_model_value:
                observation_batch = torch.as_tensor(
                    observations.copy(),
                    dtype=torch.float32,
                    device=next(self.model.parameters()).device,
                )
                with torch.inference_mode():
                    values = models.support_to_scalar(
                        self.model.initial_inference(observation_batch)[0],
                        self.config.support_size,
                    )
                values = values.reshape(-1).cpu().numpy()
            else:
                values = None

            # Re-run MCTS with the latest network to refresh policy targets. Legal
            # action masks are part of the trajectory because they cannot be
            # reconstructed from visit counts.
            if self.config.reanalyse_policy:
                policies = []
                with torch.inference_mode():
                    for observation, actions, to_play in zip(
                        observations, legal_actions, players
                    ):
                        root, _ = MCTS(self.config).run(
                            self.model,
                            observation,
                            actions,
                            to_play,
                            False,
                        )
                        policies.append(
                            GameHistory.policy_from_root(root, self.config.action_space)
                        )
            else:
                policies = None

            replay_buffer.update_reanalysed_targets.remote(
                game_id, positions, values, policies
            )
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
