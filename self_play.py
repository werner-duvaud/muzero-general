import math
import time

import numpy
import ray
import torch

import models
from game_history import GameHistory


def next_player(player, players):
    """Return the next player in the configured cyclic player order."""
    return players[(players.index(player) + 1) % len(players)]


def player_reward(game_history, player):
    """Sum trajectory rewards from the specified player's perspective."""
    return sum(
        reward if game_history.to_play_history[i] == player else -reward
        for i, reward in enumerate(game_history.reward_history)
    )


@ray.remote
class SelfPlay:
    """Ray actor collecting trajectories or evaluating a checkpoint with MCTS."""

    def __init__(self, initial_checkpoint, Game, config, seed, render=False):
        self.config = config
        self.Game = Game

        # Fix random generator seed.
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(config.num_threads)
        self.game = Game(seed, render)

        # Initialize the network.
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.model_version = initial_checkpoint["weights_version"]
        self.training_step_offset = initial_checkpoint.get("training_step_offset", 0)
        self.played_steps_offset = initial_checkpoint.get("played_steps_offset", 0)

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        """Reload published weights and play until training ends or stop is requested.

        Store training games in replay, or publish return metrics in test_mode.
        """
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

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=status["training_step"]
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    self.config.muzero_player,
                )

                # Wait for storage before collecting another game. This bounds
                # pending replay saves to one per worker when replay is slower.
                ray.get(replay_buffer.save_game.remote(game_history, shared_storage))

            else:
                # Take the best action (no exploration) in test mode.
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                    False,
                )

                # Save to the shared storage.
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history),
                        "total_reward": sum(game_history.reward_history)
                        * self.config.reward_scale,
                        "mean_value": numpy.mean(
                            [v for v in game_history.root_values if v is not None]
                            or [0]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": player_reward(
                                game_history, self.config.muzero_player
                            )
                            * self.config.reward_scale,
                            "opponent_reward": player_reward(
                                game_history,
                                next_player(
                                    self.config.muzero_player, self.config.players
                                ),
                            )
                            * self.config.reward_scale,
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    max(
                        0,
                        ray.get(shared_storage.get_info.remote("training_step"))
                        - self.training_step_offset,
                    )
                    / max(
                        1,
                        ray.get(shared_storage.get_info.remote("num_played_steps"))
                        - self.played_steps_offset,
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self,
        temperature,
        temperature_threshold,
        render,
        opponent,
        muzero_player,
        add_exploration_noise=True,
        seed=None,
    ):
        """Play one episode and return its training-scaled GameHistory.

        MuZero selects actions with MCTS; other players use the requested opponent.
        Play stops at episode end or max_moves. Passing seed recreates the game and
        reseeds NumPy and PyTorch before the episode.

        Args:
            temperature (float): Visit-count sampling temperature; zero is greedy.
            temperature_threshold (int, optional): Move index after which play is
                greedy. None keeps the supplied temperature throughout.
            render (bool): Render the game after reset and each move.
            opponent (str): "self", "human", "expert", or "random".
            muzero_player (int): Player controlled by MuZero unless opponent is "self".
            add_exploration_noise (bool): Mix Dirichlet noise into root priors.
            seed (int, optional): Seed for an independently initialized episode.

        Returns:
            GameHistory: Observations, actions, scaled rewards, and search targets.
        """
        if seed is not None:
            self.game.close()
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            self.game = self.Game(seed, render)
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.observation_history.append(observation)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.inference_mode():
            while not done and len(game_history.action_history) < self.config.max_moves:
                assert len(numpy.array(observation).shape) == 3, (
                    f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                )
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), (
                    f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                )
                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                    getattr(self.config, "observation_scale", 1),
                )

                # Choose the action.
                if opponent == "self" or muzero_player == self.game.to_play():
                    legal_actions = self.game.legal_actions()
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        legal_actions,
                        self.game.to_play(),
                        add_exploration_noise,
                    )
                    action = self.select_action(
                        root,
                        (
                            temperature
                            if temperature_threshold is None
                            or len(game_history.action_history) < temperature_threshold
                            else 0
                        ),
                    )

                    if render:
                        print(f"Tree depth: {mcts_info['max_tree_depth']}")
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    legal_actions = self.game.legal_actions()
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                game_history.store_search_statistics(
                    root, self.config.action_space, legal_actions
                )
                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.action_history.append(action)
                game_history.reward_history.append(reward / self.config.reward_scale)
                game_history.observation_history.append(observation)
                game_history.to_play_history.append(self.game.to_play())

        game_history.observation_history = numpy.asarray(
            game_history.observation_history
        )
        return game_history

    def close_game(self):
        """Close the underlying game environment."""
        self.game.close()

    def evaluate_episode(self, seed):
        """Return seed, raw reward, and length from greedy MCTS without root noise."""
        history = self.play_game(
            0,
            0,
            False,
            self.config.opponent,
            self.config.muzero_player,
            False,
            seed,
        )
        reward = (
            sum(history.reward_history)
            if len(self.config.players) == 1
            else player_reward(history, self.config.muzero_player)
        )
        return {
            "seed": seed,
            "reward": float(reward * self.config.reward_scale),
            "length": len(history.action_history),
        }

    def select_opponent_action(self, opponent, stacked_observations):
        """Return an opponent action and a diagnostic search root if available.

        Human play includes a MuZero suggestion; expert and random play skip search.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f"Tree depth: {mcts_info['max_tree_depth']}")
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert self.game.legal_actions(), (
                f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            )
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """Select an action from search visits using the supplied temperature.

        Zero chooses the most visited action. Infinite temperature samples uniformly;
        positive finite temperatures sample in proportion to visits ** (1 /
        temperature).
        If every visit count is zero, positive temperatures sample uniformly. The
        caller obtains scheduled temperatures from visit_softmax_temperature_fn.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="float64"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation.
            positive = visit_counts > 0
            if not positive.any():
                return numpy.random.choice(actions)
            logits = numpy.full_like(visit_counts, -numpy.inf)
            log_visits = numpy.log(visit_counts[positive])
            # Center before dividing so tiny temperatures cannot produce +inf.
            with numpy.errstate(over="ignore"):
                logits[positive] = (log_visits - log_visits.max()) / temperature
            visit_count_distribution = numpy.exp(logits)
            visit_count_distribution /= visit_count_distribution.sum()
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """Search learned dynamics by repeatedly selecting, expanding, and backing up.

    Each simulation starts at the root and follows UCB scores to an unexpanded
    node. The learned network supplies latent transitions, rewards, policies,
    and values; real environment steps are not used inside search.
    """

    def __init__(self, config):
        self.config = config

    @torch.inference_mode()
    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """Run simulations from an observation or an expanded override root.

        Args:
            model (AbstractNetwork): Network used for initial and recurrent inference.
            observation (array-like): Stacked and scaled network observation.
            legal_actions (list[int]): Actions allowed at the real root. Imagined
                states use the full configured action space.
            to_play (int): Player at the root.
            add_exploration_noise (bool): Mix Dirichlet noise into root priors.
            override_root_with (Node, optional): Expanded root whose latent state is
                reused.

        Returns:
            tuple: Search root and a dict containing max_tree_depth and
            root_predicted_value. The predicted value is None for an override root.
        """
        if override_root_with is not None:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = numpy.asarray(observation)
            if not observation.flags.writeable:
                observation = observation.copy()
            observation = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=next(model.parameters()).device,
            ).unsqueeze(0)
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert legal_actions, (
                f"Legal actions should not be an empty array. Got {legal_actions}."
            )
            assert set(legal_actions).issubset(set(self.config.action_space)), (
                "Legal actions should be a subset of the action space."
            )
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn.
                virtual_to_play = next_player(virtual_to_play, self.config.players)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state.
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                parent.hidden_state.new_tensor([[action]], dtype=torch.long),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """Return the action and child with highest UCB, breaking ties randomly."""
        # Use priors even before the first simulation, without adding a virtual
        # visit that would change the root value or policy target counts.
        exploration_rate = (
            math.log(
                (node.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        ) * math.sqrt(max(1, node.visit_count))
        scores = {
            action: self.ucb_score(node, child, min_max_stats, exploration_rate)
            for action, child in node.children.items()
        }
        max_ucb = max(scores.values())
        action = numpy.random.choice(
            [action for action, score in scores.items() if score == max_ucb]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats, exploration_rate=None):
        """Combine a prior exploration bonus with normalized reward and child value."""
        if exploration_rate is None:
            exploration_rate = (
                math.log(
                    (parent.visit_count + self.config.pb_c_base + 1)
                    / self.config.pb_c_base
                )
                + self.config.pb_c_init
            ) * math.sqrt(max(1, parent.visit_count))
        prior_score = exploration_rate * child.prior / (child.visit_count + 1)

        if child.visit_count > 0:
            child_value = (
                child.value() if len(self.config.players) == 1 else -child.value()
            )
            value_score = min_max_stats.normalize(
                child.reward + self.config.discount * child_value
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """Back up discounted values, player signs, visits, and search value bounds."""
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward - self.config.discount * node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    """Store a search node's prior, latent state, reward, visits, and value sum."""

    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        """Return whether this node has children."""
        return len(self.children) > 0

    def value(self):
        """Return the mean backed-up value, or zero before any visits."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """Store player, reward, and latent state, then create children for actions.

        Softmax policy logits over the supplied actions to obtain child priors.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = policy_logits[0, actions].softmax(dim=0).detach().cpu().tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """Mix Dirichlet noise into existing child priors for root exploration."""
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MinMaxStats:
    """Track search value bounds, optionally initialized from configured bounds."""

    def __init__(self, known_bounds=None):
        self.minimum = known_bounds[0] if known_bounds is not None else float("inf")
        self.maximum = known_bounds[1] if known_bounds is not None else -float("inf")

    def update(self, value):
        """Expand the tracked range to include value."""
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        """Scale value if bounds have a positive range; otherwise leave it unchanged."""
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
