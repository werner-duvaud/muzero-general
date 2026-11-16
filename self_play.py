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
        self.seed = seed
        self.extra_games = []
        self.active_games = []

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
                for game_history in self.play_games(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=status["training_step"]
                    ),
                    self.config.temperature_threshold,
                ):
                    # Bound pending replay saves to one per worker.
                    ray.get(
                        replay_buffer.save_game.remote(game_history, shared_storage)
                    )

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
                    self.config.action_space,
                    self.config.observation_scale,
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
                action_description = (
                    self.game.action_to_string(action) if render else None
                )
                observation, reward, done = self.game.step(action.value)

                if render:
                    print(f"Played action: {action_description}")
                    self.game.render()

                game_history.action_history.append(action.value)
                game_history.reward_history.append(reward / self.config.reward_scale)
                game_history.observation_history.append(observation)
                game_history.to_play_history.append(self.game.to_play())

        game_history.observation_history = numpy.asarray(
            game_history.observation_history
        )
        return game_history

    def play_games(self, temperature, temperature_threshold):
        """Yield completed self-play games, batching search across active games.

        Each tree still completes one simulation before selecting its next leaf.
        Finished games enter replay immediately and their slots restart. Return
        after a batch of completions so the caller can refresh weights, retaining
        unfinished games for the next call.
        """
        batch_size = self.config.selfplay_batch_size
        if batch_size == 1:
            yield self.play_game(
                temperature,
                temperature_threshold,
                False,
                "self",
                self.config.muzero_player,
            )
            return
        while len(self.extra_games) < batch_size - 1:
            # Reserve one seed per worker and the monitoring worker in each block.
            seed = self.seed + (len(self.extra_games) + 1) * (
                self.config.num_workers + 1
            )
            self.extra_games.append(self.Game(seed))
        if not self.active_games:
            for game in [self.game, *self.extra_games]:
                history = GameHistory()
                history.observation_history.append(game.reset())
                history.to_play_history.append(game.to_play())
                self.active_games.append((game, history))
        active = self.active_games
        search = MCTS(self.config)
        completed = 0
        while completed < batch_size:
            for _, history in active:
                assert (
                    numpy.shape(history.observation_history[-1])
                    == self.config.observation_shape
                ), "Observation must match the configured observation_shape."
            observations = [
                history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                    self.config.action_space,
                    self.config.observation_scale,
                )
                for _, history in active
            ]
            legal_actions = [game.legal_actions() for game, _ in active]
            roots, _ = search.run_batch(
                self.model,
                observations,
                legal_actions,
                [game.to_play() for game, _ in active],
                True,
            )
            for index, ((game, history), root, actions) in enumerate(
                zip(active, roots, legal_actions)
            ):
                move = len(history.action_history)
                action = self.select_action(
                    root,
                    temperature
                    if temperature_threshold is None or move < temperature_threshold
                    else 0,
                )
                history.store_search_statistics(root, self.config.action_space, actions)
                observation, reward, done = game.step(action.value)
                history.action_history.append(action.value)
                history.reward_history.append(reward / self.config.reward_scale)
                history.observation_history.append(observation)
                history.to_play_history.append(game.to_play())
                if done or len(history.action_history) >= self.config.max_moves:
                    history.observation_history = numpy.asarray(
                        history.observation_history
                    )
                    next_history = GameHistory()
                    next_history.observation_history.append(game.reset())
                    next_history.to_play_history.append(game.to_play())
                    active[index] = (game, next_history)
                    completed += 1
                    yield history

    def close_game(self):
        """Close all environments owned by this worker."""
        for game in [self.game, *self.extra_games]:
            game.close()

    def evaluate_episode(self, seed):
        """Return seed, raw reward, and length from greedy MCTS without root noise."""
        player = (
            self.config.players[seed % len(self.config.players)]
            if self.config.evaluation_both_players
            else self.config.muzero_player
        )
        history = self.play_game(
            0,
            0,
            False,
            self.config.opponent,
            player,
            False,
            seed,
        )
        reward = (
            sum(history.reward_history)
            if len(self.config.players) == 1
            else player_reward(history, player)
        )
        episode = {
            "seed": seed,
            "reward": float(reward * self.config.reward_scale),
            "length": len(history.action_history),
        }
        if len(self.config.players) > 1:
            episode["player"] = player
        return episode

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
            return Action(self.game.human_to_action()), root
        elif opponent == "expert":
            return Action(self.game.expert_agent()), None
        elif opponent == "random":
            normalized = numpy.random.uniform(
                -1 + 1e-6, 1 - 1e-6, self.config.action_space
            )
            return Action(numpy.arctanh(normalized)), None
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

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """Search one root; see run_batch for the shared search implementation."""
        roots, info = self.run_batch(
            model,
            [observation],
            [legal_actions],
            [to_play],
            add_exploration_noise,
            None if override_root_with is None else [override_root_with],
        )
        return roots[0], info[0]

    @torch.inference_mode()
    def run_batch(
        self,
        model,
        observations,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_roots=None,
    ):
        """Search independent roots with one batched network call per simulation.

        Observations, legal-action lists and players have one entry per root.
        Each root has separate value bounds and receives num_simulations visits;
        selection and backup within a tree remain sequential. The model must be
        in evaluation mode. Optional expanded roots reuse their latent states.

        Return roots and per-root max_tree_depth/root_predicted_value dictionaries.
        """
        batch_size = len(to_play)
        if (
            not batch_size
            or len(observations) != batch_size
            or len(legal_actions) != batch_size
        ):
            raise ValueError("Search requires equally sized, nonempty input batches")
        if override_roots is None:
            roots = [Node(0) for _ in to_play]
            observations = numpy.asarray(observations)
            if not observations.flags.writeable:
                observations = observations.copy()
            observations = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            root_values = self.expand_nodes(
                roots, model.initial_inference(observations), legal_actions, to_play
            )
        else:
            if len(override_roots) != batch_size:
                raise ValueError("Expected one override root per input")
            roots = override_roots
            root_values = [None] * batch_size
        if add_exploration_noise:
            for root in roots:
                root.add_exploration_noise(
                    self.config.root_dirichlet_alpha,
                    self.config.root_exploration_fraction,
                )
        bounds = [MinMaxStats(self.config.known_bounds) for _ in roots]
        depths = [0] * batch_size
        for _ in range(self.config.num_simulations):
            paths, actions, players = [], [], []
            for i, root in enumerate(roots):
                node, player = root, to_play[i]
                path = [node]
                while node.expanded():
                    action, node = self.select_child(node, bounds[i])
                    path.append(node)
                    player = next_player(player, self.config.players)
                paths.append(path)
                actions.append(action)
                players.append(player)
                depths[i] = max(depths[i], len(path) - 1)
            states = torch.cat([path[-2].hidden_state for path in paths])
            outputs = model.recurrent_inference(
                states,
                torch.as_tensor(
                    numpy.asarray([action.value for action in actions]),
                    dtype=torch.float32,
                    device=states.device,
                ),
            )
            values = self.expand_nodes(
                [path[-1] for path in paths],
                outputs,
                [self.config.action_space] * batch_size,
                players,
            )
            for path, value, player, stats in zip(paths, values, players, bounds):
                self.backpropagate(path, value, player, stats)
        return roots, [
            {"max_tree_depth": depth, "root_predicted_value": value}
            for depth, value in zip(depths, root_values)
        ]

    def expand_nodes(self, nodes, outputs, legal_actions, players):
        """Transfer predictions to CPU once per batch, retaining latents on device."""
        values, rewards, policies, states = outputs
        scalars = (
            models.support_to_scalar(
                torch.cat((values, rewards)), self.config.support_size
            )
            .flatten()
            .tolist()
        )
        policies = policies.cpu()
        for i, node in enumerate(nodes):
            node.expand(
                legal_actions[i],
                players[i],
                scalars[len(nodes) + i],
                policies[i : i + 1],
                states[i : i + 1],
                self.config,
            )
        return scalars[: len(nodes)]

    def select_child(self, node, min_max_stats):
        """Return the action and child with highest UCB, breaking ties randomly."""
        # Add candidates as visits grow, with a finite retry budget for collapsed policies.
        target = max(
            self.config.num_sampled_actions,
            math.ceil((node.visit_count + 1) ** self.config.pw_alpha),
        )
        node.sample_children(target)
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

    def expand(self, actions, to_play, reward, policy_parameters, hidden_state, config):
        """Store the latent state and sample continuous candidates from its policy."""
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        self.config = config
        self.distribution = models.policy_distribution(
            policy_parameters[0].detach().cpu()
        )
        self.exploration_noise = None
        self.sample_children(config.num_sampled_actions)

    def sample_children(self, target):
        """Draw distinct Gaussian candidates, then normalize their search priors."""
        missing = target - len(self.children)
        if missing <= 0:
            return
        for value in self.distribution.sample((max(8, missing * 2),)).numpy():
            action = Action(value)
            if action not in self.children:
                child = Node(0)
                child.log_prior = (
                    float(self.distribution.log_prob(torch.from_numpy(value)).sum())
                    if self.config.node_prior == "density"
                    else 0.0
                )
                self.children[action] = child
            if len(self.children) >= target:
                break
        if not self.children:
            raise RuntimeError("Continuous policy produced no candidates")
        self.normalize_priors()

    def normalize_priors(self):
        """Uniform sampled-policy priors or a normalized latent-density heuristic."""
        children = list(self.children.values())
        if self.config.node_prior == "uniform":
            priors = numpy.full(len(children), 1 / len(children))
        else:
            log_priors = numpy.asarray([child.log_prior for child in children])
            priors = numpy.exp(log_priors - log_priors.max())
            priors /= priors.sum()
        if self.exploration_noise is not None:
            alpha, fraction = self.exploration_noise
            priors = (1 - fraction) * priors + fraction * numpy.random.dirichlet(
                [alpha] * len(children)
            )
        for child, prior in zip(children, priors):
            child.prior = float(prior)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """Mix Dirichlet noise into existing child priors for root exploration."""
        self.exploration_noise = (dirichlet_alpha, exploration_fraction)
        self.normalize_priors()


class Action:
    """Immutable float32 vector with value-based hashing for search-tree edges."""

    def __init__(self, value):
        value = numpy.asarray(value, dtype="float32").reshape(-1).copy()
        if not numpy.isfinite(value).all():
            raise ValueError("Actions must be finite")
        value[value == 0] = 0  # Canonicalize negative zero for hashing.
        value.flags.writeable = False
        self.value = value
        self._key = value.tobytes()

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, Action) and self._key == other._key

    def __repr__(self):
        return numpy.array2string(self.value, precision=3)


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
