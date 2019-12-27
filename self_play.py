import math

import numpy
import ray
import torch


@ray.remote  # (num_gpus=1) #  Uncoomment num_gpus and model.to(config.device) to self-play on GPU
def run_selfplay(Game, config, shared_storage, replay_buffer, model, seed):
    """
    Function which run simultaneously in multiple threads and is continuously playing games and saving them to the replay-buffer.
    """
    # model.to(config.device) # Uncoomment this line and num_gpus from the decorator to self-play on GPU
    with torch.no_grad():
        while True:
            # Initialize a self-play
            model.load_state_dict(ray.get(shared_storage.get_weights.remote()))
            game = Game(seed)
            done = False
            game_history = GameHistory(config.discount)

            # Self-play with actions based on the Monte Carlo tree search at each moves
            observation = game.reset()
            game_history.observation_history.append(observation)
            while not done and len(game_history.history) < config.max_moves:
                root = MCTS(config).run(model, observation, True)

                temperature = config.visit_softmax_temperature_fn(
                    num_moves=len(game_history.history),
                    trained_steps=ray.get(shared_storage.get_training_step.remote()),
                )
                action = select_action(root, temperature)

                observation, reward, done = game.step(action)

                game_history.observation_history.append(observation)
                game_history.rewards.append(reward)
                game_history.history.append(action)
                game_history.store_search_statistics(root, config.action_space)

            game.close()
            # Save the game history
            replay_buffer.save_game.remote(game_history)


# Game independant
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(self, model, observation, add_exploration_noise):
        """
        At the root of the search tree we use the representation function to obtain a hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model learned by the network.
        """
        root = Node(0)
        observation = (
            torch.from_numpy(observation).to(self.config.device).float().unsqueeze(0)
        )
        _, expected_reward, policy_logits, hidden_state = model.initial_inference(
            observation
        )
        root.expand(
            self.config.action_space, expected_reward, policy_logits, hidden_state
        )
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats(
            self.config.min_known_bound, self.config.max_known_bound
        )

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                last_action = action
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[last_action]]).to(parent.hidden_state.device),
            )
            node.expand(self.config.action_space, reward, policy_logits, hidden_state)

            self.backpropagate(search_path, value.item(), min_max_stats)

        return root

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
        """
        for node in search_path:
            # Always the same player, the other players minds should be modeled in network because environment do not act always in the best way to make you lose
            node.value_sum += value  # if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.discount * value


def select_action(node, temperature, random=False):
    """
    Select action according to the vivist count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function in the config.
    """
    visit_counts = numpy.array(
        [[child.visit_count, action] for action, child in node.children.items()]
    ).T
    if temperature == 0:
        action_pos = numpy.argmax(visit_counts[0])
    else:
        # See paper Data Generation appendix
        visit_count_distribution = visit_counts[0] ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action_pos = numpy.random.choice(
            len(visit_counts[1]), p=visit_count_distribution
        )

    if random:
        action_pos = numpy.random.choice(len(visit_counts[1]))

    return visit_counts[1][action_pos]


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the neural network.
        """
        self.reward = reward
        self.hidden_state = hidden_state
        policy = {a: math.exp(policy_logits[0][a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self, discount):
        self.observation_history = []
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def store_search_statistics(self, root, action_space):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())


@ray.remote
class ReplayBuffer:
    # Store list of game history and generate batch
    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game_history):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)

    def sample_batch(self, num_unroll_steps, td_steps):
        observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.batch_size):
            game_history = self.sample_game()
            game_pos = self.sample_position(game_history)
            actions = game_history.history[game_pos : game_pos + num_unroll_steps]
            # Repeat precedent action to make "actions" of length "num_unroll_steps"
            actions.extend(
                [actions[-1] for _ in range(num_unroll_steps - len(actions) + 1)]
            )
            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value, reward, policy = self.make_target(
                game_history, game_pos, num_unroll_steps, td_steps
            )
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)

        return observation_batch, action_batch, reward_batch, value_batch, policy_batch

    def sample_game(self):
        """
        Sample game from buffer either uniformly or according to some priority.
        """
        # TODO: sample with probability link to the highest difference between real and predicted value (see paper appendix Training)
        return self.buffer[numpy.random.choice(range(len(self.buffer)))]

    def sample_position(self, game):
        """
        Sample position from game either uniformly or according to some priority.
        """
        # TODO: according to some priority
        return numpy.random.choice(range(len(game.rewards)))

    def make_target(self, game, state_index, num_unroll_steps, td_steps):
        """
        The value target is the discounted root value of the search tree td_steps into the future, plus the discounted sum of all rewards until then.
        """
        target_values, target_rewards, target_policies = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(game.root_values):
                value = game.root_values[bootstrap_index] * game.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                value += reward * game.discount ** i

            if current_index < len(game.root_values):
                target_values.append(value)
                target_rewards.append(game.rewards[current_index])
                target_policies.append(game.child_visits[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy to give the tensor a valid dimension
                target_policies.append(
                    [
                        1 / len(game.child_visits[0])
                        for _ in range(len(game.child_visits[0]))
                    ]
                )

        return target_values, target_rewards, target_policies

    def length(self):
        return len(self.buffer)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self, min_value_bound, max_value_bound):
        self.maximum = min_value_bound if min_value_bound else -float("inf")
        self.minimum = max_value_bound if max_value_bound else float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
