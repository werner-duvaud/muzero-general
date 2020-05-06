import collections
import copy

import numpy
import ray
import torch

import models


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config):
        self.config = config
        self.buffer = {}
        self.game_priorities = collections.deque(maxlen=self.config.window_size)
        self.max_recorded_game_priority = 1.0
        self.self_play_count = 0
        self.total_samples = 0

        # Used only for the Reanalyze options
        self.model = None
        if self.config.use_last_model_value:
            self.model = models.MuZeroNetwork(self.config)
            self.model.to(torch.device("cpu"))
            self.model.eval()

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def save_game(self, game_history):
        if game_history.priorities is not None:
            # Avoid read only array when loading replay buffer from pickle
            game_history.priorities = game_history.priorities.copy()
        else:
            if self.config.use_max_priority:
                game_history.priorities = numpy.full(
                    len(game_history.root_values), self.max_recorded_game_priority
                )
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                for i, root_value in enumerate(game_history.root_values):
                    priorities = []
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype=numpy.float32)

        self.buffer[self.self_play_count] = game_history
        self.total_samples += len(game_history.priorities)
        self.game_priorities.append(numpy.max(game_history.priorities))

        self.self_play_count += 1

        if self.config.window_size < len(self.buffer):
            del_id = self.self_play_count - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].priorities)
            del self.buffer[del_id]

    def get_self_play_count(self):
        return self.self_play_count

    def get_buffer(self):
        return self.buffer

    def get_batch(self, model_weights):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            weight_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [])

        if self.config.use_last_model_value:
            self.model.set_weights(model_weights)

        for _ in range(self.config.batch_size):
            game_id, game_history, game_prob = self.sample_game(self.buffer)
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos, self.config.stacked_observations
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            weight_batch.append(
                (self.total_samples * game_prob * pos_prob) ** (-self.config.PER_beta)
            )
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

        weight_batch = numpy.array(weight_batch, dtype=numpy.float32) / max(
            weight_batch
        )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, buffer):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_probs = numpy.array(self.game_priorities, dtype=numpy.float32)
        game_probs /= numpy.sum(game_probs)
        game_index = numpy.random.choice(len(self.buffer), p=game_probs)
        game_prob = game_probs[game_index]
        game_id = self.self_play_count - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_probs = game_history.priorities / sum(game_history.priorities)
        position_index = numpy.random.choice(len(position_probs), p=position_probs)
        position_prob = position_probs[position_index]

        return position_index, position_prob

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        min_priorities = numpy.min(priorities)
        if not min_priorities or numpy.isnan(min_priorities) or min_priorities < 1e-5:
            print(
                "Warning : Extreme values ({}) in game priorities. Could be underfitting or overfitting.".format(
                    min_priorities
                )
            )
        else:
            for i in range(len(index_info)):
                game_id, game_pos = index_info[i]

                # The element could be removed since its selection and training
                if game_id in self.buffer:
                    # Update position priorities
                    priority = priorities[i, :]
                    start_index = game_pos
                    end_index = min(
                        game_pos + len(priority), len(self.buffer[game_id].priorities)
                    )
                    self.buffer[game_id].priorities[start_index:end_index] = priority[
                        : end_index - start_index
                    ]

                    # Update game priorities
                    game_index = game_id - (self.self_play_count - len(self.buffer))
                    self.game_priorities[game_index] = numpy.max(
                        self.buffer[game_id].priorities
                    )  # option: mean, sum, max

                    self.max_recorded_game_priority = numpy.max(self.game_priorities)

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            if self.config.use_last_model_value:
                # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
                observation = (
                    torch.tensor(
                        game_history.get_stacked_observations(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )
                    .float()
                    .unsqueeze(0)
                )
                last_step_value = models.support_to_scalar(
                    self.model.initial_inference(observation)[0],
                    self.config.support_size,
                ).item()
            else:
                last_step_value = game_history.root_values[bootstrap_index]

            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + 1 + i]
                else -reward
            ) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(game_history.action_history))

        return target_values, target_rewards, target_policies, actions
