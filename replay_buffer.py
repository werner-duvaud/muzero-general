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
        self.buffer = []
        self.game_priorities = []
        self.max_recorded_game_priority = 1.0
        self.self_play_count = 0

        self.model = models.MuZeroNetwork(self.config)

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
            self.game_priorities.pop(0)

        if self.config.use_max_priority:
            game_history.priorities = (
                numpy.ones(len(game_history.root_values))
                * self.max_recorded_game_priority
            )
        self.buffer.append(game_history)
        self.game_priorities.append(numpy.mean(game_history.priorities))
        self.self_play_count += 1

    def get_self_play_count(self):
        return self.self_play_count

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

        total_samples = sum(
            (len(game_history.priorities) for game_history in self.buffer)
        )

        if self.config.use_last_model_value:
            self.model.set_weights(model_weights)

        for _ in range(self.config.batch_size):
            game_index, game_history, game_prob = self.sample_game(self.buffer)
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_index, game_pos])
            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            weight_batch.append(
                (total_samples * game_prob * pos_prob) ** (-self.config.PER_beta)
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

        weight_batch = numpy.array(weight_batch) / max(weight_batch)

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
        game_probs = numpy.array(self.game_priorities) / sum(self.game_priorities)
        game_index_candidates = numpy.arange(0, len(self.buffer), dtype=int)
        game_index = numpy.random.choice(game_index_candidates, p=game_probs)
        game_prob = game_probs[game_index]

        return game_index, self.buffer[game_index], game_prob

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_probs = numpy.array(game_history.priorities) / sum(
            game_history.priorities
        )
        position_index_candidates = numpy.arange(0, len(position_probs), dtype=int)
        position_index = numpy.random.choice(
            position_index_candidates, p=position_probs
        )
        position_prob = position_probs[position_index]

        return position_index, position_prob

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_index, game_pos = index_info[i]

            # update position priorities
            priority = priorities[i, :]
            start_index = game_pos
            end_index = min(
                game_pos + len(priority), len(self.buffer[game_index].priorities)
            )
            self.buffer[game_index].priorities[start_index:end_index] = priority[
                : end_index - start_index
            ]

            # update game priorities
            self.game_priorities[game_index] = numpy.max(
                self.buffer[game_index].priorities
            )  # option: mean, sum, max

            self.max_recorded_game_priority = numpy.max(self.game_priorities)

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            # The value target is the discounted root value of the search tree td_steps into the
            # future, plus the discounted sum of all rewards until then.
            bootstrap_index = current_index + self.config.td_steps
            if bootstrap_index < len(game_history.root_values):
                if self.config.use_last_model_value:
                    # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
                    observation = torch.tensor(
                        game_history.observation_history[bootstrap_index]
                    ).float()
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
                game_history.reward_history[current_index + 1 : bootstrap_index + 1]
            ):
                value += (
                    reward
                    if game_history.to_play_history[current_index]
                    == game_history.to_play_history[current_index + 1 + i]
                    else -reward
                ) * self.config.discount ** i

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
