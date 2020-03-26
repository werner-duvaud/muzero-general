import numpy
import ray


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config):
        self.config = config
        self.buffer = []
        self.self_play_count = 0

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)
        self.self_play_count += 1

    def get_self_play_count(self):
        return self.self_play_count

    def get_batch(self):
        (
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = [], [], [], [], [], []
        for _ in range(self.config.batch_size):
            game_history = self.sample_game(self.buffer)
            game_pos = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            observation_batch,
            action_batch,
            value_batch,
            reward_batch,
            policy_batch,
            gradient_scale_batch,
        )

    def sample_game(self, buffer):
        """
        Sample game from buffer either uniformly or according to some priority.
        """
        # TODO: sample with probability link to the highest difference between real and
        # predicted value (See paper appendix Training)
        return numpy.random.choice(self.buffer)

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        """
        # TODO: sample according to some priority
        return numpy.random.choice(range(len(game_history.reward_history)))

    def make_target(self, game_history, state_index):
        """
        The value target is the discounted root value of the search tree td_steps into the
        future, plus the discounted sum of all rewards until then.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            bootstrap_index = current_index + self.config.td_steps
            if bootstrap_index < len(game_history.root_values):
                value = (
                    game_history.root_values[bootstrap_index]
                    * self.config.discount ** self.config.td_steps
                )
            else:
                value = 0

            for i, reward in enumerate(
                game_history.reward_history[current_index:bootstrap_index]
            ):
                value += (
                    reward
                    if game_history.to_play_history[current_index]
                    == game_history.to_play_history[current_index + i]
                    else -reward
                ) * self.config.discount ** i

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
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
                # Uniform policy to give the tensor a valid dimension
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(game_history.action_history))

        return target_values, target_rewards, target_policies, actions
