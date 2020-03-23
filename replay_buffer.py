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
        self.game_priorities = []
        self.max_recorded_game_priority = 1.0
        self.self_play_count = 0

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
            self.game_priorities.pop(0)
        game_history.priorities = numpy.ones(len(game_history.observation_history)) * self.max_recorded_game_priority
        self.buffer.append(game_history)
        self.game_priorities.append(numpy.mean(game_history.priorities))

        self.self_play_count += 1

    def get_self_play_count(self):
        return self.self_play_count

    def get_batch(self):
        index_batch, observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
            [],
            [],
            [],
            [],
            [],
            []
        )
        for _ in range(self.config.batch_size):
            game_index, game_history = self.sample_game(self.buffer)
            game_pos = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_index, game_pos])
            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        return index_batch, (observation_batch, action_batch, value_batch, reward_batch, policy_batch)

    def sample_game(self, buffer):
        """
        Sample game from buffer either uniformly or according to some priority.
        """
        # TODO: sample with probability link to the highest difference between real and
        # predicted value (See paper appendix Training)
        game_probs = numpy.array(self.game_priorities) / sum(self.game_priorities)
        game_index_candidates = numpy.arange(0, len(self.buffer), dtype=int)
        game_index = numpy.random.choice(game_index_candidates, p=game_probs)

        return game_index, self.buffer[game_index]

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        """
        # TODO: sample according to some priority
        position_probs = numpy.array(game_history.priorities) / sum(game_history.priorities)
        position_index_candidates = numpy.arange(0, len(position_probs), dtype=int)
        position_index = numpy.random.choice(position_index_candidates, p=position_probs)

        return position_index

    def update_priorities(self, priorities, index_info):

        for i in range(len(index_info)):
            game_index, game_pos = index_info[i]

            # update position priorities
            priority = priorities[i, :]
            start_index = game_pos
            end_index = min(game_pos + len(priority), len(self.buffer[game_index].priorities))
            numpy.put(self.buffer[game_index].priorities, range(start_index, end_index), priority)

            # update game priorities
            self.game_priorities[game_index] = numpy.mean(self.buffer[game_index].priorities)

            self.max_recorded_game_priority = numpy.max(self.game_priorities)

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
