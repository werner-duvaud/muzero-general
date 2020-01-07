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
        observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.config.batch_size):
            game_history = sample_game(self.buffer)
            game_pos = sample_position(game_history)
            actions = game_history.history[
                game_pos : game_pos + self.config.num_unroll_steps
            ]
            # Repeat precedent action to make "actions" of length "num_unroll_steps"
            actions.extend(
                [
                    actions[-1]
                    for _ in range(self.config.num_unroll_steps - len(actions) + 1)
                ]
            )
            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value, reward, policy = make_target(
                game_history,
                game_pos,
                self.config.num_unroll_steps,
                self.config.td_steps,
            )
            value_batch.append(value)
            reward_batch.append(reward)
            policy_batch.append(policy)

        return observation_batch, action_batch, value_batch, reward_batch, policy_batch


def sample_game(buffer):
    """
    Sample game from buffer either uniformly or according to some priority.
    """
    # TODO: sample with probability link to the highest difference between real and
    # predicted value (see paper appendix Training)
    return numpy.random.choice(buffer)


def sample_position(game_history):
    """
    Sample position from game either uniformly or according to some priority.
    """
    # TODO: according to some priority
    return numpy.random.choice(range(len(game_history.rewards)))


def make_target(game_history, state_index, num_unroll_steps, td_steps):
    """
    The value target is the discounted root value of the search tree td_steps into the
    future, plus the discounted sum of all rewards until then.
    """
    target_values, target_rewards, target_policies = [], [], []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
        bootstrap_index = current_index + td_steps
        if bootstrap_index < len(game_history.root_values):
            value = (
                game_history.root_values[bootstrap_index]
                * game_history.discount ** td_steps
            )
        else:
            value = 0

        for i, reward in enumerate(game_history.rewards[current_index:bootstrap_index]):
            value += reward * game_history.discount ** i

        if current_index < len(game_history.root_values):
            target_values.append(value)
            target_rewards.append(game_history.rewards[current_index])
            target_policies.append(game_history.child_visits[current_index])
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

    return target_values, target_rewards, target_policies
