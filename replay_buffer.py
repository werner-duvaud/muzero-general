import collections
import pickle
import time

import math
import numpy
import random
import ray
import sqlite3
import torch

import models


class GameHistoryDao(collections.MutableMapping):
    """
    Data Access Object for the game histories comprising the replay buffer
    """

    @staticmethod
    def assemble_game_history(result):

        # assemble priorities into the game history
        # structure: (id, game_priority, priorities, reanalysed_predicted_root_values, object)
        game_history = pickle.loads(result[4])
        game_history.game_priority = result[1]
        game_history.priorities = pickle.loads(result[2])
        game_history.reanalysed_predicted_root_values = pickle.loads(result[3])

        return result[0], game_history

    @staticmethod
    def disassemble_game_history(value):

        # disassemble the priorities from the game history
        game_priority = value.game_priority
        priorities = value.priorities
        reanalysed_predicted_root_values = value.reanalysed_predicted_root_values

        # avoid storing duplicate data (it will be reassembled later)
        value.game_priority = None
        value.priorities = None
        value.reanalysed_predicted_root_values = None

        return game_priority, priorities, reanalysed_predicted_root_values

    def __init__(self, file):
        self.connection = sqlite3.connect(file)
        self.connection.create_function('log', 1, math.log10)
        self.connection.create_function('rand', 0, random.random)
        self.connection.execute("CREATE TABLE IF NOT EXISTS game_history("
                                "   id INTEGER PRIMARY KEY ASC,"
                                "   game_priority REAL,"
                                "   priorities TEXT,"
                                "   reanalysed_predicted_root_values TEXT,"
                                "   object TEXT"
                                ")")
        self.connection.commit()

    def __len__(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM game_history")
        result = cursor.fetchone()[0]
        return result

    def __getitem__(self, key):
        cursor = self.connection.cursor()
        cursor.execute("SELECT  id,"
                       "        game_priority,"
                       "        priorities,"
                       "        reanalysed_predicted_root_values,"
                       "        object"
                       "    FROM game_history"
                       "    WHERE id = ?", (int(key),))
        result = cursor.fetchone()
        if result is None:
            raise KeyError()

        return self.assemble_game_history(result)

    def __setitem__(self, key, value):

        game_priority, priorities, reanalysed_predicted_root_values = self.disassemble_game_history(value)

        cursor = self.connection.cursor()
        cursor.execute("REPLACE INTO game_history("
                       "    id,"
                       "    game_priority,"
                       "    priorities,"
                       "    reanalysed_predicted_root_values,"
                       "    object"
                       ") VALUES(?, ?, ?, ?, ?)", (
                            int(key),
                            float(game_priority),
                            pickle.dumps(priorities),
                            pickle.dumps(reanalysed_predicted_root_values),
                            pickle.dumps(value)
                        ))
        self.connection.commit()

    def __delitem__(self, key):
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM game_history WHERE id = ?", (int(key),))
        self.connection.commit()
        if cursor.rowcount == 0:
            raise KeyError()

    def keys(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM game_history ORDER BY id ASC")
        for row in cursor.fetchall():
            yield row[0]

    def values(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id,"
                       "       game_priority,"
                       "       priorities,"
                       "       reanalysed_predicted_root_values,"
                       "       object"
                       "    FROM game_history ORDER BY id ASC")
        for row in cursor:
            yield self.assemble_game_history(row)[1]

    def items(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id,"
                       "       game_priority,"
                       "       priorities,"
                       "       reanalysed_predicted_root_values,"
                       "       object"
                       "    FROM game_history ORDER BY id ASC")
        for row in cursor:
            yield self.assemble_game_history(row)

    def __contains__(self, key):
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM game_history WHERE id = ?", (int(key),))
        return cursor.fetchone()[0] > 0

    def __iter__(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM game_history ORDER BY id ASC")
        for row in cursor:
            yield row[0]

    def priorities(self, game_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT priorities FROM game_history WHERE id = ?", (game_id,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError()
        return pickle.loads(result[0])

    def min_id(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT MIN(id) FROM game_history")
        return cursor.fetchone()[0]

    def sample_n(self, n):
        cursor = self.connection.cursor()
        cursor.execute("SELECT  id,"
                       "        game_priority,"
                       "        priorities,"
                       "        reanalysed_predicted_root_values,"
                       "        object"
                       "    FROM game_history WHERE id IN ("
                       "        SELECT id FROM game_history"
                       "        ORDER BY RANDOM()"
                       "        LIMIT ?"
                       "    )", (int(n),))
        for row in cursor:
            yield self.assemble_game_history(row)

    def sample_n_ranked(self, n):
        # reference: https://stackoverflow.com/a/12301949
        cursor = self.connection.cursor()
        cursor.execute("SELECT  id,"
                       "        game_priority,"
                       "        priorities,"
                       "        reanalysed_predicted_root_values,"
                       "        object"
                       "    FROM game_history WHERE id IN ("
                       "        SELECT id FROM game_history"
                       "        ORDER BY -LOG(1.0 - RAND()) / game_priority"
                       "        LIMIT ?"
                       "    )", (int(n),))
        for row in cursor:
            yield self.assemble_game_history(row)

    def update_priorities(self, game_id, game_priority, priorities):
        cursor = self.connection.cursor()
        cursor.execute("UPDATE game_history"
                       "    SET game_priority = ?,"
                       "        priorities = ?"
                       "    WHERE"
                       "        id = ?", (
                            float(game_priority),
                            pickle.dumps(priorities),
                            int(game_id)
                        ))
        self.connection.commit()

    def update_reanalysed_values(self, game_id, reanalysed_predicted_root_values):
        cursor = self.connection.cursor()
        cursor.execute("UPDATE game_history"
                       "    SET reanalysed_predicted_root_values = ?"
                       "    WHERE"
                       "        id = ?", (
                            pickle.dumps(reanalysed_predicted_root_values),
                            int(game_id)
                        ))
        self.connection.commit()


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer_file = initial_buffer
        self.buffer = GameHistoryDao(file=self.buffer_file)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer_file

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
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
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
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

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        return next(iter(self.sample_n_games(1)))

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            samples = self.buffer.sample_n_ranked(n_games)
        else:
            samples = self.buffer.sample_n(n_games)

        for sample in samples:
            game_id = sample[0]
            game_history = sample[1]
            game_prob = game_history.game_priority
            yield game_id, game_history, game_prob

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_reanalysed_values(self, game_id, reanalysed_predicted_root_values):
        self.buffer.update_reanalysed_values(game_id, reanalysed_predicted_root_values)

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if self.buffer.min_id() <= game_id:

                # select record from database (can't update in place)
                priorities_record = self.buffer.priorities(game_id)

                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(priorities_record)
                )
                priorities_record[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                game_priority = numpy.max(
                    priorities_record
                )

                # update record
                self.buffer.update_priorities(game_id, game_priority, priorities_record)

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
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
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            reanalysed_predicted_root_values = game_history.reanalysed_predicted_root_values
            if self.config.use_last_model_value:
                observations = [
                    game_history.get_stacked_observations(
                        i, self.config.stacked_observations
                    )
                    for i in range(len(game_history.root_values))
                ]

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_reanalysed_values.remote(game_id, reanalysed_predicted_root_values)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
