import math
import time

import numpy
import ray
import torch

import models
from simplifiedMuZero.search_policy.RHEA2 import RHEA
from self_play import GameHistory


@ray.remote
class SelfPlayRhea:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        # self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ): # 如果当前的训练步数低于训练总步数，并且没有终止的话，继续进行训练
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights"))) # 从shared_storage中获取当前的参数

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    #play game 运行
    # 合法的actions是固定的，由游戏文件提供(在本函数中，可以看到调用legal_actions函数没有使用env，这表面现游戏环境于的改变于动作无关)。
    # 运行步骤：
    #   1. 创建GameHistory用来存储数据
    #   2. 检查游戏是否结束或者到底最大移动次数
    #   3. 获取stacked observation（因为有些游戏需要考虑之前的历史数据和移动轨迹)
    #   4. 运行MCTS搜索下一步的action
    #   5. 调用游戏函数step（action），获取下一步action之后的observation、reward和done
    #   6. 持续运行2-5步直到结束
    #   7. 返回GameHistory
    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation) # 添加reset之后的observation
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ): # 游戏没有结束且运行步数小于最大移动步长
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )
                # index是-1，game_history 会在创建时添加reset的observation，因此其长度为1.index取模（%）之后时1
                # config.stacked_observationis是存储之前的observation的数量，如果不要之前的信息，可以设为0，这样就不会存储之前的信息

                # 一下的if-else部分主要是为了选择一个动作
                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    # root, mcts_info = MCTS(self.config).run(
                    #     self.model,
                    #     stacked_observations,
                    #     self.game.legal_actions(),
                    #     self.game.to_play(), # to_play返回当期玩游戏的玩家ID，默认是0
                    #     True,
                    # )
                    # action = self.select_action(
                    #     root,
                    #     temperature
                    #     if not temperature_threshold
                    #     or len(game_history.action_history) < temperature_threshold
                    #     else 0,
                    # ) # 根据temperature选择动作
                    actions = RHEA(self.config, self.game).run(self.model,
                                          stacked_observations,
                                          self.game.legal_actions(),
                                          self.game.to_play(),
                                          self.config.action_replace,
                                          )
                    action = actions[0]

                else:
                    action, root = self.select_opponent_action( #选择对手动作，分为随机，human和expert三种
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action) # 运行游戏

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                # game_history.store_search_statistics(root, self.config.action_space)
                game_history.root_values.append(reward)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation) #添加到observation的队列。取数据是使用stacked_observation函数，从后往前取
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            return self.game.human_to_action(), None
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )
