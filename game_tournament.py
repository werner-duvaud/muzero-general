import pickle

import torch
import copy
import numpy

from games.tictactoe import MuZeroConfig, Game
import models
from self_play import MCTS, GameHistory,SelfPlay
from simplifiedMuZero.search_policy.self_play_uniform_search import UniformSearch

class GameTournament:
    def __init__(self, config:MuZeroConfig):
        self.models = []
        self.game = Game(config.seed)
        self.config = config
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 0

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
            self.board[0, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[2, 2] == self.player
        ):
            return True
        if (
            self.board[2, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[0, 2] == self.player
        ):
            return True

        return False

    def play_competition(self, model1, search_policy1, model2, search_policy2):
        game_history = GameHistory()

        observation = self.game.reset()

        game_history.action_history.append(0)
        game_history.observation_history.append(observation)  # 添加reset之后的observation
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        model1.eval()
        model2.eval()

        is_model1 = True
        while not done:
            assert (
                    len(numpy.array(observation).shape) == 3
            ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
            assert (
                    numpy.array(observation).shape == self.config.observation_shape
            ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
            stacked_observations = game_history.get_stacked_observations(
                -1, self.config.stacked_observations, len(self.config.action_space)
            )

            model = model1 if is_model1 else model2
            search_policy = search_policy1 if is_model1 else search_policy2

            root, mcts_info = search_policy(self.config).run(
                model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),  # to_play返回当期玩游戏的玩家ID，默认是0
                True,
            )

            action = SelfPlay.select_action(root, 0)  # 第二个参数阈值为0表示不会偏移，选择最大的
            observation, reward, done = self.game.step(action)

            game_history.store_search_statistics(root, self.config.action_space)

            # Next batch
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)  # 添加到observation的队列。取数据是使用stacked_observation函数，从后往前取
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.game.to_play())

            # 如果没有结束，就取反
            if not done:
                is_model1 = not is_model1

            # print("is model",is_model1,  "reward is ", reward)

        # 将player的id变回之前的id，否则检查是否有圣者时会发生错误
        self.game.env.player *= -1

        # 返回值处理
        # |-----|-----|-----|
        # |  True  |  True  |  True  | 表示模型1结束，结果为获胜。因此获胜的模型为模型1
        # |  True  |  False  |  False  | 表示模型1结束，结果为失败。因此获胜的模型为模型2
        # |  False  |  True  |  False  | 表示模型2结束，结果为获胜。因此获胜的模型为模型2
        # |  False  |  False  |  True  | 表示模型2结束，结果为失败。因此获胜的模型为模型1
        return self.game.env.have_winner(), is_model1 == (reward > 0)

    def close_game(self):
        self.game.close()

    def play_tournament(self, models, rollnum=1000):
        model_num = len(models)

        for i in range(model_num):
            for j in range(i+1, model_num):
                model1 = models[i]["model"]
                model2 = models[j]["model"]

                # model1_win_num = sum([game_tournament.play_tournament(model2, "", model1, "") for i in range(rollnum)])
                model1_win_num = 0
                model2_win_num = 0
                no_winner_num = 0

                for _ in range(rollnum):
                    have_winner, is_model1 = game_tournament.play_competition(model1, MCTS, model2, MCTS)

                    if have_winner:
                        if is_model1:
                            model1_win_num += 1
                        else:
                            model2_win_num += 1
                    else:
                        no_winner_num += 1

                 # 交换顺序，再来一遍
                for _ in range(rollnum):
                    have_winner, is_model1 = game_tournament.play_competition(model2, MCTS, model1, MCTS)

                    if have_winner:
                        if is_model1:
                            model2_win_num += 1
                        else:
                            model1_win_num += 1
                    else:
                        no_winner_num += 1

                # print(is_model1)

                print(models[i]["name"],"   ,", models[j]["name"]," :   ")

                print(models[i]["name"], " win  :   ", model1_win_num)
                print(models[j]["name"], " win  :   ", model2_win_num)
                print("No Winner", no_winner_num)
                print("===================================")



def load_model(model_cls, model_path):
    checkpoint = torch.load(model_path)
    model = model_cls(config)
    model.set_weights(checkpoint["weights"])

    return model


if __name__ == "__main__":
    config = MuZeroConfig()

    checkpoint_path1 = r"C:\Users\chunchang\workspace\muzero-general\results\tictactoe\2023-08-10--20-03-39\model.checkpoint"
    muzero_model = load_model(models.MuZeroNetwork, checkpoint_path1)

    muzero_2net_checkpoint_path = r"C:\Users\chunchang\workspace\muzero-general\results\tictactoe\2023-08-15--11-08-42\muzero_2net\model.checkpoint"
    muzero_2net_model = load_model(models.MuZeroNetwork, muzero_2net_checkpoint_path)

    uniform_checkpoint_path = r"C:\Users\chunchang\workspace\muzero-general\results\tictactoe\2023-08-15--08-20-50\muzero_uniform\model.checkpoint"
    uniform_model = load_model(models.MuZeroNetwork, uniform_checkpoint_path)

    without_rb_checkpoint_path = r"C:\Users\chunchang\workspace\muzero-general\results\tictactoe\2023-08-16--04-35-40\muzero_without_rb\model.checkpoint"
    without_rb_model = load_model(models.MuZeroNetwork, without_rb_checkpoint_path)

    game_tournament = GameTournament(config)

    models = [
        {"name":"muzero_2net", "model":muzero_2net_model},
        {"name":"uniform", "model":uniform_model},
        {"name":"muzero", "model":muzero_model},
        {"name": "without_rb", "model": without_rb_model},
    ]

    # rollnum = 1000
    #
    # # model1_win_num = sum([game_tournament.play_tournament(model2, "", model1, "") for i in range(rollnum)])
    # model1_win_num = 0
    # model2_win_num = 0
    # no_winner_num = 0
    #
    # for i in range(rollnum):
    #     have_winner, is_model1 = game_tournament.play_competition(muzero_2net_model, MCTS, uniform_model, MCTS)
    #
    #     if have_winner:
    #         if is_model1:
    #             model1_win_num += 1
    #         else:
    #             model2_win_num += 1
    #     else:
    #         no_winner_num += 1
    #
    # # print(is_model1)
    #
    # print(model1_win_num)
    # print(model2_win_num)
    # print(no_winner_num)

    game_tournament.play_tournament(models, rollnum=100)

    game_tournament.close_game()


    # print(checkpoint)
