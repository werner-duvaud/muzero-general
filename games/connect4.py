import datetime
import os

import gym
import numpy
import torch

from games.game import Game
from games.base_config import BaseConfig

class MuZeroConfig(BaseConfig):
    def __init__(self):
        super(MuZeroConfig, self).__init__()
        self._results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    
    @property
    def game_class_name(self):
        return "Connect4"
    
    @property 
    def observation_shape(self):
        return (3, 6, 7)

    @property
    def action_space(self):
        return [i for i in range(7)]

    @property
    def players(self):
        return [i for i in range(2)]
    
    @property
    def stacked_observations(self):
        return 0

    @property
    def network(self):
        return "resnet" 
    
    @property
    def test_episodes(self):
        return 2


    @property
    def results_path(self):
        return self._results_path

    @property
    def training_steps(self):
        return 40000

class Connect4(Game):

    def __init__(self, seed=None):
        self.board = numpy.zeros((6, 7)).astype(int)
        self.player = 1

    def step(self, action):
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break

        done = self.is_finished()

        reward = 1 if done and 0 < len(self.legal_actions()) else 0

        self.player *= -1

        return self.get_observation(), reward * 10, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((6, 7), self.player).astype(float)
        return numpy.array([board_player1, board_player2, board_to_play])

    def to_play(self):
        return 0 if self.player == 1 else 1

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def reset(self):
        self.board = numpy.zeros((6, 7)).astype(int)
        self.player = 1
        return self.get_observation()


    def close(self):
        pass

    def render(self):
        print(self.board[::-1])
        input("Press enter to take a step ")

    def input_action(self):
        choice = input("Enter the column to play for the player {}: ".format(self.to_play()))
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def output_action(self, action):
        # returning action directly to match muzero's suggestion
        return "Play column {}, note column starts from 0".format(action)

    def is_finished(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j][i + 1] == self.player
                    and self.board[j][i + 2] == self.player
                    and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i] == self.player
                    and self.board[j + 2][i] == self.player
                    and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i + 1] == self.player
                    and self.board[j + 2][i + 2] == self.player
                    and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j - 1][i + 1] == self.player
                    and self.board[j - 2][i + 2] == self.player
                    and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        if len(self.legal_actions()) == 0:
            return True

        return False
