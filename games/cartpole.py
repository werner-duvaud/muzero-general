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
        return "Cartpole"
    
    @property 
    def observation_shape(self):
        return (1, 1, 4)

    @property
    def action_space(self):
        return [i for i in range(2)]

    @property
    def players(self):
        return [i for i in range(1)]
    
    @property
    def stacked_observations(self):
        return 0

    @property
    def network(self):
        return "fullyconnected" 
    
    @property
    def test_episodes(self):
        return 2


    @property
    def results_path(self):
        return self._results_path

    @property
    def training_steps(self):
        return 5000

class Cartpole(Game):

    def __init__(self, seed=None):
        self.env = gym.make("CartPole-v1")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, done

    def to_play(self):
        return 0

    def legal_actions(self):
        return [i for i in range(2)]

    def reset(self):
        return numpy.array([[self.env.reset()]])

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def input_action(self):
        pass

    def output_action(self, action_number):
        descriptions = [
            "Push cart to the left",
            "Push cart to the right"
        ]
        return "{}. {}".format(action_number, descriptions[action_number])
