import copy
import numpy as np
from functools import partial

from deap import base, creator, tools, algorithms

from games.abstract_game import AbstractGame

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

class RHEA:
    def __init__(self):
        self.game = None
        self.play_id = 0
        self.toolbox = base.Toolbox()
        self.register("mate", tools.cxTwoPoint)
        self.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.register("select", tools.selStochasticUniversalSampling)

    def game_evaluate(self, actions, game_stat=None, play_id=None):
        game_stat = copy.deepcopy(game_stat)
        game_stat.reset()

        for i in range(len(actions)):
            player = game_stat.to_play()
            observation, reward, done = game_stat.step(actions[i])
            if done:
                break

        game_stat.close()
        reward = reward if play_id == player else -reward
        # 因为i是从0开始的，如果第一个action就结束，会出现NAN异常
        reward /= i + 1  # 路径越长，回报越低。以便寻找到最近的路径
        return reward,

    def evaluate(self, actions):
        game_stat = copy.deepcopy(self.game)
        play_id = self.play_id

        game_stat.reset()

        for i in range(len(actions)):
            player = game_stat.to_play()
            observation, reward, done = game_stat.step(actions[i])
            if done:
                break

        game_stat.close()
        reward = reward if play_id == player else -reward
        # 因为i是从0开始的，如果第一个action就结束，会出现NAN异常
        reward /= i + 1  # 路径越长，回报越低。以便寻找到最近的路径
        return reward,

    def individual(self, actions, max_moves, replace=False):
        max_moves = max_moves if replace else len(actions)
        return tools.initIterate(creator.Individual, partial(np.random.choice, actions, max_moves, replace=replace))
    def population(self, actions, max_moves, N, replace=False):
        return tools.initRepeat(list, partial(self.individual, actions, max_moves, replace), N)

    def rhea(self, game_state:AbstractGame, config, play_id):
        actions = game_state.legal_actions()
        pop = self.population(actions. config.max_moves)
        self.toolbox.register("evaluate", self.game_evaluate, game=game_state, play_id=play_id)
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

        results = tools.selBest(pop, k=1)

        # 返回第一个动作和评分
        return [(r[0],self.game_evaluate(actions, game_state, play_id)[0]) for r in results] # r[0]表示第一个动作





