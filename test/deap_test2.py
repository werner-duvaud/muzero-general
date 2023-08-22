import copy
import random

import deap
from games.tictactoe import Game, MuZeroConfig
import numpy as np
from functools import partial

config = MuZeroConfig()

from deap import base, creator, tools
import numpy as np
# 定义问题
# creator创建的是类，第一个参数是类名，第二个参数是基类，后面的是其它参数
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

legal_actions = 9

toolbox = base.Toolbox()
# 注册生成基因的函数。第一个参数是函数名，因此下面的调用是toolbox.Actions。
# 第二鸽参数是生成action的函数。
# 后边的参数是生成函数的参数，如此为np.random.choice(range(n), N, replace=False)
# toolbox.register("Actions", np.random.choice, range(legal_actions), config.max_moves, replace=False)
# # tools.initIterate返回一个生成的动作序列
# toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.Actions)

def individual(actions, max_moves, replace=False):
    max_moves = max_moves if replace else len(actions)
    return tools.initIterate(creator.Individual, partial(np.random.choice, actions, max_moves, replace=replace))

# print(individual([0,1,2,3,4], 9, replace=False))
# print(individual([0,1,2,3,4], 9, replace=True))
# exit()

def population(actions, max_moves, N, replace=False):
    return tools.initRepeat(list, partial(individual, actions, max_moves, replace), N)

pop = population(range(9),9,  N=4, replace=False)
print(pop)

# exit()
#
# # 重复生成动作序列
# toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

game = Game(0)

actions = game.legal_actions()
np.random.shuffle(actions)

def evaluate(actions):
    game = Game(1)
    game.reset()

    for i in range(len(actions)):
        player = game.to_play()
        observation, reward, done = game.step(actions[i])
        if done:
            break

    game.close()
    reward = reward if 0 == player else -reward
    # 因为i是从0开始的，如果第一个action就结束，会出现NAN异常
    reward /= i + 1  # 路径越长，回报越低。以便寻找到最近的路径
    return reward,


def game_evaluate(actions, game=None, play_id=None):
    game = copy.deepcopy(game)
    game.reset()

    for i in range(len(actions)):
        player = game.to_play()
        observation, reward, done = game.step(actions[i])
        if done:
            break

    game.close()
    reward = reward if play_id == player else -reward
    # 因为i是从0开始的，如果第一个action就结束，会出现NAN异常
    reward /= i+1 # 路径越长，回报越低。以便寻找到最近的路径
    return reward,
        # print(actions[i])
        # game.render()

toolbox.register("evaluate", game_evaluate, game=game, play_id = 0)
# toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# toolbox.register("select", tools.selTournament, tournsize=2000)
# toolbox.register("select", tools.selBest)
toolbox.register("select", tools.selStochasticUniversalSampling)

# pop = toolbox.population(n=100)
# pop = [[0, 6, 8, 7, 4, 5, 2, 1, 3], [0, 6, 3, 7, 4, 5, 2, 1, 8]]

from deap import algorithms
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
# # print(logbook)
results = tools.selBest(pop, k=1)

# results = [[0, 6, 8, 7, 4, 5, 2, 1, 3]]
print(results)
print(evaluate(results[0]))
reward = game_evaluate(results[0],game,0)
print(reward)

# reward = game_evaluate([0,1,3,4,6,7,2,5,9],game,0)
# print(reward)
#
# for i in range(20):
#     print(game_evaluate(pop[i], game, 0))

# print(evaluate(actions, game, 0))

# print(actions[:i])
# game.render()
# game2.render()
