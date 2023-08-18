import random

import deap
from games.tictactoe import Game, MuZeroConfig
import numpy as np

config = MuZeroConfig()
print(config.max_moves)

from deap import base, creator, tools
import numpy as np
# 定义问题
creator.create('FitnessMax', base.Fitness, weights=(-1.0,)) #优化目标：单变量，求最小值
creator.create('Individual', list, fitness = creator.FitnessMax) #创建Individual类，继承list

legal_actions = 9

toolbox = base.Toolbox()
toolbox.register("Indices", random.sample, range(legal_actions), legal_actions)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.Indices)

ind1 = toolbox.Individual()
print(ind1)

toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

pop = toolbox.population(n=36)
print(len(pop))

def ea(game):
    pass

# game = Game(0)
# game.reset()
#
# for i in range(9):
#     game.render()
#     print(game.legal_actions())
#     observation, reward, done = game.step(np.random.choice(game.legal_actions()))
#
#     if done:
#         break
#
# game.render()
