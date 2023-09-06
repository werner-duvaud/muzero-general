import numpy as np

from games.simple_grid import Game
import random
import time

g = Game()
observation = g.env.get_observation()

# print(observer)
for i in range(1000):
    actions = g.legal_actions()
    observation, reward, done = g.step(random.choice(actions))
    # g.render()
    print(np.array(observation).shape)

    if done:
        break


    # time.sleep(10)

g.close()
