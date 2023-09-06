import copy
import numpy as np
from functools import partial
import torch

from deap import base, creator, tools, algorithms

from games.abstract_game import AbstractGame
from self_play import Node
import models

from games.tictactoe import MuZeroConfig, Game

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)


def evaluate(actions, model, observation, config):
    (
        root_predicted_value,
        reward,
        policy_logits,
        hidden_state,
    ) = model.initial_inference(observation)

    for action in actions:
        value, reward, policy_logits, hidden_state = model.recurrent_inference(
            hidden_state,
            torch.tensor([[action]]).to(observation.device),
        )

    reward = models.support_to_scalar(reward, config.support_size).item()
    return reward,

class RHEA:
    def __init__(self, config, game):
        self.game = game
        self.config = config
        self.play_id = -1
        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selStochasticUniversalSampling)

    # def game_evaluate(self, actions, game_stat=None, play_id=None):
    #     game_stat = copy.deepcopy(game_stat)
    #     game_stat.reset()
    #
    #     for i in range(len(actions)):
    #         player = game_stat.to_play()
    #         observation, reward, done = game_stat.step(actions[i])
    #         if done:
    #             break
    #
    #     game_stat.close()
    #     reward = reward if play_id == player else -reward
    #     # 因为i是从0开始的，如果第一个action就结束，会出现NAN异常
    #     reward /= i + 1  # 路径越长，回报越低。以便寻找到最近的路径
    #     return reward,
    #
    # def action_evaluate(self, actions):
    #     game_stat = copy.deepcopy(self.game)
    #     game_stat.reset()
    #
    #     for i in range(len(actions)):
    #         player = game_stat.to_play()
    #         observation, reward, done = game_stat.step(actions[i])
    #         if done:
    #             break
    #
    #     game_stat.close()
    #     reward = reward if self.play_id == player else -reward
    #
    #     return reward, actions[:(i+1)]
    #
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
        max_moves = max_moves if replace else min(len(actions), max_moves)
        return tools.initIterate(creator.Individual, partial(np.random.choice, actions, max_moves, replace=replace))
    def population(self, actions, max_moves, N, replace=False):
        return tools.initRepeat(list, partial(self.individual, actions, max_moves, replace), N)

    # def rhea(self, game_state:AbstractGame):
    #     self.game = game_state
    #     self.play_id = game_state.to_play()
    #     actions = game_state.legal_actions()
    #     self.toolbox.register("evaluate", evaluate, )
    #     pop = self.population(actions. self.config.max_moves)
    #
    #     pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
    #
    #     results = tools.selBest(pop, k=1)
    #
    #     return self.action_evaluate(results[0])



        # # 返回第一个动作和评分
        # return [(r[0],self.game_evaluate(actions, game_state, play_id)[0]) for r in results] # r[0]表示第一个动作

    def run(self,
            model,
            observation,
            legal_actions,
            to_play,
            action_replace,
            override_root_with=None,
            ):
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
        )

        # 检查可用的动作空间，如果小于等于1，则直接返回。因为进化算法无法杂交，会报错
        if len(legal_actions) <=1:
            return legal_actions
        else:
            # self.toolbox.register("evaluate", evaluate, model=model, observation=observation, config=self.config)
            self.toolbox.register("evaluate", self.evaluate)
            pop = self.population(legal_actions, self.config.max_moves, self.config.num_simulations, replace=action_replace)

            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=len(legal_actions), verbose=False)

            results = tools.selBest(pop, k=1)

            return results[0]

if __name__=="__main__":
    game = Game()
    config = MuZeroConfig()
    game.reset()
    done = False

    # rhea = RHEA(config, game)
    # pop = rhea.population(game.legal_actions(), 9, config.num_simulations, config.action_replace)
    #
    # print(pop)
    # rhea.toolbox.register("evaluate", rhea.evaluate)
    # pop, logbook = algorithms.eaSimple(pop, rhea.toolbox, cxpb=0.5, mutpb=0, ngen=9, verbose=False)
    #
    # results = tools.selBest(pop, k=1)
    # print(results)

    legal_actions = game.legal_actions()
    while not done and len(legal_actions) >1:
        legal_actions = game.legal_actions()
        rhea = RHEA(config, game)
        rhea.play_id = game.to_play()

        pop = rhea.population(legal_actions, config.max_moves, config.num_simulations, config.action_replace)

        rhea.toolbox.register("evaluate", rhea.evaluate)

        pop, logbook = algorithms.eaSimple(pop, rhea.toolbox, cxpb=0.5, mutpb=0.2, ngen=len(legal_actions), verbose=False)

        print(pop)
        results = tools.selBest(pop, k=1)
        print(results)
        action = results[0][0]
        observation, reward, done = game.step(action)
        # print(observation)












