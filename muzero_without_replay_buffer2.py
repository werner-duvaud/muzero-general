import pathlib
import importlib
import ray

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

import math
import time
import copy
import nevergrad
import sys
import json

from simplifiedMuZero.without_rb.game_play import GamePlay
from simplifiedMuZero.without_rb.play_buffer import PlayBuffer
from simplifiedMuZero.without_rb.trainer import Trainer
from muzero import load_model_menu, hyperparameter_search

import models


class CPUActorWithClass:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config, model_cls):
        model = model_cls(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary

class MuZeroWithoutRB:
    def __init__(self, game_name, model_cls, config=None, split_resources_in=1, save_path_ex=None):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            print("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
            if save_path_ex:
                config.results_path /= save_path_ex
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        self.model_cls = model_cls

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActorWithClass()
        cpu_weights = cpu_actor.get_initial_weights(self.config, self.model_cls)
        self.checkpoint["weights"], self.summary = copy.deepcopy((cpu_weights))


    def logging_loop(self, writer, training_steps):
        # writer = SummaryWriter(config.results_path)

        # print(
        #     "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        # )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # # Save model representation
        # writer.add_text(
        #     "Model summary",
        #     str(model).replace("\n", " \n\n") # self.summary, 换成其它的
        # )
        # Loop for updating the training performance
        counter = training_steps

        try:
            if True:
            # while checkpoint["training_step"] < config.training_steps:
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    self.checkpoint["total_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    self.checkpoint["mean_value"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    self.checkpoint["episode_length"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward",
                    self.checkpoint["muzero_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward",
                    self.checkpoint["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    self.checkpoint["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", self.checkpoint["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", self.checkpoint["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    self.checkpoint["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    self.checkpoint["training_step"] / max(1, self.checkpoint["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", self.checkpoint["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", self.checkpoint["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", self.checkpoint["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", self.checkpoint["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", self.checkpoint["policy_loss"], counter)
                print(
                    f'Last test reward: {self.checkpoint["total_reward"]:.2f}. Training step: {self.checkpoint["training_step"]}/{self.config.training_steps}. Played games: {self.checkpoint["num_played_games"]}. Loss: {self.checkpoint["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                # time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        # if config.save_model:
        #     # Persist replay buffer to disk
        #     path = config.results_path / "replay_buffer.pkl"
        #     print(f"\n\nPersisting replay buffer games to disk at {path}")
        #     pickle.dump(
        #         {
        #             "buffer": buffer,
        #             "num_played_games": checkpoint["num_played_games"],
        #             "num_played_steps": checkpoint["num_played_steps"],
        #             "num_reanalysed_games": checkpoint["num_reanalysed_games"],
        #         },
        #         open(path, "wb"),
        #     )

    def update_gameplay_checkpoint(self, game_history):
        self.checkpoint["episode_length"] = len(game_history.action_history) - 1
        self.checkpoint["total_reward"] = sum(game_history.reward_history)
        self.checkpoint["mean_value"] = numpy.mean( [value for value in game_history.root_values if value])

        if 1 < len(self.config.players):
            self.checkpoint["muzero_reward"] = sum(
                        reward
                        for i, reward in enumerate(game_history.reward_history)
                        if game_history.to_play_history[i - 1]
                        == self.config.muzero_player
                    )
            self.checkpoint["opponent_reward"] = sum(
                        reward
                        for i, reward in enumerate(game_history.reward_history)
                        if game_history.to_play_history[i - 1]
                        != self.config.muzero_player
                    )

    def save_checkpoint(self, path=None): #将模型存储在文件中
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.checkpoint, path)

    def train(self, log_in_tensorboard=True):
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)


        trainer = Trainer(models.MuZeroNetwork, self.checkpoint, self.config)
        game_play = GamePlay(trainer.model, self.checkpoint, self.Game, self.config, self.config.seed)
        buffer = {}
        play_buffer = PlayBuffer(self.checkpoint, buffer, self.config)

        step = 1 # 间隔，即每次模拟后训练多少次
        max_steps = int(self.config.training_steps/step)
        # max_steps = 2000

        writer = SummaryWriter(self.config.results_path)

        for episode in range(max_steps):
            game_id, game_history = game_play.play_game(game_play.config.visit_softmax_temperature_fn(0), game_play.config.temperature_threshold, False, "self",0)

            # print(game_id)
            # print(game_history.action_history)
            # print(game_history.reward_history)
            # print(game_history.to_play_history)
            # # print(game_history.observation_history)
            # print("child visits", game_history.child_visits)
            # print(game_history.root_values) # root value指的是root节点的UCB值

            play_buffer.update_game_history(game_id, game_history)
            self.update_gameplay_checkpoint( game_history)

            for i in range(step):
                index_batch, batch = play_buffer.get_batch()
                # print(batch[1])
                trainer.update_lr()
                (
                    priorities,
                    total_loss,
                    value_loss,
                    reward_loss,
                    policy_loss,
                ) = trainer.update_weights(batch)


                training_step = episode * step + i
                if training_step % self.config.checkpoint_interval == 0:
                    self.checkpoint["weights"] = copy.deepcopy(trainer.model.get_weights())
                    self.checkpoint["optimizer_state"] =copy.deepcopy(models.dict_to_cpu(trainer.optimizer.state_dict()) )

                    if self.config.save_model:
                        self.save_checkpoint()
                self.checkpoint["training_step"] = training_step
                self.checkpoint["lr"] = trainer.optimizer.param_groups[0]["lr"]
                self.checkpoint["total_loss"] = total_loss
                self.checkpoint["value_loss"] = value_loss
                self.checkpoint["reward_loss"] = reward_loss
                self.checkpoint["policy_loss"] = policy_loss

            # print(training_step)
            # if training_step % 500 == 0:
            # if training_step % config.checkpoint_interval == 0:
            #     # print(training_step)
            #     logging_loop(config, checkpoint, writer)

            self.logging_loop(writer, training_step)


        writer.close()

        game_play.close_game()

if __name__ == "__main__":
    # muzero = MuZeroWithoutRB("",models.MuZeroNetwork, save_path_ex="muzero_without_rb")
    # start_time = time.time()
    # muzero.train()
    # end_time = time.time()
    # print("耗时: {:.2f}秒".format(end_time - start_time))
    model_cls = models.MuZeroNetwork
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py cartpole
        muzero = MuZeroWithoutRB(sys.argv[1], model_cls=model_cls)
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        muzero = MuZeroWithoutRB(sys.argv[1], config, model_cls=model_cls)
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
            if filename.name != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZeroWithoutRB(game_name, model_cls=model_cls)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                start_time = time.time()
                muzero.train()
                end_time = time.time()
                print("耗时: {:.2f}秒".format(end_time - start_time))
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZeroWithoutRB(game_name, best_hyperparameters , model_cls=model_cls)
            else:
                break
            print("\nDone")
