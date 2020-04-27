import copy
import importlib
import os
import pickle
import time

import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import models
import replay_buffer
import self_play
import shared_storage
import trainer


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True, opponent="self", muzero_player=None)
    """

    def __init__(self, game_name):
        self.game_name = game_name

        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + self.game_name)
            self.config = game_module.MuZeroConfig()
            self.Game = game_module.Game
        except Exception as err:
            print(
                '{} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'.format(
                    self.game_name
                )
            )
            raise err

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Weights and replay buffer used to initialize workers
        self.muzero_weights = models.MuZeroNetwork(self.config).get_weights()
        self.replay_buffer = None

    def train(self):
        ray.init()
        os.makedirs(self.config.results_path, exist_ok=True)

        # Initialize workers
        training_worker = trainer.Trainer.options(
            num_gpus=1 if "cuda" in self.config.training_device else 0
        ).remote(copy.deepcopy(self.muzero_weights), self.config)
        shared_storage_worker = shared_storage.SharedStorage.remote(
            copy.deepcopy(self.muzero_weights), self.game_name, self.config,
        )
        replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.config)
        # Pre-load buffer if pulling from persistent storage
        if self.replay_buffer:
            for game_history_id in self.replay_buffer:
                replay_buffer_worker.save_game.remote(
                    self.replay_buffer[game_history_id]
                )
            print(
                "\nLoaded {} games from replay buffer.".format(len(self.replay_buffer))
            )
        self_play_workers = [
            self_play.SelfPlay.remote(
                copy.deepcopy(self.muzero_weights),
                self.Game(self.config.seed + seed),
                self.config,
            )
            for seed in range(self.config.num_actors)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                shared_storage_worker, replay_buffer_worker
            )
            for self_play_worker in self_play_workers
        ]
        training_worker.continuous_update_weights.remote(
            replay_buffer_worker, shared_storage_worker
        )

        # Save performance in TensorBoard
        self._logging_loop(shared_storage_worker, replay_buffer_worker)

        self.muzero_weights = ray.get(shared_storage_worker.get_weights.remote())
        self.replay_buffer = ray.get(replay_buffer_worker.get_buffer.remote())
        # Persist replay buffer to disk
        print("\n\nPersisting replay buffer games to disk...")
        pickle.dump(
            self.replay_buffer,
            open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
        )
        # End running actors
        ray.shutdown()

    def _logging_loop(self, shared_storage_worker, replay_buffer_worker):
        """
        Keep track of the training performance
        """
        # Launch the test worker to get performance metrics
        test_worker = self_play.SelfPlay.remote(
            copy.deepcopy(self.muzero_weights),
            self.Game(self.config.seed + self.config.num_actors),
            self.config,
        )
        test_worker.continuous_self_play.remote(shared_storage_worker, None, True)

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            "| {} | {} |".format(key, value)
            for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            str(models.MuZeroNetwork(self.config)).replace("\n", " \n\n"),
        )
        # Loop for updating the training performance
        counter = 0
        infos = ray.get(shared_storage_worker.get_infos.remote())
        try:
            while infos["training_step"] < self.config.training_steps:
                infos = ray.get(shared_storage_worker.get_infos.remote())
                writer.add_scalar(
                    "1.Total reward/1.Total reward", infos["total_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/2.Mean value", infos["mean_value"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/3.Episode length", infos["episode_length"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/4.MuZero reward", infos["muzero_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/5.Opponent reward",
                    infos["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self played games",
                    ray.get(replay_buffer_worker.get_self_play_count.remote()),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training steps", infos["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self played games per training step ratio",
                    ray.get(replay_buffer_worker.get_self_play_count.remote())
                    / max(1, infos["training_step"]),
                    counter,
                )
                writer.add_scalar("2.Workers/4.Learning rate", infos["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total weighted loss", infos["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value loss", infos["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward loss", infos["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy loss", infos["policy_loss"], counter)
                print(
                    "Last test reward: {0:.2f}. Training step: {1}/{2}. Played games: {3}. Loss: {4:.2f}".format(
                        infos["total_reward"],
                        infos["training_step"],
                        self.config.training_steps,
                        ray.get(replay_buffer_worker.get_self_play_count.remote()),
                        infos["total_loss"],
                    ),
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt as err:
            # Comment the line below to be able to stop the training but keep running
            # raise err
            pass

    def test(self, render, opponent, muzero_player):
        """
        Test the model in a dedicated thread.

        Args:
            render: Boolean to display or not the environment.

            opponent: "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent.

            muzero_player: Integer with the player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn.
        """
        print("\nTesting...")
        ray.init()
        self_play_workers = self_play.SelfPlay.remote(
            copy.deepcopy(self.muzero_weights),
            self.Game(numpy.random.randint(1000)),
            self.config,
        )
        history = ray.get(
            self_play_workers.play_game.remote(0, 0, render, opponent, muzero_player)
        )
        ray.shutdown()
        return sum(history.reward_history)

    def load_model(self, weights_path=None, replay_buffer_path=None):
        # Load weights
        if weights_path:
            if os.path.exists(weights_path):
                self.muzero_weights = torch.load(weights_path)
                print("\nUsing weights from {}".format(weights_path))
            else:
                print("\nThere is no model saved in {}.".format(weights_path))

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                self.replay_buffer = pickle.load(open(replay_buffer_path, "rb"))
                print("\nInitializing replay buffer with {}".format(replay_buffer_path))
            else:
                print(
                    "Warning: Replay buffer path '{}' doesn't exist.  Using empty buffer.".format(
                        replay_buffer_path
                    )
                )


if __name__ == "__main__":
    print("\nWelcome to MuZero! Here's a list of games:")
    # Let user pick a game
    games = [
        filename[:-3]
        for filename in sorted(os.listdir("./games"))
        if filename.endswith(".py") and filename != "abstract_game.py"
    ]
    for i in range(len(games)):
        print("{}. {}".format(i, games[i]))
    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i) for i in range(len(games))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    # Initialize MuZero
    choice = int(choice)
    muzero = MuZero(games[choice])

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print("{}. {}".format(i, options[i]))

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        if choice == 0:
            muzero.train()
        elif choice == 1:
            weights_path = input(
                "Enter a path to the model.weights, or ENTER if none: "
            )
            while weights_path and not os.path.isfile(weights_path):
                weights_path = input("Invalid weights path. Try again: ")
            replay_buffer_path = input(
                "Enter path for existing replay buffer, or ENTER if none: "
            )
            while replay_buffer_path and not os.path.isfile(replay_buffer_path):
                replay_buffer_path = input("Invalid replay buffer path. Try again: ")
            muzero.load_model(
                weights_path=weights_path, replay_buffer_path=replay_buffer_path
            )
        elif choice == 2:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 3:
            muzero.test(render=True, opponent="human", muzero_player=0)
        elif choice == 4:
            env = muzero.Game()
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(
                    "\nAction: {}\nReward: {}".format(
                        env.action_to_string(action), reward
                    )
                )
                env.render()
        else:
            break
        print("\nDone")

    ## Successive training, create a new config file for each experiment
    # experiments = ["cartpole", "tictactoe"]
    # for experiment in experiments:
    #     print("\nStarting experiment {}".format(experiment))
    #     try:
    #         muzero = MuZero(experiment)
    #         muzero.train()
    #     except:
    #         print("Skipping {}, an error has occurred.".format(experiment))
