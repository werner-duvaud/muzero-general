import copy
import importlib
import math
import os
import pickle
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
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

        config (dict, MuZeroConfig, optional): Override the default config of the game.
        
        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    setattr(self.config, param, value)
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        total_gpus = (
            self.config.max_num_gpus
            if self.config.max_num_gpus is not None
            else torch.cuda.device_count()
        )
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Trick to force DataParallel to stay on CPU
        @ray.remote(num_gpus=0)
        def get_initial_weights(config):
            model = models.MuZeroNetwork(config)
            weigths = model.get_weights()
            summary = str(model).replace("\n", " \n\n")
            return weigths, summary

        # Weights and replay buffer used to initialize workers
        self.muzero_weights, self.summary = ray.get(
            get_initial_weights.remote(self.config)
        )
        self.replay_buffer = None

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.self_play_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray actors and launch the training.

        Args:
            num_gpus (int): Number of GPUS (if exists) that will be available for the training.

            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.            
        """
        if log_in_tensorboard or self.config.save_weights:
            os.makedirs(self.config.results_path, exist_ok=True)

        # Manage resources
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0, num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(copy.deepcopy(self.muzero_weights), self.config)
        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            copy.deepcopy(self.muzero_weights), self.config,
        )
        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.config)
        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(copy.deepcopy(self.muzero_weights), self.config)
        # Pre-load buffer if pulling from persistent storage
        if self.replay_buffer:
            for game_history_id in self.replay_buffer:
                self.replay_buffer_worker.save_game.remote(
                    self.replay_buffer[game_history_id], self.shared_storage_worker
                )
            print(f"\nLoaded {len(self.replay_buffer)} games from replay buffer.")
        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                copy.deepcopy(self.muzero_weights),
                self.Game,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0, num_gpus=num_gpus,
        ).remote(
            copy.deepcopy(self.muzero_weights),
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary", self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        info = ray.get(self.shared_storage_worker.get_info.remote())
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote())
                writer.add_scalar(
                    "1.Total reward/1.Total reward", info["total_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/2.Mean value", info["mean_value"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/3.Episode length", info["episode_length"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/4.MuZero reward", info["muzero_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/5.Opponent reward",
                    info["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self played games", info["num_played_games"], counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self played steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training steps per self played step ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total weighted loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy loss", info["policy_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.muzero_weights = ray.get(self.shared_storage_worker.get_weights.remote())
        self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        self.terminate_workers()

        if self.config.save_weights:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            pickle.dump(
                self.replay_buffer,
                open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb"),
            )

    def terminate_workers(self):
        """
        Kill the running workers if exist.
        """
        print("\nShutting down workers...")
        if self.self_play_workers:
            for worker in self.self_play_workers:
                ray.kill(worker)
        if self.test_worker:
            ray.kill(self.test_worker)
        if self.training_worker:
            ray.kill(self.training_worker)
        if self.reanalyse_worker:
            ray.kill(self.reanalyse_worker)
        if self.replay_buffer_worker:
            ray.kill(self.replay_buffer_worker)
        if self.shared_storage_worker:
            ray.kill(self.shared_storage_worker)
        if self.self_play_worker:
            ray.kill(self.self_play_worker)

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.self_play_worker = None

    def test(self, render, opponent=None, muzero_player=None, num_tests=1):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent.

            muzero_player (int): Integer with the player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self.self_play_worker = self_play.SelfPlay.options(
            num_cpus=0, num_gpus=self.num_gpus if self.config.selfplay_on_gpu else 0,
        ).remote(
            copy.deepcopy(self.muzero_weights),
            self.Game,
            self.config,
            numpy.random.randint(1000),
        )
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self.self_play_worker.play_game.remote(
                        0, 0, render, opponent, muzero_player,
                    )
                )
            )
        ray.get(self.self_play_worker.close_game.remote())

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        self.terminate_workers()
        return result

    def load_model(self, weights_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            weights_path (str): Path to model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load weights
        if weights_path:
            if os.path.exists(weights_path):
                self.muzero_weights = torch.load(weights_path)
                print(f"\nUsing weights from {weights_path}")
            else:
                print(f"\nThere is no model saved in {weights_path}.")

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                self.replay_buffer = pickle.load(open(replay_buffer_path, "rb"))
                print(f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.muzero_weights, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experience to launch in total.

        parallel_experiments (int): Number of experience to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    try:
        running_experiments = []
        best_training = None
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train(False)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if (
                    experiment
                    and experiment.config.training_steps
                    <= ray.get(experiment.shared_storage_worker.get_info.remote())[
                        "training_step"
                    ]
                ):
                    weights = ray.get(
                        experiment.shared_storage_worker.get_weights.remote()
                    )
                    time.sleep(2)
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "weights": copy.deepcopy(weights),
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(False)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        os.makedirs(best_training["config"].results_path, exist_ok=True)
        torch.save(
            best_training["weights"],
            os.path.join(best_training["config"].results_path, "model.weights"),
        )
        # Save the recommended hyperparameters
        text_file = open(
            os.path.join(best_training["config"].results_path, "best_parameters.txt"),
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


if __name__ == "__main__":
    print("\nWelcome to MuZero! Here's a list of games:")
    # Let user pick a game
    games = [
        filename[:-3]
        for filename in sorted(
            os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/games")
        )
        if filename.endswith(".py") and filename != "abstract_game.py"
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
    muzero = MuZero(game_name)

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Diagnose model",
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Test agaisnt MCTS",
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
            muzero.diagnose_model(30)
        elif choice == 3:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 4:
            muzero.test(render=True, opponent="human", muzero_player=0)
        elif choice == 5:
            muzero.test(render=True, opponent="MCTS", muzero_player=0)
        elif choice == 6:
            env = muzero.Game()
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                env.render()
        elif choice == 7:
            # Define here the parameters to tune
            # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
            muzero.terminate_workers()
            del muzero
            budget = 20
            parallel_experiments = 2
            lr_init = nevergrad.p.Log(a_min=0.0001, a_max=0.1)
            discount = nevergrad.p.Scalar(lower=0.95, upper=0.9999)
            parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
            best_hyperparameters = hyperparameter_search(
                game_name, parametrization, budget, parallel_experiments, 10
            )
            muzero = MuZero(game_name, best_hyperparameters)
        else:
            break
        print("\nDone")

    ray.shutdown()

    ## Successive training, create a new config file for each experiment
    # experiments = ["cartpole", "tictactoe"]
    # for experiment in experiments:
    #     print(f"\nStarting experiment {experiment}")
    #     try:
    #         muzero = MuZero(experiment)
    #         muzero.train()
    #     except:
    #         print(f"Skipping {experiment}, an error has occurred.")

    # import argparse
    # args = argparse.Namespace()
    # config = vars(args)
    # muzero = MuZero(config.game_name, config)
    # muzero.train()
