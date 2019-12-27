import importlib
import os
import time

import numpy
import ray
import torch

import network
import self_play


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file in the "./games" directory.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test()
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

        # Fix random generator seed for reproductibility
        # TODO: check if results do not change from one run to another
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.best_model = network.Network(
            self.config.observation_shape,
            len(self.config.action_space),
            self.config.encoding_size,
            self.config.hidden_size,
        )

    def train(self):
        # Initialize and launch components that work simultaneously
        ray.init()
        model = self.best_model
        model.train()
        storage = network.SharedStorage.remote(model)
        replay_buffer = self_play.ReplayBuffer.remote(self.config)
        for seed in range(self.config.num_actors):
            self_play.run_selfplay.remote(
                self.Game,
                self.config,
                storage,
                replay_buffer,
                model,
                self.config.seed + seed,
            )

        # Initialize network for training
        model = model.to(self.config.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.lr_init,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        # Wait for replay buffer to be non-empty
        while ray.get(replay_buffer.length.remote()) == 0:
            time.sleep(0.1)

        # Training loop
        best_test_rewards = None
        for training_step in range(self.config.training_steps):
            model.train()
            storage.set_training_step.remote(training_step)

            # Make the model available for self-play
            if training_step % self.config.checkpoint_interval == 0:
                storage.set_weights.remote(model.state_dict())

            # Update learning rate
            lr = self.config.lr_init * self.config.lr_decay_rate ** (
                training_step / self.config.lr_decay_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Train on a batch.
            batch = ray.get(
                replay_buffer.sample_batch.remote(
                    self.config.num_unroll_steps, self.config.td_steps
                )
            )
            loss = network.update_weights(optimizer, model, batch, self.config)

            # Test the current model and save it on disk if it is the best
            if training_step % self.config.test_interval == 0:
                total_test_rewards = self.test(model=model, render=False)
                if best_test_rewards is None or sum(total_test_rewards) >= sum(
                    best_test_rewards
                ):
                    self.best_model = model
                    best_test_rewards = total_test_rewards
                    self.save_model()

            print(
                "Training step: {}\nBuffer Size: {}\nLearning rate: {}\nLoss: {}\nLast test score: {}\nBest sest score: {}\n".format(
                    training_step,
                    ray.get(replay_buffer.length.remote()),
                    lr,
                    loss,
                    str(total_test_rewards),
                    str(best_test_rewards),
                )
            )

        # Finally, save the latest network in the shared storage and end the self-play
        storage.set_weights.remote(model.state_dict())
        ray.shutdown()

    def test(self, model=None, render=True):
        if not model:
            model = self.best_model

        model.to(self.config.device)
        test_rewards = []
        game = self.Game()

        model.eval()
        with torch.no_grad():
            for _ in range(self.config.test_episodes):
                observation = game.reset()
                done = False
                total_reward = 0
                while not done:
                    if render:
                        game.render()
                    root = self_play.MCTS(self.config).run(model, observation, False)
                    action = self_play.select_action(root, temperature=0)
                    observation, reward, done = game.step(action)
                    total_reward += reward
                test_rewards.append(total_reward)

        return test_rewards

    def save_model(self, model=None, path=None):
        if not model:
            model = self.best_model
        if not path:
            path = os.path.join(self.config.results_path, self.game_name)

        torch.save(model.state_dict(), path)

    def load_model(self, path=None):
        if not path:
            path = os.path.join(self.config.results_path, self.game_name)
        self.best_model = network.Network(
            self.config.observation_shape,
            len(self.config.action_space),
            self.config.encoding_size,
            self.config.hidden_size,
        )
        try:
            self.best_model.load_state_dict(torch.load(path))
        except FileNotFoundError:
            print("There is no model saved in {}.".format(path))


if __name__ == "__main__":
    # Load the game and the parameters from ./games/file_name.py
    muzero = MuZero("cartpole")
    muzero.load_model()
    muzero.train()
    muzero.test()
