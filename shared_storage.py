import ray
import torch
import os


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, weights, config):
        self.config = config
        self.weights = weights
        self.info = {
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
        }

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, path=None):
        self.weights = weights
        if not path:
            path = os.path.join(self.config.results_path, "model.weights")

        if self.config.save_weights:
            torch.save(self.weights, path)

    def get_info(self):
        return self.info

    def set_info(self, key, value):
        self.info[key] = value
