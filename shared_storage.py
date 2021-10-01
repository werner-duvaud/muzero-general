import ray
import torch
import os


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, weights, game_name, config):
        self.config = config
        self.game_name = game_name
        self.weights = weights
        self.infos = {
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
        }
        self.video = None

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, path=None):
        self.weights = weights
        if not path:
            path = os.path.join(self.config.results_path, "model.weights")

        torch.save(self.weights, path)

    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value

    def set_video(self, value):
        self.video = value

    def get_video(self):
        return self.video
