import copy

import ray
import torch


@ray.remote
class SharedStorage:
    """Ray actor holding the current checkpoint and training coordination state."""

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self):
        """Atomically save the current checkpoint to results_path/model.checkpoint."""
        path = self.config.results_path / "model.checkpoint"
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.get_checkpoint(), temporary_path)
        temporary_path.replace(path)

    def get_checkpoint(self):
        """Return a snapshot whose step matches its weights and optimizer state."""
        checkpoint = copy.deepcopy(self.current_checkpoint)
        checkpoint["training_step"] = checkpoint["weights_version"]
        return checkpoint

    def get_info(self, keys):
        """Return one field for a string key, or a field dict for a list of keys."""
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        """Update fields from a dict, or set one string key to a non-None value."""
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
