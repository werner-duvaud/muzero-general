import copy

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """
    # SharedStorage定义了一个类，包含一个dict类型的config和当前的模型参数checkpoint。类通过ray访问
    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None): #将模型存储在文件中
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self): # 返回当前的模型参数，返回的是一个深拷贝，防止对当前模型的修改
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys): # 从config中获取参数
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None): # 向config中写入参数
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
