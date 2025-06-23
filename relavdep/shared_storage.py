import copy
import ray
import torch

@ray.remote
class SharedStorage:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, save_path):
        torch.save(self.checkpoint, save_path)

    def get_checkpoint(self):
        return copy.deepcopy(self.checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.checkpoint.update(keys)
        else:
            raise TypeError
