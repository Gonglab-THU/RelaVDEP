import copy
import numpy
import ray
import torch
import time
from network import *

@ray.remote
class Reanalyse:
    def __init__(self, config, initial_checkpoint):
        self.config = config

        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.model = Network(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device('cuda' if self.config.reanalyse_on_gpu else 'cpu'))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def _reanalyse(self, shared_storage, replay_buffer):
        while ray.get(shared_storage.get_info.remote("training_step")) < 1:
            time.sleep(0.1)

        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps:
            with torch.no_grad():
                self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
                play_idx, play_history = ray.get(replay_buffer.sample_reanalysed_play.remote())
                observations = play_history.observation_history
                value = self.model.initial_inference(observations)[-1]
                value = support_to_scalar(value, self.config.support_size)
                play_history.reanalysed_values = value.squeeze().detach().cpu().numpy()
                replay_buffer.update_play_history.remote(play_idx, play_history)
                self.num_reanalysed_games += 1
                shared_storage.set_info.remote("num_reanalysed_games", self.num_reanalysed_games)
                if self.config.reanalyse_delay:
                    time.sleep(self.config.reanalyse_delay)
