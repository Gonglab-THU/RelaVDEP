import time
import ray
import torch
from scripts.models import *

@ray.remote
class RewardModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.predict_on_gpu else 'cpu')
        self.base_model = BaseModel(config.data_dir, self.device)

        if config.rm_type == 'SmallFitness':
            self.fitness_model = SmallFitness(n_layer=config.n_layer)
        elif config.rm_type == 'LargeFitness':  
            self.fitness_model = LargeFitness()
        elif config.rm_type == 'SmallStab':
            self.fitness_model = SmallStab(n_layer=config.n_layer)
        elif config.rm_type == 'LargeStab':
            self.fitness_model = LargeStab()
        else:
            raise ValueError(f"Invalid model type: {config.rm_type}")
        
        best_dict = torch.load(config.rm_params, map_location=torch.device(self.device))
        self.fitness_model.load_state_dict(best_dict)
        self.fitness_model.eval().to(self.device)
    
    def pred_fitness(self, wt_data, mut_data):
        wt_data = dict_to_device(wt_data, self.device)
        mut_data = dict_to_device(mut_data, self.device)
        with torch.no_grad():
            fitness = self.fitness_model(wt_data, mut_data)
        return fitness.item()
    
    def inference(self, seq):
        data = self.base_model.inference(seq)
        return data

    def _predict(self, shared_storage, manager):
        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps:
            while ray.get(manager.get_tasks_keys.remote()) == []:
                time.sleep(0.1)
            for task, [wt_seq, mut_seq] in ray.get(manager.get_tasks.remote()).items():
                if wt_seq in ray.get(manager.get_predictions.remote()):
                    wt_data = ray.get(manager.get_prediction_item.remote(wt_seq))
                else:
                    wt_data = self.inference(wt_seq)
                    wt_data = dict_to_device(wt_data, 'cpu')
                    manager.save_prediction.remote(wt_seq, wt_data)
                
                if mut_seq in ray.get(manager.get_predictions.remote()):
                    mut_data = ray.get(manager.get_prediction_item.remote(mut_seq))
                else:
                    mut_data = self.inference(mut_seq)
                    mut_data = dict_to_device(mut_data, 'cpu')
                    manager.save_prediction.remote(mut_seq, mut_data)
                
                fitness = self.pred_fitness(wt_data, mut_data)
                manager.save_result.remote(task, fitness)
                manager.remove_task.remote(task)
