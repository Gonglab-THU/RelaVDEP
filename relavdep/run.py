import os
import ray
import time
import timeit
import copy
import pickle
import shutil
import config
import environment
import manager
import shared_storage
import replay_buffer
import reanalyse
import trainer
import reward_model
import player
import network
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = "1.0"

parser = argparse.ArgumentParser(description='Reinforcement Learning assisted Directed Evolution for Proteins')
parser.add_argument('--fasta', type=str, required=True, help='Protein sequence')
parser.add_argument('--rm_params', type=str, required=True, help='Supervised fine-tuned reward model parameters')
parser.add_argument('--rm_type', type=str, default='SmallFitness', choices=['SmallFitness', 'LargeFitness', 'SmallStab', 'LargeStab'], 
                    help='Type of the reward model (default: %(default)s)')
parser.add_argument('--n_layer', type=int, default=1, help='Number of downstream MLP layers (default: %(default)s)')
parser.add_argument('--restraint', type=str, default=None, help='Restraint file (.npz format)')
parser.add_argument('--data_dir', type=str, default='data/params', help='Directory for model parameters (default: %(default)s)')
parser.add_argument('--output', type=str, default='tasks', help='Output directory (default: %(default)s)')
parser.add_argument('--temp_dir', type=str, default='/tmp/ray', help='Temporary directory for spilling object store (default: %(default)s)')
parser.add_argument('--max_mut', type=int, default=4, help='Maximum mutation counts (default: %(default)s)')
parser.add_argument('--n_players', type=int, default=6, help='Number of self-play workers (default: %(default)s)')
parser.add_argument('--n_sim', type=int, default=1200, help='Number of MCTS simulations (default: %(default)s)')
parser.add_argument('--train_delay', type=int, default=2, help='Training delay (default: %(default)s)')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)s)')
parser.add_argument('--no_buffer', action='store_true', help='Skip saving replay buffer during training (default: %(default)s)')
parser.add_argument('--unroll_steps', type=int, default=None, help='Number of unroll steps (default: %(default)s)')
parser.add_argument('--td_steps', type=int, default=None, help='Number of td steps (default: %(default)s)')
parser.add_argument('--init_checkpoint', type=str, default=None, help='Initialized checkpoint (default: %(default)s)')
parser.add_argument('--init_buffer', type=str, default=None, help='Initialized replay buffer (default: %(default)s)')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Fasta file does not exist!"
assert os.path.exists(args.data_dir), "Params directory does not exist!"
assert args.n_gpus <= torch.cuda.device_count(), "Insufficient number of available GPUs!"
os.makedirs(args.output, exist_ok=True)

if args.restraint: 
    assert os.path.exists(args.restraint), "Restraint file (rst.npz) does not exist!"
if args.unroll_steps:
    assert args.unroll_steps <= args.max_mut, "Unroll steps cannot exceed the number of maximum mutations!"
if args.td_steps:
    assert args.td_steps <= args.max_mut, "Td steps cannot exceed the number of maximum mutations!"
if args.init_checkpoint:
    assert os.path.exists(args.init_checkpoint), "Checkpoint file (checkpoint.pth) does not exist!"
if args.init_buffer:
    assert os.path.exists(args.init_buffer), "Replay buffer file (replay_buffer.pkl) does not exist!"

class Evolution:
    def __init__(self, args):
        self.config = config.Config(args)
        self.config.avaliables = environment.Environment(self.config, 'cpu').legal_actions()
        self.config.pb_c_init = 1.25 * (len(self.config.avaliables) // 19 + 1)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if args.init_checkpoint:
            print("Loading initial checkpoint...")
            self.checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        else:
            initial_weight = network.Network(self.config).get_weights()
            self.checkpoint = {"weights": copy.deepcopy(initial_weight), "optimizer": None, 
                               "training_step": 0, "learning_rate": 0, "total_loss": 0, 
                               "policy_loss": 0, "value_loss": 0, "reward_loss": 0, 
                               "total_reward": 0, "mean_value": 0, "episode_length": 0, 
                               "max_reward": 0, "num_played_games": 0, "num_reanalysed_games": 0}

        if args.init_buffer:
            print("Loading initial replay buffer...")
            with open(args.init_buffer, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        else:
            self.replay_buffer = {}

        self.temp_dir = os.path.join(args.temp_dir, self.config.task_name)
        os.makedirs(self.temp_dir, exist_ok=True)
        ray.init(log_to_driver=False, _temp_dir=os.path.abspath(self.temp_dir), num_gpus=self.config.n_gpus)

    def evolve(self):
        gpu_per_worker = self.config.n_gpus / (3 * self.config.train_on_gpu + 
                                               5 * self.config.predict_on_gpu + 
                                               self.config.reanalyse_on_gpu + 
                                               self.config.test_on_gpu + 
                                               self.config.n_players * self.config.play_on_gpu)
            
        if self.config.predict_on_gpu:
            num_predictor_gpus = 5 * gpu_per_worker if 5 * gpu_per_worker < 1 else 1
        else:
            num_predictor_gpus = 0
            
        if self.config.train_on_gpu:
            num_trainer_gpus = 3 * gpu_per_worker if 3 * gpu_per_worker < 1 else 1
        else:
            num_trainer_gpus = 0

        self.manager_worker = manager.Manager.remote()
        self.shared_storage_worker = shared_storage.SharedStorage.remote(self.config, self.checkpoint)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.config, self.checkpoint, self.replay_buffer)
        self.rewarder_worker = reward_model.RewardModel.options(num_cpus=0, num_gpus=num_predictor_gpus).remote(self.config)
        
        self.config.structure = network.dict_to_cpu(ray.get(self.rewarder_worker.inference.remote(self.config.sequence)))
        while self.config.structure == None:
            time.sleep(0.1)
        self.config.graph = network.structure_to_graph(self.config.structure)
        
        self.training_worker = trainer.Trainer.options(num_gpus=num_trainer_gpus).remote(self.config, self.checkpoint)
        self.reanalyse_worker = reanalyse.Reanalyse.options(num_gpus=gpu_per_worker if self.config.reanalyse_on_gpu else 0).remote(self.config, self.checkpoint)
        self.player_workers = [player.Player.options(num_gpus=gpu_per_worker if self.config.play_on_gpu else 0).remote(
            self.config, self.checkpoint, seed + self.config.seed) for seed in range(self.config.n_players)]

        print(f"Training losses and testing performance are logged in TensorBoard, with the logging file saved to: {os.path.abspath(args.output)}.")
        self.rewarder_worker._predict.remote(self.shared_storage_worker, self.manager_worker)
        [worker._play.remote(self.shared_storage_worker, self.replay_buffer_worker, self.manager_worker) for worker in self.player_workers]
        self.training_worker._train.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.reanalyse_worker._reanalyse.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.logging(gpu_per_worker if self.config.test_on_gpu else 0)
        
    def logging(self, num_gpus_per_worker):
        writer = SummaryWriter(args.output)
        print(f"View training results with: tensorboard --logdir {os.path.abspath(args.output)}.")
        self.test_worker = player.Player.options(num_gpus=num_gpus_per_worker).remote(self.config, self.checkpoint, self.config.seed)
        self.test_worker._play.remote(self.shared_storage_worker, None, self.manager_worker, test=True)

        keys = ["training_step", "total_reward", "mean_value", "max_reward", "episode_length", "num_played_games", "num_reanalysed_games"]
        
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        while info['training_step'] < 1:
            time.sleep(self.config.test_delay)
            info = ray.get(self.shared_storage_worker.get_info.remote(keys))

        test_step = 0
        try:
            while info["training_step"] < self.config.training_steps:
                writer.add_scalar('Testing worker/Total Reward', info['total_reward'], test_step)
                writer.add_scalar('Testing worker/Mean Value', info['mean_value'], test_step)
                writer.add_scalar('Testing worker/Episode Length', info['episode_length'], test_step)
                writer.add_scalar('Testing worker/Max Reward', info['max_reward'], test_step)
                
                writer.add_scalar('Self-play workers/Num Played Games', info['num_played_games'], test_step)
                writer.add_scalar('Self-play workers/Nnum Reanalysed Games', info['num_reanalysed_games'], test_step)
                
                test_step += 1
                if self.config.test_delay:
                    time.sleep(self.config.test_delay)
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        except KeyboardInterrupt:
            pass

        writer.close()
        self.terminate_workers()

    def terminate_workers(self):
        self.replay_buffer_worker.output_sequences.remote()

        self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())
        with open(os.path.join(args.output, 'replay_buffer.pkl'), 'wb') as f:
            pickle.dump(self.replay_buffer, f)

        self.manager_worker = None
        self.predictor_worker = None
        self.shared_storage_worker = None
        self.replay_buffer_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.player_workers = None
        self.test_worker = None

if __name__ == '__main__':
    start_time = timeit.default_timer()
    vir_evo = Evolution(args)
    vir_evo.evolve()
    ray.shutdown()
    end_time = timeit.default_timer()
    print(">>> RelaVDEP task completed successfully. Total execution time: {:.2f}s <<<".format(end_time - start_time))
