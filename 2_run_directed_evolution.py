import argparse
import copy
import os
import pickle
import random
import time
import timeit

import ray
import numpy as np
import torch
import wandb

from relavdep.modules import (
    config,
    environment,
    manager,
    shared_storage,
    replay_buffer,
    reanalyse,
    trainer,
    reward_model,
    player,
    network
)

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = "1.0"


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def print_section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

parser = argparse.ArgumentParser(description='Reinforcement Learning assisted Directed Evolution for Proteins')
parser.add_argument('--fasta', type=str, required=True, help='Protein sequence')
parser.add_argument('--rm_params', type=str, required=True, help='Supervised fine-tuned reward model parameters')
parser.add_argument('--constraint', type=str, default=None, help='Constraint file (.npz format)')

parser.add_argument('--rm_type', type=str, default='small', choices=['large', 'small'], help='Reward model type (default: %(default)s)')
parser.add_argument('--output', type=str, default='outputs', help='Output directory (default: %(default)s)')
parser.add_argument('--data_dir', type=str, default='models', help='Directory for model parameters (default: %(default)s)')
parser.add_argument('--temp_dir', type=str, default='/tmp/ray', help='Temporary directory for spilling object store (default: %(default)s)')
parser.add_argument('--n_layer', type=int, default=2, help='Number of downstream MLP layers (default: %(default)s)')
parser.add_argument('--max_mut', type=int, default=4, help='Maximum mutation counts (default: %(default)s)')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs (default: %(default)s)')
parser.add_argument('--n_player', type=int, default=6, help='Number of self-play workers (default: %(default)s)')
parser.add_argument('--n_sim', type=int, default=1200, help='Number of MCTS simulations (default: %(default)s)')
parser.add_argument('--train_delay', type=int, default=2, help='Training delay (default: %(default)s)')
parser.add_argument('--log_interval', type=int, default=50, help='Console logging interval in test iterations (default: %(default)s)')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training (default: %(default)s)')
parser.add_argument('--training_steps', type=int, default=20000, help='Total training steps (default: %(default)s)')
parser.add_argument('--warmup_steps', type=int, default=0, help='Learning rate warmup steps (default: disabled)')
parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Learning rate decay rate (default: %(default)s)')
parser.add_argument('--lr_decay_steps', type=int, default=1000, help='Learning rate decay steps (default: %(default)s)')
parser.add_argument('--checkpoint_interval', type=int, default=200, help='Checkpoint interval for training (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)s)')
parser.add_argument('--save_buffer', action='store_true', help='Save replay buffer to output directory (default: disabled)')
parser.add_argument('--wandb_project', type=str, default='RelaVDEP', help='Weights & Biases project name (default: %(default)s)')
parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'], help='Weights & Biases run mode (default: %(default)s)')
args = parser.parse_args()

assert os.path.exists(args.fasta), "!!! Fasta file does not exist !!!"
assert os.path.exists(args.data_dir), "!!! Params directory does not exist !!!"
assert args.n_gpus <= torch.cuda.device_count(), "!!! Insufficient number of available GPUs !!!"
os.makedirs(args.output, exist_ok=True)
if args.constraint:
    assert os.path.exists(args.constraint), "!!! Constraint file does not exist !!!"

class VirtualDE:
    def __init__(self, args):
        print_section("Stage 1: Initialize Directed Evolution")
        self.config = config.Config(args)
        self.config.avaliables = environment.Environment(self.config, 'cpu').legal_actions()
        self.config.pb_c_init = 1.25 * (len(self.config.avaliables) // 19 + 1)

        total_actions = self.config.action_space_size
        legal_actions = len(self.config.avaliables)
        constrained_actions = len(self.config.legal) + len(self.config.illegal)
        print(f"Task name: {self.config.task_name}")
        print(f"Sequence length: {self.config.length}")
        print(f"Output directory: {os.path.abspath(args.output)}")
        print(f"Reward model parameters: {os.path.abspath(args.rm_params)}")
        print(f"Constraint file: {os.path.abspath(args.constraint) if args.constraint else 'None'}")
        print(f"Action space: {legal_actions}/{total_actions} legal actions after filtering")
        print(f"Max training steps: {self.config.training_steps}")
        print(f"Learning rate warmup steps: {self.config.warmup_steps}")
        print(f"Learning rate decay: rate={self.config.lr_decay_rate}, steps={self.config.lr_decay_steps}")
        print(f"MCTS simulations per move: {self.config.n_sim}")
        print(f"Self-play workers: {self.config.n_player}")
        print(f"Max mutations per episode: {self.config.max_mutations}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Save replay buffer: {args.save_buffer}")
        print(f"Seed: {self.config.seed}")

        initial_weight = network.Network(self.config).get_weights()
        self.checkpoint = {
            "weights": copy.deepcopy(initial_weight), "optimizer": None,
            "training_step": 0, "learning_rate": 0, "total_loss": 0,
            "policy_loss": 0, "value_loss": 0, "reward_loss": 0,
            "total_reward": 0, "mean_value": 0, "episode_length": 0,
            "max_reward": 0, "num_played_games": 0, "num_reanalysed_games": 0
        }

        self.replay_buffer = {}

        self.temp_dir = os.path.join(args.temp_dir, self.config.task_name)
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Ray temp directory: {os.path.abspath(self.temp_dir)}")
        ray.init(log_to_driver=False, _temp_dir=os.path.abspath(self.temp_dir), num_gpus=self.config.n_gpus)
        print(f"Ray initialized with {self.config.n_gpus} GPU(s).")

    def evolve(self):
        print_section("Stage 2: Start Distributed Workers")
        gpu_per_worker = self.config.n_gpus / (
            3 * self.config.train_on_gpu +
            5 * self.config.predict_on_gpu +
            self.config.reanalyse_on_gpu +
            self.config.test_on_gpu +
            self.config.n_player * self.config.play_on_gpu
        )

        if self.config.predict_on_gpu:
            num_predictor_gpus = 5 * gpu_per_worker if 5 * gpu_per_worker < 1 else 1
        else:
            num_predictor_gpus = 0

        if self.config.train_on_gpu:
            num_trainer_gpus = 3 * gpu_per_worker if 3 * gpu_per_worker < 1 else 1
        else:
            num_trainer_gpus = 0

        print("GPU allocation:")
        print(f"  predictor worker: {num_predictor_gpus:.3f}")
        print(f"  trainer worker: {num_trainer_gpus:.3f}")
        print(f"  reanalyse worker: {(gpu_per_worker if self.config.reanalyse_on_gpu else 0):.3f}")
        print(f"  each self-play worker: {(gpu_per_worker if self.config.play_on_gpu else 0):.3f}")
        print(f"  test worker: {(gpu_per_worker if self.config.test_on_gpu else 0):.3f}")

        # initialize workers
        print("Initializing manager, shared storage, replay buffer, and reward predictor...")
        self.manager_worker = manager.Manager.remote()
        self.shared_storage_worker = shared_storage.SharedStorage.remote(self.config, self.checkpoint)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.config, self.checkpoint, self.replay_buffer)
        self.predictor_worker = reward_model.RewardModel.options(num_cpus=0, num_gpus=num_predictor_gpus).remote(self.config)

        structure_start = timeit.default_timer()
        print("Predicting wild-type structure for graph construction...")
        self.config.structure = network.dict_to_cpu(ray.get(self.predictor_worker.inference.remote(self.config.sequence)))
        while self.config.structure == None:
            time.sleep(0.1)
        self.config.graph = network.structure_to_graph(self.config.structure)
        structure_duration = timeit.default_timer() - structure_start
        print(f"Structure prediction completed in {format_duration(structure_duration)}.")

        print("Initializing trainer, reanalyse worker, and self-play workers...")
        self.training_worker = trainer.Trainer.options(num_gpus=num_trainer_gpus).remote(self.config, self.checkpoint)
        self.reanalyse_worker = reanalyse.Reanalyse.options(num_gpus=gpu_per_worker if self.config.reanalyse_on_gpu else 0).remote(self.config, self.checkpoint)
        self.player_workers = [player.Player.options(num_gpus=gpu_per_worker if self.config.play_on_gpu else 0).remote(
            self.config, self.checkpoint, self.config.seed + seed, seed) for seed in range(self.config.n_player)
        ]

        print(f"Training losses and testing performance are logged to Weights & Biases, with local files saved to: {os.path.abspath(args.output)}.")
        print("Launching predictor, self-play, training, and reanalyse loops...")
        self.predictor_worker._predict.remote(self.shared_storage_worker, self.manager_worker)
        [worker._play.remote(self.shared_storage_worker, self.replay_buffer_worker, self.manager_worker) for worker in self.player_workers]
        self.training_worker._train.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.reanalyse_worker._reanalyse.remote(self.shared_storage_worker, self.replay_buffer_worker)
        self.logging(gpu_per_worker if self.config.test_on_gpu else 0)

    def logging(self, num_gpus_per_worker):
        print_section("Stage 3: Monitor Training Progress")
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": self.config.task_name,
            "dir": args.output,
            "config": {
                **vars(args),
                "task_name": self.config.task_name,
                "sequence_length": self.config.length,
                "action_space_size": self.config.action_space_size,
                "legal_action_count": len(self.config.avaliables),
                "training_steps": self.config.training_steps,
                "checkpoint_interval": self.config.checkpoint_interval,
                "test_delay": self.config.test_delay,
            },
        }
        if args.wandb_mode:
            wandb_kwargs["mode"] = args.wandb_mode
        wandb.init(**wandb_kwargs)
        wandb.define_metric("training_step")
        wandb.define_metric("*", step_metric="training_step")
        print(f"Weights & Biases mode: {args.wandb_mode}")
        print("View training results in Weights & Biases.")
        print("Starting test worker for periodic policy evaluation...")
        self.test_worker = player.Player.options(num_gpus=num_gpus_per_worker).remote(self.config, self.checkpoint, self.config.seed)
        self.test_worker._play.remote(self.shared_storage_worker, None, self.manager_worker, test=True)

        keys = ["training_step", "total_reward", "mean_value", "max_reward",
                "episode_length", "num_played_games", "num_reanalysed_games",
                "learning_rate", "total_loss", "policy_loss", "value_loss", "reward_loss"]

        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        wait_start = timeit.default_timer()
        print("Waiting for trainer to finish the first optimization step...")
        while info['training_step'] < 1:
            time.sleep(self.config.test_delay)
            info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        print(f"First training step received after {format_duration(timeit.default_timer() - wait_start)}.")
        print(
            f"{'step':>8} | {'progress':>8} | {'reward':>10} | {'max':>10} | "
            f"{'games':>8} | {'loss':>10} | {'lr':>10} | {'elapsed':>10}"
        )

        test_step = 0
        monitor_start = timeit.default_timer()

        try:
            while info["training_step"] < self.config.training_steps:
                elapsed = timeit.default_timer() - monitor_start
                progress = info['training_step'] / self.config.training_steps
                wandb.log({
                    "Testing worker/Total Reward": info['total_reward'],
                    "Testing worker/Mean Value": info['mean_value'],
                    "Testing worker/Episode Length": info['episode_length'],
                    "Testing worker/Max Reward": info['max_reward'],
                    "Self-play workers/Num Played Games": info['num_played_games'],
                    "Self-play workers/Num Reanalysed Games": info['num_reanalysed_games'],
                    "Training worker/Learning Rate": info['learning_rate'],
                    "Training worker/Total Loss": info['total_loss'],
                    "Training worker/Policy Loss": info['policy_loss'],
                    "Training worker/Value Loss": info['value_loss'],
                    "Training worker/Reward Loss": info['reward_loss'],
                    "Progress/Completion": progress,
                    "Progress/Elapsed Seconds": elapsed,
                    "training_step": info['training_step'],
                    "test_step": test_step,
                }, step=info['training_step'])

                if test_step % max(args.log_interval, 1) == 0:
                    print(
                        f"{info['training_step']:8d} | "
                        f"{progress:7.1%} | "
                        f"{info['total_reward']:10.4f} | "
                        f"{info['max_reward']:10.4f} | "
                        f"{info['num_played_games']:8d} | "
                        f"{info['total_loss']:10.4f} | "
                        f"{info['learning_rate']:10.2e} | "
                        f"{format_duration(elapsed):>10}"
                    )

                test_step += 1
                if self.config.test_delay:
                    time.sleep(self.config.test_delay)
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Shutting down workers...")
        finally:
            final_elapsed = timeit.default_timer() - monitor_start
            wandb.log({
                "Progress/Final Training Step": info["training_step"],
                "Progress/Final Completion": min(info["training_step"] / self.config.training_steps, 1.0),
                "Progress/Total Monitor Seconds": final_elapsed,
            })
            print(f"Monitoring completed at step {info['training_step']} after {format_duration(final_elapsed)}.")
            wandb.finish()
            self.terminate_workers()

    def terminate_workers(self):
        print_section("Stage 4: Save Outputs and Terminate Workers")
        print("Writing final mutant sequences...")
        self.replay_buffer_worker.output_sequences.remote()

        if args.save_buffer:
            print("Saving replay buffer...")
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())
            replay_buffer_path = os.path.join(args.output, 'replay_buffer.pkl')
            with open(replay_buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)
            print(f"Replay buffer saved to: {os.path.abspath(replay_buffer_path)}")
            print(f"Replay buffer games: {len(self.replay_buffer)}")
        else:
            print("Skipping replay buffer save. Use --save_buffer to enable it.")

        self.manager_worker = None
        self.predictor_worker = None
        self.shared_storage_worker = None
        self.replay_buffer_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.player_workers = None
        self.test_worker = None
        print("Workers released.")

if __name__ == '__main__':
    start_time = timeit.default_timer()

    print_section("RelaVDEP Directed Evolution")
    print("Setting random seeds...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    evolver = VirtualDE(args)
    evolver.evolve()
    ray.shutdown()
    end_time = timeit.default_timer()
    print(">>> RelaVDEP task completed successfully. Total execution time: {} <<<".format(format_duration(end_time - start_time)))
