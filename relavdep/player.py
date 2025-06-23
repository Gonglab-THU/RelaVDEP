import os
import ray
import torch
import mcts
import time
import numpy as np
from network import Network
from environment import Environment

class PlayHistory:
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.priorities = None
        self.play_priority = None
        self.reanalysed_values = None

    def store_search_stats(self, root, action_space):
        if root is not None:
            sum_visits = sum([child.visit_count for child in root.children.values()])
            self.child_visits.append([root.children[a].visit_count / sum_visits 
                                      if a in root.children else 0 for a in action_space])
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

@ray.remote
class Player:
    def __init__(self, config, initial_checkpoint, seed):
        self.config = config

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = Network(config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device('cuda' if config.play_on_gpu else 'cpu'))
        self.model.eval()
        
        self.mut_env = Environment(config, torch.device('cuda' if config.play_on_gpu else 'cpu'))
        self.mut_env.init_model()

    def _play(self, shared_storage, replay_buffer, manager, test=False):
        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps:
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            if not test:
                trained_steps = ray.get(shared_storage.get_info.remote("training_step"))
                temperature = self.config.visit_softmax_temperature_fn(trained_steps)
                play_history = self.play_game(manager, temperature, self.config.temp_threshold, True)
                replay_buffer.save_play.remote(play_history, shared_storage)
            elif ray.get(shared_storage.get_info.remote("training_step")) >= 1:
                max_reward = ray.get(shared_storage.get_info.remote("max_reward"))
                all_length, all_values, all_rewards = [], [], []
                for _ in range(5):
                    play_history = self.play_game(manager, 0.1, self.config.temp_threshold, False)
                    values = [value for value in play_history.root_values if value]
                    all_length.append(len(play_history.action_history) - 1)
                    all_values.append(np.mean(values) if values else 0)
                    all_rewards.append(play_history.reward_history)
                shared_storage.set_info.remote({"episode_length": np.mean(all_length), "mean_value": np.mean(all_values), 
                                                "total_reward": np.mean([sum(rewards) for rewards in all_rewards])})
                if np.mean([max(rewards) for rewards in all_rewards]) > max_reward:
                    max_reward = np.mean([max(rewards) for rewards in all_rewards])
                    shared_storage.set_info.remote({"max_reward": max_reward})
                    shared_storage.save_checkpoint.remote(os.path.join(self.config.output_path, 'best.pth'))
            else:
                continue

    def play_game(self, manager, temperature, temp_threshold, add_exploration_noise):
        play_history = PlayHistory()
        observation = self.mut_env.reset()
        play_history.action_history.append(0)
        play_history.observation_history.append(observation)
        play_history.reward_history.append(0)
        done = False
        
        with torch.no_grad():
            while not done and len(play_history.action_history) <= self.config.max_mutations:
                root = mcts.MCTS(self.config).search(self.model, [observation], self.config.avaliables, 
                                                     play_history.action_history, self.mut_env.mut_count,
                                                     add_exploration_noise)
                action = self.select_action(root, temperature if not temp_threshold 
                                            or len(play_history.action_history) < temp_threshold else 0)
                observation, done = self.mut_env.step(action)
                
                task = self.get_mut(self.config.sequence, self.mut_env.curr_seq)
                if task not in ray.get(manager.get_results_keys.remote()):
                    manager.add_task.remote(task, self.config.sequence, self.mut_env.curr_seq)
                    while task not in ray.get(manager.get_results_keys.remote()):
                        time.sleep(self.config.play_delay)
                prediction = ray.get(manager.get_result_item.remote(task))
                reward = self.mut_env.get_reward(prediction)
                
                play_history.store_search_stats(root, self.config.action_space)
                play_history.action_history.append(action)
                play_history.observation_history.append(observation)
                play_history.reward_history.append(reward)
                torch.cuda.empty_cache()
        return play_history

    @staticmethod
    def select_action(node, temperature):
        visit_counts = np.array([child.visit_count for child in node.children.values()], dtype='int32')
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action
    
    @staticmethod
    def get_mut(wt_seq, mut_seq):
        mut_info = [wt_seq[i] + f'{i+1}' + mut_seq[i] for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
        return ':'.join(mut_info)
