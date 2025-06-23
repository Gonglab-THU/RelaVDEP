import os
import ray
import copy
import numpy as np
import pandas as pd

@ray.remote
class ReplayBuffer:
    def __init__(self, config, initial_checkpoint, initial_buffer):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.total_samples = sum([len(play_history.root_values) for 
                                  play_history in self.buffer.values()])
        np.random.seed(self.config.seed)

    def save_play(self, play_history, shared_storage=None):
        all_trajectories = [value.action_history for _, value in self.buffer.items()] if self.buffer else []
        if play_history.action_history not in all_trajectories:
            priorities = []
            for idx, root_value in enumerate(play_history.root_values):
                priority = np.abs(root_value - self.computer_target_value(play_history, idx)) ** self.config.prob_alpha
                priorities.append(priority)
            
            play_history.priorities = np.array(priorities, dtype='float32')
            play_history.play_priority = np.max(play_history.priorities)

            self.buffer[self.num_played_games] = play_history
            self.num_played_games += 1
            self.total_samples += len(play_history.root_values)

            if len(self.buffer) > self.config.buffer_size:
                del_idx = self.num_played_games - len(self.buffer)
                self.total_samples -= len(self.buffer[del_idx].root_values)
                del self.buffer[del_idx]

            if shared_storage:
                shared_storage.set_info.remote("num_played_games", self.num_played_games)
    
    def sample_batch(self):
        idx_batch, obs_batch, act_batch, reward_batch, value_batch, policy_batch, grad_batch = [], [], [], [], [], [], []
        weight_batch = []

        for play_idx, play_history, play_prob in self.sample_plays(self.config.batch_size):
            position_idx, position_prob = self.sample_position(play_history)
            values, rewards, policies, actions = self.make_target(play_history, position_idx)
            
            idx_batch.append([play_idx, position_idx])
            obs_batch.append(play_history.observation_history[position_idx])
            act_batch.append(actions)
            reward_batch.append(rewards)
            value_batch.append(values)
            policy_batch.append(policies)
            grad_batch.append(len(actions) * [min(self.config.num_unroll_steps, len(play_history.action_history) - position_idx)])
            weight_batch.append(1 / (self.total_samples * play_prob * position_prob))
        
        weight_batch = np.array(weight_batch, dtype='float32') / max(weight_batch)
        return idx_batch, obs_batch, act_batch, reward_batch, value_batch, policy_batch, grad_batch, weight_batch
    
    def sample_reanalysed_play(self):
        idx = np.random.choice(list(self.buffer.keys()))
        play = self.buffer[idx]
        return idx, play

    def sample_plays(self, n_plays):
        play_idx_list, play_probs = [], []
        for play_idx, play_history in self.buffer.items():
            play_idx_list.append(play_idx)
            play_probs.append(play_history.play_priority)
        play_probs = np.array(play_probs, dtype='float32')
        play_probs /= np.sum(play_probs)
        play_probs_dict = {k: v for k, v in zip(play_idx_list, play_probs)}
        selected = np.random.choice(play_idx_list, n_plays, p=play_probs)
        selected_plays = [(idx, self.buffer[idx], play_probs_dict.get(idx)) for idx in selected]
        return selected_plays
    
    def sample_position(self, play_history):
        position_prob = None
        position_probs = play_history.priorities / sum(play_history.priorities)
        position_index = np.random.choice(len(position_probs), p=position_probs)
        position_prob = position_probs[position_index]
        return position_index, position_prob
    
    def make_target(self, play_history, position_idx):
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for idx in range(position_idx, position_idx + self.config.num_unroll_steps + 1):
            value = self.computer_target_value(play_history, idx)
            if idx < len(play_history.root_values):
                target_values.append(value)
                target_rewards.append(play_history.reward_history[idx])
                target_policies.append(play_history.child_visits[idx])
                actions.append(play_history.action_history[idx] - 1)
            elif idx == len(play_history.root_values):
                target_values.append(0)
                target_rewards.append(play_history.reward_history[idx])
                target_policies.append([1 / len(play_history.child_visits[0]) 
                                        for _ in range(len(play_history.child_visits[0]))])
                actions.append(play_history.action_history[idx] - 1)
            else:
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([1 / len(play_history.child_visits[0]) 
                                        for _ in range(len(play_history.child_visits[0]))])
                actions.append(np.random.choice(self.config.avaliables) - 1)
        return target_values, target_rewards, target_policies, actions
    
    def computer_target_value(self, play_history, idx):
        bootstrap_index = idx + self.config.td_steps
        if bootstrap_index < len(play_history.root_values):
            if play_history.reanalysed_values is None:
                root_values = play_history.root_values
            else:
                root_values = play_history.reanalysed_values

            last_step_value = root_values[bootstrap_index]
            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(play_history.reward_history[idx + 1: bootstrap_index + 1]):
            value += reward * (self.config.discount ** i)
        return value
    
    def update_priorities(self, priorities, index_batch):
        for i in range(len(index_batch)):
            play_idx, position_index = index_batch[i]
            
            if next(iter(self.buffer)) <= play_idx:
                priority = priorities[i, :]
                start_idx = position_index
                end_idx = min(position_index + len(priority), len(self.buffer[play_idx].priorities))
                self.buffer[play_idx].priorities[start_idx : end_idx] = priority[:end_idx-start_idx]
                self.buffer[play_idx].play_priority = np.max(self.buffer[play_idx].priorities)

    def update_play_history(self, play_idx, play_history):
        if next(iter(self.buffer)) <= play_idx:
            play_history.priorities = np.copy(play_history.priorities)
            self.buffer[play_idx] = play_history
            
    def output_sequences(self):
        [sequences, fitness, mutants] = [[] for _ in range(3)]
        for _, value in self.buffer.items():
            for i in range(len(value.action_history) - 1):
                sequences.append(value.observation_history[i+1].seq)
                fitness.append(value.reward_history[i+1])
                mutants.append(self.get_mutation(self.config.sequence, value.observation_history[i+1].seq))
        
        sequence_data = pd.DataFrame({"mutant": mutants, "sequence": sequences, "fitness": fitness})
        sequence_data = sequence_data.sort_values(by="fitness", ascending=False).drop_duplicates(subset="sequence")
        sequence_data.to_csv(os.path.join(self.config.output_path, 'mutants.csv'), index=False)
    
    def get_mutation(self, wt_seq, mut_seq):
        mutation = []
        for i in range(len(wt_seq)):
            if wt_seq[i] != mut_seq[i]:
                mutation.append(wt_seq[i] + str(i+1) + mut_seq[i])
        mutation = ':'.join(mutation)
        return mutation

    def size(self):
        return len(self.buffer)
    
    def get_buffer(self):
        return self.buffer
    