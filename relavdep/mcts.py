import torch
import copy
import math
import numpy as np
from network import *

class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    
    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, prior_p):
        self.prior_p = prior_p
        self.visit_count = 0
        self.value_sum = 0
        self.reward = 0
        self.hidden_state = None
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, actions, reward, logits, hidden_state):
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(torch.tensor([logits[0][a-1] for a in actions]), dim=0).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, prob in policy.items():
            self.children[action] = Node(prob)
    
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior_p = self.children[a].prior_p * (1 - frac) + n * frac

class MCTS:
    def __init__(self, config):
        self.config = config

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior_p

        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value())
        else:
            value_score = 0
        return prior_score + value_score
    
    def select_child(self, node, min_max_stats):
        max_ucb_score = max(self.ucb_score(node, child, min_max_stats) 
                            for _, child in node.children.items())
        best_action = np.random.choice([action for action, child in node.children.items() 
                                        if self.ucb_score(node, child, min_max_stats) == max_ucb_score])
        return best_action, node.children[best_action]
    
    def backpropagate(self, search_path, value, min_max_stats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())
            value = node.reward + self.config.discount * value
    
    def search(self, model, observation:list, legal_actions, action_history, tree_depth, add_exploration_noise=True):
        root = Node(0)
        hidden_state, reward, logits, value = model.initial_inference(observation)
        expanded_actions = self.get_legal_actions(legal_actions, action_history)
        root.expand(expanded_actions, reward, logits, hidden_state)
        if add_exploration_noise:
            root.add_exploration_noise(self.config.dirichlet_alpha, self.config.exploration_fraction)
        min_max_stats = MinMaxStats()
        
        for _ in range(self.config.n_sim):
            node = root
            search_path = [node]
            action_path = []
            curr_depth = copy.deepcopy(tree_depth)

            while curr_depth < self.config.max_mutations and node.expanded():
                action, node = self.select_child(node, min_max_stats)
                action_path.append(action)
                search_path.append(node)
                curr_depth += 1

            parent = search_path[-2]
            action = torch.tensor([action - 1]).unsqueeze(1)
            hidden_state, reward, logits, value = model.recurrent_inference(parent.hidden_state, action)
            reward = support_to_scalar(reward, self.config.support_size)
            value = support_to_scalar(value, self.config.support_size)
            expanded_actions = self.get_legal_actions(expanded_actions, action_path)
            node.expand(expanded_actions, reward.item(), logits, hidden_state)
            self.backpropagate(search_path, value.item(), min_max_stats)
        return root
    
    def get_legal_actions(self, legal_actions, action_path):
        mut_pos = [(a - 1) // 20 for a in action_path if a != 0]
        actions = [a for a in legal_actions if (a - 1) // 20 not in mut_pos]
        return actions
    