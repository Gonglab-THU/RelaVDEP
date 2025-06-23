import os
import esm
import copy
import torch
from utils import *

class Environment:
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def init_model(self):
        model_path = os.path.join(self.config.data_dir, 'esm2_t33_650M_UR50D.pt')
        self.esm_model, self.esm2_alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.esm_batch_converter = self.esm2_alphabet.get_batch_converter()
        self.esm_model.eval()
        self.esm_model.to(self.device)

    def reset(self):
        self.mut_count = 0
        self.rewards = []
        self.curr_seq = copy.deepcopy(self.config.sequence)
        return self.get_observation(self.curr_seq)
    
    def step(self, action):
        self.last_seq = copy.deepcopy(self.curr_seq)
        self.mut_count += 1
        self.curr_seq = mutation(self.curr_seq, action)
        return self.get_observation(self.curr_seq), self.mut_done()

    def get_reward(self, prediction):
        reward = prediction
        self.rewards.append(reward)
        return reward
    
    def mut_done(self):
        if self.mut_count >= self.config.max_mutations:
            return True
        return False
    
    def get_observation(self, curr_seq):
        graph = self.config.graph.clone()
        graph.node_s = self.get_embedding(curr_seq)
        legal_onehot, avaliable_pos = self.legal_onehot(curr_seq)
        graph.legal_onehot = legal_onehot
        graph.avaliable_pos = avaliable_pos
        graph.seq = curr_seq
        return graph
    
    def get_embedding(self, seq):
        with torch.no_grad():
            _, _, target_tokens = self.esm_batch_converter([('', seq)])
            tmp = self.esm_model(target_tokens.to(self.device),
                                 repr_layers=[33], 
                                 return_contacts=False)
            embedding = tmp['representations'][33][:, 1:-1, :]
        return embedding.squeeze().clone().detach().cpu()
    
    def legal_onehot(self, curr_seq):
        legal_onehot = np.zeros((self.config.length, 20))
        avaliable_pos = np.ones(self.config.length)
        for a in self.legal_actions():
            legal_onehot[(a - 1) // 20][(a - 1) % 20] = 1
        mut_pos = [i for i in range(len(curr_seq)) if curr_seq[i] != self.config.sequence[i]]
        if mut_pos:
            legal_onehot[mut_pos] = 0
            avaliable_pos[mut_pos] = 0
        return torch.tensor(legal_onehot, dtype=torch.float32), torch.tensor(avaliable_pos, dtype=torch.float32)
    
    def legal_actions(self):
        legal_action_space = copy.deepcopy(self.config.action_space)
        wt_pos, wt_res = np.where(seq2onehot(self.config.sequence) == 1)
        for action in list(wt_pos * 20 + wt_res + 1):
            legal_action_space.remove(action)
        if self.config.illegal:
            for action in self.config.illegal:
                if action in legal_action_space:
                    legal_action_space.remove(action)
        if self.config.legal:
            legal_action_space = [action for action in legal_action_space if action in self.config.legal]
        return legal_action_space