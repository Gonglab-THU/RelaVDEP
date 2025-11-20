import esm
import torch
import torch.nn as nn
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_path, '../../..'))
from scripts.SPIRED_Fitness.models import PretrainEncoder, SPIRED_Model, GAT

def dict_to_device(dict_input, device):
    dict_output = {}
    for key, value in dict_input.items():
        if isinstance(value, torch.Tensor):
            dict_output[key] = value.to(device)
        elif isinstance(value, dict):
            dict_output[key] = dict_to_device(value, device)
        else:
            dict_output[key] = value
    return dict_output


"""
Adapted from https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/scripts/model.py
"""
class PretrainModel(nn.Module):
    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()
        self.pretrain_encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
    
    def forward(self, data):
        single_feat, _ = self.pretrain_encoder(data['embedding'], data['pair'], data['plddt'])
        return single_feat


class DownStreamModel(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        self.esm2_transform = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 640),
            nn.LeakyReLU(),
            nn.Linear(640, 320),
            nn.LeakyReLU(),
            nn.Linear(320, 128)
        )
        if n_layer == 1:
            self.read_out = nn.Linear(128 + 32, 1)
        elif n_layer == 2:
            self.read_out = nn.Sequential(
                nn.Linear(128 + 32, 80),
                nn.LeakyReLU(),
                nn.Linear(80, 1)
            )
        elif n_layer == 3:
            self.read_out = nn.Sequential(
                nn.Linear(128 + 32, 80),
                nn.LeakyReLU(),
                nn.Linear(80, 40),
                nn.LeakyReLU(),
                nn.Linear(40, 1)
            )
        elif n_layer == 4:
            self.read_out = nn.Sequential(
                nn.Linear(128 + 32, 120),
                nn.LeakyReLU(),
                nn.Linear(120, 80),
                nn.LeakyReLU(),
                nn.Linear(80, 40),
                nn.LeakyReLU(),
                nn.Linear(40, 1)
            )
        elif n_layer == 5:
            self.read_out = nn.Sequential(
                nn.Linear(128 + 32, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 96),
                nn.LeakyReLU(),
                nn.Linear(96, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            )
    
    def forward(self, pretrained_embedding, esm2_embedding):
        x = torch.cat((self.esm2_transform(esm2_embedding), pretrained_embedding), dim = -1)
        x = self.read_out(x)
        return x

##### Base Model #####

class BaseModel:
    def __init__(self, data_dir, device):
        # esm2-650M model
        self.esm2_650M, _ = esm.pretrained.load_model_and_alphabet(os.path.join(data_dir, 'esm2_t33_650M_UR50D.pt'))
        self.esm2_650M.eval().to(device)

        # esm2-3B model
        self.esm2_3B, self.esm2_alphabet = esm.pretrained.load_model_and_alphabet(os.path.join(data_dir, 'esm2_t36_3B_UR50D.pt'))
        self.esm2_3B.eval().to(device)
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()

        # SPIRED model
        self.SPIRED_model = SPIRED_Model(depth=2, channel=128, device_list=[device])
        SPIRED_model_dict = self.SPIRED_model.state_dict().copy()
        SPIRED_best_dict = torch.load(os.path.join(data_dir, 'SPIRED-Fitness.pth'))
        pretrain_params = {k.split('SPIRED.')[-1]: v for k, v in SPIRED_best_dict.copy().items() if k.startswith('SPIRED')}
        SPIRED_model_dict.update(pretrain_params)
        self.SPIRED_model.load_state_dict(SPIRED_model_dict)
        self.SPIRED_model.eval().to(device)
    
    def inference(self, seq):
        with torch.no_grad():
            _, _, target_tokens = self.esm2_batch_converter([('', seq)])
            # esm2-650M embedding
            tmp = self.esm2_650M(target_tokens.to(next(self.esm2_650M.parameters()).device), 
                                 repr_layers = [33], return_contacts = False)
            esm2_650M_embedding = tmp['representations'][33][:, 1:-1, :]
            # esm2-3B embedding
            tmp = self.esm2_3B(target_tokens.to(next(self.esm2_3B.parameters()).device), 
                               repr_layers = range(37), need_head_weights = False, return_contacts = False)
            esm2_3B_embedding = torch.stack([v[:, 1:-1, :] for _, v in sorted(tmp['representations'].items())], dim = 2)
            preds = self.SPIRED_model(target_tokens[:, 1:-1], esm2_3B_embedding, no_recycles = 1)
        data = {'tokens':target_tokens[:, 1:-1], 'embedding': esm2_650M_embedding, 
                'pair': preds[0]['4th'][-1].permute(0, 2, 3, 1).contiguous(), 'plddt': preds[2]['4th'][-1]}
        return data

##### Fitness Model #####

class FitnessModel(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        self.Fitness = PretrainModel(node_dim = 32, num_layer = 2, n_head = 8, pair_dim = 32)
        self.down_stream_model = DownStreamModel(n_layer)
        self.finetune_coef = nn.Parameter(torch.tensor([0.01], requires_grad = True))
        self.finetune_cons = nn.Parameter(torch.tensor([0.0], requires_grad = True))
    
    def forward(self, wt_data, mut_data):
        wt_pretrained_embedding = self.Fitness({'embedding': wt_data['embedding'], 
                                                'pair': wt_data['pair'], 
                                                'plddt': wt_data['plddt']})
        
        mut_pretrained_embedding = self.Fitness({'embedding': mut_data['embedding'], 
                                                 'pair': mut_data['pair'], 
                                                 'plddt': mut_data['plddt']})
        
        mut_pos = (wt_data['tokens'] != mut_data['tokens']).int()
        
        wt_value = self.down_stream_model(wt_pretrained_embedding, wt_data['embedding'])
        mut_value = self.down_stream_model(mut_pretrained_embedding, mut_data['embedding'])
        
        delta_value = torch.sum((mut_value - wt_value).squeeze(-1) * mut_pos, dim=1)
        return self.finetune_coef * delta_value + self.finetune_cons

##### Stability Model (optional) #####

class StabilityModel(nn.Module):
    def __init__(self, n_layer, node_dim = 32, num_layer = 3, n_head = 8, pair_dim = 64):
        super().__init__()
        self.esm2_transform = nn.Sequential(nn.LayerNorm(1280),
                                            nn.Linear(1280, 640),
                                            nn.LeakyReLU(),
                                            nn.Linear(640, 320),
                                            nn.LeakyReLU(),
                                            nn.Linear(320, node_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(node_dim, node_dim))
        self.pair_encoder = nn.Linear(3, pair_dim)
        self.blocks = nn.ModuleList([GAT(node_dim, n_head, pair_dim) for _ in range(num_layer)])
        self.down_stream_model = DownStreamModel(n_layer)
        self.finetune_coef = nn.Parameter(torch.tensor([0.01], requires_grad = True))
        self.finetune_cons = nn.Parameter(torch.tensor([0.0], requires_grad = True))
    
    def forward(self, wt_data, mut_data):
        wt_embedding = self.esm2_transform(wt_data['embedding'])
        mut_embedding = self.esm2_transform(mut_data['embedding'])
                
        wt_pair = self.pair_encoder(wt_data['pair'])
        mut_pair = self.pair_encoder(mut_data['pair'])
        
        for block in self.blocks:
            wt_embedding, wt_pair = block(wt_embedding, wt_pair, wt_data['plddt'].unsqueeze(-1))
            mut_embedding, mut_pair = block(mut_embedding, mut_pair, mut_data['plddt'].unsqueeze(-1))

        mut_pos = (wt_data['tokens'] != mut_data['tokens']).int()

        wt_value = self.down_stream_model(wt_embedding, wt_data['embedding'])
        mut_value = self.down_stream_model(mut_embedding, mut_data['embedding'])

        delta_value = torch.sum((mut_value - wt_value).squeeze(-1) * mut_pos, dim=1)
        return self.finetune_coef * delta_value + self.finetune_cons
