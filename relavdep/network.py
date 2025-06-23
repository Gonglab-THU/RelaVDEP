import numpy as np
import torch
import torch_cluster
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as GeoData
from torch_geometric.data import Batch
from scripts.gvp import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_sum

class RepresentationNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.esm_transform = nn.Sequential(nn.LayerNorm(1280),
                                           nn.Linear(1280, 512),
                                           nn.LeakyReLU(),
                                           nn.Linear(512, 256),
                                           nn.LeakyReLU(),
                                           nn.Linear(256, config.hidden_dim))
        
        self.representation_network = nn.Sequential(nn.Conv1d(config.hidden_dim + 20, config.hidden_dim, 3, 1, 1),
                                                    nn.ReLU(), 
                                                    nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, 1, 1),
                                                    nn.ReLU())
        
    def forward(self, observation):
        observation = Batch.from_data_list(observation).to(next(self.representation_network.parameters()).device)
        esm_embedding = self.esm_transform(observation.node_s)
        node_scalar = torch.cat([esm_embedding, observation.legal_onehot], dim=-1)
        node_scalar = node_scalar.view(len(observation), -1, self.config.hidden_dim + 20).permute(0, 2, 1)
        node_scalar = self.representation_network(node_scalar).permute(0, 2, 1)
        observation.node_s = node_scalar.reshape(-1, self.config.hidden_dim)
        observation.legal_onehot = None
        return observation

class PredictionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_size = 2 * config.support_size + 1
        self.gvp_node_p = nn.Sequential(LayerNorm(config.node_in_dim), 
                                        GVP(config.node_in_dim, config.node_h_dim, activations=(None, None)))

        self.gvp_edge_p = nn.Sequential(LayerNorm(config.edge_in_dim), 
                                        GVP(config.edge_in_dim, config.edge_h_dim, activations=(None, None)))

        self.prediction_layers = nn.ModuleList(GVPConvLayer(config.node_h_dim, config.edge_h_dim, drop_rate=config.dropout) 
                                               for _ in range(config.n_layers))
        
        self.prediction_policy_network = nn.Sequential(LayerNorm(config.node_h_dim), 
                                                       GVP(config.node_h_dim, (config.node_out_dim, 0), activations=(None, None)))
        
        self.prediction_policy_output = nn.Sequential(nn.Linear(config.length * config.node_out_dim, config.length * 20),
                                                      nn.ReLU(),
                                                      nn.Linear(config.length * 20, config.length * 20))
        
        self.prediction_value_network = nn.Sequential(LayerNorm(config.node_h_dim), 
                                                      GVP(config.node_h_dim, (config.node_out_dim, 0), activations=(None, None)))
        
        self.prediction_value_output = nn.Sequential(nn.Linear(config.node_out_dim, config.node_out_dim // 2), 
                                                     nn.LeakyReLU(),
                                                     nn.Linear(config.node_out_dim // 2, self.output_size))

        self.prediction_value_output[-1].weight.data.fill_(0)
        self.prediction_value_output[-1].bias.data.fill_(0)

    def forward(self, hidden_state):
        batch_size = len(hidden_state.ptr) - 1
        nodes = (hidden_state.node_s, hidden_state.node_v)
        edges = (hidden_state.edge_s, hidden_state.edge_v)
        
        global_nodes = self.gvp_node_p(nodes)
        global_edges = self.gvp_edge_p(edges)
        for layer in self.prediction_layers:
            global_nodes = layer(global_nodes, hidden_state.edge_index, global_edges)
        
        # policy
        policy_output = self.prediction_policy_network(global_nodes)                            # [N*L, node_out_dim]
        policy_output = policy_output.reshape(batch_size, -1, self.config.node_out_dim)         # [N, L, node_out_dim]
        policy_output = policy_output * hidden_state.avaliable_pos.reshape(batch_size, -1, 1)   # [N, L, 1]
        policy_output = self.prediction_policy_output(policy_output.view(batch_size, -1))       # [N, L*20]
        
        # value
        value_output = self.prediction_value_network(global_nodes)                              # [N*L, node_out_dim]
        value_output = scatter_sum(value_output, hidden_state.batch, dim=0)                     # [N, node_out_dim]
        value_output = self.prediction_value_output(value_output)                               # [N, output_size]
        return policy_output, value_output

class DynamicsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_size = 2 * config.support_size + 1
        self.dynamics_state_network = nn.Sequential(nn.Conv1d(config.hidden_dim + 20, config.hidden_dim, 3, 1, 1),
                                                    nn.ReLU(), 
                                                    nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, 1, 1),
                                                    nn.ReLU())

        self.gvp_node_r = nn.Sequential(LayerNorm((config.node_in_dim[0] + 20, config.node_in_dim[1])),
                                        GVP((config.node_in_dim[0] + 20, config.node_in_dim[1]), 
                                            config.node_h_dim, activations=(None, None)))

        self.gvp_edge_r = nn.Sequential(LayerNorm(config.edge_in_dim), 
                                        GVP(config.edge_in_dim, config.edge_h_dim, activations=(None, None)))

        self.dynamics_layers = nn.ModuleList(GVPConvLayer(config.node_h_dim, config.edge_h_dim, drop_rate=config.dropout) 
                                             for _ in range(config.n_layers))
        
        self.dynamics_reward_network = nn.Sequential(LayerNorm(config.node_h_dim), 
                                                     GVP(config.node_h_dim, (config.node_out_dim, 0), activations=(None, None)))

        self.dynamics_reward_output = nn.Sequential(nn.Linear(config.node_out_dim, config.node_out_dim // 2), 
                                                    nn.LeakyReLU(),
                                                    nn.Linear(config.node_out_dim // 2, self.output_size))

        self.dynamics_reward_output[-1].weight.data.fill_(0)
        self.dynamics_reward_output[-1].bias.data.fill_(0)

    def forward(self, hidden_state, action):
        assert len(action.shape) == 2
        assert action.shape[1] == 1

        action = action.to(next(self.dynamics_state_network.parameters()).device)
        action_onehot = torch.zeros(size=(action.size(0), self.config.length * 20), dtype=torch.float32, 
                                    device=next(self.dynamics_state_network.parameters()).device)
        action_onehot.scatter_(1, action, 1.0)
        action_onehot = action_onehot.view(action.size(0), -1, 20)

        # dynamics state
        next_hidden_state = hidden_state.clone()
        node_scalar = next_hidden_state.node_s.view(action.size(0), -1, self.config.hidden_dim)         # [N, L, hidden_dim]
        curr_seq_state = torch.cat([node_scalar, action_onehot], dim=-1).permute(0, 2, 1)               # [N, hidden_dim+20, L]
        next_seq_state = self.dynamics_state_network(curr_seq_state)                                    # [N, hidden_dim, L]
        next_hidden_state.node_s = next_seq_state.permute(0, 2, 1).reshape(-1, self.config.hidden_dim)  # [N*L, hidden_dim]
        
        for i in range(action.size(0)):
            next_hidden_state[i].avaliable_pos[torch.div(action[i], 20, rounding_mode='trunc')] = 0

        # dynamics reward
        hidden_state_input = hidden_state.clone()
        hidden_state_input.node_s = torch.cat([node_scalar, action_onehot], dim=-1).reshape(-1, self.config.hidden_dim + 20)
        
        nodes = (hidden_state_input.node_s, hidden_state_input.node_v)
        edges = (hidden_state_input.edge_s, hidden_state_input.edge_v)
        nodes = self.gvp_node_r(nodes)
        edges = self.gvp_edge_r(edges)
        
        for layer in self.dynamics_layers:
            nodes = layer(nodes, hidden_state_input.edge_index, edges)
        reward = self.dynamics_reward_network(nodes)
        reward = scatter_sum(reward, hidden_state.batch, dim=0)
        reward = self.dynamics_reward_output(reward)
        return next_hidden_state, reward
    
class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.representation_network = RepresentationNetwork(config)
        self.dynamics_network = DynamicsNetwork(config)
        self.prediction_network = PredictionNetwork(config)

    def initial_inference(self, observation: list):
        hidden_state = self.representation_network(observation)
        logits, value = self.prediction_network(hidden_state)
        return hidden_state, 0, logits, value
    
    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        logits, value = self.prediction_network(next_hidden_state)
        return next_hidden_state, reward, logits, value
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

def support_to_scalar(logits, support_size):
    value_probs = torch.softmax(logits, dim=1)
    support = torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(
        value_probs.shape).float().to(device=value_probs.device)
    value = torch.sum(support * value_probs, dim=1, keepdim=True)

    output = (((torch.sqrt(1 + 4 * 1e-3 * (torch.abs(value) + 1 + 1e-3)) - 1) / (2 * 1e-3)) ** 2 - 1)
    output = torch.sign(value) * output
    return output

def dict_to_cpu(dict_input):
    dict_output = {}
    for key, value in dict_input.items():
        if isinstance(value, torch.Tensor):
            dict_output[key] = value.cpu()
        elif isinstance(value, dict):
            dict_output[key] = dict_to_cpu(value)
        else:
            dict_output[key] = value
    return dict_output
    
def scalar_to_support(x, support_size):
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 1e-3 * x
    x = torch.clamp(x, -support_size, support_size)
    x_low = x.floor()

    prob = x - x_low
    prob_low = 1 - prob
    set_size = 2 * support_size + 1

    logits = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
    logits.scatter_(2, (x_low + support_size).long().unsqueeze(-1), prob_low.unsqueeze(-1))
    indexes = x_low + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

def structure_to_graph(structure, n_embedding=16, top_k=12, n_rbf=16):
    plddt = structure["plddt"][0].clone().detach()
    coords = structure["pair"][0][torch.argmax(plddt.mean(1))].clone().detach()

    with torch.no_grad():
        edge_index = torch_cluster.knn_graph(coords, k=top_k)
        edge_vectors = coords[edge_index[0]] - coords[edge_index[1]]

        # node vector: [n_nodes, 2, 3]
        node_v = orientation(coords)

        # edge scalar: [n_edges, 32]
        feature_pos = pos_embedding(n_embedding, edge_index)
        feature_rbf = rbf(torch.linalg.norm(edge_vectors, dim=-1), D_count=n_rbf)
        edge_s = torch.cat([feature_pos, feature_rbf], dim=-1)

        # edge vector: [n_edges, 1, 3]
        edge_v = F.normalize(edge_vectors).unsqueeze(-2)

        graph = GeoData.Data(edge_index=edge_index, 
                             node_s=None, node_v=node_v, 
                             edge_s=edge_s, edge_v=edge_v)
    return graph

def orientation(coords):
    # [n_nodes, 2, 3]
    with torch.no_grad():
        forward = F.pad(F.normalize(coords[1:] - coords[:-1]), [0, 0, 0, 1])
        backward = F.pad(F.normalize(coords[1:] - coords[:-1]), [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def pos_embedding(n_embed, edge_index):
    # [n_edges, 16]
    with torch.no_grad():
        position = edge_index[0] - edge_index[1]
        frequency = torch.exp(torch.arange(0, n_embed, 2, dtype=torch.float32) * -(np.log(10000.0) / n_embed))
        angles = position.unsqueeze(-1) * frequency
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

def rbf(D, D_min=0., D_max=20., D_count=16):
    # [n_edges, 16]
    with torch.no_grad():
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view(1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
