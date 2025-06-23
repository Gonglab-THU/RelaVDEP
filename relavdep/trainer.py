import os
import ray
import copy
import torch
import pickle
import time
import numpy as np
from network import *
from torch.utils.tensorboard import SummaryWriter

@ray.remote
class Trainer:
    def __init__(self, config, initial_checkpoint):
        self.config = config
        self.writer = SummaryWriter(config.output_path)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        self.model = Network(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device('cuda' if self.config.train_on_gpu else'cpu'))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.learning_rate, 
                                          weight_decay=self.config.weight_decay)

    def _train(self, shared_storage, replay_buffer):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < self.config.batch_size * 2:
            time.sleep(0.1)
        
        next_batch = replay_buffer.sample_batch.remote()
        while self.training_step < self.config.training_steps:
            batch = ray.get(next_batch)
            next_batch = replay_buffer.sample_batch.remote()
            priorities, idx_batch, losses = self.train_loop(batch)
            self.update_lr()
            
            replay_buffer.update_priorities.remote(priorities, idx_batch)
            
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote({"weights": copy.deepcopy(self.model.get_weights()),
                                                "optimizer": copy.deepcopy(dict_to_cpu(self.optimizer.state_dict()))})
                shared_storage.save_checkpoint.remote(os.path.join(self.config.output_path, 'checkpoint.pth'))
                replay_buffer.output_sequences.remote()
                
                if not self.config.no_buffer:
                    buffer = ray.get(replay_buffer.get_buffer.remote())
                    with open(os.path.join(self.config.output_path, 'replay_buffer.pkl'), 'wb') as f:
                        pickle.dump(buffer, f)
                torch.cuda.empty_cache()

            shared_storage.set_info.remote({"training_step": self.training_step, 
                                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                                            "total_loss": losses[0], "policy_loss": losses[1], 
                                            "value_loss": losses[2], "reward_loss": losses[3]})
            
            self.writer.add_scalar('Training worker/Learning_rate', self.optimizer.param_groups[0]["lr"], self.training_step)
            self.writer.add_scalar('Training worker/Total_loss', losses[0], self.training_step)
            self.writer.add_scalar('Training worker/Policy_loss', losses[1], self.training_step)
            self.writer.add_scalar('Training worker/Value_loss', losses[2], self.training_step)
            self.writer.add_scalar('Training worker/Reward_loss', losses[3], self.training_step)

            if self.config.train_delay:
                time.sleep(self.config.train_delay)
        self.writer.close()

    def train_loop(self, batch):
        idx_batch, obs_batch, act_batch, reward_batch, value_batch, policy_batch, grad_batch, weight_batch = batch
        target_value_scalar = np.array(value_batch, dtype='float32')
        priorities = np.zeros_like(target_value_scalar)
        model_device = next(self.model.parameters()).device
        
        action_batch = torch.tensor(np.array(act_batch)).long().to(model_device).unsqueeze(-1)
        target_reward = torch.tensor(np.array(reward_batch)).float().to(model_device)
        target_value = torch.tensor(np.array(value_batch)).float().to(model_device)
        target_policy = torch.tensor(np.array(policy_batch)).float().to(model_device)
        grad_batch = torch.tensor(np.array(grad_batch)).float().to(model_device)
        weight_batch = torch.tensor(weight_batch.copy()).float().to(model_device)
        
        target_value = scalar_to_support(target_value, self.config.support_size)
        target_reward = scalar_to_support(target_reward, self.config.support_size)
        
        hidden_state, reward, logits, value = self.model.initial_inference(obs_batch)
        predictions = [(logits, value, reward)]

        for i in range(1, self.config.num_unroll_steps+1):
            hidden_state, reward, logits, value = self.model.recurrent_inference(hidden_state, action_batch[:, i])
            hidden_state.node_s.register_hook(lambda grad: grad * 0.5)
            predictions.append((logits, value, reward))

        policy_loss, value_loss, reward_loss = 0, 0, 0
        logits, value, reward = predictions[0]
        
        current_policy_loss = torch.sum(-target_policy[:, 0] * torch.log_softmax(logits, dim=1), dim=1)
        current_value_loss = torch.sum(-target_value[:, 0] * torch.log_softmax(value.squeeze(), dim=1), dim=1)
        
        policy_loss += current_policy_loss
        value_loss += current_value_loss
        
        pred_value_scalar = support_to_scalar(value, self.config.support_size).squeeze().detach().cpu().numpy()
        priorities[:, 0] = np.abs(pred_value_scalar - target_value_scalar[:, 0]) ** self.config.prob_alpha

        for i in range(1, len(predictions)):
            logits, value, reward = predictions[i]
            current_policy_loss = torch.sum(-target_policy[:, i] * torch.log_softmax(logits, dim=1), dim=1)
            current_value_loss = torch.sum(-target_value[:, i] * torch.log_softmax(value.squeeze(), dim=1), dim=1)
            current_reward_loss = torch.sum(-target_reward[:, i] * torch.log_softmax(reward.squeeze(), dim=1), dim=1)
            
            current_policy_loss.register_hook(lambda grad: grad / grad_batch[:, i])
            current_value_loss.register_hook(lambda grad: grad / grad_batch[:, i])
            current_reward_loss.register_hook(lambda grad: grad / grad_batch[:, i])
            
            policy_loss += current_policy_loss
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            
            pred_value_scalar = support_to_scalar(value, self.config.support_size).squeeze().detach().cpu().numpy()
            priorities[:, i] = np.abs(pred_value_scalar - target_value_scalar[:, i]) ** self.config.prob_alpha

        policy_loss_item = policy_loss
        value_loss_item = self.config.value_loss_weight * value_loss
        reward_loss_item = reward_loss
        
        policy_loss_item *= weight_batch
        value_loss_item *= weight_batch
        reward_loss_item *= weight_batch
        total_loss = (policy_loss_item + value_loss_item + reward_loss_item).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.training_step += 1

        losses = (total_loss.item(), 
                  policy_loss_item.mean().item(), 
                  value_loss_item.mean().item(), 
                  reward_loss_item.mean().item())       
        return priorities, idx_batch, losses

    def update_lr(self):
        lr = self.config.learning_rate * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
