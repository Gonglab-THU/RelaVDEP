import pandas as pd
import statistics
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
from relavdep.modules.utils._models import *
from scripts.utils.loss import spearman_loss

def train_step(model, optimizer, loader, finetune=False):
    model.train()
    total_loss = 0

    for wt_data, mut_data, label in loader:
        wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
        mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
        label = label.to(next(model.parameters()).device)
        optimizer.zero_grad()
        pred = model(wt_data, mut_data)
        if not finetune:
            loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
        else:
            loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def val_step(model, loader, finetune=False):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for wt_data, mut_data, label in loader:
            wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            label = label.to(next(model.parameters()).device)
            pred = model(wt_data, mut_data)
            if not finetune:
                loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
            else:
                loss = F.mse_loss(pred, label)
            total_loss += loss.item()

    return total_loss / len(loader)

class EmbeddingData(Dataset):
    def __init__(self, data, wt_data, embeddings):
        self.data = data
        self.wt_data = wt_data
        self.embeddings = embeddings

    def __getitem__(self, idx):
        mutant = self.data.iloc[idx].mutant
        embedding = self.embeddings[mutant]
        embedding = {k: v.squeeze() for k, v in embedding.items()}
        label = self.data.iloc[idx].label
        return self.wt_data, embedding, torch.tensor(label).to(torch.float32)
    
    def __len__(self):
        return len(self.data)

def sft(cfg, test_fold, model, DMS_id, wt_embedding, embeddings_train, label_train, finetune=False):
    mutants = list(embeddings_train.keys())
    labels = label_train.cpu().numpy().tolist()
    raw_data = pd.DataFrame({'mutant': mutants, 'label': labels})
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    wt_data = {k: v.squeeze() for k, v in wt_embedding.items()}

    val_data = raw_data.sample(frac=0.2, random_state=cfg.seed, axis=0)
    val_dataset = EmbeddingData(val_data, wt_data, embeddings_train)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=4, generator=g, pin_memory=True
    )

    val_mutants_set = set(val_data['mutant'])
    mask = ~raw_data['mutant'].isin(val_mutants_set)
    train_data = raw_data[mask].copy()
    train_dataset = EmbeddingData(train_data, wt_data, embeddings_train)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, 
        shuffle=True, num_workers=4, drop_last=True,
        generator=g, pin_memory=True
    )
    
    if finetune:
        params_to_optimize = []

        for name, param in model.named_parameters():
            if "coef" in name:
                params_to_optimize.append({'params': param, 'lr': cfg.coef_lr})
            if "cons" in name:
                params_to_optimize.append({'params': param, 'lr': cfg.cons_lr})

        optimizer = torch.optim.Adam(params_to_optimize)
    else:
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.init_lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)
    
    best_loss = float('inf')
    stop_step = 0
    train_losses, val_losses = [], []

    pbar = tqdm(range(cfg.epochs), desc='Training' if not finetune else 'Fine-tuning')
    
    for _ in range(cfg.epochs):
        train_loss = train_step(model, optimizer, train_loader, finetune)
        train_losses.append(train_loss)

        val_loss = val_step(model, val_loader, finetune)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            stop_step = 0
            best_loss = val_loss
            if not finetune:
                torch.save(model.state_dict(), os.path.join(cfg.output_folder, f'{cfg.cv_scheme}/{DMS_id}/fold{test_fold}_best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.output_folder, f'{cfg.cv_scheme}/{DMS_id}/fold{test_fold}_{DMS_id}.pth'))
        else:
            stop_step += 1
        
        pbar.update(1)
        if stop_step >= 15:
            pbar.total = pbar.n
            break
    pbar.close()

def inference(test_idx, test_fold, model, wt_embedding, embeddings_test, label_test, df_out):
    pred_fitness = []

    with torch.no_grad():
        for _, mut_embedding in embeddings_test.items():
            wt_data = dict_to_device(wt_embedding, device=next(model.parameters()).device)
            mut_data = dict_to_device(mut_embedding, device=next(model.parameters()).device)
            fitness_value = model(wt_data, mut_data)
            pred_fitness.append(fitness_value.item())
        df_out.loc[test_idx, "fold"] = test_fold
        df_out.loc[test_idx, "y"] = label_test.detach().cpu().numpy()
        df_out.loc[test_idx, "y_pred"] = pred_fitness
        df_out.loc[test_idx, "y_var"] = statistics.stdev(pred_fitness)
    
    return df_out
