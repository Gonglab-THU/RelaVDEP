import os
import torch
import timeit
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append('..')
from utils import *
from scripts.models import *
from scripts.utils.loss import spearman_loss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

parser = argparse.ArgumentParser(description='Supervised fine-tune the reward model')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--mutant', type=str, required=True, help='Mutation data')
parser.add_argument('--embedding', type=str, required=True, help='Directory for DHR embeddings')
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument('--rm_type', type=str, required=True, help='Reward model type', 
                    choices=['SmallFitness', 'LargeFitness', 'SmallStab', 'LargeStab'])

parser.add_argument('--n_layer', type=int, default=5, help='Number of MLP layers (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: %(default)s)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', help="Device to use (default: %(default)s)")
parser.add_argument('--finetune', action='store_true', default=False, help='Fine-tune the reward model (default: %(default)s)')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Input protein sequence does not exist!"
assert os.path.exists(args.mutant), "Input mutation data does not exist!"
target, wt_seq = read_fasta(args.fasta)
raw_data = pd.read_csv(args.mutant)
os.makedirs(args.output, exist_ok=True)
if args.finetune:
    assert os.path.exists(os.path.join(args.output, 'best.pth')), "The 'best.pth' file does not exist, please run train.py without `--finetune` at first!"

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

class BatchData(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, idx):
        mutant = self.subset.iloc[idx].mutant
        label = self.subset.iloc[idx].label
        wt_data = torch.load(os.path.join(args.embedding, f'{target}.pt'), map_location='cpu')
        wt_data = {k: v.squeeze() for k, v in wt_data.items()}
        mut_data = torch.load(os.path.join(args.embedding, f'{mutant}.pt'), map_location='cpu')
        mut_data = {k: v.squeeze() for k, v in mut_data.items()}
        return mutant, wt_data, mut_data, torch.tensor(label).to(torch.float32)
    
    def __len__(self):
        return len(self.subset)

train_length = int(0.8 * len(raw_data))
val_length = len(raw_data) - train_length
train_index, val_index = random_split(
    range(len(raw_data)), [train_length, val_length], 
    generator=torch.Generator().manual_seed(args.seed)
)

train_data = raw_data.loc[list(train_index)].copy()
train_dataset = BatchData(train_data)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

val_data = raw_data.loc[list(val_index)].copy()
val_dataset = BatchData(val_data)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

def train_step(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for _, wt_data, mut_data, label in train_loader:
        wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
        mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
        label = label.to(next(model.parameters()).device)
        optimizer.zero_grad()
        pred = model(wt_data, mut_data)
        if not args.finetune:
            loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
        else:
            loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def val_step(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, wt_data, mut_data, label in val_loader:
            wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            label = label.to(next(model.parameters()).device)
            pred = model(wt_data, mut_data)
            if not args.finetune:
                loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
            else:
                loss = F.mse_loss(pred, label)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    s_time = timeit.default_timer()

    if 'Large' in args.rm_type:
        model = globals()[args.rm_type]()
    else:
        model = globals()[args.rm_type](n_layer=args.n_layer)

    if not args.finetune:
        model_dict = model.state_dict().copy()
        if 'Stab' in args.rm_type:
            best_model = torch.load('../data/params/SPIRED-Stab.pth', map_location='cpu').copy()
            best_dict = {k.split('Stab.')[-1]: v for k, v in best_model.items() if k.split('Stab.')[-1] in model_dict}
        else:
            best_model = torch.load('../data/params/SPIRED-Fitness.pth', map_location='cpu').copy()
            best_dict = {k: v for k, v in best_model.items() if k in model_dict}
        model_dict.update(best_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(torch.load(os.path.join(args.output, 'best.pth')))
    model.to(args.device)

    if not args.finetune:
        for name, param in model.named_parameters():
            if 'Large' in args.rm_type:
                if 'finetune' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                if 'down_stream_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if 'finetune' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.5, patience = 10)
    
    best_loss = float('inf')
    stop_step = 0
    train_losses, val_losses = [], []
    for _ in tqdm(range(args.epochs), desc='Training' if not args.finetune else 'Fine-tuning'):
        train_loss = train_step(model, optimizer, train_loader)
        train_losses.append(train_loss)

        val_loss = val_step(model, val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        losses = pd.DataFrame({"Train": train_losses, "Val": val_losses})
        if not args.finetune:
            losses.to_csv(os.path.join(args.output, 'train_loss.csv'), index=False)
        else:
            losses.to_csv(os.path.join(args.output, 'finetune_loss.csv'), index=False)

        if val_loss < best_loss:
            stop_step = 0
            best_loss = val_loss
            if not args.finetune:
                torch.save(model.state_dict(), os.path.join(args.output, 'best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.output, f'{target}.pth'))
        else:
            stop_step += 1
        
        if stop_step >= 15:
            break
    
    if args.finetune:
        model.load_state_dict(torch.load(os.path.join(args.output, f'{target}.pth')))
        model.eval()

        pred_scores = []
        for i in tqdm(range(len(raw_data)), "Inference"):
            mutant = raw_data.loc[i, "mutant"]
            wt_data = torch.load(os.path.join(args.embedding, f'{target}.pt'))
            wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)

            mut_data = torch.load(os.path.join(args.embedding, f'{mutant}.pt'))
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            
            with torch.no_grad():
                score = model(wt_data, mut_data)
            pred_scores.append(score.item())
        
        with torch.no_grad():
            wt_score = model(wt_data, wt_data)
        print("The fitness of wild-type is {:.4f}.".format(wt_score.item()))
        
        raw_data['pred'] = pred_scores
        raw_data.to_csv(os.path.join(args.output, 'pred.csv'), index=False)
    
    e_time = timeit.default_timer()
    print(">>> Task finished. Execution time: {:.2f}s <<<".format(e_time - s_time))

if __name__ == "__main__":
    main()
