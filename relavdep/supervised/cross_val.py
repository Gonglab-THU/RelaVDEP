import os
import torch
import timeit
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from utils import *
from scripts.models import *
from scripts.utils.loss import spearman_loss
from scripts.utils.metrics import spearman_corr
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='Cross-validation of the reward model')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--mutant', type=str, required=True, help='Mutation data')
parser.add_argument('--embedding', type=str, required=True, help='Directory for DHR embeddings')
parser.add_argument('--output', type=str, required=True, help='Output directory')
parser.add_argument('--rm_type', type=str, required=True, help='Reward model type', choices=['fitness', 'stab'])

parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: %(default)s)')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Testing ratio (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: %(default)s)')
parser.add_argument('--n_fold', type=int, default=5, help='Number of CV folds (default: %(default)s)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', help="Device to use (default: %(default)s)")
args = parser.parse_args()

assert os.path.exists(args.fasta), "Input protein sequence does not exist!"
assert os.path.exists(args.mutant), "Input mutation data does not exist!"
target, wt_seq = read_fasta(args.fasta)
raw_data = pd.read_csv(args.mutant)
os.makedirs(args.output, exist_ok=True)

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

test_data = raw_data.sample(frac=args.test_ratio, random_state=args.seed, axis=0)
test_dataset = BatchData(test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

train_val_index = [i for i in range(len(raw_data)) if raw_data.loc[i, 'mutant'] not in list(test_data['mutant'])]
train_val_data = raw_data.loc[train_val_index].copy()
splitor = KFold(n_splits=args.n_fold, shuffle=False)

def train_step(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for _, wt_data, mut_data, label in train_loader:
        wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
        mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
        label = label.to(next(model.parameters()).device)
        optimizer.zero_grad()
        pred = model(wt_data, mut_data)
        loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
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
            loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, 'kl')
            total_loss += loss.item()
    return total_loss / len(val_loader)

def test_step(model, test_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for _, wt_data, mut_data, label in test_loader:
            wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            pred = model(wt_data, mut_data)
            preds.append(pred.item())
            trues.append(label.item())
    loss = -spearman_corr(torch.tensor(preds), torch.tensor(trues))
    return loss.item()

def main():
    s_time = timeit.default_timer()
    for n_layer in [1, 2, 3, 4, 5]:
        os.makedirs(os.path.join(args.output, f'CV/layer_{n_layer}'), exist_ok=True)
        for k, (train_index, val_index) in enumerate(splitor.split(train_val_data)):
            train_data = train_val_data.iloc[train_index].copy()
            train_dataset = BatchData(train_data)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, 
                drop_last=True, num_workers=4, pin_memory=True
            )

            val_data = train_val_data.iloc[val_index].copy()
            val_dataset = BatchData(val_data)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True, 
                drop_last=True, num_workers=4, pin_memory=True
            )

            if args.rm_type == 'fitness':
                model = SmallFitness(n_layer)
                model_dict = model.state_dict().copy()
                best_model = torch.load('../data/params/SPIRED-Fitness.pth').copy()
                best_dict = {k: v for k, v in best_model.items() if k in model_dict}
            elif args.rm_type == 'stab':
                model = SmallStab(n_layer)
                model_dict = model.state_dict().copy()
                best_model = torch.load('../data/params/SPIRED-Stab.pth').copy()
                best_dict = {k.split('Stab.')[-1]: v for k, v in best_model.items() if k.split('Stab.')[-1] in model_dict}
            else:
                raise ValueError("Invalid model type!")

            model_dict.update(best_dict)
            model.load_state_dict(model_dict)
            model.to(args.device)

            for name, param in model.named_parameters():
                if 'down_stream_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)
            
            best_loss = float('inf')
            stop_step = 0
            train_losses, val_losses, test_losses = [], [], []
            for _ in tqdm(range(args.epochs), desc=f'Layer-{n_layer} Fold-{k}'):
                train_loss = train_step(model, optimizer, train_loader)
                train_losses.append(train_loss)

                val_loss = val_step(model, val_loader)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                test_loss = test_step(model, test_loader)
                test_losses.append(test_loss)

                losses = pd.DataFrame({"Train": train_losses, "Val": val_losses, "Test": test_losses})
                losses.to_csv(os.path.join(args.output, f'CV/layer_{n_layer}/fold_{k}.csv'), index=False)

                if val_loss < best_loss:
                    stop_step = 0
                    best_loss = val_loss
                else:
                    stop_step += 1
                
                if stop_step >= 15:
                    break
    
    e_time = timeit.default_timer()
    print(">>> Task finished. Execution time: {:.2f}s <<<".format(e_time - s_time))

if __name__ == "__main__":
    main()
