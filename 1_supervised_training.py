import os
import timeit
import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from scripts.utils.loss import spearman_loss
from scripts.utils.metrics import spearman_corr
from scripts.utils import *
from relavdep.modules.reward_model import *
from relavdep.modules.utils._functions import *

parser = argparse.ArgumentParser(description='Supervised fine-tuning the reward model')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--data', type=str, required=True, help='Mutation data')

parser.add_argument('--output', type=str, default='outputs', help='Output directory')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: %(default)s)')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Testing ratio (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size (default: %(default)s)')
parser.add_argument('--n_fold', type=int, default=5, help='Number of CV folds (default: %(default)s)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
parser.add_argument('--init_lr', type=float, default=1e-3, help='Learning rate (default: %(default)s)')
parser.add_argument('--cross_val', action='store_true', default=False, help='Perform cross validation (default: %(default)s)')
args = parser.parse_args()

print(f"{'=' * 60}")
assert os.path.exists(args.fasta), "!!! Input protein sequence does not exist !!!"
target_name, target_sequence = read_fasta(args.fasta)

raw_data = process_and_check_csv(args.data, target_sequence)
if raw_data is None:
    raise ValueError("!!! Data loading error. Please verify the file format !!!")
os.makedirs(args.output, exist_ok=True)

set_worker_seed(args.seed)
g = torch.Generator()
g.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

###### 1. Extract embeddings ######

current_path = os.path.abspath(os.path.dirname(__file__))
cst_path = os.path.join('relavdep', 'data', 'mutation_constraint')
base_model = BaseModel(data_dir=os.path.join(current_path, 'models'), device=device)
fitness_params = os.path.join(current_path, 'models', 'SPIRED-Fitness.pth')
all_targets = [target_name] + raw_data['mutant'].tolist()
all_sequences = [target_sequence] + raw_data['sequence'].tolist()
embeddings_path = os.path.join(args.output, 'embeddings')
rm_params = os.path.join(args.output, f'{target_name}.pth')
os.makedirs(embeddings_path, exist_ok=True)

print(">>> Stage 1: Extract ESM-2 embeddings and predict structures.")
s1_start = timeit.default_timer()

for target, sequence in tqdm(zip(all_targets, all_sequences), total=len(all_targets), desc="Extracting Embeddings"):
    if not os.path.exists(os.path.join(embeddings_path, f'{target}.pt')):
        mut_data = base_model.inference(sequence)
        mut_data = dict_to_device(mut_data, 'cpu')
        torch.save(mut_data, os.path.join(embeddings_path, f'{target}.pt'))

s1_end = timeit.default_timer()
print(f"Stage completed. Duration: {s1_end - s1_start:.2f}s")

###### 2. Cross-Validation ######

class EmbeddingData(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, idx):
        mutant = self.subset.iloc[idx].mutant
        label = self.subset.iloc[idx].label

        wt_data = torch.load(os.path.join(embeddings_path, f'{target_name}.pt'), map_location='cpu')
        wt_data = {k: v.squeeze() for k, v in wt_data.items()}

        mut_data = torch.load(os.path.join(embeddings_path, f'{mutant}.pt'), map_location='cpu')
        mut_data = {k: v.squeeze() for k, v in mut_data.items()}
        return wt_data, mut_data, torch.tensor(label).to(torch.float32)
    
    def __len__(self):
        return len(self.subset)

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

def test_step(model, loader, finetune=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for wt_data, mut_data, label in loader:
            wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            pred = model(wt_data, mut_data)
            preds.append(pred.item())
            trues.append(label.item())
    if not finetune:
        loss = spearman_corr(torch.tensor(preds), torch.tensor(trues))
    else:
        loss = F.mse_loss(torch.tensor(preds), torch.tensor(trues))
    return loss.item()

def calc_best_layers():
    result = pd.DataFrame()
    calculate = pd.DataFrame()

    for n in range(5):
        for k in range(args.n_fold):
            tmp = pd.read_csv(os.path.join(args.output, f'cv/layer_{n+1}/fold_{k+1}.csv'))
            result.loc[n * args.n_fold + k, 'Spearman Correlation'] = tmp.loc[tmp['Validation'].argmin(), 'Test']
            result.loc[n * args.n_fold + k, 'MLP Layer Count'] = str(n+1)
            calculate.loc[f'fold-{k+1}', f'layer-{n+1}'] = tmp.loc[tmp['Validation'].argmin(), 'Test']
    
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans'],
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 24, 
        'ytick.labelsize': 24,
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    custom_colors = ['#cbe5f2', '#95adcf', '#98b7ba', '#b4b6d4', '#f3dba9']
    
    sns.stripplot(x='MLP Layer Count', y='Spearman Correlation', 
                  data=result, hue='MLP Layer Count', jitter=True, 
                  size=6, alpha=0.8, edgecolor='black', linewidth=0.5, 
                  palette=custom_colors)
    
    sns.boxplot(x='MLP Layer Count', y='Spearman Correlation', 
                data=result, hue='MLP Layer Count', showfliers=False, 
                width=0.5, palette=custom_colors)

    calculate_result = calculate.describe().loc['50%']
    for index, value in enumerate(calculate_result):
        ax.text(x=index + 0.5, y=value, s='{:.2f}'.format(value),
                color='black', fontsize=18, va='center', ha='center')
        
    ax.xaxis.grid(True)
    ax.set(xlabel='MLP Layer Count', ylabel=r'Spearman Correlation ($\rho$)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'cross_validation.png'), dpi=300)
    
    best_idx = np.argmax(calculate_result)
    best_layer = best_idx + 1
    return best_layer

def cross_validation(train_val_data, splitor, test_loader):
    pbar = tqdm(range(25), desc=f'Cross-Validation')
    for n_layer in range(1, 6):
        os.makedirs(os.path.join(args.output, f'cv/layer_{n_layer}'), exist_ok=True)
        for k, (train_index, val_index) in enumerate(splitor.split(train_val_data)):
            train_data = train_val_data.iloc[train_index].copy()
            train_dataset = EmbeddingData(train_data)
            train_loader = DataLoader(
                train_dataset, batch_size=min(args.batch_size, len(train_data)), 
                shuffle=True, drop_last=True, num_workers=4, 
                generator=g, pin_memory=True
            )

            val_data = train_val_data.iloc[val_index].copy()
            val_dataset = EmbeddingData(val_data)
            val_loader = DataLoader(
                val_dataset, batch_size=min(args.batch_size, len(val_data)), 
                shuffle=True, drop_last=True, num_workers=4, 
                generator=g, pin_memory=True
            )

            model = FitnessModel(n_layer)
            model_dict = model.state_dict().copy()
            best_model = torch.load(fitness_params, map_location=torch.device('cpu')).copy()
            best_dict = {k: v for k, v in best_model.items() if k in model_dict}
            model_dict.update(best_dict)
            model.load_state_dict(model_dict)
            model.to(device)

            for name, param in model.named_parameters():
                if 'down_stream_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.init_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)
            best_loss = float('inf')
            stop_step = 0
            train_losses, val_losses, test_losses = [], [], []

            for _ in range(args.epochs):
                train_loss = train_step(model, optimizer, train_loader)
                train_losses.append(train_loss)

                val_loss = val_step(model, val_loader)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                test_loss = test_step(model, test_loader)
                test_losses.append(test_loss)

                losses = pd.DataFrame({"Train": train_losses, "Validation": val_losses, "Test": test_losses})
                losses.to_csv(os.path.join(args.output, f'cv/layer_{n_layer}/fold_{k+1}.csv'), index=False)

                if val_loss < best_loss:
                    stop_step = 0
                    best_loss = val_loss
                else:
                    stop_step += 1
                
                if stop_step >= 15:
                    break
            pbar.update(1)
    pbar.close()

###### 3. Normal Training ######

def normal_training(model, train_loader, val_loader, finetune=False):
    if finetune:
        params_to_optimize = []

        for name, param in model.named_parameters():
            if "coef" in name:
                params_to_optimize.append({'params': param, 'lr': 1e-3})
            if "cons" in name:
                params_to_optimize.append({'params': param, 'lr': 1e-3})

        optimizer = torch.optim.Adam(params_to_optimize)
    else:
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.init_lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)

    best_loss = float('inf')
    stop_step = 0
    train_losses, val_losses = [], []

    pbar = tqdm(range(args.epochs), desc='Training' if not finetune else 'Fine-tuning')
    
    for _ in range(args.epochs):
        train_loss = train_step(model, optimizer, train_loader, finetune)
        train_losses.append(train_loss)

        val_loss = val_step(model, val_loader, finetune)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            stop_step = 0
            best_loss = val_loss
            torch.save(model.state_dict(), rm_params)
        else:
            stop_step += 1
        
        pbar.update(1)
        if stop_step >= 15:
            pbar.total = pbar.n
            break
    pbar.close()

test_data = raw_data.sample(frac=args.test_ratio, random_state=args.seed, axis=0)
test_dataset = EmbeddingData(test_data)

test_mutants_set = set(test_data['mutant'])
mask = ~raw_data['mutant'].isin(test_mutants_set)
train_data = raw_data[mask].copy()

print(">>> Stage 2: Determine the depth of MLP.")
s2_start = timeit.default_timer()

if args.cross_val:
    try:
        test_loader = DataLoader(
            test_dataset, batch_size=1, 
            shuffle=False, num_workers=4, 
            generator=g, pin_memory=True
        )
        
        splitor = KFold(n_splits=args.n_fold, shuffle=False)
        cross_validation(train_data, splitor, test_loader)
        best_layer = calc_best_layers()
    except Exception as e:
        print(f"!!! An unexpected error occurred during the cross-validation process: {e} !!!")
else:
    best_layer = 2

s2_end = timeit.default_timer()
print(f"Stage completed. Duration: {s2_end - s2_start:.2f}s")

model = FitnessModel(best_layer)
model_dict = model.state_dict().copy()
best_model = torch.load(fitness_params, map_location=torch.device('cpu')).copy()
best_dict = {k: v for k, v in best_model.items() if k in model_dict}
model_dict.update(best_dict)
model.load_state_dict(model_dict)
model.to(device)

for name, param in model.named_parameters():
    if 'down_stream_model' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

try:
    train_dataset = EmbeddingData(train_data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_data)), 
        shuffle=True, num_workers=4, drop_last=True,
        generator=g, pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=min(args.batch_size, len(test_data)), 
        shuffle=True, drop_last=True, num_workers=4, 
        generator=g, pin_memory=True
    )
    
    print(">>> Stage 3: Supervised training the reward model.")
    s3_start = timeit.default_timer()

    normal_training(model, train_loader, val_loader, finetune=False)

    s3_end = timeit.default_timer()
    print(f"Stage completed. Duration: {s3_end - s3_start:.2f}s")

    model.load_state_dict(torch.load(rm_params, map_location=torch.device('cpu')))
    model.to(device)

    for name, param in model.named_parameters():
        if 'finetune' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print(">>> Stage 4: Fine-tuning the reward model.")
    s4_start = timeit.default_timer()

    normal_training(model, train_loader, val_loader, finetune=True)

    s4_end = timeit.default_timer()
    print(f"Stage completed. Duration: {s4_end - s4_start:.2f}s")

    print(">>> Stage 5: Predicting fitness for test mutants.")
    s5_start = timeit.default_timer()

    model.load_state_dict(torch.load(rm_params, map_location=torch.device('cpu')))
    model.eval().to(device)

    wt_data = torch.load(os.path.join(embeddings_path, f'{target_name}.pt'), map_location=torch.device('cpu'))
    wt_data = dict_to_device(wt_data, device=next(model.parameters()).device)
    
    pred_targets = [target_name] + test_data['mutant'].tolist()
    pred_sequences = [target_sequence] + test_data['sequence'].tolist()
    
    pred_fitness = []
    for target, sequence in tqdm(zip(pred_targets, pred_sequences), total=len(pred_targets), desc="Predicting"):
        mut_data = torch.load(os.path.join(embeddings_path, f'{target}.pt'), map_location=torch.device('cpu'))
        mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
        
        with torch.no_grad():
            fitness_value = model(wt_data, mut_data)
        pred_fitness.append(fitness_value.item())
    
    test_data['pred'] = pred_fitness[1:]
    test_spearman = test_data['label'].corr(test_data['pred'], 'spearman')
    print(f"Spearman correlation of validation set: {test_spearman:.2f}")
    
    s5_end = timeit.default_timer()
    print(f"Stage completed. Duration: {s5_end - s5_start:.2f}s")
    
    print(">>> Stage 6: Extract predicted beneficial mutations.")
    s6_start = timeit.default_timer()

    all_single_mutations = []
    for i, original_residue in enumerate(target_sequence):
        for residue in list(A2int.keys()):
            if residue != original_residue:
                mutation_name = f"{original_residue}{i + 1}{residue}"
                mutated_sequence = target_sequence[:i] + residue + target_sequence[i+1:]
                all_single_mutations.append({'mutant': mutation_name, 'sequence': mutated_sequence})
    
    mutations_df = pd.DataFrame(all_single_mutations)
    pred_scores = []
    
    for mutant in tqdm(mutations_df['sequence']):
        with torch.no_grad():
            mut_data = base_model.inference(mutant)
            mut_data = dict_to_device(mut_data, device=next(model.parameters()).device)
            score = model(wt_data, mut_data)
        pred_scores.append(score.item())
    
    mutations_df['fitness'] = pred_scores
    mutations_df = mutations_df.sort_values(by='fitness', ascending=False)
    mutations_df = mutations_df.reset_index(drop=True)
    result_df = mutations_df[mutations_df['fitness'] > pred_fitness[0]].copy()
    
    legal, illegal = [], []
    for mutant in result_df['mutant']:
        pos = int(mutant[1:-1]) - 1
        res = A2int[mutant[-1]]
        action = pos * 20 + res + 1
        legal.append(action)
    
    cst_file = os.path.join(cst_path, f'{target_name}.npz')
    np.savez(cst_file, illegal=illegal, legal=legal)

    s6_end = timeit.default_timer()
    print(f"Stage completed. Duration: {s6_end - s6_start:.2f}s")

    print(f"All processes completed. Duration: {s6_end - s1_start:.2f}s")
    print(f"{'=' * 60}")
    print("** Next Steps: Parameters for Subsequent Scripts **")
    print("------------------------------------------------------------")
    print("1. For 2_directed_evolution.py (Virtual Directed Evolution), add the following arguments:")
    if args.cross_val:
        print(f"--n_layer {best_layer} \\")
    print(f"--rm_params {rm_params} \\")
    print(f"--constraint {cst_file}")
    print("------------------------------------------------------------")
    print("2. For 3_construct_library.py (Mutant Library Construction), add the following argument:")
    print(f"--cutoff {pred_fitness[0]:.4f}")
    print(f"{'=' * 60}")
except Exception as e:
    print(f"!!! An unexpected error occurred: {e} !!!")
