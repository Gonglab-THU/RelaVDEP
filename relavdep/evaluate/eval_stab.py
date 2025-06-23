import os
import torch
import timeit
import argparse
import pandas as pd
import sys
sys.path.append('..')
from scripts.models import *
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description='Evaluate ΔΔG and ΔTm of candidate mutants with SPIRED-Stab')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--library', type=str, required=True, help='Mutant library')
parser.add_argument('--output', type=str, required=True, help='Output directory')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Protein sequence does not exist!"
assert os.path.exists(args.library), "Mutant library does not exist!"
output_path = os.path.join(args.output, 'SPIRED')
os.makedirs(output_path, exist_ok=True)

s_time = timeit.default_timer()
target, wt_seq = read_fasta(args.fasta)
library = pd.read_csv(args.library)
selected_mutants = list(library['sequence'])
selected_names = list(library['mutant'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Initialize SPIRED-Stab...")
base_model = BaseModel(params_dir='../data/params', device=device)
stab_model = StabModel(node_dim = 32, num_layer = 3, n_head = 8, pair_dim = 64)
stab_params = torch.load('../data/params/SPIRED-Stab.pth').copy()
best_dict = {k.split('Stab.')[-1]: v for k, v in stab_params.items() if k.startswith('Stab')}
stab_model.load_state_dict(best_dict)
stab_model.eval().to(device)

def dump2pdb(name, sequence, coords):
    with open(os.path.join(output_path, f'{name}.pdb'), 'w') as f:
        natom = 1
        for l in range(len(sequence)):
            f.write('{:<6}{:>5} {:^4} {:<3} {:>1}{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>12}\n'.format(
                'ATOM', int(natom), 'CA', int2AA.get(A2int.get(sequence[l])), 'A', l+1, coords[l, 0], coords[l, 1], coords[l, 2], 1.00, 0.00, 'C'))
            natom += 1

def process_data(data):
    pair, plddt = data['pair'][0], data['plddt'][0]
    max_index = torch.argmax(plddt.mean(1))
    pair_max = pair[max_index].clone().detach().cpu().numpy()
    plddt_max = plddt[max_index].clone().detach().cpu().numpy()
    return pair_max, plddt_max

print("2. Predicte ΔΔG and ΔTm of candidate mutants...")
wt_data = base_model.inference(wt_seq)
pair_max, _ = process_data(wt_data)
dump2pdb(target, wt_seq, pair_max)

ddG_preds, dTm_preds, plddts = [], [], []
for i in tqdm(range(len(selected_mutants)), desc=f"Inferencing"):
    mut_seq = selected_mutants[i]
    mut_data = base_model.inference(mut_seq)
    mut_pos = (wt_data['tokens'] != mut_data['tokens']).int().to(device)
    with torch.no_grad():
        ddG, dTm = stab_model(wt_data, mut_data, mut_pos)
    ddG_preds.append(ddG.item())
    dTm_preds.append(dTm.item())
    pair_max, plddt_max = process_data(mut_data)
    dump2pdb(selected_names[i], mut_seq, pair_max)
    plddts.append(plddt_max.mean())

library['ddG'] = ddG_preds
library['dTm'] = dTm_preds
library['plddt (SPIRED)'] = plddts
library.to_csv(os.path.join(args.output, 'library_stab.csv'), index=False)

e_time = timeit.default_timer()
print(">>> Task finished. Execution time: {:.2f}s <<<".format(e_time - s_time))
