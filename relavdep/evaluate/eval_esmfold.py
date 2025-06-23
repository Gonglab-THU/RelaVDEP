import os
import esm
import torch
import timeit
import argparse
import pandas as pd
from tqdm import tqdm
from utils import *
import biotite.structure.io as bsio

parser = argparse.ArgumentParser(description='Evaluate foldability of candidate mutants with ESMFold')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--library', type=str, required=True, help='Mutant library')
parser.add_argument('--output', type=str, required=True, help='Output directory')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Protein sequence does not exist!"
assert os.path.exists(args.library), "Mutant library does not exist!"
output_path = os.path.join(args.output, 'ESMFold')
os.makedirs(output_path, exist_ok=True)

print("====== Predict the mutant structures with ESMFold ======")
s_time = timeit.default_timer()
target, wt_seq = read_fasta(args.fasta)
library = pd.read_csv(args.library)
selected_mutants = list(library['sequence'])
selected_names = list(library['mutant'])
targets_seq = [wt_seq] + selected_mutants
targets_name = [target] + selected_names
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = esm.pretrained.esmfold_v1().eval()
model = model.to(device)

plddts = []
for i in tqdm(range(len(targets_seq)), desc=f"Inferencing"):
    sequence = targets_seq[i]
    with torch.no_grad():
        prediction = model.infer_pdb(sequence)
    with open(os.path.join(output_path, f'{targets_name[i]}.pdb'), 'w') as f:
        f.write(prediction)
    struct = bsio.load_structure(os.path.join(output_path, f'{targets_name[i]}.pdb'), extra_fields=["b_factor"])
    plddts.append(struct.b_factor.mean())

library['plddt (ESMFold)'] = plddts[1:]
library.to_csv(os.path.join(args.output, 'library_esmfold.csv'), index=False)

e_time = timeit.default_timer()
print(">>> Task finished! Execution time: {:.2f}s <<<".format(e_time - s_time))
