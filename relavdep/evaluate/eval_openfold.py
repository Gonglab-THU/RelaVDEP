import os
import sys
import torch
import timeit
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description='Evaluate foldability of candidate mutants with OpenFold')
parser.add_argument('--fasta', type=str, required=True, help='Wild-type protein sequence')
parser.add_argument('--library', type=str, required=True, help='Mutant library')
parser.add_argument('--output', type=str, required=True, help='Output directory')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Protein sequence does not exist!"
assert os.path.exists(args.library), "Mutant library does not exist!"
output_path = os.path.join(args.output, 'OpenFold')
os.makedirs(output_path, exist_ok=True)

s_time = timeit.default_timer()
target, wt_seq = read_fasta(args.fasta)
library = pd.read_csv(args.library)
selected_mutants = list(library['sequence'])
selected_names = list(library['mutant'])
use_AF_ptm_weight = True
model_idx = 1
weight_set = "AlphaFold"
module_dir = "OpenFold/openfold/"
jackhmmer_binary_path = "env/openfold/bin/jackhmmer"
uniref_90_db = "database/uniref90_2020_0422/uniref90.fasta"
ALPHAFOLD_PARAMS_DIR = "data/OpenFold/AF_weight/"
sys.path.insert(0, os.path.abspath(os.path.join(module_dir)))

from openfold import config
from openfold.data import feature_pipeline
from openfold.data import data_pipeline, data_transforms
from openfold.data import parsers
from openfold.data.tools import jackhmmer
from openfold.model import model
from openfold.np import protein
from openfold.np.relax import relax
from openfold.np.relax import utils
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map, np_tree_map

def _placeholder_template_feats(num_templates_, num_res_):
    return {'template_aatype': np.zeros((num_templates_, num_res_, 22), dtype=np.int64),
            'template_all_atom_positions': np.zeros((num_templates_, num_res_, 37, 3), dtype=np.float32),
            'template_all_atom_mask': np.zeros((num_templates_, num_res_, 37), dtype=np.float32),
            'template_domain_names': np.zeros((num_templates_,), dtype=np.float32),
            'template_sum_probs': np.zeros((num_templates_, 1), dtype=np.float32),}

config_preset = f"model_{model_idx}"
if use_AF_ptm_weight:
    config_preset += "_ptm"

cfg = config.model_config(config_preset)
openfold_model = model.AlphaFold(cfg)
openfold_model = openfold_model.eval()
params_name = os.path.join(ALPHAFOLD_PARAMS_DIR, f"params_{config_preset}.npz")
import_jax_weights_(openfold_model, params_name, version=config_preset)
openfold_model = openfold_model.cuda()

if not os.path.exists(os.path.join(output_path, f'{target}.fasta')):
    with open(os.path.join(output_path, f'{target}.fasta'), 'wt') as f:
        f.write(f'>query\n{wt_seq}')

def search_msa(target_sequence):
    print("Searching MSAs in uniref90...")
    dbs = []
    jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(n_cpu=20, binary_path=jackhmmer_binary_path,
                                                    database_path=uniref_90_db, get_tblout=True,
                                                    z_value=135301051)
    dbs.append(('uniref90', jackhmmer_uniref90_runner.query(target_sequence)))

    msas, deletion_matrices, full_msa = [], [], []
    for db_name, db_results in dbs:
        unsorted_results = []
        for i, result in enumerate(db_results):
            msa, deletion_matrix, target_names = parsers.parse_stockholm(result['sto'])
            e_values_dict = parsers.parse_e_values_from_tblout(result['tbl'])
            e_values = [e_values_dict[t.split('/')[0]] for t in target_names]
            zipped_results = zip(msa, deletion_matrix, target_names, e_values)
            if i != 0:
                zipped_results = [x for x in zipped_results if x[2] != 'query']
            unsorted_results.extend(zipped_results)
        sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
        db_msas, db_deletion_matrices, _, _ = zip(*sorted_by_evalue)
        if db_msas:
            full_msa.extend(db_msas)
            msas.append(db_msas)
            deletion_matrices.append(db_deletion_matrices)
            print(f'{len(set(db_msas))} homologous sequences were found in {db_name}!')

    with open(os.path.join(output_path, f'{target}.a3m'), 'wt') as f:
        f.write(parsers.convert_stockholm_to_a3m(dbs[0][1][0]['sto']))

if not os.path.exists(os.path.join(output_path, f'{target}.a3m')):
    search_msa(os.path.join(output_path, f'{target}.fasta'))

with open(os.path.join(output_path, f'{target}.a3m'), 'rt') as f:
    full_msa = f.read()

plddts, ptms = [], []
targets_seq = [wt_seq] + selected_mutants
targets_name = [target] + selected_names

for n in tqdm(range(len(targets_seq)), desc="Inferencing"):
    sequence = targets_seq[n]
    full_msa = full_msa[:8] + sequence + full_msa[8 + len(sequence):]
    db_msa, db_deletion_matrices = parsers.parse_a3m(full_msa)
    msas = [tuple(db_msa)]
    deletion_matrices = [tuple(db_deletion_matrices)]
    aa_map = {restype: i for i, restype in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')}
    msa_arr = np.array([[aa_map[aa] for aa in seq] for msa in msas for seq in msa], dtype=object)
    num_alignments, num_res = msa_arr.shape
    
    feature_dict = {}
    feature_dict.update(data_pipeline.make_sequence_features(sequence, 'test', num_res))
    feature_dict.update(data_pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))
    feature_dict.update(_placeholder_template_feats(1, num_res))

    pipeline = feature_pipeline.FeaturePipeline(cfg.data)

    processed_feature_dict = pipeline.process_features(feature_dict, mode='predict')
    processed_feature_dict = tensor_tree_map(lambda t: t.cuda(), processed_feature_dict)
    
    with torch.no_grad():
        prediction_result = openfold_model(processed_feature_dict)
        
    processed_feature_dict = tensor_tree_map(lambda t: np.array(t[..., -1].cpu()), processed_feature_dict)
    prediction_result = tensor_tree_map(lambda t: np.array(t.cpu()), prediction_result)
    if processed_feature_dict['aatype'].ndim == 1:
        prot = protein.from_prediction(processed_feature_dict, prediction_result, 
                                       prediction_result['plddt'][:, None] * prediction_result['final_atom_mask'])
    else:
        prot_idx = 0
        prediction_result.pop('max_predicted_aligned_error')
        prot = protein.from_prediction(np_tree_map(lambda t: t[prot_idx], processed_feature_dict), 
                                       np_tree_map(lambda t: t[prot_idx], prediction_result), 
                                       prediction_result['plddt'][prot_idx, :, None] * prediction_result['final_atom_mask'][prot_idx])
    plddts.append(prediction_result['predicted_tm_score'].item())
    ptms.append(prediction_result['plddt'].mean().item())
    
    with open(os.path.join(output_path, f'{targets_name[n]}.pdb'), 'w') as f:
        f.write(protein.to_pdb(prot))

library['plddt (OpenFold)'] = plddts[1:]
library['pTm (OpenFold)'] = ptms[1:]
library.to_csv(os.path.join(args.output, 'library_openfold.csv'), index=False)

e_time = timeit.default_timer()
print(">>> Task finished! Execution time: {:.2f}s <<<".format(e_time - s_time))
