import os
import re
import random
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from ._alphabet import *

def read_fasta(fasta):
    FastaIterator = SeqIO.parse(fasta, "fasta")
    names, sequences = [], []
    for item in FastaIterator:
        names.append(item.id)
        sequences.append(''.join([s for s in item.seq]))
    assert len(sequences) == 1, "Input fasta file must contain only one sequence!"
    for s in sequences[0]:
        assert s in list(A2int.keys()), "Non-standard residue is not allowed in sequence!"
    return names[0], sequences[0]

def mutation(curr_seq, action):
    mutant = list(curr_seq)
    mut_pos, mut_res = (action - 1) // 20, (action - 1) % 20
    mutant[mut_pos] = int2A[mut_res]
    mutant = ''.join(mutant)
    return mutant

def seq2onehot(seq):
    onehot = np.zeros((len(seq), 20))
    for i in range(len(seq)):
        onehot[i, A2int[seq[i]]] = 1
    return onehot

def set_worker_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def apply_mutation(mutant_str, base_seq):
    current_seq_list = list(base_seq)
    mutations = mutant_str.strip().upper().split(':')

    for mutation in mutations:
        mutation = mutation.strip()
        if not mutation:
            continue
        
        match = re.match(r'([A-Z])(\d+)([A-Z])', mutation)
        if not match:
            return f"ERROR: Invalid single mutation format in '{mutation}' of full mutant '{mutant_str}'"
        
        try:
            original_aa = match.group(1)
            position = int(match.group(2))
            mutated_aa = match.group(3)
            
            idx = position - 1
            if idx < 0 or idx >= len(base_seq):
                return f"ERROR: Position {position} out of bounds ({len(base_seq)}) in mutation '{mutation}'"
            
            expected_aa_at_pos = base_seq[idx]
            if expected_aa_at_pos != original_aa:
                return (f"ERROR: Expected '{original_aa}' at position {position} in mutation '{mutation}', "
                        f"but base sequence has '{expected_aa_at_pos}'")
            current_seq_list[idx] = mutated_aa
        except Exception as e:
            return f"ERROR: Cannot process mutation '{mutation}' in '{mutant_str}' - Details: {e}"
    return "".join(current_seq_list)

def process_and_check_csv(file_path, target_sequence):
    if not os.path.exists(file_path):
        print(f"ERROR: File path '{file_path}' does not exist. Please check the file path.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read file '{file_path}'.")
    except pd.errors.EmptyDataError:
        print("!!! ERROR: File is empty. !!!")
        return None
    except pd.errors.ParserError:
        print("!!! ERROR: File format is incorrect and cannot be parsed as CSV. !!!")
        return None
    except Exception as e:
        print(f"!!! An unknown error occurred while reading the file: {e} !!!")
        return None
    
    if 'mutated_sequence' and 'DMS_score' in df.columns:
        raw_data = df[['mutant', 'mutated_sequence', 'DMS_score']].copy()
        raw_data.rename(columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'}, inplace=True)
    else:
        required_cols = ['mutant', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print("!!! Data Naming Error !!!")
            print(f"Missing required columns: {missing_cols}")
            print("The data must include both 'mutant' and 'label' columns. Please check and correct your CSV file.")
            return None
        else:
            raw_data = df.copy()
            raw_data['sequence'] = raw_data['mutant'].apply(lambda x: apply_mutation(x, target_sequence))

            error_count = raw_data['sequence'].astype(str).str.startswith("ERROR:").sum()
            if error_count > 0:
                print(f"!!! NOTE: {error_count} records had errors during 'sequence' generation. Please check them. !!!")
                return None
            else:
                print("Sequences successfully generated.")
    return raw_data