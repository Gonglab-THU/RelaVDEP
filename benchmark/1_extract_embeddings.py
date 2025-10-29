"""
Script to extract ESM-2 embeddings and predict structures.
Adapated from https://github.com/petergroth/kermut/blob/main/proteingym_benchmark.py
"""
import h5py
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_path, '..'))
from relavdep.modules.utils._models import *

def _filter_datasets(cfg: DictConfig, embedding_dir: Path) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.reference_file)

    if cfg.dataset == "all":
        df_ref = df_ref[df_ref["seq_len"] < cfg.len_cutoff]
        if cfg.cv_splits == "multiples":
            df_ref = df_ref[df_ref["includes_multiple_mutants"]]
            df_ref = df_ref[df_ref["DMS_total_number_mutants"] < cfg.num_cutoff]
            df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
        if cfg.cv_splits == "singles":
            df_ref = df_ref[df_ref["DMS_number_single_mutants"] < cfg.num_cutoff]
    elif cfg.dataset == "single":
        if (df_ref["DMS_id"] == cfg.single_id).any():
            df_ref = df_ref[df_ref["DMS_id"] == cfg.single_id]
        else:
            raise ValueError(f"Invalid single dataset id: {cfg.single_id}")
    else:
        raise ValueError(f"Invalid dataset: {cfg.dataset}")

    existing_results = []
    for DMS_id in df_ref["DMS_id"]:
        output_file = embedding_dir / f"{DMS_id}.h5"
        if output_file.exists():
            existing_results.append(DMS_id)
    df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="benchmark",
)
def extract_embeddings(cfg:DictConfig):
    if cfg.cv_splits == "singles":
        embedding_dir = Path(cfg.embedding_singles)
        DMS_dir = Path(cfg.DMS_singles)
    elif cfg.cv_splits == "multiples":
        embedding_dir = Path(cfg.embedding_multiples)
        DMS_dir = Path(cfg.DMS_multiples)
    else:
        raise ValueError(f"!!! Invalid cv_splits: {cfg.cv_splits} !!!")

    df_ref = _filter_datasets(cfg, embedding_dir)

    if len(df_ref) == 0:
        print("All embeddings already exist. Exiting.")
        return
    
    device = 'cuda' if cfg.use_gpu and torch.cuda.is_available() else 'cpu'
    base_model = BaseModel(data_dir=cfg.model_path, device=device)

    for i in tqdm(range(len(df_ref))):
        DMS_id = df_ref.iloc[i].DMS_id
        wt_seq = df_ref.iloc[i].target_seq

        print(f"--- Extracting embeddings for {DMS_id} ({i+1}/{len(df_ref)}) ---")
        df = pd.read_csv(DMS_dir / f"{DMS_id}.csv")

        mutants = df["mutant"].tolist()
        sequences = df["mutated_sequence"].tolist()

        with h5py.File(embedding_dir / f"{DMS_id}.h5", "w") as f:
            for mutant, sequence in zip(mutants, sequences):
                mut_data = base_model.inference(sequence)
                mut_data = dict_to_device(mut_data, 'cpu')
                
                group = f.create_group(mutant)
                for key, value in mut_data.items():
                    value_numpy = value.cpu().clone().numpy()
                    group.create_dataset(key, data=value_numpy)
            
            wt_data = base_model.inference(wt_seq)
            wt_data = dict_to_device(wt_data, 'cpu')
            group = f.create_group("WT")
            for key, value in wt_data.items():
                value_numpy = value.cpu().clone().numpy()
                group.create_dataset(key, data=value_numpy)
    
if __name__ == "__main__":
    extract_embeddings()
