"""
Script to prepare inputs.
Adapated from https://github.com/petergroth/kermut/blob/main/proteingym_benchmark.py
"""
import h5py
import pandas as pd
import torch
from pathlib import Path
from omegaconf import DictConfig

def dict_to_device(dict_input, device):
    dict_output = {}
    for key, value in dict_input.items():
        if isinstance(value, torch.Tensor):
            dict_output[key] = value.to(device)
        elif isinstance(value, dict):
            dict_output[key] = dict_to_device(value, device)
        else:
            dict_output[key] = value
    return dict_output

def _load_embeddings(cfg: DictConfig, df: pd.DataFrame, DMS_id: str):
    if cfg.cv_scheme == "fold_rand_multiples":
        embedding_path = Path(cfg.embedding_multiples) / f"{DMS_id}.h5"
    else:
        embedding_path = Path(cfg.embedding_singles) / f"{DMS_id}.h5"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}")

    try:
        with h5py.File(embedding_path, "r", locking=True) as f:
            embeddings_dict = {}
            for mutant in f.keys():
                group = f[mutant]
                embedding = {}
                for key in group.keys():
                    value_np = group[key][:]
                    embedding[key] = torch.from_numpy(value_np)
                embeddings_dict[mutant] = embedding
    except Exception as e:
        print(e)

    order = df["mutant"].tolist()
    embeddings = {key: embeddings_dict[key] for key in order}
    return embeddings_dict["WT"], embeddings

def prepare_inputs(cfg: DictConfig, DMS_id: str):
    if cfg.cv_splits == "multiples":
        df = pd.read_csv(Path(cfg.DMS_multiples) / f"{DMS_id}.csv")
    elif cfg.cv_splits == "singles":
        df = pd.read_csv(Path(cfg.DMS_singles) / f"{DMS_id}.csv")
    else:
        raise ValueError(f"Invalid cv_splits: {cfg.cv_splits}")

    label = torch.tensor(df[cfg.DMS_label].values, dtype=torch.float32)
    wt_embedding, embeddings = _load_embeddings(cfg, df, DMS_id)

    return df, label, wt_embedding, embeddings
