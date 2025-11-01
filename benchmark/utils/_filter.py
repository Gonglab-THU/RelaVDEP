from pathlib import Path
import pandas as pd
from omegaconf import DictConfig

def filter_datasets(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.reference_file)

    if cfg.dataset == "all":
        df_ref = df_ref[df_ref["includes_multiple_mutants"]]
        df_ref = df_ref[df_ref["seq_len"] < cfg.len_cutoff]
        df_ref = df_ref[df_ref["DMS_total_number_mutants"] < cfg.num_cutoff]
        # special dataset
        df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
    elif cfg.dataset == "single":
        if (df_ref["DMS_id"] == cfg.single_id).any():
            df_ref = df_ref[df_ref["DMS_id"] == cfg.single_id]
        else:
            raise ValueError(f"Invalid single dataset id: {cfg.single_id}")
    else:
        raise ValueError(f"Invalid dataset: {cfg.dataset}")

    df_ref = df_ref[["DMS_id", "target_seq"]]
    existing_results = []
    for DMS_id in df_ref["DMS_id"]:
        output_dir = Path(cfg.output_folder) / cfg.cv_scheme / DMS_id
        existing_results = []
        for DMS_id in df_ref["DMS_id"]:
            if (output_dir / f"{DMS_id}.csv").exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref