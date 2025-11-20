import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

import ray
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path

from utils._filter import filter_datasets
from utils._prepare_inputs import prepare_inputs
from utils._split_inputs import split_inputs
from utils._train_rm import *

@ray.remote(num_gpus=0.5)
def _evaluate_single_dms(cfg: DictConfig, DMS_id: str) -> None:
    try:
        from relavdep.modules.utils._models import FitnessModel

        os.makedirs(os.path.join(cfg.output_folder, cfg.cv_scheme, DMS_id), exist_ok=True)
        df, labels, wt_embedding, embeddings = prepare_inputs(cfg, DMS_id)
        device = 'cuda' if cfg.use_gpu and torch.cuda.is_available() else 'cpu'
        df_out = df[["mutant"]].copy()
        df_out = df_out.assign(fold=np.nan, y=np.nan, y_pred=np.nan, y_var=np.nan)
        
        unique_folds = df[cfg.cv_scheme].unique()
        
        for test_fold in unique_folds:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
    
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.seed)
                torch.cuda.manual_seed_all(cfg.seed)

            train_idx = (df[cfg.cv_scheme] != test_fold).tolist()
            test_idx = (df[cfg.cv_scheme] == test_fold).tolist()

            label_train, label_test = split_inputs(train_idx, test_idx, labels)
            embeddings_train, embeddings_test = split_inputs(train_idx, test_idx, embeddings)

            model = FitnessModel(cfg.n_layer)
            model_dict = model.state_dict().copy()
            best_model = torch.load(cfg.fitness_params, map_location=torch.device('cpu')).copy()
            best_dict = {k: v for k, v in best_model.items() if k in model_dict}
            model_dict.update(best_dict)
            model.load_state_dict(model_dict)
            model.to(device)

            for name, param in model.named_parameters():
                if 'down_stream_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            best_params = os.path.join(cfg.output_folder, f'{cfg.cv_scheme}/{DMS_id}/fold{test_fold}_best.pth')
            
            if not os.path.exists(best_params):
                sft(cfg, test_fold, model, DMS_id, wt_embedding, embeddings_train, label_train, finetune=False)
            
            model.load_state_dict(torch.load(best_params, map_location=torch.device('cpu')))
            model.to(device)

            for name, param in model.named_parameters():
                if 'finetune' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            model_params = os.path.join(cfg.output_folder, f'{cfg.cv_scheme}/{DMS_id}/fold{test_fold}_{DMS_id}.pth')
            
            if not os.path.exists(model_params):
                sft(cfg, test_fold, model, DMS_id, wt_embedding, embeddings_train, label_train, finetune=True)

            model.load_state_dict(torch.load(model_params, map_location=torch.device('cpu')))
            model.eval().to(device)

            df_out = inference(test_idx, test_fold, model, wt_embedding, embeddings_test, label_test, df_out)

        out_path = Path(cfg.output_folder) / cfg.cv_scheme / DMS_id / f"{DMS_id}.csv"
        os.makedirs(str(out_path.parent), exist_ok=True)

        spearman = df_out["y"].corr(df_out["y_pred"], "spearman")
        mae = np.mean(np.abs(df_out["y"] - df_out["y_pred"]))
        df_out.to_csv(out_path, index=False)

        return {
            "DMS_id": DMS_id, 
            "status": "SUCCESS", 
            "spearman": f"{spearman:.4f}",
            "MAE": f"{mae:.4f}"
        }

    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"{error_msg} (DMS ID: {DMS_id})", flush=True)
        return {
            "DMS_id": DMS_id, 
            "status": "FAILURE", 
            "error_message": error_msg
        }

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="benchmark",
)
def main(cfg: DictConfig) -> None:
    ray.init(
        log_to_driver=False, 
        _temp_dir=os.path.abspath("/tmp/ray"), 
        num_gpus=cfg.num_gpus
    )
    
    df_ref = filter_datasets(cfg)
    futures = []

    for i, (DMS_id, _) in enumerate(df_ref.itertuples(index=False)):
        print(f"[{i+1}/{len(df_ref)}] Submitting {DMS_id}", flush=True)
        future = _evaluate_single_dms.remote(cfg, DMS_id)
        futures.append(future)

    pending_futures = futures.copy()
    total_tasks = len(futures)
    completed_count = 0

    print("--- Waiting for Tasks (Real-time Status) ---")

    while completed_count < total_tasks:
        ready_futures, pending_futures = ray.wait(
            pending_futures, 
            num_returns=1, 
            timeout=cfg.timeout
        )

        if ready_futures:
            try:
                result = ray.get(ready_futures[0])
                completed_count += 1

                if result["status"] == "SUCCESS":
                    print(f"[{completed_count}/{total_tasks}] SUCCESS: {result['DMS_id']}"
                          f"(Spearman: {result['spearman']}, MAE: {result['MAE']})")
                else:
                    print(f"[{completed_count}/{total_tasks}] FAILURE: {result['DMS_id']} ({result['error_message']})")
            
            except ray.exceptions.RayTaskError as e:
                completed_count += 1
                print(f"[{completed_count}/{total_tasks}] CRASHED: A task failed unexpectedly. Error details: {e}", flush=True)
        
        elif pending_futures:
            print(f"[PROGRESS] {completed_count}/{total_tasks} tasks completed. Still running...", flush=True)

    print("--- All tasks completed. ---")
    ray.shutdown()

if __name__ == "__main__":
    main()
