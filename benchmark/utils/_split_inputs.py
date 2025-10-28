import torch
from typing import List

def split_inputs(train_idx: List[bool], test_idx: List[bool], values):
    if isinstance(values, torch.Tensor):
        return values[train_idx], values[test_idx]
    elif isinstance(values, dict):
        ordered_keys = list(values.keys())
        train_dict = {key: values[key] for key, is_train in zip(ordered_keys, train_idx) if is_train}
        test_dict = {key: values[key] for key, is_test in zip(ordered_keys, test_idx) if is_test}
        return train_dict, test_dict
    else:
        raise TypeError("Unsupported type for input values. Must be torch.Tensor or dict.")
