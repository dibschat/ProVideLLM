from functools import partial

from torch.utils.data import ConcatDataset, Dataset

# all datasets loaded here
from .egoexo4d import *
from .ego4d import *
from .coin import *
from .data_collator import get_data_collator

__all__ = [
    "build_concat_train_dataset",
    "build_eval_dataset_dict",
    "get_data_collator",
    "get_compute_metrics_dict",
]


def _build_list_datasets(datasets: list, is_training: bool, **kwargs):
    datasets = [
        globals()[f"build_{dataset}"](is_training=is_training, **kwargs)
        for dataset in datasets
    ]
    return datasets


def build_concat_train_dataset(train_datasets: list, is_training=True, **kwargs):
    if train_datasets is None or len(train_datasets) == 0:
        return None
    return ConcatDataset(
        _build_list_datasets(datasets=train_datasets, is_training=is_training, **kwargs)
    )


def build_eval_dataset_dict(eval_datasets: list, is_training=False, **kwargs):
    if eval_datasets is None or len(eval_datasets) == 0:
        return None
    list_datasets = _build_list_datasets(
        datasets=eval_datasets, is_training=is_training, **kwargs
    )
    return {name: dataset for name, dataset in zip(eval_datasets, list_datasets)}


def get_compute_metrics_dict(dataset_dict: dict, **kwargs):
    if not dataset_dict:
        return None
    # add eval_ since transformers default metrics prefix is eval
    return {k: partial(v.compute_metrics, **kwargs) for k, v in dataset_dict.items()}
