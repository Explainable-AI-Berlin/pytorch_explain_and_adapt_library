"""Evaluation dataset utilities for multi-dataset support."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from data.unified_dataloader import prepare_unified_dataloader

logger = logging.getLogger(__name__)


@dataclass
class EvalDatasetInfo:
    """Container for eval dataset info."""
    dataset: Optional[Dataset]
    loader: object
    sampler: Optional[object]
    dataset_size: int
    is_iterable: bool
    reference_npz: Optional[Union[str, List[str]]]  # Only required when 'fid' in metrics
    condition_type: str
    metrics: List[str] = field(default_factory=lambda: ['fid'])
    num_samples: Optional[int] = None  # cap eval at this many samples; None -> full val set
    data_dir: Optional[str] = None     # passed to fd_evaluator for MIND raw-image lookup

    def __len__(self) -> int:
        return int(self.dataset_size)

    def require_indexable_dataset(self, dataset_name: str):
        if self.dataset is None or not hasattr(self.dataset, "__getitem__"):
            raise TypeError(
                f"Eval dataset {dataset_name!r} is iterable-only. The current distributed eval path still "
                f"expects an indexable dataset; use a map-style eval source for now."
            )
        return self.dataset


def _metrics_require_reference_npz(metrics: List[str]) -> bool:
    lowered = [str(metric).lower() for metric in metrics]
    return any(metric == "fid" or metric.startswith("fid_") for metric in lowered)


def _validate_eval_dataset_config(ds_name: str, ds_cfg: dict, metrics: List[str]) -> None:
    num_samples = ds_cfg.get('num_samples')
    if num_samples is not None and int(num_samples) <= 0:
        raise ValueError(f"Eval dataset {ds_name!r} has invalid num_samples={num_samples!r}; expected > 0.")
    if _metrics_require_reference_npz(metrics) and ds_cfg.get('reference_npz') is None:
        raise ValueError(
            f"Eval dataset {ds_name!r} requests metrics {metrics!r} but is missing reference_npz."
        )


def normalize_eval_datasets(datasets_cfg):
    """
    Normalize eval.datasets config to dict of {name: dataset_config}.

    Supported format:
        eval.datasets = {mscoco: {...}, mjhq: {...}}

    Returns dict of {name: dataset_config}. If no datasets are configured, returns an empty dict.
    """
    if datasets_cfg is None:
        return {}
    result = {}
    for name, cfg in datasets_cfg.items():
        result[name] = cfg.copy()
        # set target to name if not explicitly provided (for simpleeval different versions)
        if 'target' not in result[name]:
            result[name]['target'] = name
    return result


def prepare_eval_datasets(
    eval_datasets_config: Dict[str, dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Dict[str, EvalDatasetInfo]:
    """
    Prepare eval datasets from normalized config.

    Returns dict of {name: EvalDatasetInfo}.
    """
    eval_datasets = {}

    for ds_name, ds_cfg in eval_datasets_config.items():
        ds_cond_type = ds_cfg.get('condition_type', 'text')
        metrics = list(ds_cfg.get('metrics', ['fid']))
        _validate_eval_dataset_config(ds_name, ds_cfg, metrics)

        result = prepare_unified_dataloader(
            config=ds_cfg,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            condition_type=ds_cond_type,
            shuffle=False,
        )

        dataset_obj = getattr(result.loader, "dataset", None)
        if (dataset_obj is None or not hasattr(dataset_obj, "__len__")) and getattr(result, "_wds_pipeline", None) is not None:
            dataset_obj = result._wds_pipeline

        eval_datasets[ds_name] = EvalDatasetInfo(
            dataset=dataset_obj,
            loader=result.loader,
            sampler=result.sampler,
            dataset_size=int(result.dataset_size),
            is_iterable=bool(result.is_iterable),
            reference_npz=ds_cfg.get('reference_npz'),
            condition_type=ds_cond_type,
            metrics=metrics,
            num_samples=ds_cfg.get('num_samples'),
            data_dir=ds_cfg.get('data_dir'),
        )
        logger.info(f"Eval dataset loaded: {ds_name}, {int(result.dataset_size)} samples")

    return eval_datasets
