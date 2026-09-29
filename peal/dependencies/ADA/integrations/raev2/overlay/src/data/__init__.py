"""Data loading utilities for RAE training."""

try:
    from .histo_manifest import HistoManifestDataset
except ModuleNotFoundError:
    HistoManifestDataset = None
from .imagenet_classes import IMAGENET_CLASSES
from .unified_dataloader import (
    DataloaderResult,
    MixedDataloader,
    prepare_unified_dataloader,
)
from .latent_cache_dataset import Stage2LatentCacheDataset, ShardBatchDistributedSampler

try:
    from .imagenet_hf_dataset import ImageNetHFDataset
except ModuleNotFoundError:
    ImageNetHFDataset = None

try:
    from .blip3o_wds_dataset import BLIP3O_METADATA, BLIP3OWebDataset
except ModuleNotFoundError:
    BLIP3OWebDataset = None
    BLIP3O_METADATA = None

try:
    from .wds_image_dataset import GENERIC_WDS_METADATA, GenericWebDataset
except ModuleNotFoundError:
    GenericWebDataset = None
    GENERIC_WDS_METADATA = None

try:
    from .histo_wds_dataset import DEFAULT_HISTO_META_FIELDS, HistoWebDataset
except ModuleNotFoundError:
    HistoWebDataset = None
    DEFAULT_HISTO_META_FIELDS = None

__all__ = [
    "ImageNetHFDataset",
    "HistoManifestDataset",
    "HistoWebDataset",
    "DEFAULT_HISTO_META_FIELDS",
    "IMAGENET_CLASSES",
    "prepare_unified_dataloader",
    "DataloaderResult",
    "MixedDataloader",
    "BLIP3OWebDataset",
    "BLIP3O_METADATA",
    "GenericWebDataset",
    "GENERIC_WDS_METADATA",
    "Stage2LatentCacheDataset",
    "ShardBatchDistributedSampler",
]
