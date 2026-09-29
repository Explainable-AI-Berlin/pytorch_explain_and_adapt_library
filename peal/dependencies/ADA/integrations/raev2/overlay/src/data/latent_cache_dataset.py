"""Map-style Stage-2 latent-cache dataset for native RAEv2 training."""

from __future__ import annotations

import bisect
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Sampler


def _to_int_index(idx) -> int:
    if torch.is_tensor(idx):
        return int(idx.item())
    return int(idx)


class Stage2LatentCacheDataset(Dataset):
    """Dataset over torch-saved Stage-2 latent shards.

    Each shard is a dictionary with:
      z: Tensor[N, C, H, W]
      y: Tensor[N]

    Optional keys:
      cls: Tensor[N, D]
      aux: Tensor[N, ...]
      meta: Tensor[N, ...]
      source_index: Tensor[N]
      view: Tensor[N]
    """

    def __init__(
        self,
        root: str | Path,
        cache_mode: str = "lazy",
        max_open_shards: int = 2,
        cls_normalization: str = "raw",
    ):
        self.root = Path(root).expanduser()
        self.cache_mode = str(cache_mode)
        self.max_open_shards = max(1, int(max_open_shards))
        self.cls_normalization = str(cls_normalization or "raw").lower()
        if self.cls_normalization == "none":
            self.cls_normalization = "raw"
        if self.cls_normalization not in {"raw", "train_standardize"}:
            raise ValueError(
                f"Unsupported cls_normalization={self.cls_normalization!r}; "
                "expected raw/train_standardize."
            )

        self.metadata_path = self.root / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing latent-cache metadata: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        shard_entries = self.metadata.get("shards", [])
        if not shard_entries:
            raise ValueError(f"Latent-cache metadata has no shards: {self.metadata_path}")

        self.shards = [
            {
                "path": self.root / str(entry["file"]),
                "num_samples": int(entry["num_samples"]),
            }
            for entry in shard_entries
        ]
        self.cumulative = []
        total = 0
        for shard in self.shards:
            if not shard["path"].exists():
                raise FileNotFoundError(f"Missing latent-cache shard: {shard['path']}")
            total += shard["num_samples"]
            self.cumulative.append(total)
        self.total = total

        self.y_vocab = self.metadata.get("y_vocab", None) or self.metadata.get("class_to_idx", None)
        self.meta_vocabs = self.metadata.get("meta_vocabs", None)
        self.meta_fields = list(self.metadata.get("meta_fields", []))
        self.exclude_summary = self.metadata.get("exclude_summary", None)
        self.exclude_specs = self.metadata.get("exclude_specs", None)
        self._has_cls = bool(self.metadata.get("cls_condition", {}).get("included", False))
        self._cls_mean = None
        self._cls_std = None
        if self._has_cls and self.cls_normalization == "train_standardize":
            stats_file = self.metadata.get("cls_condition", {}).get("stats_file", "cls_stats.pt")
            stats_path = self.root / str(stats_file)
            if not stats_path.exists():
                raise FileNotFoundError(f"CLS normalization requested but stats file is missing: {stats_path}")
            stats = torch.load(stats_path, map_location="cpu")
            self._cls_mean = stats["mean"].float()
            self._cls_std = stats["std"].float().clamp_min(1.0e-6)

        self._z = None
        self._y = None
        self._aux = None
        self._meta = None
        self._cls = None
        self._source_index = None
        self._view = None
        self._cache = OrderedDict()
        if self.cache_mode == "memory":
            self._load_all_to_memory()
        elif self.cache_mode != "lazy":
            raise ValueError(f"Unknown latent cache_mode={self.cache_mode!r}; expected lazy/memory.")

    def _validate_payload(self, payload: dict, path: Path) -> dict:
        if "z" not in payload or "y" not in payload:
            raise KeyError(f"Latent-cache shard must contain 'z' and 'y': {path}")
        shard_has_cls = "cls" in payload and payload["cls"] is not None
        if self._has_cls and not shard_has_cls:
            raise ValueError(f"Latent-cache metadata expects CLS, but shard is missing 'cls': {path}")
        if (not self._has_cls) and shard_has_cls:
            raise ValueError(f"Shard contains 'cls' but metadata does not declare it: {path}")
        return payload

    def _load_all_to_memory(self) -> None:
        z_parts = []
        y_parts = []
        aux_parts = []
        meta_parts = []
        cls_parts = []
        source_parts = []
        view_parts = []
        has_aux = has_meta = has_cls = has_source = has_view = False
        for shard in self.shards:
            payload = self._validate_payload(torch.load(shard["path"], map_location="cpu"), shard["path"])
            z_parts.append(payload["z"])
            y_parts.append(payload["y"].long())
            if "aux" in payload and payload["aux"] is not None:
                has_aux = True
                aux_parts.append(payload["aux"])
            if "meta" in payload and payload["meta"] is not None:
                has_meta = True
                meta_parts.append(payload["meta"].long())
            if "cls" in payload and payload["cls"] is not None:
                has_cls = True
                cls_parts.append(payload["cls"])
            if "source_index" in payload and payload["source_index"] is not None:
                has_source = True
                source_parts.append(payload["source_index"].long())
            if "view" in payload and payload["view"] is not None:
                has_view = True
                view_parts.append(payload["view"].long())
        self._z = torch.cat(z_parts, dim=0)
        self._y = torch.cat(y_parts, dim=0)
        self._aux = torch.cat(aux_parts, dim=0) if has_aux else None
        self._meta = torch.cat(meta_parts, dim=0) if has_meta else None
        self._cls = torch.cat(cls_parts, dim=0) if has_cls else None
        self._source_index = torch.cat(source_parts, dim=0) if has_source else None
        self._view = torch.cat(view_parts, dim=0) if has_view else None

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        idx = _to_int_index(idx)
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.cumulative, idx)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        return shard_idx, idx - prev

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._cache:
            payload = self._cache.pop(shard_idx)
            self._cache[shard_idx] = payload
            return payload
        payload = self._validate_payload(torch.load(self.shards[shard_idx]["path"], map_location="cpu"), self.shards[shard_idx]["path"])
        self._cache[shard_idx] = payload
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return payload

    def _normalize_cls(self, cls: torch.Tensor | None):
        if cls is None:
            return None
        cls = cls.float()
        if self.cls_normalization == "train_standardize":
            return (cls - self._cls_mean.to(cls.device)) / self._cls_std.to(cls.device)
        return cls

    def _make_item(self, z, y, aux=None, meta=None, cls=None, source_index=None, view=None):
        if cls is None:
            if aux is None and meta is None:
                return z, y.long()
            if aux is None:
                return z, y.long(), meta.long()
            if meta is None:
                return z, y.long(), aux
            return z, y.long(), aux, meta.long()
        item = {"z": z, "y": y.long(), "cls": self._normalize_cls(cls)}
        if aux is not None:
            item["aux"] = aux
        if meta is not None:
            item["meta"] = meta.long()
        if source_index is not None:
            item["source_index"] = source_index.long()
        if view is not None:
            item["view"] = view.long()
        return item

    def __getitem__(self, idx: int):
        idx = _to_int_index(idx)
        if self.cache_mode == "memory":
            return self._make_item(
                self._z[idx],
                self._y[idx],
                None if self._aux is None else self._aux[idx],
                None if self._meta is None else self._meta[idx],
                None if self._cls is None else self._cls[idx],
                None if self._source_index is None else self._source_index[idx],
                None if self._view is None else self._view[idx],
            )
        shard_idx, local_idx = self._locate(idx)
        payload = self._load_shard(shard_idx)
        return self._make_item(
            payload["z"][local_idx],
            payload["y"][local_idx].long(),
            None if payload.get("aux", None) is None else payload["aux"][local_idx],
            None if payload.get("meta", None) is None else payload["meta"][local_idx].long(),
            None if payload.get("cls", None) is None else payload["cls"][local_idx],
            None if payload.get("source_index", None) is None else payload["source_index"][local_idx].long(),
            None if payload.get("view", None) is None else payload["view"][local_idx].long(),
        )

    def __getitems__(self, indices):
        indices = [_to_int_index(idx) for idx in indices]
        if self.cache_mode == "memory":
            return [self[idx] for idx in indices]

        grouped = OrderedDict()
        for out_pos, idx in enumerate(indices):
            shard_idx, local_idx = self._locate(idx)
            grouped.setdefault(shard_idx, []).append((out_pos, local_idx))

        out = [None] * len(indices)
        for shard_idx, positions in grouped.items():
            payload = self._load_shard(shard_idx)
            for out_pos, local_idx in positions:
                out[out_pos] = self._make_item(
                    payload["z"][local_idx],
                    payload["y"][local_idx].long(),
                    None if payload.get("aux", None) is None else payload["aux"][local_idx],
                    None if payload.get("meta", None) is None else payload["meta"][local_idx].long(),
                    None if payload.get("cls", None) is None else payload["cls"][local_idx],
                    None if payload.get("source_index", None) is None else payload["source_index"][local_idx].long(),
                    None if payload.get("view", None) is None else payload["view"][local_idx].long(),
                )
        return out


class ShardBatchDistributedSampler(Sampler):
    """Distributed sampler that keeps rank-local batches contiguous by shard.

    Loading one cache shard deserializes hundreds of megabytes. Assigning whole
    shards to ranks and consuming all batches consecutively avoids reloading a
    payload for every randomly interleaved microbatch.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        shard_interleave_window: int = 1,
    ):
        if not hasattr(dataset, "shards") or not hasattr(dataset, "cumulative"):
            raise TypeError("ShardBatchDistributedSampler requires a dataset with shards and cumulative offsets.")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank={rank} for num_replicas={num_replicas}.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shard_interleave_window = max(1, int(shard_interleave_window))
        self.epoch = 0
        self._rank_shards, rank_batch_counts = self._assign_shards_to_ranks()
        self.num_batches = min(rank_batch_counts)
        if self.num_batches <= 0:
            raise ValueError("Shard assignment produced no complete batches.")
        self.num_samples = self.num_batches * self.batch_size

    def _shard_batch_count(self, shard_idx: int) -> int:
        shard_len = int(self.dataset.shards[shard_idx]["num_samples"])
        if self.drop_last:
            return shard_len // self.batch_size
        return int(math.ceil(shard_len / self.batch_size))

    def _assign_shards_to_ranks(self):
        """Greedily balance whole shards while never sharing one across ranks."""
        rank_shards = [[] for _ in range(self.num_replicas)]
        rank_batch_counts = [0] * self.num_replicas
        shard_order = sorted(
            range(len(self.dataset.shards)),
            key=lambda shard_idx: (-self._shard_batch_count(shard_idx), shard_idx),
        )
        for shard_idx in shard_order:
            rank = min(
                range(self.num_replicas),
                key=lambda candidate: (rank_batch_counts[candidate], candidate),
            )
            rank_shards[rank].append(shard_idx)
            rank_batch_counts[rank] += self._shard_batch_count(shard_idx)
        return rank_shards, rank_batch_counts

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        shard_batches = []
        shard_order = list(self._rank_shards[self.rank])
        if self.shuffle:
            order = torch.randperm(len(shard_order), generator=generator).tolist()
            shard_order = [shard_order[i] for i in order]

        for shard_idx in shard_order:
            start = 0 if shard_idx == 0 else self.dataset.cumulative[shard_idx - 1]
            end = self.dataset.cumulative[shard_idx]
            shard_len = end - start
            local_order = torch.randperm(shard_len, generator=generator).tolist() if self.shuffle else list(range(shard_len))
            usable = (shard_len // self.batch_size) * self.batch_size if self.drop_last else shard_len
            batches = []
            for offset in range(0, usable, self.batch_size):
                chunk = local_order[offset : offset + self.batch_size]
                if len(chunk) < self.batch_size:
                    if self.drop_last:
                        continue
                    chunk = chunk + local_order[: self.batch_size - len(chunk)]
                batches.append([start + int(local_idx) for local_idx in chunk])
            shard_batches.append(batches)

        # Interleave a bounded set of resident shards. This preserves the
        # one-load-per-shard I/O behavior while preventing optimizer updates
        # from being composed entirely of one index-ordered cache shard.
        batches = []
        window_size = self.shard_interleave_window
        for window_start in range(0, len(shard_batches), window_size):
            window = shard_batches[window_start : window_start + window_size]
            max_batches = max((len(group) for group in window), default=0)
            for batch_idx in range(max_batches):
                for group in window:
                    if batch_idx < len(group):
                        batches.append(group[batch_idx])

        # All ranks must execute the same number of DDP steps. The greedy shard
        # assignment is usually exact; trim only the small balancing remainder.
        batches = batches[: self.num_batches]
        if len(batches) != self.num_batches:
            raise RuntimeError(
                f"Rank sampler produced {len(batches)} batches, expected {self.num_batches}."
            )
        return iter(idx for batch in batches for idx in batch)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
