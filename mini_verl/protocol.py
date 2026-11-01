"""DataProto: the one data structure that flows between the driver and the workers.

Mirrors ``verl/protocol.py``. A DataProto has three parts:

* ``batch``            -- a ``TensorDict``: tensors that share the first (batch) dimension,
* ``non_tensor_batch`` -- a dict of per-row Python lists (verl uses numpy object arrays),
* ``meta_info``        -- a plain dict that is *not* split along the batch (temperature, metrics...).

verl uses the ``tensordict`` package; mini_verl ships a ~60-line ``TensorDict`` with just the
operations DataProto needs, so the only dependency stays torch.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

_padding_size_key = "_padding_size_key_x123d"  # same magic key as verl/protocol.py


class DataProtoConfig:
    """Global auto-padding switch (verl/protocol.py ``DataProtoConfig``)."""

    auto_padding = False
    auto_padding_key = "_verl_auto_padding"


class TensorDict:
    """A dict of tensors with a common leading batch dimension (tiny stand-in for tensordict)."""

    def __init__(self, source: Dict[str, torch.Tensor], batch_size):
        self.batch_size = torch.Size(batch_size)
        self._d = {}
        for k, v in source.items():
            self[k] = v

    def __setitem__(self, key, value):
        assert value.shape[:1] == self.batch_size, f"{key}: {tuple(value.shape)} vs batch {tuple(self.batch_size)}"
        self._d[key] = value

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._d[item]
        out = {k: v[item] for k, v in self._d.items()}  # slice / index tensor / list
        n = len(range(*item.indices(self.batch_size[0]))) if isinstance(item, slice) else len(item)
        return TensorDict(out, (n,))

    def __contains__(self, key):
        return key in self._d

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def pop(self, key):
        return self._d.pop(key)

    def select(self, *keys):
        return TensorDict({k: self._d[k] for k in keys}, self.batch_size)

    @staticmethod
    def cat(tds: List["TensorDict"]) -> "TensorDict":
        keys = list(tds[0].keys())
        out = {k: torch.cat([td[k] for td in tds], dim=0) for k in keys}
        return TensorDict(out, (sum(td.batch_size[0] for td in tds),))


def _index_list(values: list, idxs: List[int]) -> list:
    return [values[i] for i in idxs]


def list_of_dict_to_dict_of_list(list_of_dict: List[dict]) -> dict:
    """verl/utils/py_functional.py: [{'a': 1}, {'a': 2}] -> {'a': [1, 2]} (lists are flattened)."""
    out = {}
    for d in list_of_dict:
        for k, v in d.items():
            out.setdefault(k, []).extend(v if isinstance(v, list) else [v])
    return out


@dataclass
class DataProto:
    """verl/protocol.py ``DataProto``."""

    batch: Optional[TensorDict] = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def check_consistency(self):
        n = len(self)
        for k, v in self.non_tensor_batch.items():
            assert isinstance(v, list), f"non_tensor_batch[{k!r}] must be a list"
            assert len(v) == n or self.batch is None, f"non_tensor_batch[{k!r}] has {len(v)} rows, batch has {n}"

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        if isinstance(item, (list, torch.Tensor)):
            return self.select_idxs(item)
        raise TypeError(f"Indexing with {type(item)} is not supported")

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None, auto_padding=False):
        tensors, non_tensors, meta_info = tensors or {}, dict(non_tensors or {}), dict(meta_info or {})
        sizes = {k: v.shape[0] for k, v in tensors.items()}
        assert len(set(sizes.values())) <= 1, f"Not all tensors have the same batch size: {sizes}"
        batch = TensorDict(tensors, (next(iter(sizes.values())),)) if tensors else None
        non_tensors = {k: list(v) for k, v in non_tensors.items()}
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=batch, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_single_dict(cls, data: dict, meta_info=None, auto_padding=False):
        tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        non_tensors = {k: v for k, v in data.items() if not isinstance(v, torch.Tensor)}
        return cls.from_dict(tensors, non_tensors, meta_info, auto_padding)

    # ---------------------------------------------------------------- selection
    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False):
        batch = self.batch.select(*batch_keys) if batch_keys is not None else self.batch
        ntb = self.non_tensor_batch
        if non_tensor_batch_keys is not None:
            ntb = {k: v for k, v in ntb.items() if k in non_tensor_batch_keys}
        meta = self.meta_info
        if meta_info_keys is not None:
            meta = {k: v for k, v in meta.items() if k in meta_info_keys}
        if deepcopy:
            ntb, meta = copy.deepcopy(ntb), copy.deepcopy(meta)
        return DataProto(batch=batch, non_tensor_batch=dict(ntb), meta_info=dict(meta))

    def select_idxs(self, idxs):
        if isinstance(idxs, torch.Tensor) and idxs.dtype == torch.bool:
            idxs = idxs.nonzero().flatten()
        idxs = [int(i) for i in idxs]
        batch = self.batch[torch.tensor(idxs, dtype=torch.long)] if self.batch is not None else None
        ntb = {k: _index_list(v, idxs) for k, v in self.non_tensor_batch.items()}
        return DataProto(batch=batch, non_tensor_batch=ntb, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        s = slice(start, end, step)
        batch = self.batch[s] if self.batch is not None else None
        return DataProto(batch=batch, non_tensor_batch={k: v[s] for k, v in self.non_tensor_batch.items()},
                         meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None):
        """Remove keys from self and return them as a new DataProto (verl ``DataProto.pop``)."""
        tensors = {k: self.batch.pop(k) for k in (batch_keys or [])}
        non_tensors = {k: self.non_tensor_batch.pop(k) for k in (non_tensor_batch_keys or [])}
        meta = {k: self.meta_info.pop(k) for k in (meta_info_keys or [])}
        return DataProto.from_dict(tensors, non_tensors, meta)

    def union(self, other: "DataProto") -> "DataProto":
        """Merge ``other`` into self; overlapping keys must hold equal values (verl ``union``)."""
        if self.batch is None or len(self.batch.keys()) == 0:
            self.batch = other.batch
        elif other.batch is not None:
            assert self.batch.batch_size == other.batch.batch_size, "union: batch sizes differ"
            for k, v in other.batch.items():
                if k in self.batch:
                    assert torch.equal(self.batch[k], v), f"union: conflicting values for {k!r}"
                else:
                    self.batch[k] = v
        for k, v in other.non_tensor_batch.items():
            assert k not in self.non_tensor_batch or self.non_tensor_batch[k] == v, f"union: conflict on {k!r}"
            self.non_tensor_batch[k] = v
        for k, v in other.meta_info.items():
            assert k not in self.meta_info or self.meta_info[k] == v, f"union: conflict on meta {k!r}"
            self.meta_info[k] = v
        return self

    # ---------------------------------------------------------------- split / merge
    def is_padding_enabled(self):
        return self.meta_info.get(DataProtoConfig.auto_padding_key, False) or DataProtoConfig.auto_padding

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split along dim 0 into ``chunks`` pieces (torch.chunk sizes); meta_info is shared."""
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, f"only support equal chunk. Got size {len(self)} and chunk {chunks}."
        size = -(-len(self) // chunks)
        out = [self[i * size:(i + 1) * size] for i in range(chunks)]
        return out

    def split(self, split_size: int) -> List["DataProto"]:
        return [self[i:i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concatenate along dim 0. ``meta_info['metrics']`` from all workers become lists."""
        batches = [d.batch for d in data if d.batch is not None and len(d.batch.keys()) > 0]
        batch = TensorDict.cat(batches) if batches else None
        ntb = {k: [x for d in data for x in d.non_tensor_batch[k]] for k in data[0].non_tensor_batch}
        meta, all_metrics = {}, []
        for d in data:
            for k, v in d.meta_info.items():
                if k == "metrics":
                    all_metrics.extend(v if isinstance(v, list) else [v])
                else:
                    assert meta.get(k, v) == v, f"Conflicting values for meta_info key '{k}'"
                    meta[k] = v
        if all_metrics:
            meta["metrics"] = list_of_dict_to_dict_of_list(all_metrics)
        return DataProto(batch=batch, non_tensor_batch=ntb, meta_info=meta)

    def reorder(self, indices: torch.Tensor):
        """In-place row permutation (used by ``balance_batch``)."""
        new = self.select_idxs(indices)
        self.batch, self.non_tensor_batch = new.batch, new.non_tensor_batch

    def repeat(self, repeat_times=2, interleave=True):
        """interleave=True: [a, b] -> [a, a, b, b] (the rollout.n layout); False: [a, b, a, b]."""
        n = len(self)
        if interleave:
            idxs = [i for i in range(n) for _ in range(repeat_times)]
        else:
            idxs = [i for _ in range(repeat_times) for i in range(n)]
        return self.select_idxs(idxs)


def pad_dataproto_to_divisor(data: DataProto, size_divisor: int):
    """Pad by repeating rows from the front until len % size_divisor == 0. Returns (padded, pad_size)."""
    if len(data) % size_divisor == 0:
        return data, 0
    pad_size = size_divisor - len(data) % size_divisor
    pads, remaining = [], pad_size
    while remaining > 0:
        take = min(remaining, len(data))
        pads.append(data[:take])
        remaining -= take
    padded = DataProto.concat([data] + pads)
    padded.meta_info = data.meta_info
    return padded, pad_size


def unpad_dataproto(data: DataProto, pad_size: int) -> DataProto:
    return data[:-pad_size] if pad_size else data
