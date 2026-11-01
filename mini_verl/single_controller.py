"""The single-controller programming model: Worker, WorkerGroup, @register, Dispatch / Execute.

Mirrors ``verl/single_controller/base/decorator.py`` (``Dispatch``, ``Execute``, ``register``,
``dispatch_*`` / ``collect_*`` functions), ``verl/single_controller/base/worker.py`` (``Worker``),
``verl/single_controller/base/worker_group.py`` (``ResourcePool``, ``ClassWithInitArgs``,
``WorkerGroup._bind_worker_method``) and ``verl/single_controller/ray/base.py`` (``func_generator``,
``execute_all``, ``spawn``, ``create_colocated_worker_cls``).

In verl every worker is a Ray actor (one GPU process) and the driver calls ``wg.method(data)``;
the call is split by a dispatch function, run on every actor, and merged by a collect function.
mini_verl keeps the exact same control flow but the "actors" are N in-process objects, each run in
its own Python thread so that collectives (``all_reduce``) can block like NCCL does.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List

import torch

from .protocol import DataProto, _padding_size_key, pad_dataproto_to_divisor

MAGIC_ATTR = "attrs_3141562937"  # same magic attribute name as verl


class Dispatch(Enum):
    RANK_ZERO = 0
    ONE_TO_ALL = 1
    ALL_TO_ALL = 2
    DP_COMPUTE = 3
    DP_COMPUTE_PROTO = 4


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1


# ----------------------------------------------------------------------------- dispatch / collect
def dispatch_one_to_all(worker_group, *args, **kwargs):
    """Broadcast: every rank receives the same arguments."""
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def dispatch_dp_compute(worker_group, *args, **kwargs):
    """Arguments are already per-rank lists of length world_size."""
    for arg in list(args) + list(kwargs.values()):
        assert isinstance(arg, (tuple, list)) and len(arg) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    """Chunk every DataProto argument into world_size shards (auto-padding if enabled)."""
    padding_size = None

    def _pad_and_split(obj: DataProto):
        nonlocal padding_size
        assert isinstance(obj, DataProto)
        if obj.is_padding_enabled():
            obj, pad = pad_dataproto_to_divisor(obj, worker_group.world_size)
            assert padding_size in (None, pad), "all DataProto args must share the same length"
            padding_size = pad
        return obj.chunk(chunks=worker_group.world_size)

    splitted_args = [_pad_and_split(arg) for arg in args]
    splitted_kwargs = {k: _pad_and_split(v) for k, v in kwargs.items()}
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size
    return splitted_args, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    output = collect_dp_compute(worker_group, output)
    assert all(isinstance(o, DataProto) for o in output), "expecting DataProto outputs"
    return DataProto.concat(output)


DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.RANK_ZERO: {"dispatch_fn": dispatch_all_to_all, "collect_fn": collect_all_to_all},
    Dispatch.ONE_TO_ALL: {"dispatch_fn": dispatch_one_to_all, "collect_fn": collect_all_to_all},
    Dispatch.ALL_TO_ALL: {"dispatch_fn": dispatch_all_to_all, "collect_fn": collect_all_to_all},
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {"dispatch_fn": dispatch_dp_compute_data_proto,
                                "collect_fn": collect_dp_compute_data_proto},
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def get_predefined_execute_fn(execute_mode):
    return {Execute.ALL: {"execute_fn_name": "execute_all"},
            Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"}}[execute_mode]


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True):
    """Tag a worker method with how the driver should dispatch / execute / collect it.

    Like verl, the decorator does not change the method; it only stores
    ``{"dispatch_mode", "execute_mode", "blocking"}`` under ``MAGIC_ATTR`` so that
    ``WorkerGroup._bind_worker_method`` can generate the driver-side method.
    """
    assert isinstance(dispatch_mode, (Dispatch, dict)) and isinstance(execute_mode, Execute)

    def decorator(func):
        setattr(func, MAGIC_ATTR, {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode,
                                   "blocking": blocking})
        return func

    return decorator


# ----------------------------------------------------------------------------- resources & workers
class ResourcePool:
    """``process_on_nodes=[2]`` = one node with 2 GPUs (here: 2 in-process worker slots)."""

    def __init__(self, process_on_nodes=None, n_gpus_per_node=8):
        self._store = list(process_on_nodes or [])
        self.n_gpus_per_node = n_gpus_per_node

    @property
    def world_size(self):
        return sum(self._store)

    def __call__(self):
        return self._store


class ClassWithInitArgs:
    """Deferred construction: remember (cls, args, kwargs) and build the worker later on each rank."""

    def __init__(self, cls, *args, **kwargs):
        self.cls, self.args, self.kwargs = cls, args, kwargs

    def __call__(self):
        return self.cls(*self.args, **self.kwargs)


class ProcessGroup:
    """An in-process stand-in for a torch.distributed / NCCL process group.

    Each rank runs in its own thread; ``all_reduce`` blocks on a barrier until every rank has
    contributed, then every rank sums the contributions in rank order -- so all ranks end up with
    bit-identical results, exactly the property NCCL all-reduce gives data-parallel training.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=600)
        self._slots: List[Any] = [None] * world_size

    def all_reduce(self, rank: int, tensors: List[torch.Tensor], op: str = "sum"):
        if self.world_size > 1:
            self._slots[rank] = [t.detach().clone() for t in tensors]
            self._barrier.wait()
            for j, t in enumerate(tensors):
                total = self._slots[0][j].clone()
                for r in range(1, self.world_size):
                    total += self._slots[r][j]
                t.copy_(total)
            self._barrier.wait()  # nobody may overwrite its slot before everyone has read it
        if op == "mean":
            for t in tensors:
                t.div_(self.world_size)

    def abort(self):
        self._barrier.abort()


_worker_env = threading.local()


@contextmanager
def _worker_context(rank: int, world_size: int, group: ProcessGroup):
    """Plays the role of the RANK / WORLD_SIZE env vars Ray sets before a verl Worker starts."""
    _worker_env.info = (rank, world_size, group)
    try:
        yield
    finally:
        _worker_env.info = None


class Worker:
    """Base class of every worker (verl/single_controller/base/worker.py ``Worker``)."""

    def __init__(self):
        info = getattr(_worker_env, "info", None) or (0, 1, ProcessGroup(1))
        self._rank, self._world_size, self.process_group = info

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def all_reduce(self, tensors: List[torch.Tensor], op: str = "sum"):
        self.process_group.all_reduce(self._rank, tensors, op)


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    """Build the driver-side method: dispatch -> execute on workers -> collect -> unpad."""

    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        padding_count = kwargs.pop(_padding_size_key, 0)
        output = execute_fn(method_name, *args, **kwargs)
        output = collect_fn(self, output)
        if padding_count > 0:
            if isinstance(output, DataProto) and len(output) > 0:
                output = output[:-padding_count]
            elif isinstance(output, list):
                output = output[:-padding_count]
        return output

    func.__name__ = method_name
    return func


class WorkerGroup:
    """A handle to ``world_size`` workers of the same class, driven from the single controller."""

    def __init__(self, resource_pool: ResourcePool = None, ray_cls_with_init: ClassWithInitArgs = None,
                 workers: List[Worker] = None):
        if workers is None:
            world_size = resource_pool.world_size
            group = ProcessGroup(world_size)
            workers = []
            for rank in range(world_size):
                with _worker_context(rank, world_size, group):
                    workers.append(ray_cls_with_init())
        self._workers = workers
        self.ray_cls_with_init = ray_cls_with_init
        self._bind_worker_method(type(workers[0]), func_generator)

    @property
    def world_size(self):
        return len(self._workers)

    @property
    def workers(self):
        return self._workers

    def _bind_worker_method(self, user_defined_cls, func_generator):
        """For every @register-ed method of the worker class, attach a driver-side twin."""
        method_names = []
        for method_name in dir(user_defined_cls):
            method = getattr(user_defined_cls, method_name, None)
            if not callable(method) or not hasattr(method, MAGIC_ATTR):
                continue
            attrs = getattr(method, MAGIC_ATTR)
            dispatch_mode = attrs["dispatch_mode"]
            if isinstance(dispatch_mode, Dispatch):
                fns = get_predefined_dispatch_fn(dispatch_mode)
            else:  # user-defined {"dispatch_fn", "collect_fn"}
                fns = dispatch_mode
            execute_fn = getattr(self, get_predefined_execute_fn(attrs["execute_mode"])["execute_fn_name"])
            setattr(self, method_name, func_generator(self, method_name, fns["dispatch_fn"], fns["collect_fn"],
                                                      execute_fn, attrs["blocking"]))
            method_names.append(method_name)
        return method_names

    # ------------------------------------------------------------------ execution
    def execute_all(self, method_name: str, *args, **kwargs):
        """Run ``method_name`` on every rank. Per-rank lists of length world_size are scattered;
        anything else is broadcast (same rule as verl's ``RayWorkerGroup.execute_all_async``)."""
        n = self.world_size
        scatter = all(isinstance(a, (list, tuple)) and len(a) == n for a in list(args) + list(kwargs.values()))

        def run(rank):
            a = [arg[rank] for arg in args] if scatter else list(args)
            kw = {k: v[rank] for k, v in kwargs.items()} if scatter else dict(kwargs)
            try:
                return getattr(self._workers[rank], method_name)(*a, **kw)
            except BaseException:
                self._workers[rank].process_group.abort()  # wake up ranks blocked in all_reduce
                raise

        if n == 1:
            return [run(0)]
        with ThreadPoolExecutor(max_workers=n) as pool:  # like ray.get on n remote calls
            futures = [pool.submit(run, r) for r in range(n)]
            return [f.result() for f in futures]

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return getattr(self._workers[0], method_name)(*args, **kwargs)

    def spawn(self, prefix_set) -> Dict[str, "WorkerGroup"]:
        """Split a colocated group into one WorkerGroup per role: ``actor_rollout_X`` becomes ``X``."""
        out = {}
        for prefix in prefix_set:
            wg = WorkerGroup(workers=self._workers, ray_cls_with_init=self.ray_cls_with_init)
            for name in dir(wg):
                if name.startswith(prefix + "_"):
                    setattr(wg, name[len(prefix) + 1:], getattr(wg, name))
            out[prefix] = wg
        return out


def create_colocated_worker_cls(class_dict: Dict[str, ClassWithInitArgs]) -> ClassWithInitArgs:
    """Fuse several roles into ONE worker class that lives on the same GPUs (verl colocation).

    ``{"actor_rollout": ..., "critic": ...}`` -> class ``WorkerDict`` that owns one instance of each
    role in ``self.worker_dict`` and exposes ``actor_rollout_generate_sequences``,
    ``critic_compute_values``, ... each carrying the original @register attributes.
    """

    class WorkerDict(Worker):
        def __init__(self):
            super().__init__()
            self.worker_dict = {key: cia() for key, cia in class_dict.items()}

    for key, cia in class_dict.items():
        for method_name in dir(cia.cls):
            method = getattr(cia.cls, method_name, None)
            if not callable(method) or not hasattr(method, MAGIC_ATTR):
                continue

            def generate_function(name, key=key):
                def func(self, *args, **kwargs):
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)
                return func

            func = generate_function(method_name)
            setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
            setattr(WorkerDict, f"{key}_{method_name}", func)
    return ClassWithInitArgs(WorkerDict)
