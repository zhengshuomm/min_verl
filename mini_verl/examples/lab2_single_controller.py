"""Lab 2 -- The single controller: @register, Dispatch / Execute, WorkerGroup, colocation.

Run:  python3 -m mini_verl.examples.lab2_single_controller

The driver calls ``wg.method(data)`` once; the dispatch function decides what each rank gets,
every rank runs the method (in its own thread here, in its own Ray actor / GPU in verl), and the
collect function merges the results.
"""

import torch

from mini_verl.protocol import DataProto
from mini_verl.single_controller import (ClassWithInitArgs, Dispatch, Execute, ResourcePool, Worker, WorkerGroup,
                                         create_colocated_worker_cls, register)


class DemoWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, name):
        return f"rank {self.rank}/{self.world_size} loaded {name}"

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute(self, data: DataProto) -> DataProto:
        rows = data.batch["row"].tolist()
        print(f"    rank {self.rank} received rows {rows}")
        return DataProto.from_dict({"row": data.batch["row"], "squared": data.batch["row"] ** 2,
                                    "rank": torch.full((len(data),), self.rank)})

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def save_checkpoint(self):
        return f"only rank {self.rank} saved"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def all_reduce_demo(self):
        t = torch.tensor([10.0 * (self.rank + 1)])
        self.all_reduce([t], op="mean")
        return float(t)


class CriticDemo(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, name):
        return f"critic on rank {self.rank} loaded {name}"


def main():
    wg = WorkerGroup(ResourcePool([2]), ClassWithInitArgs(DemoWorker))
    print("== ONE_TO_ALL: the same arguments go to every rank")
    print("  ", wg.init_model("tiny-policy"))

    print("\n== DP_COMPUTE_PROTO: 6 rows split into 2 contiguous shards, results concatenated")
    out = wg.compute(DataProto.from_dict({"row": torch.arange(6)}))
    print("  collected rows   :", out.batch["row"].tolist())
    print("  computed by rank :", out.batch["rank"].tolist())

    print("\n== DP_COMPUTE_PROTO with auto_padding: 5 rows on 3 ranks (one pad row, removed after)")
    wg3 = WorkerGroup(ResourcePool([3]), ClassWithInitArgs(DemoWorker))
    out = wg3.compute(DataProto.from_dict({"row": torch.arange(5)}, auto_padding=True))
    print("  collected rows   :", out.batch["row"].tolist(), "squared", out.batch["squared"].tolist())

    print("\n== Execute.RANK_ZERO and an all_reduce between ranks")
    print("  ", wg.save_checkpoint())
    print("   all_reduce(mean) of [10, 20] on every rank:", wg.all_reduce_demo())

    print("\n== create_colocated_worker_cls + spawn: two roles share the same 2 'GPUs'")
    fused = create_colocated_worker_cls({"actor_rollout": ClassWithInitArgs(DemoWorker),
                                         "critic": ClassWithInitArgs(CriticDemo)})
    wg_dict = WorkerGroup(ResourcePool([2]), fused)
    print("   fused methods:", sorted(m for m in dir(wg_dict) if m.startswith(("actor_rollout_", "critic_"))))
    spawned = wg_dict.spawn(prefix_set=["actor_rollout", "critic"])
    print("  ", spawned["actor_rollout"].init_model("policy"))
    print("  ", spawned["critic"].init_model("value-model"))


if __name__ == "__main__":
    main()
