import unittest

import torch

from mini_verl.protocol import DataProto
from mini_verl.single_controller import (MAGIC_ATTR, ClassWithInitArgs, Dispatch, Execute, ResourcePool, Worker,
                                         WorkerGroup, create_colocated_worker_cls, register)


class EchoWorker(Worker):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.seen = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def whoami(self, tag):
        return (self.rank, self.world_size, tag)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def double(self, data: DataProto) -> DataProto:
        self.seen = data.batch["x"].tolist()
        return DataProto.from_dict({"y": data.batch["x"] * self.scale, "rank": torch.full((len(data),), self.rank)})

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def leader(self):
        return self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def allreduce(self):
        t = torch.tensor([float(self.rank + 1)])
        self.all_reduce([t], op="sum")
        return float(t)


class OtherWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def whoami(self, tag):
        return ("other", self.rank, tag)


def group(n, cls=EchoWorker, **kw):
    return WorkerGroup(ResourcePool([n]), ClassWithInitArgs(cls, **kw))


class TestSingleController(unittest.TestCase):
    def test_register_only_tags(self):
        attrs = getattr(EchoWorker.double, MAGIC_ATTR)
        self.assertEqual(attrs["dispatch_mode"], Dispatch.DP_COMPUTE_PROTO)
        self.assertEqual(attrs["execute_mode"], Execute.ALL)

    def test_one_to_all(self):
        self.assertEqual(group(3).whoami("hi"), [(0, 3, "hi"), (1, 3, "hi"), (2, 3, "hi")])

    def test_dp_compute_proto_roundtrip(self):
        wg = group(2, scale=10)
        out = wg.double(DataProto.from_dict({"x": torch.arange(6)}))
        self.assertEqual(out.batch["y"].tolist(), [0, 10, 20, 30, 40, 50])
        self.assertEqual(out.batch["rank"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual([w.seen for w in wg.workers], [[0, 1, 2], [3, 4, 5]])

    def test_dp_compute_proto_auto_padding(self):
        wg = group(3)
        out = wg.double(DataProto.from_dict({"x": torch.arange(5)}, auto_padding=True))
        self.assertEqual(out.batch["y"].tolist(), [0, 2, 4, 6, 8])          # padding removed again
        self.assertEqual([w.seen for w in wg.workers], [[0, 1], [2, 3], [4, 0]])  # rank 2 got a pad row
        with self.assertRaises(AssertionError):
            wg.double(DataProto.from_dict({"x": torch.arange(5)}))           # no auto padding -> must divide

    def test_rank_zero_and_all_reduce(self):
        wg = group(4)
        self.assertEqual(wg.leader(), 0)
        self.assertEqual(wg.allreduce(), [10.0] * 4)  # 1+2+3+4 on every rank

    def test_colocated_worker_and_spawn(self):
        cls = create_colocated_worker_cls({"actor": ClassWithInitArgs(EchoWorker, scale=3),
                                           "critic": ClassWithInitArgs(OtherWorker)})
        wg = WorkerGroup(ResourcePool([2]), cls)
        self.assertTrue(hasattr(wg, "actor_double") and hasattr(wg, "critic_whoami"))
        spawned = wg.spawn(["actor", "critic"])
        self.assertEqual(spawned["critic"].whoami("t"), [("other", 0, "t"), ("other", 1, "t")])
        self.assertEqual(spawned["actor"].double(DataProto.from_dict({"x": torch.arange(2)})).batch["y"].tolist(), [0, 3])
        inner = wg.workers[1].worker_dict
        self.assertEqual((inner["actor"].rank, inner["critic"].rank), (1, 1))  # roles share the same rank / "GPU"


if __name__ == "__main__":
    unittest.main()
