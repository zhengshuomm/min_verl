#!/usr/bin/env python3
"""Standalone book listing — Section 4.2: DataProto: Three Drawers of a Batch
Pinned reference: verl v0.9.1 (commit 1876b06).
Expected stdout:
    pad8 = 0 | rank3 rows = 640 | pad6 = 4 | rank0/rank5 rows = 854 850
"""

# Combine batch & non_tensor_batch as dict of lists; meta is meta_info
class MiniProto:
    def __init__(self, cols, meta=None):
        self.cols, self.meta = cols, dict(meta or {})

    def __len__(self):
        return len(next(iter(self.cols.values())))

    def _map(self, f):  # transform each column, preserve meta_info
        return MiniProto({k: f(v) for k, v in self.cols.items()},
                         self.meta)

    def repeat(self, repeat_times, interleave=True):
        # True: a,a,b,b; False: a,b,a,b
        if interleave:
            return self._map(
                lambda v: [x for x in v for _ in range(repeat_times)]
            )
        return self._map(lambda v: v * repeat_times)

    def chunk(self, chunks):
        assert len(self) % chunks == 0, "only support equal chunk"
        s = len(self) // chunks
        return [
            self._map(lambda v, i=i: v[i * s:(i + 1) * s])
            for i in range(chunks)
        ]

    @staticmethod
    def concat(parts):
        cols = {
            c: [x for p in parts for x in p.cols[c]]
            for c in parts[0].cols
        }
        return MiniProto(cols, parts[0].meta)

def call_dp_compute_proto(world_size, data, fn):
    pad = (world_size - len(data) % world_size) % world_size
    if pad:  # DataProto.padding: duplicate last row pad times
        tail = data._map(lambda v: v[-1:] * pad)
        data = MiniProto.concat([data, tail])
    shards = data.chunk(world_size)                   # dispatch
    # execute: .remote + ray.get in verl
    outs = [fn(rank, sh) for rank, sh in enumerate(shards)]
    out = MiniProto.concat(outs)                      # collect
    return out._map(lambda v: v[:len(v) - pad]), pad  # strip pad rows

def worker(rank, shard):
    cols = {"prompt": shard.cols["prompt"], "rank": [rank] * len(shard)}
    return MiniProto(cols, shard.meta)

rollout_n = 5
gen_batch = MiniProto(
    {"prompt": list(range(1024))}, {"temperature": 1.0}
)
batch = gen_batch.repeat(rollout_n, interleave=True)
out8, pad8 = call_dp_compute_proto(8, batch, worker)
out6, pad6 = call_dp_compute_proto(6, batch, worker)
r3 = out8.cols["rank"].count(3)
r0, r5 = out6.cols["rank"].count(0), out6.cols["rank"].count(5)
print("pad8 =", pad8, "| rank3 rows =", r3, "| pad6 =", pad6,
      "| rank0/rank5 rows =", r0, r5)

# Verification assertions
assert len(batch) == 5120
assert batch.cols['prompt'][:6] == [0, 0, 0, 0, 0, 1]
assert pad8 == 0 and r3 == 640
assert out8.cols['prompt'] == batch.cols['prompt']
assert pad6 == 4 and len(out6) == 5120
assert r0 == 854 and r5 == 850
assert out6.cols['prompt'] == batch.cols['prompt']
assert out6.meta == {'temperature': 1.0}
