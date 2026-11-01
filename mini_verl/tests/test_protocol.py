import unittest

import torch

from mini_verl.protocol import DataProto, DataProtoConfig, pad_dataproto_to_divisor, unpad_dataproto


def make(n=4):
    return DataProto.from_dict(tensors={"x": torch.arange(n), "y": torch.arange(2 * n).view(n, 2)},
                               non_tensors={"uid": [f"u{i}" for i in range(n)]}, meta_info={"temperature": 1.0})


class TestDataProto(unittest.TestCase):
    def test_from_dict_len_and_slicing(self):
        d = make()
        self.assertEqual(len(d), 4)
        self.assertEqual(d[1:3].batch["x"].tolist(), [1, 2])
        self.assertEqual(d[1:3].non_tensor_batch["uid"], ["u1", "u2"])
        self.assertEqual(d[[3, 0]].batch["y"].tolist(), [[6, 7], [0, 1]])
        self.assertEqual(d[1:3].meta_info, {"temperature": 1.0})
        with self.assertRaises(AssertionError):
            DataProto.from_dict(tensors={"a": torch.zeros(2), "b": torch.zeros(3)})

    def test_from_single_dict_splits_tensors(self):
        d = DataProto.from_single_dict({"x": torch.zeros(2), "s": ["a", "b"]})
        self.assertEqual(list(d.batch.keys()), ["x"])
        self.assertEqual(d.non_tensor_batch["s"], ["a", "b"])

    def test_select_and_pop(self):
        d = make()
        s = d.select(batch_keys=["x"], non_tensor_batch_keys=[])
        self.assertEqual(list(s.batch.keys()), ["x"])
        self.assertEqual(s.non_tensor_batch, {})
        popped = d.pop(batch_keys=["y"], meta_info_keys=["temperature"])
        self.assertEqual(list(d.batch.keys()), ["x"])
        self.assertEqual(list(popped.batch.keys()), ["y"])
        self.assertEqual(popped.meta_info, {"temperature": 1.0})
        self.assertEqual(d.meta_info, {})

    def test_union(self):
        a, b = make(), make()
        b.pop(batch_keys=["x"])
        b.batch["z"] = torch.ones(4)
        a.union(b)
        self.assertEqual(sorted(a.batch.keys()), ["x", "y", "z"])
        c = make()
        c.batch["x"] = torch.zeros(4, dtype=torch.long)
        with self.assertRaises(AssertionError):
            a.union(c)  # conflicting values for the same key

    def test_union_into_empty_batch(self):
        d = make()
        gen = d.pop(batch_keys=["x", "y"])  # what the trainer does with input_ids etc.
        d = d.repeat(2)
        d.union(gen.repeat(2))
        self.assertEqual(d.batch["x"].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_chunk_concat_roundtrip(self):
        d = make(6)
        chunks = d.chunk(3)
        self.assertEqual([c.batch["x"].tolist() for c in chunks], [[0, 1], [2, 3], [4, 5]])
        back = DataProto.concat(chunks)
        self.assertTrue(torch.equal(back.batch["y"], d.batch["y"]))
        self.assertEqual(back.non_tensor_batch["uid"], d.non_tensor_batch["uid"])
        with self.assertRaises(AssertionError):
            make(5).chunk(2)
        self.assertEqual([len(c) for c in d.split(4)], [4, 2])

    def test_concat_merges_metrics(self):
        parts = [DataProto(meta_info={"metrics": {"loss": [1.0, 2.0]}}), DataProto(meta_info={"metrics": {"loss": [3.0]}})]
        self.assertEqual(DataProto.concat(parts).meta_info["metrics"], {"loss": [1.0, 2.0, 3.0]})

    def test_repeat(self):
        d = make(2)
        self.assertEqual(d.repeat(3, interleave=True).batch["x"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(d.repeat(3, interleave=False).non_tensor_batch["uid"], ["u0", "u1"] * 3)

    def test_reorder(self):
        d = make()
        d.reorder(torch.tensor([3, 2, 1, 0]))
        self.assertEqual(d.batch["x"].tolist(), [3, 2, 1, 0])
        self.assertEqual(d.non_tensor_batch["uid"], ["u3", "u2", "u1", "u0"])

    def test_pad_unpad(self):
        d = make(5)
        padded, pad = pad_dataproto_to_divisor(d, 4)
        self.assertEqual((len(padded), pad), (8, 3))
        self.assertEqual(padded.batch["x"].tolist(), [0, 1, 2, 3, 4, 0, 1, 2])  # repeats from the front
        self.assertEqual(unpad_dataproto(padded, pad).batch["x"].tolist(), [0, 1, 2, 3, 4])
        same, pad = pad_dataproto_to_divisor(d, 5)
        self.assertEqual(pad, 0)

    def test_auto_padding_flag(self):
        d = DataProto.from_dict(tensors={"x": torch.arange(5)}, auto_padding=True)
        self.assertTrue(d.is_padding_enabled())
        self.assertFalse(make().is_padding_enabled())
        self.assertFalse(DataProtoConfig.auto_padding)


if __name__ == "__main__":
    unittest.main()
