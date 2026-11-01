"""Lab 1 -- DataProto: the batch that flows between the driver and the workers.

Run:  python3 -m mini_verl.examples.lab1_dataproto

Builds the first RL batch of the toy task exactly as RayPPOTrainer.fit does: tensors in
``batch`` (left-padded prompts), per-row Python objects in ``non_tensor_batch``, run-wide values
in ``meta_info``; then the operations the trainer and the dispatcher use.
"""

import torch

from mini_verl.data import RLHFDataset, Tokenizer, collate_fn
from mini_verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto


def show(name, d: DataProto):
    print(f"{name}: len={len(d)} batch={ {k: tuple(v.shape) for k, v in d.batch.items()} if d.batch else None}")
    for k, v in d.non_tensor_batch.items():
        print(f"    non_tensor[{k!r}] = {v}")


def main():
    tok = Tokenizer()
    ds = RLHFDataset([(7, 5), (23, 48), (99, 9)], tok, max_prompt_length=6, data_source="arithmetic/addition")
    batch = DataProto.from_single_dict(collate_fn([ds[i] for i in range(3)]), meta_info={"temperature": 1.0})
    batch.non_tensor_batch["uid"] = ["p0", "p1", "p2"]
    print("== from_single_dict: 3 prompts, LEFT-padded to max_prompt_length=6")
    for i in range(3):
        print(f"  {batch.non_tensor_batch['raw_prompt'][i]:>7}  input_ids={batch.batch['input_ids'][i].tolist()}  "
              f"attention_mask={batch.batch['attention_mask'][i].tolist()}  position_ids={batch.batch['position_ids'][i].tolist()}")
    print("  meta_info =", batch.meta_info)

    print("\n== pop: the trainer moves the prompt tensors into gen_batch (_get_gen_batch)")
    gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
    show("gen_batch", gen_batch)
    print("  batch keeps:", list(batch.non_tensor_batch))

    print("\n== repeat(n=2, interleave=True): every prompt twice, side by side (rollout.n)")
    rep = gen_batch.repeat(repeat_times=2, interleave=True)
    print("  uid order, interleave=True :", batch.repeat(2, interleave=True).non_tensor_batch["uid"])
    print("  uid order, interleave=False:", batch.repeat(2, interleave=False).non_tensor_batch["uid"])
    print("  gen_batch rows -> prompts  :", [tok.decode(r) for r in rep.batch["input_ids"].tolist()])

    print("\n== union: attach new columns (e.g. the rollout output) to the same rows")
    rep.union(DataProto.from_dict({"responses": torch.arange(6 * 4).view(6, 4)}))
    show("after union", rep)

    print("\n== chunk(2) / concat: what Dispatch.DP_COMPUTE_PROTO does for 2 ranks")
    parts = rep.chunk(2)
    for r, p in enumerate(parts):
        print(f"  rank {r} gets responses[:,0] = {p.batch['responses'][:, 0].tolist()}")
    back = DataProto.concat(parts)
    print("  concat == original:", torch.equal(back.batch["responses"], rep.batch["responses"]))

    print("\n== select / slicing / index lists")
    print("  rep[1:3].batch keys :", list(rep[1:3].batch.keys()), "len", len(rep[1:3]))
    print("  rep.select(['responses']) keys:", list(rep.select(batch_keys=["responses"]).batch.keys()))
    print("  rep[[5, 0]].responses[:,0]   :", rep[[5, 0]].batch["responses"][:, 0].tolist())

    print("\n== pad_dataproto_to_divisor: 3 rows for 2 ranks")
    padded, pad = pad_dataproto_to_divisor(gen_batch, 2)
    print(f"  padded len={len(padded)} pad_size={pad}; unpad -> len={len(unpad_dataproto(padded, pad))}")

    print("\n== concat merges worker metrics into lists")
    m = DataProto.concat([DataProto(meta_info={"metrics": {"actor/pg_loss": [0.1]}}),
                          DataProto(meta_info={"metrics": {"actor/pg_loss": [0.3]}})])
    print("  ", m.meta_info)


if __name__ == "__main__":
    main()
