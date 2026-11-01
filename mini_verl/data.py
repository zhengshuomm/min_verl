"""The toy task: two-number addition ("23+48=" -> "71<eos>"), a character tokenizer and an RL dataset.

Mirrors the role of ``verl/utils/dataset/rl_dataset.py`` (``RLHFDataset``, ``collate_fn``): every
item carries LEFT-padded ``input_ids`` / ``attention_mask`` / ``position_ids`` plus the non-tensor
fields verl's reward function reads (``data_source``, ``reward_model.ground_truth``).
"""

from typing import Dict, List, Tuple

import torch

from .models import compute_position_id_with_mask

VOCAB = ["<pad>", "<eos>"] + [str(d) for d in range(10)] + ["+", "=", "<c>", "</c>", "<r>", "</r>"]
SPECIAL_TOKENS = ("<pad>", "<eos>")


class Tokenizer:
    """Character-level tokenizer; tags like ``<c>`` are single tokens."""

    def __init__(self):
        self.vocab = VOCAB
        self.token_to_id = {t: i for i, t in enumerate(VOCAB)}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    def encode(self, text: str) -> List[int]:
        ids, i = [], 0
        while i < len(text):
            for tag in ("</c>", "</r>", "<c>", "<r>", "<eos>", "<pad>"):
                if text.startswith(tag, i):
                    ids.append(self.token_to_id[tag])
                    i += len(tag)
                    break
            else:
                ids.append(self.token_to_id[text[i]])
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        toks = [self.vocab[int(i)] for i in ids]
        return "".join(t for t in toks if not (skip_special_tokens and t in SPECIAL_TOKENS))


def make_addition_pairs(operand_max: int, val_size: int, seed: int) -> Tuple[List, List]:
    """All (a, b) with 0 <= a, b <= operand_max, shuffled; the first val_size pairs are held out."""
    pairs = [(a, b) for a in range(operand_max + 1) for b in range(operand_max + 1)]
    perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [pairs[i] for i in perm]
    return pairs[val_size:], pairs[:val_size]


def left_pad(seqs: List[List[int]], length: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), length), dtype=torch.long)
    for i, s in enumerate(seqs):
        assert len(s) <= length, f"prompt of {len(s)} tokens > max_prompt_length={length}"
        if s:
            ids[i, length - len(s):] = torch.tensor(s)
            mask[i, length - len(s):] = 1
    return ids, mask


class RLHFDataset:
    """List of addition prompts. ``__getitem__`` returns one row; ``collate_fn`` batches rows."""

    def __init__(self, pairs, tokenizer: Tokenizer, max_prompt_length: int, data_source: str):
        self.pairs, self.tokenizer = pairs, tokenizer
        self.max_prompt_length, self.data_source = max_prompt_length, data_source

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i) -> Dict:
        a, b = self.pairs[i]
        prompt = f"{a}+{b}="
        ids, mask = left_pad([self.tokenizer.encode(prompt)], self.max_prompt_length, self.tokenizer.pad_token_id)
        return {"input_ids": ids[0], "attention_mask": mask[0], "position_ids": compute_position_id_with_mask(mask)[0],
                "raw_prompt": prompt, "data_source": self.data_source,
                "reward_model": {"style": "rule", "ground_truth": str(a + b)}}


def collate_fn(rows: List[Dict]) -> Dict:
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def dataloader(dataset: RLHFDataset, batch_size: int, seed: int, shuffle: bool = True):
    """Endless iterator over shuffled batches (a new permutation each epoch), like a DataLoader."""
    g = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(dataset), generator=g).tolist() if shuffle else list(range(len(dataset)))
        for s in range(0, len(order) - batch_size + 1, batch_size):
            yield collate_fn([dataset[i] for i in order[s:s + batch_size]])
