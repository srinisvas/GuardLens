import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer

def conversation_to_text(turns: List[Dict]) -> str:

    parts = []
    for t in turns:
        parts.append(f"[{t['role'].upper()}]: {t['text']}")
    return " ".join(parts)

class JailbreakDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int = 512):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item  = self.data[idx]
        text  = conversation_to_text(item["turns"])
        label = item["label"]

        max_sr = max(
            (t.get("surface_risk", 0.0) for t in item["turns"] if t["role"] == "user"),
            default=0.0,
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float),
            "max_surface_risk": torch.tensor(max_sr, dtype=torch.float),
            "difficulty":     item.get("difficulty", "unknown"),
            "family":         item.get("family", "unknown"),
            "pair_id":        item.get("pair_id", ""),
        }

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def pair_level_split(
    data: List[Dict],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    random.seed(seed)

    # Group by pair_id
    pair_groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in data:
        pid = record.get("pair_id") or record["conversation_id"]
        pair_groups[pid].append(record)

    group_keys = list(pair_groups.keys())
    random.shuffle(group_keys)

    n = len(group_keys)
    train_end = int(train_frac * n)
    val_end   = int((train_frac + val_frac) * n)

    train_data = [r for k in group_keys[:train_end]  for r in pair_groups[k]]
    val_data   = [r for k in group_keys[train_end:val_end] for r in pair_groups[k]]
    test_data  = [r for k in group_keys[val_end:]    for r in pair_groups[k]]

    return train_data, val_data, test_data

def collate_fn(batch: List[Dict]) -> Dict:

    return {
        "input_ids":        torch.stack([b["input_ids"] for b in batch]),
        "attention_mask":   torch.stack([b["attention_mask"] for b in batch]),
        "label":            torch.stack([b["label"] for b in batch]),
        "max_surface_risk": torch.stack([b["max_surface_risk"] for b in batch]),
        "difficulty":       [b["difficulty"] for b in batch],
        "family":           [b["family"] for b in batch],
        "pair_id":          [b["pair_id"] for b in batch],
    }


def create_dataloaders(
    path:       str,
    batch_size: int = 16,
    max_len:    int = 512,
    seed:       int = 42,
):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data      = load_jsonl(path)

    train_data, val_data, test_data = pair_level_split(data, seed=seed)

    print(f"Split (pair-level): train={len(train_data)}, "
          f"val={len(val_data)}, test={len(test_data)}")

    train_ds = JailbreakDataset(train_data, tokenizer, max_len)
    val_ds   = JailbreakDataset(val_data,   tokenizer, max_len)
    test_ds  = JailbreakDataset(test_data,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader