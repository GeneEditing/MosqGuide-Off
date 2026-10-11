import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# DNA alphabet for both reference(sgRNA) and off-target sequences
BASES = ["A", "C", "G", "T"]

# 4 x 4 = 16 pair tokens, fixed and reproducible order
PAIR_TOKENS = [f"{b1}{b2}" for b1 in BASES for b2 in BASES]
PAIR_TO_ID = {pair: idx for idx, pair in enumerate(PAIR_TOKENS)}
ID_TO_PAIR = {idx: pair for pair, idx in PAIR_TO_ID.items()}
NUM_PAIR_TOKENS = len(PAIR_TO_ID)

REQUIRED_COLUMNS = ["sgRNA", "off_target", "label", "mismatch_count"]


def encode_pair(sgrna: str, off_target: str) -> List[int]:
    """
    Encode one 23nt sgRNA/off-target pair into 23 pair-token IDs.

    Input:
        sgrna: str, length must be 23, characters in A/C/G/T
        off_target: str, length must be 23, characters in A/C/G/T

    Output:
        List[int] of length 23
    """
    s = sgrna.strip().upper()
    o = off_target.strip().upper()

    if len(s) != 23 or len(o) != 23:
        raise ValueError(
            f"Both sgrna and off_target must have length 23, got {len(s)} and {len(o)}"
        )

    if any(ch not in BASES for ch in s):
        raise ValueError(f"sgrna contains invalid characters: {s}")
    if any(ch not in BASES for ch in o):
        raise ValueError(f"off_target contains invalid characters: {o}")

    pair_ids = [PAIR_TO_ID[f"{sb}{ob}"] for sb, ob in zip(s, o)]
    return pair_ids


class OffTargetDataset(Dataset):
    """
    Off-target binary classification dataset.

    Returns each sample as:
    {
        "pair_ids": torch.LongTensor shape [23],
        "label": torch.FloatTensor scalar,
        "mismatch_count": torch.LongTensor scalar,
    }
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {csv_path}: {missing_cols}. "
                f"Required columns: {REQUIRED_COLUMNS}"
            )

        # Normalize sequence columns
        df["sgRNA"] = df["sgRNA"].astype(str).str.strip().str.upper()
        df["off_target"] = df["off_target"].astype(str).str.strip().str.upper()

        # Validate sequence lengths
        sgrna_len_ok = df["sgRNA"].str.len().eq(23)
        off_len_ok = df["off_target"].str.len().eq(23)
        if not sgrna_len_ok.all():
            bad_idx = df.index[~sgrna_len_ok][:5].tolist()
            raise ValueError(f"Found sgRNA length != 23 at rows: {bad_idx}")
        if not off_len_ok.all():
            bad_idx = df.index[~off_len_ok][:5].tolist()
            raise ValueError(f"Found off_target length != 23 at rows: {bad_idx}")

        # Validate DNA alphabet A/C/G/T only
        sgrna_char_ok = df["sgRNA"].str.fullmatch(r"[ACGT]{23}")
        off_char_ok = df["off_target"].str.fullmatch(r"[ACGT]{23}")
        if not sgrna_char_ok.all():
            bad_idx = df.index[~sgrna_char_ok][:5].tolist()
            raise ValueError(f"Found invalid characters in sgRNA at rows: {bad_idx}")
        if not off_char_ok.all():
            bad_idx = df.index[~off_char_ok][:5].tolist()
            raise ValueError(f"Found invalid characters in off_target at rows: {bad_idx}")

        # Validate label: only 0/1
        label_num = pd.to_numeric(df["label"], errors="coerce")
        label_ok = label_num.isin([0, 1])
        if not label_ok.all():
            bad_idx = df.index[~label_ok][:5].tolist()
            raise ValueError(f"Found invalid label (must be 0/1) at rows: {bad_idx}")

        # Validate mismatch_count: integer-like and in [0, 23]
        mm_num = pd.to_numeric(df["mismatch_count"], errors="coerce")
        mm_int_like = mm_num.notna() & (mm_num == mm_num.astype(int))
        mm_range_ok = mm_num.between(0, 23, inclusive="both")
        mm_ok = mm_int_like & mm_range_ok
        if not mm_ok.all():
            bad_idx = df.index[~mm_ok][:5].tolist()
            raise ValueError(
                f"Found invalid mismatch_count (must be integer in [0, 23]) at rows: {bad_idx}"
            )

        # Store compact columns for fast __getitem__
        self.sgrna = df["sgRNA"].tolist()
        self.off_target = df["off_target"].tolist()
        self.labels = label_num.astype(int).tolist()
        self.mismatch_counts = mm_num.astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_ids = encode_pair(self.sgrna[idx], self.off_target[idx])

        return {
            "pair_ids": torch.tensor(pair_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "mismatch_count": torch.tensor(self.mismatch_counts[idx], dtype=torch.long),
        }


def create_dataloader(
    csv_path: str,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader for off-target binary classification.
    """
    dataset = OffTargetDataset(csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    default_train_csv = str(Path(__file__).resolve().parents[1] / "data" / "train.csv")

    print("PAIR_TO_ID:", PAIR_TO_ID)
    print("NUM_PAIR_TOKENS:", NUM_PAIR_TOKENS)

    if NUM_PAIR_TOKENS != 16:
        raise RuntimeError(f"NUM_PAIR_TOKENS should be 16, got {NUM_PAIR_TOKENS}")

    dataset = OffTargetDataset(default_train_csv)
    print("Total samples:", len(dataset))

    print("First raw sample:")
    print("  sgRNA:", dataset.sgrna[0])
    print("  off_target:", dataset.off_target[0])
    print("  label:", dataset.labels[0])
    print("  mismatch_count:", dataset.mismatch_counts[0])

    first_item = dataset[0]
    print("First encoded sample:")
    print("  pair_ids:", first_item["pair_ids"])
    print("  pair_ids shape:", first_item["pair_ids"].shape)

    loader = create_dataloader(
        csv_path=default_train_csv,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    first_batch = next(iter(loader))
    print("First batch shapes:")
    print('  batch["pair_ids"].shape:', first_batch["pair_ids"].shape)
    print('  batch["label"].shape:', first_batch["label"].shape)
    print('  batch["mismatch_count"].shape:', first_batch["mismatch_count"].shape)

    print("Dataset test PASS.")
