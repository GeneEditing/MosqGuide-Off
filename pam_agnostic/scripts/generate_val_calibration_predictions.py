#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
FEATURE_DIR = PROJECT_ROOT / "feature_enhancement"

for path in [SRC_DIR, FEATURE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dataset import encode_pair
from model import OffTargetModel
from inspect_features import build_handcrafted_features


FEATURE_DIM = 81

DEFAULT_DL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "offtarget_pam_agnostic_focal_best.pt"
)

DEFAULT_RF_MODEL = (
    PROJECT_ROOT
    / "checkpoints"
    / "rf_handcrafted_baseline.joblib"
)

DEFAULT_ENSEMBLE_WEIGHT = 0.84


# ============================================================
# Column aliases
# ============================================================

GUIDE_ALIASES = [
    "guide_id",
]

SGRNA_ALIASES = [
    "sgRNA_20nt",
    "sgRNA",
]

OFF_TARGET_ALIASES = [
    "off_target_20nt",
    "off_target",
]

LABEL_ALIASES = [
    "label",
]


def resolve_column(df, aliases, role):

    for col in aliases:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {role} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        sgrnas,
        off_targets
    ):
        self.sgrnas = sgrnas
        self.off_targets = off_targets

    def __len__(self):
        return len(self.sgrnas)

    def __getitem__(self, idx):

        pair_ids = encode_pair(
            self.sgrnas[idx],
            self.off_targets[idx]
        )

        return torch.tensor(
            pair_ids,
            dtype=torch.long
        )


# ============================================================
# Device
# ============================================================

def resolve_device(device_arg):

    import os

    value = str(
        device_arg
    ).strip().lower()

    if value.isdigit():

        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES",
            value,
        )

        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "[WARN] CUDA not available. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    if value == "cuda":

        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "[WARN] CUDA not available. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    if value == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "--device must be "
        "'cpu', 'cuda', or a GPU ID such as '6'"
    )


# ============================================================
# Load / QC
# ============================================================

def load_val(path):

    print(
        f"Loading VAL: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    guide_col = resolve_column(
        df,
        GUIDE_ALIASES,
        "guide_id"
    )

    sgrna_col = resolve_column(
        df,
        SGRNA_ALIASES,
        "sgRNA"
    )

    off_col = resolve_column(
        df,
        OFF_TARGET_ALIASES,
        "off_target"
    )

    label_col = resolve_column(
        df,
        LABEL_ALIASES,
        "label"
    )

    # --------------------------------------------------------
    # Normalize sequences
    # --------------------------------------------------------

    df[sgrna_col] = (
        df[sgrna_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:20]
    )

    df[off_col] = (
        df[off_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:20]
    )

    # --------------------------------------------------------
    # Sequence QC
    # --------------------------------------------------------

    sgrna_ok = (
        df[sgrna_col]
        .str.fullmatch(
            r"[ACGT]{20}"
        )
    )

    off_ok = (
        df[off_col]
        .str.fullmatch(
            r"[ACGT]{20}"
        )
    )

    valid_seq = (
        sgrna_ok
        & off_ok
    )

    invalid_seq_count = int(
        (~valid_seq).sum()
    )

    if invalid_seq_count > 0:

        raise ValueError(
            f"Invalid 20nt sequences detected: "
            f"{invalid_seq_count:,}"
        )

    # --------------------------------------------------------
    # Label QC
    # --------------------------------------------------------

    if df[
        label_col
    ].isna().any():

        raise ValueError(
            "NaN found in label column."
        )

    df[label_col] = (
        df[label_col]
        .astype(int)
    )

    unique_labels = set(
        df[label_col]
        .unique()
        .tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Unexpected labels: "
            f"{sorted(unique_labels)}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n = len(df)

    positives = int(
        df[label_col].sum()
    )

    negatives = (
        n - positives
    )

    positive_ratio = (
        positives / n
    )

    print()
    print("=" * 70)
    print("VAL dataset QC")
    print("=" * 70)

    print(
        f"Rows:                  "
        f"{n:,}"
    )

    print(
        f"Guides:                "
        f"{df[guide_col].nunique():,}"
    )

    print(
        f"Positive:              "
        f"{positives:,}"
    )

    print(
        f"Negative:              "
        f"{negatives:,}"
    )

    print(
        f"Positive ratio:        "
        f"{positive_ratio:.6f}"
    )

    print(
        f"Invalid sequences:     "
        f"{invalid_seq_count:,}"
    )

    return (
        df,
        guide_col,
        sgrna_col,
        off_col,
        label_col,
    )


# ============================================================
# DL prediction
# ============================================================

def predict_dl(
    df,
    sgrna_col,
    off_col,
    checkpoint_path,
    device,
    batch_size,
    num_workers,
):

    print()
    print("=" * 70)
    print("DL prediction")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = OffTargetModel()

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    dataset = PairDataset(
        df[sgrna_col].tolist(),
        df[off_col].tolist()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    predictions = []

    total_batches = len(loader)

    with torch.no_grad():

        for batch_idx, pair_ids in enumerate(
            loader,
            start=1
        ):

            pair_ids = pair_ids.to(
                device,
                non_blocking=True
            )

            logits = model(
                pair_ids
            )

            probs = torch.sigmoid(
                logits
            )

            probs = (
                probs
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            predictions.append(
                probs
            )

            if (
                batch_idx % 100
                == 0
                or batch_idx
                == total_batches
            ):

                print(
                    f"DL batches: "
                    f"{batch_idx:,} / "
                    f"{total_batches:,}"
                )

    predictions = np.concatenate(
        predictions
    ).astype(
        np.float32
    )

    if len(predictions) != len(df):

        raise RuntimeError(
            "DL prediction length mismatch: "
            f"{len(predictions)} "
            f"vs {len(df)}"
        )

    print()
    print(
        f"DL predictions:       "
        f"{len(predictions):,}"
    )

    print(
        f"DL mean:              "
        f"{predictions.mean():.6f}"
    )

    print(
        f"DL min:               "
        f"{predictions.min():.6f}"
    )

    print(
        f"DL max:               "
        f"{predictions.max():.6f}"
    )

    return predictions


# ============================================================
# RF prediction
# ============================================================

def predict_rf(
    df,
    sgrna_col,
    off_col,
    rf_model_path,
):

    print()
    print("=" * 70)
    print("RF prediction")
    print("=" * 70)

    print(
        f"RF model: "
        f"{rf_model_path}"
    )

    rf_model = joblib.load(
        rf_model_path
    )

    n = len(df)

    features = np.empty(
        (
            n,
            FEATURE_DIM
        ),
        dtype=np.float32
    )

    sgrnas = (
        df[sgrna_col]
        .tolist()
    )

    off_targets = (
        df[off_col]
        .tolist()
    )

    for i, (
        sgrna,
        off_target
    ) in enumerate(
        zip(
            sgrnas,
            off_targets
        )
    ):

        _, vector = (
            build_handcrafted_features(
                sgrna,
                off_target
            )
        )

        vector = np.asarray(
            vector,
            dtype=np.float32
        ).reshape(-1)

        if len(vector) != FEATURE_DIM:

            raise RuntimeError(
                f"Feature dimension error "
                f"at row {i}: "
                f"{len(vector)} "
                f"!= {FEATURE_DIM}"
            )

        features[i] = vector

        if (
            (i + 1)
            % 50000
            == 0
            or (i + 1) == n
        ):

            print(
                f"RF features: "
                f"{i + 1:,} / "
                f"{n:,}"
            )

    print()
    print(
        "Running RF predict_proba..."
    )

    probabilities = (
        rf_model
        .predict_proba(
            features
        )[:, 1]
        .astype(
            np.float32
        )
    )

    if len(probabilities) != n:

        raise RuntimeError(
            "RF prediction length mismatch."
        )

    print()
    print(
        f"RF predictions:       "
        f"{len(probabilities):,}"
    )

    print(
        f"RF mean:              "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"RF min:               "
        f"{probabilities.min():.6f}"
    )

    print(
        f"RF max:               "
        f"{probabilities.max():.6f}"
    )

    return probabilities


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen DL/RF raw predictions "
            "on the original-distribution VAL set "
            "for probability calibration."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Original-distribution VAL CSV."
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output calibration CSV."
        )
    )

    parser.add_argument(
        "--dl-checkpoint",
        type=Path,
        default=DEFAULT_DL_CHECKPOINT
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
        default=DEFAULT_RF_MODEL
    )

    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=DEFAULT_ENSEMBLE_WEIGHT
    )

    parser.add_argument(
        "--device",
        type=str,
        default="6"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.ensemble_weight
        <= 1.0
    ):

        raise ValueError(
            "--ensemble-weight must "
            "be between 0 and 1."
        )

    if not args.input.exists():

        raise FileNotFoundError(
            f"VAL file not found: "
            f"{args.input}"
        )

    if not args.dl_checkpoint.exists():

        raise FileNotFoundError(
            f"DL checkpoint not found: "
            f"{args.dl_checkpoint}"
        )

    if not args.rf_model.exists():

        raise FileNotFoundError(
            f"RF model not found: "
            f"{args.rf_model}"
        )

    device = resolve_device(
        args.device
    )

    print("=" * 70)
    print("VAL prediction for probability calibration")
    print("=" * 70)

    print(
        f"Device:                "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:                   "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"DL ensemble weight:    "
        f"{args.ensemble_weight:.2f}"
    )

    print(
        f"RF ensemble weight:    "
        f"{1.0 - args.ensemble_weight:.2f}"
    )

    # --------------------------------------------------------
    # Load VAL
    # --------------------------------------------------------

    (
        df,
        guide_col,
        sgrna_col,
        off_col,
        label_col,
    ) = load_val(
        args.input
    )

    # --------------------------------------------------------
    # DL prediction
    # --------------------------------------------------------

    dl_raw = predict_dl(
        df,
        sgrna_col,
        off_col,
        args.dl_checkpoint,
        device,
        args.batch_size,
        args.num_workers,
    )

    # --------------------------------------------------------
    # RF prediction
    # --------------------------------------------------------

    rf_raw = predict_rf(
        df,
        sgrna_col,
        off_col,
        args.rf_model,
    )

    # --------------------------------------------------------
    # Frozen ensemble
    # --------------------------------------------------------

    w = args.ensemble_weight

    ensemble_raw = (
        w * dl_raw
        +
        (1.0 - w) * rf_raw
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = pd.DataFrame({

        "guide_id":
            df[guide_col]
            .astype(str),

        "sgRNA_20nt":
            df[sgrna_col],

        "off_target_20nt":
            df[off_col],

        "label":
            df[label_col]
            .astype(int),

        "dl_raw_score":
            dl_raw,

        "rf_raw_score":
            rf_raw,

        "ensemble_raw_score":
            ensemble_raw,
    })

    # Preserve useful metadata if available.
    optional_columns = [
        "mismatch_count",
        "source",
        "assay",
    ]

    for col in optional_columns:

        if col in df.columns:
            output[col] = df[col]

    # --------------------------------------------------------
    # Final QC
    # --------------------------------------------------------

    score_columns = [
        "dl_raw_score",
        "rf_raw_score",
        "ensemble_raw_score",
    ]

    if (
        output[
            score_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "NaN found in raw predictions."
        )

    for col in score_columns:

        if not (
            output[col]
            .between(
                0.0,
                1.0
            )
            .all()
        ):

            raise RuntimeError(
                f"{col} contains "
                f"values outside [0,1]."
            )

    if len(output) != len(df):

        raise RuntimeError(
            "Output row count mismatch."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        args.output,
        index=False
    )

    print()
    print("=" * 70)
    print("VAL prediction complete")
    print("=" * 70)

    print(
        f"Rows written:          "
        f"{len(output):,}"
    )

    print(
        f"Guides:                "
        f"{output['guide_id'].nunique():,}"
    )

    print(
        f"Positive ratio:        "
        f"{output['label'].mean():.6f}"
    )

    print()
    print(
        f"DL raw mean:           "
        f"{output['dl_raw_score'].mean():.6f}"
    )

    print(
        f"RF raw mean:           "
        f"{output['rf_raw_score'].mean():.6f}"
    )

    print(
        f"Ensemble raw mean:     "
        f"{output['ensemble_raw_score'].mean():.6f}"
    )

    print()
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()