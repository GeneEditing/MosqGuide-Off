#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import encode_pair
from model import OffTargetModel
from inspect_features import build_handcrafted_features


# ============================================================
# Frozen configuration
# ============================================================

DEFAULT_DL_WEIGHT = 0.86


# ============================================================
# DL prediction
# ============================================================

def predict_dl(
    df,
    checkpoint_path,
    device,
    batch_size,
):

    print()
    print("=" * 70)
    print("DL prediction")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "DL checkpoint does not contain model_state_dict."
        )

    model = OffTargetModel().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    outputs = []

    n = len(df)

    with torch.no_grad():

        for start in range(
            0,
            n,
            batch_size,
        ):

            end = min(
                start + batch_size,
                n,
            )

            batch_df = df.iloc[
                start:end
            ]

            pair_ids = np.asarray(
                [
                    encode_pair(
                        sgrna,
                        off_target,
                    )
                    for (
                        sgrna,
                        off_target,
                    ) in zip(
                        batch_df["sgRNA"],
                        batch_df["off_target"],
                    )
                ],
                dtype=np.int64,
            )

            pair_tensor = torch.tensor(
                pair_ids,
                dtype=torch.long,
                device=device,
            )

            logits = model(
                pair_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            outputs.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            if (
                start == 0
                or end == n
                or end % 50000 < batch_size
            ):

                print(
                    f"  predicted "
                    f"{end:,}/{n:,}"
                )

    result = np.concatenate(
        outputs
    ).astype(float)

    if len(result) != len(df):

        raise RuntimeError(
            "DL prediction length mismatch."
        )

    if not np.isfinite(
        result
    ).all():

        raise RuntimeError(
            "DL predictions contain NaN or Inf."
        )

    print(
        f"DL mean raw probability: "
        f"{result.mean():.6f}"
    )

    return result


# ============================================================
# RF prediction
# ============================================================

def predict_rf(
    df,
    rf_path,
):

    print()
    print("=" * 70)
    print("RF prediction")
    print("=" * 70)

    rf_model = joblib.load(
        rf_path
    )

    n_features = getattr(
        rf_model,
        "n_features_in_",
        None,
    )

    if n_features != 93:

        raise ValueError(
            f"RF expects {n_features} features; "
            f"expected 93."
        )

    features = np.empty(
        (
            len(df),
            93,
        ),
        dtype=np.float32,
    )

    for i, (
        sgrna,
        off_target,
    ) in enumerate(
        zip(
            df["sgRNA"],
            df["off_target"],
        )
    ):

        _, vector = (
            build_handcrafted_features(
                sgrna,
                off_target,
            )
        )

        if vector.shape != (93,):

            raise RuntimeError(
                f"Unexpected feature shape "
                f"at row {i}: "
                f"{vector.shape}"
            )

        features[i] = vector

        if (
            i == 0
            or i + 1 == len(df)
            or (i + 1) % 50000 == 0
        ):

            print(
                f"  features "
                f"{i + 1:,}/{len(df):,}"
            )

    probabilities = (
        rf_model
        .predict_proba(
            features
        )[:, 1]
        .astype(float)
    )

    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "RF predictions contain NaN or Inf."
        )

    print(
        f"RF mean raw probability: "
        f"{probabilities.mean():.6f}"
    )

    return probabilities


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate raw NGG DL/RF predictions "
            "on the fixed VAL set for probability calibration."
        )
    )

    parser.add_argument(
        "--val",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "val.csv"
        ),
    )

    parser.add_argument(
        "--dl-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "offtarget_focal_best.pt"
        ),
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "rf_handcrafted_baseline.joblib"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "calibration"
            / "val_raw_predictions.csv"
        ),
    )

    parser.add_argument(
        "--dl-weight",
        type=float,
        default=DEFAULT_DL_WEIGHT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.dl_weight
        <= 1.0
    ):

        raise ValueError(
            "--dl-weight must be in [0,1]."
        )

    if not args.val.exists():
        raise FileNotFoundError(
            args.val
        )

    if not args.dl_checkpoint.exists():
        raise FileNotFoundError(
            args.dl_checkpoint
        )

    if not args.rf_model.exists():
        raise FileNotFoundError(
            args.rf_model
        )

    # ========================================================
    # Device
    # ========================================================

    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda"
        )

    elif str(args.device).strip().isdigit():

        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES",
            str(args.device).strip(),
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            args.device
        )

    print("=" * 70)
    print("NGG VAL raw prediction generation")
    print("=" * 70)

    print(
        f"VAL:            {args.val}"
    )

    print(
        f"DL checkpoint:  {args.dl_checkpoint}"
    )

    print(
        f"RF model:       {args.rf_model}"
    )

    print(
        f"Device:         {device}"
    )

    print(
        f"DL weight:      {args.dl_weight:.2f}"
    )

    # ========================================================
    # Load VAL
    # ========================================================

    df = pd.read_csv(
        args.val,
        low_memory=False,
    )

    required = {
        "guide_id",
        "sgRNA",
        "off_target",
        "mismatch_count",
        "label",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"VAL missing columns: "
            f"{sorted(missing)}"
        )

    df["sgRNA"] = (
        df["sgRNA"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["off_target"] = (
        df["off_target"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["guide_id"] = (
        df["guide_id"]
        .astype(str)
    )

    df["label"] = (
        pd.to_numeric(
            df["label"],
            errors="raise",
        )
        .astype(int)
    )

    if not (
        df["label"]
        .isin(
            [0, 1]
        )
        .all()
    ):

        raise ValueError(
            "VAL label must contain only 0/1."
        )

    if not (
        df["sgRNA"]
        .str.fullmatch(
            r"[ACGT]{23}"
        )
        .all()
    ):

        raise ValueError(
            "Invalid 23nt sgRNA found."
        )

    if not (
        df["off_target"]
        .str.fullmatch(
            r"[ACGT]{23}"
        )
        .all()
    ):

        raise ValueError(
            "Invalid 23nt off_target found."
        )

    print()
    print(
        f"VAL rows:       {len(df):,}"
    )

    print(
        f"VAL guides:     "
        f"{df['guide_id'].nunique():,}"
    )

    print(
        f"VAL positives:  "
        f"{int(df['label'].sum()):,}"
    )

    print(
        f"Positive rate:  "
        f"{df['label'].mean():.6f}"
    )

    # ========================================================
    # Predictions
    # ========================================================

    dl_prob = predict_dl(
        df,
        args.dl_checkpoint,
        device,
        args.batch_size,
    )

    rf_prob = predict_rf(
        df,
        args.rf_model,
    )

    w = args.dl_weight

    ensemble_raw = (
        w
        * dl_prob
        +
        (1.0 - w)
        * rf_prob
    )

    # ========================================================
    # Save
    # ========================================================

    output = df[
        [
            "guide_id",
            "sgRNA",
            "off_target",
            "mismatch_count",
            "label",
        ]
    ].copy()

    output[
        "dl_raw_score"
    ] = dl_prob

    output[
        "rf_raw_score"
    ] = rf_prob

    output[
        "ensemble_raw_score"
    ] = ensemble_raw

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("VAL raw predictions complete")
    print("=" * 70)

    print(
        f"Mean DL score:       "
        f"{dl_prob.mean():.6f}"
    )

    print(
        f"Mean RF score:       "
        f"{rf_prob.mean():.6f}"
    )

    print(
        f"Mean ensemble score: "
        f"{ensemble_raw.mean():.6f}"
    )

    print(
        f"Observed positive:   "
        f"{df['label'].mean():.6f}"
    )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()