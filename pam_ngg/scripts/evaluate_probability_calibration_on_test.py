#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


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


DEFAULT_DL_WEIGHT = 0.86


# ============================================================
# Calibration
# ============================================================

def apply_calibrator(bundle, scores):

    method = bundle["method"]
    model = bundle["model"]

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if method == "platt":

        return model.predict_proba(
            scores.reshape(-1, 1)
        )[:, 1]

    if method == "isotonic":

        return np.asarray(
            model.predict(scores),
            dtype=float,
        )

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=15,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        y_prob,
        edges[1:-1],
        right=False,
    )

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):

        mask = (
            bin_ids == i
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_prediction = float(
            y_prob[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        ece += (
            count / n
        ) * abs(
            mean_prediction
            - observed_rate
        )

    return float(ece)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    n_bins=15,
):

    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    safe_prob = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "brier":
            float(
                brier_score_loss(
                    labels,
                    probabilities,
                )
            ),

        "ece":
            float(
                expected_calibration_error(
                    labels,
                    probabilities,
                    n_bins=n_bins,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    labels,
                    safe_prob,
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),

        "mean_probability":
            float(
                probabilities.mean()
            ),

        "observed_positive_rate":
            float(
                labels.mean()
            ),
    }


# ============================================================
# DL prediction
# ============================================================

def predict_dl(
    df,
    checkpoint_path,
    device,
    batch_size,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = OffTargetModel().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    outputs = []

    with torch.no_grad():

        for start in range(
            0,
            len(df),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(df),
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
                    for sgrna, off_target
                    in zip(
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

            prob = torch.sigmoid(
                logits
            )

            outputs.append(
                prob
                .detach()
                .cpu()
                .numpy()
            )

            if (
                end == len(df)
                or end % 50000 < batch_size
            ):

                print(
                    f"DL predicted "
                    f"{end:,}/{len(df):,}"
                )

    return np.concatenate(
        outputs
    ).astype(float)


# ============================================================
# RF prediction
# ============================================================

def predict_rf(
    df,
    rf_path,
):

    rf_model = joblib.load(
        rf_path
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

        features[i] = vector

        if (
            i + 1 == len(df)
            or (i + 1) % 50000 == 0
        ):

            print(
                f"RF features "
                f"{i + 1:,}/{len(df):,}"
            )

    return (
        rf_model
        .predict_proba(
            features
        )[:, 1]
        .astype(float)
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen NGG probability calibration "
            "on untouched TEST."
        )
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "test.csv"
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
        "--dl-calibrator",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "dl_probability_calibrator.joblib"
        ),
    )

    parser.add_argument(
        "--rf-calibrator",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "rf_probability_calibrator.joblib"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "calibration"
            / "test_calibrated_predictions.csv"
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
        "--ece-bins",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()

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

    # ========================================================
    # Load TEST
    # ========================================================

    df = pd.read_csv(
        args.test,
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
            f"TEST missing columns: "
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

    df["label"] = (
        df["label"]
        .astype(int)
    )

    print("=" * 70)
    print("NGG probability calibration TEST evaluation")
    print("=" * 70)

    print(
        f"TEST rows:      "
        f"{len(df):,}"
    )

    print(
        f"TEST guides:    "
        f"{df['guide_id'].nunique():,}"
    )

    print(
        f"TEST positive:  "
        f"{int(df['label'].sum()):,}"
    )

    print(
        f"Positive rate:  "
        f"{df['label'].mean():.6f}"
    )

    # ========================================================
    # Raw predictions
    # ========================================================

    dl_raw = predict_dl(
        df,
        args.dl_checkpoint,
        device,
        args.batch_size,
    )

    rf_raw = predict_rf(
        df,
        args.rf_model,
    )

    w = args.dl_weight

    ensemble_raw = (
        w * dl_raw
        + (1.0 - w) * rf_raw
    )

    # ========================================================
    # Calibration
    # ========================================================

    dl_calibrator = joblib.load(
        args.dl_calibrator
    )

    rf_calibrator = joblib.load(
        args.rf_calibrator
    )

    dl_calibrated = apply_calibrator(
        dl_calibrator,
        dl_raw,
    )

    rf_calibrated = apply_calibrator(
        rf_calibrator,
        rf_raw,
    )

    ensemble_calibrated = (
        w * dl_calibrated
        + (1.0 - w) * rf_calibrated
    )

    # ========================================================
    # Metrics
    # ========================================================

    raw_metrics = calculate_metrics(
        df["label"],
        ensemble_raw,
        args.ece_bins,
    )

    calibrated_metrics = (
        calculate_metrics(
            df["label"],
            ensemble_calibrated,
            args.ece_bins,
        )
    )

    print()
    print("=" * 70)
    print("RAW ensemble")
    print("=" * 70)

    for key, value in raw_metrics.items():

        print(
            f"{key:24s}: "
            f"{value:.6f}"
        )

    print()
    print("=" * 70)
    print("CALIBRATED ensemble")
    print("=" * 70)

    for key, value in calibrated_metrics.items():

        print(
            f"{key:24s}: "
            f"{value:.6f}"
        )

    # ========================================================
    # Save
    # ========================================================

    output = df.copy()

    output[
        "dl_raw_score"
    ] = dl_raw

    output[
        "rf_raw_score"
    ] = rf_raw

    output[
        "ensemble_raw_score"
    ] = ensemble_raw

    output[
        "dl_calibrated_probability"
    ] = dl_calibrated

    output[
        "rf_calibrated_probability"
    ] = rf_calibrated

    output[
        "off_target_probability"
    ] = ensemble_calibrated

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()