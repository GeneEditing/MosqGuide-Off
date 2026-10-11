#!/usr/bin/env python3

import argparse
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

from torch.utils.data import (
    Dataset,
    DataLoader,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "feature_enhancement"
)

for path in [
    SRC_DIR,
    FEATURE_DIR,
]:

    if str(path) not in sys.path:

        sys.path.insert(
            0,
            str(path)
        )


from dataset import encode_pair
from model import OffTargetModel
from inspect_features import (
    build_handcrafted_features,
)


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

DEFAULT_DL_CALIBRATOR = (
    PROJECT_ROOT
    / "checkpoints"
    / "dl_probability_calibrator.joblib"
)

DEFAULT_RF_CALIBRATOR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rf_probability_calibrator.joblib"
)

DEFAULT_ENSEMBLE_WEIGHT = 0.84


# ============================================================
# Calibration
# ============================================================

def apply_calibrator(
    bundle,
    scores,
):

    method = bundle[
        "method"
    ]

    model = bundle[
        "model"
    ]

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if method == "platt":

        return (
            model
            .predict_proba(
                scores.reshape(-1, 1)
            )[:, 1]
        )

    if method == "isotonic":

        return np.asarray(
            model.predict(
                scores
            ),
            dtype=float,
        )

    raise ValueError(
        f"Unknown calibration method: "
        f"{method}"
    )


# ============================================================
# Metrics
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

    n = len(
        y_true
    )

    for bin_index in range(
        n_bins
    ):

        mask = (
            bin_ids
            == bin_index
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            y_prob[
                mask
            ].mean()
        )

        observed_rate = float(
            y_true[
                mask
            ].mean()
        )

        ece += (
            count / n
        ) * abs(
            mean_probability
            - observed_rate
        )

    return float(
        ece
    )


def calculate_metrics(
    y_true,
    y_prob,
    n_bins,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    safe_prob = np.clip(
        y_prob,
        1e-7,
        1.0 - 1e-7,
    )

    return {

        "brier":
            float(
                brier_score_loss(
                    y_true,
                    y_prob,
                )
            ),

        "ece":
            float(
                expected_calibration_error(
                    y_true,
                    y_prob,
                    n_bins=n_bins,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    safe_prob,
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    y_true,
                    y_prob,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y_true,
                    y_prob,
                )
            ),

        "mean_probability":
            float(
                y_prob.mean()
            ),

        "observed_positive_rate":
            float(
                y_true.mean()
            ),
    }


def build_reliability_table(
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

    rows = []

    for i in range(
        n_bins
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (y_prob >= lower)
                & (y_prob <= upper)
            )

        else:

            mask = (
                (y_prob >= lower)
                & (y_prob < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        rows.append({

            "bin":
                i + 1,

            "lower":
                lower,

            "upper":
                upper,

            "count":
                count,

            "mean_predicted_probability":
                float(
                    y_prob[
                        mask
                    ].mean()
                ),

            "observed_positive_rate":
                float(
                    y_true[
                        mask
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        sgrnas,
        off_targets,
    ):

        self.sgrnas = sgrnas
        self.off_targets = off_targets

    def __len__(self):

        return len(
            self.sgrnas
        )

    def __getitem__(
        self,
        idx,
    ):

        ids = encode_pair(
            self.sgrnas[
                idx
            ],
            self.off_targets[
                idx
            ],
        )

        return torch.tensor(
            ids,
            dtype=torch.long,
        )


# ============================================================
# Device
# ============================================================

def resolve_device(
    device_arg
):

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

            return torch.device(
                "cuda"
            )

        print(
            "[WARN] CUDA unavailable. "
            "Using CPU."
        )

        return torch.device(
            "cpu"
        )

    if value == "cuda":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    if value == "cpu":

        return torch.device(
            "cpu"
        )

    raise ValueError(
        "--device must be cpu, cuda, "
        "or a GPU ID such as 6."
    )


# ============================================================
# Column helper
# ============================================================

def resolve_column(
    df,
    aliases,
    role,
):

    for name in aliases:

        if name in df.columns:

            return name

    raise ValueError(
        f"Could not find {role}. "
        f"Tried {aliases}. "
        f"Available: {list(df.columns)}"
    )


# ============================================================
# Load TEST
# ============================================================

def load_test(
    path
):

    print(
        f"Loading TEST: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    guide_col = resolve_column(
        df,
        [
            "guide_id",
        ],
        "guide_id",
    )

    sgrna_col = resolve_column(
        df,
        [
            "sgRNA_20nt",
            "sgRNA",
        ],
        "sgRNA",
    )

    off_col = resolve_column(
        df,
        [
            "off_target_20nt",
            "off_target",
        ],
        "off-target",
    )

    label_col = resolve_column(
        df,
        [
            "label",
        ],
        "label",
    )

    # --------------------------------------------------------
    # Sequence QC
    # --------------------------------------------------------

    df[
        sgrna_col
    ] = (
        df[
            sgrna_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:20]
    )

    df[
        off_col
    ] = (
        df[
            off_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:20]
    )

    sequence_valid = (
        df[
            sgrna_col
        ]
        .str.fullmatch(
            r"[ACGT]{20}"
        )
        &
        df[
            off_col
        ]
        .str.fullmatch(
            r"[ACGT]{20}"
        )
    )

    invalid_sequences = int(
        (
            ~sequence_valid
        ).sum()
    )

    if invalid_sequences > 0:

        raise ValueError(
            f"Invalid sequences: "
            f"{invalid_sequences:,}"
        )

    # --------------------------------------------------------
    # Label QC
    # --------------------------------------------------------

    if df[
        label_col
    ].isna().any():

        raise ValueError(
            "NaN detected in TEST labels."
        )

    df[
        label_col
    ] = (
        df[
            label_col
        ]
        .astype(int)
    )

    unique_labels = set(
        df[
            label_col
        ].unique()
    )

    if not unique_labels.issubset(
        {
            0,
            1,
        }
    ):

        raise ValueError(
            f"Unexpected TEST labels: "
            f"{unique_labels}"
        )

    return (
        df,
        guide_col,
        sgrna_col,
        off_col,
        label_col,
    )


# ============================================================
# DL
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
    print("DL TEST prediction")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = OffTargetModel()

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = (
            checkpoint
        )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    loader = DataLoader(

        PairDataset(
            df[
                sgrna_col
            ].tolist(),

            df[
                off_col
            ].tolist(),
        ),

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    outputs = []

    total_batches = len(
        loader
    )

    with torch.no_grad():

        for batch_index, pair_ids in enumerate(
            loader,
            start=1,
        ):

            pair_ids = pair_ids.to(
                device,
                non_blocking=True,
            )

            logits = model(
                pair_ids
            )

            probability = torch.sigmoid(
                logits
            )

            outputs.append(
                probability
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if (
                batch_index % 100 == 0
                or batch_index
                == total_batches
            ):

                print(
                    f"DL batches: "
                    f"{batch_index:,} / "
                    f"{total_batches:,}"
                )

    prediction = np.concatenate(
        outputs
    ).astype(
        np.float32
    )

    if len(
        prediction
    ) != len(
        df
    ):

        raise RuntimeError(
            "DL prediction length mismatch."
        )

    print(
        f"DL predictions: "
        f"{len(prediction):,}"
    )

    return prediction


# ============================================================
# RF
# ============================================================

def predict_rf(
    df,
    sgrna_col,
    off_col,
    model_path,
):

    print()
    print("=" * 70)
    print("RF TEST prediction")
    print("=" * 70)

    model = joblib.load(
        model_path
    )

    n = len(
        df
    )

    features = np.empty(
        (
            n,
            FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    sgrnas = df[
        sgrna_col
    ].tolist()

    off_targets = df[
        off_col
    ].tolist()

    for i, (
        sgrna,
        off_target,
    ) in enumerate(
        zip(
            sgrnas,
            off_targets,
        )
    ):

        _, vector = (
            build_handcrafted_features(
                sgrna,
                off_target,
            )
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if len(
            vector
        ) != FEATURE_DIM:

            raise RuntimeError(
                f"Feature dimension error "
                f"at row {i}: "
                f"{len(vector)} "
                f"!= {FEATURE_DIM}"
            )

        features[
            i
        ] = vector

        if (
            (i + 1) % 50000 == 0
            or (i + 1) == n
        ):

            print(
                f"RF features: "
                f"{i + 1:,} / "
                f"{n:,}"
            )

    print(
        "Running RF predict_proba..."
    )

    probability = (
        model
        .predict_proba(
            features
        )[:, 1]
        .astype(
            np.float32
        )
    )

    return probability


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final probability-calibration "
            "evaluation on untouched TEST data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dl-checkpoint",
        type=Path,
        default=DEFAULT_DL_CHECKPOINT,
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
        default=DEFAULT_RF_MODEL,
    )

    parser.add_argument(
        "--dl-calibrator",
        type=Path,
        default=DEFAULT_DL_CALIBRATOR,
    )

    parser.add_argument(
        "--rf-calibrator",
        type=Path,
        default=DEFAULT_RF_CALIBRATOR,
    )

    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=DEFAULT_ENSEMBLE_WEIGHT,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="6",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    # ========================================================
    # Check frozen files
    # ========================================================

    for path in [
        args.input,
        args.dl_checkpoint,
        args.rf_model,
        args.dl_calibrator,
        args.rf_calibrator,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if not (
        0.0
        <= args.ensemble_weight
        <= 1.0
    ):

        raise ValueError(
            "--ensemble-weight must be in [0,1]."
        )

    device = resolve_device(
        args.device
    )

    # ========================================================
    # Load TEST
    # ========================================================

    (
        df,
        guide_col,
        sgrna_col,
        off_col,
        label_col,
    ) = load_test(
        args.input
    )

    y_true = (
        df[
            label_col
        ]
        .to_numpy(
            dtype=int
        )
    )

    print()
    print("=" * 70)
    print(
        "Final TEST probability calibration evaluation"
    )
    print("=" * 70)

    print(
        f"Rows:              "
        f"{len(df):,}"
    )

    print(
        f"Guides:            "
        f"{df[guide_col].nunique():,}"
    )

    print(
        f"Positive:          "
        f"{y_true.sum():,}"
    )

    print(
        f"Negative:          "
        f"{len(y_true) - y_true.sum():,}"
    )

    print(
        f"Positive ratio:    "
        f"{y_true.mean():.6f}"
    )

    print(
        f"Device:            "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:               "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Raw DL/RF predictions
    # ========================================================

    dl_raw = predict_dl(
        df,
        sgrna_col,
        off_col,
        args.dl_checkpoint,
        device,
        args.batch_size,
        args.num_workers,
    )

    rf_raw = predict_rf(
        df,
        sgrna_col,
        off_col,
        args.rf_model,
    )

    w = args.ensemble_weight

    ensemble_raw = (
        w
        * dl_raw
        +
        (1.0 - w)
        * rf_raw
    )

    # ========================================================
    # Load frozen calibrators
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Loading frozen calibrators"
    )
    print("=" * 70)

    dl_calibrator = joblib.load(
        args.dl_calibrator
    )

    rf_calibrator = joblib.load(
        args.rf_calibrator
    )

    print(
        f"DL calibration method: "
        f"{dl_calibrator['method']}"
    )

    print(
        f"RF calibration method: "
        f"{rf_calibrator['method']}"
    )

    # ========================================================
    # Apply probability calibration
    # ========================================================

    dl_probability = (
        apply_calibrator(
            dl_calibrator,
            dl_raw,
        )
    )

    rf_probability = (
        apply_calibrator(
            rf_calibrator,
            rf_raw,
        )
    )

    off_target_probability = (
        w
        * dl_probability
        +
        (1.0 - w)
        * rf_probability
    )

    # ========================================================
    # QC
    # ========================================================

    for name, values in [
        (
            "DL probability",
            dl_probability,
        ),
        (
            "RF probability",
            rf_probability,
        ),
        (
            "Off-target probability",
            off_target_probability,
        ),
    ]:

        if np.isnan(
            values
        ).any():

            raise RuntimeError(
                f"NaN detected in {name}."
            )

        if not np.all(
            (
                values >= 0
            )
            &
            (
                values <= 1
            )
        ):

            raise RuntimeError(
                f"{name} outside [0,1]."
            )

    # ========================================================
    # Metrics
    # ========================================================

    metric_rows = []

    prediction_sets = [

        (
            "DL_RAW",
            dl_raw,
        ),

        (
            "RF_RAW",
            rf_raw,
        ),

        (
            "ENSEMBLE_RAW",
            ensemble_raw,
        ),

        (
            "DL_CALIBRATED",
            dl_probability,
        ),

        (
            "RF_CALIBRATED",
            rf_probability,
        ),

        (
            "ENSEMBLE_CALIBRATED",
            off_target_probability,
        ),
    ]

    for model_name, probability in (
        prediction_sets
    ):

        result = calculate_metrics(
            y_true,
            probability,
            args.ece_bins,
        )

        result[
            "model"
        ] = model_name

        metric_rows.append(
            result
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    # ========================================================
    # Output candidate-level table
    # ========================================================

    output = pd.DataFrame({

        "guide_id":
            df[
                guide_col
            ].astype(str),

        "sgRNA_20nt":
            df[
                sgrna_col
            ],

        "off_target_20nt":
            df[
                off_col
            ],

        "label":
            y_true,

        "dl_raw_score":
            dl_raw,

        "rf_raw_score":
            rf_raw,

        "ensemble_raw_score":
            ensemble_raw,

        "dl_probability":
            dl_probability,

        "rf_probability":
            rf_probability,

        "off_target_probability":
            off_target_probability,
    })

    # Preserve useful metadata if present.
    for column in [
        "mismatch_count",
        "source",
        "assay",
    ]:

        if column in df.columns:

            output[
                column
            ] = df[
                column
            ]

    # ========================================================
    # Save
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        args.output_dir
        / "test_calibrated_predictions.csv"
    )

    metrics_path = (
        args.output_dir
        / "test_calibration_metrics.csv"
    )

    reliability_path = (
        args.output_dir
        / "test_ensemble_reliability.csv"
    )

    output.to_csv(
        prediction_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    reliability_df = (
        build_reliability_table(
            y_true,
            off_target_probability,
            args.ece_bins,
        )
    )

    reliability_df.to_csv(
        reliability_path,
        index=False,
    )

    # ========================================================
    # Console metrics
    # ========================================================

    print()
    print("=" * 70)
    print("TEST metrics")
    print("=" * 70)

    display_columns = [
        "model",
        "brier",
        "ece",
        "log_loss",
        "auroc",
        "auprc",
        "mean_probability",
        "observed_positive_rate",
    ]

    print(
        metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    final = (
        metrics_df.loc[
            metrics_df[
                "model"
            ]
            == "ENSEMBLE_CALIBRATED"
        ]
        .iloc[0]
    )

    print()
    print("=" * 70)
    print(
        "FINAL calibrated ensemble"
    )
    print("=" * 70)

    print(
        f"Brier:             "
        f"{final['brier']:.6f}"
    )

    print(
        f"ECE:               "
        f"{final['ece']:.6f}"
    )

    print(
        f"LogLoss:           "
        f"{final['log_loss']:.6f}"
    )

    print(
        f"AUROC:             "
        f"{final['auroc']:.6f}"
    )

    print(
        f"AUPRC:             "
        f"{final['auprc']:.6f}"
    )

    print(
        f"Mean probability:  "
        f"{final['mean_probability']:.6f}"
    )

    print(
        f"Observed positive: "
        f"{final['observed_positive_rate']:.6f}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        prediction_path
    )

    print(
        metrics_path
    )

    print(
        reliability_path
    )


if __name__ == "__main__":
    main()