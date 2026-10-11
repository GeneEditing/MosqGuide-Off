#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Calibration utilities
# ============================================================

def fit_calibrator(
    method,
    scores,
    labels,
):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if method == "platt":

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

        model.fit(
            scores.reshape(-1, 1),
            labels,
        )

    elif method == "isotonic":

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

        model.fit(
            scores,
            labels,
        )

    else:

        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    return {
        "method": method,
        "model": model,
    }


def apply_calibrator(
    bundle,
    scores,
):

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
        f"Unknown calibrator method: {method}"
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
    n = len(y_true)

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


# ============================================================
# Guide-level OOF calibration
# ============================================================

def run_group_oof_calibration(
    scores,
    labels,
    groups,
    method,
    n_splits,
):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    groups = np.asarray(
        groups,
    )

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    oof_probability = np.full(
        len(labels),
        np.nan,
        dtype=float,
    )

    fold_ids = np.zeros(
        len(labels),
        dtype=int,
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        splitter.split(
            scores,
            labels,
            groups=groups,
        ),
        start=1,
    ):

        y_train = labels[
            train_idx
        ]

        y_valid = labels[
            valid_idx
        ]

        if len(
            np.unique(
                y_train
            )
        ) < 2:

            raise RuntimeError(
                f"Fold {fold} calibration "
                "training data contains one class."
            )

        bundle = fit_calibrator(
            method,
            scores[train_idx],
            y_train,
        )

        fold_probability = (
            apply_calibrator(
                bundle,
                scores[valid_idx],
            )
        )

        oof_probability[
            valid_idx
        ] = fold_probability

        fold_ids[
            valid_idx
        ] = fold

        print(
            f"  fold {fold}: "
            f"train guides="
            f"{len(np.unique(groups[train_idx]))}, "
            f"valid guides="
            f"{len(np.unique(groups[valid_idx]))}, "
            f"train positive="
            f"{y_train.mean():.4f}, "
            f"valid positive="
            f"{y_valid.mean():.4f}"
        )

    if np.isnan(
        oof_probability
    ).any():

        raise RuntimeError(
            "NaN detected in OOF calibrated probabilities."
        )

    return (
        oof_probability,
        fold_ids,
    )


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    df,
    score_column,
    model_name,
    n_splits,
    n_bins,
):

    labels = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["guide_id"]
        .astype(str)
        .to_numpy()
    )

    scores = (
        df[score_column]
        .astype(float)
        .to_numpy()
    )

    all_results = []

    calibrated_predictions = {}

    # RAW

    raw_metrics = calculate_metrics(
        labels,
        scores,
        n_bins,
    )

    raw_metrics.update({
        "model": model_name,
        "method": "raw",
    })

    all_results.append(
        raw_metrics
    )

    print()
    print(
        f"{model_name}: RAW"
    )

    print(
        f"  Brier: "
        f"{raw_metrics['brier']:.6f}"
    )

    print(
        f"  ECE:   "
        f"{raw_metrics['ece']:.6f}"
    )

    print(
        f"  mean prediction: "
        f"{raw_metrics['mean_probability']:.6f}"
    )

    # Platt + Isotonic

    for method in [
        "platt",
        "isotonic",
    ]:

        print()
        print(
            f"{model_name}: {method}"
        )

        (
            oof_probability,
            fold_ids,
        ) = run_group_oof_calibration(
            scores,
            labels,
            groups,
            method,
            n_splits,
        )

        result = calculate_metrics(
            labels,
            oof_probability,
            n_bins,
        )

        result.update({
            "model": model_name,
            "method": method,
        })

        all_results.append(
            result
        )

        calibrated_predictions[
            method
        ] = {
            "probability":
                oof_probability,

            "fold":
                fold_ids,
        }

        print(
            f"  Brier:   "
            f"{result['brier']:.6f}"
        )

        print(
            f"  ECE:     "
            f"{result['ece']:.6f}"
        )

        print(
            f"  LogLoss: "
            f"{result['log_loss']:.6f}"
        )

        print(
            f"  Mean calibrated: "
            f"{result['mean_probability']:.6f}"
        )

    # Select:
    # primary Brier
    # secondary ECE

    calibrated_results = [
        result
        for result in all_results
        if result["method"] != "raw"
    ]

    calibrated_results.sort(
        key=lambda result: (
            result["brier"],
            result["ece"],
        )
    )

    selected_method = (
        calibrated_results[0][
            "method"
        ]
    )

    selected_metrics = (
        calibrated_results[0]
    )

    print()
    print(
        f"{model_name} selected: "
        f"{selected_method}"
    )

    print(
        f"  Brier = "
        f"{selected_metrics['brier']:.6f}"
    )

    print(
        f"  ECE   = "
        f"{selected_metrics['ece']:.6f}"
    )

    return (
        all_results,
        calibrated_predictions,
        selected_method,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Select and fit probability calibrators "
            "for frozen NGG DL and RF models."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "checkpoints",
        help="Where to save the fitted DL/RF calibrator weights (joblib).",
    )

    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.86,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.ensemble_weight
        <= 1.0
    ):

        raise ValueError(
            "--ensemble-weight must be in [0,1]."
        )

    # ========================================================
    # Load
    # ========================================================

    df = pd.read_csv(
        args.input,
        low_memory=False,
    )

    required = {
        "guide_id",
        "label",
        "dl_raw_score",
        "rf_raw_score",
        "ensemble_raw_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        list(required)
    ].isna().any().any():

        raise ValueError(
            "NaN detected in calibration input."
        )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    labels = (
        df["label"]
        .to_numpy()
    )

    n_guides = (
        df["guide_id"]
        .nunique()
    )

    if n_guides < args.folds:

        raise ValueError(
            f"Only {n_guides} guides "
            f"but folds={args.folds}."
        )

    args.checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NGG probability calibration")
    print("=" * 70)

    print(
        f"Rows:              "
        f"{len(df):,}"
    )

    print(
        f"Guides:            "
        f"{n_guides:,}"
    )

    print(
        f"Positive:          "
        f"{labels.sum():,}"
    )

    print(
        f"Positive ratio:    "
        f"{labels.mean():.6f}"
    )

    print(
        f"Guide CV folds:    "
        f"{args.folds}"
    )

    # ========================================================
    # DL
    # ========================================================

    print()
    print("=" * 70)
    print("DL calibration")
    print("=" * 70)

    (
        dl_results,
        dl_oof,
        dl_selected_method,
    ) = evaluate_model(
        df,
        "dl_raw_score",
        "DL",
        args.folds,
        args.ece_bins,
    )

    # ========================================================
    # RF
    # ========================================================

    print()
    print("=" * 70)
    print("RF calibration")
    print("=" * 70)

    (
        rf_results,
        rf_oof,
        rf_selected_method,
    ) = evaluate_model(
        df,
        "rf_raw_score",
        "RF",
        args.folds,
        args.ece_bins,
    )

    # ========================================================
    # OOF calibrated ensemble
    # ========================================================

    dl_selected_probability = (
        dl_oof[
            dl_selected_method
        ]["probability"]
    )

    rf_selected_probability = (
        rf_oof[
            rf_selected_method
        ]["probability"]
    )

    w = args.ensemble_weight

    ensemble_probability_oof = (
        w
        * dl_selected_probability
        +
        (1.0 - w)
        * rf_selected_probability
    )

    ensemble_metrics = calculate_metrics(
        labels,
        ensemble_probability_oof,
        args.ece_bins,
    )

    ensemble_metrics.update({
        "model":
            "CALIBRATED_ENSEMBLE",

        "method":
            (
                f"{w:.2f}*DL_{dl_selected_method}"
                f"+"
                f"{1-w:.2f}*RF_{rf_selected_method}"
            ),
    })

    print()
    print("=" * 70)
    print("Calibrated NGG ensemble")
    print("=" * 70)

    print(
        f"DL method:          "
        f"{dl_selected_method}"
    )

    print(
        f"RF method:          "
        f"{rf_selected_method}"
    )

    print(
        f"DL weight:          "
        f"{w:.2f}"
    )

    print(
        f"RF weight:          "
        f"{1-w:.2f}"
    )

    print()

    print(
        f"Brier:              "
        f"{ensemble_metrics['brier']:.6f}"
    )

    print(
        f"ECE:                "
        f"{ensemble_metrics['ece']:.6f}"
    )

    print(
        f"LogLoss:            "
        f"{ensemble_metrics['log_loss']:.6f}"
    )

    print(
        f"Mean probability:   "
        f"{ensemble_metrics['mean_probability']:.6f}"
    )

    print(
        f"Observed positive:  "
        f"{ensemble_metrics['observed_positive_rate']:.6f}"
    )

    # ========================================================
    # Fit FINAL calibrators on full VAL
    # ========================================================

    print()
    print("=" * 70)
    print("Fitting final calibrators on full VAL")
    print("=" * 70)

    dl_final_bundle = fit_calibrator(
        dl_selected_method,
        df[
            "dl_raw_score"
        ].to_numpy(),
        labels,
    )

    rf_final_bundle = fit_calibrator(
        rf_selected_method,
        df[
            "rf_raw_score"
        ].to_numpy(),
        labels,
    )

    dl_final_bundle[
        "model_name"
    ] = "NGG_DL"

    dl_final_bundle[
        "val_rows"
    ] = int(
        len(df)
    )

    dl_final_bundle[
        "val_guides"
    ] = int(
        n_guides
    )

    dl_final_bundle[
        "val_positive_ratio"
    ] = float(
        labels.mean()
    )

    rf_final_bundle[
        "model_name"
    ] = "NGG_RF"

    rf_final_bundle[
        "val_rows"
    ] = int(
        len(df)
    )

    rf_final_bundle[
        "val_guides"
    ] = int(
        n_guides
    )

    rf_final_bundle[
        "val_positive_ratio"
    ] = float(
        labels.mean()
    )

    dl_path = (
        args.checkpoints_dir
        / "dl_probability_calibrator.joblib"
    )

    rf_path = (
        args.checkpoints_dir
        / "rf_probability_calibrator.joblib"
    )

    joblib.dump(
        dl_final_bundle,
        dl_path,
    )

    joblib.dump(
        rf_final_bundle,
        rf_path,
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)
    print("NGG calibration fitting complete")
    print("=" * 70)

    print(
        f"Selected DL calibrator: "
        f"{dl_selected_method}"
    )

    print(
        f"Selected RF calibrator: "
        f"{rf_selected_method}"
    )

    print()

    print(
        f"DL calibrator: {dl_path}"
    )

    print(
        f"RF calibrator: {rf_path}"
    )


if __name__ == "__main__":
    main()