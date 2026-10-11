import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import create_dataloader
from model import OffTargetModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PAM-agnostic 20nt off-target model")

    parser.add_argument(
        "--val_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "val.csv"),
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "test.csv"),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "offtarget_pam_agnostic_focal_best.pt"),
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help=(
            "GPU setting. Examples: '3' (equivalent to CUDA_VISIBLE_DEVICES=3), "
            "'cuda:3', or 'cuda'."
        ),
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "offtarget_pam_agnostic_test_predictions.csv"),
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "offtarget_pam_agnostic_test_metrics.csv"),
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve GPU device setting consistent with train.py behavior."""
    d = str(device_arg).strip().lower()

    if d.isdigit():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", d)
    elif d.startswith("cuda:"):
        gpu_id = d.split(":", 1)[1]
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid --device value: {device_arg}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    elif d == "cuda":
        pass
    else:
        raise ValueError(
            f"Invalid --device value: {device_arg}. "
            "Use 'cuda', '<gpu_id>' (e.g. '3'), or 'cuda:<gpu_id>' (e.g. 'cuda:3')."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This evaluation script is GPU-only.")

    return torch.device("cuda")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[OffTargetModel, Dict]:
    """Load model and checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = OffTargetModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict(model: OffTargetModel, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels, probabilities and mismatch_count from a loader."""
    model.eval()

    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_mismatch: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            pair_ids = batch["pair_ids"].to(device)
            logits = model(pair_ids)
            probs = torch.sigmoid(logits)

            all_labels.append(batch["label"].detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_mismatch.append(batch["mismatch_count"].detach().cpu())

    labels = torch.cat(all_labels).numpy().astype(int)
    probabilities = torch.cat(all_probs).numpy().astype(float)
    mismatch_counts = torch.cat(all_mismatch).numpy().astype(int)

    return labels, probabilities, mismatch_counts


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, probs))
    except Exception:
        return float("nan")


def find_best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Find threshold on validation set that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    # thresholds has length n-1 relative to precision/recall
    if thresholds.size == 0:
        threshold = 0.5
    else:
        p = precision[:-1]
        r = recall[:-1]
        denom = p + r
        f1_scores = np.where(denom > 0, 2.0 * p * r / denom, 0.0)
        best_idx = int(np.nanargmax(f1_scores))
        threshold = float(thresholds[best_idx])

    preds = (probs >= threshold).astype(int)
    p_at_t = float(precision_score(labels, preds, zero_division=0))
    r_at_t = float(recall_score(labels, preds, zero_division=0))
    f1_at_t = float(f1_score(labels, preds, zero_division=0))
    mcc_at_t = float(matthews_corrcoef(labels, preds))

    return threshold, p_at_t, r_at_t, f1_at_t, mcc_at_t


def calculate_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calculate threshold-independent and threshold-dependent metrics."""
    preds = (probs >= threshold).astype(int)

    auroc = safe_auroc(labels, probs)
    auprc = safe_auprc(labels, probs)

    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    mcc = float(matthews_corrcoef(labels, preds))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_by_mismatch(
    labels: np.ndarray,
    probs: np.ndarray,
    mismatch_counts: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate MM1-MM6 groups on test set (20nt protospacer mismatch counts)."""
    rows = []

    for mm in [1, 2, 3, 4, 5, 6]:
        mask = mismatch_counts == mm
        y = labels[mask]
        p = probs[mask]

        samples = int(mask.sum())
        positive = int(y.sum()) if samples > 0 else 0
        negative = int(samples - positive)

        if samples == 0:
            row = {
                "mismatch_count": mm,
                "samples": 0,
                "positive": 0,
                "negative": 0,
                "AUROC": float("nan"),
                "AUPRC": float("nan"),
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "MCC": float("nan"),
            }
            rows.append(row)
            continue

        preds = (p >= threshold).astype(int)

        if positive > 0 and negative > 0:
            auroc = safe_auroc(y, p)
            auprc = safe_auprc(y, p)
        else:
            auroc = float("nan")
            auprc = float("nan")

        precision = float(precision_score(y, preds, zero_division=0))
        recall = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        mcc = float(matthews_corrcoef(y, preds))

        row = {
            "mismatch_count": mm,
            "samples": samples,
            "positive": positive,
            "negative": negative,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_section_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.predictions_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

    mismatch_metrics_path = str(PROJECT_ROOT / "results" / "offtarget_pam_agnostic_test_by_mismatch.csv")
    os.makedirs(os.path.dirname(mismatch_metrics_path), exist_ok=True)

    device = resolve_device(args.device)

    model, checkpoint = load_model(args.checkpoint_path, device)
    ckpt_epoch = checkpoint.get("epoch", "N/A")
    ckpt_best_auprc = checkpoint.get("best_val_auprc", float("nan"))

    print_section_header("CHECKPOINT")
    print(f"Best epoch: {ckpt_epoch}")
    print(f"Best validation AUPRC: {ckpt_best_auprc}")
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Data loaders (no resampling, no reshuffling)
    val_loader = create_dataloader(
        csv_path=args.val_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        csv_path=args.test_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Validation prediction and threshold selection
    val_labels, val_probs, _ = predict(model, val_loader, device)
    val_auroc = safe_auroc(val_labels, val_probs)
    val_auprc = safe_auprc(val_labels, val_probs)

    best_threshold, val_p, val_r, val_f1, val_mcc = find_best_f1_threshold(val_labels, val_probs)

    print_section_header("VALIDATION THRESHOLD")
    print(f"Validation AUROC: {val_auroc:.6f}")
    print(f"Validation AUPRC: {val_auprc:.6f}")
    print(f"Optimal threshold: {best_threshold:.6f}")
    print(f"Precision: {val_p:.6f}")
    print(f"Recall: {val_r:.6f}")
    print(f"F1: {val_f1:.6f}")
    print(f"MCC: {val_mcc:.6f}")

    # Test prediction (single pass)
    test_labels, test_probs, test_mismatch_counts = predict(model, test_loader, device)

    # Threshold = 0.5
    test_metrics_05 = calculate_metrics(test_labels, test_probs, threshold=0.5)

    print_section_header("TEST — THRESHOLD 0.5")
    print(f"AUROC: {test_metrics_05['AUROC']:.6f}")
    print(f"AUPRC: {test_metrics_05['AUPRC']:.6f}")
    print(f"Precision: {test_metrics_05['Precision']:.6f}")
    print(f"Recall: {test_metrics_05['Recall']:.6f}")
    print(f"F1: {test_metrics_05['F1']:.6f}")
    print(f"MCC: {test_metrics_05['MCC']:.6f}")
    print(f"Specificity: {test_metrics_05['Specificity']:.6f}")
    print("\nConfusion matrix:")
    print(f"TN: {test_metrics_05['TN']}")
    print(f"FP: {test_metrics_05['FP']}")
    print(f"FN: {test_metrics_05['FN']}")
    print(f"TP: {test_metrics_05['TP']}")

    # Threshold = validation-derived optimal threshold
    test_metrics_val_thr = calculate_metrics(test_labels, test_probs, threshold=best_threshold)

    print_section_header("TEST — VALIDATION-DERIVED THRESHOLD")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"AUROC: {test_metrics_val_thr['AUROC']:.6f}")
    print(f"AUPRC: {test_metrics_val_thr['AUPRC']:.6f}")
    print(f"Precision: {test_metrics_val_thr['Precision']:.6f}")
    print(f"Recall: {test_metrics_val_thr['Recall']:.6f}")
    print(f"F1: {test_metrics_val_thr['F1']:.6f}")
    print(f"MCC: {test_metrics_val_thr['MCC']:.6f}")
    print(f"Specificity: {test_metrics_val_thr['Specificity']:.6f}")
    print("\nConfusion matrix:")
    print(f"TN: {test_metrics_val_thr['TN']}")
    print(f"FP: {test_metrics_val_thr['FP']}")
    print(f"FN: {test_metrics_val_thr['FN']}")
    print(f"TP: {test_metrics_val_thr['TP']}")

    # Save predictions.csv (keep all original test columns)
    test_df = pd.read_csv(args.test_path)
    if len(test_df) != len(test_probs):
        raise RuntimeError(
            f"Prediction count ({len(test_probs)}) does not match test rows ({len(test_df)})."
        )

    test_df["prediction_probability"] = test_probs
    test_df["prediction_fixed_0.5"] = (test_probs >= 0.5).astype(int)
    test_df["prediction_val_threshold"] = (test_probs >= best_threshold).astype(int)
    test_df.to_csv(args.predictions_path, index=False)

    # Save metrics.csv
    metrics_rows = [
        {
            "evaluation": "test_fixed_0.5",
            "threshold": test_metrics_05["threshold"],
            "AUROC": test_metrics_05["AUROC"],
            "AUPRC": test_metrics_05["AUPRC"],
            "Precision": test_metrics_05["Precision"],
            "Recall": test_metrics_05["Recall"],
            "F1": test_metrics_05["F1"],
            "MCC": test_metrics_05["MCC"],
            "Specificity": test_metrics_05["Specificity"],
            "TN": test_metrics_05["TN"],
            "FP": test_metrics_05["FP"],
            "FN": test_metrics_05["FN"],
            "TP": test_metrics_05["TP"],
        },
        {
            "evaluation": "test_val_optimal_threshold",
            "threshold": test_metrics_val_thr["threshold"],
            "AUROC": test_metrics_val_thr["AUROC"],
            "AUPRC": test_metrics_val_thr["AUPRC"],
            "Precision": test_metrics_val_thr["Precision"],
            "Recall": test_metrics_val_thr["Recall"],
            "F1": test_metrics_val_thr["F1"],
            "MCC": test_metrics_val_thr["MCC"],
            "Specificity": test_metrics_val_thr["Specificity"],
            "TN": test_metrics_val_thr["TN"],
            "FP": test_metrics_val_thr["FP"],
            "FN": test_metrics_val_thr["FN"],
            "TP": test_metrics_val_thr["TP"],
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(args.metrics_path, index=False)

    # Mismatch stratified evaluation (MM1-MM6) using val-derived threshold
    mismatch_df = evaluate_by_mismatch(
        labels=test_labels,
        probs=test_probs,
        mismatch_counts=test_mismatch_counts,
        threshold=best_threshold,
    )
    mismatch_df.to_csv(mismatch_metrics_path, index=False)

    print_section_header("TEST BY MISMATCH COUNT")
    print(mismatch_df.to_string(index=False))

    print("\nSaved files:")
    print(f"Predictions: {args.predictions_path}")
    print(f"Metrics: {args.metrics_path}")
    print(f"Mismatch metrics: {mismatch_metrics_path}")


if __name__ == "__main__":
    main()
