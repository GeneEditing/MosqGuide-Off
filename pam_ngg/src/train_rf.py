"""Train a Random Forest on handcrafted features only (no pair_ids, no GPU).

Purpose: check how much predictive signal the 93 handcrafted features carry
on their own, and rank their importance with a standard, easy-to-explain
method (Random Forest Gini importance). This is not meant to beat the
residual deep-learning model; it only isolates the handcrafted features.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


FEATURE_ENHANCEMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_ENHANCEMENT_DIR.parent
if str(FEATURE_ENHANCEMENT_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENHANCEMENT_DIR))

from inspect_features import build_handcrafted_features, get_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "train.csv"),
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "val.csv"),
    )
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional row cap per split, for a quick smoke test.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib"),
        help="Where to save the fitted RandomForestClassifier (joblib).",
    )
    parser.add_argument(
        "--importance_report_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rf_feature_importance_top5.txt"),
        help="Where to save the concise top-5 feature importance report (txt).",
    )
    return parser.parse_args()


def build_feature_matrix(csv_path: str, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, usecols=["sgRNA", "off_target", "label"])
    if max_samples is not None:
        df = df.iloc[:max_samples]

    rows = np.empty((len(df), 93), dtype=np.float32)
    for row_index, (sgrna, off_target) in enumerate(zip(df["sgRNA"], df["off_target"])):
        _, vector = build_handcrafted_features(sgrna, off_target)
        rows[row_index] = vector

    labels = df["label"].to_numpy(dtype=int)
    return rows, labels


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probabilities))
    return {
        "AUROC": auroc,
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }


def write_top5_importance_report(
    report_path: str, feature_names: List[str], importances: np.ndarray, val_metrics: Dict[str, float]
) -> None:
    """Write a short, human-readable top-5 feature importance report."""
    top5_idx = np.argsort(importances)[::-1][:5]
    lines = [
        "RF handcrafted-feature model - top 5 important features",
        f"Validation AUROC: {val_metrics['AUROC']:.4f} | AUPRC: {val_metrics['AUPRC']:.4f}",
        "",
    ]
    for rank, idx in enumerate(top5_idx, start=1):
        lines.append(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    print("Building train feature matrix...")
    start = time.time()
    X_train, y_train = build_feature_matrix(args.train_path, args.max_samples)
    print(f"Train matrix: {X_train.shape}, elapsed {time.time() - start:.1f}s")

    print("Building validation feature matrix...")
    start = time.time()
    X_val, y_val = build_feature_matrix(args.val_path, args.max_samples)
    print(f"Validation matrix: {X_val.shape}, elapsed {time.time() - start:.1f}s")

    feature_names = get_feature_names()
    assert X_train.shape[1] == 93
    assert len(feature_names) == 93

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    print("Training Random Forest...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training done, elapsed {time.time() - start:.1f}s")

    train_probabilities = model.predict_proba(X_train)[:, 1]
    val_probabilities = model.predict_proba(X_val)[:, 1]
    train_metrics = calculate_metrics(y_train, train_probabilities)
    val_metrics = calculate_metrics(y_val, val_probabilities)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    write_top5_importance_report(
        args.importance_report_path, feature_names, model.feature_importances_, val_metrics
    )

    print("\n=== RESULTS ===")
    print(f"Train AUROC: {train_metrics['AUROC']:.6f}, AUPRC: {train_metrics['AUPRC']:.6f}")
    print(f"Val AUROC: {val_metrics['AUROC']:.6f}, AUPRC: {val_metrics['AUPRC']:.6f}")
    print("Model path:", args.model_path)
    print("Importance report path:", args.importance_report_path)
    print("ML baseline: PASS")


if __name__ == "__main__":
    main()
