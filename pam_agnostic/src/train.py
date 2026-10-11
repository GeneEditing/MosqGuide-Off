"""Train the PAM-agnostic 20nt off-target model with focal loss.

This reuses the focal-loss training scheme from
`pam_ngg/src/train.py` (currently the
best-performing DL training scheme in the NGG project), adapted to the
20nt PAM-agnostic dataset and model.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from dataset import create_dataloader
from model import OffTargetModel


class BinaryFocalLoss(nn.Module):
    """Binary focal loss on logits for imbalanced classification."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic setup (may reduce speed, but improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        pair_ids = batch["pair_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(pair_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bsz = labels.size(0)
        total_loss += loss.item() * bsz
        total_samples += bsz

    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate on validation set.

    Notes:
    - Off-target data is highly imbalanced.
    - We monitor AUPRC as the primary metric because it focuses on positive-class
      precision-recall quality under imbalance.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            pair_ids = batch["pair_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(pair_ids)
            loss = criterion(logits, labels)

            bsz = labels.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)

    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np = (probs_np >= 0.5).astype(int)

    val_loss = total_loss / max(total_samples, 1)

    # Safe AUROC for single-class edge case
    if len(np.unique(labels_np)) < 2:
        val_auroc = float("nan")
    else:
        val_auroc = float(roc_auc_score(labels_np, probs_np))

    val_auprc = float(average_precision_score(labels_np, probs_np))
    val_f1 = float(f1_score(labels_np, preds_np, zero_division=0))
    val_mcc = float(matthews_corrcoef(labels_np, preds_np))

    return {
        "val_loss": val_loss,
        "val_auroc": val_auroc,
        "val_auprc": val_auprc,
        "val_f1": val_f1,
        "val_mcc": val_mcc,
    }


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_auprc: float,
    args: argparse.Namespace,
) -> None:
    """Save best checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_auprc": best_val_auprc,
        "args": vars(args),
    }
    torch.save(payload, checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PAM-agnostic 20nt off-target model (focal loss)")

    parser.add_argument(
        "--train_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "train.csv"),
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "val.csv"),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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
        "--checkpoint_path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "offtarget_pam_agnostic_focal_best.pt"
        ),
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha for focal loss. Typical range: 0.1~0.75",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss. Typical range: 1.0~3.0",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Device selection (GPU-only by request; no CPU fallback)
    # An externally set CUDA_VISIBLE_DEVICES always wins over --device.
    device_arg = str(args.device).strip().lower()

    if device_arg.isdigit():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", device_arg)
    elif device_arg.startswith("cuda:"):
        gpu_id = device_arg.split(":", 1)[1]
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid --device value: {args.device}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    elif device_arg == "cuda":
        pass
    else:
        raise ValueError(
            f"Invalid --device value: {args.device}. "
            "Use 'cuda', '<gpu_id>' (e.g. '3'), or 'cuda:<gpu_id>' (e.g. 'cuda:3')."
        )

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. This script is configured for GPU-only training."
        )

    # If CUDA_VISIBLE_DEVICES is set to a single GPU id, this process sees it as cuda:0.
    device = torch.device("cuda")

    print(f"Device: {device}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # DataLoaders
    train_loader = create_dataloader(
        csv_path=args.train_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        csv_path=args.val_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # One-time shape sanity check from first train batch
    first_batch = next(iter(train_loader))
    print(f"First train batch pair_ids shape: {first_batch['pair_ids'].shape}")
    print(f"First train batch labels shape: {first_batch['label'].shape}")

    # Model
    model = OffTargetModel().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Loss and optimizer (Focal Loss)
    criterion = BinaryFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        reduction="mean",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auprc = -float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)

        val_loss = metrics["val_loss"]
        val_auroc = metrics["val_auroc"]
        val_auprc = metrics["val_auprc"]
        val_f1 = metrics["val_f1"]
        val_mcc = metrics["val_mcc"]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"AUROC: {val_auroc:.6f} | "
            f"AUPRC: {val_auprc:.6f} | "
            f"F1: {val_f1:.6f} | "
            f"MCC: {val_mcc:.6f}"
        )

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoint_path=args.checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_auprc=best_auprc,
                args=args,
            )
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUPRC: {best_auprc:.6f}")
    print(f"Best checkpoint path: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
