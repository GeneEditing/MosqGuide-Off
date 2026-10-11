#!/usr/bin/env bash
# Chains DL training -> RF training -> ensemble weight search -> probability
# calibration for the PAM-agnostic off-target model. Only final model weights
# are produced under checkpoints/; run with an env that has
# torch/sklearn/joblib/pandas installed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DEVICE="${DEVICE:-0}"
CALIBRATION_INPUT="results/calibration/val_raw_predictions.csv"

echo "=== [1/5] Train DL model ==="
python src/train.py --device "$DEVICE"

echo "=== [2/5] Train RF model ==="
python src/train_rf.py

echo "=== [3/5] Search DL/RF ensemble blend weight ==="
python src/ensemble_focal_rf.py --device "$DEVICE"

echo "=== [4/5] Generate VAL raw scores for calibration ==="
python scripts/generate_val_calibration_predictions.py \
    --input data/val.csv \
    --output "$CALIBRATION_INPUT" \
    --device "$DEVICE"

echo "=== [5/5] Fit probability calibrators ==="
python scripts/fit_probability_calibrators.py --input "$CALIBRATION_INPUT"

# Calibrators are persisted under checkpoints/; the raw VAL scores are disposable.
rm -rf "$PROJECT_DIR/results/calibration"

echo "Done. Weights saved under: $PROJECT_DIR/checkpoints/"
