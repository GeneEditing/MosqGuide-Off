#!/usr/bin/env python3
"""Predict calibrated off-target probabilities for supplied sequence pairs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {"ngg": "pam_ngg", "pam-agnostic": "pam_agnostic"}
REQUIRED_COLUMNS = (
    "candidate_id", "guide_id", "sgRNA_20nt", "off_target_20nt",
    "candidate_pam", "guide_pam",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "demo_pairs.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "demo_predictions.csv")
    parser.add_argument("--model-profile", choices=MODEL_DIRS, default="ngg")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<GPU index>")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    return args


def read_pairs(path, profile):
    """Validate supplied pairs without loading models or training data."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("Input CSV contains duplicate column names")
        missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains no sequence pairs")
    seen = set()
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {index}: column count does not match header")
        for column in REQUIRED_COLUMNS:
            row[column] = row[column].strip()
            if not row[column]:
                raise ValueError(f"CSV row {index}: {column} is empty")
        if row["candidate_id"] in seen:
            raise ValueError(f"CSV row {index}: duplicate candidate_id")
        seen.add(row["candidate_id"])
        for column in ("sgRNA_20nt", "off_target_20nt", "candidate_pam", "guide_pam"):
            value = row[column].upper()
            row[column] = value
            if set(value) - set("ACGT"):
                raise ValueError(f"CSV row {index}: {column} must contain concrete A/C/G/T bases")
            if column.endswith("20nt") and len(value) != 20:
                raise ValueError(f"CSV row {index}: {column} must be exactly 20 nt")
            if profile == "ngg" and column.endswith("pam"):
                if len(value) != 3 or value[1:] != "GG":
                    raise ValueError(f"CSV row {index}: {column} must match NGG for the NGG model")
    return columns, rows


def model_sequences(row, profile):
    guide, target = row["sgRNA_20nt"], row["off_target_20nt"]
    if profile == "ngg":
        guide += row["guide_pam"]
        target += row["candidate_pam"]
    return guide, target


def load_profile(profile):
    base = ROOT / MODEL_DIRS[profile]
    config_path = base / "checkpoints" / "ensemble_focal_rf_weight.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight = float(config["blend_weight_dl"])
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid blend_weight_dl in ensemble configuration")
    paths = {
        "dl": base / config["dl_checkpoint_path"],
        "rf": base / config["rf_model_path"],
        "dl_calibrator": base / "checkpoints" / "dl_probability_calibrator.joblib",
        "rf_calibrator": base / "checkpoints" / "rf_probability_calibrator.joblib",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required model file missing: {path}. See README downloads.")
    return base, paths, weight


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_sources(base):
    """Isolate same-named model modules without changing training scripts."""
    name = base.name
    dataset = load_source(base / "src" / "dataset.py", f"{name}_demo_dataset")
    previous = sys.modules.get("dataset")
    sys.modules["dataset"] = dataset
    try:
        model = load_source(base / "src" / "model.py", f"{name}_demo_model")
    finally:
        if previous is None:
            sys.modules.pop("dataset", None)
        else:
            sys.modules["dataset"] = previous
    features = load_source(base / "src" / "inspect_features.py", f"{name}_demo_features")
    return dataset, model, features


def apply_calibrator(bundle, scores, np):
    if not isinstance(bundle, dict) or not {"method", "model"} <= bundle.keys():
        raise ValueError("Invalid probability calibrator bundle")
    scores = np.asarray(scores, dtype=float)
    if bundle["method"] == "platt":
        values = bundle["model"].predict_proba(scores.reshape(-1, 1))[:, 1]
    elif bundle["method"] == "isotonic":
        values = bundle["model"].predict(scores)
    else:
        raise ValueError(f"Unknown calibration method: {bundle['method']}")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != len(scores) or not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
        raise ValueError("Calibrator returned invalid probabilities")
    return values


def predict(rows, profile, device_arg, batch_size):
    base, paths, weight = load_profile(profile)
    # Lazy imports keep --help and input/configuration validation dependency-free.
    import joblib
    import numpy as np
    import torch

    device = torch.device(device_arg)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("--device must be cpu, cuda, or cuda:<GPU index>")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    dataset_source, model_source, features_source = load_model_sources(base)
    checkpoint = torch.load(paths["dl"], map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    dl_model = model_source.OffTargetModel()
    dl_model.load_state_dict(state)
    dl_model.to(device)
    dl_model.eval()
    rf_model = joblib.load(paths["rf"])
    feature_dim = 93 if profile == "ngg" else 81
    if getattr(rf_model, "n_features_in_", None) != feature_dim:
        raise ValueError(f"RF model does not match {profile}: expected {feature_dim} features")
    if list(rf_model.classes_) != [0, 1]:
        raise ValueError("RF model must have class order [0, 1]")
    dl_calibrator = joblib.load(paths["dl_calibrator"])
    rf_calibrator = joblib.load(paths["rf_calibrator"])
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            pairs = [model_sequences(row, profile) for row in rows[start:start + batch_size]]
            tokens = [dataset_source.encode_pair(guide, target) for guide, target in pairs]
            pair_ids = torch.tensor(tokens, dtype=torch.long, device=device)
            dl_raw = torch.sigmoid(dl_model(pair_ids)).detach().cpu().numpy().reshape(-1).astype(np.float32)
            vectors = [features_source.build_handcrafted_features(guide, target)[1] for guide, target in pairs]
            features = np.asarray(vectors, dtype=np.float32)
            if features.shape != (len(pairs), feature_dim) or not np.isfinite(features).all():
                raise ValueError("Invalid RF feature matrix")
            rf_raw = rf_model.predict_proba(features)[:, 1].astype(np.float32)
            dl_probability = apply_calibrator(dl_calibrator, dl_raw, np)
            rf_probability = apply_calibrator(rf_calibrator, rf_raw, np)
            blended = (weight * dl_probability + (1.0 - weight) * rf_probability).astype(np.float32)
            if not np.all(np.isfinite(blended) & (blended >= 0) & (blended <= 1)):
                raise ValueError("Invalid final probabilities")
            probabilities.extend(blended.tolist())
    return probabilities


def main(argv=None):
    args = parse_args(argv)
    columns, rows = read_pairs(args.input, args.model_profile)
    probabilities = predict(rows, args.model_profile, args.device, args.batch_size)
    output_columns = [column for column in columns if column != "off_target_probability"] + ["off_target_probability"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow({**row, "off_target_probability": probability})
    print(f"Scored {len(rows)} pair(s) with {args.model_profile}; output: {args.output}")
    if len(rows) == 1:
        print(f"off_target_probability: {probabilities[0]:.8f}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc