"""
evaluate.py
===========
Evaluate a trained AestheticMobileNet on the test split.

Usage
-----
python evaluate.py
python evaluate.py --task classification
python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import get_dataloaders, CLASS_NAMES
from model import build_model


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate AestheticMobileNet"
    )

    parser.add_argument(
        "--pairs_csv",
        default="image_label_pairs.csv"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth"
    )

    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification"]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--save_dir",
        default="eval_results"
    )

    return parser.parse_args()


# ============================================================
# Prediction Loop
# ============================================================

@torch.no_grad()
def get_predictions(model, loader, device):

    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in loader:

        images = images.to(device)

        outputs = model(images)

        # Classification
        if outputs.dim() > 1:

            preds = outputs.argmax(1).cpu().numpy()

        # Regression
        else:

            preds = outputs.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


# ============================================================
# Regression Evaluation
# ============================================================

def evaluate_regression(preds, targets, save_dir):

    # --------------------------------------------------------
    # Convert normalized scores back to [1,10]
    # --------------------------------------------------------
    preds = preds * 9.0 + 1.0
    targets = targets * 9.0 + 1.0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    mae = mean_absolute_error(targets, preds)

    rmse = np.sqrt(
        mean_squared_error(targets, preds)
    )

    try:
        pearson_r, _ = pearsonr(targets, preds)
    except:
        pearson_r = 0.0

    try:
        spearman_rho, _ = spearmanr(targets, preds)
    except:
        spearman_rho = 0.0

    print("\n" + "=" * 55)
    print("Regression Metrics")
    print("=" * 55)

    print(f"MAE           : {mae:.4f}")
    print(f"RMSE          : {rmse:.4f}")
    print(f"Pearson r     : {pearson_r:.4f}")
    print(f"Spearman rho  : {spearman_rho:.4f}")

    print("=" * 55 + "\n")

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Pearson_r": float(pearson_r),
        "Spearman_rho": float(spearman_rho)
    }

    metrics_path = os.path.join(
        save_dir,
        "regression_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval] Saved metrics -> {metrics_path}")

    # ========================================================
    # Visualization
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------------
    axes[0].scatter(
        targets,
        preds,
        alpha=0.3,
        s=10
    )

    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())

    axes[0].plot(
        [lo, hi],
        [lo, hi],
        linestyle="--"
    )

    axes[0].set_title(
        f"Predicted vs Actual (r={pearson_r:.3f})"
    )

    axes[0].set_xlabel("Ground Truth Score")
    axes[0].set_ylabel("Predicted Score")

    # --------------------------------------------------------
    # Residual Histogram
    # --------------------------------------------------------
    residuals = preds - targets

    axes[1].hist(
        residuals,
        bins=50
    )

    axes[1].axvline(
        0,
        linestyle="--"
    )

    axes[1].set_title(
        f"Residuals (MAE={mae:.3f})"
    )

    axes[1].set_xlabel(
        "Residual (Predicted - Actual)"
    )

    axes[1].set_ylabel("Count")

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "regression_results.png"
    )

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"[eval] Saved plot -> {save_path}")

    return metrics


# ============================================================
# Classification Evaluation
# ============================================================

def evaluate_classification(preds, targets, save_dir):

    report = classification_report(
        targets,
        preds,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n" + "=" * 55)
    print("Classification Report")
    print("=" * 55)

    print(report)

    print("=" * 55 + "\n")

    report_path = os.path.join(
        save_dir,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print(f"[eval] Saved report -> {report_path}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------
    cm = confusion_matrix(targets, preds)

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(ax=ax, colorbar=False)

    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "confusion_matrix.png"
    )

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"[eval] Saved matrix -> {save_path}")


# ============================================================
# Training Curves
# ============================================================

def plot_training_curves(
    history_json="checkpoints/history.json",
    save_dir="eval_results"
):

    if not os.path.exists(history_json):

        print(
            f"[eval] history.json not found -> {history_json}"
        )

        return

    with open(history_json, "r") as f:
        history = json.load(f)

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # --------------------------------------------------------
    # Loss Curves
    # --------------------------------------------------------
    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Val Loss"
    )

    axes[0].set_title("Loss Curves")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[0].legend()

    # --------------------------------------------------------
    # Accuracy Curves (Classification Only)
    # --------------------------------------------------------
    if any(v is not None for v in history.get("train_acc", [])):

        axes[1].plot(
            epochs,
            history["train_acc"],
            label="Train Accuracy"
        )

        axes[1].plot(
            epochs,
            history["val_acc"],
            label="Val Accuracy"
        )

        axes[1].set_title("Accuracy Curves")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")

        axes[1].legend()

    else:

        axes[1].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "training_curves.png"
    )

    plt.savefig(save_path, dpi=150)

    plt.close()

    print(f"[eval] Saved curves -> {save_path}")


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------------
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    saved_args = checkpoint.get("args", {})

    task = saved_args.get(
        "task",
        args.task
    )

    model = build_model(
        task=task,
        pretrained=False,
        device=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print(
        f"\n[eval] Loaded checkpoint -> {args.checkpoint}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    loaders, _ = get_dataloaders(
        pairs_csv=args.pairs_csv,
        task=task,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    preds, targets = get_predictions(
        model,
        loaders["test"],
        device
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    if task == "regression":

        evaluate_regression(
            preds,
            targets,
            args.save_dir
        )

    else:

        evaluate_classification(
            preds,
            targets,
            args.save_dir
        )

    # --------------------------------------------------------
    # Curves
    # --------------------------------------------------------
    plot_training_curves(
        save_dir=args.save_dir
    )

    print("\n[eval] Evaluation Complete!\n")


if __name__ == "__main__":
    main()

