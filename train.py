"""
train.py
========
Fine-tune MobileNetV3-Small on the AVA aesthetic dataset.

Usage
-----
python train.py                        # regression, default settings
python train.py --task classification  # 3-class (Low / Medium / High)
python train.py --epochs 20 --lr 1e-4 --batch_size 64
"""

import os
import argparse
import time
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model   import build_model


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train AestheticMobileNet')
    p.add_argument('--pairs_csv',   default='image_label_pairs.csv')
    p.add_argument('--task',        default='regression',
                   choices=['regression', 'classification'])
    p.add_argument('--epochs',      type=int,   default=3)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--dropout',     type=float, default=0.3)
    p.add_argument('--img_size',    type=int,   default=224)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--save_dir',    default='checkpoints')
    p.add_argument('--freeze_backbone', action='store_true',
                   help='Freeze MobileNet features; train head only')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs    = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss  = criterion(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            total      += len(imgs)

            # Accuracy for classification only
            if preds.dim() > 1:                    # (B, num_classes)
                correct += (preds.argmax(1) == targets).sum().item()

    avg_loss = total_loss / total
    acc      = correct / total if correct else None
    return avg_loss, acc


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[train] Device: {device}  |  Task: {args.task}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    loaders, _ = get_dataloaders(
        pairs_csv   = args.pairs_csv,
        task        = args.task,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(task=args.task, dropout=args.dropout, device=device)

    if args.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        print("[train] Backbone frozen — training head only.")

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer & scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ───────────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, loaders['train'], criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(
            model, loaders['val'],   criterion, optimizer, device, train=False)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        acc_str = (f"  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
                   if train_acc is not None else "")
        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"{acc_str}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"[{elapsed:.1f}s]")

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'val_loss':   val_loss,
                'args':       vars(args),
            }, ckpt_path)
            print(f"           ✔ Best model saved → {ckpt_path}")

    print(f"\n[train] Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    # ── Save history ─────────────────────────────────────────────────────────
    hist_path = os.path.join(args.save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[train] History saved → {hist_path}\n")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    train(args)