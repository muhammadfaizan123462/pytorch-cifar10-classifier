"""Training loop: augmentation + AdamW + cosine schedule + checkpointing.

Examples:
    python src/train.py --epochs 30
    python src/train.py --smoke            # 1 epoch on 512 samples (CI check)
    python src/train.py --config configs/default.yaml --lr 0.002
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_loaders
from model import TinyVGG, count_parameters

CHECKPOINT_DIR = Path("checkpoints")


def load_config(path: str | None) -> dict:
    cfg: dict = {}
    if path:
        cfg = yaml.safe_load(Path(path).read_text())
    return cfg


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        running_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="Train TinyVGG on CIFAR-10.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch on a tiny subset (correctness check)")
    args = p.parse_args()

    cfg = load_config(args.config)
    epochs = 1 if args.smoke else (args.epochs or cfg.get("epochs", 30))
    lr = args.lr or cfg.get("lr", 3e-3)
    batch_size = args.batch_size or cfg.get("batch_size", 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | epochs: {epochs} | lr: {lr}")

    train_loader, test_loader = get_loaders(
        batch_size=batch_size, smoke=args.smoke)
    model = TinyVGG(num_classes=10,
                    dropout=0.0 if args.smoke else cfg.get("dropout", 0.25))
    model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history, best_acc, patience, best_epoch = [], 0.0, 0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, va_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                        "val_acc": round(va_acc, 4)})
        print(f"Epoch {epoch:3d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.3f} "
              f"| val {va_loss:.4f}/{va_acc:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if va_acc > best_acc + 1e-4:
            best_acc, best_epoch, patience = va_acc, epoch, 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "val_acc": va_acc},
                       CHECKPOINT_DIR / "best.pt")
        else:
            patience += 1
        if patience >= cfg.get("early_stopping_patience", 8) and not args.smoke:
            print(f"Early stopping at epoch {epoch}.")
            break

    mins = (time.time() - start) / 60
    print(f"Best val acc: {best_acc:.4f} (epoch {best_epoch}) in {mins:.1f} min")
    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    if math.isnan(best_acc):
        raise RuntimeError("Training diverged (NaN accuracy).")


if __name__ == "__main__":
    main()
