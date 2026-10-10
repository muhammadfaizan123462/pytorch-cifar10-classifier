"""Evaluate the best checkpoint: accuracy, per-class accuracy, confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from data import CLASSES, get_loaders  # noqa: E402
from model import TinyVGG  # noqa: E402


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the trained model.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyVGG(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val acc {ckpt['val_acc']:.4f})")

    _, test_loader = get_loaders(batch_size=args.batch_size)
    all_pred, all_true = [], []
    for x, y in test_loader:
        all_pred += model(x.to(device)).argmax(1).cpu().tolist()
        all_true += y.tolist()

    acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    print(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(all_true, all_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for cls, a in zip(CLASSES, per_class):
        print(f"  {cls:>6}: {a:.3f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path("checkpoints").mkdir(exist_ok=True)
    fig.savefig("checkpoints/confusion_matrix.png", dpi=120)
    print("Saved checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    main()
