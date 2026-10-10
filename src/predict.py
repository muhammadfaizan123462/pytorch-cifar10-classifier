"""Run inference on a single image with the trained checkpoint.

Example:
    python src/predict.py --image path/to/cat.png --topk 3
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from data import CIFAR10_MEAN, CIFAR10_STD, CLASSES
from model import TinyVGG

_tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_model(checkpoint: str, device: str) -> TinyVGG:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TinyVGG(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: TinyVGG, image_path: str, device: str, topk: int = 3):
    img = _tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(img)[0], dim=0)
    top_p, top_c = probs.topk(topk)
    return [(CLASSES[c], float(p)) for c, p in zip(top_c.tolist(), top_p.tolist())]


def main() -> None:
    p = argparse.ArgumentParser(description="Classify a single image.")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    for label, prob in predict_image(model, args.image, device, args.topk):
        print(f"{label:>6}: {prob:.3f}")


if __name__ == "__main__":
    main()
