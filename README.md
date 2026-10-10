# CIFAR-10 Image Classification with PyTorch

A from-scratch convolutional neural network (VGG-style) trained on CIFAR-10,
with modern training practices: augmentation, AdamW, cosine LR schedule,
label smoothing, gradient clipping, checkpointing and early stopping.

## Model

`TinyVGG` — three conv blocks (64→128→256 channels, BatchNorm + ReLU),
global average pooling and a linear classifier (~1.2M parameters).
Small enough to train on CPU, strong enough to learn real features.

## Project structure

```
pytorch-cifar10-classifier/
├── src/
│   ├── data.py      # CIFAR-10 loaders + train-time augmentation
│   ├── model.py     # TinyVGG architecture
│   ├── train.py     # training loop, scheduling, checkpointing
│   ├── evaluate.py  # accuracy, per-class report, confusion matrix
│   └── predict.py   # single-image inference CLI
└── configs/
    └── default.yaml # hyperparameters
```

## Quickstart

```bash
pip install -r requirements.txt

# Train (30 epochs, ~1-2h on CPU, minutes on GPU)
python src/train.py

# Quick correctness check (1 epoch, 512 samples)
python src/train.py --smoke

# Evaluate the best checkpoint
python src/evaluate.py

# Classify your own image
python src/predict.py --image my_photo.png --topk 3
```

## Training details

- **Augmentation**: random crop, horizontal flip, rotation
- **Optimiser**: AdamW (weight decay 5e-4) + cosine annealing schedule
- **Regularisation**: label smoothing (0.05), dropout, gradient clipping
- **Selection**: best checkpoint by validation accuracy, early stopping
