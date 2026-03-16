# Training

Scripts for training CIFAR-10 models suitable for homomorphic encryption inference.
Results target those reported in the paper:
> *High-Resolution Convolutional Neural Networks on Homomorphically Encrypted Data via Sharding Ciphertexts* (Maloney et al., 2024)

## Setup

From the repo root:

```bash
uv sync
```

CIFAR-10 downloads automatically on first run.

## Running

All architectures are trained with `train_cifar10.py`, run from the **repo root**:

```bash
uv run python training/train_cifar10.py --arch <arch> [options]
```

### ResNet50 (recommended starting point)

Single run, ~3–5 hours on an NVIDIA RTX 3070 Ti (8 GB VRAM).

```bash
uv run python training/train_cifar10.py --arch resnet50 --data-dir ./data
```

Target: **98.3%** validation accuracy. Saves to `weights/resnet50_CIFAR10.pt`.

### Multiplexed ResNets (ResNet20/32/44/56/110)

Five independent runs each; results are reported as mean ± std.

```bash
uv run python training/train_cifar10.py --arch resnet32 --nruns 5 --data-dir ./data
```

| Architecture | Target accuracy    | Approx. time per run |
|--------------|--------------------|----------------------|
| resnet20     | 90.6% ± 0.3%       | ~1 h                 |
| resnet32     | 92.2% ± 0.2%       | ~1.5 h               |
| resnet44     | 92.2% ± 0.1%       | ~2 h                 |
| resnet56     | 92.8% ± 0.2%       | ~2.5 h               |
| resnet110    | 92.7% ± 0.2%       | ~4 h                 |

Saves to `weights/resnet{N}_CIFAR10_run{i}.pt` for each run `i`.

### ResNet9

Five independent runs, ~30 min each.

```bash
uv run python training/train_cifar10.py --arch resnet9 --nruns 5 --data-dir ./data
```

Target: **94.5% ± 0.1%** (best: 94.7%). Saves to `weights/resnet9_CIFAR10_run{i}.pt`.

## Smoke test

Before committing to a full run, verify the script works end-to-end with two epochs:

```bash
uv run python training/train_cifar10.py --arch resnet50 --max-epochs 2 --data-dir ./data
```

Then confirm the saved model has the right structure:

```bash
uv run python -c "
import torch
m = torch.load('weights/resnet50_CIFAR10.pt', weights_only=False)
print(m.conv1)    # Conv2d(3, 64, kernel_size=3, stride=1)
print(m.maxpool)  # Identity
print(m.fc)       # Linear(2048, 10)
"
```

## Output

| File | Description |
|------|-------------|
| `weights/<arch>_CIFAR10.pt` | Best EMA model (full model, float32) |
| `weights/<arch>_CIFAR10_run{i}.pt` | Per-run weights for multi-run architectures |
| `logs/<arch>_CIFAR10.json` | Accuracy per run, mean, std |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | *(required)* | `resnet9`, `resnet20`, `resnet32`, `resnet44`, `resnet56`, `resnet110`, `resnet50` |
| `--data-dir` | `./data` | CIFAR-10 data directory (downloaded automatically) |
| `--nruns` | 5 (small archs), 1 (resnet50) | Number of independent training runs |
| `--max-epochs` | None | Hard epoch cap — use `--max-epochs 2` for smoke testing |
| `--cuda` | 0 | CUDA device index |
| `--batch-size` | 512 (resnet9), 256 (others) | Training batch size |
| `--no-pretrained` | — | Train ResNet50 from scratch (omit for ImageNet init) |
| `--lr`, `--momentum`, `--weight-decay` | arch-specific | Override Optuna-tuned defaults |

## Notes

- **ResNet50** is trained with ImageNet pretrained weights by default. The stem is modified for 32×32 input: `conv1` is replaced with a 3×3 stride-1 convolution and `maxpool` with `Identity`.
- All architectures use **GELU** activations and **kurtosis regularization** to keep pre-activation distributions close to Gaussian, which is required for the polynomial GELU approximation used in HE inference.
- Training uses **float16** (ResNet50) or **float32** (all others), with BatchNorm layers kept in float32 throughout.
- Saved models are passed directly to the inference scripts in `inference/`.
