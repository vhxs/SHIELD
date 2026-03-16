# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
#
# Train ResNet50 on CIFAR-10 with GELU activations and kurtosis regularization,
# following the procedure described in:
#   "High-Resolution Convolutional Neural Networks on Homomorphically Encrypted
#    Data via Sharding Ciphertexts" (Maloney et al., 2024)
#
# Training mirrors train_resnetN.py: Nesterov momentum, EMA validation model,
# float16 weights with float32 BatchNorm, and the same kurtosis schedule.
#
# Usage (from repo root):
#   uv run python training/train_resnet50_cifar.py
#
# The saved model is loaded by inference/resnet50_cifar_inference.py.
# Hardware: tested on NVIDIA RTX 3070 Ti (8GB VRAM), batch size 256.
# Approximate wall time: 3-5 hours for ~59 epochs.

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
import torchvision

# Allow running from either the repo root or the training/ directory.
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils_dataloading import load_cifar10, random_crop
from utils.utils_kurtosis import get_intermediate_output, get_statistics, remove_all_hooks
from utils.utils_resnetN import update_nesterov, update_ema, label_smoothing_loss


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Return a ResNet50 modified for CIFAR-10 HE-compatible inference.

    Changes vs. stock torchvision ResNet50:
    - conv1: 7×7 stride-2 → 3×3 stride-1 (CIFAR-10 images are 32×32)
    - maxpool: replaced with Identity (the HE pipeline skips the initial maxpool)
    - All ReLU activations replaced with GELU
    - fc: 1000-class head → num_classes

    The model is trained with standard 3-channel inputs. At HE inference time,
    pad_conv_input_channels() extends conv1 to accept the 4th zero-padded channel.
    """
    weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet50(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    _replace_relu_with_gelu(model)
    return model


def _replace_relu_with_gelu(model: nn.Module) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.GELU())
        else:
            _replace_relu_with_gelu(module)


def train(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(device.index)}")
    else:
        print("Device: CPU")

    # -------------------------------------------------------------------------
    # Data — loaded fully to GPU (CIFAR-10 is ~150 MB, fits comfortably).
    # train_data shape: (50000, 3, 40, 40)  — reflection-padded from 32×32
    # valid_data shape: (10000, 3, 32, 32)
    # -------------------------------------------------------------------------
    train_data, train_targets, valid_data, valid_targets = load_cifar10(
        device, dtype, args.data_dir
    )

    N = len(train_data) // args.batch_size  # batches per epoch

    # -------------------------------------------------------------------------
    # Schedules — identical structure to train_resnetN.py.
    # -------------------------------------------------------------------------
    lr = args.lr
    lr_schedule = torch.cat([
        torch.linspace(0.0, lr,  1 * N),   # warmup:   1 epoch
        torch.linspace(lr,  1e-4, 3 * N),  # decay:    3 epochs
        torch.linspace(1e-4, 1e-4, 30 * N),  # constant: 30 epochs
        torch.linspace(1e-5, 1e-5, 15 * N),  # constant: 15 epochs
        torch.linspace(1e-6, 1e-6, 10 * N),  # constant: 10 epochs
    ])  # total: 59 epochs
    lr_schedule_bias = args.lr_bias * lr_schedule

    kurt_schedule = torch.cat([
        torch.linspace(0,    0,    10 * N),   # 0     for first 10 epochs
        torch.linspace(0.05, 0.05,  2 * N),   # 0.05  for next  2 epochs
        torch.linspace(0.1,  0.1, 200 * N),   # 0.1   thereafter
    ])

    # -------------------------------------------------------------------------
    # Model — float16 with BatchNorm in float32, same as train_resnetN.py.
    # -------------------------------------------------------------------------
    train_model = build_model(num_classes=10, pretrained=args.pretrained).to(device)
    train_model.to(dtype)
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    valid_model = copy.deepcopy(train_model)  # EMA model used for validation

    ema_update_freq = 5
    ema_rho = 0.99 ** ema_update_freq

    # Nesterov state: (parameter, velocity) pairs split into weights and biases.
    weights = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    biases = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    batch_count = 0
    train_time = 0.0
    total_epochs = len(lr_schedule) // N

    print(f"\nepoch    batch    train time [sec]    validation accuracy")

    for epoch in range(1, total_epochs + 1):
        t0 = perf_counter()
        train_model.train(True)

        # Shuffle
        indices = torch.randperm(len(train_data), device=device)
        data = train_data[indices]
        targets = train_targets[indices]

        # Random 32×32 crop from 40×40 padded training images
        data = torch.cat([
            random_crop(data[i:i + args.batch_size], (32, 32))
            for i in range(0, len(data), args.batch_size)
        ])

        # Random horizontal flip on half the training data
        data[:len(data) // 2] = torch.flip(data[:len(data) // 2], [-1])

        for i in range(0, len(data), args.batch_size):
            if i + args.batch_size > len(data):
                break

            inputs = data[i:i + args.batch_size]
            target = targets[i:i + args.batch_size]
            batch_count += 1

            train_model.zero_grad()

            # Register GELU input hooks, run forward, then remove hooks.
            remove_all_hooks(train_model)
            activations = get_intermediate_output(train_model)

            logits = train_model(inputs)
            loss = label_smoothing_loss(logits, target, alpha=0.2)

            kurt_idx = min(batch_count, len(kurt_schedule) - 1)
            kurt_scale = kurt_schedule[kurt_idx]

            remove_all_hooks(train_model)
            activations = {k: torch.cat(v) for k, v in activations.items()}
            loss_means, loss_stds, loss_kurts = get_statistics(activations)
            loss += (loss_means + loss_stds + loss_kurts) * kurt_scale

            loss.sum().backward()

            lr_idx = min(batch_count, len(lr_schedule) - 1)
            update_nesterov(weights, lr_schedule[lr_idx], args.weight_decay, args.momentum)
            update_nesterov(biases, lr_schedule_bias[lr_idx], 0.004, args.momentum)

            if (i // args.batch_size % ema_update_freq) == 0:
                update_ema(train_model, valid_model, ema_rho)

        train_time += perf_counter() - t0

        # Validation using the EMA model
        valid_correct = []
        for i in range(0, len(valid_data), args.batch_size):
            valid_model.train(False)
            logits = valid_model(valid_data[i:i + args.batch_size]).detach()
            correct = logits.max(dim=1)[1] == valid_targets[i:i + args.batch_size]
            valid_correct.append(correct.detach().to(torch.float64))

        valid_acc = torch.mean(torch.cat(valid_correct)).item()
        print(f"{epoch:5d} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            # Save in float32 so weight extraction at inference works correctly.
            torch.save(copy.deepcopy(valid_model).float(), str(output_path))
            print(f"  -> Saved (val={100 * valid_acc:.2f}%)")

    print(f"\nDone. Best val_acc={100 * best_acc:.2f}%, saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on CIFAR-10 for HE-compatible inference"
    )
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size; 256 fits in 8 GB VRAM (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Peak Nesterov learning rate (default: 0.001)")
    parser.add_argument("--lr-bias", type=float, default=64.0,
                        help="Bias LR multiplier — lr_schedule_bias = lr_bias * lr_schedule "
                             "(default: 64.0)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Nesterov momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.256,
                        help="Weight decay for non-bias parameters (default: 0.256)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="CIFAR-10 data directory (auto-downloaded if absent)")
    parser.add_argument("--output", type=str, default="weights/resnet50_cifar_gelu_kurt.pt",
                        help="Path to save the best model")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Train from scratch instead of using ImageNet pretrained weights")
    args = parser.parse_args()

    train(args)
