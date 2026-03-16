# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
#
# Unified CIFAR-10 training script for all architectures evaluated in:
#   "High-Resolution Convolutional Neural Networks on Homomorphically Encrypted
#    Data via Sharding Ciphertexts" (Maloney et al., 2024)
#
# Supported architectures:
#   resnet9                       — 94.5% avg / 94.7% best (5 runs)
#   resnet20/32/44/56/110         — 90.6–92.8% avg (5 runs each, multiplexed)
#   resnet50                      — 98.3% (1 run, ImageNet pretrained)
#
# All architectures use Nesterov momentum, an EMA validation model, and
# kurtosis regularization targeting Gaussian pre-activation distributions.
#
# Usage (from repo root):
#   uv run python training/train_cifar10.py --arch resnet50
#   uv run python training/train_cifar10.py --arch resnet9 --nruns 5
#   uv run python training/train_cifar10.py --arch resnet32 --nruns 5

import argparse
import copy
import json
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
from utils.utils_resnetN import update_nesterov, update_ema, label_smoothing_loss, patch_whitening

# ---------------------------------------------------------------------------
# Hyperparameters per architecture (from Optuna tuning in the paper).
# ResNet9 and ResNet50 use the default bag-of-tricks params.
# ---------------------------------------------------------------------------
ARCH_PARAMS = {
    'resnet9':   {'lr': 2e-3,                   'lr_bias': 64.0, 'momentum': 0.9,                  'weight_decay': 0.256},
    'resnet20':  {'lr': 0.0016822249163093617,   'lr_bias': 63.934695046801245, 'momentum': 0.8484574950771097,  'weight_decay': 0.11450934135118791},
    'resnet32':  {'lr': 0.0013205254360784781,   'lr_bias': 61.138281101282544, 'momentum': 0.873508553678625,   'weight_decay': 0.26911634559915815},
    'resnet44':  {'lr': 0.0017177668853317557,   'lr_bias': 72.4258603207131,   'momentum': 0.8353896320183106,  'weight_decay': 0.16749858871622},
    'resnet56':  {'lr': 0.0012022823706985977,   'lr_bias': 71.31108702685964,  'momentum': 0.8252747623136261,  'weight_decay': 0.26463818739336625},
    'resnet110': {'lr': 0.001477698037686629,    'lr_bias': 61.444988882569774, 'momentum': 0.7241645867415002,  'weight_decay': 0.23586225065185779},
    'resnet50':  {'lr': 1e-3,                    'lr_bias': 64.0, 'momentum': 0.9,                  'weight_decay': 0.256},
}

RESNETN_ARCHS = {'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_resnet9(train_data: torch.Tensor, num_classes: int, device, dtype) -> nn.Module:
    """ResNet9 with patch-whitened first layer (frozen)."""
    from models.resnet9 import ResNet9

    # Compute patch-whitening weights from a 10k subset of unpadded training data.
    subset = train_data[:10000, :, 4:-4, 4:-4]  # un-pad the 40x40 → 32x32
    pw_weights = patch_whitening(subset)

    model = ResNet9(
        c_in=pw_weights.size(1),
        c_out=pw_weights.size(0),
        num_classes=num_classes,
        scale_out=0.125,
    ).to(device)
    model.set_conv1_weights(
        weights=pw_weights.to(device),
        bias=torch.zeros(pw_weights.size(0), device=device),
    )
    return model


def build_resnetn(arch: str, num_classes: int) -> nn.Module:
    """Multiplexed ResNet (20/32/44/56/110). Takes 4-channel input."""
    from models.resnetN_multiplexed import resnet20, resnet32, resnet44, resnet56, resnet110
    builders = {
        'resnet20': resnet20, 'resnet32': resnet32, 'resnet44': resnet44,
        'resnet56': resnet56, 'resnet110': resnet110,
    }
    return builders[arch](num_classes=num_classes)


def build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    """ResNet50 modified for 32×32 CIFAR-10 input with GELU activations.

    Changes vs. stock torchvision ResNet50:
    - conv1: 7×7 stride-2 → 3×3 stride-1
    - maxpool: replaced with Identity
    - All ReLU replaced with GELU
    - fc: 1000-class → num_classes

    Trained with 3-channel input. At HE inference time, pad_conv_input_channels()
    extends conv1 to accept the 4th zero-padded channel.
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


# ---------------------------------------------------------------------------
# LR and kurtosis schedules
# ---------------------------------------------------------------------------

def make_schedules(arch: str, N: int, lr: float, lr_bias: float):
    """Return (lr_schedule, lr_schedule_bias, kurt_schedule) tensors.

    ResNet9 uses a short fixed-batch schedule (faithful to the original).
    ResNetN/50 use an epoch-proportional schedule (~59 epochs total).
    """
    if arch == 'resnet9':
        # Hardcoded batch counts from original train_resnet9.py.
        # After 776 batches the LR stays at its final value.
        lr_sched = torch.cat([
            torch.linspace(0, lr, 194),
            torch.linspace(lr, lr / 10, 582),
        ])
        kurt_sched = torch.linspace(0, 0.1, 2000)
    else:
        lr_sched = torch.cat([
            torch.linspace(0.0, lr,   1 * N),   # warmup:   1 epoch
            torch.linspace(lr,  1e-4, 3 * N),   # decay:    3 epochs
            torch.linspace(1e-4, 1e-4, 30 * N), # constant: 30 epochs
            torch.linspace(1e-5, 1e-5, 15 * N), # constant: 15 epochs
            torch.linspace(1e-6, 1e-6, 10 * N), # constant: 10 epochs
        ])
        kurt_sched = torch.cat([
            torch.linspace(0,    0,    10 * N),  # 0    for first 10 epochs
            torch.linspace(0.05, 0.05,  2 * N),  # 0.05 for next   2 epochs
            torch.linspace(0.1,  0.1, 200 * N),  # 0.1  thereafter
        ])

    return lr_sched, lr_bias * lr_sched, kurt_sched


# ---------------------------------------------------------------------------
# Training loop (shared across all architectures)
# ---------------------------------------------------------------------------

def train_one_run(
    arch: str,
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    valid_data: torch.Tensor,
    valid_targets: torch.Tensor,
    num_classes: int,
    batch_size: int,
    lr: float,
    lr_bias: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    pretrained: bool,
    device,
    dtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    N = len(train_data) // batch_size
    lr_schedule, lr_schedule_bias, kurt_schedule = make_schedules(arch, N, lr, lr_bias)

    # Build model
    if arch == 'resnet9':
        model = build_resnet9(train_data, num_classes, device, dtype)
    elif arch in RESNETN_ARCHS:
        model = build_resnetn(arch, num_classes).to(device)
    else:
        model = build_resnet50(num_classes, pretrained).to(device)

    model.to(dtype)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    valid_model = copy.deepcopy(model)
    ema_update_freq = 5
    ema_rho = 0.99 ** ema_update_freq

    weights = [
        (w, torch.zeros_like(w))
        for w in model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    biases = [
        (w, torch.zeros_like(w))
        for w in model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    # For schedule-driven architectures, derive epoch count from schedule.
    if arch == 'resnet9':
        total_epochs = epochs
    else:
        total_epochs = len(lr_schedule) // N

    best_acc = 0.0
    best_model = None
    batch_count = 0
    train_time = 0.0

    print(f"\nepoch    batch    train time [sec]    validation accuracy")

    for epoch in range(1, total_epochs + 1):
        t0 = perf_counter()
        model.train(True)

        indices = torch.randperm(len(train_data), device=device)
        data = train_data[indices]
        targets = train_targets[indices]

        # Random 32×32 crop from 40×40 padded training images
        data = torch.cat([
            random_crop(data[i:i + batch_size], (32, 32))
            for i in range(0, len(data), batch_size)
        ])

        # Random horizontal flip on half the data
        data[:len(data) // 2] = torch.flip(data[:len(data) // 2], [-1])

        for i in range(0, len(data), batch_size):
            if i + batch_size > len(data):
                break

            inputs = data[i:i + batch_size]
            target = targets[i:i + batch_size]
            batch_count += 1

            model.zero_grad()

            remove_all_hooks(model)
            activations = get_intermediate_output(model)

            logits = model(inputs)
            loss = label_smoothing_loss(logits, target, alpha=0.2)

            kurt_idx = min(batch_count, len(kurt_schedule) - 1)
            kurt_scale = kurt_schedule[kurt_idx]

            remove_all_hooks(model)
            activations = {k: torch.cat(v) for k, v in activations.items()}
            loss_means, loss_stds, loss_kurts = get_statistics(activations)
            loss += (loss_means + loss_stds + loss_kurts) * kurt_scale

            loss.sum().backward()

            lr_idx = min(batch_count, len(lr_schedule) - 1)
            update_nesterov(weights, lr_schedule[lr_idx], weight_decay, momentum)
            update_nesterov(biases, lr_schedule_bias[lr_idx], 0.004, momentum)

            if (i // batch_size % ema_update_freq) == 0:
                update_ema(model, valid_model, ema_rho)

        train_time += perf_counter() - t0

        valid_correct = []
        for i in range(0, len(valid_data), batch_size):
            valid_model.train(False)
            logits = valid_model(valid_data[i:i + batch_size]).detach()
            correct = logits.max(dim=1)[1] == valid_targets[i:i + batch_size]
            valid_correct.append(correct.detach().to(torch.float64))

        valid_acc = torch.mean(torch.cat(valid_correct)).item()
        print(f"{epoch:5d} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(valid_model).float()

    return best_acc, best_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 models for HE-compatible inference"
    )
    parser.add_argument("--arch", required=True,
                        choices=list(ARCH_PARAMS.keys()),
                        help="Model architecture")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 512 for resnet9, 256 otherwise)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (default: architecture-specific)")
    parser.add_argument("--lr-bias", type=float, default=None,
                        help="Bias LR multiplier (default: architecture-specific)")
    parser.add_argument("--momentum", type=float, default=None,
                        help="Nesterov momentum (default: architecture-specific)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay (default: architecture-specific)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for resnet9; ignored for others "
                             "(schedule-driven). Default: 100")
    parser.add_argument("--nruns", type=int, default=None,
                        help="Number of independent training runs "
                             "(default: 1 for resnet50, 5 for all others)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="CIFAR-10 data directory (auto-downloaded if absent)")
    parser.add_argument("--output-dir", type=str, default="weights",
                        help="Directory to save model weights (default: weights/)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Train ResNet50 from scratch instead of ImageNet weights")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default="CIFAR10",
                        help="Dataset (default: CIFAR10)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.arch == 'resnet50' else torch.float32

    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(device.index)}")
    print(f"Dtype:  {dtype}")

    # Resolve hyperparams (CLI overrides architecture defaults)
    defaults = ARCH_PARAMS[args.arch]
    lr           = args.lr           if args.lr           is not None else defaults['lr']
    lr_bias      = args.lr_bias      if args.lr_bias      is not None else defaults['lr_bias']
    momentum     = args.momentum     if args.momentum     is not None else defaults['momentum']
    weight_decay = args.weight_decay if args.weight_decay is not None else defaults['weight_decay']
    batch_size   = args.batch_size   if args.batch_size   is not None else (512 if args.arch == 'resnet9' else 256)
    nruns        = args.nruns        if args.nruns        is not None else (1 if args.arch == 'resnet50' else 5)
    num_classes  = 10 if args.dataset == 'CIFAR10' else 100

    print(f"arch={args.arch}  lr={lr}  lr_bias={lr_bias}  momentum={momentum}  "
          f"weight_decay={weight_decay}  batch_size={batch_size}  nruns={nruns}")

    # Load data
    if args.dataset == "CIFAR10":
        train_data, train_targets, valid_data, valid_targets = load_cifar10(
            device, dtype, args.data_dir
        )
    else:
        from utils.utils_dataloading import load_cifar100
        train_data, train_targets, valid_data, valid_targets = load_cifar100(
            device, dtype, args.data_dir
        )

    # ResNet9 and multiplexed ResNets take 4-channel input; ResNet50 takes 3.
    if args.arch != 'resnet50':
        zeros = torch.zeros(train_data.size(0), 1, train_data.size(2), train_data.size(3),
                            device=device, dtype=dtype)
        train_data = torch.cat([train_data, zeros], dim=1)
        zeros = torch.zeros(valid_data.size(0), 1, valid_data.size(2), valid_data.size(3),
                            device=device, dtype=dtype)
        valid_data = torch.cat([valid_data, zeros], dim=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_params = dict(
        arch=args.arch,
        train_data=train_data, train_targets=train_targets,
        valid_data=valid_data, valid_targets=valid_targets,
        num_classes=num_classes,
        batch_size=batch_size,
        lr=lr, lr_bias=lr_bias, momentum=momentum, weight_decay=weight_decay,
        epochs=args.epochs,
        pretrained=args.pretrained,
        device=device, dtype=dtype,
    )

    accuracies = []
    log = {"arch": args.arch, "dataset": args.dataset,
           "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
           "batch_size": batch_size, "nruns": nruns}

    for run in range(nruns):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{nruns}  (seed={run})")
        print(f"{'='*60}")

        best_acc, best_model = train_one_run(**run_params, seed=run)
        accuracies.append(best_acc)
        log[f"run{run}"] = best_acc
        print(f"Run {run + 1} best val_acc: {100 * best_acc:.2f}%")

        weight_file = output_dir / f"{args.arch}_{args.dataset}_run{run}.pt"
        torch.save(best_model, str(weight_file))
        print(f"Saved: {weight_file}")

    mean = sum(accuracies) / len(accuracies)
    std = (sum((a - mean) ** 2 for a in accuracies) / len(accuracies)) ** 0.5
    print(f"\nFinal: {100 * mean:.2f}% ± {100 * std:.2f}% over {nruns} run(s)")
    log["mean_accuracy"] = mean
    log["std_accuracy"] = std

    log_file = log_dir / f"{args.arch}_{args.dataset}.json"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=4)
    print(f"Log:   {log_file}")

    # For single-run architectures (resnet50), also save without run suffix.
    if nruns == 1:
        canonical = output_dir / f"{args.arch}_{args.dataset}.pt"
        torch.save(best_model, str(canonical))
        print(f"Also saved as: {canonical}")


if __name__ == "__main__":
    main()
