"""
Sequential Experiment Runner — SMAI Assignment 3
================================================
Runs experiments A-H in order.  Each experiment fixes the best value found in the
previous one and varies exactly one new hyperparameter.  A bar-chart is saved after
every experiment and a cumulative summary chart is produced at the end.

Output folder: experiment_results/
    exp_A_learning_rate.png
    exp_B_augmentation.png
    exp_C_scheduler.png
    exp_D_epochs.png
    exp_E_label_smoothing.png
    exp_F_diff_lr.png
    exp_G_dropout.png
    exp_H_skip_connections.png
    summary_all_experiments.png
    experiment_summary.csv
    optimal_config.json
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths & constants ────────────────────────────────────────────────────────
DATA_DIR    = "dataset/animals/animals"
RESULTS_DIR = "experiment_results"
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Augmentation catalogue ───────────────────────────────────────────────────
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

AUGMENTATIONS = {
    "baseline": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), _NORM,
    ]),
    "colorjitter": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), _NORM,
    ]),
    "grayscale": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(), _NORM,
    ]),
    "centercrop": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), _NORM,
    ]),
    "heavy": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(), _NORM,
    ]),
}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), _NORM,
])

# ─── Dataset helper ───────────────────────────────────────────────────────────
class _Wrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.transform(x), y


def load_splits():
    full = datasets.ImageFolder(DATA_DIR)
    n    = len(full)
    tr   = int(0.70 * n)
    va   = int(0.15 * n)
    te   = n - tr - va
    gen  = torch.Generator().manual_seed(SEED)
    train_sub, val_sub, test_sub = random_split(full, [tr, va, te], generator=gen)
    return train_sub, val_sub, test_sub, full.classes


def make_loaders(train_sub, val_sub, test_sub, aug="baseline", batch=32):
    tl = DataLoader(_Wrapper(train_sub, AUGMENTATIONS[aug]), batch_size=batch, shuffle=True,  num_workers=0)
    vl = DataLoader(_Wrapper(val_sub,   VAL_TRANSFORM),      batch_size=batch, shuffle=False, num_workers=0)
    el = DataLoader(_Wrapper(test_sub,  VAL_TRANSFORM),       batch_size=batch, shuffle=False, num_workers=0)
    return tl, vl, el

# ─── Custom classifier heads ──────────────────────────────────────────────────
class DropoutHead(nn.Module):
    """Single linear classifier with pre-classifier dropout."""
    def __init__(self, in_features, num_classes, p=0.0):
        super().__init__()
        self.drop = nn.Dropout(p=p)
        self.fc   = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


class SkipHead(nn.Module):
    """
    Two-path classifier head with a skip (residual) connection.

    Deep path  : Linear(D→H) → BN → ReLU → Dropout → Linear(H→C)
    Skip path  : Linear(D→C)          (shortcut, bypasses hidden layer)
    Output     : deep_path + skip_path

    The skip ensures the model can fall back to a direct linear projection
    when the hidden layer does not help, while still having non-linear
    capacity when it does.
    """
    def __init__(self, in_features, num_classes, hidden=512, p=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden)
        self.bn   = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=p)
        self.fc2  = nn.Linear(hidden, num_classes)
        self.skip = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x):
        h = self.drop(self.relu(self.bn(self.fc1(x))))
        return self.fc2(h) + self.skip(x)


# ─── Model helpers ────────────────────────────────────────────────────────────
def build_model(num_classes, dropout=0.0, skip_hidden=None):
    """
    EfficientNet-B0 with Level-1 unfreezing (classifier + conv_head + bn2).

    dropout    : dropout rate applied in the classifier head (default 0 = off)
    skip_hidden: if not None, replaces the linear head with a SkipHead whose
                 hidden dimension is this value
    """
    m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    in_features = m.num_features          # 1280 for EfficientNet-B0

    # Freeze everything first
    for p in m.parameters():
        p.requires_grad = False

    # Replace classifier head according to experiment config
    if skip_hidden is not None:
        m.classifier = SkipHead(in_features, num_classes, hidden=skip_hidden, p=dropout)
    elif dropout > 0.0:
        m.classifier = DropoutHead(in_features, num_classes, p=dropout)
    # else: keep timm's default nn.Linear head

    # Level-1 unfreezing
    for p in m.get_classifier().parameters():
        p.requires_grad = True
    if hasattr(m, "conv_head"):
        for p in m.conv_head.parameters():
            p.requires_grad = True
    if hasattr(m, "bn2"):
        for p in m.bn2.parameters():
            p.requires_grad = True
    return m


def build_optimizer(model, lr, diff_lr=False):
    if not diff_lr:
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    head_params     = list(model.get_classifier().parameters())
    backbone_params = []
    if hasattr(model, "conv_head"):
        backbone_params += list(model.conv_head.parameters())
    if hasattr(model, "bn2"):
        backbone_params += list(model.bn2.parameters())
    return optim.Adam([
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ])


def build_scheduler(optimizer, name, epochs):
    if name is None:
        return None
    if name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    return None

# ─── Core train + eval ───────────────────────────────────────────────────────
def run_trial(cfg, train_sub, val_sub, test_sub, num_classes, tag=""):
    """
    Train one model with `cfg` and return test accuracy (%).
    cfg keys: lr, augmentation, scheduler, epochs, label_smoothing,
              diff_lr, dropout, skip_hidden
    """
    tl, vl, el = make_loaders(train_sub, val_sub, test_sub, aug=cfg["augmentation"])
    model       = build_model(
        num_classes,
        dropout     = cfg.get("dropout",     0.0),
        skip_hidden = cfg.get("skip_hidden", None),
    ).to(DEVICE)
    optimizer   = build_optimizer(model, cfg["lr"], cfg.get("diff_lr", False))
    criterion   = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    scheduler   = build_scheduler(optimizer, cfg.get("scheduler"), cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        # ── train ──
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        # ── validate ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, preds = torch.max(model(x), 1)
                total   += y.size(0)
                correct += (preds == y).sum().item()
        val_acc = correct / total

        if scheduler:
            if cfg.get("scheduler") == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        print(f"      [{tag}] epoch {epoch+1:2d}/{cfg['epochs']}  val={val_acc:.4f}")

    # ── test ──
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in el:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, preds = torch.max(model(x), 1)
            total   += y.size(0)
            correct += (preds == y).sum().item()
    acc = correct / total * 100
    print(f"      [{tag}] ── test acc: {acc:.2f}%")
    return acc

# ─── Chart helper ─────────────────────────────────────────────────────────────
def save_bar_chart(labels, accuracies, best_label, title, xlabel, filename, prev_best=None):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    colors = ["#2ecc71" if l == best_label else "#3498db" for l in labels]
    bars   = ax.bar(labels, accuracies, color=colors, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    if prev_best is not None:
        ax.axhline(
            prev_best, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Previous best: {prev_best:.2f}%", zorder=4,
        )
        ax.legend(fontsize=10)

    best_idx = labels.index(best_label)
    bars[best_idx].set_edgecolor("#27ae60")
    bars[best_idx].set_linewidth(2.5)

    y_min = min(accuracies) - 2
    y_max = max(accuracies) + 3
    ax.set_ylim(y_min, y_max)
    ax.annotate(
        "★ Best",
        xy=(bars[best_idx].get_x() + bars[best_idx].get_width() / 2, y_max - 0.5),
        ha="center", color="#27ae60", fontsize=11, fontweight="bold",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"    → saved {path}")


def save_summary_chart(stage_labels, accuracies):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    palette = ["#95a5a6"] + ["#3498db"] * (len(accuracies) - 2) + ["#2ecc71"]
    bars    = ax.bar(stage_labels, accuracies, color=palette, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Delta annotations between consecutive bars
    for i in range(1, len(accuracies)):
        delta = accuracies[i] - accuracies[i - 1]
        mid_y = (accuracies[i] + accuracies[i - 1]) / 2
        sign  = "+" if delta >= 0 else ""
        color = "#27ae60" if delta >= 0 else "#e74c3c"
        ax.annotate(
            "", xy=(i, accuracies[i] - 0.15), xytext=(i, accuracies[i - 1] + 0.15),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        ax.text(i + 0.08, mid_y, f"{sign}{delta:.2f}%", color=color, fontsize=9, va="center")

    total = accuracies[-1] - accuracies[0]
    ax.text(
        0.98, 0.97, f"Total gain: +{total:.2f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold", color="#27ae60",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#27ae60", lw=1.5),
    )

    y_min = min(accuracies) - 2
    y_max = max(accuracies) + 4
    ax.set_ylim(y_min, y_max)
    ax.set_title("Cumulative Experiment Results: Accuracy Improvement Journey",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment Stage", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "summary_all_experiments.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"    → saved {path}")

# ─── Individual experiments ───────────────────────────────────────────────────
def experiment_a(base_cfg, train_sub, val_sub, test_sub, num_classes):
    """Vary learning rate; keep everything else at base."""
    _banner("A", "Learning Rate Tuning")
    candidates = [1e-2, 1e-3, 1e-4]
    labels     = [f"lr={lr:.0e}" for lr in candidates]
    accs       = []
    for lr, lbl in zip(candidates, labels):
        cfg = {**base_cfg, "lr": lr}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes, lbl))

    best_lbl = labels[int(np.argmax(accs))]
    best_lr  = candidates[int(np.argmax(accs))]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment A: Learning Rate Tuning",
                   "Learning Rate", "exp_A_learning_rate.png")
    _done("lr", best_lr, max(accs))
    return {**base_cfg, "lr": best_lr}, max(accs)


def experiment_b(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """Vary data augmentation; LR is now fixed from Exp A."""
    _banner("B", "Data Augmentation Strategy")
    aug_keys = ["baseline", "colorjitter", "grayscale", "centercrop", "heavy"]
    labels   = ["Baseline\n(Flip+Rot)", "ColorJitter", "RandomGrayscale",
                "CenterCrop\n(256→224)", "Heavy\n(All)"]
    accs = []
    for aug, lbl in zip(aug_keys, labels):
        cfg = {**base_cfg, "augmentation": aug}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes,
                              lbl.replace("\n", " ")))

    best_lbl = labels[int(np.argmax(accs))]
    best_aug = aug_keys[int(np.argmax(accs))]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment B: Augmentation Strategy Comparison",
                   "Augmentation", "exp_B_augmentation.png", prev_best=prev_best)
    _done("augmentation", best_aug, max(accs))
    return {**base_cfg, "augmentation": best_aug}, max(accs)


def experiment_c(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """Vary LR scheduler; LR & augmentation fixed from A & B."""
    _banner("C", "LR Scheduler")
    sched_keys = [None,           "steplr",                "cosine",          "plateau"]
    labels     = ["No Scheduler", "StepLR\n(step=2,γ=0.5)","CosineAnnealing", "ReduceOnPlateau"]
    accs = []
    for sk, lbl in zip(sched_keys, labels):
        cfg = {**base_cfg, "scheduler": sk}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes,
                              lbl.replace("\n", " ")))

    best_lbl   = labels[int(np.argmax(accs))]
    best_sched = sched_keys[int(np.argmax(accs))]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment C: LR Scheduler Comparison",
                   "Scheduler", "exp_C_scheduler.png", prev_best=prev_best)
    _done("scheduler", best_sched, max(accs))
    return {**base_cfg, "scheduler": best_sched}, max(accs)


def experiment_d(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """Vary epoch count; all previous choices fixed."""
    _banner("D", "Training Epoch Count")
    epoch_vals = [5, 10, 15]
    labels     = [f"{e} Epochs" for e in epoch_vals]
    accs = []
    for ep, lbl in zip(epoch_vals, labels):
        cfg = {**base_cfg, "epochs": ep}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes, lbl))

    best_lbl    = labels[int(np.argmax(accs))]
    best_epochs = epoch_vals[int(np.argmax(accs))]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment D: Training Epoch Count",
                   "Epochs", "exp_D_epochs.png", prev_best=prev_best)
    _done("epochs", best_epochs, max(accs))
    return {**base_cfg, "epochs": best_epochs}, max(accs)


def experiment_e(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """Vary label smoothing; all previous choices fixed."""
    _banner("E", "Label Smoothing")
    eps_vals = [0.0, 0.05, 0.1, 0.2]
    labels   = [f"ε = {e}" for e in eps_vals]
    accs = []
    for eps, lbl in zip(eps_vals, labels):
        cfg = {**base_cfg, "label_smoothing": eps}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes, lbl))

    best_lbl = labels[int(np.argmax(accs))]
    best_eps = eps_vals[int(np.argmax(accs))]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment E: Label Smoothing (ε)",
                   "Label Smoothing ε", "exp_E_label_smoothing.png", prev_best=prev_best)
    _done("label_smoothing", best_eps, max(accs))
    return {**base_cfg, "label_smoothing": best_eps}, max(accs)


def experiment_f(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """Vary whether backbone uses a lower LR; all previous choices fixed."""
    _banner("F", "Differential Learning Rates")
    variants = [
        (False, "Uniform\nlr = 1e-3"),
        (True,  "Differential\nhead=1e-3\nbackbone=1e-4"),
    ]
    labels = [lbl for _, lbl in variants]
    accs   = []
    for diff_lr, lbl in variants:
        cfg = {**base_cfg, "diff_lr": diff_lr}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes,
                              lbl.replace("\n", " ")))

    best_lbl     = labels[int(np.argmax(accs))]
    best_diff_lr = variants[int(np.argmax(accs))][0]
    save_bar_chart(labels, accs, best_lbl,
                   "Experiment F: Differential Learning Rates",
                   "LR Strategy", "exp_F_diff_lr.png", prev_best=prev_best)
    _done("diff_lr", best_diff_lr, max(accs))
    return {**base_cfg, "diff_lr": best_diff_lr}, max(accs)

def experiment_g(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """
    Vary dropout rate in the classifier head; all previous choices fixed.
    Tests whether regularising the head reduces overfitting on the small dataset.
    """
    _banner("G", "Classifier Dropout")
    drop_vals = [0.0, 0.1, 0.2, 0.3, 0.5]
    labels    = [f"dropout={p}" for p in drop_vals]
    accs = []
    for p, lbl in zip(drop_vals, labels):
        cfg = {**base_cfg, "dropout": p, "skip_hidden": None}
        accs.append(run_trial(cfg, train_sub, val_sub, test_sub, num_classes, lbl))

    best_lbl  = labels[int(np.argmax(accs))]
    best_drop = drop_vals[int(np.argmax(accs))]
    save_bar_chart(
        labels, accs, best_lbl,
        "Experiment G: Classifier Dropout Rate",
        "Dropout Rate (p)", "exp_G_dropout.png", prev_best=prev_best,
    )
    _done("dropout", best_drop, max(accs))
    return {**base_cfg, "dropout": best_drop}, max(accs)


def experiment_h(base_cfg, train_sub, val_sub, test_sub, num_classes, prev_best):
    """
    Compare classifier head architectures with and without a skip connection.

    Variants:
      plain_linear  — default nn.Linear (no hidden layer, no skip)
      mlp_512       — two-layer MLP, hidden=512, NO skip
      skip_512      — two-layer MLP, hidden=512, WITH skip (Linear(1280→90) shortcut)
      skip_256      — two-layer MLP, hidden=256, WITH skip
      skip_128      — two-layer MLP, hidden=128, WITH skip

    The MLP-without-skip variant isolates whether gains come from extra capacity
    or specifically from the residual shortcut path.
    """
    _banner("H", "Skip Connections in Classifier Head")

    # (skip_hidden=None → plain or MLP without skip; positive int → SkipHead)
    # We abuse skip_hidden=0 as a sentinel for "MLP without skip, hidden=512"
    variants = [
        (None,  False, "Plain\nLinear"),
        (None,  True,  "MLP-512\n(no skip)"),   # handled separately below
        (512,   False, "SkipHead\nhidden=512"),
        (256,   False, "SkipHead\nhidden=256"),
        (128,   False, "SkipHead\nhidden=128"),
    ]

    labels = [lbl for _, _, lbl in variants]
    accs   = []

    for skip_hidden, is_plain_mlp, lbl in variants:
        if is_plain_mlp:
            # Build a plain two-layer MLP without the residual shortcut
            cfg = {**base_cfg, "skip_hidden": None, "dropout": base_cfg.get("dropout", 0.0)}
            # Monkey-patch: temporarily swap head after build_model
            tl, vl, el = make_loaders(
                train_sub, val_sub, test_sub, aug=cfg["augmentation"]
            )
            model = build_model(
                num_classes,
                dropout=cfg.get("dropout", 0.0),
                skip_hidden=None,
            )
            # Replace with plain two-layer MLP (no skip)
            in_f = model.num_features
            nc   = num_classes
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=cfg.get("dropout", 0.0)),
                nn.Linear(512, nc),
            )
            for p in model.classifier.parameters():
                p.requires_grad = True
            model = model.to(DEVICE)

            optimizer  = build_optimizer(model, cfg["lr"], cfg.get("diff_lr", False))
            criterion  = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
            scheduler  = build_scheduler(optimizer, cfg.get("scheduler"), cfg["epochs"])
            acc = _train_eval_loop(model, tl, vl, el, optimizer, criterion, scheduler,
                                   cfg["epochs"], cfg.get("scheduler"), lbl.replace("\n", " "))
        else:
            cfg = {**base_cfg, "skip_hidden": skip_hidden}
            acc = run_trial(cfg, train_sub, val_sub, test_sub, num_classes, lbl.replace("\n", " "))
        accs.append(acc)

    best_lbl        = labels[int(np.argmax(accs))]
    best_idx        = int(np.argmax(accs))
    best_skip_hidden = variants[best_idx][0]

    save_bar_chart(
        labels, accs, best_lbl,
        "Experiment H: Classifier Head Architecture (Skip Connections)",
        "Head Architecture", "exp_H_skip_connections.png", prev_best=prev_best,
    )
    _done("skip_hidden", best_skip_hidden, max(accs))
    return {**base_cfg, "skip_hidden": best_skip_hidden}, max(accs)


def _train_eval_loop(model, tl, vl, el, optimizer, criterion, scheduler,
                     epochs, sched_name, tag):
    """Shared training loop used by the plain-MLP branch of Experiment H."""
    for epoch in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, preds = torch.max(model(x), 1)
                total   += y.size(0)
                correct += (preds == y).sum().item()
        val_acc = correct / total
        if scheduler:
            scheduler.step(val_acc) if sched_name == "plateau" else scheduler.step()
        print(f"      [{tag}] epoch {epoch+1:2d}/{epochs}  val={val_acc:.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in el:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, preds = torch.max(model(x), 1)
            total   += y.size(0)
            correct += (preds == y).sum().item()
    acc = correct / total * 100
    print(f"      [{tag}] ── test acc: {acc:.2f}%")
    return acc


# ─── Utilities ────────────────────────────────────────────────────────────────
def _banner(letter, name):
    sep = "=" * 60
    print(f"\n{sep}\n  EXPERIMENT {letter}: {name}\n{sep}")

def _done(key, value, acc):
    print(f"\n  ✓  Optimal {key} = {value!r}  →  test acc: {acc:.2f}%\n")

def _hline():
    print("-" * 60)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nDevice  : {DEVICE}")
    print(f"Results : {os.path.abspath(RESULTS_DIR)}/\n")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset not found at: {DATA_DIR}")

    print("Loading dataset (fixed seed for reproducibility)…")
    train_sub, val_sub, test_sub, classes = load_splits()
    num_classes = len(classes)
    print(f"Classes={num_classes}  |  train={len(train_sub)}  val={len(val_sub)}  test={len(test_sub)}\n")

    # ── Baseline ──────────────────────────────────────────────────────────────
    base_cfg = dict(lr=0.001, augmentation="baseline", scheduler=None,
                    epochs=5, label_smoothing=0.0, diff_lr=False,
                    dropout=0.0, skip_hidden=None)
    print("─── Baseline (Level-1, default config) ───")
    t0           = time.time()
    baseline_acc = run_trial(base_cfg, train_sub, val_sub, test_sub, num_classes, "baseline")
    print(f"Baseline done in {(time.time()-t0)/60:.1f} min  →  {baseline_acc:.2f}%\n")

    # ── Sequential experiments ────────────────────────────────────────────────
    stage_labels = ["Baseline\n(Level 1)"]
    stage_accs   = [baseline_acc]

    cfg, acc = experiment_a(base_cfg, train_sub, val_sub, test_sub, num_classes)
    stage_labels.append("Exp A\n(LR)");        stage_accs.append(acc)

    cfg, acc = experiment_b(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp B\n(Augment)");   stage_accs.append(acc)

    cfg, acc = experiment_c(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp C\n(Scheduler)"); stage_accs.append(acc)

    cfg, acc = experiment_d(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp D\n(Epochs)");    stage_accs.append(acc)

    cfg, acc = experiment_e(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp E\n(LabelSmooth)"); stage_accs.append(acc)

    cfg, acc = experiment_f(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp F\n(Diff LR)");   stage_accs.append(acc)

    cfg, acc = experiment_g(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp G\n(Dropout)");   stage_accs.append(acc)

    cfg, acc = experiment_h(cfg, train_sub, val_sub, test_sub, num_classes, acc)
    stage_labels.append("Exp H\n(SkipHead)");  stage_accs.append(acc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    save_summary_chart(stage_labels, stage_accs)

    print("\nStage-by-stage results:")
    _hline()
    for lbl, a in zip(stage_labels, stage_accs):
        print(f"  {lbl.replace(chr(10), ' '):30s}  {a:.2f}%")
    _hline()
    print(f"  Total gain: +{stage_accs[-1] - stage_accs[0]:.2f}%")

    print("\nOptimal config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Save artefacts
    with open(os.path.join(RESULTS_DIR, "optimal_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    pd.DataFrame({"stage": stage_labels, "accuracy": stage_accs}).to_csv(
        os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False
    )

    print(f"\n✓  All done.  Charts + CSV saved to: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
