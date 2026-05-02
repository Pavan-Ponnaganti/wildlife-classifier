"""
CLIP Pipeline — T7.6 Indian Wildlife & Bird Identifier
=======================================================
Five progressively stronger approaches, each building on the previous:

  Step 1  Zero-shot          openai/clip-vit-base-patch32 + single prompt
  Step 2  Prompt ensemble    average embeddings over 7 text templates
  Step 3  Linear probe       CLIP image features → LogisticRegression (no GPU)
  Step 4  Ensemble           CLIP linear probe + EfficientNet (tuned weight)
  Step 5  Ensemble + TTA     8 augmented views averaged at inference time

Output folder: clip_results/
    clip_comparison.png        bar chart of all five methods
    clip_results_summary.csv   stage × accuracy table
    cache/                     cached CLIP feature arrays (avoids re-extraction)

New dependency (add to requirements.txt):
    transformers>=4.35
"""

import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ─── paths & constants ────────────────────────────────────────────────────────
DATA_DIR     = "dataset/animals/animals"
MODEL_PATH   = "efficientnet_b0_animals.pth"
CLASSES_JSON = "classes.json"
RESULTS_DIR  = "clip_results"
CACHE_DIR    = os.path.join(RESULTS_DIR, "cache")
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Seven prompt templates used in Step 2
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a wildlife photo of a {}",
    "a {} in its natural habitat",
    "a close-up photo of a {}",
    "an image of a {}",
    "a {}, a type of animal",
    "a {} in the wild",
]

# ─── dataset helpers ──────────────────────────────────────────────────────────
_EFFNET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

EFFNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _EFFNET_NORM,
])

# PIL-level augmentations used for TTA (compatible with CLIP processor and EfficientNet)
PIL_TTA_AUGS = [
    lambda img: img,
    lambda img: TF.hflip(img),
    lambda img: TF.rotate(img, 10),
    lambda img: TF.rotate(img, -10),
    lambda img: TF.adjust_brightness(img, 1.2),
    lambda img: TF.adjust_contrast(img, 1.1),
    lambda img: TF.hflip(TF.rotate(img, 10)),
    lambda img: TF.hflip(TF.adjust_brightness(img, 0.9)),
]


class _Wrapper(torch.utils.data.Dataset):
    """Applies a torchvision transform on top of a raw Subset."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]   # PIL image, int
        return self.transform(img), label


def load_splits():
    """70/15/15 split with seed=42, matching the fixed seed now in train.py."""
    full = datasets.ImageFolder(DATA_DIR)
    n  = len(full)
    tr = int(0.70 * n)
    va = int(0.15 * n)
    te = n - tr - va
    gen = torch.Generator().manual_seed(SEED)
    train_sub, val_sub, test_sub = random_split(full, [tr, va, te], generator=gen)
    return train_sub, val_sub, test_sub, full.classes


def load_class_names():
    with open(CLASSES_JSON) as f:
        return json.load(f)


# ─── CLIP model ───────────────────────────────────────────────────────────────
def load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError("Run: pip install transformers")

    print("Loading CLIP (openai/clip-vit-base-patch32)…")
    t0 = time.time()
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print(f"  done in {time.time()-t0:.1f}s")
    return model, processor


# ─── text embeddings ──────────────────────────────────────────────────────────
@torch.no_grad()
def get_text_embeddings(clip_model, clip_processor, class_names, templates=None):
    """
    Returns L2-normalised text embeddings, shape (C, 512).
    With multiple templates the per-template embeddings are averaged
    and re-normalised — the standard CLIP prompt-ensembling trick.
    """
    if templates is None:
        templates = ["a photo of a {}"]

    per_template = []
    for tmpl in templates:
        texts  = [tmpl.format(c.replace("_", " ")) for c in class_names]
        inputs     = clip_processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True).to(DEVICE)
        text_out   = clip_model.text_model(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"])
        embeds     = clip_model.text_projection(text_out.pooler_output)  # (C, 512)
        per_template.append(F.normalize(embeds, dim=-1))

    avg = torch.stack(per_template).mean(0)                    # (C, 512)
    return F.normalize(avg, dim=-1)


# ─── image feature extraction ─────────────────────────────────────────────────
@torch.no_grad()
def extract_clip_features(clip_model, clip_processor, subset, cache_name, batch=64):
    """
    Extracts CLIP image embeddings for every sample in `subset`.
    Results are cached to disk so repeated runs are instant.
    Returns (features: np.float32 (N,512), labels: np.int64 (N,)).
    """
    path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(path):
        print(f"  Loading cached features '{cache_name}'…")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Extracting CLIP features for {len(subset)} images ({cache_name})…")
    feats, labels = [], []
    buf_imgs, buf_labs = [], []

    for i in range(len(subset)):
        img, lbl = subset[i]
        buf_imgs.append(img)
        buf_labs.append(lbl)

        if len(buf_imgs) == batch or i == len(subset) - 1:
            inp          = clip_processor(images=buf_imgs, return_tensors="pt",
                                          padding=True).to(DEVICE)
            vision_out   = clip_model.vision_model(pixel_values=inp["pixel_values"])
            projected    = clip_model.visual_projection(vision_out.pooler_output)
            f            = F.normalize(projected, dim=-1)
            feats.append(f.cpu().numpy())
            labels.extend(buf_labs)
            buf_imgs, buf_labs = [], []

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(subset)}")

    feats  = np.vstack(feats).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)

    with open(path, "wb") as f:
        pickle.dump((feats, labels), f)
    print(f"  Cached → {path}")
    return feats, labels


# ─── zero-shot evaluation ─────────────────────────────────────────────────────
def zero_shot_eval(test_feats, test_labels, text_embeds_np):
    """
    Dot-product similarity (cosine, since both sides are L2-normalised).
    Returns (accuracy %, softmax_probs (N, C)).
    """
    logits = test_feats @ text_embeds_np.T * 100.0        # (N, C)  scale = CLIP temperature
    probs  = _softmax(logits)
    preds  = logits.argmax(axis=1)
    acc    = (preds == test_labels).mean() * 100
    return acc, probs


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─── linear probe ─────────────────────────────────────────────────────────────
def train_linear_probe(train_feats, train_labels, C=4.0):
    """LogisticRegression on CLIP features. Trains in seconds, no GPU needed."""
    print(f"  Training linear probe (C={C})…")
    t0 = time.time()
    probe = LogisticRegression(
        C=C, max_iter=1000, solver="lbfgs",
        random_state=SEED, n_jobs=-1,
    )
    probe.fit(train_feats, train_labels)
    print(f"  Probe trained in {time.time()-t0:.1f}s")
    return probe


# ─── EfficientNet helpers ─────────────────────────────────────────────────────
def load_efficientnet(num_classes):
    if not os.path.exists(MODEL_PATH):
        print(f"  WARNING: {MODEL_PATH} not found — skipping EfficientNet steps.")
        return None
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"  EfficientNet loaded from {MODEL_PATH}")
    return model


@torch.no_grad()
def get_effnet_probs(effnet_model, subset, pil_aug=None, batch=32):
    """
    Returns softmax probabilities (N, C) from EfficientNet.
    `pil_aug`: optional callable PIL→PIL applied before the standard transform.
    """
    class _AugWrapper(torch.utils.data.Dataset):
        def __init__(self, sub, aug, tf):
            self.sub = sub
            self.aug = aug
            self.tf  = tf
        def __len__(self):
            return len(self.sub)
        def __getitem__(self, i):
            img, lbl = self.sub[i]
            if self.aug:
                img = self.aug(img)
            return self.tf(img), lbl

    loader = DataLoader(
        _AugWrapper(subset, pil_aug, EFFNET_TRANSFORM),
        batch_size=batch, shuffle=False, num_workers=0,
    )
    probs_all, labs_all = [], []
    for x, y in loader:
        logits = effnet_model(x.to(DEVICE))
        probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labs_all.extend(y.numpy())
    return np.vstack(probs_all), np.array(labs_all)


# ─── ensemble weight tuning ───────────────────────────────────────────────────
def tune_weight(probs_a, probs_b, labels, steps=17):
    """Sweep blending weight on val set; return best (weight, val_acc)."""
    best_w, best_acc = 0.5, 0.0
    for w in np.linspace(0.05, 0.95, steps):
        blended = w * probs_a + (1 - w) * probs_b
        acc = (blended.argmax(1) == labels).mean() * 100
        if acc > best_acc:
            best_acc, best_w = acc, w
    return round(float(best_w), 2), best_acc


def blend(probs_a, probs_b, w):
    return w * probs_a + (1 - w) * probs_b


def acc(probs, labels):
    return (probs.argmax(1) == labels).mean() * 100


# ─── chart ────────────────────────────────────────────────────────────────────
def save_chart(results: dict, effnet_baseline: float):
    sns.set_theme(style="whitegrid")
    labels = list(results.keys())
    accs   = [results[k] for k in labels]
    best   = max(accs)
    colors = ["#2ecc71" if a == best else "#3498db" for a in accs]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 2.2), 6))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{a:.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(effnet_baseline, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"EfficientNet fine-tuned baseline: {effnet_baseline:.2f}%", zorder=4)
    ax.legend(fontsize=10)

    best_idx = accs.index(best)
    bars[best_idx].set_edgecolor("#27ae60")
    bars[best_idx].set_linewidth(2.5)
    ax.annotate("★ Best",
                xy=(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
                    max(accs) + 2.5),
                ha="center", color="#27ae60", fontsize=11, fontweight="bold")

    y_min = min(accs + [effnet_baseline]) - 4
    y_max = max(accs) + 5
    ax.set_ylim(y_min, y_max)
    ax.set_title("CLIP vs EfficientNet: All Methods Comparison",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "clip_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  → Chart saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\nDevice : {DEVICE}")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}/\n")

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"Classes: {num_classes}\n")

    print("Loading dataset splits (same seed as train.py)…")
    train_sub, val_sub, test_sub, _ = load_splits()
    print(f"  train={len(train_sub)}  val={len(val_sub)}  test={len(test_sub)}\n")

    clip_model, clip_processor = load_clip()

    # ── Extract CLIP features once, cache to disk ──────────────────────────────
    print("\n── Feature Extraction ──────────────────────────────────────────────")
    train_feats, train_labels = extract_clip_features(clip_model, clip_processor, train_sub, "train")
    val_feats,   val_labels   = extract_clip_features(clip_model, clip_processor, val_sub,   "val")
    test_feats,  test_labels  = extract_clip_features(clip_model, clip_processor, test_sub,  "test")

    results = {}   # will accumulate {label: test_acc}

    # ── Step 1: Zero-shot (single prompt) ─────────────────────────────────────
    print("\n── Step 1: CLIP Zero-Shot (single prompt) ──────────────────────────")
    text_single = get_text_embeddings(clip_model, clip_processor, class_names,
                                      templates=["a photo of a {}"])
    text_np = text_single.cpu().numpy()
    zs_acc, zs_probs = zero_shot_eval(test_feats, test_labels, text_np)
    print(f"  Test accuracy: {zs_acc:.2f}%")
    results["CLIP\nZero-Shot\n(1 prompt)"] = zs_acc

    # val probs for later tuning
    _, zs_val_probs = zero_shot_eval(val_feats, val_labels, text_np)

    # ── Step 2: Prompt ensemble ────────────────────────────────────────────────
    print("\n── Step 2: CLIP Zero-Shot (prompt ensemble, 7 templates) ───────────")
    text_ens = get_text_embeddings(clip_model, clip_processor, class_names,
                                   templates=PROMPT_TEMPLATES)
    text_ens_np = text_ens.cpu().numpy()
    pe_acc, pe_probs = zero_shot_eval(test_feats, test_labels, text_ens_np)
    print(f"  Test accuracy: {pe_acc:.2f}%  (Δ = {pe_acc-zs_acc:+.2f}%)")
    results["CLIP\nPrompt\nEnsemble"] = pe_acc

    _, pe_val_probs = zero_shot_eval(val_feats, val_labels, text_ens_np)

    # ── Step 3: Linear probe ───────────────────────────────────────────────────
    print("\n── Step 3: CLIP Linear Probe ────────────────────────────────────────")
    # Tune regularisation C on val set
    best_C, best_val_acc = 1.0, 0.0
    for C in [0.1, 0.5, 1.0, 4.0, 10.0]:
        probe_c = train_linear_probe(train_feats, train_labels, C=C)
        va = probe_c.score(val_feats, val_labels) * 100
        print(f"  C={C:5.1f}  val={va:.2f}%")
        if va > best_val_acc:
            best_val_acc, best_C = va, C

    print(f"\n  Best C={best_C}  (val={best_val_acc:.2f}%)")
    probe = train_linear_probe(train_feats, train_labels, C=best_C)
    lp_probs     = probe.predict_proba(test_feats)
    lp_val_probs = probe.predict_proba(val_feats)
    lp_acc = acc(lp_probs, test_labels)
    print(f"  Test accuracy: {lp_acc:.2f}%  (Δ vs zero-shot = {lp_acc-zs_acc:+.2f}%)")
    results["CLIP\nLinear\nProbe"] = lp_acc

    # ── Step 4: Ensemble CLIP probe + EfficientNet ─────────────────────────────
    print("\n── Step 4: Ensemble (CLIP Linear Probe + EfficientNet) ──────────────")
    effnet = load_efficientnet(num_classes)

    if effnet is not None:
        print("  Getting EfficientNet probabilities…")
        en_probs,     _  = get_effnet_probs(effnet, test_sub)
        en_val_probs, _  = get_effnet_probs(effnet, val_sub)
        effnet_test_acc  = acc(en_probs, test_labels)
        effnet_val_acc   = acc(en_val_probs, val_labels)
        print(f"  EfficientNet alone — val={effnet_val_acc:.2f}%  test={effnet_test_acc:.2f}%")

        best_w, val_ens_acc = tune_weight(lp_val_probs, en_val_probs, val_labels)
        ens_probs = blend(lp_probs, en_probs, best_w)
        ens_acc   = acc(ens_probs, test_labels)
        print(f"  Optimal weight: CLIP×{best_w} + EfficientNet×{1-best_w:.2f}")
        print(f"  Ensemble val={val_ens_acc:.2f}%  test={ens_acc:.2f}%  "
              f"(Δ vs EfficientNet alone = {ens_acc-effnet_test_acc:+.2f}%)")
        results["Ensemble\n(CLIP Probe\n+ EffNet)"] = ens_acc
    else:
        effnet_test_acc = 0.0
        ens_probs = lp_probs
        best_w    = 1.0

    # ── Step 5: Ensemble + TTA ─────────────────────────────────────────────────
    print("\n── Step 5: Ensemble + TTA (8 augmented views) ───────────────────────")

    print("  EfficientNet TTA…")
    en_tta_probs_list = []
    for aug_idx, pil_aug in enumerate(PIL_TTA_AUGS):
        p, _ = get_effnet_probs(effnet, test_sub, pil_aug=pil_aug)
        en_tta_probs_list.append(p)
        print(f"    aug {aug_idx+1}/{len(PIL_TTA_AUGS)}")
    en_tta_probs = np.mean(en_tta_probs_list, axis=0)

    print("  CLIP TTA (re-using cached features + PIL augmentation)…")
    clip_tta_probs_list = []
    for aug_idx, pil_aug in enumerate(PIL_TTA_AUGS):
        aug_feats, aug_labels = [], []
        buf_imgs, buf_labs = [], []
        batch = 64
        for i in range(len(test_sub)):
            img, lbl = test_sub[i]
            buf_imgs.append(pil_aug(img))
            buf_labs.append(lbl)
            if len(buf_imgs) == batch or i == len(test_sub) - 1:
                with torch.no_grad():
                    inp        = clip_processor(images=buf_imgs, return_tensors="pt",
                                               padding=True).to(DEVICE)
                    vision_out = clip_model.vision_model(pixel_values=inp["pixel_values"])
                    projected  = clip_model.visual_projection(vision_out.pooler_output)
                    f          = F.normalize(projected, dim=-1)
                aug_feats.append(f.cpu().numpy())
                aug_labels.extend(buf_labs)
                buf_imgs, buf_labs = [], []
        aug_feats = np.vstack(aug_feats).astype(np.float32)
        _, p = zero_shot_eval(aug_feats, np.array(aug_labels, dtype=np.int64), text_ens_np)
        clip_tta_probs_list.append(p)
        print(f"    aug {aug_idx+1}/{len(PIL_TTA_AUGS)}")

    clip_tta_probs = np.mean(clip_tta_probs_list, axis=0)

    # blend TTA versions with the same optimal weight from Step 4
    if effnet is not None:
        tta_ens_probs = blend(clip_tta_probs, en_tta_probs, best_w)
    else:
        tta_ens_probs = clip_tta_probs

    tta_acc = acc(tta_ens_probs, test_labels)
    print(f"  Ensemble + TTA test accuracy: {tta_acc:.2f}%  "
          f"(Δ vs no-TTA ensemble = {tta_acc - ens_acc if effnet else tta_acc - lp_acc:+.2f}%)")
    results["Ensemble\n+ TTA"] = tta_acc

    # ── Summary ───────────────────────────────────────────────────────────────
    effnet_baseline = effnet_test_acc if effnet is not None else 0.0

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for label, a in results.items():
        marker = " ← best" if a == max(results.values()) else ""
        print(f"  {label.replace(chr(10), ' '):35s}  {a:.2f}%{marker}")
    if effnet is not None:
        print(f"  {'EfficientNet fine-tuned (baseline)':35s}  {effnet_baseline:.2f}%")

    save_chart(results, effnet_baseline)

    pd.DataFrame({
        "method":   [k.replace("\n", " ") for k in results],
        "test_acc": list(results.values()),
    }).to_csv(os.path.join(RESULTS_DIR, "clip_results_summary.csv"), index=False)
    print(f"\n✓  All done.  Results in {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
