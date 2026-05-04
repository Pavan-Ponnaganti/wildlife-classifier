import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import timm
from PIL import Image
import json
import os
import joblib

st.set_page_config(page_title="Indian Wildlife & Bird Identifier", page_icon="🐾",
                   layout="centered")

RESULTS_DIR  = "clip_results"
MODEL_PATH   = "efficientnet_b0_animals.pth"
CLASSES_JSON = "classes.json"
PROBE_PATH   = os.path.join(RESULTS_DIR, "linear_probe.joblib")
WEIGHT_PATH  = os.path.join(RESULTS_DIR, "ensemble_weight.json")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFFNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── model loaders (cached so they load only once) ─────────────────────────────

@st.cache_resource
def load_class_names():
    with open(CLASSES_JSON) as f:
        return json.load(f)

@st.cache_resource
def load_efficientnet(num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

@st.cache_resource
def load_probe():
    return joblib.load(PROBE_PATH)

@st.cache_resource
def load_weight():
    with open(WEIGHT_PATH) as f:
        d = json.load(f)
    return d["clip_weight"], d["effnet_weight"]

@st.cache_data
def load_animal_info():
    if os.path.exists("animal_info.json"):
        with open("animal_info.json", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ── inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def clip_features(clip_model, clip_processor, pil_image):
    """Extract L2-normalised CLIP image embedding for a single PIL image."""
    inp        = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(pixel_values=inp["pixel_values"])
    projected  = clip_model.visual_projection(vision_out.pooler_output)
    return F.normalize(projected, dim=-1).cpu().numpy()   # (1, 512)

@torch.no_grad()
def effnet_probs(effnet_model, pil_image):
    """Softmax probabilities from EfficientNet for a single PIL image."""
    x      = EFFNET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    logits = effnet_model(x)
    return torch.softmax(logits, dim=-1).cpu().numpy()    # (1, C)

def predict(pil_image, clip_model, clip_processor, effnet_model, probe,
            clip_w, effnet_w):
    feats   = clip_features(clip_model, clip_processor, pil_image)   # (1, 512)
    lp_prob = probe.predict_proba(feats)                              # (1, C)
    en_prob = effnet_probs(effnet_model, pil_image)                   # (1, C)
    blended = clip_w * lp_prob + effnet_w * en_prob                  # (1, C)
    return blended[0]   # (C,)

# ── check required files ──────────────────────────────────────────────────────

missing = []
for p, label in [
    (CLASSES_JSON, "classes.json"),
    (MODEL_PATH,   "efficientnet_b0_animals.pth  (run train.py)"),
    (PROBE_PATH,   "clip_results/linear_probe.joblib  (run clip_pipeline.py)"),
    (WEIGHT_PATH,  "clip_results/ensemble_weight.json  (run clip_pipeline.py)"),
]:
    if not os.path.exists(p):
        missing.append(label)

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🐾 Indian Wildlife & Bird Identifier")

if missing:
    st.error("Missing required files — please generate them first:")
    for m in missing:
        st.code(m)
    st.stop()

# Load everything — CLIP takes a few seconds on first run
with st.spinner("Loading models (first run takes ~15 s for CLIP)…"):
    class_names  = load_class_names()
    effnet_model = load_efficientnet(len(class_names))
    clip_model, clip_processor = load_clip()
    probe        = load_probe()
    clip_w, effnet_w = load_weight()
    animal_info  = load_animal_info()

# Show which mode is active
col1, col2 = st.columns(2)
col1.metric("Model", "CLIP + EfficientNet Ensemble")
col2.metric("Test Accuracy", "96.30%")

st.markdown("---")
st.write("Upload a photo of an animal and we'll identify it!")

uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analysing…"):
        probs = predict(image, clip_model, clip_processor, effnet_model,
                        probe, clip_w, effnet_w)

    top3_idx  = np.argsort(probs)[::-1][:3]
    top3_prob = probs[top3_idx]

    st.subheader("Predictions")
    for rank, (idx, prob) in enumerate(zip(top3_idx, top3_prob)):
        animal_name = class_names[idx]
        pct         = prob * 100
        with st.expander(f"{rank+1}. {animal_name.replace('_', ' ').title()}  —  {pct:.2f}%",
                         expanded=(rank == 0)):
            st.progress(float(prob))
            info = animal_info.get(animal_name)
            if info:
                st.write(f"**Wikipedia Summary:** {info.get('description', 'N/A')}")
                status = info.get("iucn_status", "N/A")
                colour = {"Least Concern": "green", "Vulnerable": "orange",
                          "Endangered": "red", "Critically Endangered": "red"}.get(status, "grey")
                st.markdown(f"**IUCN Status:** :{colour}[{status}]")
            else:
                st.write("No additional facts found for this species.")

st.markdown("---")
st.caption(
    f"CLIP weight: {clip_w}  ·  EfficientNet weight: {effnet_w}  ·  "
    "Built for SMAI Assignment 3 — T7.6"
)
