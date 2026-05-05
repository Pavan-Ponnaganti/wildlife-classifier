"""
Wildlife Species Identifier
Streamlit app: upload animal photo → EfficientNet top-3 → Wikipedia + IUCN badge
"""

import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

import timm

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Wildlife Lens · Species Identifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Animal classes (90 species) ──────────────────────────────────────────────
ANIMALS = [
    "antelope","badger","bat","bear","bee","beetle","bison","boar","butterfly",
    "cat","caterpillar","chimpanzee","cockroach","cow","coyote","crab","crow",
    "deer","dog","dolphin","donkey","dragonfly","duck","eagle","elephant",
    "flamingo","fly","fox","goat","goldfish","goose","gorilla","grasshopper",
    "hamster","hare","hedgehog","hippopotamus","hornbill","horse","hummingbird",
    "hyena","jellyfish","kangaroo","koala","ladybugs","leopard","lion","lizard",
    "lobster","mosquito","moth","mouse","octopus","okapi","orangutan","otter",
    "owl","ox","oyster","panda","parrot","pelecaniformes","penguin","pig",
    "pigeon","porcupine","possum","raccoon","rat","reindeer","rhinoceros",
    "sandpiper","seahorse","seal","shark","sheep","snake","sparrow","squid",
    "squirrel","starfish","swan","tiger","turkey","turtle","whale","wolf",
    "wombat","woodpecker","zebra",
]

# IUCN status metadata
IUCN_META = {
    "EX":  {"label": "Extinct",             "emoji": "⚫", "color": "#1a1a1a", "bg": "#3d3d3d"},
    "EW":  {"label": "Extinct in Wild",     "emoji": "⚫", "color": "#2d1b4e", "bg": "#4a3066"},
    "CR":  {"label": "Critically Endangered","emoji": "🔴","color": "#7f0000", "bg": "#c0392b"},
    "EN":  {"label": "Endangered",          "emoji": "🟠", "color": "#7f3300", "bg": "#e67e22"},
    "VU":  {"label": "Vulnerable",          "emoji": "🟡", "color": "#7f6500", "bg": "#f1c40f"},
    "NT":  {"label": "Near Threatened",     "emoji": "🟢", "color": "#1a4a1a", "bg": "#27ae60"},
    "LC":  {"label": "Least Concern",       "emoji": "🟢", "color": "#0d3320", "bg": "#2ecc71"},
    "DD":  {"label": "Data Deficient",      "emoji": "⚪", "color": "#3d3d3d", "bg": "#95a5a6"},
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

/* Root */
:root {
    --forest-dark:  #0a1f0e;
    --forest-mid:   #0f2d14;
    --forest-card:  #122918;
    --accent-green: #4ade80;
    --accent-amber: #fbbf24;
    --accent-teal:  #2dd4bf;
    --text-primary: #e8f5e9;
    --text-muted:   #86a88a;
    --border:       rgba(74, 222, 128, 0.15);
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--forest-dark) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #071510 0%, #0a1f0e 40%, #071510 100%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061209 0%, #0a1a0d 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Uploader */
[data-testid="stFileUploader"] {
    background: rgba(18, 41, 24, 0.6) !important;
    border: 2px dashed rgba(74, 222, 128, 0.3) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74, 222, 128, 0.6) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(74,222,128,0.2) !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(74,222,128,0.35) !important;
}

/* Progress bars */
.stProgress > div > div {
    background: linear-gradient(90deg, #16a34a, #4ade80) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: rgba(74,222,128,0.1) !important;
    border-radius: 999px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-green) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Selectbox / inputs */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--forest-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

/* Images */
img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(22,163,74,0.15) 0%, rgba(15,45,20,0.8) 50%, rgba(21,128,61,0.15) 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
    <div style="position:relative; z-index:2;">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
            <span style="font-size:2.5rem;">🌿</span>
            <h1 style="
                font-family:'Playfair Display', serif;
                font-size:2.4rem;
                font-weight:800;
                margin:0;
                background: linear-gradient(135deg, #4ade80, #fbbf24);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">Wildlife Lens</h1>
        </div>
        <p style="color:#86a88a; font-size:1.05rem; margin:0; font-weight:400;">
            Upload a forest photo · Identify the species · Discover its story
        </p>
    </div>
    <div style="
        position:absolute; top:-50px; right:-50px;
        width:250px; height:250px;
        background: radial-gradient(circle, rgba(74,222,128,0.08) 0%, transparent 70%);
        border-radius:50%;
    "></div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <span style="font-size:3rem;">🔭</span>
        <h2 style="font-family:'Playfair Display',serif; font-size:1.4rem; margin:0.5rem 0 0;">
            How It Works
        </h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 0 0.5rem;">
    <div style="background:rgba(74,222,128,0.08); border-left:3px solid #4ade80; border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem;">
        <b style="color:#4ade80;">① Upload</b><br>
        <span style="color:#86a88a; font-size:0.9rem;">Drag & drop any animal photo (JPG/PNG)</span>
    </div>
    <div style="background:rgba(251,191,36,0.08); border-left:3px solid #fbbf24; border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem;">
        <b style="color:#fbbf24;">② Model Identifies</b><br>
        <span style="color:#86a88a; font-size:0.9rem;">Inference over wildlife species using fine-tuned EfficientNet-B0</span>
    </div>
    <div style="background:rgba(45,212,191,0.08); border-left:3px solid #2dd4bf; border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem;">
        <b style="color:#2dd4bf;">③ Discover</b><br>
        <span style="color:#86a88a; font-size:0.9rem;">Wikipedia summary + IUCN conservation status + fun fact</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <p style="color:#86a88a; font-size:0.85rem; margin:0;">
            <b style="color:#4ade80;">Model:</b> efficientnet_b0<br>
            <b style="color:#4ade80;">Method:</b> Image classification<br>
            <b style="color:#4ade80;">Dataset:</b> Animals-90
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <p style="color:#4a6650; font-size:0.78rem; text-align:center;">
            Built with EfficientNet · Streamlit · Wikipedia API
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists('classes.json'):
        return None, None, None
    with open('classes.json', 'r') as f:
        class_names = json.load(f)
        
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_names))
    
    if os.path.exists('efficientnet_b0_animals.pth'):
        model.load_state_dict(torch.load('efficientnet_b0_animals.pth', map_location=torch.device('cpu')))
    else:
        st.warning("Could not find the trained model weights 'efficientnet_b0_animals.pth'. Using untrained weights for demonstration.")
        
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, class_names, preprocess



@st.cache_data(show_spinner=False)
def load_iucn_cache():
    cache_path = Path(__file__).parent / "iucn_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


# ── EfficientNet inference ─────────────────────────────────────────────────────────────
def predict_top3(image: Image.Image, model, class_names, preprocess):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    top3 = [(class_names[top3_idx[i].item()], float(top3_prob[i])) for i in range(3)]
    return top3


# ── Wikipedia summary ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_wikipedia_summary(species: str) -> str:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        # Try exact search first
        try:
            page = wikipedia.page(species, auto_suggest=False)
            summary = wikipedia.summary(species, sentences=3, auto_suggest=False)
            return summary
        except Exception:
            summary = wikipedia.summary(species, sentences=3)
            return summary
    except Exception as e:
        return f"Wikipedia summary unavailable for '{species}'."


# ── IUCN badge HTML ────────────────────────────────────────────────────────────
def iucn_badge_html(status_code: str) -> str:
    meta = IUCN_META.get(status_code, IUCN_META["DD"])
    return f"""<div style="
        display:inline-flex; align-items:center; gap:0.6rem;
        background: linear-gradient(135deg, {meta['bg']}22, {meta['bg']}44);
        border: 1.5px solid {meta['bg']};
        border-radius: 999px;
        padding: 0.4rem 1.2rem;
        font-size: 0.95rem;
        font-weight: 600;
    ">
        {meta['emoji']}
        <span style="color:{meta['bg']}; letter-spacing:0.05em;">{status_code}</span>
        <span style="color:#86a88a; font-size:0.85rem;">· {meta['label']}</span>
    </div>"""


# ── Confidence bar HTML ────────────────────────────────────────────────────────
def confidence_bar_html(rank: int, species: str, prob: float, is_top: bool) -> str:
    pct = prob * 100
    bar_color = "linear-gradient(90deg, #16a34a, #4ade80)" if is_top else "linear-gradient(90deg, #1e5631, #2d7d46)"
    rank_color = "#4ade80" if is_top else "#2d7d46"
    medal = ["🥇", "🥈", "🥉"][rank]
    box_shadow = "box-shadow: 0 0 20px rgba(74,222,128,0.1);" if is_top else ""
    return f"""<div style="
        background: rgba(18,41,24,0.7);
        border: 1px solid {'rgba(74,222,128,0.3)' if is_top else 'rgba(74,222,128,0.1)'};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem; {box_shadow}
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1.2rem;">{medal}</span>
                <span style="font-weight:{'700' if is_top else '500'}; font-size:{'1.05rem' if is_top else '0.95rem'}; color:{'#e8f5e9' if is_top else '#86a88a'}; text-transform:capitalize;">
                    {species}
                </span>
            </div>
            <span style="font-weight:700; color:{rank_color}; font-size:{'1.1rem' if is_top else '0.95rem'};">
                {pct:.1f}%
            </span>
        </div>
        <div style="background:rgba(74,222,128,0.08); border-radius:999px; height:6px; overflow:hidden;">
            <div style="
                background: {bar_color};
                height:100%;
                width:{min(pct * 1.5, 100):.1f}%;
                border-radius:999px;
                transition: width 0.8s ease;
            "></div>
        </div>
    </div>"""


# ── Main UI ───────────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown("""
    <h3 style="color:#4ade80; font-size:1.1rem; font-weight:600; margin-bottom:1rem; letter-spacing:0.05em;">
        📷 UPLOAD YOUR PHOTO
    </h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an animal photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True, caption="")

        # Image metadata strip
        w, h = image.size
        st.markdown(f"""
        <div style="
            display:flex; gap:1rem; margin-top:0.5rem;
            background:rgba(18,41,24,0.5); border-radius:10px; padding:0.6rem 1rem;
        ">
            <span style="color:#86a88a; font-size:0.82rem;">📐 {w}×{h}px</span>
            <span style="color:#86a88a; font-size:0.82rem;">🖼️ {uploaded_file.name}</span>
            <span style="color:#86a88a; font-size:0.82rem;">💾 {uploaded_file.size/1024:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background:rgba(18,41,24,0.3); border:1px dashed rgba(74,222,128,0.15);
            border-radius:16px; padding:3rem 2rem; text-align:center; margin-top:1rem;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🦁</div>
            <p style="color:#4a6650; font-size:0.9rem; margin:0;">
                Upload a photo to identify the species
            </p>
        </div>
        """, unsafe_allow_html=True)


with col_results:
    st.markdown("""
    <h3 style="color:#4ade80; font-size:1.1rem; font-weight:600; margin-bottom:1rem; letter-spacing:0.05em;">
        🔬 IDENTIFICATION RESULTS
    </h3>
    """, unsafe_allow_html=True)

    if uploaded_file:
        iucn_cache = load_iucn_cache()

        with st.spinner("🌿 Loading EfficientNet model..."):
            model, class_names, preprocess = load_model()

        if model is None:
            st.error("Model or classes.json not found. Please ensure train.py has been executed to map classes.")
            st.stop()

        with st.spinner("🔍 Identifying species..."):
            top3 = predict_top3(image, model, class_names, preprocess)

        top_species, top_prob = top3[0]

        # ── Top prediction hero card ──────────────────────────────────────────
        iucn_data = iucn_cache.get(top_species, {})
        iucn_status = iucn_data.get("iucn_status", "DD")
        fun_fact = iucn_data.get("fun_fact", "")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(22,163,74,0.12), rgba(15,41,20,0.9));
            border: 1px solid rgba(74,222,128,0.25);
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="position:relative; z-index:2;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.3rem;">
                    <span style="color:#4a6650; font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Top Match</span>
                </div>
                <h2 style="
                    font-family:'Playfair Display',serif;
                    font-size:2rem;
                    font-weight:800;
                    text-transform:capitalize;
                    margin:0 0 0.6rem;
                    background: linear-gradient(135deg, #4ade80, #fbbf24);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">{top_species}</h2>
                <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                    {iucn_badge_html(iucn_status).strip()}
                    <span style="color:#4a6650; font-size:0.85rem;">Confidence: <b style="color:#4ade80;">{top_prob*100:.1f}%</b></span>
                </div>
            </div>
            <div style="
                position:absolute; bottom:-30px; right:-30px;
                width:150px; height:150px;
                background: radial-gradient(circle, rgba(74,222,128,0.06) 0%, transparent 70%);
                border-radius:50%;
            "></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-3 confidence bars ─────────────────────────────────────────────
        st.markdown("""
        <p style="color:#86a88a; font-size:0.85rem; font-weight:600; letter-spacing:0.08em; margin-bottom:0.6rem;">
            TOP 3 PREDICTIONS
        </p>
        """, unsafe_allow_html=True)

        bars_html = ""
        for i, (species, prob) in enumerate(top3):
            bars_html += confidence_bar_html(i, species, prob, i == 0)
        st.markdown(bars_html, unsafe_allow_html=True)

        # ── Fun fact ──────────────────────────────────────────────────────────
        if fun_fact:
            st.markdown(f"""
            <div style="
                background: rgba(251,191,36,0.06);
                border: 1px solid rgba(251,191,36,0.2);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 1.2rem;
            ">
                <div style="display:flex; align-items:flex-start; gap:0.7rem;">
                    <span style="font-size:1.3rem; margin-top:0.1rem;">💡</span>
                    <div>
                        <p style="color:#4a6650; font-size:0.78rem; font-weight:600; letter-spacing:0.08em; margin:0 0 0.3rem; text-transform:uppercase;">Fun Fact</p>
                        <p style="color:#d4a017; font-size:0.92rem; margin:0; line-height:1.55;">{fun_fact}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Wikipedia summary ─────────────────────────────────────────────────
        st.markdown("""
        <p style="color:#86a88a; font-size:0.85rem; font-weight:600; letter-spacing:0.08em; margin-bottom:0.6rem;">
            📖 ABOUT THIS SPECIES
        </p>
        """, unsafe_allow_html=True)

        with st.spinner("📚 Fetching Wikipedia summary..."):
            wiki_text = get_wikipedia_summary(top_species)

        st.markdown(f"""
        <div style="
            background: rgba(45,212,191,0.05);
            border: 1px solid rgba(45,212,191,0.15);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
        ">
            <p style="color:#b2d8d8; font-size:0.92rem; line-height:1.7; margin:0;">{wiki_text}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state for results panel
        st.markdown("""
        <div style="
            background:rgba(18,41,24,0.3); border:1px dashed rgba(74,222,128,0.12);
            border-radius:18px; padding:4rem 2rem; text-align:center;
        ">
            <div style="font-size:4rem; margin-bottom:1.5rem; opacity:0.4;">🔬</div>
            <p style="color:#4a6650; font-size:0.95rem; margin:0 0 0.5rem;">
                Results will appear here after upload
            </p>
            <p style="color:#2d4a35; font-size:0.82rem; margin:0;">
                Supports 90 wildlife species
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    border-top: 1px solid rgba(74,222,128,0.1);
    padding-top: 1.5rem;
    text-align: center;
">
    <div style="display:flex; justify-content:center; gap:3rem; flex-wrap:wrap; margin-bottom:0.8rem;">
        <div style="text-align:center;">
            <p style="color:#4ade80; font-size:1.3rem; font-weight:700; margin:0;">90</p>
            <p style="color:#4a6650; font-size:0.78rem; margin:0;">Species</p>
        </div>
        <div style="text-align:center;">
            <p style="color:#fbbf24; font-size:1.3rem; font-weight:700; margin:0;">EfficientNet</p>
            <p style="color:#4a6650; font-size:0.78rem; margin:0;">CNN Model</p>
        </div>
        <div style="text-align:center;">
            <p style="color:#2dd4bf; font-size:1.3rem; font-weight:700; margin:0;">IUCN</p>
            <p style="color:#4a6650; font-size:0.78rem; margin:0;">Conservation Data</p>
        </div>
        <div style="text-align:center;">
            <p style="color:#818cf8; font-size:1.3rem; font-weight:700; margin:0;">Wiki</p>
            <p style="color:#4a6650; font-size:0.78rem; margin:0;">Species Info</p>
        </div>
    </div>
    <p style="color:#2d4a35; font-size:0.78rem; margin:0;">
        Wildlife Lens · SMAI Assignment 3 · efficientnet_b0
    </p>
</div>
""", unsafe_allow_html=True)
