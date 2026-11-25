import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st
import torch

from app.hf_loader import (
    HF_REPO_ID,
    load_part_a_bundle,
    load_part_b_bundle,
    load_part_c_bundle,
    load_part_d_retriever,
    load_part_d_generator,
)
from part_d_rag.generate import generate_answer, MEDICAL_NOTICE

st.set_page_config(
    page_title="Four Generative Models, One App",
    page_icon="🧬",
    layout="wide",
)

# ==============================================================================
# DESIGN SYSTEM  (light, clean, one accent per model)
# ==============================================================================
ACCENT = {
    "mae": "#4f46e5",   # indigo
    "vae": "#db2777",   # rose
    "ddpm": "#0891b2",  # cyan
    "rag": "#059669",   # emerald
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root { color-scheme: light; }

/* ---------- Base (force light colors even if the browser/theme is dark) ---------- */
html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1200px 500px at 10% -10%, #e0e7ff 0%, transparent 60%),
        radial-gradient(900px 450px at 100% 0%, #fce7f3 0%, transparent 55%),
        #f6f8fc;
    color: #0f172a;
}
.block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] * { color: #475569 !important; }
.stAppDeployButton, [data-testid="stDeployButton"] { display: none !important; }

h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #0f172a !important; letter-spacing: -0.01em; }
.stApp p, .stApp li, .stApp span { color: #1e293b; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #64748b !important; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 45%, #ecfeff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 2rem 2.2rem 1.6rem 2.2rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 12px 32px rgba(79,70,229,.08);
    margin-bottom: 1.4rem;
}
.hero .eyebrow {
    font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
    color: #4f46e5 !important; margin-bottom: .5rem;
}
.hero h1 {
    margin: 0; font-weight: 800; font-size: 2.5rem; line-height: 1.15; color: #0f172a !important;
}
.hero h1 .grad {
    background: linear-gradient(90deg, #4f46e5, #db2777 55%, #0891b2);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
}
.hero p.lead { margin: .7rem 0 1.3rem 0; font-size: 1.04rem; color: #475569 !important; max-width: 820px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; }
.tile {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: .8rem .95rem; display: flex; gap: .7rem; align-items: center;
}
.tile .ico {
    width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center;
    justify-content: center; font-size: 1.15rem; flex-shrink: 0;
}
.tile .t1 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: .92rem; color: #0f172a !important; line-height: 1.2; }
.tile .t2 { font-size: .76rem; color: #64748b !important; }
.repo {
    display: inline-block; margin-top: 1rem; padding: .3rem .8rem; border-radius: 999px;
    background: #ffffff; border: 1px solid #e2e8f0; font-size: .78rem; color: #475569 !important;
}
.repo b { color: #0f172a !important; }
@media (max-width: 900px) { .tiles { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.9rem; } }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #e9edf5; padding: 5px; border-radius: 14px; border: none;
}
.stTabs button[data-baseweb="tab"] {
    height: 44px; padding: 0 18px; border-radius: 10px; background: transparent;
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600;
}
.stTabs button[data-baseweb="tab"] p { color: #475569 !important; font-weight: 600; font-size: .92rem; }
.stTabs button[data-baseweb="tab"]:hover p { color: #0f172a !important; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff; box-shadow: 0 1px 3px rgba(15,23,42,.12);
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { color: #4f46e5 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
    display: none !important; height: 0 !important; background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* ---------- Section banner ---------- */
.banner {
    background: #ffffff; border: 1px solid #e2e8f0; border-left-width: 5px;
    border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
    display: flex; gap: 1rem; align-items: flex-start;
}
.banner .ico {
    width: 46px; height: 46px; border-radius: 13px; display: flex; align-items: center;
    justify-content: center; font-size: 1.4rem; flex-shrink: 0;
}
.banner h3 { margin: 0 0 .2rem 0; font-size: 1.25rem; font-weight: 700; }
.stApp .banner p { margin: 0; color: #475569; font-size: .94rem; }
.banner .pills { margin-top: .6rem; }
.pill {
    display: inline-block; padding: .18rem .7rem; margin: 0 .4rem .3rem 0;
    border-radius: 999px; font-size: .74rem; font-weight: 600;
}

/* ---------- Cards (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e2e8f0 !important; border-radius: 16px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 18px rgba(15,23,42,.04);
}
.card-title {
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1rem;
    color: #0f172a; margin-bottom: .5rem;
}

/* ---------- Form controls ---------- */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
    color: #334155 !important; font-weight: 500; font-size: .88rem;
}
div[data-baseweb="select"] > div {
    background: #ffffff !important; border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important; min-height: 42px;
}
div[data-baseweb="select"] *, div[data-baseweb="select"] input { color: #0f172a !important; -webkit-text-fill-color: #0f172a; }
div[data-baseweb="select"] svg { fill: #64748b !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="popover"] [role="option"] {
    background: #ffffff !important; color: #0f172a !important;
}
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] { background: #eef2ff !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: #ffffff !important; border-radius: 10px !important; border-color: #cbd5e1 !important;
}
div[data-baseweb="input"] { border: 1px solid #cbd5e1 !important; }
div[data-baseweb="input"] input {
    color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; background: #ffffff !important;
}
div[data-baseweb="input"] input::placeholder { color: #94a3b8 !important; -webkit-text-fill-color: #94a3b8 !important; }
div[data-baseweb="input"]:focus-within { border-color: #4f46e5 !important; box-shadow: 0 0 0 3px rgba(79,70,229,.15); }
[data-testid="stNumberInput"] button { background: #f1f5f9 !important; color: #334155 !important; border: none !important; }
[data-testid="stNumberInput"] button svg { fill: #334155 !important; }
[data-testid="stSlider"] [role="slider"] { background: #4f46e5 !important; box-shadow: 0 0 0 4px rgba(79,70,229,.18); }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #475569 !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: #4f46e5; border: none; border-radius: 10px; padding: .55rem 1.2rem;
    font-weight: 600; box-shadow: 0 1px 2px rgba(15,23,42,.12), 0 4px 12px rgba(79,70,229,.25);
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton > button p { color: #ffffff !important; font-weight: 600; }
.stButton > button:hover {
    background: #4338ca; transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(15,23,42,.14), 0 8px 18px rgba(79,70,229,.32);
}
.stButton > button:active { transform: translateY(0); }

/* ---------- Images ---------- */
[data-testid="stImage"] img {
    border-radius: 12px; border: 1px solid #e2e8f0; background: #f8fafc;
    box-shadow: 0 1px 3px rgba(15,23,42,.08); transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stImage"] img:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(15,23,42,.14); }
[data-testid="stImageCaption"] { color: #64748b !important; font-size: .76rem; text-align: center; }

/* ---------- Progress / alerts / expanders ---------- */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #4f46e5, #db2777); }
[data-testid="stProgress"] p { color: #334155 !important; font-size: .85rem; }
div[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden;
}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color: #0f172a !important; font-weight: 600; }
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p { color: #334155 !important; line-height: 1.6; }

/* ---------- Footer ---------- */
.footer { text-align: center; color: #64748b; font-size: .84rem; margin-top: 2.2rem; padding-top: 1.1rem; border-top: 1px solid #e2e8f0; }
.footer b { color: #0f172a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==============================================================================
# UI HELPERS
# ==============================================================================
def hero(repo_id: str):
    tiles = [
        ("mae", "🛰️", "Masked Autoencoder", "EuroSAT · satellite tiles"),
        ("vae", "✍️", "Variational Autoencoder", "UHaT · Urdu characters"),
        ("ddpm", "🌫️", "Conditional Diffusion", "UHaT · DDPM sampling"),
        ("rag", "🧪", "Biomedical RAG", "BioASQ · PubMed QA"),
    ]
    tiles_html = "".join(
        f"""<div class="tile">
                <div class="ico" style="background:{ACCENT[k]}18;">{ico}</div>
                <div><div class="t1">{name}</div><div class="t2">{sub}</div></div>
            </div>"""
        for k, ico, name, sub in tiles
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Generative AI Showcase</div>
            <h1>Four Generative Models, <span class="grad">One App</span></h1>
            <p class="lead">Explore masked reconstruction, latent-space morphing, diffusion sampling
               and retrieval-augmented biomedical QA in a single interactive workspace.</p>
            <div class="tiles">{tiles_html}</div>
            <span class="repo">🤗 Weights hosted on Hugging Face · <b>{repo_id}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_banner(key: str, emoji: str, title: str, desc: str, pills=()):
    c = ACCENT[key]
    pills_html = "".join(
        f'<span class="pill" style="background:{c}14; color:{c}; border:1px solid {c}33;">{p}</span>'
        for p in pills
    )
    st.markdown(
        f"""
        <div class="banner" style="border-left-color:{c};">
            <div class="ico" style="background:{c}14;">{emoji}</div>
            <div>
                <h3>{title}</h3>
                <p>{desc}</p>
                <div class="pills">{pills_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_title(text: str):
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Part A (MAE + Few-Label Classifier) from Hugging Face...")
def cached_part_a():
    return load_part_a_bundle()


@st.cache_resource(show_spinner="Loading Part B (ConvVAE) from Hugging Face...")
def cached_part_b():
    return load_part_b_bundle()


@st.cache_resource(show_spinner="Loading Part C (Class-Conditional DDPM) from Hugging Face...")
def cached_part_c():
    return load_part_c_bundle()


@st.cache_resource(show_spinner="Loading Part D Dense Retriever & 79,332 BioASQ Embeddings...")
def cached_part_d_retriever():
    return load_part_d_retriever()


@st.cache_resource(show_spinner="Loading Part D Generator (Qwen2.5-1.5B-Instruct)...")
def cached_part_d_generator():
    return load_part_d_generator()


def tensor_to_hwc_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [0, 1] to displayable numpy array."""
    arr = img_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


# ==============================================================================
# HEADER + TABS
# ==============================================================================
hero(HF_REPO_ID)

tab_mae, tab_vae, tab_ddpm, tab_rag = st.tabs(
    [
        "🛰️  MAE · Satellite",
        "✍️  VAE · Urdu Handwriting",
        "🌫️  Diffusion · Urdu",
        "🧪  RAG · Biomedical QA",
    ]
)

# ==============================================================================
# TAB 1: MASKED AUTOENCODER (Part A)
# ==============================================================================
with tab_mae:
    section_banner(
        "mae", "🛰️",
        "Masked Autoencoder & 16-Shot Classifier",
        "Hide 75% of a satellite tile, watch the model rebuild it, then see what a linear "
        "probe trained on just 16 labels per class predicts.",
        pills=["75% masking", "64×64 RGB", "16-shot linear probe"],
    )

    mae, classifier, mae_samples, dev_a = cached_part_a()
    class_names_a = mae_samples["class_names"]
    imgs_a = mae_samples["images"]  # (30, 3, 64, 64)
    lbls_a = mae_samples["labels"]

    options_a = [
        f"Tile #{i:02d} — True Class: {class_names_a[int(lbls_a[i])]}"
        for i in range(len(imgs_a))
    ]

    with st.container(border=True):
        col_ctrl, col_seed = st.columns([3, 1])
        with col_ctrl:
            selected_idx = st.selectbox(
                "Select a held-out EuroSAT test tile (64×64 RGB):",
                options=range(len(options_a)),
                format_func=lambda idx: options_a[idx],
            )
        with col_seed:
            mask_seed = st.number_input("Random Mask Seed", min_value=0, max_value=9999, value=42, step=1)

    tile = imgs_a[selected_idx : selected_idx + 1].to(dev_a)
    true_label = class_names_a[int(lbls_a[selected_idx])]

    torch.manual_seed(int(mask_seed))
    masked_img, recon_img, orig_img = mae.reconstruct_images(tile, mask_ratio=0.75)

    with torch.no_grad():
        logits = classifier(tile)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = class_names_a[pred_idx]
        pred_conf = float(probs[pred_idx]) * 100.0

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
        with c1:
            st.image(
                tensor_to_hwc_uint8(masked_img[0]),
                caption="Masked input (48/64 patches hidden)",
                use_container_width=True,
            )
        with c2:
            st.image(
                tensor_to_hwc_uint8(recon_img[0]),
                caption="MAE reconstruction",
                use_container_width=True,
            )
        with c3:
            st.image(
                tensor_to_hwc_uint8(orig_img[0]),
                caption=f"Original ({true_label})",
                use_container_width=True,
            )
        with c4:
            card_title("Few-label classifier prediction")
            if pred_label == true_label:
                st.success(f"Predicted: **{pred_label}** ({pred_conf:.1f}%)", icon="✅")
            else:
                st.warning(f"Predicted: **{pred_label}** ({pred_conf:.1f}%) | True: **{true_label}**", icon="⚠️")

            top3_idx = np.argsort(probs)[::-1][:3]
            for rank_i in top3_idx:
                st.progress(float(probs[rank_i]), text=f"{class_names_a[rank_i]}: {probs[rank_i]*100:.1f}%")

# ==============================================================================
# TAB 2: VARIATIONAL AUTOENCODER (Part B)
# ==============================================================================
with tab_vae:
    section_banner(
        "vae", "✍️",
        "Convolutional Variational Autoencoder",
        "Sample brand-new Urdu characters from the prior, or morph smoothly between two real "
        "characters through the latent space.",
        pills=["Latent dim = 16", "28×28 grayscale", "40 UHaT classes"],
    )

    vae, vae_samples, dev_b = cached_part_b()
    class_names_b = vae_samples["class_names"]
    imgs_b = vae_samples["images"]  # (80, 1, 28, 28)
    lbls_b = vae_samples["labels"]

    left_vae, right_vae = st.columns([1, 1.2])

    with left_vae:
        with st.container(border=True):
            card_title("1 · Prior sampling: decode z ~ N(0, I)")
            if st.button("Generate 16 new random characters", key="vae_sample_btn"):
                st.session_state["vae_prior_z"] = torch.randn(16, vae.latent_dim)
            if "vae_prior_z" not in st.session_state:
                torch.manual_seed(42)
                st.session_state["vae_prior_z"] = torch.randn(16, vae.latent_dim)

            with torch.no_grad():
                decoded_prior = vae.decode(st.session_state["vae_prior_z"].to(dev_b)).cpu()

            grid_cols = st.columns(4)
            for idx in range(16):
                with grid_cols[idx % 4]:
                    st.image(
                        tensor_to_hwc_uint8(decoded_prior[idx]),
                        caption=f"z #{idx+1}",
                        use_container_width=True,
                    )

    with right_vae:
        with st.container(border=True):
            card_title("2 · Latent-space morphing between two characters")
            col_a, col_b = st.columns(2)
            with col_a:
                char_a_idx = st.selectbox(
                    "Start character (α = 0.0):",
                    options=range(len(class_names_b)),
                    index=0,  # alif
                    format_func=lambda i: class_names_b[i],
                    key="vae_char_a",
                )
            with col_b:
                char_b_idx = st.selectbox(
                    "End character (α = 1.0):",
                    options=range(len(class_names_b)),
                    index=27,  # seen
                    format_func=lambda i: class_names_b[i],
                    key="vae_char_b",
                )

            alpha = st.slider(
                "Interpolation weight α:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

            # Pick first stored test sample for each chosen character
            sample_a = imgs_b[lbls_b == char_a_idx][0:1].to(dev_b)
            sample_b = imgs_b[lbls_b == char_b_idx][0:1].to(dev_b)

            with torch.no_grad():
                mu_a, _ = vae.encode(sample_a)
                mu_b, _ = vae.encode(sample_b)
                z_blend = (1.0 - alpha) * mu_a + alpha * mu_b
                morph_img = vae.decode(z_blend)[0].cpu()

                steps_10 = torch.linspace(0.0, 1.0, steps=10, device=dev_b).unsqueeze(1)
                strip_imgs = vae.decode((1.0 - steps_10) * mu_a + steps_10 * mu_b).cpu()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.image(
                    tensor_to_hwc_uint8(sample_a[0].cpu()),
                    caption=f"Start: {class_names_b[char_a_idx]}",
                    use_container_width=True,
                )
            with mc2:
                st.image(
                    tensor_to_hwc_uint8(morph_img),
                    caption=f"Interpolated (α = {alpha:.2f})",
                    use_container_width=True,
                )
            with mc3:
                st.image(
                    tensor_to_hwc_uint8(sample_b[0].cpu()),
                    caption=f"End: {class_names_b[char_b_idx]}",
                    use_container_width=True,
                )

            st.caption("Full 10-step latent trajectory")
            strip_cols = st.columns(10)
            for s_i in range(10):
                with strip_cols[s_i]:
                    st.image(
                        tensor_to_hwc_uint8(strip_imgs[s_i]),
                        caption=f"{s_i/9:.1f}",
                        use_container_width=True,
                    )

# ==============================================================================
# TAB 3: CLASS-CONDITIONAL DDPM (Part C)
# ==============================================================================
with tab_ddpm:
    section_banner(
        "ddpm", "🌫️",
        "Class-Conditional DDPM",
        "Start from pure Gaussian noise and denoise step by step into a chosen Urdu character. "
        "Pick a class, set a seed, and generate.",
        pills=["T = 1,000 steps", "8 samples per run", "Class-conditioned U-Net"],
    )

    unet, schedule, class_names_c, dev_c = cached_part_c()

    with st.container(border=True):
        dc1, dc2, dc3 = st.columns([2, 1, 1.3])
        with dc1:
            chosen_char = st.selectbox(
                "Urdu character class (40 UHaT classes):",
                options=range(len(class_names_c)),
                index=27,  # seen
                format_func=lambda i: f"{i:02d} — {class_names_c[i]}",
                key="ddpm_char_select",
            )
        with dc2:
            ddpm_seed = st.number_input("Sampling seed", min_value=0, max_value=9999, value=42, step=1)
        with dc3:
            st.write("")
            st.write("")
            run_ddpm_btn = st.button("Generate 8 new samples", type="primary", key="ddpm_gen_btn")

    if run_ddpm_btn or "ddpm_last_imgs" not in st.session_state:
        progress_bar = st.progress(0, text="Denoising 8 samples from pure Gaussian noise (T=1000)...")
        gen = torch.Generator(device="cpu").manual_seed(int(ddpm_seed))
        x_t = torch.randn((8, 1, 28, 28), generator=gen, dtype=torch.float32).to(dev_c)
        y_cond = torch.full((8,), int(chosen_char), device=dev_c, dtype=torch.long)

        with torch.no_grad():
            for step_i, t_idx in enumerate(reversed(range(schedule.num_timesteps))):
                x_t = schedule.p_sample_step(unet, x_t, t_idx, y_cond)
                if (step_i + 1) % 100 == 0 or t_idx == 0:
                    progress_bar.progress(
                        (step_i + 1) / schedule.num_timesteps,
                        text=f"Reverse diffusion step {step_i + 1}/{schedule.num_timesteps}...",
                    )
            samples_01 = ((x_t.clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()

        progress_bar.empty()
        st.session_state["ddpm_last_imgs"] = samples_01
        st.session_state["ddpm_last_label"] = class_names_c[int(chosen_char)]

    with st.container(border=True):
        card_title(f"Generated samples · character: {st.session_state['ddpm_last_label']}")
        cols_8 = st.columns(8)
        for i in range(8):
            with cols_8[i]:
                st.image(
                    tensor_to_hwc_uint8(st.session_state["ddpm_last_imgs"][i]),
                    caption=f"Sample #{i+1}",
                    use_container_width=True,
                )

# ==============================================================================
# TAB 4: BIOMEDICAL RAG (Part D)
# ==============================================================================
with tab_rag:
    section_banner(
        "rag", "🧪",
        "Biomedical Research QA",
        "Ask a biomedical question. A dense retriever finds the most relevant PubMed passages "
        "and a small LLM writes a cited answer.",
        pills=["40,181 abstracts", "79,332 chunks", "Qwen2.5-1.5B-Instruct"],
    )
    st.info(MEDICAL_NOTICE, icon="⚕️")

    example_questions = [
        "Which hormone abnormalities are characteristic to Pendred syndrome?",
        "Is capmatinib effective for glioblastoma?",
        "What is the mechanism of action of voxelotor in sickle cell disease?",
        "Which gene mutations cause Dravet syndrome?",
    ]

    with st.container(border=True):
        selected_example = st.selectbox(
            "Choose a sample biomedical question or type your own below:",
            options=["(Custom question)"] + example_questions,
        )
        default_q = "" if selected_example == "(Custom question)" else selected_example
        user_question = st.text_input(
            "Biomedical research question:",
            value=default_q,
            placeholder="Enter a biomedical research question...",
        )
        run_rag = st.button("Retrieve & generate answer", type="primary", key="rag_run_btn")

    if run_rag:
        if not user_question.strip():
            st.warning("Please enter a biomedical question first.")
        else:
            retriever = cached_part_d_retriever()
            gen_tok, gen_model, gen_dev = cached_part_d_generator()

            with st.spinner("Searching 79,332 PubMed chunks & generating cited answer..."):
                _, top5_chunks = retriever.retrieve_single(
                    user_question.strip(), top_k_abstracts=10, top_k_chunks=5
                )
                answer = generate_answer(
                    question=user_question.strip(),
                    top_chunks=top5_chunks,
                    tokenizer=gen_tok,
                    model=gen_model,
                    device=gen_dev,
                    max_new_tokens=256,
                )

            st.markdown("### Generated answer")
            st.success(answer)
            st.warning(f"**Notice:** {MEDICAL_NOTICE}")
            st.markdown("### Cited PubMed passages ([1]–[5])")
            for ch in top5_chunks:
                with st.expander(
                    f"[{ch['rank']}] PubMed Abstract ID: {ch['abstract_id']}  |  Cosine Similarity: {ch['score']:.4f}",
                    expanded=True,
                ):
                    st.write(ch["text"])

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    '<div class="footer">Built with <b>PyTorch</b> · <b>Streamlit</b> · <b>Hugging Face</b>'
    " &nbsp;|&nbsp; MAE · VAE · DDPM · RAG</div>",
    unsafe_allow_html=True,
)