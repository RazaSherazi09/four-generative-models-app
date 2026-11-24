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


st.title("Four Generative Models, One App")
st.caption(
    f"Masked Autoencoder (EuroSAT) · Variational Autoencoder (UHaT) · "
    f"Class-Conditional DDPM (UHaT) · Biomedical RAG (BioASQ) | "
    f"Weights: `{HF_REPO_ID}`"
)

tab_mae, tab_vae, tab_ddpm, tab_rag = st.tabs(
    [
        "1. MAE (Satellite Land Cover)",
        "2. VAE (Urdu Handwriting)",
        "3. Diffusion (Conditional Urdu)",
        "4. RAG (Biomedical QA)",
    ]
)

# ==============================================================================
# TAB 1: MASKED AUTOENCODER (Part A)
# ==============================================================================
with tab_mae:
    st.subheader("Part A: Masked Autoencoder (75% Masking) & 16-Shot Linear Classifier")
    mae, classifier, mae_samples, dev_a = cached_part_a()
    class_names_a = mae_samples["class_names"]
    imgs_a = mae_samples["images"]  # (30, 3, 64, 64)
    lbls_a = mae_samples["labels"]

    options_a = [
        f"Tile #{i:02d} — True Class: {class_names_a[int(lbls_a[i])]}"
        for i in range(len(imgs_a))
    ]
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

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    with c1:
        st.image(
            tensor_to_hwc_uint8(masked_img[0]),
            caption="Masked Input (48/64 patches hidden)",
            use_container_width=True,
        )
    with c2:
        st.image(
            tensor_to_hwc_uint8(recon_img[0]),
            caption="MAE Reconstruction",
            use_container_width=True,
        )
    with c3:
        st.image(
            tensor_to_hwc_uint8(orig_img[0]),
            caption=f"Original ({true_label})",
            use_container_width=True,
        )
    with c4:
        st.markdown("**Few-Label Classifier Prediction (Unmasked)**")
        if pred_label == true_label:
            st.success(f"Predicted: **{pred_label}** ({pred_conf:.1f}%)")
        else:
            st.warning(f"Predicted: **{pred_label}** ({pred_conf:.1f}%) | True: **{true_label}**")

        top3_idx = np.argsort(probs)[::-1][:3]
        for rank_i in top3_idx:
            st.progress(float(probs[rank_i]), text=f"{class_names_a[rank_i]}: {probs[rank_i]*100:.1f}%")

# ==============================================================================
# TAB 2: VARIATIONAL AUTOENCODER (Part B)
# ==============================================================================
with tab_vae:
    st.subheader("Part B: Convolutional Variational Autoencoder (Latent Dim = 16)")
    vae, vae_samples, dev_b = cached_part_b()
    class_names_b = vae_samples["class_names"]
    imgs_b = vae_samples["images"]  # (80, 1, 28, 28)
    lbls_b = vae_samples["labels"]

    left_vae, right_vae = st.columns([1, 1.2])

    with left_vae:
        st.markdown("**1. Prior Sampling: Decode $z \\sim \\mathcal{N}(0, I)$**")
        if st.button("Generate 16 New Random Characters", key="vae_sample_btn"):
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
        st.markdown("**2. Latent Space Morphing Between Two Characters**")
        col_a, col_b = st.columns(2)
        with col_a:
            char_a_idx = st.selectbox(
                "Start Character (α = 0.0):",
                options=range(len(class_names_b)),
                index=0,  # alif
                format_func=lambda i: class_names_b[i],
                key="vae_char_a",
            )
        with col_b:
            char_b_idx = st.selectbox(
                "End Character (α = 1.0):",
                options=range(len(class_names_b)),
                index=27,  # seen
                format_func=lambda i: class_names_b[i],
                key="vae_char_b",
            )

        alpha = st.slider(
            "Interpolation Weight α:",
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

        st.caption("Full 10-step latent trajectory:")
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
    st.subheader("Part C: Class-Conditional DDPM (1,000-Step Reverse Sampling)")
    unet, schedule, class_names_c, dev_c = cached_part_c()

    dc1, dc2, dc3 = st.columns([2, 1, 1.5])
    with dc1:
        chosen_char = st.selectbox(
            "Select Urdu Character Class (40 UHaT classes):",
            options=range(len(class_names_c)),
            index=27,  # seen
            format_func=lambda i: f"{i:02d} — {class_names_c[i]}",
            key="ddpm_char_select",
        )
    with dc2:
        ddpm_seed = st.number_input("Sampling Seed", min_value=0, max_value=9999, value=42, step=1)
    with dc3:
        st.write("")
        st.write("")
        run_ddpm_btn = st.button("Generate 8 New Samples", type="primary", key="ddpm_gen_btn")

    cache_key = f"ddpm_imgs_{chosen_char}_{ddpm_seed}"
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

    st.markdown(f"**Generated Samples for Character: `{st.session_state['ddpm_last_label']}`**")
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
    st.subheader("Part D: Biomedical Research QA over 40,181 PubMed Abstracts (BioASQ)")
    st.info(f"⚕️ **{MEDICAL_NOTICE}**")

    example_questions = [
        "Which hormone abnormalities are characteristic to Pendred syndrome?",
        "Is capmatinib effective for glioblastoma?",
        "What is the mechanism of action of voxelotor in sickle cell disease?",
        "Which gene mutations cause Dravet syndrome?",
    ]
    selected_example = st.selectbox(
        "Choose a sample biomedical question or type your own below:",
        options=["(Custom question)"] + example_questions,
    )
    default_q = "" if selected_example == "(Custom question)" else selected_example
    user_question = st.text_input(
        "Biomedical Research Question:",
        value=default_q,
        placeholder="Enter a biomedical research question...",
    )

    if st.button("Retrieve & Generate Answer", type="primary", key="rag_run_btn"):
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

            st.markdown("### Generated Answer")
            st.success(answer)
            st.warning(f"**Notice:** {MEDICAL_NOTICE}")

            st.markdown("### Cited PubMed Passages ([1]–[5])")
            for ch in top5_chunks:
                with st.expander(
                    f"[{ch['rank']}] PubMed Abstract ID: {ch['abstract_id']}  |  Cosine Similarity: {ch['score']:.4f}",
                    expanded=True,
                ):
                    st.write(ch["text"])