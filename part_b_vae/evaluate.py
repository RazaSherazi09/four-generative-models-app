import json
import torch
from torch.utils.data import TensorDataset, DataLoader

from starter.viz import save_grid
from part_b_vae.model import ConvVAE, verify_vae_correctness
from part_b_vae.train import load_uhat_cached


@torch.no_grad()
def evaluate_part_b(
    uhat_root: str,
    ckpt_path: str,
    recon_plot_path: str,
    samples_plot_path: str,
    interp_plot_path: str,
    summary_json_path: str,
    correctness_txt_path: str,
    app_samples_path: str,
    batch_size: int = 128,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ConvVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    c_tr, _, c_te, l_te, class_names = load_uhat_cached(uhat_root)

    # Section 3.6 correctness check
    correctness_report = verify_vae_correctness(model, c_te[:32].to(device))
    with open(correctness_txt_path, "w", encoding="utf-8") as f:
        f.write(correctness_report + "\n")

    # Section 3.2: Mean negative ELBO on full test set split into reconstruction & KL (nats/image)
    test_loader = DataLoader(TensorDataset(c_te), batch_size=batch_size, shuffle=False)
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0
    for (imgs,) in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits, mu, logvar, _ = model(imgs)
        neg_elbo, recon, kl = ConvVAE.elbo_loss(logits, imgs, mu, logvar)
        bs = imgs.size(0)
        total_elbo += neg_elbo.item() * bs
        total_recon += recon.item() * bs
        total_kl += kl.item() * bs
        total_samples += bs

    mean_neg_elbo = total_elbo / total_samples
    mean_recon = total_recon / total_samples
    mean_kl = total_kl / total_samples

    print(
        f"[Part B Evaluation] Test Negative ELBO: {mean_neg_elbo:.4f} nats/image "
        f"(Recon: {mean_recon:.4f} + KL: {mean_kl:.4f})",
        flush=True,
    )

    # Figure 2b (Task B.4a): 16 test images and their reconstructions (4x8 interleaved rows)
    gen = torch.Generator().manual_seed(seed)
    pick16 = torch.randperm(len(c_te), generator=gen)[:16]
    orig16 = c_te[pick16].to(device)
    lbl16 = [class_names[int(l_te[i])] for i in pick16]
    logits16, _, _, _ = model(orig16)
    recon16 = torch.sigmoid(logits16)

    paired_imgs = torch.cat(
        [orig16[:8], recon16[:8], orig16[8:16], recon16[8:16]], dim=0
    )
    paired_titles = (
        [f"Orig: {n}" for n in lbl16[:8]]
        + [f"Recon: {n}" for n in lbl16[:8]]
        + [f"Orig: {n}" for n in lbl16[8:16]]
        + [f"Recon: {n}" for n in lbl16[8:16]]
    )
    save_grid(paired_imgs, recon_plot_path, nrow=8, titles=paired_titles)

    # Figure 2c (Task B.4b): 64 new characters decoded from z ~ N(0, I)
    z_prior = torch.randn(64, model.latent_dim, generator=gen, device="cpu").to(device)
    prior_samples = model.decode(z_prior)
    save_grid(prior_samples, samples_plot_path, nrow=8)

    # Figure 2d (Task B.4c): Linear interpolation between mu of two different characters (10 points)
    idx_a = int(torch.nonzero(l_te == 0, as_tuple=True)[0][0])   # 'alif'
    idx_b = int(torch.nonzero(l_te == 27, as_tuple=True)[0][0])  # 'seen'
    img_a = c_te[idx_a : idx_a + 1].to(device)
    img_b = c_te[idx_b : idx_b + 1].to(device)
    name_a = class_names[int(l_te[idx_a])]
    name_b = class_names[int(l_te[idx_b])]

    mu_a, _ = model.encode(img_a)
    mu_b, _ = model.encode(img_b)
    alphas = torch.linspace(0.0, 1.0, steps=10, device=device).unsqueeze(1)
    z_interp = (1.0 - alphas) * mu_a + alphas * mu_b
    interp_imgs = model.decode(z_interp)
    interp_titles = [
        f"{name_a} (0.0)" if i == 0 else (f"{name_b} (1.0)" if i == 9 else f"t={i/9:.2f}")
        for i in range(10)
    ]
    save_grid(interp_imgs, interp_plot_path, nrow=10, titles=interp_titles)

    # Save 2 test images per character (80 tiny 28x28 tensors) for the Streamlit interpolation tab
    app_idx = []
    for c in range(len(class_names)):
        c_indices = torch.nonzero(l_te == c, as_tuple=True)[0][:2]
        app_idx.append(c_indices)
    app_idx = torch.cat(app_idx, dim=0)
    torch.save(
        {
            "images": c_te[app_idx].cpu(),
            "labels": l_te[app_idx].cpu(),
            "class_names": class_names,
        },
        app_samples_path,
    )

    summary = {
        "dataset": {
            "num_classes": len(class_names),
            "train_images": int(len(c_tr)),
            "test_images": int(len(c_te)),
        },
        "training": {
            "total_parameters": int(ckpt["total_params"]),
            "final_train_neg_elbo": float(ckpt["final_train_loss"]),
            "train_time_sec": float(ckpt["train_time_sec"]),
            "gpu_name": ckpt["gpu_name"],
        },
        "evaluation": {
            "test_neg_elbo_nats": round(mean_neg_elbo, 4),
            "test_recon_nats": round(mean_recon, 4),
            "test_kl_nats": round(mean_kl, 4),
            "interpolation_pair": [name_a, name_b],
            "correctness_check": correctness_report,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary