import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.uhat import load_uhat_characters
from part_b_vae.model import ConvVAE


def load_uhat_cached(uhat_root: str, cache_path: str = "/vol/data/uhat_tensors.pt"):
    """Loads UHaT via starter.uhat.load_uhat_characters once and caches tensors on Modal Volume."""
    if os.path.exists(cache_path):
        print(f"Loading cached UHaT tensors from {cache_path}...", flush=True)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return (
            data["c_tr"],
            data["l_tr"],
            data["c_te"],
            data["l_te"],
            data["class_names"],
        )

    print("Reading UHaT image files via starter.uhat.load_uhat_characters (one-time cache build)...", flush=True)
    c_tr, l_tr, class_names = load_uhat_characters(root=uhat_root, split="train")
    c_te, l_te, _ = load_uhat_characters(
        root=uhat_root, split="test", class_names=class_names
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "c_tr": c_tr,
            "l_tr": l_tr,
            "c_te": c_te,
            "l_te": l_te,
            "class_names": class_names,
        },
        cache_path,
    )
    print(f"Cached UHaT tensors to {cache_path}.", flush=True)
    return c_tr, l_tr, c_te, l_te, class_names


@torch.no_grad()
def _evaluate_epoch(model: ConvVAE, loader: DataLoader, device: torch.device):
    model.eval()
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0
    for (imgs,) in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits, mu, logvar, _ = model(imgs)
        neg_elbo, recon, kl = ConvVAE.elbo_loss(logits, imgs, mu, logvar)
        bs = imgs.size(0)
        total_elbo += neg_elbo.item() * bs
        total_recon += recon.item() * bs
        total_kl += kl.item() * bs
        total_samples += bs
    return (
        total_elbo / total_samples,
        total_recon / total_samples,
        total_kl / total_samples,
    )


def train_vae(
    uhat_root: str,
    ckpt_path: str,
    loss_plot_path: str,
    history_csv_path: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training ConvVAE on device: {gpu_name}", flush=True)

    c_tr, _, c_te, _, class_names = load_uhat_cached(uhat_root)

    train_loader = DataLoader(
        TensorDataset(c_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(c_te), batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = ConvVAE(latent_dim=latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE total parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_elbo_sum = 0.0
        tr_recon_sum = 0.0
        tr_kl_sum = 0.0
        tr_count = 0

        for (imgs,) in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar, _ = model(imgs)
            neg_elbo, recon, kl = ConvVAE.elbo_loss(logits, imgs, mu, logvar)
            neg_elbo.backward()
            optimizer.step()

            bs = imgs.size(0)
            tr_elbo_sum += neg_elbo.item() * bs
            tr_recon_sum += recon.item() * bs
            tr_kl_sum += kl.item() * bs
            tr_count += bs

        tr_elbo = tr_elbo_sum / tr_count
        tr_recon = tr_recon_sum / tr_count
        tr_kl = tr_kl_sum / tr_count
        te_elbo, te_recon, te_kl = _evaluate_epoch(model, test_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_neg_elbo": tr_elbo,
                "train_recon": tr_recon,
                "train_kl": tr_kl,
                "test_neg_elbo": te_elbo,
                "test_recon": te_recon,
                "test_kl": te_kl,
            }
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[VAE] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train -ELBO: {tr_elbo:.2f} (Recon: {tr_recon:.2f}, KL: {tr_kl:.2f}) | "
                f"Test -ELBO: {te_elbo:.2f} (Recon: {te_recon:.2f}, KL: {te_kl:.2f})",
                flush=True,
            )

    elapsed_sec = time.perf_counter() - t0
    print(f"VAE training completed in {elapsed_sec:.2f}s on {gpu_name}.", flush=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"latent_dim": latent_dim},
            "class_names": class_names,
            "total_params": total_params,
            "final_train_loss": history[-1]["train_neg_elbo"],
            "final_test_loss": history[-1]["test_neg_elbo"],
            "train_time_sec": elapsed_sec,
            "gpu_name": gpu_name,
            "history": history,
        },
        ckpt_path,
    )

    with open(history_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_neg_elbo,train_recon_nats,train_kl_nats,"
            "test_neg_elbo,test_recon_nats,test_kl_nats\n"
        )
        for row in history:
            f.write(
                f"{row['epoch']},{row['train_neg_elbo']:.6f},{row['train_recon']:.6f},"
                f"{row['train_kl']:.6f},{row['test_neg_elbo']:.6f},"
                f"{row['test_recon']:.6f},{row['test_kl']:.6f}\n"
            )

    ep_x = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(ep_x, [r["train_neg_elbo"] for r in history], label="Train", lw=2)
    axes[0].plot(ep_x, [r["test_neg_elbo"] for r in history], label="Test", lw=2, ls="--")
    axes[0].set_title("Negative ELBO (Total Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Nats / image")
    axes[0].grid(True, ls="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(ep_x, [r["train_recon"] for r in history], label="Train", lw=2)
    axes[1].plot(ep_x, [r["test_recon"] for r in history], label="Test", lw=2, ls="--")
    axes[1].set_title("Reconstruction Term (-log p(x|z))")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Nats / image")
    axes[1].grid(True, ls="--", alpha=0.4)
    axes[1].legend()

    axes[2].plot(ep_x, [r["train_kl"] for r in history], label="Train", lw=2)
    axes[2].plot(ep_x, [r["test_kl"] for r in history], label="Test", lw=2, ls="--")
    axes[2].set_title("KL Divergence D_KL(q(z|x) || N(0, I))")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Nats / image")
    axes[2].grid(True, ls="--", alpha=0.4)
    axes[2].legend()

    fig.suptitle("Part B: ConvVAE Training & Test Loss Terms on UHaT", fontsize=12)
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    return model