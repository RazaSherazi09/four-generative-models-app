import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.eurosat import load_eurosat_split, CLASS_NAMES
from starter.viz import save_grid
from part_a_mae.model import MaskedAutoencoder


def pretrain_mae(
    ckpt_path: str,
    loss_plot_path: str,
    recon_plot_path: str,
    history_csv_path: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Pretraining MAE on device: {gpu_name}")

    x_tr, _ = load_eurosat_split("train")
    x_te, y_te = load_eurosat_split("test")

    dataset = TensorDataset(x_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = MaskedAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MAE total parameters (encoder + decoder): {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for (batch_imgs,) in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(batch_imgs, mask_ratio=0.75)
            loss.backward()
            optimizer.step()

            bs = batch_imgs.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

        epoch_loss = running_loss / total_samples
        losses.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"[MAE Pretrain] Epoch {epoch:03d}/{epochs:03d} | Masked MSE Loss: {epoch_loss:.6f}")

    elapsed_sec = time.perf_counter() - t0
    print(f"MAE pretraining completed in {elapsed_sec:.2f}s on {gpu_name}.")

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "img_size": 64,
                "patch_size": 8,
                "in_chans": 3,
                "enc_dim": 192,
                "enc_depth": 6,
                "enc_heads": 3,
                "enc_mlp_dim": 768,
                "dec_dim": 128,
                "dec_depth": 2,
                "dec_heads": 4,
                "dec_mlp_dim": 512,
                "mask_ratio": 0.75,
            },
            "total_params": total_params,
            "final_train_loss": losses[-1],
            "train_time_sec": elapsed_sec,
            "gpu_name": gpu_name,
            "loss_history": losses,
        },
        ckpt_path,
    )

    # Save epoch loss CSV
    with open(history_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,masked_mse_loss\n")
        for ep, val in enumerate(losses, start=1):
            f.write(f"{ep},{val:.8f}\n")

    # Figure 1a: Pretraining loss curve
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, epochs + 1), losses, color="#1f77b4", linewidth=2)
    ax.set_title("Part A: MAE Pretraining Loss on EuroSAT (75% Masking)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked Patch MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    # Figure 1b (Task A.2): 8 test images -> 3 rows: masked input, reconstruction, original
    gen = torch.Generator().manual_seed(42)
    sample_idx = torch.randperm(len(x_te), generator=gen)[:8]
    sample_imgs = x_te[sample_idx].to(device)
    sample_labels = [CLASS_NAMES[int(y_te[i])] for i in sample_idx]

    masked_imgs, recon_imgs, orig_imgs = model.reconstruct_images(sample_imgs, mask_ratio=0.75)
    grid_imgs = torch.cat([masked_imgs, recon_imgs, orig_imgs], dim=0)
    titles = (
        [f"Masked: {lbl}" for lbl in sample_labels]
        + [f"Recon: {lbl}" for lbl in sample_labels]
        + [f"Orig: {lbl}" for lbl in sample_labels]
    )
    save_grid(grid_imgs, recon_plot_path, nrow=8, titles=titles)

    return model, {
        "total_params": total_params,
        "final_train_loss": losses[-1],
        "train_time_sec": elapsed_sec,
        "gpu_name": gpu_name,
    }
