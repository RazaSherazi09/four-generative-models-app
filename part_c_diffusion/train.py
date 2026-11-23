import time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.viz import save_grid
from part_b_vae.train import load_uhat_cached
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule


def train_ddpm(
    uhat_root: str,
    ckpt_path: str,
    loss_plot_path: str,
    forward_plot_path: str,
    history_csv_path: str,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-4,
    num_timesteps: int = 1000,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training Class-Conditional DDPM on device: {gpu_name}", flush=True)

    c_tr, l_tr, _, _, class_names = load_uhat_cached(uhat_root)
    # Rescale pixels from [0, 1] to [-1, 1] as required by Section 2.4
    x0_tr = c_tr * 2.0 - 1.0

    train_loader = DataLoader(
        TensorDataset(x0_tr, l_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    unet = build_unet(num_classes=len(class_names)).to(device)
    schedule = DDPMSchedule(num_timesteps=num_timesteps).to(device)

    total_params = sum(p.numel() for p in unet.parameters())
    print(f"DDPM UNet2DModel total parameters: {total_params:,}", flush=True)

    # Task C.2: Forward process visualization on one training image at t in {0, 100, 250, 500, 750, 999}
    vis_steps = [0, 100, 250, 500, 750, 999]
    x_clean = x0_tr[0:1].to(device)
    lbl_clean = class_names[int(l_tr[0])]
    gen_vis = torch.Generator(device="cpu").manual_seed(seed)
    fixed_eps = torch.randn(x_clean.shape, generator=gen_vis).to(device)

    forward_imgs = []
    for t_step in vis_steps:
        if t_step == 0:
            xt = x_clean
        else:
            t_tensor = torch.tensor([t_step], device=device, dtype=torch.long)
            xt, _ = schedule.q_sample(x_clean, t_tensor, noise=fixed_eps)
        # Map [-1, 1] -> [0, 1] for visualization
        forward_imgs.append(((xt.clamp(-1.0, 1.0) + 1.0) * 0.5).cpu())

    save_grid(
        torch.cat(forward_imgs, dim=0),
        forward_plot_path,
        nrow=len(vis_steps),
        titles=[f"{lbl_clean} (t={t})" if t == 0 else f"t={t}" for t in vis_steps],
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    losses = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        unet.train()
        running_loss = 0.0
        total_samples = 0

        for x0, y in train_loader:
            x0 = x0.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bs = x0.size(0)

            t = torch.randint(0, num_timesteps, (bs,), device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t, _ = schedule.q_sample(x0, t, noise=eps)

            optimizer.zero_grad(set_to_none=True)
            eps_pred = unet(x_t, t, class_labels=y).sample
            loss = F.mse_loss(eps_pred, eps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bs
            total_samples += bs

        epoch_loss = running_loss / total_samples
        losses.append(epoch_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[DDPM] Epoch {epoch:02d}/{epochs:02d} | L_simple MSE Loss: {epoch_loss:.6f}",
                flush=True,
            )

    elapsed_sec = time.perf_counter() - t0
    print(f"DDPM training completed in {elapsed_sec:.2f}s on {gpu_name}.", flush=True)

    torch.save(
        {
            "model_state_dict": unet.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "num_timesteps": num_timesteps,
            "total_params": total_params,
            "final_train_loss": losses[-1],
            "train_time_sec": elapsed_sec,
            "gpu_name": gpu_name,
            "loss_history": losses,
        },
        ckpt_path,
    )

    with open(history_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,l_simple_mse_loss\n")
        for ep, val in enumerate(losses, start=1):
            f.write(f"{ep},{val:.8f}\n")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, epochs + 1), losses, color="#2ca02c", linewidth=2)
    ax.set_title("Part C: DDPM Training Loss (L_simple) on UHaT")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise Prediction MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    return unet, schedule