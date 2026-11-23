import torch
import torch.nn as nn


class DDPMSchedule(nn.Module):
    """Linear beta noise schedule, forward process q(x_t|x_0), and Algorithm 2 sampling."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_coeff", (1.0 - alphas) / torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer("sigma", torch.sqrt(betas))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample_step(
        self,
        unet: nn.Module,
        x_t: torch.Tensor,
        t_idx: int,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step from t to t-1 (DDPM Algorithm 2)."""
        batch_size = x_t.size(0)
        t_tensor = torch.full(
            (batch_size,), t_idx, device=x_t.device, dtype=torch.long
        )
        eps_theta = unet(x_t, t_tensor, class_labels=class_labels).sample

        sqrt_recip_alpha = self.sqrt_recip_alphas[t_idx]
        coeff = self.posterior_coeff[t_idx]
        mean = sqrt_recip_alpha * (x_t - coeff * eps_theta)

        if t_idx > 0:
            z = torch.randn_like(x_t)
            sigma_t = self.sigma[t_idx]
            return mean + sigma_t * z
        else:
            # z = 0 at the final step
            return mean

    @torch.no_grad()
    def sample(
        self,
        unet: nn.Module,
        class_labels: torch.Tensor,
        capture_steps: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """DDPM Algorithm 2 sampling loop from x_T ~ N(0, I) down to x_0 clipped to [-1, 1]."""
        was_training = unet.training
        unet.eval()

        device = class_labels.device
        batch_size = class_labels.size(0)
        x_t = torch.randn(
            (batch_size, 1, 28, 28),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device)

        snapshots: dict[int, torch.Tensor] = {}
        if capture_steps and (self.num_timesteps - 1) in capture_steps:
            snapshots[self.num_timesteps - 1] = x_t.clamp(-1.0, 1.0).cpu()

        for t_idx in reversed(range(self.num_timesteps)):
            x_t = self.p_sample_step(unet, x_t, t_idx, class_labels)
            if capture_steps and t_idx in capture_steps and t_idx != (self.num_timesteps - 1):
                snapshots[t_idx] = x_t.clamp(-1.0, 1.0).cpu()

        x_0 = x_t.clamp(-1.0, 1.0)
        if capture_steps and 0 in capture_steps:
            snapshots[0] = x_0.cpu()

        if was_training:
            unet.train()
        return x_0, snapshots


def verify_ddpm_correctness(schedule: DDPMSchedule, sample_x0: torch.Tensor) -> str:
    """Section 3.6 correctness check: alpha_bar_T ~ 4e-5 and x_{t=999} ~ N(0, I)."""
    alpha_bar_T = float(schedule.alpha_bars[-1].item())
    assert 1e-5 < alpha_bar_T < 1e-4, f"Unexpected alpha_bar_T: {alpha_bar_T}"

    t_last = torch.full(
        (sample_x0.size(0),),
        schedule.num_timesteps - 1,
        device=sample_x0.device,
        dtype=torch.long,
    )
    x_999, _ = schedule.q_sample(sample_x0, t_last)
    mean_999 = float(x_999.mean().item())
    std_999 = float(x_999.std().item())
    assert abs(mean_999) < 0.1 and 0.9 < std_999 < 1.1, (
        f"Noised x_999 stats deviated from N(0,1): mean={mean_999:.4f}, std={std_999:.4f}"
    )

    report = (
        f"[DDPM Correctness Check PASSED] T={schedule.num_timesteps} | "
        f"alpha_bar_T={alpha_bar_T:.6e} (approx 4e-5 verified) | "
        f"Noised x_(t=999) stats: mean={mean_999:+.4f}, std={std_999:.4f} (pure Gaussian noise verified)."
    )
    print(report, flush=True)
    return report