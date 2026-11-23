import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for 28x28 grayscale Urdu characters."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder q_phi(z | x): 28x28 -> 14x14 -> 7x7 -> flatten(64*7*7) -> mu, logvar
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder p_theta(x | z): Linear(16, 64*7*7) -> reshape -> Deconv -> ReLU -> Deconv logits
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, where sigma = exp(0.5 * logvar) and eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 64, 7, 7)
        logits = self.decoder_deconv(h)
        return logits

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar, z

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Closed-form KL(q(z|x) || N(0, I)) per sample in nats:
        -0.5 * sum_{j=1}^d (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2).
        """
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    @classmethod
    def elbo_loss(
        cls,
        logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (negative_elbo, recon_bce_nats, kl_nats) averaged over the batch."""
        recon_per_sample = F.binary_cross_entropy_with_logits(
            logits, x, reduction="none"
        ).sum(dim=(1, 2, 3))
        kl_per_sample = cls.kl_divergence(mu, logvar)

        recon_loss = recon_per_sample.mean()
        kl_loss = kl_per_sample.mean()
        neg_elbo = recon_loss + kl_loss
        return neg_elbo, recon_loss, kl_loss


def verify_vae_correctness(model: ConvVAE, sample_images: torch.Tensor) -> str:
    """Section 3.6 correctness check: KL >= 0 always, and KL == 0 when mu=0, logvar=0."""
    model.eval()
    with torch.no_grad():
        _, mu, logvar, _ = model(sample_images)
        kl_vals = ConvVAE.kl_divergence(mu, logvar)
        min_kl = float(kl_vals.min().item())
        assert min_kl >= -1e-7, f"KL divergence became negative: {min_kl}"

        zero_mu = torch.zeros(8, model.latent_dim, device=sample_images.device)
        zero_logvar = torch.zeros(8, model.latent_dim, device=sample_images.device)
        kl_zero = float(ConvVAE.kl_divergence(zero_mu, zero_logvar).abs().max().item())
        assert kl_zero < 1e-7, f"KL divergence at mu=0, logvar=0 was non-zero: {kl_zero}"

    report = (
        f"[VAE Correctness Check PASSED] Batch={sample_images.shape[0]} | "
        f"Min batch KL={min_kl:.6f} nats (>= 0 verified) | "
        f"KL at (mu=0, log_sigma^2=0)={kl_zero:.2e} nats (exact 0 verified)."
    )
    print(report)
    return report