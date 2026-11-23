import torch
import torch.nn as nn
from starter.eurosat import patchify, unpatchify, PATCH, CLASS_NAMES


class MaskedAutoencoder(nn.Module):
    """Asymmetric Vision Transformer Masked Autoencoder (75% patch masking)."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = PATCH,
        in_chans: int = 3,
        enc_dim: int = 192,
        enc_depth: int = 6,
        enc_heads: int = 3,
        enc_mlp_dim: int = 768,
        dec_dim: int = 128,
        dec_depth: int = 2,
        dec_heads: int = 4,
        dec_mlp_dim: int = 512,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) ** 2  # 64
        self.patch_dim = patch_size * patch_size * in_chans  # 192
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.mask_ratio = mask_ratio

        # Encoder components
        self.patch_embed = nn.Linear(self.patch_dim, enc_dim)
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, enc_dim))
        self.encoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=enc_dim,
                    nhead=enc_heads,
                    dim_feedforward=enc_mlp_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(enc_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(enc_dim)

        # Decoder components
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dec_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dec_dim,
                    nhead=dec_heads,
                    dim_feedforward=dec_mlp_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(dec_depth)
            ]
        )
        self.decoder_pred = nn.Linear(dec_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Per-image random masking: keeps 25% (16 patches), masks 75% (48 patches)."""
        N, L, D = x.shape
        len_keep = int(round(L * (1.0 - mask_ratio)))  # 16

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # True = masked (48 patches), False = visible (16 patches)
        mask = torch.ones((N, L), device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return x_masked, mask, ids_restore

    def forward_encoder(self, patches: torch.Tensor, mask_ratio: float = 0.75):
        x = self.patch_embed(patches) + self.enc_pos_embed
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            N, L, _ = x.shape
            mask = torch.zeros((N, L), device=x.device, dtype=torch.bool)
            ids_restore = None

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x_vis: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x_vis)
        N, L_vis, D_dec = x.shape
        L = ids_restore.shape[1]

        mask_tokens = self.mask_token.expand(N, L - L_vis, D_dec)
        x_shuffled = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        )
        x_full = x_full + self.dec_pos_embed

        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        pred = self.decoder_pred(x_full)
        return pred

    @staticmethod
    def compute_loss(
        target_patches: torch.Tensor, pred_patches: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error computed strictly on the masked patches."""
        sq_err = (pred_patches - target_patches) ** 2
        return sq_err[mask].mean()

    def forward(self, images: torch.Tensor, mask_ratio: float | None = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        target = patchify(images, p=self.patch_size)
        latent, mask, ids_restore = self.forward_encoder(target, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.compute_loss(target, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct_images(self, images: torch.Tensor, mask_ratio: float = 0.75):
        """Returns (masked_input_imgs, reconstructed_imgs, original_imgs) in [0, 1]."""
        was_training = self.training
        self.eval()
        target = patchify(images, p=self.patch_size)
        _, pred, mask = self.forward(images, mask_ratio=mask_ratio)

        # Masked input: grey (0.5) on masked patches, original pixels on visible patches
        masked_patches = target.clone()
        masked_patches[mask] = 0.5
        masked_imgs = unpatchify(masked_patches, p=self.patch_size, C=self.in_chans)

        # Reconstruction: predicted pixels on masked patches, original on visible patches
        recon_patches = target.clone()
        recon_patches[mask] = pred.clamp(0.0, 1.0)[mask]
        recon_imgs = unpatchify(recon_patches, p=self.patch_size, C=self.in_chans)

        if was_training:
            self.train()
        return masked_imgs, recon_imgs, images

    @torch.no_grad()
    def encode_all_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Feeds all 64 patches (no masking) and averages encoder outputs -> (N, 192)."""
        patches = patchify(images, p=self.patch_size)
        latent, _, _ = self.forward_encoder(patches, mask_ratio=0.0)
        return latent.mean(dim=1)


class MAEClassifier(nn.Module):
    """Frozen MAE encoder + linear classification head nn.Linear(192, 10)."""

    def __init__(self, mae: MaskedAutoencoder, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.mae = mae
        for param in self.mae.parameters():
            param.requires_grad = False
        self.mae.eval()
        self.head = nn.Linear(mae.enc_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep pretrained MAE encoder strictly in eval mode
        self.mae.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mae.encode_all_patches(images)
        return self.head(features)


def verify_mae_correctness(model: MaskedAutoencoder, sample_images: torch.Tensor) -> str:
    """Section 3.6 correctness check: 48/64 patches masked & loss strictly on masked patches."""
    model.eval()
    with torch.no_grad():
        target = patchify(sample_images, p=model.patch_size)
        loss, pred, mask = model(sample_images, mask_ratio=0.75)

        masked_counts = mask.sum(dim=1)
        assert mask.shape[1] == 64, f"Expected 64 patches, got {mask.shape[1]}"
        assert torch.all(masked_counts == 48), (
            f"Expected exactly 48 masked patches per image, got {masked_counts.tolist()}"
        )

        # Perturb visible patches -> loss must remain identical
        pred_vis_perturbed = pred.clone()
        pred_vis_perturbed[~mask] += 100.0
        loss_vis_perturbed = model.compute_loss(target, pred_vis_perturbed, mask)
        vis_diff = abs(loss_vis_perturbed.item() - loss.item())
        assert vis_diff < 1e-7, f"Visible patches leaked into loss! diff={vis_diff}"

        # Perturb masked patches -> loss must change
        pred_mask_perturbed = pred.clone()
        pred_mask_perturbed[mask] += 1.0
        loss_mask_perturbed = model.compute_loss(target, pred_mask_perturbed, mask)
        assert abs(loss_mask_perturbed.item() - loss.item()) > 0.1

    report = (
        f"[MAE Correctness Check PASSED] Batch={sample_images.shape[0]} | "
        f"Total patches=64 | Masked patches per image={int(masked_counts[0].item())}/64 (75.0%) | "
        f"Loss delta when perturbing visible patches={vis_diff:.2e} (strictly masked-only loss verified)."
    )
    print(report)
    return report
