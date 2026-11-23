from diffusers import UNet2DModel


def build_unet(num_classes: int = 40) -> UNet2DModel:
    """Exact UNet2DModel constructor specified in Section 2.4 (random initialization)."""
    unet = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=num_classes,
    )
    return unet