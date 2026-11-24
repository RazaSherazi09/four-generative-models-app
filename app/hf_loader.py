import os
import torch
from huggingface_hub import hf_hub_download

from part_a_mae.model import MaskedAutoencoder, MAEClassifier
from part_b_vae.model import ConvVAE
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule
from part_d_rag.index import load_retriever
from part_d_rag.retriever import DenseRetriever
from part_d_rag.generate import load_generator

HF_REPO_ID = "razaasherazi/four-generative-models-checkpoints"


def _resolve_artifact(hf_filename: str, vol_rel_path: str, vol_prefix: str = "/vol") -> str:
    """Use Modal Volume path if running inside Modal, otherwise download from Hugging Face Hub."""
    vol_path = os.path.join(vol_prefix, vol_rel_path)
    if os.path.exists(vol_path):
        return vol_path
    return hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename, repo_type="model")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_part_a_bundle():
    device = get_device()
    mae_path = _resolve_artifact(
        "part_a_mae/mae_pretrained.pt", "checkpoints/part_a_mae/mae_pretrained.pt"
    )
    cls_path = _resolve_artifact(
        "part_a_mae/mae_classifier.pt", "checkpoints/part_a_mae/mae_classifier.pt"
    )
    samples_path = _resolve_artifact(
        "part_a_mae/app_test_samples.pt", "checkpoints/part_a_mae/app_test_samples.pt"
    )

    mae_ckpt = torch.load(mae_path, map_location=device, weights_only=True)
    cls_ckpt = torch.load(cls_path, map_location=device, weights_only=True)
    samples_data = torch.load(samples_path, map_location="cpu", weights_only=True)

    mae = MaskedAutoencoder(**mae_ckpt["config"]).to(device)
    mae.load_state_dict(mae_ckpt["model_state_dict"])
    mae.eval()

    classifier = MAEClassifier(mae=mae, num_classes=len(cls_ckpt["class_names"])).to(device)
    classifier.head.load_state_dict(cls_ckpt["head_state_dict"])
    classifier.eval()

    return mae, classifier, samples_data, device


def load_part_b_bundle():
    device = get_device()
    vae_path = _resolve_artifact(
        "part_b_vae/vae_uhat.pt", "checkpoints/part_b_vae/vae_uhat.pt"
    )
    samples_path = _resolve_artifact(
        "part_b_vae/app_uhat_samples.pt", "checkpoints/part_b_vae/app_uhat_samples.pt"
    )

    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=True)
    samples_data = torch.load(samples_path, map_location="cpu", weights_only=True)

    vae = ConvVAE(**vae_ckpt["config"]).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    return vae, samples_data, device


def load_part_c_bundle():
    device = get_device()
    ddpm_path = _resolve_artifact(
        "part_c_diffusion/ddpm_unet_uhat.pt", "checkpoints/part_c_diffusion/ddpm_unet_uhat.pt"
    )
    ckpt = torch.load(ddpm_path, map_location=device, weights_only=True)
    class_names = ckpt["class_names"]

    unet = build_unet(num_classes=len(class_names)).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    schedule = DDPMSchedule(num_timesteps=ckpt["num_timesteps"]).to(device)
    return unet, schedule, class_names, device


def load_part_d_retriever():
    device = get_device()
    emb_path = _resolve_artifact(
        "part_d_rag/chunk_embeddings.pt", "indices/part_d_rag/chunk_embeddings.pt"
    )
    meta_path = _resolve_artifact(
        "part_d_rag/chunks_metadata.pt", "indices/part_d_rag/chunks_metadata.pt"
    )

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)

    ret_tok, ret_model, _ = load_retriever(device=device)
    retriever = DenseRetriever(
        embeddings=embeddings,
        abstract_ids=meta["abstract_ids"],
        texts=meta["texts"],
        tokenizer=ret_tok,
        model=ret_model,
        device=device,
    )
    return retriever


def load_part_d_generator():
    device = get_device()
    return load_generator(device=device)