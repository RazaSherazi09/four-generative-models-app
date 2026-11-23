import time
import torch
import torch.nn as nn

from starter.eurosat import load_eurosat_split, few_label_indices, CLASS_NAMES
from part_a_mae.model import MaskedAutoencoder, MAEClassifier


def train_few_label_classifier(
    mae_ckpt_path: str,
    cls_ckpt_path: str,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    ckpt = torch.load(mae_ckpt_path, map_location=device, weights_only=True)
    mae = MaskedAutoencoder(**ckpt["config"]).to(device)
    mae.load_state_dict(ckpt["model_state_dict"])

    classifier = MAEClassifier(mae=mae, num_classes=len(CLASS_NAMES)).to(device)
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Few-label linear classifier trainable parameters: {trainable_params:,}")

    x_tr, y_tr = load_eurosat_split("train")
    idx_160 = few_label_indices(y_tr, per_class=16, seed=42)
    x_few = x_tr[idx_160].to(device)
    y_few = y_tr[idx_160].to(device)
    assert len(x_few) == 160, f"Expected 160 labelled images, got {len(x_few)}"

    optimizer = torch.optim.AdamW(classifier.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Precompute frozen encoder features once for fast, exact 200-epoch full-batch training
    with torch.no_grad():
        features_few = classifier.mae.encode_all_patches(x_few)

    t0 = time.perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier.head(features_few)
        loss = criterion(logits, y_few)
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(dim=1) == y_few).float().mean().item()
        history.append((epoch, loss.item(), train_acc))
        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(
                f"[MAE Linear Probe] Epoch {epoch:03d}/{epochs:03d} | "
                f"CE Loss: {loss.item():.4f} | 160-Sample Train Acc: {train_acc * 100:.2f}%"
            )

    elapsed_sec = time.perf_counter() - t0
    torch.save(
        {
            "head_state_dict": classifier.head.state_dict(),
            "class_names": CLASS_NAMES,
            "trainable_params": trainable_params,
            "train_time_sec": elapsed_sec,
            "gpu_name": gpu_name,
            "final_train_loss": history[-1][1],
            "final_train_acc": history[-1][2],
        },
        cls_ckpt_path,
    )
    return classifier
