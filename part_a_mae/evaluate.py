import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.eurosat import load_eurosat_split, few_label_indices, CLASS_NAMES
from part_a_mae.model import MaskedAutoencoder, MAEClassifier, verify_mae_correctness


@torch.no_grad()
def evaluate_part_a(
    mae_ckpt_path: str,
    cls_ckpt_path: str,
    cm_plot_path: str,
    cm_csv_path: str,
    summary_json_path: str,
    correctness_txt_path: str,
    app_samples_path: str,
    batch_size: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_ckpt = torch.load(mae_ckpt_path, map_location=device, weights_only=True)
    cls_ckpt = torch.load(cls_ckpt_path, map_location=device, weights_only=True)

    mae = MaskedAutoencoder(**mae_ckpt["config"]).to(device)
    mae.load_state_dict(mae_ckpt["model_state_dict"])

    classifier = MAEClassifier(mae=mae, num_classes=len(CLASS_NAMES)).to(device)
    classifier.head.load_state_dict(cls_ckpt["head_state_dict"])
    classifier.eval()

    x_tr, y_tr = load_eurosat_split("train")
    x_te, y_te = load_eurosat_split("test")
    few_idx = few_label_indices(y_tr, per_class=16, seed=42)

    # Section 3.6 correctness check
    correctness_report = verify_mae_correctness(mae, x_te[:16].to(device))
    with open(correctness_txt_path, "w", encoding="utf-8") as f:
        f.write(correctness_report + "\n")

    # Evaluate on all 5,400 test images
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    all_preds = []
    all_targets = []
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = classifier(imgs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(labels)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    test_acc = (preds_cat == targets_cat).float().mean().item()
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets_cat.numpy(), preds_cat.numpy()):
        cm[int(t), int(p)] += 1

    print(f"[Part A Evaluation] Test Accuracy (5,400 images): {test_acc * 100:.2f}% (Chance: 10.00%)")

    # Save 10x10 confusion matrix CSV
    with open(cm_csv_path, "w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(CLASS_NAMES) + "\n")
        for i, cname in enumerate(CLASS_NAMES):
            row_vals = ",".join(str(int(v)) for v in cm[i])
            f.write(f"{cname},{row_vals}\n")

    # Figure 1c: 10x10 Confusion Matrix heatmap
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"Part A: MAE Few-Label Confusion Matrix (Test Acc: {test_acc * 100:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(cm_plot_path, dpi=150)
    plt.close(fig)

    # Save 30 lightweight EuroSAT test tiles (3 per class) for the Streamlit app
    app_indices = []
    for c in range(num_classes):
        c_idx = torch.nonzero(y_te == c, as_tuple=True)[0][:3]
        app_indices.append(c_idx)
    app_indices = torch.cat(app_indices, dim=0)
    torch.save(
        {
            "images": x_te[app_indices].cpu(),
            "labels": y_te[app_indices].cpu(),
            "class_names": CLASS_NAMES,
        },
        app_samples_path,
    )

    summary = {
        "dataset": {
            "train_images": int(len(x_tr)),
            "test_images": int(len(x_te)),
            "few_label_images_used": int(len(few_idx)),
            "num_classes": num_classes,
        },
        "training": {
            "mae_total_parameters": int(mae_ckpt["total_params"]),
            "classifier_trainable_parameters": int(cls_ckpt["trainable_params"]),
            "mae_final_train_loss": float(mae_ckpt["final_train_loss"]),
            "mae_train_time_sec": float(mae_ckpt["train_time_sec"]),
            "classifier_train_time_sec": float(cls_ckpt["train_time_sec"]),
            "gpu_name": mae_ckpt["gpu_name"],
        },
        "evaluation": {
            "test_accuracy": float(test_acc),
            "test_accuracy_pct": round(test_acc * 100.0, 2),
            "correctness_check": correctness_report,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
