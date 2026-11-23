import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starter.viz import save_grid
from part_b_vae.train import load_uhat_cached
from part_c_diffusion.model import build_unet
from part_c_diffusion.schedule import DDPMSchedule, verify_ddpm_correctness


def _save_labeled_40x8_grid(images_01: torch.Tensor, class_names: list[str], path: str):
    """Saves a 40-row x 8-column grid with each row's Urdu character label on the left."""
    imgs = images_01.detach().cpu().clamp(0, 1)
    rows, cols = 40, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15 + 1.4, rows * 0.95))
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            idx = r * cols + c
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            if c == 0:
                ax.text(
                    -3.5,
                    14.0,
                    class_names[r],
                    fontsize=8,
                    fontweight="bold",
                    ha="right",
                    va="center",
                )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.995, bottom=0.005, wspace=0.05, hspace=0.08)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def evaluate_and_sample_part_c(
    uhat_root: str,
    ckpt_path: str,
    grid_40x8_path: str,
    denoise_plot_path: str,
    blind_grid_path: str,
    blind_key_json_path: str,
    human_eval_csv_path: str,
    summary_json_path: str,
    correctness_txt_path: str,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    unet = build_unet(num_classes=num_classes).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    schedule = DDPMSchedule(num_timesteps=ckpt["num_timesteps"]).to(device)

    c_tr, _, _, _, _ = load_uhat_cached(uhat_root)
    x0_sample = (c_tr[:32] * 2.0 - 1.0).to(device)

    # Section 3.6 correctness check
    correctness_report = verify_ddpm_correctness(schedule, x0_sample)
    with open(correctness_txt_path, "w", encoding="utf-8") as f:
        f.write(correctness_report + "\n")

    # Task C.5a: 8 samples for each of the 40 characters (320 total)
    print("Sampling 40x8 character grid (320 samples across 1000 DDPM steps)...", flush=True)
    labels_40x8 = torch.arange(num_classes, device=device).repeat_interleave(8)
    gen_grid = torch.Generator(device="cpu").manual_seed(seed)
    samples_40x8, _ = schedule.sample(unet, labels_40x8, generator=gen_grid)
    samples_40x8_01 = (samples_40x8 + 1.0) * 0.5
    _save_labeled_40x8_grid(samples_40x8_01, class_names, grid_40x8_path)

    # Task C.5b: Denoising trajectory of one sample at t in {999, 750, 500, 250, 100, 0}
    print("Sampling denoising progression at t in {999, 750, 500, 250, 100, 0}...", flush=True)
    denoise_steps = [999, 750, 500, 250, 100, 0]
    target_cls = 27  # 'seen'
    gen_denoise = torch.Generator(device="cpu").manual_seed(seed + 10)
    _, snapshots = schedule.sample(
        unet,
        torch.tensor([target_cls], device=device, dtype=torch.long),
        capture_steps=set(denoise_steps),
        generator=gen_denoise,
    )
    denoise_imgs = torch.cat([(snapshots[t] + 1.0) * 0.5 for t in denoise_steps], dim=0)
    denoise_titles = [
        f"t={t} ({class_names[target_cls]})" if t == 0 else f"t={t}" for t in denoise_steps
    ]
    save_grid(denoise_imgs, denoise_plot_path, nrow=len(denoise_steps), titles=denoise_titles)

    # Section 3.3: Blind recognisability test (1 sample of each of 40 characters, shuffled)
    print("Generating 40-character shuffled blind recognisability test sheet...", flush=True)
    gen_blind = torch.Generator(device="cpu").manual_seed(seed + 99)
    blind_labels = torch.arange(num_classes, device=device)
    blind_imgs, _ = schedule.sample(unet, blind_labels, generator=gen_blind)
    blind_imgs_01 = (blind_imgs + 1.0) * 0.5

    perm = torch.randperm(num_classes, generator=gen_blind)
    shuffled_imgs = blind_imgs_01[perm]
    shuffled_labels = [class_names[int(i)] for i in perm]

    save_grid(
        shuffled_imgs,
        blind_grid_path,
        nrow=8,
        titles=[f"#{i + 1}" for i in range(num_classes)],
    )

    blind_records = [
        {"sample_id": i + 1, "true_character": shuffled_labels[i]}
        for i in range(num_classes)
    ]
    with open(blind_key_json_path, "w", encoding="utf-8") as f:
        json.dump(blind_records, f, indent=2)

    # Create results/human_eval.csv template for Section 3.3 (and ready for Section 3.5)
    with open(human_eval_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "task,item_id,true_label_or_qid,member1_response,member2_response,"
            "member1_correct,member2_correct,both_agreed,rag_m1_supported,rag_m2_supported\n"
        )
        for rec in blind_records:
            f.write(
                f"diffusion_blind,{rec['sample_id']},{rec['true_character']},,,,,,,\n"
            )

    summary = {
        "training": {
            "total_parameters": int(ckpt["total_params"]),
            "final_train_l_simple": float(ckpt["final_train_loss"]),
            "train_time_sec": float(ckpt["train_time_sec"]),
            "gpu_name": ckpt["gpu_name"],
        },
        "evaluation": {
            "num_classes": num_classes,
            "grid_samples_generated": 320,
            "blind_test_samples": 40,
            "correctness_check": correctness_report,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary