import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import modal

APP_NAME = "genai-a4-four-generative-models"
VOLUME_NAME = "genai-a4-models-data-vol"
HF_REPO_ID = "razaasherazi/four-generative-models-checkpoints"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "curl")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "diffusers",
        "rouge-score",
        "matplotlib",
        "pillow",
        "numpy",
        "pandas",
        "huggingface-hub",
        "accelerate",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[".venv", ".git", "__pycache__", "results", ".vscode"],
    )
)


def _get_hf_secret():
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return [modal.Secret.from_dict({"HF_TOKEN": tok})]
    except Exception:
        pass
    return []


def upload_dir_to_hf(local_dir: str, path_in_repo: str):
    """Push trained checkpoints/indices from Modal container directly to Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in container; skipping Hugging Face Hub upload.", flush=True)
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Upload {path_in_repo} checkpoints and artifacts",
    )
    print(f"Uploaded {local_dir} -> https://huggingface.co/{HF_REPO_ID}/tree/main/{path_in_repo}", flush=True)


def _download_uhat_zip(zip_path: str) -> None:
    urls = [
        "https://zenodo.org/api/records/3670611/files/data.zip/content",
        "https://www.kaggle.com/api/v1/datasets/download/hazrat/uhat-urdu-handwritten-text-dataset",
        "https://zenodo.org/records/3670611/files/data.zip?download=1",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    last_err = None
    for url in urls:
        try:
            print(f"Fetching UHaT archive from: {url}", flush=True)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if zipfile.is_zipfile(zip_path):
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                print(f"Downloaded valid UHaT ZIP ({size_mb:.2f} MB).", flush=True)
                return
            os.remove(zip_path)
            last_err = RuntimeError(f"Response from {url} was not a valid ZIP archive.")
        except Exception as exc:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            last_err = exc
            print(f"Source failed ({exc}); trying next mirror...", flush=True)
    raise RuntimeError(f"All UHaT download sources failed. Last error: {last_err}")


def setup_vol_env():
    """Configure Hugging Face caches and directories on the persistent Modal Volume."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_HOME"] = "/vol/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/vol/hf_cache/datasets"
    os.makedirs("/vol/hf_cache/datasets", exist_ok=True)
    os.makedirs("/vol/data", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_a_mae", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_b_vae", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_c_diffusion", exist_ok=True)
    os.makedirs("/vol/indices/part_d_rag", exist_ok=True)


def ensure_uhat_tensor_cache(cache_path: str = "/vol/data/uhat_tensors.pt"):
    """Downloads & extracts UHaT on fast local container SSD (/tmp) and persists one .pt file to /vol."""
    if os.path.exists(cache_path):
        print(f"Found cached UHaT tensors at {cache_path}.", flush=True)
        return

    import sys
    import torch
    sys.path.insert(0, "/root/project")
    from starter.uhat import load_uhat_characters

    tmp_zip = "/tmp/uhat_data.zip"
    tmp_dir = "/tmp/uhat_extracted"
    os.makedirs(tmp_dir, exist_ok=True)

    _download_uhat_zip(tmp_zip)
    print("Extracting UHaT archive on fast container NVMe SSD...", flush=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(tmp_dir)
    os.remove(tmp_zip)

    print("Preprocessing 40 Urdu character classes via starter.uhat.load_uhat_characters...", flush=True)
    c_tr, l_tr, class_names = load_uhat_characters(root=tmp_dir, split="train")
    c_te, l_te, _ = load_uhat_characters(root=tmp_dir, split="test", class_names=class_names)

    torch.save(
        {
            "c_tr": c_tr,
            "l_tr": l_tr,
            "c_te": c_te,
            "l_te": l_te,
            "class_names": class_names,
        },
        cache_path,
    )
    vol.commit()
    print(f"Persisted UHaT tensor cache to {cache_path}: train={tuple(c_tr.shape)}, test={tuple(c_te.shape)}.", flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def run_task0_remote():
    setup_vol_env()
    tmp_zip = "/tmp/uhat_data.zip"
    tmp_dir = "/root/project/starter/data/uhat"
    os.makedirs(tmp_dir, exist_ok=True)
    _download_uhat_zip(tmp_zip)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(tmp_dir)
    os.remove(tmp_zip)

    res = subprocess.run(
        ["python", "check_starter.py"],
        cwd="/root/project/starter",
        capture_output=True,
        text=True,
        check=True,
    )
    vol.commit()
    return {
        "stdout": res.stdout,
        "eurosat_png": Path("/root/project/starter/results/eurosat_examples.png").read_bytes(),
        "uhat_png": Path("/root/project/starter/results/uhat_examples.png").read_bytes(),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": vol},
    secrets=_get_hf_secret(),
    timeout=3600,
)
def run_part_a_remote():
    import sys
    sys.path.insert(0, "/root/project")
    setup_vol_env()

    from part_a_mae.train_mae import pretrain_mae
    from part_a_mae.train_classifier import train_few_label_classifier
    from part_a_mae.evaluate import evaluate_part_a

    ckpt_dir = "/vol/checkpoints/part_a_mae"
    out_dir = "/tmp/part_a_results"
    os.makedirs(out_dir, exist_ok=True)

    mae_ckpt = f"{ckpt_dir}/mae_pretrained.pt"
    cls_ckpt = f"{ckpt_dir}/mae_classifier.pt"
    app_samples = f"{ckpt_dir}/app_test_samples.pt"

    loss_plot = f"{out_dir}/mae_pretrain_loss.png"
    recon_plot = f"{out_dir}/mae_reconstructions.png"
    cm_plot = f"{out_dir}/mae_confusion_matrix.png"
    history_csv = f"{out_dir}/mae_pretrain_history.csv"
    cm_csv = f"{out_dir}/mae_confusion_matrix.csv"
    summary_json = f"{out_dir}/part_a_summary.json"
    correctness_txt = f"{out_dir}/part_a_correctness.txt"

    pretrain_mae(
        ckpt_path=mae_ckpt,
        loss_plot_path=loss_plot,
        recon_plot_path=recon_plot,
        history_csv_path=history_csv,
        epochs=100,
        batch_size=256,
        lr=5e-4,
        weight_decay=0.05,
        seed=42,
    )

    train_few_label_classifier(
        mae_ckpt_path=mae_ckpt,
        cls_ckpt_path=cls_ckpt,
        epochs=200,
        lr=1e-3,
        seed=42,
    )

    summary = evaluate_part_a(
        mae_ckpt_path=mae_ckpt,
        cls_ckpt_path=cls_ckpt,
        cm_plot_path=cm_plot,
        cm_csv_path=cm_csv,
        summary_json_path=summary_json,
        correctness_txt_path=correctness_txt,
        app_samples_path=app_samples,
        batch_size=256,
    )

    vol.commit()
    upload_dir_to_hf(ckpt_dir, "part_a_mae")

    return {
        "summary": summary,
        "loss_plot": Path(loss_plot).read_bytes(),
        "recon_plot": Path(recon_plot).read_bytes(),
        "cm_plot": Path(cm_plot).read_bytes(),
        "history_csv": Path(history_csv).read_text(encoding="utf-8"),
        "cm_csv": Path(cm_csv).read_text(encoding="utf-8"),
        "summary_json": Path(summary_json).read_text(encoding="utf-8"),
        "correctness_txt": Path(correctness_txt).read_text(encoding="utf-8"),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": vol},
    secrets=_get_hf_secret(),
    timeout=3600,
)
def run_part_b_remote():
    import sys
    sys.path.insert(0, "/root/project")
    setup_vol_env()
    ensure_uhat_tensor_cache()

    from part_b_vae.train import train_vae
    from part_b_vae.evaluate import evaluate_part_b

    ckpt_dir = "/vol/checkpoints/part_b_vae"
    out_dir = "/tmp/part_b_results"
    os.makedirs(out_dir, exist_ok=True)

    vae_ckpt = f"{ckpt_dir}/vae_uhat.pt"
    app_samples = f"{ckpt_dir}/app_uhat_samples.pt"

    loss_plot = f"{out_dir}/vae_loss_terms.png"
    recon_plot = f"{out_dir}/vae_reconstructions.png"
    samples_plot = f"{out_dir}/vae_prior_samples.png"
    interp_plot = f"{out_dir}/vae_interpolation.png"
    history_csv = f"{out_dir}/vae_train_history.csv"
    summary_json = f"{out_dir}/part_b_summary.json"
    correctness_txt = f"{out_dir}/part_b_correctness.txt"

    train_vae(
        uhat_root="/tmp/uhat_extracted",
        ckpt_path=vae_ckpt,
        loss_plot_path=loss_plot,
        history_csv_path=history_csv,
        epochs=30,
        batch_size=128,
        lr=1e-3,
        latent_dim=16,
        seed=42,
    )

    summary = evaluate_part_b(
        uhat_root="/tmp/uhat_extracted",
        ckpt_path=vae_ckpt,
        recon_plot_path=recon_plot,
        samples_plot_path=samples_plot,
        interp_plot_path=interp_plot,
        summary_json_path=summary_json,
        correctness_txt_path=correctness_txt,
        app_samples_path=app_samples,
        batch_size=128,
        seed=42,
    )

    vol.commit()
    upload_dir_to_hf(ckpt_dir, "part_b_vae")

    return {
        "loss_plot": Path(loss_plot).read_bytes(),
        "recon_plot": Path(recon_plot).read_bytes(),
        "samples_plot": Path(samples_plot).read_bytes(),
        "interp_plot": Path(interp_plot).read_bytes(),
        "history_csv": Path(history_csv).read_text(encoding="utf-8"),
        "summary_json": Path(summary_json).read_text(encoding="utf-8"),
        "correctness_txt": Path(correctness_txt).read_text(encoding="utf-8"),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": vol},
    secrets=_get_hf_secret(),
    timeout=3600,
)
def run_part_c_remote():
    import sys
    sys.path.insert(0, "/root/project")
    setup_vol_env()
    ensure_uhat_tensor_cache()

    from part_c_diffusion.train import train_ddpm
    from part_c_diffusion.sample import evaluate_and_sample_part_c

    ckpt_dir = "/vol/checkpoints/part_c_diffusion"
    out_dir = "/tmp/part_c_results"
    os.makedirs(out_dir, exist_ok=True)

    ddpm_ckpt = f"{ckpt_dir}/ddpm_unet_uhat.pt"
    loss_plot = f"{out_dir}/ddpm_train_loss.png"
    forward_plot = f"{out_dir}/ddpm_forward_noising.png"
    grid_40x8_plot = f"{out_dir}/ddpm_40x8_samples.png"
    denoise_plot = f"{out_dir}/ddpm_denoising_steps.png"
    blind_grid_plot = f"{out_dir}/ddpm_blind_test_40.png"
    blind_key_json = f"{out_dir}/ddpm_blind_key.json"
    human_eval_csv = f"{out_dir}/human_eval.csv"
    history_csv = f"{out_dir}/ddpm_train_history.csv"
    summary_json = f"{out_dir}/part_c_summary.json"
    correctness_txt = f"{out_dir}/part_c_correctness.txt"

    train_ddpm(
        uhat_root="/tmp/uhat_extracted",
        ckpt_path=ddpm_ckpt,
        loss_plot_path=loss_plot,
        forward_plot_path=forward_plot,
        history_csv_path=history_csv,
        epochs=40,
        batch_size=128,
        lr=2e-4,
        num_timesteps=1000,
        seed=42,
    )

    evaluate_and_sample_part_c(
        uhat_root="/tmp/uhat_extracted",
        ckpt_path=ddpm_ckpt,
        grid_40x8_path=grid_40x8_plot,
        denoise_plot_path=denoise_plot,
        blind_grid_path=blind_grid_plot,
        blind_key_json_path=blind_key_json,
        human_eval_csv_path=human_eval_csv,
        summary_json_path=summary_json,
        correctness_txt_path=correctness_txt,
        seed=42,
    )

    vol.commit()
    upload_dir_to_hf(ckpt_dir, "part_c_diffusion")

    return {
        "loss_plot": Path(loss_plot).read_bytes(),
        "forward_plot": Path(forward_plot).read_bytes(),
        "grid_40x8_plot": Path(grid_40x8_plot).read_bytes(),
        "denoise_plot": Path(denoise_plot).read_bytes(),
        "blind_grid_plot": Path(blind_grid_plot).read_bytes(),
        "blind_key_json": Path(blind_key_json).read_text(encoding="utf-8"),
        "human_eval_csv": Path(human_eval_csv).read_text(encoding="utf-8"),
        "history_csv": Path(history_csv).read_text(encoding="utf-8"),
        "summary_json": Path(summary_json).read_text(encoding="utf-8"),
        "correctness_txt": Path(correctness_txt).read_text(encoding="utf-8"),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": vol},
    secrets=_get_hf_secret(),
    timeout=3600,
)
def run_part_d_remote(existing_human_eval_csv: str = ""):
    import sys
    sys.path.insert(0, "/root/project")
    setup_vol_env()

    from part_d_rag.evaluate import evaluate_part_d

    index_dir = "/vol/indices/part_d_rag"
    out_dir = "/tmp/part_d_results"
    os.makedirs(out_dir, exist_ok=True)

    recall_plot = f"{out_dir}/rag_recall_at_k.png"
    retrieval_csv = f"{out_dir}/rag_retrieval_curve.csv"
    rag_answers_jsonl = f"{out_dir}/rag_answers.jsonl"
    human_eval_csv = f"{out_dir}/human_eval.csv"
    human_eval_md = f"{out_dir}/rag_human_eval_20_prompts.md"
    samples_md = f"{out_dir}/samples.md"
    summary_json = f"{out_dir}/part_d_summary.json"
    correctness_txt = f"{out_dir}/part_d_correctness.txt"

    if existing_human_eval_csv.strip():
        Path(human_eval_csv).write_text(existing_human_eval_csv, encoding="utf-8")

    evaluate_part_d(
        index_dir=index_dir,
        recall_plot_path=recall_plot,
        retrieval_csv_path=retrieval_csv,
        rag_answers_jsonl_path=rag_answers_jsonl,
        human_eval_csv_path=human_eval_csv,
        human_eval_md_path=human_eval_md,
        samples_md_path=samples_md,
        summary_json_path=summary_json,
        correctness_txt_path=correctness_txt,
        num_gen_questions=100,
        seed=42,
    )

    vol.commit()
    upload_dir_to_hf(index_dir, "part_d_rag")

    return {
        "recall_plot": Path(recall_plot).read_bytes(),
        "retrieval_csv": Path(retrieval_csv).read_text(encoding="utf-8"),
        "rag_answers_jsonl": Path(rag_answers_jsonl).read_text(encoding="utf-8"),
        "human_eval_csv": Path(human_eval_csv).read_text(encoding="utf-8"),
        "human_eval_md": Path(human_eval_md).read_text(encoding="utf-8"),
        "samples_md": Path(samples_md).read_text(encoding="utf-8"),
        "summary_json": Path(summary_json).read_text(encoding="utf-8"),
        "correctness_txt": Path(correctness_txt).read_text(encoding="utf-8"),
    }


@app.local_entrypoint()
def task0_starter():
    print("Running Task 0 (starter/check_starter.py) on Modal...")
    out = run_task0_remote.remote()
    print("\n=== Task 0 Output ===")
    print(out["stdout"])
    os.makedirs("results/figures", exist_ok=True)
    Path("results/figures/eurosat_examples.png").write_bytes(out["eurosat_png"])
    Path("results/figures/uhat_examples.png").write_bytes(out["uhat_png"])
    Path("results/task0_output.txt").write_text(out["stdout"], encoding="utf-8")


@app.local_entrypoint()
def part_a():
    print("Running Part A (MAE Pretraining + Few-Label Classifier + Evaluation) on Modal A10G GPU...")
    out = run_part_a_remote.remote()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    Path("results/figures/mae_pretrain_loss.png").write_bytes(out["loss_plot"])
    Path("results/figures/mae_reconstructions.png").write_bytes(out["recon_plot"])
    Path("results/figures/mae_confusion_matrix.png").write_bytes(out["cm_plot"])
    Path("results/tables/mae_pretrain_history.csv").write_text(out["history_csv"], encoding="utf-8")
    Path("results/tables/mae_confusion_matrix.csv").write_text(out["cm_csv"], encoding="utf-8")
    Path("results/tables/part_a_summary.json").write_text(out["summary_json"], encoding="utf-8")
    Path("results/tables/part_a_correctness.txt").write_text(out["correctness_txt"], encoding="utf-8")

    print("\n=== Part A Summary ===")
    print(out["summary_json"])
    print("=== Correctness Check ===")
    print(out["correctness_txt"])


@app.local_entrypoint()
def part_b():
    print("Running Part B (ConvVAE Training + Evaluation on UHaT) on Modal A10G GPU...")
    out = run_part_b_remote.remote()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    Path("results/figures/vae_loss_terms.png").write_bytes(out["loss_plot"])
    Path("results/figures/vae_reconstructions.png").write_bytes(out["recon_plot"])
    Path("results/figures/vae_prior_samples.png").write_bytes(out["samples_plot"])
    Path("results/figures/vae_interpolation.png").write_bytes(out["interp_plot"])
    Path("results/tables/vae_train_history.csv").write_text(out["history_csv"], encoding="utf-8")
    Path("results/tables/part_b_summary.json").write_text(out["summary_json"], encoding="utf-8")
    Path("results/tables/part_b_correctness.txt").write_text(out["correctness_txt"], encoding="utf-8")

    print("\n=== Part B Summary ===")
    print(out["summary_json"])
    print("=== Correctness Check ===")
    print(out["correctness_txt"])


@app.local_entrypoint()
def part_c():
    print("Running Part C (Class-Conditional DDPM Training + Sampling on UHaT) on Modal A10G GPU...")
    out = run_part_c_remote.remote()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    Path("results/figures/ddpm_train_loss.png").write_bytes(out["loss_plot"])
    Path("results/figures/ddpm_forward_noising.png").write_bytes(out["forward_plot"])
    Path("results/figures/ddpm_40x8_samples.png").write_bytes(out["grid_40x8_plot"])
    Path("results/figures/ddpm_denoising_steps.png").write_bytes(out["denoise_plot"])
    Path("results/figures/ddpm_blind_test_40.png").write_bytes(out["blind_grid_plot"])
    Path("results/tables/ddpm_blind_key.json").write_text(out["blind_key_json"], encoding="utf-8")
    Path("results/human_eval.csv").write_text(out["human_eval_csv"], encoding="utf-8")
    Path("results/tables/ddpm_train_history.csv").write_text(out["history_csv"], encoding="utf-8")
    Path("results/tables/part_c_summary.json").write_text(out["summary_json"], encoding="utf-8")
    Path("results/tables/part_c_correctness.txt").write_text(out["correctness_txt"], encoding="utf-8")

    print("\n=== Part C Summary ===")
    print(out["summary_json"])
    print("=== Correctness Check ===")
    print(out["correctness_txt"])


@app.local_entrypoint()
def part_d():
    print("Running Part D (BioASQ Dense Retrieval + Qwen2.5-1.5B-Instruct RAG Evaluation) on Modal A10G GPU...")
    existing_csv = ""
    if Path("results/human_eval.csv").exists():
        existing_csv = Path("results/human_eval.csv").read_text(encoding="utf-8")

    out = run_part_d_remote.remote(existing_human_eval_csv=existing_csv)

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    Path("results/figures/rag_recall_at_k.png").write_bytes(out["recall_plot"])
    Path("results/tables/rag_retrieval_curve.csv").write_text(out["retrieval_csv"], encoding="utf-8")
    Path("results/rag_answers.jsonl").write_text(out["rag_answers_jsonl"], encoding="utf-8")
    Path("results/human_eval.csv").write_text(out["human_eval_csv"], encoding="utf-8")
    Path("results/tables/rag_human_eval_20_prompts.md").write_text(out["human_eval_md"], encoding="utf-8")
    Path("results/samples.md").write_text(out["samples_md"], encoding="utf-8")
    Path("results/tables/part_d_summary.json").write_text(out["summary_json"], encoding="utf-8")
    Path("results/tables/part_d_correctness.txt").write_text(out["correctness_txt"], encoding="utf-8")

    print("\n=== Part D Summary ===")
    print(out["summary_json"])
    print("=== Correctness Check ===")
    print(out["correctness_txt"])
    print("All Part D dense embeddings persisted on Modal Volume & Hugging Face Hub; evaluation artifacts saved locally.")