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
    """Push trained checkpoints from Modal container directly to Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in container; skipping Hugging Face Hub upload.")
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
    print(f"Uploaded {local_dir} -> https://huggingface.co/{HF_REPO_ID}/tree/main/{path_in_repo}")


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
            print(f"Fetching UHaT archive from: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if zipfile.is_zipfile(zip_path):
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                print(f"Downloaded valid UHaT ZIP ({size_mb:.2f} MB).")
                return
            os.remove(zip_path)
            last_err = RuntimeError(f"Response from {url} was not a valid ZIP archive.")
        except Exception as exc:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            last_err = exc
            print(f"Source failed ({exc}); trying next mirror...")
    raise RuntimeError(f"All UHaT download sources failed. Last error: {last_err}")


def setup_vol_env():
    """Configure Hugging Face caches and UHaT symlink on the persistent Modal Volume."""
    os.environ["HF_HOME"] = "/vol/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/vol/hf_cache/datasets"
    os.makedirs("/vol/hf_cache/datasets", exist_ok=True)
    os.makedirs("/vol/data/uhat", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_a_mae", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_b_vae", exist_ok=True)
    os.makedirs("/vol/checkpoints/part_c_diffusion", exist_ok=True)
    os.makedirs("/vol/indices/part_d_rag", exist_ok=True)

    uhat_dir = Path("/vol/data/uhat")
    has_char_dir = any(
        d.is_dir() and "char" in d.name.lower() for d in uhat_dir.rglob("*")
    )
    if not has_char_dir:
        zip_path = "/vol/data/uhat/data.zip"
        _download_uhat_zip(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("/vol/data/uhat")
        os.remove(zip_path)
        print("UHaT dataset extracted to /vol/data/uhat.")

    starter_data = Path("/root/project/starter/data")
    starter_data.mkdir(parents=True, exist_ok=True)
    link_path = starter_data / "uhat"
    if not link_path.exists():
        link_path.symlink_to("/vol/data/uhat")


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def run_task0_remote():
    setup_vol_env()
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
    gpu="T4",
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
    print("Running Part A (MAE Pretraining + Few-Label Classifier + Evaluation) on Modal T4 GPU...")
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
    print("All Part A checkpoints persisted on Modal Volume & Hugging Face Hub; figures/tables saved locally.")
