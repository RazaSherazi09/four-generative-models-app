import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

RETRIEVER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever(model_id: str = RETRIEVER_MODEL_ID, device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 256,
    show_progress: bool = False,
) -> torch.Tensor:
    """Encode texts with MiniLM (max_length=256), attention-masked mean pooling, and L2 norm."""
    all_embs = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state  # (B, L, 384)
        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        normed = F.normalize(pooled, p=2, dim=1)
        all_embs.append(normed.cpu())

        if show_progress and ((start // batch_size) % 50 == 0 or start + batch_size >= total):
            print(
                f"[RAG Index] Encoded {min(start + batch_size, total)}/{total} chunks...",
                flush=True,
            )

    return torch.cat(all_embs, dim=0)


def build_and_save_index(
    chunks: list[dict],
    index_dir: str,
    batch_size: int = 256,
) -> dict:
    """Embed all chunks and persist the embedding matrix + chunk metadata to disk."""
    os.makedirs(index_dir, exist_ok=True)
    tokenizer, model, device = load_retriever()

    texts = [c["text"] for c in chunks]
    abstract_ids = torch.tensor([c["abstract_id"] for c in chunks], dtype=torch.long)
    word_counts = [c["word_count"] for c in chunks]

    embeddings = encode_texts(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=256,
        show_progress=True,
    )

    emb_path = os.path.join(index_dir, "chunk_embeddings.pt")
    meta_path = os.path.join(index_dir, "chunks_metadata.pt")

    torch.save(embeddings, emb_path)
    torch.save(
        {
            "abstract_ids": abstract_ids,
            "texts": texts,
            "word_counts": word_counts,
            "model_id": RETRIEVER_MODEL_ID,
        },
        meta_path,
    )
    print(
        f"[RAG Index] Saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} to {index_dir}",
        flush=True,
    )
    return {
        "embeddings": embeddings,
        "abstract_ids": abstract_ids,
        "texts": texts,
        "word_counts": word_counts,
    }


def load_saved_index(index_dir: str) -> dict:
    emb_path = os.path.join(index_dir, "chunk_embeddings.pt")
    meta_path = os.path.join(index_dir, "chunks_metadata.pt")
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    return {
        "embeddings": embeddings,
        "abstract_ids": meta["abstract_ids"],
        "texts": meta["texts"],
        "word_counts": meta["word_counts"],
    }