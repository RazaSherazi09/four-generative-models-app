import torch
from part_d_rag.index import encode_texts


class DenseRetriever:
    """Cosine-similarity dense retriever over L2-normalized chunk embeddings."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        abstract_ids: torch.Tensor,
        texts: list[str],
        tokenizer,
        model,
        device: torch.device,
    ):
        self.device = device
        self.embeddings = embeddings.to(device)  # (N_chunks, 384)
        self.abstract_ids = abstract_ids.to(device)  # (N_chunks,)
        self.texts = texts
        self.tokenizer = tokenizer
        self.model = model

        # Precompute unique sorted abstract IDs and chunk -> dense abstract index mapping
        unique_abs, chunk_to_abs_idx = torch.unique(
            self.abstract_ids, sorted=True, return_inverse=True
        )
        self.unique_abs_ids = unique_abs  # (N_abstracts,)
        self.chunk_to_abs_idx = chunk_to_abs_idx  # (N_chunks,)
        self.num_abstracts = int(unique_abs.size(0))

    @torch.no_grad()
    def encode_queries(self, questions: list[str], batch_size: int = 128) -> torch.Tensor:
        q_emb = encode_texts(
            questions,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            batch_size=batch_size,
            max_length=256,
            show_progress=False,
        )
        return q_emb.to(self.device)

    @torch.no_grad()
    def retrieve_batch(
        self,
        questions: list[str],
        top_k_abstracts: int = 20,
        top_k_chunks: int = 5,
    ) -> tuple[list[list[int]], list[list[dict]]]:
        """Batch retrieval returning:
        1) top_k_abstracts list of abstract IDs per question (max chunk score per abstract)
        2) top_k_chunks list of chunk dicts per question for RAG generation
        """
        q_emb = self.encode_queries(questions)  # (Q, 384)
        # Single matrix multiplication for cosine similarity
        chunk_scores = q_emb @ self.embeddings.T  # (Q, N_chunks)

        # Top-k chunks directly for generation (D.5)
        top_c_scores, top_c_indices = torch.topk(
            chunk_scores, k=top_k_chunks, dim=1, largest=True, sorted=True
        )
        top_c_scores_cpu = top_c_scores.cpu()
        top_c_indices_cpu = top_c_indices.cpu()
        abs_ids_cpu = self.abstract_ids.cpu()

        batch_top_chunks: list[list[dict]] = []
        for qi in range(len(questions)):
            q_chunks = []
            for rank_idx in range(top_k_chunks):
                c_idx = int(top_c_indices_cpu[qi, rank_idx].item())
                q_chunks.append(
                    {
                        "rank": rank_idx + 1,
                        "chunk_id": c_idx,
                        "abstract_id": int(abs_ids_cpu[c_idx].item()),
                        "score": float(top_c_scores_cpu[qi, rank_idx].item()),
                        "text": self.texts[c_idx],
                    }
                )
            batch_top_chunks.append(q_chunks)

        # Abstract score = max score of its chunks (D.3)
        abs_scores = torch.full(
            (len(questions), self.num_abstracts),
            float("-inf"),
            device=self.device,
            dtype=chunk_scores.dtype,
        )
        idx_expand = self.chunk_to_abs_idx.unsqueeze(0).expand_as(chunk_scores)
        abs_scores.scatter_reduce_(
            1, idx_expand, chunk_scores, reduce="amax", include_self=False
        )

        _, top_abs_indices = torch.topk(
            abs_scores, k=top_k_abstracts, dim=1, largest=True, sorted=True
        )
        top_abs_ids = self.unique_abs_ids[top_abs_indices].cpu().tolist()
        return top_abs_ids, batch_top_chunks

    def retrieve_single(
        self,
        question: str,
        top_k_abstracts: int = 10,
        top_k_chunks: int = 5,
    ) -> tuple[list[int], list[dict]]:
        abs_list, chunks_list = self.retrieve_batch(
            [question], top_k_abstracts=top_k_abstracts, top_k_chunks=top_k_chunks
        )
        return abs_list[0], chunks_list[0]