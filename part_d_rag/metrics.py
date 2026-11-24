import torch


def compute_retrieval_metrics(
    ranked_lists: list[list[int]],
    relevant_sets: list[set[int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute Hit@k, Recall@k (denominator = min(k, |R_q|)), and MRR@10 over all queries."""
    if k_values is None:
        k_values = [1, 5, 10]

    num_q = len(ranked_lists)
    assert num_q == len(relevant_sets) and num_q > 0

    hit_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    mrr10_sum = 0.0

    for ranked, rel in zip(ranked_lists, relevant_sets):
        rel_size = len(rel)
        for k in k_values:
            top_k = ranked[:k]
            hits = sum(1 for doc_id in top_k if doc_id in rel)
            hit_sums[k] += 1.0 if hits >= 1 else 0.0
            denom = min(k, rel_size)
            recall_sums[k] += hits / denom

        rr = 0.0
        for rank_idx, doc_id in enumerate(ranked[:10], start=1):
            if doc_id in rel:
                rr = 1.0 / rank_idx
                break
        mrr10_sum += rr

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Hit@{k}"] = hit_sums[k] / num_q
        metrics[f"Recall@{k}"] = recall_sums[k] / num_q
    metrics["MRR@10"] = mrr10_sum / num_q
    return metrics


def verify_rag_correctness(
    chunks: list[dict],
    embeddings: torch.Tensor,
    questions: list[dict],
) -> str:
    """Section 3.6 correctness check:
    1) Every chunk has <= 150 words.
    2) L2 norm of every embedding is 1.0.
    3) Placing relevant_ids first in the ranking returns Hit@k = Recall@k = MRR@10 = 1.0.
    """
    max_wc = max(c["word_count"] for c in chunks)
    assert max_wc <= 150, f"Found chunk exceeding 150 words: {max_wc}"

    norms = torch.linalg.norm(embeddings, dim=1)
    max_norm_err = float((norms - 1.0).abs().max().item())
    assert max_norm_err < 1e-4, f"Embeddings not unit L2-normalized: max err={max_norm_err}"

    # Construct perfect rankings where all relevant_ids appear first
    perfect_rankings = []
    relevant_sets = []
    for q in questions:
        rel_list = list(dict.fromkeys(q["relevant_ids"]))
        relevant_sets.append(set(rel_list))
        # Pad with dummy negative IDs beyond relevant_ids
        padded = rel_list + [-i - 1 for i in range(20)]
        perfect_rankings.append(padded)

    perfect_metrics = compute_retrieval_metrics(
        perfect_rankings, relevant_sets, k_values=[1, 5, 10]
    )
    for m_name, val in perfect_metrics.items():
        assert abs(val - 1.0) < 1e-9, f"Perfect retriever failed {m_name}: got {val}"

    report = (
        f"[RAG Correctness Check PASSED] Chunks={len(chunks)} (max_words={max_wc} <= 150) | "
        f"Max L2-norm deviation={max_norm_err:.2e} (unit norm verified) | "
        f"Perfect-retriever sanity check: Hit@1={perfect_metrics['Hit@1']:.2f}, "
        f"Recall@10={perfect_metrics['Recall@10']:.2f}, MRR@10={perfect_metrics['MRR@10']:.2f}."
    )
    print(report, flush=True)
    return report