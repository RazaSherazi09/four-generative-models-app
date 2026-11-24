import csv
import json
import os
import re
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

from starter.bioasq import load_corpus, load_questions
from part_d_rag.chunking import chunk_corpus
from part_d_rag.index import (
    load_retriever,
    build_and_save_index,
    load_saved_index,
)
from part_d_rag.retriever import DenseRetriever
from part_d_rag.metrics import compute_retrieval_metrics, verify_rag_correctness
from part_d_rag.generate import load_generator, generate_answer, MEDICAL_NOTICE

IDK_PATTERNS = [
    "i do not know",
    "i don't know",
    "do not contain the answer",
    "does not contain the answer",
    "do not contain enough information",
    "does not contain enough information",
    "not mentioned in the provided passages",
    "not stated in the provided passages",
    "cannot be answered from the provided passages",
    "insufficient information",
]


def is_idk_response(answer: str) -> bool:
    low = answer.lower().strip()
    return any(pat in low for pat in IDK_PATTERNS)


def extract_cited_numbers(answer: str) -> list[int]:
    """Extract integer citation markers [1]..[5] from generated text."""
    matches = re.findall(r"\[(\d+)\]", answer)
    return [int(m) for m in matches if 1 <= int(m) <= 5]


def evaluate_part_d(
    index_dir: str,
    recall_plot_path: str,
    retrieval_csv_path: str,
    rag_answers_jsonl_path: str,
    human_eval_csv_path: str,
    human_eval_md_path: str,
    samples_md_path: str,
    summary_json_path: str,
    correctness_txt_path: str,
    num_gen_questions: int = 100,
    seed: int = 42,
):
    t_start = time.perf_counter()
    corpus = load_corpus()
    test_qs = load_questions("test", corpus)
    mean_rel_per_q = sum(len(q["relevant_ids"]) for q in test_qs) / len(test_qs)
    print(
        f"[Part D] Loaded {len(corpus)} BioASQ abstracts and {len(test_qs)} test questions "
        f"(mean relevant abstracts/question: {mean_rel_per_q:.2f}).",
        flush=True,
    )

    # D.1 Chunking
    chunks = chunk_corpus(corpus, max_words=150, overlap_words=30)
    print(f"[Part D] Created {len(chunks)} chunks (max 150 words, 30-word overlap).", flush=True)

    # D.2 Embedding & Index
    emb_file = os.path.join(index_dir, "chunk_embeddings.pt")
    if os.path.exists(emb_file):
        print(f"[Part D] Loading prebuilt dense index from {index_dir}...", flush=True)
        idx_data = load_saved_index(index_dir)
    else:
        idx_data = build_and_save_index(chunks, index_dir=index_dir, batch_size=256)

    # Section 3.6 correctness check
    correctness_report = verify_rag_correctness(chunks, idx_data["embeddings"], test_qs)
    with open(correctness_txt_path, "w", encoding="utf-8") as f:
        f.write(correctness_report + "\n")

    # D.3 & D.4 Retrieval over all 707 test questions
    ret_tok, ret_model, device = load_retriever()
    retriever = DenseRetriever(
        embeddings=idx_data["embeddings"],
        abstract_ids=idx_data["abstract_ids"],
        texts=idx_data["texts"],
        tokenizer=ret_tok,
        model=ret_model,
        device=device,
    )

    all_q_texts = [q["question"] for q in test_qs]
    relevant_sets = [set(q["relevant_ids"]) for q in test_qs]
    top20_abstracts, top5_chunks_all = retriever.retrieve_batch(
        all_q_texts, top_k_abstracts=20, top_k_chunks=5
    )

    k_range = list(range(1, 21))
    all_k_metrics = compute_retrieval_metrics(
        top20_abstracts, relevant_sets, k_values=k_range
    )

    table4_metrics = {
        "Hit@1": round(all_k_metrics["Hit@1"], 4),
        "Hit@5": round(all_k_metrics["Hit@5"], 4),
        "Hit@10": round(all_k_metrics["Hit@10"], 4),
        "Recall@1": round(all_k_metrics["Recall@1"], 4),
        "Recall@5": round(all_k_metrics["Recall@5"], 4),
        "Recall@10": round(all_k_metrics["Recall@10"], 4),
        "MRR@10": round(all_k_metrics["MRR@10"], 4),
    }
    print(f"[Part D Retrieval Metrics (707 test questions)] {table4_metrics}", flush=True)

    # Save Recall@k (k=1..20) CSV and Figure 4
    with open(retrieval_csv_path, "w", encoding="utf-8") as f:
        f.write("k,hit_at_k,recall_at_k\n")
        for k in k_range:
            f.write(f"{k},{all_k_metrics[f'Hit@{k}']:.6f},{all_k_metrics[f'Recall@{k}']:.6f}\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    recalls = [all_k_metrics[f"Recall@{k}"] for k in k_range]
    hits = [all_k_metrics[f"Hit@{k}"] for k in k_range]
    ax.plot(k_range, recalls, marker="o", linewidth=2, color="#1f77b4", label="Recall@k")
    ax.plot(k_range, hits, marker="s", linewidth=1.8, linestyle="--", color="#ff7f0e", label="Hit@k")
    ax.set_title("Part D: BioASQ Dense Retrieval Performance (k = 1 to 20)")
    ax.set_xlabel("k (Top-k Retrieved Abstracts)")
    ax.set_ylabel("Score")
    ax.set_xticks(k_range)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(recall_plot_path, dpi=150)
    plt.close(fig)

    # Free retriever GPU memory before loading Qwen2.5-1.5B-Instruct
    del retriever, ret_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # D.5 & D.6 Generation on first 100 test questions
    print(
        f"[Part D] Loading Qwen/Qwen2.5-1.5B-Instruct (fp16) and generating answers for first {num_gen_questions} questions...",
        flush=True,
    )
    gen_tok, gen_model, gen_device = load_generator()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    gen_records = []
    rouge_l_scores = []
    total_citations = 0
    valid_citations = 0
    idk_count = 0

    for idx in range(num_gen_questions):
        q_item = test_qs[idx]
        rel_set = set(q_item["relevant_ids"])
        top5_chunks = top5_chunks_all[idx]
        top10_abs = top20_abstracts[idx][:10]
        retrieval_hit10 = any(aid in rel_set for aid in top10_abs)

        ans = generate_answer(
            question=q_item["question"],
            top_chunks=top5_chunks,
            tokenizer=gen_tok,
            model=gen_model,
            device=gen_device,
            max_new_tokens=256,
        )

        # ROUGE-L F-measure against reference answer
        rl = scorer.score(q_item["answer"], ans)["rougeL"].fmeasure
        rouge_l_scores.append(rl)

        # Citation precision
        cited_nums = extract_cited_numbers(ans)
        q_valid_cites = 0
        for c_num in cited_nums:
            total_citations += 1
            cited_abs_id = top5_chunks[c_num - 1]["abstract_id"]
            if cited_abs_id in rel_set:
                valid_citations += 1
                q_valid_cites += 1

        idk_flag = is_idk_response(ans)
        if idk_flag:
            idk_count += 1

        gen_records.append(
            {
                "index": idx,
                "qid": q_item["qid"],
                "question": q_item["question"],
                "reference_answer": q_item["answer"],
                "generated_answer": ans,
                "medical_notice": MEDICAL_NOTICE,
                "relevant_ids": q_item["relevant_ids"],
                "top10_abstract_ids": top10_abs,
                "retrieval_hit_at_10": retrieval_hit10,
                "cited_passages": cited_nums,
                "valid_citations": q_valid_cites,
                "total_citations": len(cited_nums),
                "is_idk": idk_flag,
                "rouge_l": round(rl, 4),
                "retrieved_chunks": top5_chunks,
            }
        )

        if (idx + 1) % 20 == 0:
            print(
                f"[Part D Generation] Completed {idx + 1}/{num_gen_questions} questions...",
                flush=True,
            )

    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    citation_precision = (
        (valid_citations / total_citations) if total_citations > 0 else 0.0
    )
    idk_rate = idk_count / num_gen_questions

    print(
        f"[Part D Generation Metrics (100 questions)] ROUGE-L: {mean_rouge_l:.4f} | "
        f"Citation Precision: {citation_precision:.4f} ({valid_citations}/{total_citations}) | "
        f"IDK Rate: {idk_rate:.4f} ({idk_count}/{num_gen_questions})",
        flush=True,
    )

    # Save results/rag_answers.jsonl
    with open(rag_answers_jsonl_path, "w", encoding="utf-8") as f:
        for rec in gen_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Section 3.5: Select 20 questions (fixed seed=42) for human evaluation
    gen_rng = torch.Generator().manual_seed(seed)
    eval20_indices = sorted(
        torch.randperm(num_gen_questions, generator=gen_rng)[:20].tolist()
    )

    # Preserve existing diffusion_blind rows in results/human_eval.csv and append 20 rag_eval rows
    existing_rows = []
    if os.path.exists(human_eval_csv_path):
        with open(human_eval_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("task") == "diffusion_blind":
                    existing_rows.append(row)

    fieldnames = [
        "task",
        "item_id",
        "true_label_or_qid",
        "member1_response",
        "member2_response",
        "member1_correct",
        "member2_correct",
        "both_agreed",
        "rag_m1_supported",
        "rag_m2_supported",
    ]
    with open(human_eval_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for q_idx in eval20_indices:
            rec = gen_records[q_idx]
            writer.writerow(
                {
                    "task": "rag_eval",
                    "item_id": q_idx + 1,
                    "true_label_or_qid": rec["qid"],
                    "member1_response": "",
                    "member2_response": "",
                    "member1_correct": "",
                    "member2_correct": "",
                    "both_agreed": "",
                    "rag_m1_supported": "",
                    "rag_m2_supported": "",
                }
            )

    # Write human-readable review packet for the 20 RAG questions
    with open(human_eval_md_path, "w", encoding="utf-8") as f:
        f.write("# Part D: 20-Question Human Evaluation Packet (Seed = 42)\n\n")
        f.write(
            "For each item below, both team members independently mark `rag_mX_supported` (1/0) "
            "and `memberX_correct` (1/0) in `results/human_eval.csv`.\n\n"
        )
        for q_idx in eval20_indices:
            rec = gen_records[q_idx]
            f.write(f"## Item #{q_idx + 1} (QID: {rec['qid']})\n\n")
            f.write(f"**Question:** {rec['question']}\n\n")
            f.write(f"**Reference Answer:** {rec['reference_answer']}\n\n")
            f.write(f"**Generated Answer:** {rec['generated_answer']}\n\n")
            f.write("**Retrieved Passages ([1]–[5]):**\n")
            for ch in rec["retrieved_chunks"]:
                rel_tag = "RELEVANT" if ch["abstract_id"] in rec["relevant_ids"] else "non-relevant"
                f.write(
                    f"- **[{ch['rank']}]** (Abstract `{ch['abstract_id']}`, `{rel_tag}`, score={ch['score']:.3f}): {ch['text']}\n"
                )
            f.write("\n---\n\n")

    # Section 4.7 Qualitative Samples (5 RAG questions: at least 1 retrieval fail, at least 1 IDK)
    fail_records = [r for r in gen_records if not r["retrieval_hit_at_10"]]
    idk_records = [r for r in gen_records if r["is_idk"]]
    success_records = [
        r for r in gen_records if r["retrieval_hit_at_10"] and not r["is_idk"] and r["valid_citations"] > 0
    ]

    selected_5 = []
    used_qids = set()
    for pool in [success_records[:2], fail_records[:2], idk_records[:2]]:
        for item in pool:
            if item["qid"] not in used_qids and len(selected_5) < 5:
                selected_5.append(item)
                used_qids.add(item["qid"])
    for item in gen_records:
        if item["qid"] not in used_qids and len(selected_5) < 5:
            selected_5.append(item)
            used_qids.add(item["qid"])

    with open(samples_md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative RAG Samples (Section 4.7)\n\n")
        f.write(f"> *{MEDICAL_NOTICE}*\n\n")
        for s_i, rec in enumerate(selected_5, start=1):
            status = (
                "Retrieval Failed (No relevant abstract in Top-10)"
                if not rec["retrieval_hit_at_10"]
                else ("Model Abstained (I Don't Know)" if rec["is_idk"] else "Grounded Retrieval & Answer")
            )
            f.write(f"## Sample {s_i}: {status} (QID: `{rec['qid']}`)\n\n")
            f.write(f"**Question:** {rec['question']}\n\n")
            f.write(f"**Generated Answer:** {rec['generated_answer']}\n\n")
            f.write(f"**Reference Answer:** {rec['reference_answer']}\n\n")
            f.write("**Top-5 Retrieved Passages:**\n")
            for ch in rec["retrieved_chunks"]:
                f.write(
                    f"- **[{ch['rank']}]** (Abstract ID: `{ch['abstract_id']}`): {ch['text']}\n"
                )
            f.write("\n---\n\n")

    elapsed_sec = time.perf_counter() - t_start
    summary = {
        "dataset": {
            "bioasq_abstracts": len(corpus),
            "bioasq_chunks": len(chunks),
            "test_questions": len(test_qs),
            "mean_relevant_abstracts_per_question": round(mean_rel_per_q, 4),
        },
        "retrieval": table4_metrics,
        "generation_100q": {
            "rouge_l_f1": round(mean_rouge_l, 4),
            "citation_precision": round(citation_precision, 4),
            "valid_citations": valid_citations,
            "total_citations": total_citations,
            "idk_rate": round(idk_rate, 4),
            "idk_count": idk_count,
            "retrieval_failures_in_first_100": len(fail_records),
        },
        "evaluation": {
            "total_time_sec": round(elapsed_sec, 2),
            "correctness_check": correctness_report,
            "failure_examples_for_e6": [
                {
                    "qid": r["qid"],
                    "question": r["question"],
                    "relevant_ids": r["relevant_ids"],
                    "top10_abstract_ids": r["top10_abstract_ids"],
                    "generated_answer": r["generated_answer"],
                    "is_idk": r["is_idk"],
                }
                for r in fail_records[:5]
            ],
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary