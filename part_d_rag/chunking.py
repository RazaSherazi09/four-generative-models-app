def chunk_corpus(
    corpus: dict[int, str],
    max_words: int = 150,
    overlap_words: int = 30,
) -> list[dict]:
    """Split every abstract into chunks of at most 150 words with a 30-word overlap.
    Each chunk stores its abstract_id, text, and word_count.
    """
    assert max_words > overlap_words >= 0
    step = max_words - overlap_words  # 120-word stride -> 30-word consecutive overlap
    chunks: list[dict] = []

    for abs_id in sorted(corpus.keys()):
        text = corpus[abs_id]
        words = text.split()
        if not words:
            continue

        if len(words) <= max_words:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "abstract_id": int(abs_id),
                    "text": " ".join(words),
                    "word_count": len(words),
                }
            )
            continue

        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "abstract_id": int(abs_id),
                    "text": " ".join(chunk_words),
                    "word_count": len(chunk_words),
                }
            )
            if end == len(words):
                break
            start += step

    return chunks