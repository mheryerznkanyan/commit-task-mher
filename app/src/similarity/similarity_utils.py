from typing import List, Dict, Tuple, Any, Set
import logging
import numpy as np

logger = logging.getLogger(__name__)


def remove_similar_chunks(
    chunks: List[Dict],
    similar_pairs: List[Tuple[str, str, float]],
    keep_strategy: str = "first"
) -> Tuple[List[Dict], Dict]:
    """
    Remove redundant chunks based on similar pairs with configurable keep strategy.

    Args:
        chunks: List of all chunk dicts (must have 'chunk_id')
        similar_pairs: List of (chunk_id, similar_chunk_id, similarity_score) tuples
        keep_strategy: Strategy for keeping chunks ('first', 'longest', 'highest_score')

    Returns:
        Tuple of (deduplicated_chunks, removal_stats)
    """
    if not similar_pairs:
        return chunks, {"removed": 0, "kept": len(chunks)}
    
    # Build removal mapping based on strategy
    to_remove = set()
    removal_reasons = {}
    
    if keep_strategy == "first":
        # Keep the first chunk in each pair, remove the second
        for chunk_id, similar_id, score in similar_pairs:
            to_remove.add(similar_id)
            removal_reasons[similar_id] = f"Similar to {chunk_id} (score: {score:.3f})"
            
    elif keep_strategy == "longest":
        # Keep the chunk with longer text
        for chunk_id, similar_id, score in similar_pairs:
            chunk1 = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
            chunk2 = next((c for c in chunks if c["chunk_id"] == similar_id), None)
            
            if chunk1 and chunk2:
                if len(chunk1["text"]) >= len(chunk2["text"]):
                    to_remove.add(similar_id)
                    removal_reasons[similar_id] = f"Shorter than {chunk_id} (score: {score:.3f})"
                else:
                    to_remove.add(chunk_id)
                    removal_reasons[chunk_id] = f"Shorter than {similar_id} (score: {score:.3f})"
                    
    elif keep_strategy == "highest_score":
        # Keep the chunk that appears more as the "first" in pairs (higher overall similarity)
        chunk_scores = {}
        for chunk_id, similar_id, score in similar_pairs:
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score
            chunk_scores[similar_id] = chunk_scores.get(similar_id, 0) - score
        
        for chunk_id, similar_id, score in similar_pairs:
            if chunk_scores.get(chunk_id, 0) >= chunk_scores.get(similar_id, 0):
                to_remove.add(similar_id)
                removal_reasons[similar_id] = f"Lower score than {chunk_id} (score: {score:.3f})"
            else:
                to_remove.add(chunk_id)
                removal_reasons[chunk_id] = f"Lower score than {similar_id} (score: {score:.3f})"
    
    # Remove chunks
    dedup_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in to_remove]
    
    # Log removal details
    for chunk_id in to_remove:
        logger.info(f"Removing chunk {chunk_id}: {removal_reasons.get(chunk_id, 'Duplicate')}")
    
    stats = {
        "original_count": len(chunks),
        "removed_count": len(to_remove),
        "final_count": len(dedup_chunks),
        "reduction_percent": round((len(to_remove) / len(chunks)) * 100, 2)
    }
    
    logger.info(f"Deduplication complete: {stats['removed_count']} chunks removed "
                f"({stats['reduction_percent']}% reduction)")
    
    return dedup_chunks, stats


def deduplicate_chunks_pipeline(
    chunks: List[Dict],
    bi_encoder_model: Any,
    cross_encoder_model: Any,
    similarity_threshold: float = 0.95,
    top_k_v1: int = 10,
    keep_strategy: str = "first",
    batch_size: int = 100
) -> Tuple[List[Dict], Dict]:
    """
    Complete pipeline for finding and removing similar chunks using hybrid (bi-encoder + cross-encoder) method.

    Args:
        chunks: List of chunk dicts
        bi_encoder_model: Bi-encoder model instance
        cross_encoder_model: Cross-encoder model instance
        similarity_threshold: Threshold for considering chunks similar
        top_k_v1: Number of most similar chunks to check per chunk (bi-encoder)
        keep_strategy: Strategy for keeping chunks ('first', 'longest', 'highest_score')
        batch_size: Batch size for processing

    Returns:
        Tuple of (deduplicated_chunks, pipeline_stats)
    """
    logger.info("Starting chunk deduplication pipeline (hybrid method)")

    # Step 1: Find similar chunks using hybrid method
    similar_pairs = find_similar_chunks_hybrid(
        chunks=chunks,
        bi_encoder_model=bi_encoder_model,
        cross_encoder_model=cross_encoder_model,
        similarity_threshold=similarity_threshold,
        top_k_bi=top_k_v1,
        batch_size=batch_size
    )

    # Step 2: Remove similar chunks
    dedup_chunks, removal_stats = remove_similar_chunks(
        chunks=chunks,
        similar_pairs=similar_pairs,
        keep_strategy=keep_strategy
    )

    # Combine stats
    pipeline_stats = {
        "similar_pairs_found": len(similar_pairs),
        "similarity_threshold": similarity_threshold,
        "keep_strategy": keep_strategy,
        "processing_method": "hybrid",
        **removal_stats
    }

    return dedup_chunks, pipeline_stats


def find_similar_chunks_hybrid(
    chunks: List[Dict],
    bi_encoder_model: Any,  # Should have compute_similarity_matrix_batch
    cross_encoder_model: Any,  # Should have compute_cross_scores(query, candidates)
    similarity_threshold: float = 0.95,
    top_k_bi: int = 10,
    batch_size: int = 100
) -> List[Tuple[str, str, float]]:
    """
    Hybrid method: Use bi-encoder for fast candidate retrieval, then cross-encoder for re-ranking.
    Args:
        chunks: List of chunk dicts (must have 'chunk_id' and 'text')
        bi_encoder_model: Model with compute_similarity_matrix_batch
        cross_encoder_model: Model with compute_cross_scores(query, candidates)
        similarity_threshold: Threshold for cross-encoder score
        top_k_bi: Number of candidates to retrieve from bi-encoder
        batch_size: Batch size for processing
    Returns:
        List of (chunk_id, similar_chunk_id, similarity_score) tuples
    """
    similar_pairs = set()
    n = len(chunks)
    logger.info(f"Finding similar chunks among {n} chunks with hybrid (bi+cross) method, threshold {similarity_threshold}")
    all_texts = [chunk["text"] for chunk in chunks]
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_chunks = chunks[batch_start:batch_end]
        batch_texts = all_texts[batch_start:batch_end]
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(n + batch_size - 1)//batch_size} (chunks {batch_start+1}-{batch_end})")
        try:
            # Step 1: Bi-encoder similarity matrix (n x m)
            similarity_matrix = bi_encoder_model.compute_similarity_matrix_batch(all_texts, batch_texts)
            for batch_idx, batch_chunk in enumerate(batch_chunks):
                batch_chunk_id = batch_chunk["chunk_id"]
                batch_global_idx = batch_start + batch_idx
                similarities = similarity_matrix[:, batch_idx]
                similarities[batch_global_idx] = -np.inf  # Exclude self
                # Step 2: Get top_k_bi candidates
                if top_k_bi < len(similarities):
                    top_k_indices = np.argpartition(similarities, -top_k_bi)[-top_k_bi:]
                else:
                    top_k_indices = np.where(similarities > -np.inf)[0]
                candidate_ids = [chunks[i]["chunk_id"] for i in top_k_indices]
                candidate_texts = [chunks[i]["text"] for i in top_k_indices]
                # Step 3: Cross-encoder re-ranking
                cross_scores = cross_encoder_model.compute_cross_scores(batch_chunk["text"], candidate_texts)
                for idx, score in zip(top_k_indices, cross_scores):
                    if score >= similarity_threshold:
                        other_chunk_id = chunks[idx]["chunk_id"]
                        # Consistent ordering to avoid duplicates
                        pair = tuple(sorted([batch_chunk_id, other_chunk_id])) + (score,)
                        similar_pairs.add(pair)
        except Exception as e:
            logger.warning(f"Error processing batch {batch_start//batch_size + 1}: {e}")
            continue
    logger.info(f"Found {len(similar_pairs)} similar chunk pairs using hybrid method")
    return list(similar_pairs) 