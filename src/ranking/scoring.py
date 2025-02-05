import numpy as np

def calculate_final_score(
    semantic_score,
    domain_score,
    keyword_score,
    freshness=0.5,
    semantic_weight=0.4,
    domain_weight=0.3,
    keyword_weight=0.2,
    freshness_weight=0.1
):
    """
    Calculate the final score using configurable weights.
    
    Args:
        semantic_score: Score from semantic analysis.
        domain_score: Score based on domain relevance.
        keyword_score: Score from keyword matching.
        freshness: Freshness score (default 0.5).
        semantic_weight: Weight for semantic score (default 0.4).
        domain_weight: Weight for domain score (default 0.3, applied to domain_score/100.0).
        keyword_weight: Weight for keyword score (default 0.2).
        freshness_weight: Weight for freshness (default 0.1).
        
    Returns:
        Final weighted score.
    """
    return (
        semantic_weight * semantic_score +
        domain_weight * (domain_score / 100.0) +
        keyword_weight * keyword_score +
        freshness_weight * freshness
    )