import numpy as np

def calculate_final_score(semantic_score, domain_score, keyword_score, freshness=0.5):
    """
    Weighted example:
      final = 0.4 * semantic_score
            + 0.3 * (domain_score / 100.0)
            + 0.2 * keyword_score
            + 0.1 * freshness
    """
    return (
        0.4 * semantic_score
        + 0.3 * (domain_score / 100.0)
        + 0.2 * keyword_score
        + 0.1 * freshness
    )