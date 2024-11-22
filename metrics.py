import numpy as np

def calculate_dcg(relevance_scores, k):
    """Calculate DCG@k for a list of relevance scores"""
    dcg = 0
    for i in range(min(k, len(relevance_scores))):
        dcg += (2 ** relevance_scores[i] - 1) / np.log2(i + 2)
    return dcg

def calculate_ndcg(ranked_list, ground_truth, k):
    """Calculate NDCG@k given a ranked list and ground truth"""
    # Create relevance scores for ranked list (1 if document is relevant, 0 otherwise)
    relevance_scores = [1 if doc_id in ground_truth else 0 for doc_id in ranked_list[:k]]
    
    # Calculate ideal ranking (all relevant documents first)
    ideal_scores = [1] * min(len(ground_truth), k) + [0] * max(0, k - len(ground_truth))
    
    # Calculate DCG and IDCG
    dcg = calculate_dcg(relevance_scores, k)
    idcg = calculate_dcg(ideal_scores, k)
    
    # Return NDCG (if IDCG is 0, return 0 to avoid division by zero)
    return dcg / idcg if idcg > 0 else 0