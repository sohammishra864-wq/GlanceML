import numpy as np


def recall_at_k(retrieved_ids, relevant_ids, k):

    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # Did we retrieve at least one relevant image?
    if len(top_k & relevant_set) > 0:
        return 1
    return 0


def mean_reciprocal_rank(retrieved_ids, relevant_ids):

    relevant_set = set(relevant_ids)

    for rank, img_id in enumerate(retrieved_ids, start=1):
        if img_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids, relevant_ids, k): # Discounted Cumulative Gain

    relevant_set = set(relevant_ids)
    dcg = 0.0

    for rank, img_id in enumerate(retrieved_ids[:k], start=1):
        if img_id in relevant_set:
            # Binary relevance: 1 if relevant, 0 if not
            relevance = 1
            dcg += relevance / np.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):

    # Calculate actual DCG
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Calculate ideal DCG 
    num_relevant = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, num_relevant + 1))

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(retrieved_ids, relevant_ids, k_values=[5, 10]):

    results = {}

    # Recall@k for each k
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_ids, k)

    # MRR
    results['mrr'] = mean_reciprocal_rank(retrieved_ids, relevant_ids)

    # nDCG@k for each k
    for k in k_values:
        results[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results


def aggregate_metrics(query_results):

    if not query_results:
        return {}

    # Get all metric names from first query
    metric_names = query_results[0].keys()

    aggregated = {}
    for metric in metric_names:
        values = [result[metric] for result in query_results]
        aggregated[metric] = np.mean(values)

    return aggregated


if __name__ == "__main__":
    # Test the metrics
    print("Testing evaluation metrics...")

    # Example: Retrieved IDs and relevant IDs
    retrieved = ['img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8', 'img9', 'img10']
    relevant = ['img2', 'img5', 'img8']

    # Test Recall@5
    r5 = recall_at_k(retrieved, relevant, k=5)
    print(f"Recall@5: {r5} (should be 1, since img2 and img5 are in top-5)")

    # Test MRR
    mrr = mean_reciprocal_rank(retrieved, relevant)
    print(f"MRR: {mrr:.3f} (should be 0.5, since first relevant is at rank 2)")
    assert abs(mrr - 0.5) < 0.01

    # Test nDCG@5
    ndcg5 = ndcg_at_k(retrieved, relevant, k=5)
    print(f"nDCG@5: {ndcg5:.3f}")

    # Test full evaluation
    metrics = evaluate_query(retrieved, relevant, k_values=[5, 10])
    print(f"\nFull metrics: {metrics}")

    print("\n Metrics working correctly!")
