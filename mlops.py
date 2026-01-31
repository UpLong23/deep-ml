# %%
from ml import accuracy_score
# %%
def analyze_canary_deployment(
        canary_results: list,
        baseline_results: list,
        accuracy_tolerance: float = 0.05,
        latency_tolerance: float = 0.10
) -> dict:
    """
    https://www.deep-ml.com/problems/251
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_results: list of prediction results from canary (new) model
                       Each dict has 'latency_ms', 'prediction', 'ground_truth'
        baseline_results: list of prediction results from baseline (existing) model
                         Each dict has 'latency_ms', 'prediction', 'ground_truth'
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    if len(canary_results) == 0 | len(baseline_results) == 0:
        return {}
    canary_y_pred = [result['prediction'] for result in canary_results]
    canary_y_true = [result['ground_truth'] for result in canary_results]
    canary_latency = [result['latency_ms'] for result in canary_results]

    canary_accuracy = accuracy_score(canary_y_true, canary_y_pred)
    canary_avg_latency = sum(canary_latency) / len(canary_latency)

    baseline_y_pred = [result['prediction'] for result in baseline_results]
    baseline_y_true = [result['ground_truth'] for result in baseline_results]
    baseline_latency = [result['latency_ms'] for result in baseline_results]

    baseline_accuracy = accuracy_score(baseline_y_true, baseline_y_pred)
    baseline_avg_latency = sum(baseline_latency) / len(baseline_latency)


    accuracy_change_pct = 100 * (canary_accuracy - baseline_accuracy) / baseline_accuracy
    latency_change_pct = 100 * (canary_avg_latency - baseline_avg_latency) / baseline_avg_latency

    promote_recommended = (accuracy_change_pct >= -accuracy_tolerance**100) & (latency_change_pct <= latency_tolerance*100)

    results = {
        'canary_accuracy':round(canary_accuracy, 4),
        'baseline_accuracy':round(baseline_accuracy, 4),
        'accuracy_change_pct':round(accuracy_change_pct, 2),
        'canary_avg_latency':round(canary_avg_latency, 2),
        'baseline_avg_latency':round(baseline_avg_latency, 2),
        'latency_change_pct':round(latency_change_pct, 2),
        'promote_recommended':promote_recommended
    }
    return results 

# %%
canary_results = [{'latency_ms': 45, 'prediction': 1, 'ground_truth': 1},
                  {'latency_ms': 50, 'prediction': 0, 'ground_truth': 0}, 
                  {'latency_ms': 48, 'prediction': 1, 'ground_truth': 1}, 
                  {'latency_ms': 52, 'prediction': 1, 'ground_truth': 0}, 
                  {'latency_ms': 47, 'prediction': 0, 'ground_truth': 0}]

baseline_results = [{'latency_ms': 50, 'prediction': 1, 'ground_truth': 1}, 
                    {'latency_ms': 55, 'prediction': 0, 'ground_truth': 0}, 
                    {'latency_ms': 52, 'prediction': 1, 'ground_truth': 0}, 
                    {'latency_ms': 58, 'prediction': 0, 'ground_truth': 0}, 
                    {'latency_ms': 53, 'prediction': 1, 'ground_truth': 1}]

analyze_canary_deployment(canary_results, baseline_results)