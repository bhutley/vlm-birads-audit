"""Evaluation metrics with confidence interval computation."""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "macro",
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Class probabilities (for AUC-ROC).
        average: Averaging strategy for F1.

    Returns:
        Dictionary of metric name -> value.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"f1_{average}": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None:
        n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
        try:
            if n_classes == 2:
                # Binary: use probability of positive class
                prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                results["auc_roc"] = roc_auc_score(y_true, prob)
            else:
                results["auc_roc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                )
        except ValueError:
            results["auc_roc"] = float("nan")

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return results


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and confidence interval from multiple trials.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    arr = np.array(values)
    mean = arr.mean()
    if len(arr) < 2:
        return mean, mean, mean

    se = stats.sem(arr)
    ci = stats.t.interval(confidence, df=len(arr) - 1, loc=mean, scale=se)
    return mean, ci[0], ci[1]


def aggregate_trial_results(
    trial_results: list[dict[str, float]],
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Aggregate results across multiple trials with confidence intervals.

    Args:
        trial_results: List of metric dicts from each trial.
        confidence: Confidence level for intervals.

    Returns:
        Dict of metric_name -> {"mean", "ci_lower", "ci_upper"}.
    """
    # Collect scalar metrics (skip confusion_matrix)
    metric_names = [
        k for k in trial_results[0]
        if k != "confusion_matrix" and isinstance(trial_results[0][k], (int, float))
    ]

    aggregated = {}
    for name in metric_names:
        values = [r[name] for r in trial_results if not np.isnan(r.get(name, float("nan")))]
        if values:
            mean, ci_lo, ci_hi = compute_confidence_interval(values, confidence)
            aggregated[name] = {"mean": mean, "ci_lower": ci_lo, "ci_upper": ci_hi}

    return aggregated


def compute_pairwise_significance(
    trials_a: list[dict[str, float]],
    trials_b: list[dict[str, float]],
    metric: str = "accuracy",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test between two sets of trial results.

    Args:
        trials_a: Trial results for model A.
        trials_b: Trial results for model B.
        metric: Metric name to compare.

    Returns:
        Dict with statistic, p_value, and significant (at alpha=0.05).
    """
    from scipy.stats import wilcoxon

    vals_a = [t[metric] for t in trials_a]
    vals_b = [t[metric] for t in trials_b]
    stat, p_value = wilcoxon(vals_a, vals_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
    }


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm–Bonferroni step-down correction over a family of p-values.

    ``pvalues`` maps a hypothesis key -> raw one-sided/two-sided p-value (the
    family is the whole dict). Returns ``{key: {p_value, holm_threshold,
    holm_family_size, reject}}``. Semantics match the inline correction used
    elsewhere in this repo (compute_significance.py): rank ascending, compare each
    to ``alpha / (n - rank)``, and once one hypothesis fails to reject, every
    higher-p hypothesis in the family also fails (step-down).

    Also returns ``holm_adjusted_p`` -- the step-down adjusted p-value
    ``max_{j<=rank} (n-j) * p_(j)``, clipped to 1 and monotone non-decreasing in
    rank -- so a reader can reproduce the correction from the table without
    re-deriving the step-down thresholds.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    out = {}
    failed = False
    running = 0.0
    for rank, (key, p) in enumerate(items):
        threshold = alpha / (n - rank)
        running = min(1.0, max(running, float(p) * (n - rank)))
        if failed:
            reject = False
        elif p < threshold:
            reject = True
        else:
            failed = True
            reject = False
        out[key] = {
            "p_value": float(p),
            "holm_threshold": float(threshold),
            "holm_adjusted_p": running,
            "holm_family_size": n,
            "reject": bool(reject),
        }
    return out
