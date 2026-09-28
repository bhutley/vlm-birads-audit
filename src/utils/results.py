"""Standardized results saving for experiments."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _numpy_converter(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def save_results(
    experiment_name: str,
    results: dict,
    config: dict,
    summary_lines: list[str] | None = None,
) -> Path:
    """Save experiment results with metadata.

    Creates:
        results/<experiment_name>/results.json  -- full results with metadata
        results/<experiment_name>/summary.txt   -- human-readable summary

    Args:
        experiment_name: Name of the experiment (used as directory name).
        results: The experiment results dict.
        config: The config dict used for this run.
        summary_lines: Optional list of lines for the summary file.
            If not provided, a minimal summary is generated.

    Returns:
        Path to the output directory.
    """
    output_dir = PROJECT_ROOT / config.get("output", {}).get("results_dir", "results") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Build the full output with metadata
    output = {
        "metadata": {
            "experiment": experiment_name,
            "author": config.get("experiment", {}).get("author", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "git_commit": _get_git_commit(),
            "config": config,
        },
        "results": results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_numpy_converter)

    # Write summary
    summary_path = output_dir / "summary.txt"
    if summary_lines is None:
        summary_lines = [
            f"Experiment: {experiment_name}",
            f"Timestamp: {output['metadata']['timestamp']}",
            f"Git commit: {output['metadata']['git_commit']}",
        ]
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nResults saved to {output_dir}/")
    return output_dir
