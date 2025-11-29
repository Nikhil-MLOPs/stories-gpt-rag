from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import json
import mlflow

# Ensure project root is on sys.path so that "app" can be imported
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag_engine import rag_engine  # noqa: E402


EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
SAMPLE_STORY_PATH = EXPERIMENTS_DIR / "sample_story.txt"



def load_sample_text() -> str:
    if not SAMPLE_STORY_PATH.exists():
        raise FileNotFoundError(
            f"Sample story not found at {SAMPLE_STORY_PATH}. "
            "Make sure experiments/sample_story.txt exists."
        )
    return SAMPLE_STORY_PATH.read_text(encoding="utf-8")


def run_experiment(
    name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    """
    Run a single experiment:
    - chunk sample text with given params
    - compute basic metrics
    - log to MLflow
    - save JSON summary under experiments/results
    """
    text = load_sample_text()
    chunks = rag_engine.chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    num_chunks = len(chunks)
    total_words = sum(len(c.split()) for c in chunks)
    avg_chunk_words = total_words / num_chunks if num_chunks > 0 else 0.0

    summary: Dict[str, Any] = {
        "experiment_name": name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "avg_chunk_words": avg_chunk_words,
    }

    # Ensure output dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary, out_path


def main() -> None:
    mlflow.set_tracking_uri("file:" + str(BASE_DIR / "mlruns"))
    mlflow.set_experiment("chunking_experiments")

    # Define at least 5 experiments with different chunking configs
    configs: List[Dict[str, Any]] = [
        {"name": "exp_chunk_100_overlap_10", "chunk_size": 100, "chunk_overlap": 10},
        {"name": "exp_chunk_150_overlap_20", "chunk_size": 150, "chunk_overlap": 20},
        {"name": "exp_chunk_200_overlap_20", "chunk_size": 200, "chunk_overlap": 20},
        {"name": "exp_chunk_250_overlap_30", "chunk_size": 250, "chunk_overlap": 30},
        {"name": "exp_chunk_300_overlap_50", "chunk_size": 300, "chunk_overlap": 50},
    ]

    for cfg in configs:
        name = cfg["name"]
        chunk_size = cfg["chunk_size"]
        chunk_overlap = cfg["chunk_overlap"]

        print(f"Running experiment: {name} "
              f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

        with mlflow.start_run(run_name=name):
            summary, out_path = run_experiment(
                name=name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Log parameters
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)

            # Log metrics
            mlflow.log_metric("num_chunks", summary["num_chunks"])
            mlflow.log_metric("avg_chunk_words", summary["avg_chunk_words"])

            # Log artifact: JSON summary
            mlflow.log_artifact(str(out_path))

    print("All experiments completed.")
    print("You can view them with: mlflow ui --backend-store-uri file:mlruns")


if __name__ == "__main__":
    main()
