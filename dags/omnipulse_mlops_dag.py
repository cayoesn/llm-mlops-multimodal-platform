"""Apache Airflow DAG orchestrating ETL, Feast, DVC/MLflow, Batch Scoring & Drift (Unit II)."""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any


def build_omnipulse_airflow_dag_spec() -> dict[str, Any]:
    """Build and topologically verify the acyclic Apache Airflow MLOps pipeline graph."""
    dependencies: dict[str, set[str]] = {
        "extract_validate_data": set(),
        "feast_materialize_features": {"extract_validate_data"},
        "dvc_train_and_track_mlflow": {"feast_materialize_features"},
        "evaluate_and_promote_registry": {"dvc_train_and_track_mlflow"},
        "batch_score_active_customers": {"evaluate_and_promote_registry"},
        "monitor_drift_and_trigger_retrain": {"batch_score_active_customers"},
    }
    sorter = TopologicalSorter(dependencies)
    ordered_tasks = list(sorter.static_order())
    return {
        "dag_id": "omnipulse_mlops_continuous_pipeline",
        "schedule_interval": "@daily",
        "catchup": False,
        "tags": ["mlops", "feast", "dvc", "mlflow", "drift"],
        "dependencies": {k: sorted(v) for k, v in dependencies.items()},
        "topological_order": ordered_tasks,
    }
