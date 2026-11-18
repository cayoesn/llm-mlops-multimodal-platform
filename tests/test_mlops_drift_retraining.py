"""TDD tests for Pillar 1 Drift Monitoring, Auto-Retraining, Maturity Level & Airflow DAG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.mlops.drift_retrainer import DriftRetrainingEngine, MLOpsMaturityCapabilities
from app.mlops.mlflow_lifecycle import MLflowLifecycleManager
from dags.omnipulse_mlops_dag import build_omnipulse_airflow_dag_spec


def test_psi_ks_drift_triggers_automated_retraining_and_promotion(tmp_path: Path) -> None:
    """Shifted production distribution triggers PSI/KS drift and auto-promotes new model."""
    repo_root = Path(__file__).resolve().parents[1]
    mlflow_mgr = MLflowLifecycleManager(
        artifact_root=tmp_path / "mlflow_drift",
        mlproject_path=repo_root / "MLproject",
    )
    base_run = mlflow_mgr.train_and_track_experiment(
        run_name="initial-champion",
        n_estimators=50,
        max_depth=5,
        sample_size=360,
    )
    v1 = mlflow_mgr.register_model_version(base_run.run_id)
    mlflow_mgr.transition_model_stage(v1.version, target_stage="Production")

    engine = DriftRetrainingEngine(mlflow_manager=mlflow_mgr)
    rng = np.random.default_rng(42)
    ref_dist = rng.normal(loc=0.30, scale=0.08, size=300).tolist()
    stable_live = rng.normal(loc=0.31, scale=0.08, size=300).tolist()

    stable_report = engine.evaluate_and_retrain_if_needed(
        reference_feature=ref_dist,
        current_feature=stable_live,
        recent_accuracy=0.91,
        auto_retrain=True,
    )
    assert stable_report.data_drift_detected is False
    assert stable_report.concept_drift_detected is False
    assert stable_report.retraining_triggered is False

    shifted_live = rng.normal(loc=0.78, scale=0.14, size=300).tolist()
    drift_report = engine.evaluate_and_retrain_if_needed(
        reference_feature=ref_dist,
        current_feature=shifted_live,
        recent_accuracy=0.71,
        auto_retrain=True,
    )
    assert drift_report.psi_score >= 0.20
    assert drift_report.ks_p_value < 0.05
    assert drift_report.data_drift_detected is True
    assert drift_report.concept_drift_detected is True
    assert drift_report.retraining_triggered is True
    assert drift_report.promoted_model_version == 2


def test_mlops_maturity_assessor_and_airflow_dag_topology() -> None:
    """Maturity assessor differentiates Level 0, 1, and 2, and Airflow DAG is acyclic."""
    level_0 = DriftRetrainingEngine.assess_maturity_level(
        MLOpsMaturityCapabilities(
            automated_training_pipeline=False,
            data_and_model_versioning=False,
            feature_store_enabled=False,
            ci_cd_model_deployment=False,
            continuous_drift_monitoring=False,
            automated_retraining_trigger=False,
            ab_testing_registry=False,
        )
    )
    assert level_0.level == 0

    level_1 = DriftRetrainingEngine.assess_maturity_level(
        MLOpsMaturityCapabilities(
            automated_training_pipeline=True,
            data_and_model_versioning=True,
            feature_store_enabled=True,
            ci_cd_model_deployment=False,
            continuous_drift_monitoring=False,
            automated_retraining_trigger=False,
            ab_testing_registry=False,
        )
    )
    assert level_1.level == 1

    level_2 = DriftRetrainingEngine.assess_maturity_level(
        MLOpsMaturityCapabilities(
            automated_training_pipeline=True,
            data_and_model_versioning=True,
            feature_store_enabled=True,
            ci_cd_model_deployment=True,
            continuous_drift_monitoring=True,
            automated_retraining_trigger=True,
            ab_testing_registry=True,
        )
    )
    assert level_2.level == 2

    dag_spec = build_omnipulse_airflow_dag_spec()
    assert dag_spec["dag_id"] == "omnipulse_mlops_continuous_pipeline"
    assert len(dag_spec["topological_order"]) == 6
    assert dag_spec["topological_order"][0] == "extract_validate_data"
    assert dag_spec["topological_order"][-1] == "monitor_drift_and_trigger_retrain"
