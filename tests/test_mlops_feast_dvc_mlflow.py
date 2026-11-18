"""TDD tests for Pillar 1 MLOps: Feast Feature Store, DVC/CML DAG, and MLflow Lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import FeatureStoreSkewError, ModelRegistryStageError
from app.mlops.dvc_pipeline import DVCPipelineRunner
from app.mlops.feast_store import EntityObservationQuery, FeastFeatureRecord, FeastFeatureService
from app.mlops.mlflow_lifecycle import MLflowLifecycleManager


def test_feast_point_in_time_join_prevents_data_leakage_and_serving_skew(
    tmp_path: Path,
) -> None:
    """Feast offline Parquet PIT join excludes future events and matches online store parity."""
    parquet_path = tmp_path / "offline_features.parquet"
    service = FeastFeatureService(offline_parquet_path=parquet_path)

    records = [
        FeastFeatureRecord(
            customer_id="cust-101",
            event_timestamp=1_000,
            tx_velocity_24h=2.0,
            credit_utilization=0.25,
            chargeback_ratio=0.01,
            device_trust_score=0.94,
        ),
        FeastFeatureRecord(
            customer_id="cust-101",
            event_timestamp=2_000,
            tx_velocity_24h=19.0,
            credit_utilization=0.97,
            chargeback_ratio=0.45,
            device_trust_score=0.12,
        ),
    ]
    service.ingest_offline_parquet(records)
    assert parquet_path.exists()

    pit_early = service.get_historical_features_pit(
        [EntityObservationQuery(customer_id="cust-101", request_timestamp=1_500)]
    )
    assert len(pit_early) == 1
    assert pit_early[0].tx_velocity_24h == pytest.approx(2.0)
    assert pit_early[0].chargeback_ratio == pytest.approx(0.01)

    materialized_count = service.materialize_to_online_store()
    assert materialized_count == 1
    online = service.get_online_features("cust-101")
    assert online.tx_velocity_24h == pytest.approx(19.0)
    assert online.device_trust_score == pytest.approx(0.12)

    with pytest.raises(FeatureStoreSkewError):
        service.get_online_features("cust-non-existent")


def test_dvc_pipeline_executes_stages_caches_unchanged_and_renders_cml_report(
    tmp_path: Path,
) -> None:
    """DVCPipelineRunner runs modular stages, skips cached stages on repeat, and outputs CML."""
    repo_root = Path(__file__).resolve().parents[1]
    runner = DVCPipelineRunner(
        dvc_yaml_path=repo_root / "dvc.yaml",
        params_yaml_path=repo_root / "params.yaml",
        workspace_dir=tmp_path / "dvc_ws",
    )

    first_run = runner.run_pipeline()
    assert first_run.stages_executed == 6
    assert first_run.cached_stages_skipped == 0
    assert first_run.metrics.roc_auc >= 0.85
    assert first_run.metrics.avg_prec >= 0.80
    assert first_run.onnx_artifact_bytes > 0

    second_run = runner.run_pipeline()
    assert second_run.cached_stages_skipped == 6
    assert second_run.stages_executed == 0

    cml_markdown = runner.generate_cml_pr_report(
        baseline=first_run.metrics,
        candidate=second_run.metrics,
    )
    assert "## CML Experiment Comparison Report" in cml_markdown
    assert "roc_auc" in cml_markdown
    assert "avg_prec" in cml_markdown


def test_mlflow_tracking_projects_onnx_models_registry_and_ab_routing(
    tmp_path: Path,
) -> None:
    """MLflowLifecycleManager enforces SLO gates on Production promotion and splits A/B traffic."""
    repo_root = Path(__file__).resolve().parents[1]
    manager = MLflowLifecycleManager(
        artifact_root=tmp_path / "mlflow_store",
        mlproject_path=repo_root / "MLproject",
    )

    project_spec = manager.load_mlproject_spec()
    assert project_spec["name"] == "omnipulse-mlops-pipeline"
    assert "main" in project_spec["entry_points"]

    weak_run = manager.train_and_track_experiment(
        run_name="weak-challenger",
        n_estimators=5,
        max_depth=1,
        sample_size=120,
        corrupt_labels_ratio=0.48,
    )
    weak_version = manager.register_model_version(weak_run.run_id)
    with pytest.raises(ModelRegistryStageError):
        manager.transition_model_stage(weak_version.version, target_stage="Production")

    strong_run = manager.train_and_track_experiment(
        run_name="strong-champion",
        n_estimators=60,
        max_depth=5,
        sample_size=400,
    )
    champion_ver = manager.register_model_version(strong_run.run_id)
    promoted = manager.transition_model_stage(champion_ver.version, target_stage="Production")
    assert promoted.stage == "Production"

    challenger_run = manager.train_and_track_experiment(
        run_name="next-challenger",
        n_estimators=80,
        max_depth=6,
        sample_size=400,
    )
    challenger_ver = manager.register_model_version(challenger_run.run_id)
    manager.transition_model_stage(challenger_ver.version, target_stage="Staging")

    routed_variants = {
        manager.predict_with_ab_routing(
            customer_id=f"cust-{idx}",
            feature_vector=[4.0, 0.35, 0.02, 0.88],
        ).served_stage
        for idx in range(60)
    }
    assert "Production" in routed_variants
    assert "Staging" in routed_variants
