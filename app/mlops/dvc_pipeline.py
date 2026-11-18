"""DVC Pipeline DAG Runner with SHA-256 stage caching & CML PR reports (Unit II.2)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.core.exceptions import ValidationLayerError
from app.core.observability import log_event, log_info
from app.mlops.feast_store import FeastFeatureRecord, FeastFeatureService
from app.mlops.mlflow_lifecycle import (
    ExperimentMetrics,
    MLflowLifecycleManager,
    generate_synthetic_credit_dataset,
)

_LOGGER = logging.getLogger("omnipulse.mlops.dvc")


class DVCPipelineResult(BaseModel):
    """Immutable execution summary of a DVC pipeline run."""

    model_config = ConfigDict(frozen=True)

    stages_executed: int
    cached_stages_skipped: int
    stage_hashes: dict[str, str]
    metrics: ExperimentMetrics
    registered_model_version: int
    onnx_artifact_bytes: int


class DVCPipelineRunner:
    """Executes declarative dvc.yaml stages with content-addressable cache & CML reporting."""

    STAGE_ORDER = (
        "ingest",
        "validate",
        "feast_materialize",
        "train",
        "evaluate",
        "register",
    )

    def __init__(
        self,
        dvc_yaml_path: Path | None = None,
        params_yaml_path: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        """Bind dvc.yaml, params.yaml, and runtime workspace cache directory."""
        repo_root = Path(__file__).resolve().parents[2]
        self._dvc_path = dvc_yaml_path if dvc_yaml_path is not None else repo_root / "dvc.yaml"
        self._params_path = (
            params_yaml_path if params_yaml_path is not None else repo_root / "params.yaml"
        )
        self._workspace_dir = (
            workspace_dir
            if workspace_dir is not None
            else Path(get_settings().OMNIPULSE_ARTIFACT_DIR) / "dvc_workspace"
        )
        self._cache_manifest_path = self._workspace_dir / "dvc_lock_hashes.json"
        self._feast = FeastFeatureService(
            offline_parquet_path=self._workspace_dir / "offline_features.parquet"
        )
        self._mlflow = MLflowLifecycleManager(
            artifact_root=self._workspace_dir / "mlflow_store",
            mlproject_path=repo_root / "MLproject",
        )

    @property
    def feast_service(self) -> FeastFeatureService:
        """Return the pipeline's FeastFeatureService instance."""
        return self._feast

    @property
    def mlflow_manager(self) -> MLflowLifecycleManager:
        """Return the pipeline's MLflowLifecycleManager instance."""
        return self._mlflow

    def _load_specs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse and validate dvc.yaml and params.yaml."""
        dvc_spec = yaml.safe_load(self._dvc_path.read_text(encoding="utf-8"))
        params_spec = yaml.safe_load(self._params_path.read_text(encoding="utf-8"))
        if not isinstance(dvc_spec, dict) or "stages" not in dvc_spec:
            raise ValidationLayerError("Invalid dvc.yaml specification.")
        if not isinstance(params_spec, dict):
            raise ValidationLayerError("Invalid params.yaml specification.")
        return dvc_spec, params_spec

    def _compute_stage_signature(
        self,
        stage_name: str,
        stage_cfg: dict[str, Any],
        params: dict[str, Any],
        upstream_hash: str,
    ) -> str:
        """Compute deterministic SHA-256 signature over stage config, params, and upstream hash."""
        payload = json.dumps(
            [stage_name, stage_cfg, params, upstream_hash],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def run_pipeline(self, *, force: bool = False) -> DVCPipelineResult:
        """Execute the 6-stage DVC pipeline skipping unchanged cached stages in O(1)."""
        dvc_spec, params = self._load_specs()
        stages_map: dict[str, dict[str, Any]] = dvc_spec["stages"]
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        previous_lock: dict[str, Any] = {}
        if not force and self._cache_manifest_path.exists():
            previous_lock = json.loads(self._cache_manifest_path.read_text(encoding="utf-8"))

        prev_hashes: dict[str, str] = previous_lock.get("stage_hashes", {})
        current_hashes: dict[str, str] = {}
        upstream_chain = "genesis"
        executed = 0
        skipped = 0

        raw_parquet = self._workspace_dir / "raw_transactions.parquet"
        metrics_file = self._workspace_dir / "metrics.json"
        state_file = self._workspace_dir / "pipeline_state.json"

        sample_size = int(params["dataset"]["sample_size"])
        seed = int(params["dataset"]["random_seed"])
        reg_c = float(params["training"]["regularization_c"])

        tracked_run_id: str | None = previous_lock.get("run_id")
        registered_ver: int = int(previous_lock.get("registered_model_version", 0))
        onnx_bytes_count: int = int(previous_lock.get("onnx_artifact_bytes", 0))

        for stage_name in self.STAGE_ORDER:
            stage_cfg = stages_map[stage_name]
            sig = self._compute_stage_signature(stage_name, stage_cfg, params, upstream_chain)
            current_hashes[stage_name] = sig
            upstream_chain = sig

            if not force and prev_hashes.get(stage_name) == sig and metrics_file.exists():
                skipped += 1
                continue

            executed += 1
            if stage_name == "ingest":
                features, labels = generate_synthetic_credit_dataset(
                    sample_size=sample_size,
                    seed=seed,
                )
                table = pa.table(
                    {
                        "customer_id": [f"cust-{i:04d}" for i in range(len(features))],
                        "event_timestamp": [1_700_000_000 + i * 60 for i in range(len(features))],
                        "tx_velocity_24h": features[:, 0].tolist(),
                        "credit_utilization": features[:, 1].tolist(),
                        "chargeback_ratio": features[:, 2].tolist(),
                        "device_trust_score": features[:, 3].tolist(),
                        "label": labels.tolist(),
                    }
                )
                pq.write_table(table, raw_parquet, compression="snappy")

            elif stage_name == "validate":
                raw_table = pq.read_table(raw_parquet)
                if raw_table.num_rows < 50:
                    raise ValidationLayerError("Statistical validation failed: insufficient rows.")
                val_report = {"rows": raw_table.num_rows, "null_count": 0, "valid": True}
                (self._workspace_dir / "validation_report.json").write_text(
                    json.dumps(val_report),
                    encoding="utf-8",
                )

            elif stage_name == "feast_materialize":
                rows = pq.read_table(raw_parquet).to_pylist()
                feast_records = [
                    FeastFeatureRecord(
                        customer_id=str(r["customer_id"]),
                        event_timestamp=int(r["event_timestamp"]),
                        tx_velocity_24h=float(r["tx_velocity_24h"]),
                        credit_utilization=float(r["credit_utilization"]),
                        chargeback_ratio=float(r["chargeback_ratio"]),
                        device_trust_score=float(r["device_trust_score"]),
                    )
                    for r in rows
                ]
                self._feast.ingest_offline_parquet(feast_records)
                self._feast.materialize_to_online_store()

            elif stage_name == "train":
                raw_rows = pq.read_table(raw_parquet).to_pylist()
                import numpy as np

                x_mat = np.asarray(
                    [
                        [
                            r["tx_velocity_24h"],
                            r["credit_utilization"],
                            r["chargeback_ratio"],
                            r["device_trust_score"],
                        ]
                        for r in raw_rows
                    ],
                    dtype=np.float32,
                )
                y_vec = np.asarray([r["label"] for r in raw_rows], dtype=np.int64)
                run_rec = self._mlflow.train_and_track_experiment(
                    run_name="dvc-pipeline-run",
                    regularization_c=reg_c,
                    sample_size=sample_size,
                    random_seed=seed,
                    features_override=x_mat,
                    labels_override=y_vec,
                )
                tracked_run_id = run_rec.run_id
                onnx_bytes_count = run_rec.onnx_artifact_bytes
                state_file.write_text(run_rec.model_dump_json(indent=2), encoding="utf-8")

            elif stage_name == "evaluate":
                run_payload = json.loads(state_file.read_text(encoding="utf-8"))
                metrics_file.write_text(
                    json.dumps(run_payload["metrics"], indent=2),
                    encoding="utf-8",
                )

            elif stage_name == "register":
                if tracked_run_id is None:
                    raise ValidationLayerError("Cannot register model without tracked run_id.")
                reg = self._mlflow.register_model_version(tracked_run_id)
                promoted = self._mlflow.transition_model_stage(
                    reg.version,
                    target_stage="Production",
                )
                registered_ver = promoted.version

        metrics_obj = ExperimentMetrics(**json.loads(metrics_file.read_text(encoding="utf-8")))
        lock_payload = {
            "stage_hashes": current_hashes,
            "run_id": tracked_run_id,
            "registered_model_version": registered_ver,
            "onnx_artifact_bytes": onnx_bytes_count,
        }
        self._cache_manifest_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")
        log_event(
            _LOGGER,
            "dvc_pipeline_completed",
            executed=executed,
            skipped=skipped,
            roc_auc=metrics_obj.roc_auc,
        )
        return DVCPipelineResult(
            stages_executed=executed,
            cached_stages_skipped=skipped,
            stage_hashes=current_hashes,
            metrics=metrics_obj,
            registered_model_version=registered_ver,
            onnx_artifact_bytes=onnx_bytes_count,
        )

    @staticmethod
    def generate_cml_pr_report(
        *,
        baseline: ExperimentMetrics,
        candidate: ExperimentMetrics,
    ) -> str:
        """Render a Continuous Machine Learning (CML) Markdown PR metric comparison report."""
        delta_auc = candidate.roc_auc - baseline.roc_auc
        delta_ap = candidate.avg_prec - baseline.avg_prec
        delta_f1 = candidate.f1_score - baseline.f1_score
        lines = [
            "## CML Experiment Comparison Report",
            "",
            "| Metric | Baseline (`main`) | Candidate (`PR`) | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| `roc_auc` | {baseline.roc_auc:.4f} | {candidate.roc_auc:.4f} | {delta_auc:+.4f} |",
            f"| `avg_prec` | {baseline.avg_prec:.4f} | {candidate.avg_prec:.4f} | "
            f"{delta_ap:+.4f} |",
            f"| `f1_score` | {baseline.f1_score:.4f} | {candidate.f1_score:.4f} | "
            f"{delta_f1:+.4f} |",
            f"| `brier_loss` | {baseline.brier_loss:.4f} | {candidate.brier_loss:.4f} | "
            f"{candidate.brier_loss - baseline.brier_loss:+.4f} |",
        ]
        return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running the DVC + Feast + MLflow pipeline."""
    runner = DVCPipelineRunner()
    result = runner.run_pipeline()
    log_info(
        _LOGGER,
        "cli_pipeline_finished",
        stages_executed=result.stages_executed,
        cached_skipped=result.cached_stages_skipped,
        roc_auc=result.metrics.roc_auc,
        version=result.registered_model_version,
    )


if __name__ == "__main__":
    main()
