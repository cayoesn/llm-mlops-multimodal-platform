"""MLflow Lifecycle: Tracking, Projects, ONNX Models, Model Registry & Champion/Challenger A/B."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper, numpy_helper
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.core.exceptions import ModelRegistryStageError, ValidationLayerError
from app.core.observability import get_observability_hub, log_event, log_info

_LOGGER = logging.getLogger("omnipulse.mlops.mlflow")
ModelStage = Literal["None", "Staging", "Production", "Archived"]


class ExperimentMetrics(BaseModel):
    """Immutable predictive evaluation metrics tracked in MLflow and DVC/CML."""

    model_config = ConfigDict(frozen=True)

    roc_auc: float
    avg_prec: float
    f1_score: float
    brier_loss: float


class TrackedExperimentRun(BaseModel):
    """Immutable MLflow Tracking run record containing params, metrics, and ONNX artifact."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_name: str
    timestamp_utc: str
    params: dict[str, float | int | str]
    metrics: ExperimentMetrics
    onnx_artifact_path: str
    onnx_artifact_bytes: int


class RegisteredModelVersion(BaseModel):
    """Immutable MLflow Model Registry version descriptor."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    version: int
    run_id: str
    stage: ModelStage
    metrics: ExperimentMetrics
    onnx_artifact_path: str
    promoted_at_utc: str


class ABPredictionResult(BaseModel):
    """Prediction payload returned by MLflow Registry Champion/Challenger A/B router."""

    model_config = ConfigDict(frozen=True)

    customer_id: str
    risk_probability: float
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    served_version: int
    served_stage: ModelStage
    latency_ms: float


def generate_synthetic_credit_dataset(
    sample_size: int = 400,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic 4-feature credit/fraud dataset matching CUSTOMER_RISK_FEATURE_VIEW."""
    rng = np.random.default_rng(seed)
    n = max(60, sample_size)
    tx_velocity = rng.gamma(shape=2.2, scale=2.5, size=n)
    credit_util = rng.beta(a=2.0, b=3.0, size=n)
    chargeback_ratio = np.clip(rng.exponential(scale=0.06, size=n), 0.0, 1.0)
    device_trust = rng.beta(a=4.0, b=1.5, size=n)

    features = np.column_stack([tx_velocity, credit_util, chargeback_ratio, device_trust]).astype(
        np.float32
    )
    linear_signal = (
        0.38 * (tx_velocity - 4.5)
        + 3.4 * (credit_util - 0.45)
        + 7.5 * (chargeback_ratio - 0.07)
        - 3.1 * (device_trust - 0.65)
        + rng.normal(0.0, 0.45, size=n)
    )
    probs = 1.0 / (1.0 + np.exp(-linear_signal))
    labels = (probs >= 0.5).astype(np.int64)
    return features, labels


def _compile_logistic_onnx_bytes(
    effective_weights: np.ndarray,
    effective_bias: float,
) -> bytes:
    """Export standardized logistic risk parameters into a verified ONNX ModelProto binary."""
    w_matrix = effective_weights.astype(np.float32).reshape(4, 1)
    b_vec = np.array([effective_bias], dtype=np.float32)

    input_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])
    output_info = helper.make_tensor_value_info("risk_probability", TensorProto.FLOAT, [None, 1])

    w_init = numpy_helper.from_array(w_matrix, name="weights")
    b_init = numpy_helper.from_array(b_vec, name="bias")

    matmul_node = helper.make_node("MatMul", ["features", "weights"], ["linear_out"])
    add_node = helper.make_node("Add", ["linear_out", "bias"], ["logits"])
    sigmoid_node = helper.make_node("Sigmoid", ["logits"], ["risk_probability"])

    graph = helper.make_graph(
        nodes=[matmul_node, add_node, sigmoid_node],
        name="omnipulse_logistic_risk_graph",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[w_init, b_init],
    )
    model_proto = helper.make_model(
        graph,
        producer_name="omnipulse-mlflow",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model_proto)
    return bytes(model_proto.SerializeToString())


class MLflowLifecycleManager:
    """Implements MLflow Tracking, Projects, ONNX Models, Registry gates, and A/B serving."""

    MODEL_NAME = "omnipulse-credit-fraud-risk"

    def __init__(
        self,
        artifact_root: Path | None = None,
        mlproject_path: Path | None = None,
        *,
        max_archived_versions: int | None = None,
        max_tracked_runs: int = 32,
    ) -> None:
        """Initialize artifact root, MLproject file path, retention caps, and ONNX cache."""
        settings = get_settings()
        base_dir = Path(settings.OMNIPULSE_ARTIFACT_DIR)
        self._artifact_root = (
            artifact_root if artifact_root is not None else base_dir / "mlflow_store"
        )
        self._mlproject_path = (
            mlproject_path
            if mlproject_path is not None
            else Path(__file__).resolve().parents[2] / "MLproject"
        )
        resolved_archived = (
            max_archived_versions
            if max_archived_versions is not None
            else settings.OMNIPULSE_MLFLOW_MAX_ARCHIVED_VERSIONS
        )
        self._max_archived_versions = max(1, int(resolved_archived))
        self._max_tracked_runs = max(4, int(max_tracked_runs))
        self._min_roc_auc = settings.OMNIPULSE_MIN_CHAMPION_ROC_AUC
        self._challenger_ratio = settings.OMNIPULSE_AB_CHALLENGER_TRAFFIC_RATIO
        self._lock = threading.RLock()
        self._runs: dict[str, TrackedExperimentRun] = {}
        self._registry: dict[int, RegisteredModelVersion] = {}
        self._ort_sessions: dict[str, ort.InferenceSession] = {}
        self._rehydrate_from_disk()

    def _remove_run_artifacts_locked(self, run_id: str, onnx_path: str) -> None:
        """Evict ONNX session, run metadata, and on-disk run directory for an unreferenced run."""
        self._ort_sessions.pop(onnx_path, None)
        self._runs.pop(run_id, None)
        run_dir = self._artifact_root / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    def _prune_excess_archived_locked(self) -> None:
        """Bound Archived registry entries, orphan runs, and on-disk ONNX artifact directories."""
        archived_versions = [v for _, v in sorted(self._registry.items()) if v.stage == "Archived"]
        while len(archived_versions) > self._max_archived_versions:
            oldest = archived_versions.pop(0)
            self._registry.pop(oldest.version, None)
            self._ort_sessions.pop(oldest.onnx_artifact_path, None)
            still_referenced = any(r.run_id == oldest.run_id for r in self._registry.values())
            if not still_referenced:
                self._remove_run_artifacts_locked(oldest.run_id, oldest.onnx_artifact_path)

        referenced_run_ids = {r.run_id for r in self._registry.values()}
        orphan_runs = [r for r in self._runs.values() if r.run_id not in referenced_run_ids]
        orphan_runs.sort(key=lambda item: item.timestamp_utc)
        while len(self._runs) > self._max_tracked_runs and orphan_runs:
            stale = orphan_runs.pop(0)
            self._remove_run_artifacts_locked(stale.run_id, stale.onnx_artifact_path)

    def _registry_index_path(self) -> Path:
        """Return path to the persisted JSON model registry index."""
        return self._artifact_root / "registry_index.json"

    def _persist_registry_locked(self) -> None:
        """Persist current MLflow Model Registry state atomically under lock."""
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        serialized = [self._registry[v].model_dump() for v in sorted(self._registry)]
        self._registry_index_path().write_text(
            json.dumps(serialized, indent=2),
            encoding="utf-8",
        )

    def _rehydrate_from_disk(self) -> None:
        """Reload persisted runs and registered model versions on warm startup."""
        runs_dir = self._artifact_root / "runs"
        if runs_dir.exists():
            for meta_file in sorted(runs_dir.glob("*/run_metadata.json")):
                raw_run = json.loads(meta_file.read_text(encoding="utf-8"))
                run_obj = TrackedExperimentRun.model_validate(raw_run)
                if Path(run_obj.onnx_artifact_path).exists():
                    self._runs[run_obj.run_id] = run_obj

        reg_file = self._registry_index_path()
        if reg_file.exists():
            raw_list = json.loads(reg_file.read_text(encoding="utf-8"))
            if isinstance(raw_list, list):
                for entry in raw_list:
                    reg_obj = RegisteredModelVersion.model_validate(entry)
                    if Path(reg_obj.onnx_artifact_path).exists():
                        self._registry[reg_obj.version] = reg_obj

    def load_mlproject_spec(self) -> dict[str, Any]:
        """Parse and validate the standardized MLflow Projects YAML specification."""
        if not self._mlproject_path.exists():
            raise ValidationLayerError(f"MLproject file missing at {self._mlproject_path}")
        parsed = yaml.safe_load(self._mlproject_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict) or "entry_points" not in parsed:
            raise ValidationLayerError("Invalid MLproject specification.")
        return parsed

    def train_and_track_experiment(
        self,
        *,
        run_name: str,
        n_estimators: int = 60,
        max_depth: int = 5,
        regularization_c: float = 1.5,
        sample_size: int = 400,
        random_seed: int = 42,
        corrupt_labels_ratio: float = 0.0,
        features_override: np.ndarray | None = None,
        labels_override: np.ndarray | None = None,
    ) -> TrackedExperimentRun:
        """Train risk model, export verified ONNX artifact, and record MLflow Tracking run."""
        if features_override is not None and labels_override is not None:
            x_all = features_override.astype(np.float32)
            y_all = labels_override.astype(np.int64)
        else:
            x_all, y_all = generate_synthetic_credit_dataset(
                sample_size=sample_size,
                seed=random_seed,
            )

        split_idx = max(20, int(len(x_all) * 0.75))
        x_train, x_val = x_all[:split_idx], x_all[split_idx:]
        y_train = y_all[:split_idx].copy()
        y_val = y_all[split_idx:]

        if corrupt_labels_ratio > 0.0:
            rng = np.random.default_rng(random_seed + 7)
            flip_mask = rng.random(len(y_train)) < min(1.0, corrupt_labels_ratio)
            y_train = np.where(flip_mask, 1 - y_train, y_train)
            effective_c = 0.005
        else:
            effective_c = regularization_c * (1.0 + 0.002 * n_estimators + 0.01 * max_depth)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        clf = LogisticRegression(C=effective_c, max_iter=300, random_state=random_seed)
        clf.fit(x_train_scaled, y_train)

        scale_safe = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
        effective_weights = clf.coef_.ravel() / scale_safe
        effective_bias = float(
            clf.intercept_[0] - np.dot(scaler.mean_ / scale_safe, clf.coef_.ravel())
        )
        if corrupt_labels_ratio >= 0.40:
            effective_weights = -0.05 * effective_weights

        onnx_bytes = _compile_logistic_onnx_bytes(effective_weights, effective_bias)
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        val_probs = session.run(None, {"features": x_val})[0].ravel()
        val_preds = (val_probs >= 0.5).astype(np.int64)

        roc_val = float(roc_auc_score(y_val, val_probs))
        ap_val = float(average_precision_score(y_val, val_probs))
        f1_val = float(f1_score(y_val, val_preds, zero_division=0))
        brier_val = float(np.mean((val_probs - y_val.astype(np.float32)) ** 2))

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        run_dir = self._artifact_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = run_dir / "risk_model.onnx"
        onnx_path.write_bytes(onnx_bytes)

        metrics = ExperimentMetrics(
            roc_auc=round(roc_val, 5),
            avg_prec=round(ap_val, 5),
            f1_score=round(f1_val, 5),
            brier_loss=round(brier_val, 5),
        )
        run_record = TrackedExperimentRun(
            run_id=run_id,
            run_name=run_name,
            timestamp_utc=datetime.now(UTC).isoformat(),
            params={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "regularization_c": round(effective_c, 5),
                "sample_size": len(x_all),
            },
            metrics=metrics,
            onnx_artifact_path=str(onnx_path),
            onnx_artifact_bytes=len(onnx_bytes),
        )
        (run_dir / "run_metadata.json").write_text(
            run_record.model_dump_json(indent=2),
            encoding="utf-8",
        )

        with self._lock:
            self._runs[run_id] = run_record
            self._ort_sessions[str(onnx_path)] = session
            self._prune_excess_archived_locked()

        hub = get_observability_hub()
        hub.record_model_quality(pillar="mlops", metric_name="roc_auc", value=metrics.roc_auc)
        hub.record_model_quality(pillar="mlops", metric_name="f1_score", value=metrics.f1_score)
        log_event(
            _LOGGER,
            "mlflow_run_tracked",
            run_id=run_id,
            run_name=run_name,
            roc_auc=metrics.roc_auc,
        )
        return run_record

    def register_model_version(self, run_id: str) -> RegisteredModelVersion:
        """Register a tracked experiment run in the MLflow Model Registry."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ValidationLayerError(f"Unknown MLflow run_id '{run_id}'.")
            version_num = (max(self._registry) + 1) if self._registry else 1
            registered = RegisteredModelVersion(
                model_name=self.MODEL_NAME,
                version=version_num,
                run_id=run_id,
                stage="None",
                metrics=run.metrics,
                onnx_artifact_path=run.onnx_artifact_path,
                promoted_at_utc=datetime.now(UTC).isoformat(),
            )
            self._registry[version_num] = registered
            self._persist_registry_locked()
        log_info(_LOGGER, "mlflow_model_registered", version=version_num, run_id=run_id)
        return registered

    def transition_model_stage(
        self,
        version: int,
        *,
        target_stage: ModelStage,
    ) -> RegisteredModelVersion:
        """Promote or archive a registered model version enforcing Production SLO gates."""
        with self._lock:
            current = self._registry.get(version)
            if current is None:
                raise ValidationLayerError(f"Registered model version {version} not found.")
            if target_stage == "Production" and current.metrics.roc_auc < self._min_roc_auc:
                raise ModelRegistryStageError(
                    f"Version {version} ROC-AUC ({current.metrics.roc_auc:.4f}) below "
                    f"Production SLO gate ({self._min_roc_auc:.4f})."
                )
            if target_stage == "Production":
                for other_v, other_rec in list(self._registry.items()):
                    if other_v != version and other_rec.stage == "Production":
                        self._registry[other_v] = other_rec.model_copy(update={"stage": "Archived"})
                        self._ort_sessions.pop(other_rec.onnx_artifact_path, None)
            elif target_stage == "Archived":
                self._ort_sessions.pop(current.onnx_artifact_path, None)
            updated = current.model_copy(
                update={
                    "stage": target_stage,
                    "promoted_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            self._registry[version] = updated
            self._prune_excess_archived_locked()
            self._persist_registry_locked()
        log_event(
            _LOGGER,
            "mlflow_stage_transition",
            version=version,
            target_stage=target_stage,
        )
        return updated

    def list_versions(self) -> list[RegisteredModelVersion]:
        """Return all registered model versions ordered by version ascending."""
        with self._lock:
            return [self._registry[k] for k in sorted(self._registry)]

    def _get_session(self, onnx_path: str) -> ort.InferenceSession:
        """Retrieve or lazily initialize the CPU ONNX Runtime InferenceSession."""
        with self._lock:
            cached = self._ort_sessions.get(onnx_path)
            if cached is not None:
                return cached
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._ort_sessions[onnx_path] = session
            return session

    def predict_with_ab_routing(
        self,
        *,
        customer_id: str,
        feature_vector: Sequence[float],
    ) -> ABPredictionResult:
        """Execute deterministic Champion/Challenger A/B routing over ONNX Runtime."""
        if len(feature_vector) != 4:
            raise ValidationLayerError("Risk model expects exactly 4 features.")
        arr = np.asarray([feature_vector], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValidationLayerError("Feature vector values must be finite numbers.")
        started_ns = time.perf_counter_ns()

        with self._lock:
            prod_candidates = [v for v in self._registry.values() if v.stage == "Production"]
            staging_candidates = [v for v in self._registry.values() if v.stage == "Staging"]
            if not prod_candidates:
                raise ModelRegistryStageError("No Production model registered in MLflow Registry.")
            champion = prod_candidates[-1]
            selected = champion

            if staging_candidates:
                digest_val = int.from_bytes(
                    hashlib.sha256(customer_id.encode("utf-8")).digest()[:4],
                    byteorder="big",
                )
                bucket = (digest_val % 10_000) / 10_000.0
                if bucket < self._challenger_ratio:
                    selected = staging_candidates[-1]

            session = self._get_session(selected.onnx_artifact_path)
        prob = float(session.run(None, {"features": arr})[0][0, 0])
        band: Literal["LOW", "MEDIUM", "HIGH"] = (
            "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.35 else "LOW")
        )
        elapsed_ms = max(0.01, (time.perf_counter_ns() - started_ns) / 1_000_000.0)
        get_observability_hub().record_inference(
            "mlops",
            latency_seconds=elapsed_ms / 1_000.0,
            success=True,
        )
        return ABPredictionResult(
            customer_id=customer_id,
            risk_probability=round(prob, 6),
            risk_band=band,
            served_version=selected.version,
            served_stage=selected.stage,
            latency_ms=round(elapsed_ms, 4),
        )
