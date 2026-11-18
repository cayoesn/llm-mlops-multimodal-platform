"""Statistical Data/Concept Drift Monitoring, Auto-Retraining & Maturity Assessor."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import ks_2samp

from app.config import get_settings
from app.core.exceptions import ValidationLayerError
from app.core.observability import get_observability_hub, log_event
from app.mlops.mlflow_lifecycle import MLflowLifecycleManager

_LOGGER = logging.getLogger("omnipulse.mlops.drift")


class MLOpsMaturityCapabilities(BaseModel):
    """Organizational capability flags mapped to Unit I.3 MLOps Maturity Levels 0, 1, and 2."""

    model_config = ConfigDict(frozen=True)

    automated_training_pipeline: bool
    data_and_model_versioning: bool
    feature_store_enabled: bool
    ci_cd_model_deployment: bool
    continuous_drift_monitoring: bool
    automated_retraining_trigger: bool
    ab_testing_registry: bool


class MLOpsMaturityAssessment(BaseModel):
    """Diagnostic MLOps maturity score and evolutionary roadmap recommendation."""

    model_config = ConfigDict(frozen=True)

    level: Literal[0, 1, 2]
    label: str
    capabilities_enabled: int
    next_milestone: str


class DriftEvaluationReport(BaseModel):
    """Statistical data & concept drift evaluation report with closed-loop retraining status."""

    model_config = ConfigDict(frozen=True)

    psi_score: float
    ks_statistic: float
    ks_p_value: float
    recent_accuracy: float
    data_drift_detected: bool
    concept_drift_detected: bool
    retraining_triggered: bool
    promoted_model_version: int | None


def compute_population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """Compute vectorized Population Stability Index (PSI) across quantile bins."""
    ref_arr = np.asarray(reference, dtype=np.float64)
    cur_arr = np.asarray(current, dtype=np.float64)
    if ref_arr.size < 10 or cur_arr.size < 10:
        raise ValidationLayerError("At least 10 observations required to compute PSI.")

    quantiles = np.linspace(0.0, 100.0, bins + 1)
    edges = np.percentile(ref_arr, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    unique_edges = np.unique(edges)
    if unique_edges.size < 3:
        unique_edges = np.array([-np.inf, float(np.mean(ref_arr)), np.inf])

    ref_counts, _ = np.histogram(ref_arr, bins=unique_edges)
    cur_counts, _ = np.histogram(cur_arr, bins=unique_edges)

    eps = 1e-4
    ref_pct = np.clip(ref_counts / ref_arr.size, eps, 1.0)
    cur_pct = np.clip(cur_counts / cur_arr.size, eps, 1.0)
    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi_val)


class DriftRetrainingEngine:
    """Monitors PSI, Kolmogorov-Smirnov, and Concept Drift to trigger automated retraining."""

    def __init__(self, mlflow_manager: MLflowLifecycleManager) -> None:
        """Bind MLflowLifecycleManager for automated Challenger training and promotion."""
        settings = get_settings()
        self._mlflow = mlflow_manager
        self._psi_threshold = settings.OMNIPULSE_PSI_DRIFT_THRESHOLD
        self._ks_alpha = settings.OMNIPULSE_KS_DRIFT_ALPHA
        self._min_accuracy = settings.OMNIPULSE_MIN_CHAMPION_ROC_AUC

    def evaluate_and_retrain_if_needed(
        self,
        *,
        reference_feature: Sequence[float],
        current_feature: Sequence[float],
        recent_accuracy: float,
        auto_retrain: bool = True,
    ) -> DriftEvaluationReport:
        """Evaluate Data Drift (PSI + KS) & Concept Drift and trigger retraining if breached."""
        psi = compute_population_stability_index(reference_feature, current_feature)
        ks_res = ks_2samp(reference_feature, current_feature)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        data_drift = (psi >= self._psi_threshold) or (ks_pval < self._ks_alpha and ks_stat >= 0.15)
        concept_drift = recent_accuracy < self._min_accuracy

        hub = get_observability_hub()
        hub.record_model_quality(pillar="mlops", metric_name="psi_data_drift", value=psi)
        hub.record_model_quality(
            pillar="mlops",
            metric_name="recent_live_accuracy",
            value=recent_accuracy,
        )

        promoted_version: int | None = None
        triggered = False
        if auto_retrain and (data_drift or concept_drift):
            triggered = True
            new_run = self._mlflow.train_and_track_experiment(
                run_name="drift-auto-retrained-challenger",
                n_estimators=90,
                max_depth=6,
                sample_size=420,
                random_seed=108,
            )
            reg_ver = self._mlflow.register_model_version(new_run.run_id)
            promoted = self._mlflow.transition_model_stage(
                reg_ver.version,
                target_stage="Production",
            )
            promoted_version = promoted.version
            hub.audit_chain.append(
                request_id="drift-watchdog",
                pillar="mlops",
                action=f"auto_retrain_promoted_v{promoted_version}",
                payload_hash=new_run.run_id,
                confidence=promoted.metrics.roc_auc,
            )
            log_event(
                _LOGGER,
                "drift_auto_retraining_promoted",
                psi=round(psi, 4),
                ks_p_value=round(ks_pval, 6),
                promoted_version=promoted_version,
            )

        return DriftEvaluationReport(
            psi_score=round(psi, 5),
            ks_statistic=round(ks_stat, 5),
            ks_p_value=round(ks_pval, 6),
            recent_accuracy=round(recent_accuracy, 5),
            data_drift_detected=data_drift,
            concept_drift_detected=concept_drift,
            retraining_triggered=triggered,
            promoted_model_version=promoted_version,
        )

    @staticmethod
    def assess_maturity_level(
        capabilities: MLOpsMaturityCapabilities,
    ) -> MLOpsMaturityAssessment:
        """Diagnose MLOps Maturity Level (0, 1, or 2) per Unit I.3 (Fig. 5)."""
        flags = capabilities.model_dump()
        enabled_count = sum(1 for v in flags.values() if v)

        level_2_ready = all(flags.values())
        level_1_ready = (
            capabilities.automated_training_pipeline and capabilities.data_and_model_versioning
        )

        if level_2_ready:
            return MLOpsMaturityAssessment(
                level=2,
                label="Level 2 — Full CI/CD Automation, A/B Registry & Continuous Drift Monitoring",
                capabilities_enabled=enabled_count,
                next_milestone="Optimize FinOps unit economics and multi-region Edge federation.",
            )
        if level_1_ready:
            return MLOpsMaturityAssessment(
                level=1,
                label="Level 1 — Automated ML Training Pipeline & Artifact Versioning",
                capabilities_enabled=enabled_count,
                next_milestone=(
                    "Enable CI/CD deployment gates, A/B Model Registry, and automated drift "
                    "retraining triggers to reach Level 2."
                ),
            )
        return MLOpsMaturityAssessment(
            level=0,
            label="Level 0 — Manual & Script-Driven Disconnected ML Operations",
            capabilities_enabled=enabled_count,
            next_milestone=(
                "Adopt DVC, Feast, and MLflow to automate training pipelines and reach Level 1."
            ),
        )
