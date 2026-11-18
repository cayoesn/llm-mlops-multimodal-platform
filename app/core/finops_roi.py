"""FinOps cost governance and continuous ROI evaluation (Unit I Fig. 17 & Unit V.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings


class FinOpsWorkloadInput(BaseModel):
    """Immutable workload specification for FinOps and continuous ROI analysis."""

    model_config = ConfigDict(frozen=True)

    monthly_requests: int = Field(ge=0)
    business_value_per_1k_requests_usd: float = Field(ge=0.0)
    gpu_hours_baseline: float = Field(ge=0.0)
    quantization_bits: Literal[4, 8, 16] = 4
    semantic_cache_hit_ratio: float = Field(default=0.35, ge=0.0, le=0.99)
    autoscaling_efficiency_ratio: float = Field(default=0.25, ge=0.0, le=0.90)
    raw_storage_gb: float = Field(default=100.0, ge=0.0)
    use_parquet_compression: bool = True
    retraining_runs_per_month: int = Field(default=4, ge=0)


class FinOpsROIReport(BaseModel):
    """Quantified FinOps breakdown, optimization savings, and negative-ROI risk flag."""

    model_config = ConfigDict(frozen=True)

    gross_business_value_usd: float
    baseline_unoptimized_cost_usd: float
    optimized_total_cost_usd: float
    compute_cost_usd: float
    storage_cost_usd: float
    retraining_cost_usd: float
    effective_gpu_hours: float
    effective_storage_gb: float
    cost_savings_usd: float
    cost_savings_pct: float
    net_value_usd: float
    roi_ratio: float
    negative_roi_risk: bool
    governance_recommendation: str


_QUANTIZATION_COMPUTE_FACTOR: dict[int, float] = {
    16: 1.00,
    8: 0.58,
    4: 0.34,
}


class FinOpsROIEngine:
    """Evaluates continuous MLOps/LLMOps infrastructure sustainability and ROI."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Bind platform FinOps unit pricing settings."""
        self._settings = settings if settings is not None else get_settings()

    def evaluate(self, workload: FinOpsWorkloadInput) -> FinOpsROIReport:
        """Compute baseline vs optimized MLOps/LLMOps cost, storage compression, and ROI."""
        gpu_rate = self._settings.OMNIPULSE_GPU_HOUR_RATE_USD
        storage_rate = self._settings.OMNIPULSE_STORAGE_GB_MONTH_USD
        retrain_rate = self._settings.OMNIPULSE_RETRAIN_RUN_COST_USD

        gross_value = (
            workload.monthly_requests / 1_000.0
        ) * workload.business_value_per_1k_requests_usd

        retraining_cost = workload.retraining_runs_per_month * retrain_rate
        baseline_compute = workload.gpu_hours_baseline * gpu_rate
        baseline_storage = workload.raw_storage_gb * storage_rate
        baseline_total = baseline_compute + baseline_storage + retraining_cost

        quant_factor = _QUANTIZATION_COMPUTE_FACTOR[workload.quantization_bits]
        cache_factor = 1.0 - workload.semantic_cache_hit_ratio
        autoscale_factor = 1.0 - workload.autoscaling_efficiency_ratio
        effective_gpu_hours = (
            workload.gpu_hours_baseline * quant_factor * cache_factor * autoscale_factor
        )
        optimized_compute = effective_gpu_hours * gpu_rate

        effective_storage_gb = (
            workload.raw_storage_gb * 0.24
            if workload.use_parquet_compression
            else workload.raw_storage_gb
        )
        optimized_storage = effective_storage_gb * storage_rate
        optimized_total = optimized_compute + optimized_storage + retraining_cost

        savings_usd = max(0.0, baseline_total - optimized_total)
        savings_pct = (savings_usd / baseline_total * 100.0) if baseline_total > 0.0 else 0.0
        net_value = gross_value - optimized_total
        roi_ratio = (net_value / optimized_total) if optimized_total > 0.0 else 0.0
        negative_roi = net_value < 0.0

        recommendation = (
            "WARNING: Negative ROI detected — scale down GPU hours, enable INT4 quantization, "
            "Parquet tiering, and align retraining cadence with production value."
            if negative_roi
            else "HEALTHY: Positive ROI sustained via quantization, semantic caching, HPA "
            "autoscaling, and Parquet columnar storage."
        )

        return FinOpsROIReport(
            gross_business_value_usd=round(gross_value, 4),
            baseline_unoptimized_cost_usd=round(baseline_total, 4),
            optimized_total_cost_usd=round(optimized_total, 4),
            compute_cost_usd=round(optimized_compute, 4),
            storage_cost_usd=round(optimized_storage, 4),
            retraining_cost_usd=round(retraining_cost, 4),
            effective_gpu_hours=round(effective_gpu_hours, 4),
            effective_storage_gb=round(effective_storage_gb, 4),
            cost_savings_usd=round(savings_usd, 4),
            cost_savings_pct=round(savings_pct, 4),
            net_value_usd=round(net_value, 4),
            roi_ratio=round(roi_ratio, 4),
            negative_roi_risk=negative_roi,
            governance_recommendation=recommendation,
        )
