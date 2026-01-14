from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
from datetime import datetime, timedelta


class ProductionMetrics(BaseModel):
    timestamp: str
    extraction_success_rate: float
    average_confidence: float
    confidence_distribution: Dict[str, int]
    drift_alerts: int
    total_extractions: int
    failed_extractions: int
    field_extraction_rates: Dict[str, float]


class ProductionReport(BaseModel):
    metrics: List[ProductionMetrics]
    summary: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    period_start: str
    period_end: str


class ProductionMonitor:
    DEFAULT_STORAGE_PATH = None
    ANOMALY_THRESHOLD_SUCCESS_RATE = 0.10
    ANOMALY_THRESHOLD_CONFIDENCE = 0.15

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "production_metrics.json"
        )
        self.metrics_history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_history(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def record_extraction(
        self,
        prediction: Dict[str, Any],
        confidence: float,
        success: bool,
        field_extraction_results: Dict[str, bool] = None,
    ):
        if not self.metrics_history:
            self.metrics_history = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "extraction_success_rate": 1.0 if success else 0.0,
                    "average_confidence": confidence,
                    "confidence_distribution": self._init_confidence_distribution(),
                    "drift_alerts": 0,
                    "total_extractions": 1,
                    "failed_extractions": 0 if success else 1,
                    "field_extraction_rates": {},
                }
            ]
            self._save_history()
            return

        latest = self.metrics_history[-1]
        latest_time = datetime.fromisoformat(latest["timestamp"])

        if datetime.now() - latest_time > timedelta(hours=1):
            self.metrics_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "extraction_success_rate": 1.0 if success else 0.0,
                    "average_confidence": confidence,
                    "confidence_distribution": self._init_confidence_distribution(),
                    "drift_alerts": 0,
                    "total_extractions": 1,
                    "failed_extractions": 0 if success else 1,
                    "field_extraction_rates": {},
                }
            )
        else:
            total = latest["total_extractions"] + 1
            failed = latest["failed_extractions"] + (0 if success else 1)
            latest["extraction_success_rate"] = (total - failed) / total
            latest["average_confidence"] = (
                latest["average_confidence"] * latest["total_extractions"] + confidence
            ) / total
            latest["total_extractions"] = total
            latest["failed_extractions"] = failed
            self._update_confidence_distribution(latest, confidence)

        if field_extraction_results:
            for field, extracted in field_extraction_results.items():
                if field not in latest["field_extraction_rates"]:
                    latest["field_extraction_rates"][field] = {"success": 0, "total": 0}
                latest["field_extraction_rates"][field]["total"] += 1
                if extracted:
                    latest["field_extraction_rates"][field]["success"] += 1

        self._save_history()

    def _init_confidence_distribution(self) -> Dict[str, int]:
        return {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}

    def _update_confidence_distribution(
        self, metrics: Dict[str, Any], confidence: float
    ):
        if confidence < 0.2:
            metrics["confidence_distribution"]["0.0-0.2"] += 1
        elif confidence < 0.4:
            metrics["confidence_distribution"]["0.2-0.4"] += 1
        elif confidence < 0.6:
            metrics["confidence_distribution"]["0.4-0.6"] += 1
        elif confidence < 0.8:
            metrics["confidence_distribution"]["0.6-0.8"] += 1
        else:
            metrics["confidence_distribution"]["0.8-1.0"] += 1

    def get_weekly_report(self, days: int = 7) -> ProductionReport:
        cutoff = datetime.now() - timedelta(days=days)

        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]

        if not recent_metrics:
            return ProductionReport(
                metrics=[],
                summary={"message": "No metrics available for the specified period"},
                anomalies=[],
                recommendations=["Start recording production metrics to get insights"],
                period_start=cutoff.isoformat(),
                period_end=datetime.now().isoformat(),
            )

        total_extractions = sum(m["total_extractions"] for m in recent_metrics)
        total_failed = sum(m["failed_extractions"] for m in recent_metrics)
        avg_confidence = sum(m["average_confidence"] for m in recent_metrics) / len(
            recent_metrics
        )
        total_drift_alerts = sum(m.get("drift_alerts", 0) for m in recent_metrics)

        field_rates = {}
        for m in recent_metrics:
            for field, data in m.get("field_extraction_rates", {}).items():
                if field not in field_rates:
                    field_rates[field] = {"success": 0, "total": 0}
                field_rates[field]["success"] += data["success"]
                field_rates[field]["total"] += data["total"]

        final_field_rates = {
            field: data["success"] / data["total"]
            for field, data in field_rates.items()
            if data["total"] > 0
        }

        anomalies = self._detect_anomalies(recent_metrics)

        recommendations = []
        success_rate = (
            (total_extractions - total_failed) / total_extractions
            if total_extractions > 0
            else 0
        )

        if success_rate < 0.9:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Review failed extractions for patterns."
            )
        if avg_confidence < 0.7:
            recommendations.append(
                f"Low average confidence ({avg_confidence:.2f}). Consider improving prompt clarity."
            )
        if total_drift_alerts > 0:
            recommendations.append(
                f"Drift alerts detected ({total_drift_alerts}). Consider retraining with new data."
            )

        weak_fields = [f for f, r in final_field_rates.items() if r < 0.8]
        if weak_fields:
            recommendations.append(
                f"Weak fields: {', '.join(weak_fields)}. Review extraction rules."
            )

        if not recommendations:
            recommendations.append(
                "Production metrics look healthy. Continue monitoring."
            )

        summary = {
            "period": f"Last {days} days",
            "total_extractions": total_extractions,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "drift_alerts": total_drift_alerts,
            "best_field": max(final_field_rates, key=final_field_rates.get)
            if final_field_rates
            else None,
            "worst_field": min(final_field_rates, key=final_field_rates.get)
            if final_field_rates
            else None,
        }

        return ProductionReport(
            metrics=recent_metrics,
            summary=summary,
            anomalies=anomalies,
            recommendations=recommendations,
            period_start=cutoff.isoformat(),
            period_end=datetime.now().isoformat(),
        )

    def _detect_anomalies(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anomalies = []

        if len(metrics) < 2:
            return anomalies

        success_rates = [m["extraction_success_rate"] for m in metrics]
        avg_success = sum(success_rates) / len(success_rates)

        if avg_success > 0:
            for i, rate in enumerate(success_rates):
                drop = avg_success - rate
                if drop > self.ANOMALY_THRESHOLD_SUCCESS_RATE:
                    anomalies.append(
                        {
                            "type": "success_rate_drop",
                            "timestamp": metrics[i]["timestamp"],
                            "details": f"Success rate dropped by {drop:.1%} from average",
                            "severity": "high" if drop > 0.2 else "medium",
                        }
                    )

        confidences = [m["average_confidence"] for m in metrics]
        avg_conf = sum(confidences) / len(confidences)

        for i, conf in enumerate(confidences):
            drop = avg_conf - conf
            if drop > self.ANOMALY_THRESHOLD_CONFIDENCE:
                anomalies.append(
                    {
                        "type": "confidence_drop",
                        "timestamp": metrics[i]["timestamp"],
                        "details": f"Confidence dropped by {drop:.2f} from average",
                        "severity": "high" if drop > 0.3 else "medium",
                    }
                )

        return anomalies

    def should_alert(self) -> tuple:
        if not self.metrics_history:
            return False, ""

        recent = self.metrics_history[-7:]
        avg_success = sum(m["extraction_success_rate"] for m in recent) / len(recent)
        avg_confidence = sum(m["average_confidence"] for m in recent) / len(recent)

        baseline = (
            self.metrics_history[:-7] if len(self.metrics_history) > 7 else recent
        )
        if baseline:
            baseline_success = sum(
                m["extraction_success_rate"] for m in baseline
            ) / len(baseline)
            baseline_conf = sum(m["average_confidence"] for m in baseline) / len(
                baseline
            )

            if avg_success < baseline_success - 0.1:
                return (
                    True,
                    f"Success rate dropped from {baseline_success:.1%} to {avg_success:.1%}",
                )
            if avg_confidence < baseline_conf - 0.15:
                return (
                    True,
                    f"Confidence dropped from {baseline_conf:.2f} to {avg_confidence:.2f}",
                )

        return False, ""


def get_production_report(days: int = 7) -> ProductionReport:
    monitor = ProductionMonitor()
    return monitor.get_weekly_report(days)
