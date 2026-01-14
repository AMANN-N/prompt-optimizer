from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
from datetime import datetime


class ConfidenceCalibrationResult(BaseModel):
    calibrated_confidence: float
    raw_confidence: float
    temperature_scale_factor: float
    calibration_quality: str
    confidence_breakdown: Dict[str, float]
    is_calibrated: bool


class CalibrationHistory(BaseModel):
    calibration_data: List[Dict[str, Any]]
    temperature: float
    last_updated: str


class ConfidenceCalibrator:
    DEFAULT_TEMP = 1.0
    MIN_TEMP = 0.1
    MAX_TEMP = 2.0

    def __init__(self, calibration_data_path: Optional[str] = None):
        self.calibration_data_path = calibration_data_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "calibration_history.json"
        )
        self.calibration_history = self._load_calibration_history()
        self.temperature = self.DEFAULT_TEMP

    def _load_calibration_history(self) -> CalibrationHistory:
        try:
            os.makedirs(os.path.dirname(self.calibration_data_path), exist_ok=True)
            if os.path.exists(self.calibration_data_path):
                with open(self.calibration_data_path, "r") as f:
                    data = json.load(f)
                    return CalibrationHistory(**data)
        except Exception:
            pass
        return CalibrationHistory(
            calibration_data=[], temperature=self.DEFAULT_TEMP, last_updated=""
        )

    def _save_calibration_history(self):
        try:
            os.makedirs(os.path.dirname(self.calibration_data_path), exist_ok=True)
            with open(self.calibration_data_path, "w") as f:
                json.dump(self.calibration_history.model_dump(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save calibration history: {e}")

    def _compute_calibration_gap(self) -> float:
        if len(self.calibration_history.calibration_data) < 5:
            return 0.0

        bins = {}
        for entry in self.calibration_history.calibration_data:
            bin_key = int(entry["raw_confidence"] * 10) / 10
            if bin_key not in bins:
                bins[bin_key] = {"predicted_confidences": [], "actual_accuracies": []}
            bins[bin_key]["predicted_confidences"].append(entry["raw_confidence"])
            bins[bin_key]["actual_accuracies"].append(entry["actual_accuracy"])

        ece = 0.0
        total_weight = 0.0

        for bin_key, data in bins.items():
            bin_weight = len(data["actual_accuracies"])
            avg_predicted = sum(data["predicted_confidences"]) / bin_weight
            avg_actual = sum(data["actual_accuracies"]) / bin_weight
            ece += bin_weight * abs(avg_predicted - avg_actual)
            total_weight += bin_weight

        return ece / total_weight if total_weight > 0 else 0.0

    def _adjust_temperature(self, calibration_gap: float) -> float:
        if calibration_gap < 0.05:
            return self.temperature
        if calibration_gap > 0.2:
            return max(self.MIN_TEMP, self.temperature * 0.8)
        else:
            adjustment = 0.9 if calibration_gap > 0.1 else 0.95
            return max(self.MIN_TEMP, self.temperature * adjustment)

    def calibrate(
        self, raw_confidence: float, actual_accuracy: Optional[float] = None
    ) -> ConfidenceCalibrationResult:
        if raw_confidence < 0 or raw_confidence > 1:
            calibrated = 0.5
            is_calibrated = False
            calibration_quality = "invalid"
        elif actual_accuracy is None:
            calibrated = raw_confidence
            is_calibrated = False
            calibration_quality = "no_actual"
        else:
            entry = {
                "raw_confidence": raw_confidence,
                "actual_accuracy": actual_accuracy,
                "timestamp": str(datetime.now()),
            }
            self.calibration_history.calibration_data.append(entry)

            if len(self.calibration_history.calibration_data) > 1000:
                self.calibration_history.calibration_data = (
                    self.calibration_history.calibration_data[-500:]
                )

            calibration_gap = self._compute_calibration_gap()
            new_temp = self._adjust_temperature(calibration_gap)
            self.temperature = new_temp
            self.calibration_history.temperature = new_temp
            self.calibration_history.last_updated = str(datetime.now())
            self._save_calibration_history()

            calibrated = min(1.0, max(0.0, raw_confidence / self.temperature))
            is_calibrated = True

            if calibration_gap < 0.05:
                calibration_quality = "excellent"
            elif calibration_gap < 0.1:
                calibration_quality = "good"
            elif calibration_gap < 0.2:
                calibration_quality = "fair"
            else:
                calibration_quality = "poor"

        breakdown = {
            "raw_confidence": raw_confidence,
            "temperature_factor": self.temperature,
            "calibrated_value": calibrated,
        }

        return ConfidenceCalibrationResult(
            calibrated_confidence=calibrated,
            raw_confidence=raw_confidence,
            temperature_scale_factor=self.temperature,
            calibration_quality=calibration_quality,
            confidence_breakdown=breakdown,
            is_calibrated=is_calibrated,
        )

    def batch_calibrate(
        self,
        confidence_scores: List[float],
        actual_accuracies: Optional[List[float]] = None,
    ) -> List[ConfidenceCalibrationResult]:
        results = []
        for i, conf in enumerate(confidence_scores):
            actual = actual_accuracies[i] if actual_accuracies is not None else None
            results.append(self.calibrate(conf, actual))
        return results

    def get_calibration_status(self) -> Dict[str, Any]:
        calibration_gap = self._compute_calibration_gap()

        if calibration_gap < 0.05:
            quality = "excellent"
        elif calibration_gap < 0.1:
            quality = "good"
        elif calibration_gap < 0.2:
            quality = "fair"
        else:
            quality = "needs_improvement"

        return {
            "temperature": self.temperature,
            "calibration_entries": len(self.calibration_history.calibration_data),
            "calibration_gap": calibration_gap,
            "quality": quality,
            "last_updated": self.calibration_history.last_updated,
        }

    def reset_calibration(self):
        self.calibration_history = CalibrationHistory(
            calibration_data=[], temperature=self.DEFAULT_TEMP, last_updated=""
        )
        self.temperature = self.DEFAULT_TEMP
        self._save_calibration_history()
