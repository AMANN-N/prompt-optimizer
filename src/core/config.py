from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class InputModality(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    MIXED = "mixed"


class TaskConfig(BaseModel):
    task_name: str
    intent: str
    input_modality: InputModality
    output_schema: Dict[str, Any]
    hard_rules: List[str] = Field(default_factory=list)
    soft_rules: List[str] = Field(default_factory=list)
    evaluation_hints: List[str] = Field(default_factory=list)


class PromptObject(BaseModel):
    system_role: str
    task_instruction: str
    field_definitions: str
    extraction_steps: str
    error_handling: str
    output_format: str

    def to_yaml(self) -> str:
        import yaml

        return yaml.dump(self.model_dump(), sort_keys=False)


class EvaluationResult(BaseModel):
    field_level_accuracy: Dict[str, float]
    rule_violations: List[str]
    hallucinations_detected: bool
    format_validity: bool


class OverallEvaluationReport(BaseModel):
    results: List[EvaluationResult]  # Detailed results per example
    overall_score: float
    failure_patterns: Dict[str, Any]  # Grouped by type


class FailureAnalysis(BaseModel):
    failure_type: str
    frequency: int
    root_cause: str
    affected_prompt_section: str


class OptimizationResult(BaseModel):
    updated_prompt: PromptObject
    change_log: str


class FieldSimilarityScore(BaseModel):
    field_name: str
    predicted_value: str
    ground_truth_value: str
    cosine_similarity: float
    is_exact_match: bool


class SemanticScoringResult(BaseModel):
    overall_similarity_score: float
    field_scores: List[FieldSimilarityScore]
    average_similarity: float
    match_count: int
    total_fields: int


class FieldPartialCredit(BaseModel):
    field_name: str
    predicted_value: Any
    ground_truth_value: Any
    score: float
    match_type: str
    normalized_predicted: str
    normalized_ground_truth: str


class PartialCreditResult(BaseModel):
    overall_credit_score: float
    field_credits: List[FieldPartialCredit]
    exact_matches: int
    normalized_matches: int
    partial_matches: int
    mismatches: int
    missing_fields: int


class ConfidenceCalibrationResult(BaseModel):
    calibrated_confidence: float
    raw_confidence: float
    temperature_scale_factor: float
    calibration_quality: str
    confidence_breakdown: Dict[str, float]
    is_calibrated: bool


class FieldUncertainty(BaseModel):
    field_name: str
    values: List[Any]
    variance_score: float
    is_uncertain: bool
    entropy: float


class UncertaintyEstimationResult(BaseModel):
    overall_uncertainty_score: float
    field_uncertainties: Dict[str, FieldUncertainty]
    uncertain_fields: List[str]
    confidence_intervals: Dict[str, Any]
    is_reliable: bool


class EnhancedEvaluationResult(BaseModel):
    exact_match_score: float
    semantic_similarity_score: float
    partial_credit_score: float
    confidence_calibrated_score: float
    uncertainty_flags: Dict[str, bool]
    field_scores: Dict[str, Dict[str, float]]
    rule_violations: List[str]
    hallucinations_detected: bool
    format_validity: bool


class EnhancedOverallEvaluationReport(BaseModel):
    results: List[EnhancedEvaluationResult]
    exact_match_score: float
    semantic_similarity_score: float
    partial_credit_score: float
    confidence_calibrated_score: float
    overall_score: float
    uncertainty_detected: bool
    uncertain_fields: List[str]
    failure_patterns: Dict[str, Any]
    generalization_gap: Optional[float] = None
    overfitting_risk: Optional[str] = None


class CrossValidationResult(BaseModel):
    fold_results: List[Dict[str, float]]
    mean_score: float
    std_score: float
    generalization_gap: float
    is_overfitting: bool


class DiversityCheckResult(BaseModel):
    cluster_scores: Dict[str, float]
    minimum_cluster_score: float
    diversity_score: float
    is_diverse: bool


class OverfittingDetectionResult(BaseModel):
    risk_level: str
    gini_coefficient: float
    fields_at_risk: List[str]
    recommendation: str
