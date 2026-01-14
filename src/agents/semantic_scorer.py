from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import os

_semantic_model = None
_model_lock = threading.Lock()


def get_semantic_model() -> SentenceTransformer:
    global _semantic_model
    if _semantic_model is None:
        with _model_lock:
            if _semantic_model is None:
                model_name = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
                _semantic_model = SentenceTransformer(model_name)
    return _semantic_model


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


def _extract_field_values(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any], prefix: str = ""
) -> Dict[str, tuple]:
    results = {}

    for key in ground_truth:
        full_key = f"{prefix}.{key}" if prefix else key

        pred_val = predicted.get(key) if isinstance(predicted, dict) else None
        gt_val = ground_truth.get(key)

        if isinstance(gt_val, dict):
            results.update(_extract_field_values(pred_val, gt_val, full_key))
        elif isinstance(gt_val, list):
            for i, item in enumerate(gt_val):
                if isinstance(item, dict):
                    results.update(
                        _extract_field_values(
                            pred_val[i]
                            if isinstance(pred_val, list) and i < len(pred_val)
                            else {},
                            item,
                            f"{full_key}[{i}]",
                        )
                    )
                else:
                    results[f"{full_key}[{i}]"] = (
                        pred_val[i]
                        if isinstance(pred_val, list) and i < len(pred_val)
                        else None,
                        item,
                    )
        else:
            results[full_key] = (pred_val, gt_val)

    return results


def _normalize_for_embedding(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def compute_semantic_similarity(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> SemanticScoringResult:
    model = get_semantic_model()

    field_values = _extract_field_values(predicted, ground_truth)

    if not field_values:
        return SemanticScoringResult(
            overall_similarity_score=1.0,
            field_scores=[],
            average_similarity=1.0,
            match_count=0,
            total_fields=0,
        )

    field_scores = []
    similarities = []
    match_count = 0

    for field_name, (pred_val, gt_val) in field_values.items():
        pred_str = _normalize_for_embedding(pred_val)
        gt_str = _normalize_for_embedding(gt_val)

        if pred_str == gt_str:
            cosine_sim = 1.0
            is_exact = True
            match_count += 1
        elif not pred_str or not gt_str:
            cosine_sim = 0.0
            is_exact = False
        else:
            try:
                embeddings = model.encode([pred_str, gt_str])
                cosine_sim = float(
                    np.dot(embeddings[0], embeddings[1])
                    / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                )
            except Exception:
                cosine_sim = 0.0
            is_exact = False

        similarities.append(cosine_sim)
        field_scores.append(
            FieldSimilarityScore(
                field_name=field_name,
                predicted_value=str(pred_val) if pred_val else "",
                ground_truth_value=str(gt_val) if gt_val else "",
                cosine_similarity=cosine_sim,
                is_exact_match=is_exact,
            )
        )

    avg_similarity = float(np.mean(similarities)) if similarities else 1.0

    return SemanticScoringResult(
        overall_similarity_score=avg_similarity,
        field_scores=field_scores,
        average_similarity=avg_similarity,
        match_count=match_count,
        total_fields=len(field_scores),
    )


def batch_semantic_score(
    predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
) -> Dict[str, SemanticScoringResult]:
    results = {}
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        results[f"example_{i}"] = compute_semantic_similarity(pred, gt)
    return results
