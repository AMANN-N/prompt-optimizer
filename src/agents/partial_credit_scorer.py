from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re
import decimal
import numpy as np
from dateutil import parser as date_parser
from difflib import SequenceMatcher


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


def _normalize_numeric(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        val_str = str(value).strip()
        val_str = val_str.replace("₹", "").replace("Rs.", "").replace("Rs", "")
        val_str = val_str.replace(",", "").replace(" ", "")
        val_str = val_str.lower().replace("lakh", "00000").replace("lac", "00000")
        val_str = val_str.replace("crore", "0000000").replace("million", "000000")
        val_str = val_str.replace("%", "")
        val_str = re.sub(r"\.0+$", "", val_str)
        val = decimal.Decimal(val_str)
        return str(int(val))
    except:
        return str(value).strip().lower()


def _normalize_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        val_str = str(value).strip()
        dt = date_parser.parse(val_str, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except:
        return str(value).strip().lower()


def _normalize_currency(value: Any) -> Optional[str]:
    return _normalize_numeric(value)


def _normalize_percentage(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        val_str = str(value).strip()
        val_str = val_str.replace("%", "")
        val = float(val_str)
        return f"{val:.2f}"
    except:
        return str(value).strip().lower()


def _jaccard_similarity(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _token_overlap_ratio(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    tokens1 = str1.lower().split()
    tokens2 = str2.lower().split()
    if not tokens1 or not tokens2:
        return 0.0
    matches = sum(1 for t in tokens1 if t in tokens2)
    return matches / max(len(tokens1), len(tokens2))


def _compute_edit_distance_ratio(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def _is_numeric(value: Any) -> bool:
    if value is None:
        return False
    try:
        val_str = (
            str(value).strip().replace(",", "").replace("₹", "").replace("Rs.", "")
        )
        decimal.Decimal(val_str)
        return True
    except:
        return False


def _is_date(value: Any) -> bool:
    if value is None:
        return False
    try:
        date_parser.parse(str(value), fuzzy=True)
        return True
    except:
        return False


def _compute_field_credit(
    pred_val: Any, gt_val: Any, field_type: str = "text"
) -> tuple:
    pred_str = str(pred_val).strip() if pred_val else ""
    gt_str = str(gt_val).strip() if gt_val else ""

    if pred_str == gt_str:
        return 1.0, "exact", pred_str, gt_str

    if not pred_str and gt_str:
        return 0.0, "missing", "", gt_str

    if pred_str and not gt_str:
        return 0.0, "extra", pred_str, ""

    pred_norm = pred_str
    gt_norm = gt_str

    if _is_numeric(pred_val) and _is_numeric(gt_val):
        pred_norm = _normalize_numeric(pred_val) or pred_str
        gt_norm = _normalize_numeric(gt_val) or gt_str

        if pred_norm == gt_norm:
            return 0.7, "normalized_numeric", pred_str, gt_str

        try:
            pred_num = float(pred_norm)
            gt_num = float(gt_norm)
            if pred_num > 0:
                ratio = min(pred_num, gt_num) / max(pred_num, gt_num)
                if ratio >= 0.99:
                    return 0.7, "normalized_numeric", pred_str, gt_str
        except:
            pass

    if _is_date(pred_val) and _is_date(gt_val):
        pred_norm = _normalize_date(pred_val) or pred_str
        gt_norm = _normalize_date(gt_val) or gt_str

        if pred_norm == gt_norm:
            return 0.7, "normalized_date", pred_str, gt_str

    jaccard = _jaccard_similarity(pred_str, gt_str)
    if jaccard >= 0.8:
        return 0.5, "partial_text", pred_str, gt_str

    token_ratio = _token_overlap_ratio(pred_str, gt_str)
    if token_ratio >= 0.7:
        return 0.4, "partial_text", pred_str, gt_str

    edit_ratio = _compute_edit_distance_ratio(pred_str, gt_str)
    if edit_ratio >= 0.85:
        return 0.35, "partial_text", pred_str, gt_str

    return 0.0, "mismatch", pred_str, gt_str


def _extract_leaf_fields(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any], prefix: str = ""
) -> Dict[str, tuple]:
    results = {}

    for key in ground_truth:
        full_key = f"{prefix}.{key}" if prefix else key

        pred_val = predicted.get(key) if isinstance(predicted, dict) else None
        gt_val = ground_truth.get(key)

        if isinstance(gt_val, dict):
            results.update(_extract_leaf_fields(pred_val, gt_val, full_key))
        elif isinstance(gt_val, list):
            for i, item in enumerate(gt_val):
                if isinstance(item, dict):
                    pred_list_val = (
                        pred_val[i]
                        if isinstance(pred_val, list) and i < len(pred_val)
                        else {}
                    )
                    results.update(
                        _extract_leaf_fields(pred_list_val, item, f"{full_key}[{i}]")
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


def compute_partial_credit(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> PartialCreditResult:
    field_values = _extract_leaf_fields(predicted, ground_truth)

    if not field_values:
        return PartialCreditResult(
            overall_credit_score=1.0,
            field_credits=[],
            exact_matches=0,
            normalized_matches=0,
            partial_matches=0,
            mismatches=0,
            missing_fields=0,
        )

    field_credits = []
    exact_matches = 0
    normalized_matches = 0
    partial_matches = 0
    mismatches = 0
    missing_fields = 0
    scores = []

    for field_name, (pred_val, gt_val) in field_values.items():
        score, match_type, norm_pred, norm_gt = _compute_field_credit(pred_val, gt_val)

        field_credits.append(
            FieldPartialCredit(
                field_name=field_name,
                predicted_value=pred_val,
                ground_truth_value=gt_val,
                score=score,
                match_type=match_type,
                normalized_predicted=norm_pred,
                normalized_ground_truth=norm_gt,
            )
        )

        scores.append(score)

        if match_type == "exact":
            exact_matches += 1
        elif match_type in ["normalized_numeric", "normalized_date"]:
            normalized_matches += 1
        elif match_type == "partial_text":
            partial_matches += 1
        elif match_type == "missing":
            missing_fields += 1
        elif match_type == "extra":
            mismatches += 1
        elif match_type == "mismatch":
            mismatches += 1

    overall_score = float(np.mean(scores)) if scores else 1.0

    return PartialCreditResult(
        overall_credit_score=overall_score,
        field_credits=field_credits,
        exact_matches=exact_matches,
        normalized_matches=normalized_matches,
        partial_matches=partial_matches,
        mismatches=mismatches,
        missing_fields=missing_fields,
    )


def batch_partial_credit(
    predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
) -> Dict[str, PartialCreditResult]:
    results = {}
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        results[f"example_{i}"] = compute_partial_credit(pred, gt)
    return results
