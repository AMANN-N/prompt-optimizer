from typing import List, Dict, Any, Callable, Optional
from src.core.config import CrossValidationResult
import random
import numpy as np


class CrossValidator:
    DEFAULT_K_FOLDS = 5
    DEFAULT_GENERALIZATION_GAP_THRESHOLD = 0.15

    def __init__(
        self,
        k_folds: int = DEFAULT_K_FOLDS,
        generalization_gap_threshold: float = DEFAULT_GENERALIZATION_GAP_THRESHOLD,
    ):
        self.k_folds = k_folds
        self.generalization_gap_threshold = generalization_gap_threshold

    def _create_folds(
        self, data: List[Any], labels: List[Any], stratify_by: List[str] = None
    ) -> List[tuple]:
        if stratify_by is None:
            random.shuffle(data)
            fold_size = len(data) // self.k_folds
            folds = []
            for i in range(self.k_folds):
                start = i * fold_size
                end = start + fold_size if i < self.k_folds - 1 else len(data)
                val_data = data[start:end]
                val_labels = labels[start:end] if labels else [None] * len(val_data)
                train_data = data[:start] + data[end:]
                train_labels = (
                    (labels[:start] + labels[end:])
                    if labels
                    else [None] * len(train_data)
                )
                folds.append((train_data, train_labels, val_data, val_labels))
            return folds

        strat_groups = {}
        for i, s in enumerate(stratify_by):
            if s not in strat_groups:
                strat_groups[s] = []
            strat_groups[s].append(i)

        stratified_indices = []
        for group, indices in strat_groups.items():
            random.shuffle(indices)
            stratified_indices.extend(indices)

        fold_size = len(stratified_indices) // self.k_folds
        folds = []
        for i in range(self.k_folds):
            start = i * fold_size
            end = start + fold_size if i < self.k_folds - 1 else len(stratified_indices)
            val_indices = stratified_indices[start:end]
            train_indices = [
                idx
                for j in range(self.k_folds)
                if j != i
                for idx in stratified_indices[
                    j * fold_size : (j + 1) * fold_size
                    if j < self.k_folds - 1
                    else len(stratified_indices)
                ]
            ]
            val_data = [data[idx] for idx in val_indices]
            val_labels = (
                [labels[idx] for idx in val_indices]
                if labels
                else [None] * len(val_indices)
            )
            train_data = [data[idx] for idx in train_indices]
            train_labels = (
                [labels[idx] for idx in train_indices]
                if labels
                else [None] * len(train_data)
            )
            folds.append((train_data, train_labels, val_data, val_labels))

        return folds

    def cross_validate(
        self,
        data: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        evaluate_func: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], float],
        stratify_by: List[str] = None,
    ) -> CrossValidationResult:
        folds = self._create_folds(data, ground_truths, stratify_by)

        fold_results = []
        train_scores = []
        val_scores = []

        for train_data, train_labels, val_data, val_labels in folds:
            if not train_data or not val_data:
                continue

            train_score = evaluate_func(train_data, train_labels)
            val_score = evaluate_func(val_data, val_labels)

            train_scores.append(train_score)
            val_scores.append(val_score)

            fold_results.append(
                {
                    "train_score": train_score,
                    "val_score": val_score,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                }
            )

        mean_train = np.mean(train_scores) if train_scores else 0.0
        std_train = np.std(train_scores) if train_scores else 0.0
        mean_val = np.mean(val_scores) if val_scores else 0.0
        std_val = np.std(val_scores) if val_scores else 0.0

        generalization_gap = mean_train - mean_val
        is_overfitting = generalization_gap > self.generalization_gap_threshold

        return CrossValidationResult(
            fold_results=fold_results,
            mean_score=mean_val,
            std_score=std_val,
            generalization_gap=generalization_gap,
            is_overfitting=is_overfitting,
        )

    def should_pause_for_overfitting(self, cv_result: CrossValidationResult) -> tuple:
        if cv_result.generalization_gap > self.generalization_gap_threshold:
            return (
                True,
                f"Generalization gap {cv_result.generalization_gap:.2%} exceeds threshold {self.generalization_gap_threshold:.2%}. Consider adding more diverse examples.",
            )
        if cv_result.std_score > 0.15:
            return (
                True,
                f"High variance across folds (std={cv_result.std_score:.2%}). Model may be unstable.",
            )
        return False, ""


def run_cross_validation(
    data: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    evaluate_func: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], float],
    k_folds: int = 5,
    generalization_gap_threshold: float = 0.15,
) -> CrossValidationResult:
    validator = CrossValidator(k_folds, generalization_gap_threshold)
    return validator.cross_validate(data, ground_truths, evaluate_func)
