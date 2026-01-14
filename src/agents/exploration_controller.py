from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np


class ExplorationDecision(BaseModel):
    exploration_weight: float
    exploitation_weight: float
    reason: str
    should_explore: bool
    suggested_variants: int


class ExplorationController:
    DEFAULT_EXPLORATION_WEIGHT = 0.3
    DEFAULT_EXPLOITATION_WEIGHT = 0.7
    PLATEAU_THRESHOLD = 0.01
    UNSTABLE_THRESHOLD = 0.15
    PATIENCE = 3

    def __init__(
        self,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
        plateau_threshold: float = PLATEAU_THRESHOLD,
        unstable_threshold: float = UNSTABLE_THRESHOLD,
        patience: int = PATIENCE,
    ):
        self.base_exploration = exploration_weight
        self.plateau_threshold = plateau_threshold
        self.unstable_threshold = unstable_threshold
        self.patience = patience
        self.score_history = []
        self.variance_history = []

    def decide(
        self, score_history: List[float], variance_history: List[float] = None
    ) -> ExplorationDecision:
        self.score_history = score_history
        self.variance_history = variance_history or []

        if len(score_history) < 2:
            return ExplorationDecision(
                exploration_weight=self.base_exploration,
                exploitation_weight=1 - self.base_exploration,
                reason="Insufficient history for decision",
                should_explore=True,
                suggested_variants=3,
            )

        recent_scores = score_history[-self.patience :]

        if len(recent_scores) >= 2:
            improvement = recent_scores[-1] - recent_scores[0]
            if abs(improvement) < self.plateau_threshold:
                return self._handle_plateau()

        if len(self.variance_history) >= 2:
            recent_variance = np.mean(self.variance_history[-3:])
            if recent_variance > self.unstable_threshold:
                return self._handle_unstable()

        return ExplorationDecision(
            exploration_weight=self.base_exploration,
            exploitation_weight=1 - self.base_exploration,
            reason="Stable progress, maintaining exploration",
            should_explore=True,
            suggested_variants=2,
        )

    def _handle_plateau(self) -> ExplorationDecision:
        increased_exploration = min(0.6, self.base_exploration * 1.5)
        return ExplorationDecision(
            exploration_weight=increased_exploration,
            exploitation_weight=1 - increased_exploration,
            reason=f"Plateau detected (improvement < {self.plateau_threshold:.2%}). Increasing exploration.",
            should_explore=True,
            suggested_variants=4,
        )

    def _handle_unstable(self) -> ExplorationDecision:
        decreased_exploration = max(0.1, self.base_exploration * 0.5)
        return ExplorationDecision(
            exploration_weight=decreased_exploration,
            exploitation_weight=1 - decreased_exploration,
            reason=f"High variance detected (>{self.unstable_threshold:.2%}). Increasing exploitation.",
            should_explore=False,
            suggested_variants=2,
        )

    def update_history(self, score: float, variance: float = None):
        self.score_history.append(score)
        if variance is not None:
            self.variance_history.append(variance)

        if len(self.score_history) > 20:
            self.score_history = self.score_history[-20:]
        if len(self.variance_history) > 20:
            self.variance_history = self.variance_history[-20:]

    def get_status(self) -> Dict[str, Any]:
        if len(self.score_history) < 2:
            return {"status": "insufficient_data"}

        recent = self.score_history[-5:]
        trend = (
            "improving"
            if recent[-1] > recent[0]
            else "declining"
            if recent[-1] < recent[0]
            else "stable"
        )

        variance = np.std(recent) if len(recent) > 1 else 0

        return {
            "status": "plateau"
            if abs(recent[-1] - recent[0]) < self.plateau_threshold
            else "active",
            "trend": trend,
            "recent_variance": variance,
            "current_exploration": self.base_exploration,
            "score_history_length": len(self.score_history),
        }


def get_exploration_decision(
    score_history: List[float],
    variance_history: List[float] = None,
    exploration_weight: float = 0.3,
) -> ExplorationDecision:
    controller = ExplorationController(exploration_weight=exploration_weight)
    return controller.decide(score_history, variance_history)
