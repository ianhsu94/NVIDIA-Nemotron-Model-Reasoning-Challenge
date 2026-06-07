from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TeacherResult:
    answer: str
    confidence: float
    selected_solver: str
    matched_examples: int = 0
    total_examples: int = 0
    reason: str = ""


class BaseTeacher:
    name = "BaseTeacher"
    category = "base"

    def fit(self, train_df: Any) -> None:
        return None

    def predict_row(self, row: Any) -> TeacherResult:
        raise NotImplementedError

